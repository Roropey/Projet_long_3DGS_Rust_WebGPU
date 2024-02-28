use std::convert::TryInto;

use crate::{
    scene::{Scene, Splat},
    utils::{mat4_multiplication, mat4_transform, motor3d_to_mat4, perspective_projection, transmute_slice,read_and_print_radii_buffer},
};
use geometric_algebra::{
    ppga3d::{Motor, Point},
    Inverse,
};
use wgpu::util::DeviceExt;
use std::fs::File;
use std::io::Write;
use std::any::Any;
use std::path::Path;
use serde_json::to_writer_pretty;

use futures::executor::block_on;


/// Selects how splats are sorted by their distance to the camera
pub enum DepthSorting {
    /// No sorting at all
    None,
    /// Sorting takes place on the CPU and is copied over to the GPU
    Cpu,
    /// Sorting takes place internally on the GPU
    Gpu,
    /// Like [DepthSorting::Gpu] and additionally skips rendering frustum culled splats by stream compaction
    GpuIndirectDraw,
}

/// Rendering configuration
pub struct Configuration {
    /// Format of the frame buffer texture
    pub surface_configuration: wgpu::SurfaceConfiguration,
    /// Selects how splats are sorted by their distance to the camera
    pub depth_sorting: DepthSorting,
    /// Uses the parallel projected covariance for decomposition of semi axes
    pub use_covariance_for_scale: bool,
    /// Decomposes the conic sections and renders them as rotated rectangles
    pub use_unaligned_rectangles: bool,
    /// How many spherical harmonics coefficients to use, possible values are 0..=3
    pub spherical_harmonics_order: usize,
    /// Maximum number of splats to allocate memory for
    pub max_splat_count: usize,
    /// How many bits of the key to bin in a single pass. Should be 8
    pub radix_bits_per_digit: usize,
    /// Factor by which the center of a splat can be outside the frustum without being called. Should be > 1.0
    pub frustum_culling_tolerance: f32,
    /// Factor by which the raserized rectangle reaches beyond the ellipse inside. Should be 2.0
    pub ellipse_margin: f32,
    /// Factor to scale splat ellipsoids with. Should be 1.0
    pub splat_scale: f32,
}

#[repr(C)]
pub(crate) struct Uniforms {
    camera_matrix: [Point; 4],
    view_matrix: [Point; 4],
    view_projection_matrix: [Point; 4],
    view_size: [f32; 2],
    image_size: [u32; 2],
    frustum_culling_tolerance: f32,
    ellipse_size_bias: f32,
    ellipse_margin: f32,
    splat_scale: f32,
    padding: [f32; 0],
}

/// Splats forward renderer
pub struct Renderer {
    /// The rendering configuration
    pub config: Configuration,
    pub(crate) radix_digit_places: usize,
    radix_base: usize,
    workgroup_entries_a: usize,
    workgroup_entries_c: usize,
    max_tile_count_c: usize,
    pub(crate) sorting_buffer_size: usize,
    pub(crate) compute_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) render_bind_group_layout: wgpu::BindGroupLayout,
    radix_sort_a_pipeline: wgpu::ComputePipeline,
    radix_sort_b_pipeline: wgpu::ComputePipeline,
    radix_sort_c_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) sorting_pass_buffers: [wgpu::Buffer; 4],
    pub(crate) sorting_buffer: wgpu::Buffer,
    pub(crate) entry_buffer_a: wgpu::Buffer,
    pub(crate) entry_buffer_b: wgpu::Buffer,
    pub(crate) radii_buffer: wgpu::Buffer, // ce buffer contiendra les radii associcé aux splat, après le tri ? Utile pour la densification
    radii_compute_a_pipeline: wgpu::ComputePipeline,

} 

impl Renderer {
    /// Constructs a new [Renderer]
    pub fn new(device: &wgpu::Device, config: Configuration) -> Self {
        let radix_digit_places = 32 / config.radix_bits_per_digit;
        assert_eq!(32, radix_digit_places * config.radix_bits_per_digit);
        let radix_base = 1 << config.radix_bits_per_digit;
        let entries_per_invocation_a = 4;
        let entries_per_invocation_c = 4;
        let workgroup_invocations_a = radix_base * radix_digit_places;
        let workgroup_invocations_c = radix_base;
        let workgroup_entries_a = workgroup_invocations_a * entries_per_invocation_a;
        let workgroup_entries_c = workgroup_invocations_c * entries_per_invocation_c;
        let max_tile_count_c = (config.max_splat_count + workgroup_entries_c - 1) / workgroup_entries_c;
        let sorting_buffer_size =
            (radix_base * (radix_digit_places + max_tile_count_c) * std::mem::size_of::<u32>()) + std::mem::size_of::<u32>() * 5;
        let mut string: String = include_str!("shaders.wgsl").into();

        // Pipeline overrides are not implemented in wgpu yet
        // injection des paramètres dnas les shaders Webgpu
        for (name, value) in &[
            (
                "USE_DEPTH_SORTING",
                format!("{}{}", !matches!(config.depth_sorting, DepthSorting::None), ""),
            ),
            (
                "USE_INDIRECT_DRAW",
                format!(
                    "{}{}",
                    matches!(config.depth_sorting, DepthSorting::Cpu | DepthSorting::GpuIndirectDraw),
                    ""
                ),
            ),
            ("USE_COVARIANCE_FOR_SCALE", format!("{}{}", config.use_covariance_for_scale, "")),
            ("USE_UNALIGNED_RECTANGLES", format!("{}{}", config.use_unaligned_rectangles, "")),
            ("SPHERICAL_HARMONICS_ORDER", format!("{}{}", config.spherical_harmonics_order, "u")),
            ("MAX_SPLAT_COUNT", format!("{}{}", config.max_splat_count, "u")),
            ("RADIX_BITS_PER_DIGIT", format!("{}{}", config.radix_bits_per_digit, "u")),
            ("RADIX_DIGIT_PLACES", format!("{}{}", radix_digit_places, "u")),
            ("RADIX_BASE", format!("{}{}", radix_base, "u")),
            ("ENTRIES_PER_INVOCATION_A", format!("{}{}", entries_per_invocation_a, "u")),
            ("ENTRIES_PER_INVOCATION_C", format!("{}{}", entries_per_invocation_c, "u")),
            ("WORKGROUP_INVOCATIONS_A", format!("{}{}", workgroup_invocations_a, "u")),
            ("WORKGROUP_INVOCATIONS_C", format!("{}{}", workgroup_invocations_c, "u")),
            ("WORKGROUP_ENTRIES_A", format!("{}{}", workgroup_entries_a, "u")),
            ("WORKGROUP_ENTRIES_C", format!("{}{}", workgroup_entries_c, "u")),
            ("MAX_TILE_COUNT_C", format!("{}{}", max_tile_count_c, "u")),
        ] {
            string = string.replace(name, value);
        }
        
        // on compile le shader avec les configuriations
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(string.into()),
        });
        // accès des données uniformes aux shader webgpu: qui sont communes aux sommets 
        let uniform_layout = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
            },
            count: None,
        };
        let sorting_pass_layout = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as u64),
            },
            count: None,
        };
        let sorting_layout = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(sorting_buffer_size as u64),
            },
            count: None,
        };
        let entries_a_layout = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
            },
            count: None,
        };
        let entries_b_layout = wgpu::BindGroupLayoutEntry {
            binding: 4,
            ..entries_a_layout.clone()
        };
        let splats_layout = wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Splat>() as u64),
            },
            count: None,
        };
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_layout,
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX|wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<(u32, u32)>() as u64),
                    },
                    count: None,
                },
                splats_layout,
            ],
        });


     


        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_layout,
                sorting_pass_layout,
                sorting_layout,
                entries_a_layout,
                entries_b_layout,
                splats_layout,
                wgpu::BindGroupLayoutEntry {
                    binding: 7, // Assurez-vous que ce binding est unique et non utilisé
                    visibility: wgpu::ShaderStages::COMPUTE ,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<f32>() as u64),

                    },
                    count: None,
                },
            ],
        });
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        let radix_sort_a_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mon Pipelinezz de Calcul"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "radixSortA",
        });
        let radix_sort_b_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "radixSortB",
        });
        let radix_sort_c_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "radixSortC",
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vertex",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fragment",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.surface_configuration.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::DstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                unclipped_depth: false,
                cull_mode: None,
                conservative: false,
                polygon_mode: wgpu::PolygonMode::Fill,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST| wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sorting_pass_buffers = (0..4)
            .map(|pass_index| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &[pass_index as u8, 0, 0, 0],
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect::<Vec<wgpu::Buffer>>()
            .try_into()
            .unwrap();
        let sorting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sorting_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let entry_buffer_usage = if matches!(config.depth_sorting, DepthSorting::Cpu) {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        } else {
            wgpu::BufferUsages::STORAGE
        } | wgpu::BufferUsages::COPY_SRC;
        let entry_buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (config.max_splat_count * std::mem::size_of::<(u32, u32)>()) as u64,
            usage: entry_buffer_usage,
            mapped_at_creation: false,
        });
        let entry_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (config.max_splat_count * std::mem::size_of::<(u32, u32)>()) as u64,
            usage: entry_buffer_usage,
            mapped_at_creation: false,
        });



        // calcul du Radii
        // création du buffer qui contiendra les info pour le calcul du Radii
        /// Cela permet de visualiser que le calcul du bffer a bien fonctionner et qu'il y'a bien splat_count radii
        let radii_buffer_size = (config.max_splat_count * std::mem::size_of::<f32>()) as u64;
        let radii_data = vec![100.0f32; config.max_splat_count]; // Création d'un vecteur rempli de 0.0 pour chaque élément du buffer.

        let radii_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radii Buffer"),
            contents: bytemuck::cast_slice(&radii_data),


            usage:wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST ,
        });
        
       

        let radii_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        let radii_compute_a_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mon Pipeline de Calcul"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "computeRadii",
        });
      
    
        Self {
            config,
            radix_digit_places,
            radix_base,
            workgroup_entries_a,
            workgroup_entries_c,
            max_tile_count_c,
            sorting_buffer_size,
            compute_bind_group_layout,
            render_bind_group_layout,
            radix_sort_a_pipeline,
            radix_sort_b_pipeline,
            radix_sort_c_pipeline,
            render_pipeline,
            uniform_buffer,
            sorting_pass_buffers,
            sorting_buffer,
            entry_buffer_a,
            entry_buffer_b,
            radii_buffer,
            radii_compute_a_pipeline,
        }
    }
    
    /// Renders the given `scene` into `frame_view`
    pub fn render_frame(
        &self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        frame_view: &wgpu::TextureView,
        viewport_size: wgpu::Extent3d,
        camera_motor: Motor,
        scene: &Scene,
    ) {
        //camera_matrix contient la représentation matricielle de la transformation de la caméra, ce qui peut être utilisé dans le contexte de WebGPU pour déterminer la vue ou la projection d'objets 3D à l'écran.
        let camera_matrix = motor3d_to_mat4(&camera_motor);
        let view_matrix = motor3d_to_mat4(&camera_motor.inverse());
        let field_of_view_y = std::f32::consts::PI * 0.5;
        let view_height = (field_of_view_y * 0.5).tan();
        let view_width = (viewport_size.width as f32 / viewport_size.height as f32) / view_height;
        let projection_matrix = perspective_projection(view_width, view_height, 1.0, 1000.0);
        let view_projection_matrix = mat4_multiplication(&projection_matrix, &view_matrix);
        let mut splat_count = scene.splat_count;
        if matches!(self.config.depth_sorting, DepthSorting::Cpu) {
            let mut entries: Vec<(u32, u32)> = (0..scene.splat_count)
                .filter_map(|splat_index| {
                    // On récupère la position dans le monde du splat
                    let world_position = Point::new(
                        scene.splat_positions[splat_index * 3 + 0],
                        scene.splat_positions[splat_index * 3 + 1],
                        scene.splat_positions[splat_index * 3 + 2],
                        1.0,
                    );
                    //La position du monde est transformée en espace de clip à l'aide de la view_projection_matrix
                    let homogenous_position = mat4_transform(&view_projection_matrix, &world_position);
                    let clip_space_position = homogenous_position * (1.0 / homogenous_position[3]);
                    //On vérifie si le "splat" est à l'intérieur du frustum de la caméra en utilisant la frustum_culling_tolerance
                    if clip_space_position[0].abs() < self.config.frustum_culling_tolerance
                        && clip_space_position[1].abs() < self.config.frustum_culling_tolerance
                        && (clip_space_position[2] - 0.5).abs() < 0.5
                    {
                        //n prépare un tuple contenant la profondeur (convertie de f32 à u32 pour le tri) et l'index du "splat"
                        Some((unsafe { std::mem::transmute::<f32, u32>(clip_space_position[2]) }, splat_index as u32))
                    } else {
                        None
                    }
                })
                .collect();
            //Une fois triés, les indices et profondeurs des "splats" sont écrits dans le buffer self.entry_buffer_a. Ce buffer est utilisé plus tard dans le pipeline de rendu pour dessiner les "splats" dans l'ordre correct.
            splat_count = entries.len();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            queue.write_buffer(&self.entry_buffer_a, 0, transmute_slice::<_, u8>(&entries));
        }
        let uniform_data = &[Uniforms {
            camera_matrix,
            view_matrix,
            view_projection_matrix,
            view_size: [view_width, view_height],
            image_size: [viewport_size.width, viewport_size.height],
            frustum_culling_tolerance: self.config.frustum_culling_tolerance,
            ellipse_size_bias: 0.2 * view_width / viewport_size.width as f32,
            ellipse_margin: self.config.ellipse_margin,
            splat_scale: self.config.splat_scale,
            padding: [0.0; 0],
        }];
        //les données uniformes dans le buffer uniforme (uniform_buffer) utilisé par le shader. 
        queue.write_buffer(&self.uniform_buffer, 0, transmute_slice::<_, u8>(uniform_data));
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if matches!(self.config.depth_sorting, DepthSorting::Gpu | DepthSorting::GpuIndirectDraw) {
            encoder.clear_buffer(&self.sorting_buffer, 0, None);
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                compute_pass.set_bind_group(0, &scene.compute_bind_groups[1], &[]);
                compute_pass.set_pipeline(&self.radix_sort_a_pipeline);
                compute_pass.dispatch_workgroups(((splat_count + self.workgroup_entries_a - 1) / self.workgroup_entries_a) as u32, 1, 1);
                compute_pass.set_pipeline(&self.radix_sort_b_pipeline);
                compute_pass.dispatch_workgroups(1, self.radix_digit_places as u32, 1);
            }
            for pass_index in 0..self.radix_digit_places {
                if pass_index > 0 {
                    encoder.clear_buffer(
                        &self.sorting_buffer,
                        0,
                        Some(std::num::NonZeroU64::new((self.radix_base * self.max_tile_count_c * std::mem::size_of::<u32>()) as u64).unwrap()),
                    );
                }
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                compute_pass.set_pipeline(&self.radix_sort_c_pipeline);
                compute_pass.set_bind_group(0, &scene.compute_bind_groups[pass_index], &[]);
                compute_pass.dispatch_workgroups(1, ((splat_count + self.workgroup_entries_c - 1) / self.workgroup_entries_c) as u32, 1);
            }
        }
        let workgroups_x = ((splat_count + 64) /65) as u32;


        {let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_bind_group(0, &scene.compute_bind_groups[1], &[]);

        compute_pass.set_pipeline(&self.radii_compute_a_pipeline);
        compute_pass.dispatch_workgroups(workgroups_x, 1, 1);    }
       


    
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &scene.render_bind_group, &[]);
            if matches!(self.config.depth_sorting, DepthSorting::GpuIndirectDraw) {
                render_pass.draw_indirect(&self.sorting_buffer, (self.sorting_buffer_size - std::mem::size_of::<u32>() * 5) as u64);
            } else {
                render_pass.draw(0..4, 0..splat_count as u32);
            }
        }
        
        

            // VERIF DE RADII:
        let buffer_size = (self.config.max_splat_count * std::mem::size_of::<f32>()) as u64;
        
        let copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Copy Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Encoder la commande pour copier les données du radii_buffer vers le copy_buffer
        encoder.copy_buffer_to_buffer(&self.radii_buffer, 0, &copy_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));
        // Attendre que la copie soit terminée
        device.poll(wgpu::Maintain::Wait);
        
        
        let buffer_slice = copy_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait); // S'assurer que la demande de mappage est traitée
        let mut sum = 0;
        let mut max_radius_not_100 = f32::NEG_INFINITY; // Initialisation avec la plus petite valeur possible
        let mut iteration_count = 0; // Variable de comptage des itérations

        if let Ok(Ok(())) = futures::executor::block_on(receiver) {
            // Accès aux données du buffer de copie
            let data = buffer_slice.get_mapped_range();
            let radii: &[f32] = bytemuck::cast_slice(&data);
            for &radius in radii.iter() {
                println!("Radius: {}", radius);
                
                if radius != 100.0 {
                    sum += 1;

                }
                iteration_count += 1; 

                if radius != 100.0 && radius > max_radius_not_100 {
                    max_radius_not_100 = radius; // Mettre à jour le maximum
                }
                // if iteration_count >= 100 {
                //     break; // Sortir de la boucle après 10 itérations
                // }
                
                
            }

            // Démapper le buffer après utilisation
            drop(data);
            copy_buffer.unmap();
            println!("La somme est : {}", sum);
            println!("La somme est : {}", splat_count);
            println!("La valeur maximale de radii qui n'est pas égale à 100.0 est : {}", max_radius_not_100);



        }

        
       
        


    }
}