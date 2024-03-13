mod read_cam;

use projetLong3DGaussianSplatting::{
    renderer::{Configuration, DepthSorting, Renderer},
    scene::Scene,
};
use std::{env, fs::File};

async fn run(path:&str) {
    let file = File::open(env::args().nth(1).unwrap()).unwrap();
    let instance = wgpu::Instance::default();

    let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .await
    .expect("No suitable GPU adapters found on the system!");

    let required_features = wgpu::Features::default();
    let required_limits = wgpu::Limits {
        max_compute_invocations_per_workgroup: 1024,
        max_storage_buffer_binding_size: 1024 * 1024 * 1024,
        max_buffer_size: 1024 * 1024 * 1024,
        ..wgpu::Limits::default()
    };
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features: {:?}",
        required_features - adapter_features
    );
    let (device, mut queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: required_features,
                limits: required_limits,
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    let (cameras_intrinsics,cameras_extrinsic) = read_cam::read_colmap_scene_info(path);
    let camera = cameras_intrinsics.get(&(1 as u64)).unwrap();
    let width = if camera.width % 64 == 0{ camera.width as u32}else {64 * (camera.width as f32/64.0).round() as u32};


    let height = camera.height as u32;

    let texture = init_output_texture(&device, width,height);  //size.width
    let surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm, 
        view_formats: vec![wgpu::TextureFormat::Rgba8Unorm],
        width: width,
        height: height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque, 
    };

    let renderer = Renderer::new(
        &device,
        Configuration {
            surface_configuration: surface_configuration.clone(),
            depth_sorting: DepthSorting::Gpu,
            use_covariance_for_scale: true,
            use_unaligned_rectangles: true,
            spherical_harmonics_order: 2,
            max_splat_count: 1024 * 1024 * 4,
            radix_bits_per_digit: 8,
            frustum_culling_tolerance: 1.5,
            ellipse_margin: 4.0,
            splat_scale: 1.0,
        },
    );

    // lecture du fichier 
    let (file_header_size, splat_count, mut file) = Scene::parse_file_header(file);
    //creation de la scene
    let mut scene = Scene::new(&device, &renderer, splat_count);
    
    scene.load_chunk(&queue, &mut file, file_header_size, 0..splat_count);

    let viewport_size = wgpu::Extent3d {
        width: width,
        height: height,
        depth_or_array_layers: 1,};

    println!("nb_im{:?}", cameras_extrinsic.keys());
    for i in cameras_extrinsic.keys() {
        let (world_view_transform,projection_matrix,camera_center,FoVy,FoVx) = read_cam::compute_matrix(&cameras_intrinsics,&cameras_extrinsic ,i );
        let output_buffer = renderer.render_frame( &device, &mut queue, &texture, viewport_size,&scene, world_view_transform,projection_matrix,camera_center,FoVy,FoVx);

        let img = to_image(&device, &output_buffer, texture.width(), texture.height()).await;

        img.save(format!("erreur/image{}.jpg", i)).unwrap();
    }
    print!("ok");

}


fn init_output_texture(device: &wgpu::Device, width : u32, height : u32 ) -> wgpu::Texture {
    let texture_desc = wgpu::TextureDescriptor {
        label: Some("output_texture"),
        // The texture size. (layers is set to 1)
        size: wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
        dimension: wgpu::TextureDimension::D2,
        mip_level_count: 1, // the number of mip levels the texture will contain
        sample_count: 1,    // sample_count > 1 would indicate a multisampled texture
        // Use RGBA format for the output
        format: wgpu::TextureFormat::Rgba8Unorm,
        // RENDER_ATTACHMENT -> so that the GPU can render to the texture
        // COPY_SRC -> so that we can pull data out of the texture
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        // Specify the allowed formats when calling "texture.create_view()"
        view_formats: &[], //&[wgpu::TextureFormat::Rgba8Unorm],
    };
    device.create_texture(&texture_desc)
}

async fn to_image(
    device: &wgpu::Device,
    texture_buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let buffer_slice = texture_buffer.slice(..);

    // NOTE: We have to create the mapping THEN device.poll() before await the future.
    // Otherwise the application will freeze.
    let (tx, rx) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();

    // Synchronously and immediately map a buffer for reading.
    // Will panic if buffer_slice.map_async() did not finish yet.
    let data = buffer_slice.get_mapped_range();

    image::RgbaImage::from_raw(width, height, Vec::from(&data as &[u8])).unwrap()
}


fn main() {
    pollster::block_on(run("C:\\3DGS\\gaussian-splatting\\tandt_db\\lefaucheux_7mm"));
    //pollster::block_on(run("C:\\3DGS\\gaussian-splatting\\tandt_db\\tandt\\train"));
    //pollster::block_on(run("C:\\3DGS\\gaussian-splatting\\tandt_db\\synthetic_truck"));

}