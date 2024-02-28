// à voir si on garde 
const CONTINUOUS_REDRAW: bool = true;



#[cfg(not(target_arch = "wasm32"))]
struct StdOutLogger;

#[cfg(not(target_arch = "wasm32"))]
impl log::Log for StdOutLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

#[cfg(not(target_arch = "wasm32"))]
static LOGGER: StdOutLogger = StdOutLogger;

#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl std::future::Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}


// def ce qu'est une application:
pub trait Application {
    fn new(device: &wgpu::Device, queue: &mut wgpu::Queue, surface_configuration: &wgpu::SurfaceConfiguration) -> Self;
    fn resize(&mut self, device: &wgpu::Device, queue: &mut wgpu::Queue, surface_configuration: &wgpu::SurfaceConfiguration);
    fn render(&mut self, device: &wgpu::Device, queue: &mut wgpu::Queue, frame: &wgpu::Texture, frame_time: f32);//, output_buffer: wgpu::Buffer);
}



// manager de notre application rust sur webgpu:
pub struct ApplicationManager {
    window: winit::window::Window,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}


impl ApplicationManager {
    pub fn run<A: 'static + Application>(title: &'static str) {
        let event_loop = winit::event_loop::EventLoop::new();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let setup = pollster::block_on(ApplicationManager::setup(&event_loop, title));
            setup.start_loop::<A>(event_loop);
        }
     

    }
   

   
    async fn setup(event_loop: &winit::event_loop::EventLoop<()>, title: &'static str) -> Self {
        // let window = create_window(title, event_loop).await;
        // let mut builder = winit::window::WindowBuilder::new();
        // builder = builder.with_title(title);
        // let window = builder.build(event_loop).unwrap();
        

        let mut builder = winit::window::WindowBuilder::new();
        builder = builder.with_title(title);
        let window = builder.build(event_loop).unwrap();
       


        #[cfg(not(target_arch = "wasm32"))]
        log::set_logger(&LOGGER)
            .map(|()| log::set_max_level(log::LevelFilter::Info))
            .expect("Could not initialize logger");
        let instance = wgpu::Instance::default();
        let (size, surface) = unsafe {
            // taille de ma fenetre
            let size = window.inner_size();
            // surface webgpu associée à la fenetre 
            // unsafe car la ceration d'une fenetre webgpu n'est opas sans risque 
            // region de dessin en gros pour Webgpu 
            let surface = instance.create_surface(&window).expect("WebGPU is not supported or not enabled");
            (size, surface)
        };

        let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

        let adapter_info = adapter.get_info();
        log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

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
        let (device, queue) = adapter
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
        Self {
            window,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
        }
    }
    
    fn generate_surface_configuration(&self) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm, // self.surface.get_supported_formats(&self.adapter)[0],
            view_formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            width: self.size.width,
            height: self.size.height,
            present_mode: wgpu::PresentMode::Fifo, // self.surface.get_supported_present_modes(&self.adapter)[0],
            alpha_mode: wgpu::CompositeAlphaMode::Opaque, // self.surface.get_supported_alpha_modes(&adapter)[0],
        }
    }
    
    fn resize<A: 'static + Application>(&mut self, application: &mut A, size: winit::dpi::PhysicalSize<u32>) {
        self.size.width = size.width.max(1);
        self.size.height = size.height.max(1);
        self.window.request_redraw();
        let surface_configuration = self.generate_surface_configuration();
        self.surface.configure(&self.device, &surface_configuration);
        application.resize(&self.device, &mut self.queue, &surface_configuration);
    }

    fn start_loop<A: 'static + Application>(mut self, event_loop: winit::event_loop::EventLoop<()>) {
        let surface_configuration = self.generate_surface_configuration();
        // new application
        let mut application = A::new(&self.device, &mut self.queue, &surface_configuration);
        let mut rolling_average = 0.0f32;
        let mut average_window = [0.0f32; 64];
        let mut average_window_slot = 0;
        let spawner = Spawner::new();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = winit::event_loop::ControlFlow::Wait;
            match event {
                winit::event::Event::RedrawEventsCleared => {
                    // Code pour le rafraîchissement ici
                }
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::Resized(new_inner_size),
                    ..
                } => {
                    // Ajuster la taille de la fenêtre et redemander le rafraîchissement si nécessaire
                    self.resize(&mut application, new_inner_size);
                    self.window.request_redraw();
                }
                winit::event::Event::RedrawRequested(_) => {
                    // Code pour le rendu ici
                    let frame_time = 0.0; // Temps d'image factice
                    let frame = self.surface.get_current_texture().unwrap();
                    application.render(&self.device, &mut self.queue, &frame.texture, frame_time);
                    frame.present();
                    
                }
                _ => {}
            }
        });
    }
}


#[allow(dead_code)]
fn main() {}
        