pub mod cuda {
    pub mod customop;
    pub mod bindings;
    pub mod cuda_kernels;
}
mod trainer;
mod project_gaussians;
mod rasterize;
mod sh;
mod utils;
