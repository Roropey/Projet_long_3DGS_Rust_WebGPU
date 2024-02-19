struct ProjectGaussians{
    glob_scale : float,
    fx : float,
    fy : float,
    cx : float,
    cy : float,
    img_height: uint,
    img_width : uint,
    tile_bounds: &(i32, i32, i32),
    clip_thresh: float,
    viewmat : &candle::Tensor,
    projmat : &candle::Tensor,
};

impl ProjectGaussian {

    fn fwd()
}

impl CustomOp1 for ProjectGaussians {
    fn name(&self) -> &'static str {
        "project-3d-gaussians"
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
        use candle::cuda_backend::WrapErr;

        let dev = storage.device().clone();
        let slice = storage.as_cuda_slice::<f32>()?;
        let slice = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => slice.slice(o1..o2),
        };
        
    }
}