#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
    }

}


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

    fn fwd(
        &self,
        means3d_storage: &candle::CudaStorage,
        means3d_layout: &Layout,
        scale_storage: &candle::CudaStorage,
        scale_layout: &Layout,
        quats_storage: &candle::CudaStorage,
        quats_layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,
        candle::CudaStorage, Shape,)>
        { //Surement d'autre verif à faire dans le code pour voir si les inputs sont bon (taille, rank, etc...)
            use candle::backend::BackendStorage;
            use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
            use candle::cuda_backend::WrapErr;
            let devm3D = means3d_storage.device().clone();
            let devsc = scale_storage.device().clone();
            let devq = quats_storage.device().clone();
            let devproj = self.projmat.device().clone();
            let devview = self.viewmat.device().clone();
            if devm3D != devsc || devm3D != devq || devsc != devq || devproj != devview {
                candle::bail!("all inputs must be on the same device");
            }
            let dev = devm3D;

            let slice_m3D = means3d_storage.as_cuda_slice::<f32>()?;
            let slice_sc = scale_storage.as_cuda_slice::<f32>()?;
            let slice_q = quats_storage.as_cuda_slice::<f32>()?;
            let slice_m3D = match means3d_layout.contiguous_offsets() {
                None => candle::bail!("means 3d input has to be contiguous"),
                Some((o1, o2)) => slice_m3D.slice(o1..o2),
            };
            let slice_sc = match scale_layout.contiguous_offsets() {
                None => candle::bail!("scale input has to be contiguous"),
                Some((o1, o2)) => slice_sc.slice(o1..o2),
            };
            let slice_q = match quats_layout.contiguous_offsets() {
                None => candle::bail!("quat input has to be contiguous"),
                Some((o1, o2)) => slice_q.slice(o1..o2),
            };
            let func = dev.get_or_load_func("project_gaussians_forward", cuda_kernels::GSPLATKERNELS)?; //Cette partie de binding n'est pas fonctionnelle, il faut la revoir
            let num_points = means3d_layout.shape().dims2()?;
            let dst_cov3d = unsafe { dev.alloc::<f32>(num_points*6) }.w()?;
            let dst_xys_d = unsafe { dev.alloc::<f32>(num_points*2) }.w()?;
            let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst radii = unsafe { dev.alloc::<i32>(num_points) }.w()?;
            let dst_conics = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
            let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_num_tiles_hit = unsafe { dev.alloc::<i32>(num_points) }.w()?;
            
            let params = (&slice_m3D,&slice_sc,self.glob_scale,&slice_q,&self.viewmat,&self.projmat,[self.fx,self.fy,self.cx,self.cy],self.img_height,self.img_width,[self.tile_bounds.0,self.tile_bounds.1,self.tile_bounds.2],self.clip_thresh,&dst_cov3d,&dst_xys_d,&dst_depth,&dst_radii,&dst_conics,&dst_compensation,&dst_num_tiles_hit);
            let N_THREADS = 256; //TODO : voir si on peut le mettre en argument
            let N_BLOCKS = (num_points + N_THREADS - 1) / N_THREADS;
            let cfg = LaunchConfig {
                grid_dim: (N_BLOCKS, 1, 1),
                block_dim: (N_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe { func.launch(cfg, params) }.w()?;
            
            let dst_cov3d = candle::CudaStorage::wrap_cuda_slice(dst_cov3d, dev);
            let dst_xys_d = candle::CudaStorage::wrap_cuda_slice(dst_xys_d, dev);
            let dst_depth = candle::CudaStorage::wrap_cuda_slice(dst_depth, dev);
            let dst_radii = candle::CudaStorage::wrap_cuda_slice(dst_radii, dev);
            let dst_conics = candle::CudaStorage::wrap_cuda_slice(dst_conics, dev);
            let dst_compensation = candle::CudaStorage::wrap_cuda_slice(dst_compensation, dev);
            let dst_num_tiles_hit = candle::CudaStorage::wrap_cuda_slice(dst_num_tiles_hit, dev);

            Ok((dst_cov3d, Shape::from_dims(&[num_points,6]),
                dst_xys_d, Shape::from_dims(&[num_points,2]),
                dst_depth, Shape::from_dims(&[num_points]),
                dst_radii, Shape::from_dims(&[num_points]),
                dst_conics, Shape::from_dims(&[num_points,3]),
                dst_compensation, Shape::from_dims(&[num_points]),
                dst_num_tiles_hit, Shape::from_dims(&[num_points])))
                 
        }
}

impl CustomOp1 for ProjectGaussians {
    fn name(&self) -> &'static str {
        "project-3d-gaussians"
    }
}