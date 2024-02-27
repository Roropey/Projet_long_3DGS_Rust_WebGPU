
use super::cuda_kernels::FORWARD;
use candle_core::CustomOp1;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchAsync,LaunchConfig};
use candle_core::cuda_backend::{WrapErr,CudaDevice};
use candle_core::backend::BackendDevice;
use candle_core::{Layout, Result, Shape};
use candle_core::CpuStorage;


pub struct ProjectGaussians{
    pub glob_scale : f32,
    pub fx : f32,
    pub fy : f32,
    pub cx : f32,
    pub cy : f32,
    pub img_height: u32,
    pub img_width : u32,
    pub tile_bounds: (u32, u32, u32), //a passer en param
    pub clip_thresh: f32,
}

impl ProjectGaussians {

    pub fn fwd(
        &self,
        means3d_storage: candle_core::CudaStorage,
        means3d_layout: &candle_core::Layout,
        scale_storage: candle_core::CudaStorage,
        scale_layout: &candle_core::Layout,
        quats_storage: candle_core::CudaStorage,
        quats_layout: &candle_core::Layout,
        viewmat_storage: candle_core::CudaStorage,
        viewmat_layout: &candle_core::Layout,
        projmat_storage: candle_core::CudaStorage,
        projmat_layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,
        candle_core::CudaStorage, Shape,)>
        { //Surement d'autre verif à faire dans le code pour voir si les inputs sont bon (taille, rank, etc...)
            use candle_core::backend::BackendStorage;
            use candle_core::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
            use candle_core::cuda_backend::WrapErr;
            let devm3D = means3d_storage.device().clone();
            let devsc = scale_storage.device().clone();
            let devq = quats_storage.device().clone();
            let devproj = projmat_storage.device().clone();
            let devview = viewmat_storage.device().clone();
            if ! devm3D.same_device(&devsc) || ! devm3D.same_device(&devq) {//|| not devm3D.same_device(&devproj) || not devm3D.same_device(&devview) {
                candle_core::bail!("all inputs must be on the same device");
            }
            let dev = devm3D;

            let slice_m3D = means3d_storage.as_cuda_slice::<f32>()?;
            let slice_sc = scale_storage.as_cuda_slice::<f32>()?;
            let slice_q = quats_storage.as_cuda_slice::<f32>()?;
            let slice_view = viewmat_storage.as_cuda_slice::<f32>()?;
            let slice_proj = projmat_storage.as_cuda_slice::<f32>()?;

            let slice_m3D = match means3d_layout.contiguous_offsets() {
                None => candle_core::bail!("means 3d input has to be contiguous"),
                Some((o1, o2)) => slice_m3D.slice(o1..o2),
            };
            let slice_sc = match scale_layout.contiguous_offsets() {
                None => candle_core::bail!("scale input has to be contiguous"),
                Some((o1, o2)) => slice_sc.slice(o1..o2),
            };
            let slice_q = match quats_layout.contiguous_offsets() {
                None => candle_core::bail!("quat input has to be contiguous"),
                Some((o1, o2)) => slice_q.slice(o1..o2),
            };
            let slice_view = match viewmat_layout.contiguous_offsets() {
                None => candle_core::bail!("viewmat input has to be contiguous"),
                Some((o1, o2)) => slice_view.slice(o1..o2),
            };
            let slice_proj = match projmat_layout.contiguous_offsets() {
                None => candle_core::bail!("projmat input has to be contiguous"),
                Some((o1, o2)) => slice_proj.slice(o1..o2),
            };

            let func = dev.get_or_load_func("project_gaussians_forward", FORWARD)?; //Cette partie de binding n'est pas fonctionnelle, il faut la revoir
            let num_points = means3d_layout.shape().dims();
            let num_points = num_points[0];

            let dst_cov3d = unsafe { dev.alloc::<f32>(num_points*6) }.w()?;
            let dst_xys_d = unsafe { dev.alloc::<f32>(num_points*2) }.w()?;
            let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_radii = unsafe { dev.alloc::<i64>(num_points) }.w()?;
            let dst_conics = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
            let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_num_tiles_hit = unsafe { dev.alloc::<u32>(num_points) }.w()?;
            
            let params = (
                &slice_m3D,
                &slice_sc,
                self.glob_scale,
                &slice_q,
                &slice_view,
                &slice_proj,
                [self.fx,self.fy,self.cx,self.cy],
                self.img_height,self.img_width,
                [self.tile_bounds.0,self.tile_bounds.1,self.tile_bounds.2],
                self.clip_thresh,
                &dst_cov3d,
                &dst_xys_d,
                &dst_depth,
                &dst_radii,
                &dst_conics,
                &dst_compensation,
                &dst_num_tiles_hit
            );

            let N_THREADS = 256; //TODO : voir si on peut le mettre en argument
            let N_BLOCKS = (num_points + N_THREADS - 1) / N_THREADS;
            let N_THREADS = N_THREADS as u32;
            let N_BLOCKS = N_BLOCKS as u32;
            let cfg = LaunchConfig {
                grid_dim: (N_BLOCKS, 1, 1),
                block_dim: (N_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };

            //mdr func.launch peut lancer des launchs que jusqu'à 11 paramètres, sans raison particulière xd
            //unsafe { func.launch(cfg, params) }.w()?;
            
            let dst_cov3d = candle_core::CudaStorage::wrap_cuda_slice(dst_cov3d, dev.clone());
            let dst_xys_d = candle_core::CudaStorage::wrap_cuda_slice(dst_xys_d, dev.clone());
            let dst_depth = candle_core::CudaStorage::wrap_cuda_slice(dst_depth, dev.clone());
            let dst_radii = candle_core::CudaStorage::wrap_cuda_slice(dst_radii, dev.clone());
            let dst_conics = candle_core::CudaStorage::wrap_cuda_slice(dst_conics, dev.clone());
            let dst_compensation = candle_core::CudaStorage::wrap_cuda_slice(dst_compensation, dev.clone());
            let dst_num_tiles_hit = candle_core::CudaStorage::wrap_cuda_slice(dst_num_tiles_hit, dev.clone());

            Ok((dst_cov3d, Shape::from_dims(&[num_points,6]),
                dst_xys_d, Shape::from_dims(&[num_points,2]),
                dst_depth, Shape::from_dims(&[num_points]),
                dst_radii, Shape::from_dims(&[num_points]),
                dst_conics, Shape::from_dims(&[num_points,3]),
                dst_compensation, Shape::from_dims(&[num_points]),
                dst_num_tiles_hit, Shape::from_dims(&[num_points])))
                 
        }

        //dummy fwd qui va renvoyer des tenseurs à 0
        fn dummy_fwd(
            &self,
            means3d_storage: &candle_core::CudaStorage,
            means3d_layout: &Layout,
            scale_storage: &candle_core::CudaStorage,
            scale_layout: &Layout,
            quats_storage: &candle_core::CudaStorage,
            quats_layout: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
            candle_core::CudaStorage, Shape,
        )>
            {
                let num_points = means3d_layout.shape().dims();
                let num_points = num_points[0];

                let dev = means3d_storage.device().clone();
                let dst_cov3d = unsafe { dev.alloc::<f32>(num_points*6) }.w()?;
                let dst_xys_d = unsafe { dev.alloc::<f32>(num_points*2) }.w()?;
                let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
                let dst_radii = unsafe { dev.alloc::<i64>(num_points) }.w()?;
                let dst_conics = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
                let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
                let dst_num_tiles_hit = unsafe { dev.alloc::<u32>(num_points) }.w()?;
                
                let dst_cov3d = candle_core::CudaStorage::wrap_cuda_slice(dst_cov3d, dev.clone());
                let dst_xys_d = candle_core::CudaStorage::wrap_cuda_slice(dst_xys_d, dev.clone());
                let dst_depth = candle_core::CudaStorage::wrap_cuda_slice(dst_depth, dev.clone());
                let dst_radii = candle_core::CudaStorage::wrap_cuda_slice(dst_radii, dev.clone());
                let dst_conics = candle_core::CudaStorage::wrap_cuda_slice(dst_conics, dev.clone());
                let dst_compensation = candle_core::CudaStorage::wrap_cuda_slice(dst_compensation, dev.clone());
                let dst_num_tiles_hit = candle_core::CudaStorage::wrap_cuda_slice(dst_num_tiles_hit, dev.clone());


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

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        
        Ok((storage.try_clone(layout)?, layout.shape().clone()))
    }

    //après avoir testé customop.rs et fwd : IMPLEMENTER BACKWARD
}

/* #[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_func_binding() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dev = CudaDevice::new(0)?;
        let func = dev.get_or_load_func("project_gaussians_forward", FORWARD);
        match func {
            Ok(_) => assert!(true),
            Err(e) => panic!("Expected successful loading, but got error: {:?}", e),
        }
    }
    // unsafe fn launch(self, cfg: LaunchConfig, params: Params) -> Result<(), result::DriverError>;
    #[test]
    fn dummy_launch() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dev = CudaDevice::new(0)?;
        let func = dev.get_or_load_func("project_gaussians_forward", FORWARD)?;
        let params = (0,0,0,0,0,0,0,0,0,0,0,0,0);
        let cfg = LaunchConfig {
            grid_dim: (0, 0, 0),
            block_dim: (0, 0, 0),
            shared_mem_bytes: 0,
        };
        //unsafe { func.launch(cfg, params) };
        
    }
    

} */