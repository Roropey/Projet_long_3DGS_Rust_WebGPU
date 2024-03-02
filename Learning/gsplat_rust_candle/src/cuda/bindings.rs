
use super::cuda_kernels::FORWARD;
use candle_core::CustomOp1;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchAsync,LaunchConfig,DeviceRepr};
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
            let devm3d = means3d_storage.device().clone();
            
            let dev = devm3d;

            println!("getting slices");
            let slice_m3d = means3d_storage.as_cuda_slice::<f32>()?;
            let slice_sc = scale_storage.as_cuda_slice::<f32>()?;
            let slice_q = quats_storage.as_cuda_slice::<f32>()?;
            let slice_view = viewmat_storage.as_cuda_slice::<f32>()?;
            let slice_proj = projmat_storage.as_cuda_slice::<f32>()?;

            println!("slice creation");
            let slice_m3d = match means3d_layout.contiguous_offsets() {
                None => candle_core::bail!("means 3d input has to be contiguous"),
                Some((o1, o2)) => slice_m3d.slice(o1..o2),
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

            println!("loading kernel");
            let func = dev.get_or_load_func("project_gaussians_forward_kernel", FORWARD)?; //Cette partie de binding n'est pas fonctionnelle, il faut la revoir
            let num_points = means3d_layout.shape().dims();
            let num_points = num_points[0];

            println!("allocating memory for output tensors");
            let dst_cov3d = unsafe { dev.alloc::<f32>(num_points*6) }.w()?;
            let dst_xys_d = unsafe { dev.alloc::<f32>(num_points*2) }.w()?;
            let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_radii = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_conics = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
            let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
            let dst_num_tiles_hit = unsafe { dev.alloc::<u32>(num_points) }.w()?;
            
            println!("creating params for kernel launch");
            let params : &mut [_] = & mut [
                num_points.as_kernel_param(),
                (&slice_m3d).as_kernel_param(),
                (&slice_sc).as_kernel_param(),
                self.glob_scale.as_kernel_param(),
                (&slice_q).as_kernel_param(),
                (&slice_view).as_kernel_param(),
                (&slice_proj).as_kernel_param(),
                self.fx.as_kernel_param(),
                self.fy.as_kernel_param(),
                self.cx.as_kernel_param(),
                self.cy.as_kernel_param(),
                self.img_height.as_kernel_param(),
                self.img_width.as_kernel_param(),
                self.tile_bounds.0.as_kernel_param(),
                self.tile_bounds.1.as_kernel_param(),
                self.tile_bounds.2.as_kernel_param(),
                self.clip_thresh.as_kernel_param(),
                (&dst_cov3d).as_kernel_param(),
                (&dst_xys_d).as_kernel_param(),
                (&dst_depth).as_kernel_param(),
                (&dst_radii).as_kernel_param(),
                (&dst_conics).as_kernel_param(),
                (&dst_compensation).as_kernel_param(),
                (&dst_num_tiles_hit).as_kernel_param()
            ];

            
            
            let n_threads = 256; //TODO : voir si on peut le mettre en argument
            let n_blocks = (num_points + n_threads - 1) / n_threads;
            let n_threads = n_threads as u32;
            let n_blocks = n_blocks as u32;
            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (n_threads, 1, 1),
                shared_mem_bytes: 0,
            };

            println!("Launching kernel");
            unsafe { func.launch(cfg, params) }.w()?;
            
            println!("Wrapping output tensors");
            let dst_cov3d = candle_core::CudaStorage::wrap_cuda_slice(dst_cov3d, dev);
            
            let dev = dst_cov3d.device().clone();
            let dst_xys_d = candle_core::CudaStorage::wrap_cuda_slice(dst_xys_d, dev);
            
            let dev = dst_xys_d.device().clone();
            let dst_depth = candle_core::CudaStorage::wrap_cuda_slice(dst_depth, dev);
            
            let dev = dst_depth.device().clone();
            let dst_radii = candle_core::CudaStorage::wrap_cuda_slice(dst_radii, dev);
            
            let dev = dst_radii.device().clone();
            let dst_conics = candle_core::CudaStorage::wrap_cuda_slice(dst_conics, dev);
            
            let dev = dst_conics.device().clone();
            let dst_compensation = candle_core::CudaStorage::wrap_cuda_slice(dst_compensation, dev);
            
            let dev = dst_compensation.device().clone();
            let dst_num_tiles_hit = candle_core::CudaStorage::wrap_cuda_slice(dst_num_tiles_hit, dev);


            println!("Returning output tensors");
            Ok((dst_cov3d, Shape::from_dims(&[num_points,6]),
                dst_xys_d, Shape::from_dims(&[num_points,2]),
                dst_depth, Shape::from_dims(&[num_points,1]),
                dst_radii, Shape::from_dims(&[num_points,1]),
                dst_conics, Shape::from_dims(&[num_points,3]),
                dst_compensation, Shape::from_dims(&[num_points,1]),
                dst_num_tiles_hit, Shape::from_dims(&[num_points,1])))

        //PB DE DROP DU CUDA DEVICE        
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
                let dst_radii = unsafe { dev.alloc::<f32>(num_points) }.w()?;
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

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_func_binding() -> () {
        let pdev = CudaDevice::new(0);
        let mut dev;
        match pdev {
            Ok(d) => dev = d,
            Err(e) => panic!("Expected successful device creation, but got error: {:?}", e),
        }
        let func = dev.get_or_load_func("project_gaussians_forward_kernel", FORWARD);
        match func {
            Ok(_) => assert!(true),
            Err(e) => panic!("Expected successful loading, but got error: {:?}", e),
        }
    }
    // unsafe fn launch(self, cfg: LaunchConfig, params: Params) -> Result<(), result::DriverError>;
    #[test]
    
    fn dummy_launch() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dev = CudaDevice::new(0)?;
        let num_points = 1;
        let dst_cov3d = unsafe { dev.alloc::<f32>(6) }.w()?;
        let dst_xys_d = unsafe { dev.alloc::<f32>(2) }.w()?;
        let dst_depth = unsafe { dev.alloc::<f32>(1) }.w()?;
        let dst_radii = unsafe { dev.alloc::<f32>(1) }.w()?;
        let dst_conics = unsafe { dev.alloc::<f32>(3) }.w()?;
        let dst_compensation = unsafe { dev.alloc::<f32>(1) }.w()?;
        let dst_num_tiles_hit = unsafe { dev.alloc::<u32>(1) }.w()?;

        let slice_m3d = unsafe { dev.alloc::<f32>(3) }.w()?;
        let slice_sc = unsafe { dev.alloc::<f32>(1) }.w()?;
        let slice_q = unsafe { dev.alloc::<f32>(4) }.w()?;
        let slice_view = unsafe { dev.alloc::<f32>(16) }.w()?;
        let slice_proj = unsafe { dev.alloc::<f32>(16) }.w()?;
        
        let glob_scale : f32 = 1.0;
        let fx : f32 = 1.0;
        let fy : f32 = 1.0;
        let cx : f32 = 1.0;
        let cy : f32 = 1.0;
        let img_height: u32 = 1;
        let img_width : u32 = 1;
        let tile_bounds0 : u32 = 1;
        let tile_bounds1 : u32 = 1;
        let tile_bounds2 : u32 = 1;
        let clip_thresh : f32 = 1.0;

        let params : &mut [_] = & mut [
            num_points.as_kernel_param(),
            (&slice_m3d).as_kernel_param(),
            (&slice_sc).as_kernel_param(),
            glob_scale.as_kernel_param(),
            (&slice_q).as_kernel_param(),
            (&slice_view).as_kernel_param(),
            (&slice_proj).as_kernel_param(),
            fx.as_kernel_param(),
            fy.as_kernel_param(),
            cx.as_kernel_param(),
            cy.as_kernel_param(),
            img_height.as_kernel_param(),
            img_width.as_kernel_param(),
            tile_bounds0.as_kernel_param(),
            tile_bounds1.as_kernel_param(),
            tile_bounds2.as_kernel_param(),
            clip_thresh.as_kernel_param(),
            (&dst_cov3d).as_kernel_param(),
            (&dst_xys_d).as_kernel_param(),
            (&dst_depth).as_kernel_param(),
            (&dst_radii).as_kernel_param(),
            (&dst_conics).as_kernel_param(),
            (&dst_compensation).as_kernel_param(),
            (&dst_num_tiles_hit).as_kernel_param()
        ];
        


        let func = dev.get_or_load_func("project_gaussians_forward_kernel", FORWARD)?;
        let cfg = LaunchConfig {
            grid_dim: (0, 0, 0),
            block_dim: (0, 0, 0),
            shared_mem_bytes: 0,
        };
        unsafe { func.launch(cfg, params) };
        Ok(())
        
    }
    

} 