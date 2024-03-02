
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
            let devsc = scale_storage.device().clone();
            let devq = quats_storage.device().clone();
            let devproj = projmat_storage.device().clone();
            let devview = viewmat_storage.device().clone();
            if ! devm3d.same_device(&devsc) || ! devm3d.same_device(&devq) {//|| not devm3d.same_device(&devproj) || not devm3d.same_device(&devview) {
                candle_core::bail!("all inputs must be on the same device");
            }
            let dev = devm3d;

            let slice_m3d = means3d_storage.as_cuda_slice::<f32>()?;
            let slice_sc = scale_storage.as_cuda_slice::<f32>()?;
            let slice_q = quats_storage.as_cuda_slice::<f32>()?;
            let slice_view = viewmat_storage.as_cuda_slice::<f32>()?;
            let slice_proj = projmat_storage.as_cuda_slice::<f32>()?;

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
            
            let params : &mut [_] = & mut [
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

            unsafe { func.launch(cfg, params) }.w()?;
            
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

fn get_cuda_slice(tensor:&Tensor) -> Result<(CudaSlice,Layout)> {
    let (storage,layout) = tensor.storage_and_layout();
    let cuda_storage = to_cuda_storage(&storage,&layout);
    let cuda_slice = cuda_stroage.as_cuda_slice::<f32>()?;
    let cuda_slice = match cuda_slice.contiguous_offsets() {
                None => candle_core::bail!("means 3d input has to be contiguous"),
                Some((o1, o2)) => cuda_slice.slice(o1..o2),
            };
    Ok(cuda_slice,layout)
}

fn to_tensor(slice:CudaSlice,dev:Device,shape;Shape) -> Result<Tensor>{
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice,dev);
    let tensor = from_storage(candle_core::Storage::Cuda(storage), shape, BackpropOp::none(),false)
}

impl CustomOp1 for ProjectGaussians {
    fn name(&self) -> &'static str {
        "project-3d-gaussians"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        
        Ok((storage.try_clone(layout)?, layout.shape().clone()))
    }

    fn bwd (&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        let num_points = _arg.shape().dims();
        let num_points = num_points[0];
        
        let means3d = _arg.narrow(1,0,3);
        let scales = _arg.narrow(1,3,3);
        let quats = _arg.narrow(1,6,4);
        let slice_means3d = get_cuda_slice(means3d);
        let slice_scales = get_cuda_slice(scales);
        let slice_quats = get_cuda_slice(quats);
        
        let dst_v_means3d = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
        let dst_v_scales = unsafe { dev.alloc::<f32>(num_points*3) }.w()?;
        let dst_v_quats = unsafe { dev.alloc::<f32>(num_points*4) }.w()?;
        
        let cov3d = _res.narrow(1,0,6);
        let radii = _res.narrow(1,9,1);
        let conics = _res.narrow(1,10,3);
        let slice_cov3d = get_cuda_slice(cov3d);
        let slice_radii = get_cuda_slice(radii);
        let slice_conics = get_cuda_slice(conics);
        
        let v_xy = _grad_res.narrow(1,6,2);
        let v_depth = _grad_rest.narrow(1,8,1);
        let v_conics = _grad_res.narrow(1,10,3);
        let slice_v_xy = get_cuda_slice(v_xy);
        let slice_v_depth = get_cuda_slice(v_depth);
        let slice_v_conics = get_cuda_slice(v_conics);

        let params: &mut [_] = & mut [
            num_points.as_kernel_param(),
            (&slice_means3d).as_kernel_param(),
            (&slice_scales).as_kernel_param(),
            self.glob_scale.as_kernel_param(),
            (&slice_quats).as_kernel_param(),
            //viewmat,
            //projmat,
            self.fx.as_kernel_param(),
            self.fy.as_kernel_param(),
            self.cx.as_kernel_param(),
            self.cy.as_kernel_param(),
            self.img_height.as_kernel_param(),
            self.img_width.as_kernel_param(),
            //IMG SIZE Z,
            (&slice_cov3d).as_kernel_param(),
            (&slice_radii).as_kernel_param(), //il faudra modifier le kernel cuda pour qu'il prenne un tensor float et pas int
            (&slice_conics).as_kernel_param(),
            (&slice_v_xy).as_kernel_param(),
            (&slice_v_depth).as_kernel_param(),
            (&slice_v_conics).as_kernel_param(),
            //V COV 2D,
            //V COV 3D,
            (&dst_v_means3d).as_kernel_param(),
            (&dst_v_scales).as_kernel_param(),
            (&dst_v_quats).as_kernel_param(),
        ]

        let func = dev.get_or_load_func("project_gaussians_backward_kernel", BACKWARD)?;
        
        let n_threads = 256; //TODO : voir si on peut le mettre en argument
        let n_blocks = (num_points + n_threads - 1) / n_threads;
        let n_threads = n_threads as u32;
        let n_blocks = n_blocks as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (n_threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, params) }.w()?;

        let v_means3d = to_tensor(dst_v_means3d,dev,Shape::from_dims(&[num_points,3]));
        let v_scales = to_tensor(dst_v_scales,dev,Shape::from_dims(&[num_points,3]));
        let v_quats = to_tensor(dst_v_quats,dev,Shape::from_dims(&[num_points,4]));

        let gradtot = Tensor::cat(&[v_means3d,v_scales,v_quats]);
        let gradtot = gradtot.contiguous()

        //ptetre ici faudra modifier l'op de backprop pour que ça fasse pas n'imp
        //comme dans customop
        //mais ptetre pas
        //let shape = gradtot.shape();
        //let (storage, layout) = gradtot.storage_and_layout();
        //let storage = storage.try_clone(layout)?;
        
    }
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
