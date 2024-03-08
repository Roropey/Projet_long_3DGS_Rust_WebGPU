use super::cuda_kernels::BACKWARD;
use super::cuda_kernels::FORWARD;
use candle_core::backend::BackendDevice;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::CudaFunction;
use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, DeviceRepr, LaunchAsync, LaunchConfig,
};
use candle_core::cuda_backend::{CudaDevice, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::tensor::from_storage;
use candle_core::CpuStorage;
use candle_core::CudaStorage;
use candle_core::{CustomOp2, CustomOp3};
use candle_core::Device;
use candle_core::Tensor;
use candle_core::{Layout, Result, Shape};

pub struct ProjectGaussians {
    pub glob_scale: f32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub img_height: u32,
    pub img_width: u32,
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
    ) -> Result<(
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
    )> {
        //Surement d'autre verif à faire dans le code pour voir si les inputs sont bon (taille, rank, etc...)
        let devm3d = means3d_storage.device().clone();

        let dev = devm3d;

        //println!("getting slices");
        let slice_m3d = means3d_storage.as_cuda_slice::<f32>()?;
        let slice_sc = scale_storage.as_cuda_slice::<f32>()?;
        let slice_q = quats_storage.as_cuda_slice::<f32>()?;
        let slice_view = viewmat_storage.as_cuda_slice::<f32>()?;
        let slice_proj = projmat_storage.as_cuda_slice::<f32>()?;

        //println!("slice creation");
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

        //println!("loading kernel");
        let func = dev.get_or_load_func("project_gaussians_forward_kernel", FORWARD)?; //Cette partie de binding n'est pas fonctionnelle, il faut la revoir
        let num_points = means3d_layout.shape().dims();
        let num_points = num_points[0];

        //println!("allocating memory for output tensors");
        let dst_cov3d = unsafe { dev.alloc::<f32>(num_points * 6) }.w()?;
        let dst_xys_d = unsafe { dev.alloc::<f32>(num_points * 2) }.w()?;
        let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_radii = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_conics = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;
        let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_num_tiles_hit = unsafe { dev.alloc::<i64>(num_points) }.w()?;

        //println!("creating params for kernel launch");
        let params: &mut [_] = &mut [
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
            (&dst_num_tiles_hit).as_kernel_param(),
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

        println!("Launching project forward kernel");
        unsafe { func.launch(cfg, params) }.w()?;

        //println!("Wrapping output tensors");
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

        //println!("Returning output tensors");
        Ok((
            dst_cov3d,
            Shape::from_dims(&[num_points, 6]),
            dst_xys_d,
            Shape::from_dims(&[num_points, 2]),
            dst_depth,
            Shape::from_dims(&[num_points, 1]),
            dst_radii,
            Shape::from_dims(&[num_points, 1]),
            dst_conics,
            Shape::from_dims(&[num_points, 3]),
            dst_compensation,
            Shape::from_dims(&[num_points, 1]),
            dst_num_tiles_hit,
            Shape::from_dims(&[num_points, 1]),
        ))

        //PB DE DROP DU CUDA DEVICE
    }

    //dummy fwd qui va renvoyer des tenseurs à 0
    fn dummy_fwd(
        &self,
        means3d_storage: &candle_core::CudaStorage,
        means3d_layout: &Layout,
        _scale_storage: &candle_core::CudaStorage,
        _scale_layout: &Layout,
        _quats_storage: &candle_core::CudaStorage,
        _quats_layout: &Layout,
    ) -> Result<(
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
    )> {
        let num_points = means3d_layout.shape().dims();
        let num_points = num_points[0];

        let dev = means3d_storage.device().clone();
        let dst_cov3d = unsafe { dev.alloc::<f32>(num_points * 6) }.w()?;
        let dst_xys_d = unsafe { dev.alloc::<f32>(num_points * 2) }.w()?;
        let dst_depth = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_radii = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_conics = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;
        let dst_compensation = unsafe { dev.alloc::<f32>(num_points) }.w()?;
        let dst_num_tiles_hit = unsafe { dev.alloc::<i64>(num_points) }.w()?;

        let dst_cov3d = candle_core::CudaStorage::wrap_cuda_slice(dst_cov3d, dev.clone());
        let dst_xys_d = candle_core::CudaStorage::wrap_cuda_slice(dst_xys_d, dev.clone());
        let dst_depth = candle_core::CudaStorage::wrap_cuda_slice(dst_depth, dev.clone());
        let dst_radii = candle_core::CudaStorage::wrap_cuda_slice(dst_radii, dev.clone());
        let dst_conics = candle_core::CudaStorage::wrap_cuda_slice(dst_conics, dev.clone());
        let dst_compensation =
            candle_core::CudaStorage::wrap_cuda_slice(dst_compensation, dev.clone());
        let dst_num_tiles_hit =
            candle_core::CudaStorage::wrap_cuda_slice(dst_num_tiles_hit, dev.clone());

        Ok((
            dst_cov3d,
            Shape::from_dims(&[num_points, 6]),
            dst_xys_d,
            Shape::from_dims(&[num_points, 2]),
            dst_depth,
            Shape::from_dims(&[num_points]),
            dst_radii,
            Shape::from_dims(&[num_points]),
            dst_conics,
            Shape::from_dims(&[num_points, 3]),
            dst_compensation,
            Shape::from_dims(&[num_points]),
            dst_num_tiles_hit,
            Shape::from_dims(&[num_points]),
        ))
    }
}

fn get_dev(tensor: &Tensor) -> Result<CudaDevice> {
    let (storage, layout) = tensor.storage_and_layout();
    let storage = super::customop::to_cuda_storage(&storage, &layout)?;
    let dev = storage.device().clone();
    Ok(dev)
}

fn get_cuda_slice_f32<'a>(
    cuda_storage: &'a CudaStorage,
    layout: Layout,
) -> Result<CudaView<'a, f32>> {
    let cuda_slice = cuda_storage.as_cuda_slice::<f32>()?;
    let cuda_slice = match layout.contiguous_offsets() {
        None => candle_core::bail!("input frome {:#?} has to be contiguous", layout),
        Some((o1, o2)) => cuda_slice.slice(o1..o2),
    };
    Ok(cuda_slice)
}

fn get_cuda_slice_i64<'a>(
    cuda_storage: &'a CudaStorage,
    layout: Layout,
) -> Result<CudaView<'a, i64>> {
    let cuda_slice = cuda_storage.as_cuda_slice::<i64>()?;
    let cuda_slice = match layout.contiguous_offsets() {
        None => candle_core::bail!("input frome {:#?} has to be contiguous", layout),
        Some((o1, o2)) => cuda_slice.slice(o1..o2),
    };
    Ok(cuda_slice)
}

fn get_cuda_slice_u32<'a>(
    cuda_storage: &'a CudaStorage,
    layout: Layout,
) -> Result<CudaView<'a, i64>> {
    let cuda_slice = cuda_storage.as_cuda_slice::<i64>()?;
    let cuda_slice = match layout.contiguous_offsets() {
        None => candle_core::bail!("input frome {:#?} has to be contiguous", layout),
        Some((o1, o2)) => cuda_slice.slice(o1..o2),
    };
    Ok(cuda_slice)
}

fn to_tensor(slice: CudaSlice<f32>, dev: CudaDevice, shape: Shape) -> Result<Tensor> {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev);
    let tensor = from_storage(
        candle_core::Storage::Cuda(storage),
        shape,
        BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

impl CustomOp2 for ProjectGaussians {
    fn name(&self) -> &'static str {
        "project-3d-gaussians"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>{
        Ok((s1.try_clone(l1)?, l1.shape().clone()))
    }
    

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let num_points = _arg1.shape().dims();
        let num_points = num_points[0];

        let means3d = _arg1.narrow(1, 0, 3)?;
        let means3d = means3d.contiguous()?;
        let scales = _arg1.narrow(1, 3, 3)?;
        let scales = scales.contiguous()?;
        let quats = _arg1.narrow(1, 6, 4)?;
        let quats = quats.contiguous()?;

        let projmat = _arg2.narrow(1,0,4)?;
        let projmat = projmat.contiguous()?;
        let viewmat = _arg2.narrow(1,4,4)?;
        let viewmat = viewmat.contiguous()?;

        //println!("on est par ici");

        let dev = get_dev(&means3d)?;

        let (storage, layout) = viewmat.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_viewmat = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = projmat.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_projmat = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = means3d.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_means3d = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = scales.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_scales = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = quats.storage_and_layout();

        //println!("par exemple, le layout de quats est {:#?}", layout);

        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_quats = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let dst_v_means3d = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;
        let dst_v_scales = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;
        let dst_v_quats = unsafe { dev.alloc::<f32>(num_points * 4) }.w()?;
        let dst_v_cov3d = unsafe { dev.alloc::<f32>(num_points * 6) }.w()?;
        let dst_v_cov2d = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;

        let cov3d = _res.narrow(1, 0, 6)?;
        let cov3d = cov3d.contiguous()?;
        let radii = _res.narrow(1, 9, 1)?;
        let radii = radii.contiguous()?;
        let conics = _res.narrow(1, 10, 3)?;
        let conics = conics.contiguous()?;

        let (storage, layout) = cov3d.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_cov3d = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = radii.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_radii = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = conics.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_conics = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let v_xy = _grad_res.narrow(1, 6, 2)?;
        let v_xy = v_xy.contiguous()?;
        let v_depth = _grad_res.narrow(1, 8, 1)?;
        let v_depth = v_depth.contiguous()?;
        let v_conics = _grad_res.narrow(1, 10, 3)?;
        let v_conics = v_conics.contiguous()?;

        let (storage, layout) = v_xy.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_v_xy = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = v_depth.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_v_depth = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let (storage, layout) = v_conics.storage_and_layout();
        let cuda_storage = super::customop::to_cuda_storage(&storage, &layout)?;
        let slice_v_conics = get_cuda_slice_f32(&cuda_storage, layout.clone())?;

        let params: &mut [_] = &mut [
            num_points.as_kernel_param(),
            (&slice_means3d).as_kernel_param(),
            (&slice_scales).as_kernel_param(),
            self.glob_scale.as_kernel_param(),
            (&slice_quats).as_kernel_param(),
            (&slice_viewmat).as_kernel_param(),
            (&slice_projmat).as_kernel_param(),
            self.fx.as_kernel_param(),
            self.fy.as_kernel_param(),
            self.cx.as_kernel_param(),
            self.cy.as_kernel_param(),
            self.img_height.as_kernel_param(),
            self.img_width.as_kernel_param(),
            (&slice_cov3d).as_kernel_param(),
            (&slice_radii).as_kernel_param(), 
            (&slice_conics).as_kernel_param(),
            (&slice_v_xy).as_kernel_param(),
            (&slice_v_depth).as_kernel_param(),
            (&slice_v_conics).as_kernel_param(),
            (&dst_v_cov2d).as_kernel_param(),
            (&dst_v_cov3d).as_kernel_param(),
            (&dst_v_means3d).as_kernel_param(),
            (&dst_v_scales).as_kernel_param(),
            (&dst_v_quats).as_kernel_param(),
        ];


        println!("Launching project backward kernel");
        let func = dev.get_or_load_func("project_gaussians_backward_kernel", BACKWARD)?;

        let n_threads = 256; //TODO : voir si on peut le mettre en argument
        let n_blocks = (num_points + n_threads - 1) / n_threads;
        let n_threads = n_threads as u32;
        let n_blocks = n_blocks as u32;
        println!("n_threads : {:#?}", n_threads);
        println!("n_blocks : {:#?}", n_blocks);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (n_threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { func.launch(cfg, params) }.w()?;

        let v_means3d = to_tensor(
            dst_v_means3d,
            dev.clone(),
            Shape::from_dims(&[num_points, 3]),
        )?;
        let v_scales = to_tensor(
            dst_v_scales,
            dev.clone(),
            Shape::from_dims(&[num_points, 3]),
        )?;
        let v_quats = to_tensor(dst_v_quats, dev.clone(), Shape::from_dims(&[num_points, 4]))?;

        let gradtot = Tensor::cat(&[v_means3d, v_scales, v_quats], 1)?;
        let gradtot = gradtot.contiguous()?;
        Ok((Some(gradtot),None))

        //ptetre ici faudra modifier l'op de backprop pour que ça fasse pas n'imp
        //comme dans customop
        //mais ptetre pas
        //let shape = gradtot.shape();
        //let (storage, layout) = gradtot.storage_and_layout();
        //let storage = storage.try_clone(layout)?;
    }
}


pub struct RasterizeGaussians{
    pub not_nd: bool,
    pub tile_bounds: (u32, u32, u32),
    pub img_size: (u32, u32, u32),
    pub channels: u32,
    pub num_intersects: u32,
    pub block_width: u32,
    pub background: Tensor,
}

impl RasterizeGaussians {
    pub fn fwd(
        &self,
        gaussian_ids_sorted_storage: candle_core::CudaStorage,
        gaussian_ids_sorted_layout: &Layout,
        tile_bins_storage: candle_core::CudaStorage,
        tile_bins_layout: &Layout,
        xys_storage: candle_core::CudaStorage,
        xys_layout: &Layout,
        conics_storage: candle_core::CudaStorage,
        conics_layout: &Layout,
        colors_storage: candle_core::CudaStorage,
        colors_layout: &Layout,
        opacity_storage: candle_core::CudaStorage,
        opacity_layout: &Layout,
    ) -> Result<(
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape,
        candle_core::CudaStorage,
        Shape)>{
        //println!("Start fwd");
        let dev = gaussian_ids_sorted_storage.device().clone();
        let gaussian_ids_sorted_slice = get_cuda_slice_i64(&gaussian_ids_sorted_storage, gaussian_ids_sorted_layout.clone())?;
        let tile_bins_slice = get_cuda_slice_f32(&tile_bins_storage, tile_bins_layout.clone())?;
        let xys_slice = get_cuda_slice_f32(&xys_storage, xys_layout.clone())?;
        let conics_slice = get_cuda_slice_f32(&conics_storage, conics_layout.clone())?;
        let colors_slice = get_cuda_slice_f32(&colors_storage, colors_layout.clone())?;
        let opacity_slice = get_cuda_slice_f32(&opacity_storage, opacity_layout.clone())?;
        
        let (background_storage, background_layout) = self.background.storage_and_layout();
        let background_storage = super::customop::to_cuda_storage(&background_storage, &background_layout)?;
        let background_slice = get_cuda_slice_f32(&background_storage, background_layout.clone())?;
        
        //println!("get func");
        let func = if self.not_nd {
            dev.get_or_load_func("rasterize_forward", FORWARD)?
        } else {
            dev.get_or_load_func("nd_rasterize_forward", FORWARD)?
        };
        
        let dst_out_img = unsafe { dev.alloc::<f32>((self.img_size.0 * self.img_size.1 * self.channels) as usize) }.w()?;
        let dst_final_Ts = unsafe { dev.alloc::<f32>((self.img_size.0 * self.img_size.1) as usize) }.w()?;
        let dst_final_index = unsafe { dev.alloc::<f32>((self.img_size.0 * self.img_size.1) as usize) }.w()?;
        let param_norm = &mut [
        self.tile_bounds.0.as_kernel_param(),
        self.tile_bounds.1.as_kernel_param(),
        self.tile_bounds.2.as_kernel_param(),
        self.img_size.0.as_kernel_param(),
        self.img_size.1.as_kernel_param(),
        self.img_size.2.as_kernel_param(),
        (&gaussian_ids_sorted_slice).as_kernel_param(),
        (&tile_bins_slice).as_kernel_param(),
        (&xys_slice).as_kernel_param(),
        (&conics_slice).as_kernel_param(),
        (&colors_slice).as_kernel_param(),
        (&opacity_slice).as_kernel_param(),
        (&dst_final_Ts).as_kernel_param(),
        (&dst_final_index).as_kernel_param(),
        (&dst_out_img).as_kernel_param(),
        (&background_slice).as_kernel_param()
        ];
        let param_nd = &mut [
        self.tile_bounds.0.as_kernel_param(),
        self.tile_bounds.1.as_kernel_param(),
        self.tile_bounds.2.as_kernel_param(),
        self.img_size.0.as_kernel_param(),
        self.img_size.1.as_kernel_param(),
        self.img_size.2.as_kernel_param(),
        self.channels.as_kernel_param(),
        (&gaussian_ids_sorted_slice).as_kernel_param(),
        (&tile_bins_slice).as_kernel_param(),
        (&xys_slice).as_kernel_param(),
        (&conics_slice).as_kernel_param(),
        (&colors_slice).as_kernel_param(),
        (&opacity_slice).as_kernel_param(),
        (&dst_final_Ts).as_kernel_param(),
        (&dst_final_index).as_kernel_param(),
        (&dst_out_img).as_kernel_param(),
        (&background_slice).as_kernel_param()
        ];
        
        let params: &mut [_] = if self.not_nd { param_norm } else {param_nd};
        // let n_threads = 256; //TODO : voir si on peut le mettre en argument
        // let n_blocks = (num_points + n_threads - 1) / n_threads;
        // let n_threads = n_threads as u32;
        // let n_blocks = n_blocks as u32;
        let cfg = LaunchConfig {
            grid_dim: self.tile_bounds,
            block_dim: (self.block_width, self.block_width, 1),
            shared_mem_bytes: 0,
        };

        println!("Launching rasterize forward kernel");
        unsafe { func.launch(cfg, params) }.w()?;

        //println!("Wrapping output tensors");
        let dst_final_Ts = candle_core::CudaStorage::wrap_cuda_slice(dst_final_Ts, dev.clone());
        let dst_final_index = candle_core::CudaStorage::wrap_cuda_slice(dst_final_index, dev.clone());
        let dst_out_img = candle_core::CudaStorage::wrap_cuda_slice(dst_out_img, dev);
        Ok((
            dst_final_Ts,
            Shape::from_dims(&[self.img_size.1 as usize, self.img_size.0 as usize]),
            dst_final_index,
            Shape::from_dims(&[self.img_size.1 as usize, self.img_size.0 as usize]),
            dst_out_img,
            Shape::from_dims(&[self.img_size.1 as usize, self.img_size.0 as usize, self.channels as usize])       
        ))


    }
}
impl CustomOp3 for RasterizeGaussians {
    fn name(&self) -> &'static str {
        "rasterize-gaussians"
    }
    fn cpu_fwd(
            &self,
            s1: &CpuStorage,
            l1: &Layout,
            s2: &CpuStorage,
            l2: &Layout,
            s3: &CpuStorage,
            l3: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            Ok((s1.try_clone(l1)?, l1.shape().clone()))
        
    }
    fn bwd(
            &self,
            tensor_gauss: &Tensor,
            gaussians_ids_sorted: &Tensor,
            tile_bins: &Tensor,
            _res: &Tensor,
            _grad_res: &Tensor,
        ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {

        println!("Tensor_gauss shape : {:#?}", tensor_gauss.shape());
        println!("Gaussians_ids_sorted shape : {:#?}", gaussians_ids_sorted.shape());
        println!("Tile_bins shape : {:#?}", tile_bins.shape());
        println!("Res shape : {:#?}", _res.shape());
        println!("Grad_res shape : {:#?}", _grad_res.shape());

        let xys = tensor_gauss.narrow(1, 0, 2)?;
        let xys = xys.contiguous()?;
        let conics = tensor_gauss.narrow(1, 2, 3)?;
        let conics = conics.contiguous()?;
        let colors = tensor_gauss.narrow(1, 5, self.channels as usize)?;
        let colors = colors.contiguous()?;
        let opacity = tensor_gauss.narrow(1, 5 + self.channels as usize, 1)?;
        let opacity = opacity.contiguous()?;
        
        let (xys_st, xys_l) = xys.storage_and_layout();
        let xys_st = super::customop::to_cuda_storage(&xys_st, &xys_l)?;
        let slice_xys = get_cuda_slice_f32(&xys_st, xys_l.clone())?;
        let (conics_st, conics_l) = conics.storage_and_layout();
        let conics_st = super::customop::to_cuda_storage(&conics_st, &conics_l)?;
        let slice_conics = get_cuda_slice_f32(&conics_st, conics_l.clone())?;
        let (colors_st, colors_l) = colors.storage_and_layout();
        let colors_st = super::customop::to_cuda_storage(&colors_st, &colors_l)?;
        let slice_colors = get_cuda_slice_f32(&colors_st, colors_l.clone())?;
        let (opacity_st, opacity_l) = opacity.storage_and_layout();
        let opacity_st = super::customop::to_cuda_storage(&opacity_st, &opacity_l)?;
        let slice_opacity = get_cuda_slice_f32(&opacity_st, opacity_l.clone())?;
        let (gaussians_ids_sorted_st, gaussians_ids_sorted_l) = gaussians_ids_sorted.storage_and_layout();
        let gaussians_ids_sorted_st = super::customop::to_cuda_storage(&gaussians_ids_sorted_st, &gaussians_ids_sorted_l)?;
        let slice_gaussians_ids_sorted = get_cuda_slice_i64(&gaussians_ids_sorted_st, gaussians_ids_sorted_l.clone())?;
        let (tile_bins_st, tile_bins_l) = tile_bins.storage_and_layout();
        let tile_bins_st = super::customop::to_cuda_storage(&tile_bins_st, &tile_bins_l)?;
        let slice_tile_bins = get_cuda_slice_f32(&tile_bins_st, tile_bins_l.clone())?;
        let (background_st, background_l) = self.background.storage_and_layout();
        let background_st = super::customop::to_cuda_storage(&background_st, &background_l)?;
        let slice_background = get_cuda_slice_f32(&background_st, background_l.clone())?;
        
        let dev = xys_st.device();

        
        let v_out_img = _grad_res.narrow(2, 0, self.channels as usize)?;
        let v_out_img = v_out_img.contiguous()?;
        let final_Ts = _res.narrow(2, self.channels as usize, 1)?;
        let final_Ts = final_Ts.squeeze(2)?.contiguous()?;
        let final_index = _res.narrow(2, self.channels as usize + 1, 1)?;
        let final_index = final_index.squeeze(2)?.contiguous()?;
        let v_out_alpha = _grad_res.narrow(2, self.channels as usize + 2, 1)?;
        let v_out_alpha = v_out_alpha.squeeze(2)?.contiguous()?;

        let (v_out_img_st, v_out_img_l) = v_out_img.storage_and_layout();
        let v_out_img_st = super::customop::to_cuda_storage(&v_out_img_st, &v_out_img_l)?;
        let slice_v_out_img = get_cuda_slice_f32(&v_out_img_st, v_out_img_l.clone())?;
        let (final_Ts_st, final_Ts_l) = final_Ts.storage_and_layout();
        let final_Ts_st = super::customop::to_cuda_storage(&final_Ts_st, &final_Ts_l)?;
        let slice_final_Ts = get_cuda_slice_f32(&conics_st, conics_l.clone())?;
        let (final_index_st, final_index_l) = final_index.storage_and_layout();
        let final_index_st = super::customop::to_cuda_storage(&final_index_st, &final_index_l)?;
        let slice_final_index = get_cuda_slice_f32(&final_index_st, final_index_l.clone())?;
        let (v_out_alpha_st, v_out_alpha_l) = v_out_alpha.storage_and_layout();
        let v_out_alpha_st = super::customop::to_cuda_storage(&v_out_alpha_st, &v_out_alpha_l)?;
        let slice_v_out_alpha = get_cuda_slice_f32(&v_out_alpha_st, v_out_alpha_l.clone())?;
        let num_points = xys.dim(0)?;
        let dst_v_xy = unsafe { dev.alloc::<f32>(num_points * 2) }.w()?;
        let dst_v_conic = unsafe { dev.alloc::<f32>(num_points * 3) }.w()?;
        let dst_v_colors = unsafe { dev.alloc::<f32>(num_points * self.channels as usize) }.w()?;
        let dst_v_opacity = unsafe { dev.alloc::<f32>(num_points * 1) }.w()?;
        
        let func = if self.not_nd {
            dev.get_or_load_func("rasterize_backward_kernel", BACKWARD)?
        } else {
            dev.get_or_load_func("nd_rasterize_backward_kernel", BACKWARD)?
        };
        let param_norm = &mut [
            self.tile_bounds.0.as_kernel_param(),
            self.tile_bounds.1.as_kernel_param(),
            self.tile_bounds.2.as_kernel_param(),
            self.img_size.0.as_kernel_param(),
            self.img_size.1.as_kernel_param(),
            self.img_size.2.as_kernel_param(),
            (&slice_gaussians_ids_sorted).as_kernel_param(),
            (&slice_tile_bins).as_kernel_param(),
            (&slice_xys).as_kernel_param(),
            (&slice_conics).as_kernel_param(),
            (&slice_colors).as_kernel_param(),
            (&slice_opacity).as_kernel_param(),
            (&slice_background).as_kernel_param(),
            (&slice_final_Ts).as_kernel_param(),
            (&slice_final_index).as_kernel_param(),
            (&slice_v_out_img).as_kernel_param(),
            (&slice_v_out_alpha).as_kernel_param(),
            (&dst_v_xy).as_kernel_param(),
            (&dst_v_conic).as_kernel_param(),
            (&dst_v_colors).as_kernel_param(),
            (&dst_v_opacity).as_kernel_param()
        ];
        let param_nd = &mut [
                self.tile_bounds.0.as_kernel_param(),
                self.tile_bounds.1.as_kernel_param(),
                self.tile_bounds.2.as_kernel_param(),
                self.img_size.0.as_kernel_param(),
                self.img_size.1.as_kernel_param(),
                self.img_size.2.as_kernel_param(),
                self.channels.as_kernel_param(),
                (&slice_gaussians_ids_sorted).as_kernel_param(),                
                (&slice_tile_bins).as_kernel_param(),
                (&slice_xys).as_kernel_param(),
                (&slice_conics).as_kernel_param(),
                (&slice_colors).as_kernel_param(),
                (&slice_opacity).as_kernel_param(),
                (&slice_background).as_kernel_param(),
                (&slice_final_Ts).as_kernel_param(),
                (&slice_final_index).as_kernel_param(),
                (&slice_v_out_img).as_kernel_param(),
                (&slice_v_out_alpha).as_kernel_param(),
                (&dst_v_xy).as_kernel_param(),
                (&dst_v_conic).as_kernel_param(),
                (&dst_v_colors).as_kernel_param(),
                (&dst_v_opacity).as_kernel_param()
        ];
        let params: &mut [_] = if self.not_nd { param_norm } else {param_nd};
        let cfg = LaunchConfig {
            grid_dim: self.tile_bounds,
            block_dim: (self.block_width, self.block_width, 1),
            shared_mem_bytes: 0,
        };

        println!("Launching rasterize backward kernel");
        unsafe { func.launch(cfg, params) }.w()?;

        let v_xy = to_tensor(
            dst_v_xy,
            dev.clone(),
            Shape::from_dims(&[num_points, 2]),
        )?;
        let v_conic = to_tensor(
            dst_v_conic,
            dev.clone(),
            Shape::from_dims(&[num_points, 3]),
        )?;
        let v_colors = to_tensor(
            dst_v_colors,
            dev.clone(),
            Shape::from_dims(&[num_points, 3]),
        )?;
        let v_opacity = to_tensor(
            dst_v_opacity,
            dev.clone(),
            Shape::from_dims(&[num_points, 1]),
        )?;


        let gradtot = Tensor::cat(&[v_xy, v_conic, v_colors,v_opacity], 1)?;
        let gradtot = gradtot.contiguous()?;
        Ok((Some(gradtot),None,None))

    }
} 
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_func_binding() -> () {
        let pdev = CudaDevice::new(0);
        let dev;
        match pdev {
            Ok(d) => dev = d,
            Err(e) => panic!(
                "Expected successful device creation, but got error: {:?}",
                e
            ),
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
        let dst_num_tiles_hit = unsafe { dev.alloc::<i64>(1) }.w()?;

        let slice_m3d = unsafe { dev.alloc::<f32>(3) }.w()?;
        let slice_sc = unsafe { dev.alloc::<f32>(1) }.w()?;
        let slice_q = unsafe { dev.alloc::<f32>(4) }.w()?;
        let slice_view = unsafe { dev.alloc::<f32>(16) }.w()?;
        let slice_proj = unsafe { dev.alloc::<f32>(16) }.w()?;

        let glob_scale: f32 = 1.0;
        let fx: f32 = 1.0;
        let fy: f32 = 1.0;
        let cx: f32 = 1.0;
        let cy: f32 = 1.0;
        let img_height: u32 = 1;
        let img_width: u32 = 1;
        let tile_bounds0: u32 = 1;
        let tile_bounds1: u32 = 1;
        let tile_bounds2: u32 = 1;
        let clip_thresh: f32 = 1.0;

        let params: &mut [_] = &mut [
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
            (&dst_num_tiles_hit).as_kernel_param(),
        ];

        let func = dev.get_or_load_func("project_gaussians_forward_kernel", FORWARD)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { func.launch(cfg, params) };
        Ok(())
    }

    #[test]
    fn dummy_bacward_launch()->Result<()>{
        let dev = CudaDevice::new(0).unwrap();
        let num_points = 0;
        
        let dst_v_xys_d = unsafe { dev.alloc::<f32>(2) }.w().unwrap();
        let dst_v_depth = unsafe { dev.alloc::<f32>(1) }.w().unwrap();
        let dst_v_conics = unsafe { dev.alloc::<f32>(3) }.w().unwrap();

        let dst_cov3d = unsafe { dev.alloc::<f32>(6) }.w().unwrap();
        let dst_radii = unsafe { dev.alloc::<f32>(1) }.w().unwrap();
        let dst_conics = unsafe { dev.alloc::<f32>(3) }.w().unwrap();

        let slice_m3d = unsafe { dev.alloc::<f32>(3) }.w().unwrap();
        let slice_sc = unsafe { dev.alloc::<f32>(3) }.w().unwrap();
        let slice_q = unsafe { dev.alloc::<f32>(4) }.w().unwrap();
        let slice_view = unsafe { dev.alloc::<f32>(16) }.w().unwrap();
        let slice_proj = unsafe { dev.alloc::<f32>(16) }.w().unwrap();

        let dst_v_cov2d = unsafe { dev.alloc::<f32>(3) }.w().unwrap();
        let dst_v_cov3d = unsafe { dev.alloc::<f32>(6) }.w().unwrap();
        let dst_v_mean3d = unsafe { dev.alloc::<f32>(3) }.w().unwrap();
        let dst_v_scales = unsafe { dev.alloc::<f32>(3) }.w().unwrap();
        let dst_v_quats = unsafe { dev.alloc::<f32>(4) }.w().unwrap();

        let glob_scale: f32 = 1.0;
        let fx: f32 = 1.0;
        let fy: f32 = 1.0;
        let cx: f32 = 1.0;
        let cy: f32 = 1.0;
        let img_height: u32 = 1;
        let img_width: u32 = 1;
        


        let params: &mut [_] = &mut [
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
            
            (&dst_cov3d).as_kernel_param(),
            (&dst_radii).as_kernel_param(),
            (&dst_conics).as_kernel_param(),

            (&dst_v_xys_d).as_kernel_param(),
            (&dst_v_depth).as_kernel_param(),
            (&dst_v_conics).as_kernel_param(),
            
            (&dst_v_cov2d).as_kernel_param(),
            (&dst_v_cov3d).as_kernel_param(),
            (&dst_v_mean3d).as_kernel_param(),
            (&dst_v_scales).as_kernel_param(),
            (&dst_v_quats).as_kernel_param(),

        ];

        let func = dev.get_or_load_func("project_gaussians_backward_kernel", BACKWARD).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (1, 0, 0),
            block_dim: (1, 0, 0),
            shared_mem_bytes: 0,
        };
        unsafe { func.launch(cfg, params) }.w()?;
        Ok(())
    
    }

    /* #[test]
    #[ignore]
    fn test_bwd(){
        let pdev = CudaDevice::new(0);
        //Defininf every tensor that will be part of _arg1 :
        let means3d = Tensor::from_shape(vec![1, 3], &vec![0.0, 0.0, 0.0]).unwrap();
        let scales = Tensor::from_shape(vec![1, 3], &vec![0.0, 0.0, 0.0]).unwrap();
        let quats = Tensor::from_shape(vec![1, 4], &vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        
    } */
}
