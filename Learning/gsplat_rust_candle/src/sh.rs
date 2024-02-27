use candle_core as candle;
use candle::{CustomOp2, CudaStorage, CpuStorage, Layout, Shape, Tensor, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
#[cfg(feature = "cuda")]
use super::cuda::cuda_kernels::SH;



fn num_sh_bases(degree:usize) -> usize{
    if degree == 0 {
        1
    } else if degree == 1 {
        4
    } else if degree == 2 {
        9
    } else if degree == 3 {
        16
    } else {
        25
    }
}

fn deg_from_sh(num_bases:usize) -> usize{
    if num_bases == 1 {
        0
    } else if num_bases == 4 {
        1
    } else if num_bases == 9 {
        2
    } else if num_bases == 16 {
        3
    } else if num_bases == 25 {
        4
    } else {
        assert!(false,"Invalid number of SH bases");
        0
    }
}

fn spherical_harmonics(
    degrees_to_use:usize,
    viewdirs: &Tensor,
    coeffs: &Tensor
) -> Tensor {
    /*Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (isize): degree of SHs to use (<= total number available).
        viewdirs (candle::Tensor): viewing directions.
        coeffs (candle::Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    */
    assert!(coeffs.shape().dims()[coeffs.shape().rank()-2] >= num_sh_bases(degrees_to_use));
    let degree = deg_from_sh(coeffs.shape().dims()[coeffs.shape().rank()-2]);
    let num_points = coeffs.shape().dims()[0] as usize;
    let op = _SphericalHarmonicsFct(degrees_to_use,degree,num_points);
    coeffs.apply_op2(&viewdirs,op).unwrap()
}


struct _SphericalHarmonicsFct(usize,usize,usize);

impl CustomOp2 for _SphericalHarmonicsFct{
    fn name(&self) -> &'static str {
        "spherical-harmonics"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage, //coeffs
        l1: &Layout,
        s2: &CpuStorage, //viewdirs 
        l2: &Layout,
    ) -> candle::Result<(CpuStorage, Shape)> {
        Err(candle::Error::Msg("Not supported CPU".to_string()))

    }

    //#[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage, //coeffs
        l1: &Layout,
        s2: &CudaStorage, //viewdirs 
        l2: &Layout,
    ) -> candle::Result<(CudaStorage, Shape)> {
            use candle::backend::BackendStorage;
            use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
            use candle::cuda_backend::WrapErr;
            let dev = s1.device().clone();
            let slice1 = s1.as_cuda_slice::<f64>()?;
            //Copie le slice sur le host (GPU -> CPU)
            //let slice1 = dev.sync_reclaim(cuda_slice.clone()).unwrap();
            // Vérification contiguous
            let slice1 = match l1.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => slice1.slice(o1..o2),
            };
            let slice2 = s2.as_cuda_slice::<f64>()?;
            let slice2 = match l2.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => slice2.slice(o1..o2),
            };
            let _SphericalHarmonicsFct(degrees_to_use,degree,num_points) = *self;
            let dst = unsafe {dev.alloc::<f64>(num_points*3)}.w()?;
            let func = dev.get_or_load_func("compute_sh_forward_kernel", SH)?;
            let params = (num_points,degree,degrees_to_use,&slice2,&slice1,&dst);
            // Supposons que l'on a 1024 thread par block, alors
            let n_threads = 1024;
            let nb_block = (num_points + n_threads - 1 )/n_threads;
            let cfg = LaunchConfig{
                grid_dim:(nb_block as u32,1,1),
                block_dim:(n_threads as u32,1,1),
                shared_mem_bytes: 0,
            };
            unsafe{func.launch(cfg,params)}.w()?;
            let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
            Ok((dst, l2.shape().clone()))
    }

    fn bwd(
        &self,
        arg1: &Tensor,//coeffs
        arg2: &Tensor,//viewdirs 
        res: &Tensor, //v_colors
        grad_res: &Tensor,
    ) -> candle::Result<(Option<Tensor>, Option<Tensor>)> {
        let _SphericalHarmonicsFct(degrees_to_use,degree,num_points) = *self;
        let op = _SphericalHarmonics_bwd(degrees_to_use,degree,num_points,arg1.dim(0)?,arg1.dim(1)?,arg1.dim(2)?);
        Ok((Some(res.apply_op2(&arg2,op)?),None))
    }
}

// Ne voyant pas comment appeler la fonction cuda en codage "normal" (avec les tensors),
// décide d'utiliser le forward d'un CustomOp2 pour reproduire l'appel de fonction trouver en exemple dans Candle
struct _SphericalHarmonics_bwd(usize,usize,usize,usize,usize,usize);

impl CustomOp2 for _SphericalHarmonics_bwd{
    fn name(&self) -> &'static str {
        "spherical-harmonics-backward"
    }
    fn cpu_fwd(
            &self,
            s1: &candle::CpuStorage,
            l1: &candle::Layout,
            s2: &candle::CpuStorage,
            l2: &candle::Layout,
        ) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
        
            Err(candle::Error::Msg("Not supported CPU".to_string()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage, //v_colors
        l1: &Layout,
        s2: &CudaStorage, //viewdirs
        l2: &Layout,
    ) -> candle::Result<(CudaStorage, Shape)> {
            use candle::backend::BackendStorage;
            use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
            use candle::cuda_backend::WrapErr;
            let dev = s1.device.clone();
            let slice1 = s1.as_cuda_slice::<f64>()?;
            //Copie le slice sur le host (GPU -> CPU)
            //let slice1 = dev.sync_reclaim(cuda_slice.clone()).unwrap();
            // Vérification contiguous
            let slice1 = match l1.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => slice1.slice(o1..o2),
            };
            let slice2 = s2.as_cuda_slice::<f64>()?;
            let slice2 = match l2.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => slice2.slice(o1..o2),
            };
            let _SphericalHarmonics_bwd(degrees_to_use,degree,num_points,dim1,dim2,dim3) = *self;      
            let num_base = num_sh_bases(degree);
            // Coeffs de taille {num_points, num_bases, 3}
            let dst = unsafe {dev.alloc::<f64>(num_points*num_base*3)}.w()?;
            let func = dev.get_or_load_func("compute_sh_backward_kernel", SH)?;
            let params = (num_points,degree,degrees_to_use,&slice2,&slice1,&dst);
            // Supposons que l'on a 1024 thread par block, alors
            let n_threads = 1024;
            let nb_block = (num_points + n_threads - 1 )/n_threads ;
            let cfg = LaunchConfig{
                grid_dim:(nb_block as u32,1,1),
                block_dim:(n_threads as u32,1,1),
                shared_mem_bytes: 0,
            };
            unsafe{func.launch(cfg,params)}.w()?;
            let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
            Ok((dst, Shape::from_dims(&[dim1,dim2,dim3])))
    }

}


// fn compute_sh_colors(viewdirs: Tensor,sh_coeffs:Tensor){
//     let (dims,dim_sh,c) = sh_coeffs.shape().into_dims();
//     let bases = eval_sh_bases(dim_sh,viewdirs);
//     (bases.mul(sh_coeffs))

// }

// fn eval_sh_bases(basis_dim:usize,dirs:Tensor){

//     let result = Tensor::new()
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let device = Device::Cpu;
    let device = Device::new_cuda(0)?;

    let num_points = 1;
    let degree = 4;
    let gt_colors = Tensor::ones((num_points,3),candle::DType::F64,&device)?;
    let viewdirs = Tensor::randn(0f64,2.,(num_points,3),&device)?;
    //let norm = viewdirs.no
    let sh_coeffs = Tensor::rand(0.0,1.0,(num_points,num_sh_bases(degree),3),&device)?;
    let sh_coeffs = candle::Var::from_tensor(&sh_coeffs)?;
    //let _test = spherical_harmonics(degree,viewdirs,sh_coeffs);
    let lr = 0.01;
    let mut adam_optimize = AdamW::new(
        vec![
            sh_coeffs.clone(),
        ],
        ParamsAdamW {
            lr,
            ..Default::default()
        }
    )?;
    println!("coeffs {}",sh_coeffs);
    let mse_loss = candle_nn::loss::mse;
    let num_iters = 10;
    for i in 1..num_iters{
        let colors = spherical_harmonics(degree, &viewdirs, &sh_coeffs);
        let loss = mse_loss(&colors,&gt_colors)?;//.powf(2.0)?.mean_all()?;
        if i%1==0{
            println!("Iter {}, colors {}, coeffs {}, loss {} end",i,colors,sh_coeffs,loss);
            let grad = loss.backward()?;
            println!("Grad {grad:?}");
        }
        adam_optimize.backward_step(&loss);
        
    }
    Ok(())
}