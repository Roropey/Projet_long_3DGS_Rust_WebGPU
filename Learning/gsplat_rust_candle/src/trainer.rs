// Implementation basé sur le simple_trainer.py d'exmpales de gsplat


use candle_core as candle;
use candle::{Tensor, Var, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW,  ops::sigmoid};
use num;
//use tch::nn::Adam;

//use crate::project_gaussians;
use super::cuda::customop;
//use crate::rasterize;

pub struct Trainer{
    device: Device,
    gt_image: Tensor,
    img_path_res: Path,
    num_points: usize,
    h: usize,
    w: usize,
    focal: f64,
    tile_bounds: (isize,isize,isize),
    img_size: Tensor,
    block: Tensor,
    means: Var, // Changement de tensor à Var pour avoir élément variable/modifiable
    scales: Var,
    rgbs: Var,
    quats: Var,
    opacities: Var,
    viewmat: Tensor,
    background: Tensor,
    
}

impl Trainer {
    pub fn __init__(
        img_path: Path,
        gt_image: Tensor,
        num_points: Option<usize> // When use, put Some(...), if default value, put None
    )-> Self{
        let num_points = num_points.unwrap_or(2000);
        let device = Device::new_cuda(0).unwrap();
        let gt_image = gt_image.to_device(&device).unwrap();
        let num_points = num_points;

        let block_x = 16;
        let block_y = 16;
        let fov_x = std::f64::consts::PI / 2.0;
        let h = gt_image.dim(0).unwrap();
        let w = gt_image.dim(1).unwrap();
        let focal = 0.5*(w as f64) / ((0.5*fov_x).tan());
        let tile_bounds = (
            num::integer::div_floor(w as isize + block_x as isize - 1, block_x as isize),
            num::integer::div_floor(h as isize + block_y as isize - 1, block_y as isize),
            1
        );
        let img_size = Tensor::new(&[w as f32, h as f32, 1.], &device).unwrap();
        let block = Tensor::new(&[block_x as f32, block_y as f32, 1.], &device).unwrap();
        let (means,scales,quats,rgbs,opacities,viewmat,background) = Trainer::_init_gaussians(num_points,device);
        Trainer {device: device,
             gt_image: gt_image,
             img_path_res: img_path,
             num_points: num_points,
             h: h,
             w: w,
             focal: focal,
             tile_bounds: tile_bounds,
             img_size: img_size,
             block: block,
             means: means,
             scales: scales,
             rgbs: rgbs,
             quats: quats,
             opacities: opacities,
             viewmat: viewmat,
             background: background }
    }
    pub fn _init_gaussians(num_points:usize,device:Device) -> (Var,Var,Var,Var,Var,Tensor,Tensor){
        // Random gaussians
        let bd = 2.0;

        let means = Tensor::rand(0.0,1.0,&[num_points,3],&device).unwrap().affine(1.0,-0.5).unwrap().affine(bd,0.0).unwrap();
        let scales = Tensor::rand(0.0,1.0,&[num_points,3],&device).unwrap();
        let rgbs = Tensor::rand(0.0,1.0,&[num_points,3],&device).unwrap();

        let u = Tensor::rand(0.0,1.0,&[num_points,1],&device).unwrap();
        let v = Tensor::rand(0.0,1.0,&[num_points,1],&device).unwrap();
        let w = Tensor::rand(0.0,1.0,&[num_points,1],&device).unwrap();

        let quats = Tensor::cat(
            &[
                u.affine(-1.0,1.0).unwrap().sqrt().unwrap().mul(&v.affine(2.0*std::f64::consts::PI,0.0).unwrap().sin().unwrap()).unwrap(),                
                u.affine(-1.0,1.0).unwrap().sqrt().unwrap().mul(&v.affine(2.0*std::f64::consts::PI,0.0).unwrap().sin().unwrap()).unwrap(),
                u.sqrt().unwrap().mul(&w.affine(2.0*std::f64::consts::PI,0.0).unwrap().sin().unwrap()).unwrap(),
                u.sqrt().unwrap().mul(&w.affine(2.0*std::f64::consts::PI,0.0).unwrap().sin().unwrap()).unwrap()
            ],
            1, //Normalement -1 mais 1 en supposant que c'est selon 1...
        ).unwrap();
        let opacities = Tensor::ones(&[num_points,1],candle::DType::F64,&device).unwrap(); //Supposer f32 en type
        let viewmat = Tensor::new(&[
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],&device).unwrap();

        let background = Tensor::zeros(3, candle::DType::F64, &device).unwrap();


        
        let means = candle::Var::from_tensor(&means).unwrap(); //self.means.requires_grad = true;
        let scales = candle::Var::from_tensor(&scales).unwrap();//self.scales.requires_grad = true;
        let quats = candle::Var::from_tensor(&quats).unwrap();//self.quats.requires_grad = true;
        let rgbs = candle::Var::from_tensor(&rgbs).unwrap();//self.rgbs.requires_grad = true;
        let opacities = candle::Var::from_tensor(&opacities).unwrap();//self.opacities.requires_grad = true;       
        viewmat.detach();  //self.viewmat.requires_grad = false;

        (means,scales,quats,rgbs,opacities,viewmat,background)
    }

    fn train(&mut self,
        iterations: Option<isize>,// When use, put Some(...), if default value, put None
        lr: Option<f64>,// When use, put Some(...), if default value, put None
        save_imgs: Option<bool>,// When use, put Some(...), if default value, put None
    ){
        let iterations = iterations.unwrap_or(1000);
        let lr = lr.unwrap_or(0.01);
        let save_imgs = save_imgs.unwrap_or(false);
        let mut adam_optimize = AdamW::new(
            vec![
                self.rgbs.clone(),
                self.means.clone(),
                self.scales.clone(),
                self.opacities.clone(),
                self.quats.clone()],
            ParamsAdamW {
                lr,
                ..Default::default()
            }
        ).unwrap(); // Utilisation de AdamW au lieu de Adam (trouve pas Adam alors qu'il existe dans optimizer.rs .unwrap())
        let mse_loss = candle_nn::loss::mse;

        let mut frames = Vec::new();

        for iter in 0..iterations{
            let (xys,depths,radii,conics,compensation,num_tiles_hit,cov3d) = customop::ProjectGaussians(
                &self.means,                
                &self.scales,
                1,
                &self.quats,
                &self.viewmat,
                &self.viewmat,
                self.focal,
                self.focal,
                self.w / 2,
                self.h / 2,
                self.h,
                self.w,
                self.tile_bounds,
                None // none pour le threshold car sélection de la valeur par défaut 
            );
            //cuda.synchronize ... Pas trouver comment faire
            let (out_img, out_alpha) = customop::RasterizeGaussians(
                &xys,
                &depths,
                &radii,
                &conics,
                &num_tiles_hit,
                sigmoid(&self.rgbs).unwrap(),
                sigmoid(&self.opacities).unwrap(),
                self.h,
                self.w,
                &self.background,
            );
            //cuda.synchronize ... Pas trouver comment faire
            let loss = mse_loss(out_img,&self.gt_image).unwrap();

            //adam_optimize.zero_grad().unwrap();
            
            adam_optimize.backward_step(&loss).unwrap();
            println!("Iteraion {}/{}, Loss {}",iter+1,iterations,loss);
            if save_imgs && iter%5==0{
                let out_img = (out_img * 255)?.to_dtype(DType::U8);
                frames.push(out_img);
                let image_path = self.img_path_res.set_extension(format!("_{}.jpg",iter)); 
                save_image(out_img,image_path);

            }
        }

    }
}

// Fonctions venant de candle-example src/lib.rs

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((2, 0, 1))
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

// fn image_path_to_tensor(image_path:&std::path::Path,width:usize,height:usize)-> Tensor{
//     let original_image = image::io::Reader::open(&image_path).unwrap()
//             .decode()
//             .map_err(candle::Error::wrap).unwrap();
//     let data = original_image
//                 .resize_exact(
//                     width as u32,
//                     height as u32,
//                     image::imageops::FilterType::Triangle,
//                 )
//                 .to_rgb8()
//                 .into_raw();
            
//     let image = Tensor::from_vec(data, (width, height, 3), &Device::Cpu).unwrap().permute((1, 2, 0)).unwrap();
//     (image.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?
// }

fn main(height:Option<usize>,
    width:Option<usize>,
    num_points:Option<usize>,
    save_imgs:Option<bool>,
    img_path:&std::path::Path,
    iterations:Option<isize>,
    lr:Option<f64>,
){
    let height = height.unwrap_or(256);
    let width = width.unwrap_or(256);
    let num_points = num_points.unwrap_or(100000);
    let iterations = iterations.unwrap_or(1000);
    let save_imgs = save_imgs.unwrap_or(true);
    let lr = lr.unwrap_or(0.01);
    //if Some(img_path){
    let gt_image = (load_image_and_resize(img_path, width, height)?.to_dtype(DType::f32)? * (1./255.))?;
    //} //else {
        //Vois pas comment accéder à certaines valeurs d'un torseurs pour les modifiés
    // }
    let mut trainer = Trainer::__init__(img_path, gt_image, Some(num_points));
    trainer.train(Some(iterations), Some(lr), Some(save_imgs));
    
}