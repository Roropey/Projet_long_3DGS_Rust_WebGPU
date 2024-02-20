// Implementation basé sur le simple_trainer.py d'exmpales de gsplat

use project_gaussians;
use rasterize;
use candle_core as candle;
use candle::{quantized::k_quants::BlockQ2K, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW,  ops::sigmoid};
use tch::nn::Adam;

pub struct Trainer{
    device: Device,
    gt_image: Tensor,
    num_points: isize,
    h: isize,
    w: isize,
    focal: f32,
    tile_bounds: (isize,isize,isize),
    img_size: Tensor,
    block: Tensor,
    means: Tensor,
    scales: Tensor,
    rgbs: Tensor,
    quats: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    background: Tensor,
    
}

impl Trainer {
    pub fn __init__(
        &mut self,
        gt_image: Tensor,
        num_points: Option<isize> // When use, put Some(...), if default value, put None
    )-> Self{
        let num_points = num_points.unwrap_or(2000);
        self.device = Device::new_cuda(0)?;
        self.gt_image = gt_image.to_device(&self.device);
        self.num_points = num_points;

        let block_x = 16;
        let block_y = 16;
        let fov_x = std::f64::consts::PI / 2.0;
        self.h = gt_image.shape().dims()[0];
        self.w = gt_image.shape().dims()[1];
        self.focal = 0.5*(self.w as f32) / ((0.5*fov_x).tan());
        self.tile_bounds = (
            num::integer::div_floor((self.w + block_x - 1), block_x),
            num::integer::div_floor((self.h + block_y - 1), block_y),
            1
        );
        self.img_size = Tensor::new(&[self.w, self.h, 1], &self.device)?;
        self.block = Tensor::new(&[block_x, block_y, 1], &self.device)?;
        self._init_gaussians();
        self
    }
    pub fn _init_gaussians(&mut self){
        // Random gaussians
        let bd = 2.0;

        self.means = Tensor::rand(0.0,1.0,&[self.num_points,3],&self.device)?.affine(1.0,-0.5)?.affine(bd,0.0)?;
        self.scales = Tensor::rand(0.0,1.0,&[self.num_points,3],&self.device)?;
        self.rgbs = Tensor::rand(0.0,1.0,&[self.num_points,3],&self.device)?;

        let u = Tensor::rand(0.0,1.0,&[self.num_points,1],&self.device)?;
        let v = Tensor::rand(0.0,1.0,&[self.num_points,1],&self.device)?;
        let w = Tensor::rand(0.0,1.0,&[self.num_points,1],&self.device)?;

        self.quats = Tensor::cat(
            &[
                u.affine(-1.0,1.0)?.sqrt()?.mul(&v.affine(2.0*std::f64::consts::PI,0.0)?.sin()?)?,                
                u.affine(-1.0,1.0)?.sqrt()?.mul(&v.affine(2.0*std::f64::consts::PI,0.0)?.sin()?)?,
                u.sqrt()?.mul(&w.affine(2.0*std::f64::consts::PI,0.0)?.sin()?),
                u.sqrt()?.mul(&w.affine(2.0*std::f64::consts::PI,0.0)?.sin()?)
            ],
            1, //Normalement -1 mais 1 en supposant que c'est selon 1...
        );
        self.opacities = Tensor::ones(&[self.num_points,1],f32,&self.device)?; //Supposer f32 en type
        self.viewmat = Tensor::new(&[
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],&self.device)?;

        self.background = Tensor::zeros(3, f32, &self.device)?;


        // Utiles et/ou nécessaires dans notre cas ?
        //self.means.requires_grad = true;
        //self.scales.requires_grad = true;
        //self.quats.requires_grad = true;
        //self.rgbs.requires_grad = true;
        //self.opacities.requires_grad = true;
        //self.viewmat.requires_grad = false;
        self.viewmat.detach();

    }

    fn train(&mut self,
        iterations: Option<isize>,// When use, put Some(...), if default value, put None
        lr: Option<f32>,// When use, put Some(...), if default value, put None
        save_imgs: Option<bool>,// When use, put Some(...), if default value, put None
    ){
        let iterations = iterations.unwrap_or(1000);
        let lr = lr.unwrap_or(0.01);
        let save_imgs = save_imgs.unwrap_or(False);
        let mut adam_optimize = AdamW::new(
            vec![
                candle::Var::from_tensor(&self.rgbs),
                candle::Var::from_tensor(&self.means),
                candle::Var::from_tensor(&self.scales),
                candle::Var::from_tensor(&self.opacities),
                candle::Var::from_tensor(&self.quats)],
            ParamsAdamW {
                lr,
                ..Default::default()
            }
        )?; // Utilisation de AdamW au lieu de Adam (trouve pas Adam alors qu'il existe dans optimizer.rs ?)
        let mse_loss = candle_nn::loss::mse;

        let mut frames = Vec::new();

        for iter in 0..iterations{
            let (xys,depths,radii,conics,compensation,num_tiles_hit,cov3d) = _ProjectGaussian.apply(
                self.means,
                self.scales,
                1,
                self.quats,
                self.viewmat,
                self.viewmat,
                self.focal,
                self,focal,
                self.w / 2,
                self.h / 2,
                self.h,
                self.w,
                self.tile_bounds,
            );
            //cuda.synchronize ... Pas trouver comment faire
            out_img = _RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                sigmoid(&self.rgbs)?,
                sigmoid(&self.opacities)?,
                self.h,
                self.w,
                self.background,
            );
            //cuda.synchronize ... Pas trouver comment faire
            let loss = mse_loss(out_img,&self.gt_image);

            adam_optimize.zero_grad()?;
            
            adam_optimize.backward_step(&loss)?;
            println!("Iteraion {}/{}, Loss {}",iter+1,iterations,loss.item());
            if save_imgs && iter%5==0{

            }
        }

    }
}

fn image_path_to_tensor(image_path:Path,width:usize,height:usize)-> Tensor{
    let original_image = image::io::Reader::open(&image_path)?
            .decode()
            .map_err(candle::Error::wrap)?;
    let data = original_image
                .resize_exact(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::Triangle,
                )
                .to_rgb8()
                .into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((1, 2, 0))?
}

fn main(height:Option<usize>,
    width:Option<usize>,
    num_points:Option<isize>,
    save_imgs:Option<bool>,
    img_path:Option<Path>,
    iterations:Option<isize>,
    lr:Option<f32>,
){
    let height = height.unwrap_or(256);
    let width = width.unwrap_or(256);
    let num_points = num_points.unwrap_or(100000);
    let iterations = iterations.unwrap_or(1000);
    let lr = lr.unwrap_or(0.01);
    if Some(img_path){
        let gt_image = image_path_to_tensor(image_path, width, height);
    } //else {
        //Vois pas comment accéder à certaines valeurs d'un torseurs pour les modifiés
    // }
    let mut trainer = Trainer::__init__(&mut self, gt_image, Some(num_points));
    trainer.train(Some(iterations), Some(lr), Some(save_imgs));
    
}