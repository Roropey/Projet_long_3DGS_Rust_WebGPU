// Implementation basé sur le simple_trainer.py d'exmpales de gsplat

use project_gaussians;
use rasterize;
use candle_core as candle;
use candle::{quantized::k_quants::BlockQ2K, Tensor};

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
        num_points: isize // When use, put Some(...), if default value, put None
    ){
        num_points = num_points.unwrap_or(2000);
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

    }

    fn train(&mut self,
        iteration: isize,
        lr: f32,
        save_imgs: bool,
    ) {
        

    }
}
