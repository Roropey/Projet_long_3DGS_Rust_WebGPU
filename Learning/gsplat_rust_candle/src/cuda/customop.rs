use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::cudarc::driver::result::event::elapsed;
use candle_core::{CustomOp2, CustomOp3};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{CudaDevice, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::op::Op;
use candle_core::tensor::from_storage;
use candle_core::Device;
use candle_core::Result;
use candle_core::Shape;
use candle_core::Storage;
use candle_core::Tensor;
use candle_core::TensorId;
use std::sync::{Arc, RwLock};
#[path = "../utils.rs"] mod utils;

pub(crate) fn to_cuda_storage(
    storage: &Storage,
    layout: &candle_core::Layout,
) -> Result<candle_core::CudaStorage> {
    match storage {
        Storage::Cuda(s) => Ok(s.try_clone(layout)?),
        _ => unreachable!(),
    }
}

/*/// Projette des gaussiennes 3D dans un espace 2D en utilisant les paramètres spécifiés.
///
/// Cette fonction prend en compte un ensemble de gaussiennes définies par leurs moyennes, échelles,
/// et quaternions, puis les projette dans un espace 2D en utilisant une matrice de vue
/// et une matrice de projection spécifiques.
///
/// # Arguments
///
/// * `num_points` - Le nombre de points (gaussiennes) à projeter.
/// * `means3d` - Un tensor représentant les positions des gaussiennes dans l'espace 3D.
/// * `scales` - Un tensor représentant les échelles des gaussiennes.
/// * `glob_scale` - Un facteur d'échelle global appliqué à toutes les gaussiennes.
/// * `quats` - Un tensor contenant les quaternions (w,x,y,z) pour la rotation des splats.
/// * `viewmat` - La matrice de vue pour la projection.
/// * `projmat` - La matrice de projection pour la projection.
/// * `fx`, `fy` - Les composantes focales de la caméra.
/// * `cx`, `cy` - Les coordonnées du centre de la caméra.
/// * `img_height`, `img_width` - Les dimensions de l'image.
/// * `tile_bounds` - Les bornes des tuiles pour limiter la projection (xmin, ymin, xmax).
/// * `clip_thresh` - Le seuil.
///
/// # Retour
///
/// Cette fonction retourne un tuple contenant les tensors suivants :
/// * `cov3d` - Les covariances des gaussiennes projetées.
/// * `xys` - Les coordonnées 2D des gaussiennes projetées.
/// * `depth` - La profondeur des gaussiennes projetées.
/// * `radii` - Les rayons des gaussiennes projetées.
/// * `conics` - Les coniques des gaussiennes projetées.
/// * `compensation` - Les compensations des gaussiennes projetées.
/// * `num_tiles_hit` - Le nombre de tuiles touchées par les gaussiennes projetées.
///
/// # Exemples
///
/// ```
/// // Exemple d'utilisation de la fonction `ProjectGaussians`
/// let result = ProjectGaussians(/* valeurs des arguments */);
/// match result {
///     Ok((positions, ...)) => println!("Projection réussie"),
///     Err(e) => println!("Erreur lors de la projection: {}", e),
/// }
/// ```
///
/// # Erreurs
///
/// Cette fonction peut renvoyer une erreur si les tensors fournis ne sont pas conformes aux attentes
/// ou si la projection ne peut pas être réalisée pour une autre raison. */

pub fn project_gaussians(
    means3d: &candle_core::Tensor,
    scales: &candle_core::Tensor,
    glob_scale: f32,
    quats: &candle_core::Tensor,
    viewmat: &candle_core::Tensor,
    projmat: &candle_core::Tensor,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    img_height: u32,
    img_width: u32,
    tile_bounds: (u32, u32, u32),
    clip_thresh: Option<f32>,
) -> Result<(
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
)> {
    let clip_thresh = clip_thresh.unwrap_or(0.01);
    let mut max = 0;
    let a = [means3d, scales, quats];
    for arg in a.iter() {
        if arg.rank() > max {
            max = arg.rank();
        }
    }
    for arg in a.iter() {
        for _i in 0..((max as i64) - (arg.rank() as i64)) {
            arg.unsqueeze(0)?;
        }
    }
    let tensor_in = Tensor::cat(&a, 1)?;
    let tensor_in = tensor_in.contiguous()?;
    let (_,layout) = tensor_in.storage_and_layout();
    let projview = Tensor::cat(&[projmat, viewmat], 1)?;
    let projview = projview.contiguous()?;
    let (_,projview_layout) = projview.storage_and_layout();
    let c = super::bindings::ProjectGaussians {
        glob_scale,
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    };

    //reecriture de la fonction apply_op1_arc de tensor.rs pour bypass le fonctionnement normal
    let (means3d_storage, means3d_layout) = means3d.storage_and_layout();
    let (scales_storage, scales_layout) = scales.storage_and_layout();
    let (quats_storage, quats_layout) = quats.storage_and_layout();
    let (viewmat_storage, viewmat_layout) = viewmat.storage_and_layout();
    let (projmat_storage, projmat_layout) = projmat.storage_and_layout();


    
    let means3d_storage = to_cuda_storage(&means3d_storage, &means3d_layout)?;
    let scales_storage = to_cuda_storage(&scales_storage, &scales_layout)?;
    let quats_storage = to_cuda_storage(&quats_storage, &quats_layout)?;
    let viewmat_storage = to_cuda_storage(&viewmat_storage, &viewmat_layout)?;
    let projmat_storage = to_cuda_storage(&projmat_storage, &projmat_layout)?;

    let (
        storage_cov3d,
        shape_cov3d,
        storage_xys,
        shape_xys,
        storage_depth,
        shape_depth,
        storage_radii,
        shape_radii,
        storage_conics,
        shape_conics,
        storage_compensation,
        shape_compensation,
        storage_num_tiles_hit,
        shape_num_tiles_hit,
    ) = c.fwd(
        means3d_storage,
        means3d_layout,
        scales_storage,
        scales_layout,
        quats_storage,
        quats_layout,
        viewmat_storage,
        viewmat_layout,
        projmat_storage,
        projmat_layout,
    )?;

    let tensor_cov3d = from_storage(
        candle_core::Storage::Cuda(storage_cov3d),
        shape_cov3d,
        BackpropOp::none(),
        false,
    );
    let tensor_xys = from_storage(
        candle_core::Storage::Cuda(storage_xys),
        shape_xys,
        BackpropOp::none(),
        false,
    );
    let tensor_depth = from_storage(
        candle_core::Storage::Cuda(storage_depth),
        shape_depth,
        BackpropOp::none(),
        false,
    );
    let tensor_radii = from_storage(
        candle_core::Storage::Cuda(storage_radii),
        shape_radii,
        BackpropOp::none(),
        false,
    );
    let tensor_conics = from_storage(
        candle_core::Storage::Cuda(storage_conics),
        shape_conics,
        BackpropOp::none(),
        false,
    );
    let tensor_compensation = from_storage(
        candle_core::Storage::Cuda(storage_compensation),
        shape_compensation,
        BackpropOp::none(),
        false,
    );
    let tensor_num_tiles_hit = from_storage(
        candle_core::Storage::Cuda(storage_num_tiles_hit),
        shape_num_tiles_hit,
        BackpropOp::none(),
        false,
    );


    let (_, tensor_cov3d_layout) = tensor_cov3d.storage_and_layout();
    //Il a fallu modifier le type de radii (from i32 to f32) dans le kernel cuda, puisque il le faut pour la retropropagation, et il faut que tout ai le même type pour cat.
    let tensortot = Tensor::cat(
        &[
            tensor_cov3d,
            tensor_xys,
            tensor_depth,
            tensor_radii,
            tensor_conics,
            tensor_compensation,
        ],
        1,
    )?;

    //réécriture de from_storage et copy
    let shape = tensortot.shape();

    let tensortot = tensortot.contiguous()?;

    let (storage, layout) = tensortot.storage_and_layout();
    let storage = storage.try_clone(layout)?;

    let structc = Arc::new(Box::new(c) as Box<dyn CustomOp2 + Send + Sync>);

    let op = BackpropOp::new2(&tensor_in,&projview, |t1,t2| Op::CustomOp2(t1,t2, structc.clone())); 
    
    let tensor_out = from_storage(storage, shape, op, false);
    let (_, tensor_out_layout) = tensor_out.storage_and_layout();

    
    //la backpropagation va marcher puisque narrow qui va associer l'opération Backprop narrow aux tenseurs de sorte à ce que les gradients des tenseurs
    //de sortie reforme un tenseur unique de ces gradient, qui pourra etre rentré dans le backward de ProjectGaussian (le struct) qu'on re splitera pour donner au kernel cuda
    //et les gradients de seront remis dans 1 gradient dans bckward de ProjectGaussian(le struct) qui sera ensuite re-split dans les bons tenseur par la backpropagation grace à l'opération Cat

    //ATTENTION : Dans tensor_out il n'y PAS num_tiles_hit (soucis de type)
    //Mais ce n'est pas un pb, il faut juste le prendre en compte

    let cov3d = tensor_out.narrow(1, 0, 6)?.contiguous()?;


    

    let xys = tensor_out.narrow(1, 6, 2)?.contiguous()?;
    let depth = tensor_out.narrow(1, 8, 1)?.contiguous()?;
    let radii = tensor_out.narrow(1, 9, 1)?.contiguous()?;
    let conics = tensor_out.narrow(1, 10, 3)?.contiguous()?;
    let compensation = tensor_out.narrow(1, 13, 1)?.contiguous()?;
    let num_tiles_hit = tensor_num_tiles_hit;

    Ok((
        cov3d,
        xys,
        depth,
        radii,
        conics,
        compensation,
        num_tiles_hit,
    ))
}


pub fn rasterize_gaussians(
    xys: &Tensor,
    depths: &Tensor,
    radii: &Tensor,
    conics: &Tensor,
    num_tiles_hit: &Tensor,
    colors: &Tensor,
    opacity: &Tensor,
    img_height: u32,
    img_width: u32,
    block_width: u32,
    background: Option<Tensor>, // When use, put Some(...) and not use, put None
    return_alpha: Option<bool> // When use, put Some(...), if not, put None
) -> candle_core::Result<(Tensor,Option<Tensor>)> {

    let background = background.unwrap_or(Tensor::ones(colors.dim(candle_core::D::Minus1)?, candle_core::DType::F32, colors.device()).unwrap());
    let return_alpha = return_alpha.unwrap_or(false);
    assert!(background.shape().dims()[0] == colors.shape().dims()[colors.shape().rank()-1], "Incorrect shape of background color tensor, expected shape {}",colors.shape().dims()[colors.shape().rank()-1]);
    
    assert!(xys.shape().rank()==2 && xys.shape().dims()[1] == 2, "xys, must have dimensions (N,2)");
    assert!(colors.shape().rank() == 2, "colors must have dimensions (N,D)");
    let num_points = xys.dim(0)?;
    let tile_bounds = (
        num::integer::div_floor(img_width as isize + block_width as isize - 1, block_width as isize) as u32,
        num::integer::div_floor(img_height as isize + block_width as isize - 1, block_width as isize) as u32,
        1 as u32
    );
    let _block = (block_width,block_width,1);
    let img_size = (img_width,img_height,1);
    let (num_intersects, cum_tiles_hit)= utils::compute_cumulative_intersects(num_tiles_hit)?;
    let (out_img, out_alpha) =  /* if num_intersects < 1 {
        (background.unsqueeze(0)?.unsqueeze(0)?.repeat((img_height as usize,img_width as usize,1))?,
        Tensor::ones((img_height as usize,img_width as usize),candle_core::DType::F32,xys.device())?)
    } else  */ {
        let (
            _isect_ids_unsorted,
            _gaussians_ids_unsorted,
            _isect_ids_sorted,
            gaussians_ids_sorted,
            tile_bins,
        ) = utils::bin_and_sort_gaussians(
            num_points,
            num_intersects,
            xys,
            depths,
            radii,
            &cum_tiles_hit,
            (tile_bounds.0 as usize, tile_bounds.1 as usize, tile_bounds.2 as usize),
            block_width as usize,
        )?;

        let not_nd = colors.dim(candle_core::D::Minus1)? == 3;
        let channels = colors.dim(1)? as u32; // Baser sur bindings.cu, rasterize_forward_tensor
        
        let num_intersects = num_intersects as u32;
        let mut max = 0;
        let a = [xys, conics, colors,opacity];
        for arg in a.iter() {
            if arg.rank() > max {
                max = arg.rank();
            }
        }
        for arg in a.iter() {
            for _i in 0..((max as i64) - (arg.rank() as i64)) {
                arg.unsqueeze(0)?;
            }
        }
        let tensor_gauss = Tensor::cat(&a, 1)?;
        let tensor_gauss = tensor_gauss.contiguous()?;
        let (_,layout) = tensor_gauss.storage_and_layout();
    

        let c = super::bindings::RasterizeGaussians{
            not_nd,
            tile_bounds,
            img_size,
            channels,
            num_intersects,
            block_width,
            background
        };
        let (gaussian_ids_sorted_storage, gaussian_ids_sorted_layout) = gaussians_ids_sorted.storage_and_layout();
        let (tile_bins_storage, tile_bins_layout) = tile_bins.storage_and_layout();
        let (xys_storage, xys_layout) = xys.storage_and_layout();
        let (conics_storage, conics_layout) = conics.storage_and_layout();
        let (colors_storage, colors_layout) = colors.storage_and_layout();
        let (opacity_storage, opacity_layout) = opacity.storage_and_layout();

        let gaussian_ids_sorted_storage = to_cuda_storage(&gaussian_ids_sorted_storage, &gaussian_ids_sorted_layout)?;
        let tile_bins_storage = to_cuda_storage(&tile_bins_storage, &tile_bins_layout)?;
        let xys_storage = to_cuda_storage(&xys_storage, &xys_layout)?;
        let conics_storage = to_cuda_storage(&conics_storage, &conics_layout)?;
        let colors_storage = to_cuda_storage(&colors_storage, &colors_layout)?;
        let opacity_storage = to_cuda_storage(&opacity_storage, &opacity_layout)?;

        
        let (
            storage_final_ts,
            shape_final_ts,
            storage_final_index,
            shape_final_index,
            storage_out_img,
            shape_out_img
        ) = c.fwd(
            gaussian_ids_sorted_storage,
            gaussian_ids_sorted_layout,
            tile_bins_storage,
            tile_bins_layout,
            xys_storage,
            xys_layout,
            conics_storage,
            conics_layout,
            colors_storage,
            colors_layout,
            opacity_storage,
            opacity_layout
        )?;
        let tensor_final_ts = from_storage(
            candle_core::Storage::Cuda(storage_final_ts),
            shape_final_ts,
            BackpropOp::none(),
            false,
        );
        let tensor_out_img = from_storage(
            candle_core::Storage::Cuda(storage_out_img),
            shape_out_img,
            BackpropOp::none(),
            false,
        );
        let tensor_final_index = from_storage(
            candle_core::Storage::Cuda(storage_final_index),
            shape_final_index,
            BackpropOp::none(),
            false,
        );
        let out_alpha = if return_alpha {
            tensor_final_ts.affine(-1.0,1.0).unwrap()
        } else {
            Tensor::zeros(tensor_final_ts.shape(),candle_core::DType::F32,tensor_final_ts.device())?
        };
                let tensortot = Tensor::cat(
            &[
                tensor_out_img,
                tensor_final_ts.unsqueeze(2)?,
                tensor_final_index.unsqueeze(2)?,
                out_alpha.unsqueeze(2)?,
            ],
            2,
        )?;          

        //réécriture de from_storage et copy
        let shape = tensortot.shape();

        let tensortot = tensortot.contiguous()?;

        let (storage, layout) = tensortot.storage_and_layout();
        let storage = storage.try_clone(layout)?;

        
        let structc = Arc::new(Box::new(c) as Box<dyn CustomOp3 + Send + Sync>);

        let op = BackpropOp::new3(&tensor_gauss,&gaussians_ids_sorted, &tile_bins, |t1,t2, t3| Op::CustomOp3(t1,t2,t3, structc.clone())); 
        
        let tensor_out = from_storage(storage, shape, op, false).contiguous()?;

        let (_, tensor_out_layout) = tensor_out.storage_and_layout();
        
        let out_img = tensor_out.narrow(2, 0, channels as usize)?.contiguous()?;
        let out_alpha = tensor_out.narrow(2, channels as usize + 2, 1)?.squeeze(2)?.contiguous()?;
        (out_img,out_alpha)
    };
    if return_alpha {
        Ok((out_img,Some(out_alpha)))
    } else {
        Ok((out_img,None))
    }

}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::*;

    fn projection_matrix(
        fx: f32,
        fy: f32,
        W: u32,
        H: u32,
        n: f32,
        f: f32,
        device: &Device,
    ) -> candle_core::Tensor {
        let H = H as f32;
        let W = W as f32;
        let projslice: &[f32] = &[
            2.0 * fx / W,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * fy / H,
            0.0,
            0.0,
            0.0,
            0.0,
            (f + n) / (f - n),
            -2.0 * f * n / (f - n),
            0.0,
            0.0,
            1.0,
            0.0,
        ];
        let projmat = candle_core::Tensor::from_slice(
            projslice,
            &candle_core::Shape::from_dims(&[4, 4]),
            device,
        );
        match projmat {
            Ok(projmat) => projmat,
            Err(e) => panic!(
                "Erreur lors de la création de la matrice de projection : {}",
                e
            ),
        }
    }

    

    #[test]
    fn test_project_gaussians_fwd_small() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::new_cuda(0)?;
        let _num_points = 2;
        let means3d_slice: &[f32] = &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
        let means3d = candle_core::Tensor::from_slice(
            means3d_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let scales_slice: &[f32] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let scales = candle_core::Tensor::from_slice(
            scales_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let glob_scale = 1.0;
        let quats_slice: &[f32] = &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let quats = candle_core::Tensor::from_slice(
            quats_slice,
            &candle_core::Shape::from_dims(&[2, 4]),
            &device,
        )?;
        //let quats = quats / quats.norm(candle_core::Norm::L2, &[1], true);
        let H = 512;
        let W = 512;
        let cx = W as f32 / 2.0;
        let cy = H as f32 / 2.0;
        let fx = W as f32 / 2.0;
        let fy = W as f32 / 2.0;
        let clip_thresh = 0.01;
        let viewmat_slice: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let viewmat = candle_core::Tensor::from_slice(
            viewmat_slice,
            &candle_core::Shape::from_dims(&[4, 4]),
            &device,
        )?;
        let projmat = projection_matrix(fx, fy, W, H, 0.01, 1000.0, &device);
        let _fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = project_gaussians(
            &means3d,
            &scales,
            glob_scale,
            &quats,
            &viewmat,
            &projmat,
            fx,
            fy,
            cx,
            cy,
            H as u32,
            W as u32,
            tile_bounds,
            Some(clip_thresh),
        )?;

        println!("cov3d : {}", cov3d);
        println!("xys : {}", xys);
        println!("depths : {}", depths);
        println!("radii : {}", radii);
        println!("conics : {}", conics);
        println!("compensation : {}", compensation);
        println!("num_tiles_hit : {}", num_tiles_hit);
        Ok(())
    }

    #[test]
    fn test_rasterize_gaussian_fwd_small() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::new_cuda(0)?;
        let xys_slice: &[f32] = &[255.5000, 255.5000, 255.5000, 255.5000];
        let xys = candle_core::Tensor::from_slice(
            xys_slice,
            &candle_core::Shape::from_dims(&[2, 2]),
            &device,
        )?;
        let depths_slice: &[f32] = &[18.0, 18.0];
        let depths = candle_core::Tensor::from_slice(
            depths_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let radii_slice: &[f32] = &[43.0, 43.0];
        let radii = candle_core::Tensor::from_slice(
            radii_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let conics_slice: &[f32] = &[0.0049, -0.0000, 0.0049, 0.0049, -0.0000, 0.0049];
        let conics = candle_core::Tensor::from_slice(
            conics_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let num_tiles_hit_slice: &[i64] = &[36, 36];
        let num_tiles_hit = candle_core::Tensor::from_slice(
            num_tiles_hit_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let colors_slice: &[f32] = &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let colors = candle_core::Tensor::from_slice(
            colors_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let opacity_slice: &[f32] = &[1.0, 1.0];
        let opacity = candle_core::Tensor::from_slice(
            opacity_slice,
            &candle_core::Shape::from_dims(&[2,1]),
            &device,
        )?;
        let H = 512;
        let W = 512;
        let block_width = 16;
        let background = None;
        let return_alpha = None;

        let (out_img, out_alpha) = rasterize_gaussians(
            &xys,
            &depths,
            &radii,
            &conics,
            &num_tiles_hit,
            &colors,
            &opacity,
            H,
            W,
            block_width,
            background,
            return_alpha,
        )?;

        println!("out_img : {}", out_img);
        println!("out_alpha : {:?}", out_alpha);
        Ok(())
    }
    

    
    #[test]
    
    fn test_dummy_bwd_project() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::new_cuda(0)?;
        let _num_points = 2;
        let means3d_slice: &[f32] = &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0];
        let means3d = candle_core::Tensor::from_slice(
            means3d_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let scales_slice: &[f32] = &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let scales = candle_core::Tensor::from_slice(
            scales_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let glob_scale = 1.0;
        let quats_slice: &[f32] = &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let quats = candle_core::Tensor::from_slice(
            quats_slice,
            &candle_core::Shape::from_dims(&[2, 4]),
            &device,
        )?;
        //let quats = quats / quats.norm(candle_core::Norm::L2, &[1], true);
        let H = 512;
        let W = 512;
        let cx = W as f32 / 2.0;
        let cy = H as f32 / 2.0;
        let fx = W as f32 / 2.0;
        let fy = W as f32 / 2.0;
        let clip_thresh = 0.01;
        let viewmat_slice: &[f32] = &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let viewmat = candle_core::Tensor::from_slice(
            viewmat_slice,
            &candle_core::Shape::from_dims(&[4, 4]),
            &device,
        )?;
        let projmat = projection_matrix(fx, fy, W, H, 0.01, 1000.0, &device);
        let _fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = project_gaussians(
            &means3d,
            &scales,
            glob_scale,
            &quats,
            &viewmat,
            &projmat,
            fx,
            fy,
            cx,
            cy,
            H as u32,
            W as u32,
            tile_bounds,
            Some(clip_thresh),
        )?;

        println!("IIIIIIICIIIIIII");
        println!("cov3d : {}", cov3d);
        if let Some(op) = cov3d.op() {
            println!("some op");
            match op {
                Op::Copy(t) =>{
                    if let Some(op2) = t.op(){
                        if let Some(op2) = t.op(){
                            match op2 {
                                Op::Narrow(t,_,_,_) => {
                                    println!("narrow");
                                    let grad = Tensor::rand(0.0 as f32,1.0 as f32,t.shape(),t.device())?;
                                    let (_,layout) = t.storage_and_layout();
                                    println!("layout de tensor_in dans backward: {:#?}", layout);
                                    println!("tensor id : {:?}", t.id());
                                    if let Some(op2) = t.op(){
                                        
                                        println!("some op2");
                                        match op2 {
                                            Op::CustomOp2(arg1,arg2,c) => {
                                                println!("customop2");
                                                let structc = c.bwd(&arg1,&arg2,t,&grad)?;
                                                if let (Some(grad),_) = structc {
                                                    println!("grad : {}", grad);
                                                }
                                                else{
                                                    println!("no grad");
                                                }
                                            }
                                            _ => panic!("Op2 n'est pas un CustomOp2")
                                        }
                                    }
                                }
                                _ => panic!("Op n'est pas un Narrow")
                            }
                            
                        }
                    }
                }
                _ => panic!("Op n'est pas un Copy")
        }
        }
        else{
            println!("no op");
        }

        Ok(())
    }

    #[test]
    
    fn test_dummy_bwd_rasterize() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = Device::new_cuda(0)?;
        let xys_slice: &[f32] = &[255.5000, 255.5000, 255.5000, 255.5000];
        let xys = candle_core::Tensor::from_slice(
            xys_slice,
            &candle_core::Shape::from_dims(&[2, 2]),
            &device,
        )?;
        let depths_slice: &[f32] = &[18.0, 18.0];
        let depths = candle_core::Tensor::from_slice(
            depths_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let radii_slice: &[f32] = &[43.0, 43.0];
        let radii = candle_core::Tensor::from_slice(
            radii_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let conics_slice: &[f32] = &[0.0049, -0.0000, 0.0049, 0.0049, -0.0000, 0.0049];
        let conics = candle_core::Tensor::from_slice(
            conics_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let num_tiles_hit_slice: &[i64] = &[36, 36];
        let num_tiles_hit = candle_core::Tensor::from_slice(
            num_tiles_hit_slice,
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let colors_slice: &[f32] = &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let colors = candle_core::Tensor::from_slice(
            colors_slice,
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let opacity_slice: &[f32] = &[1.0, 1.0];
        let opacity = candle_core::Tensor::from_slice(
            opacity_slice,
            &candle_core::Shape::from_dims(&[2,1]),
            &device,
        )?;
        let H = 512;
        let W = 512;
        let block_width = 16;
        let background = None;
        let return_alpha = None;


        let (out_img, out_alpha) = rasterize_gaussians(
            &xys,
            &depths,
            &radii,
            &conics,
            &num_tiles_hit,
            &colors,
            &opacity,
            H,
            W,
            block_width,
            background,
            return_alpha,
        )?;



        println!("out_img : {}", out_img);
        if let Some(op) = out_img.op() {
            println!("some op");
            match op {
                Op::Copy(t) =>{
                    if let Some(op2) = t.op(){
                        match op2 {
                            Op::Narrow(t,_,_,_) => {
                                println!("narrow");
                                let grad = Tensor::rand(0.0 as f32,1.0 as f32,t.shape(),t.device())?;
                                let (_,layout) = t.storage_and_layout();
                                println!("layout de tensor_in dans backward: {:#?}", layout);
                                println!("tensor id : {:?}", t.id());
                                if let Some(op2) = t.op(){
                                    
                                    println!("some op2");
                                    match op2 {
                                        Op::CustomOp3(arg1,arg2,arg3,c) => {
                                            println!("customop3");
                                            let structc = c.bwd(&arg1,&arg2,&arg3,t,&grad)?;
                                            if let (Some(grad),_,_) = structc {
                                                println!("grad : {}", grad);
                                            }
                                            else{
                                                println!("no grad");
                                            }
                                        }
                                        _ => panic!("Op2 n'est pas un CustomOp3")
                                    }
                                }
                            }
                            _ => panic!("Op n'est pas un Narrow")
                        }
                    }
                }
                _ => panic!("Op n'est pas un Copy")
            }
        }
        else{
            println!("no op");
        }

        Ok(())
    }


}
