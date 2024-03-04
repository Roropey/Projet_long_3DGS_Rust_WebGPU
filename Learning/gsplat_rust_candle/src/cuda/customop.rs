use candle_core::backend::BackendDevice;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::{CudaDevice, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::op::Op;
use candle_core::tensor::from_storage;
use candle_core::CustomOp1;
use candle_core::Device;
use candle_core::Result;
use candle_core::Shape;
use candle_core::Storage;
use candle_core::Tensor;
use candle_core::TensorId;
use std::sync::{Arc, RwLock};

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

pub fn ProjectGaussians(
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
    println!("on est sorti de la fonction de bindings qui appelle cuda");

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

    println!("Cov3d avant le cat : {}", tensor_cov3d);

    let (_, tensor_cov3d_layout) = tensor_cov3d.storage_and_layout();
    println!("layout de cov3d : {:#?}", tensor_cov3d_layout);
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

    println!("tensortot apres cat : {}", tensortot);

    //réécriture de from_storage et copy
    let shape = tensortot.shape();

    println!("shape : {:#?}", shape);

    let tensortot = tensortot.contiguous()?;

    let (storage, layout) = tensortot.storage_and_layout();
    let storage = storage.try_clone(layout)?;

    println!("layout de tensortot: {:#?}", layout);

    let structc = Arc::new(Box::new(c) as Box<dyn CustomOp1 + Send + Sync>);

    let op = BackpropOp::new1(&tensor_in, |s| Op::CustomOp1(s, structc.clone()));
    let tensor_out = from_storage(storage, shape, op, false);

    let (_, tensor_out_layout) = tensor_out.storage_and_layout();
    println!("layout de tensor_out : {:#?}", tensor_out_layout);

    println!("tensor_out apres from_storage : {}", tensor_out);

    //la backpropagation va marcher puisque narrow qui va associer l'opération Backprop narrow aux tenseurs de sorte à ce que les gradients des tenseurs
    //de sortie reforme un tenseur unique de ces gradient, qui pourra etre rentré dans le backward de ProjectGaussian (le struct) qu'on re splitera pour donner au kernel cuda
    //et les gradients de seront remis dans 1 gradient dans bckward de ProjectGaussian(le struct) qui sera ensuite re-split dans les bons tenseur par la backpropagation grace à l'opération Cat

    //ATTENTION : Dans tensor_out il n'y PAS num_tiles_hit (soucis de type)
    //Mais ce n'est pas un pb, il faut juste le prendre en compte

    let cov3d = tensor_out.narrow(1, 0, 6)?;

    println!("cov3d apres narrow: {}", cov3d);

    let xys = tensor_out.narrow(1, 6, 2)?;
    let depth = tensor_out.narrow(1, 8, 1)?;
    let radii = tensor_out.narrow(1, 9, 1)?;
    let conics = tensor_out.narrow(1, 10, 3)?;
    let compensation = tensor_out.narrow(1, 13, 1)?;
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

#[cfg(test)]
mod tests {
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

    /* fn check_close(a: &candle_core::Tensor, b: &candle_core::Tensor, atol: f32, rtol: f32){
        let diff = a.sub(b)?;
        let mut max = diff.max();
        let mut mean = diff.mean();
        if max < 0.0 {
            max = -diff.min();
            mean = -mean;
        }
        assert!(max <= atol,
                "La valeur max de la différence est supérieure à la tolérance : max = {}, atol = {}",
                max,
                atol);
        assert!(mean <= rtol,
                "La valeur moyenne de la différence est supérieure à la tolérance : mean = {}, rtol = {}",
                mean,
                rtol);

    } */

    /*def projection_matrix(fx, fy, W, H, n=0.01, f=1000.0):
    return torch.tensor(
        [
            [2.0 * fx / W, 0.0, 0.0, 0.0],
            [0.0, 2.0 * fy / H, 0.0, 0.0],
            [0.0, 0.0, (f + n) / (f - n), -2 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

    using the results from the python code test_project_gaussians_fwd_small() in order to test
    using only 2 points the rust implementation of the function

     */

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
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = ProjectGaussians(
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
        println!("on est sorti de la fonction");
        /*cov3d:  tensor([[1., 0., 0., 1., 0., 1.],
        [1., 0., 0., 1., 0., 1.]], device='cuda:0')
        xys:  tensor([[255.5000, 255.5000],
                [255.5000, 255.5000]], device='cuda:0')
        depths:  tensor([18., 18.], device='cuda:0')
        radii:  tensor([43, 43], device='cuda:0', dtype=torch.int32)
        conics:  tensor([[0.0049, -0.0000, 0.0049],
                [0.0049, -0.0000, 0.0049]], device='cuda:0')
        compensation:  tensor([0.9985, 0.9985], device='cuda:0')
        num_tiles_hit:  tensor([36, 36], device='cuda:0', dtype=torch.int32) */
        let _python_cov3d = candle_core::Tensor::from_slice(
            &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            &candle_core::Shape::from_dims(&[2, 6]),
            &device,
        )?;
        let _python_xys = candle_core::Tensor::from_slice(
            &[255.5000, 255.5000, 255.5000, 255.5000],
            &candle_core::Shape::from_dims(&[2, 2]),
            &device,
        )?;
        let _python_depths = candle_core::Tensor::from_slice(
            &[18.0, 18.0],
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let _python_radii = candle_core::Tensor::from_slice(
            &[43 as f32, 43 as f32],
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let _python_conics = candle_core::Tensor::from_slice(
            &[0.0049, -0.0000, 0.0049, 0.0049, -0.0000, 0.0049],
            &candle_core::Shape::from_dims(&[2, 3]),
            &device,
        )?;
        let _python_compensation = candle_core::Tensor::from_slice(
            &[0.9985, 0.9985],
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        let _python_num_tiles_hit = candle_core::Tensor::from_slice(
            &[36 as u32, 36 as u32],
            &candle_core::Shape::from_dims(&[2]),
            &device,
        )?;
        println!("cov3d : {}", cov3d);
        println!("xys : {}", xys);
        println!("depths : {}", depths);
        println!("radii : {}", radii);
        println!("conics : {}", conics);
        println!("compensation : {}", compensation);
        println!("num_tiles_hit : {}", num_tiles_hit);

        //check_close(&cov3d, &python_cov3d, 1e-5, 1e-5);
        //check_close(&xys, &python_xys, 1e-5, 1e-5);
        //check_close(&depths, &python_depths, 1e-5, 1e-5);
        //check_close(&radii, &python_radii, 1e-5, 1e-5);
        //check_close(&conics, &python_conics, 1e-5, 1e-5);
        //check_close(&compensation, &python_compensation, 1e-5, 1e-5);
        //check_close(&num_tiles_hit, &python_num_tiles_hit, 1e-5, 1e-5);
        Ok(())
    }

    /* #[test]
    #[ignore]
    fn full_test_project_gaussians_forward(){
        let num_points = 100;
        let means3d = candle_core::Tensor::randn((num_points, 3), &candle_core::Device::cuda(0), true);
        let scales = candle_core::Tensor::rand((num_points, 3), &candle_core::Device::cuda(0)) + 0.2;
        let glob_scale = 1.0;
        let quats = candle_core::Tensor::randn((num_points, 4), &candle_core::Device::cuda(0));
        let quats = quats / quats.norm(candle_core::Norm::L2, &[1], true);
        let H = 512;
        let W = 512;
        let cx = W / 2;
        let cy = H / 2;
        let fx = W / 2;
        let fy = W / 2;
        let clip_thresh = 0.01;
        let viewmat = candle_core::Tensor::from_data(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0], &candle_core::Shape::from_dims(&[4,4]));
        let projmat = projection_matrix(fx, fy, W, H, 0.01, 1000.0);
        let fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = ProjectGaussians(num_points, means3d, scales, glob_scale, quats, viewmat, projmat, fx, fy, cx, cy, H, W, tile_bounds, clip_thresh);
        let masks = num_tiles_hit.gt(0);

        //Ici il faudrait reussir a invoquer le code python
        let (_cov3d, _xys, _depths, _radii, _conics, _compensation, _num_tiles_hit, _masks) = _torch_impl::project_gaussians_forward(means3d, scales, glob_scale, quats, viewmat, fullmat, (fx, fy, cx, cy), (W, H), tile_bounds, clip_thresh);

        check_close(&masks, &_masks, 1e-5, 1e-5);
        check_close(&cov3d, &_cov3d, 1e-5, 1e-5);
        check_close(&xys, &_xys, 1e-5, 1e-5);
        check_close(&depths, &_depths, 1e-5, 1e-5);
        check_close(&radii, &_radii, 1e-5, 1e-5);
        check_close(&conics, &_conics, 1e-5, 1e-5);
        check_close(&compensation, &_compensation, 1e-5, 1e-5);
        check_close(&num_tiles_hit, &_num_tiles_hit, 1e-5, 1e-5);
    }

    #[test]
    #[ignore]
    fn test_dummy_fwd() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let means3d = candle_core::Tensor::randn((100, 3), &candle_core::Device::cuda(0), true);
        let scales = candle_core::Tensor::rand((100, 3), &candle_core::Device::cuda(0)) + 0.2;
        let glob_scale = 1.0;
        let quats = candle_core::Tensor::randn((100, 4), &candle_core::Device::cuda(0));
        let quats = quats / quats.norm(candle_core::norm::L2, &[1], true);
        let viewmat = candle_core::Tensor::from_data(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0], &candle_core::Shape::from_dims(&[4,4]));
        let projmat = projection_matrix(1.0, 1.0, 512, 512, 0.01, 1000.0);
        let fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((512 + BLOCK_X - 1) / BLOCK_X, (512 + BLOCK_Y - 1) / BLOCK_Y, 1);
        let c = ProjectGaussians{glob_scale,1.0,1.0,1.0,1.0,512,512,(1,1,1),0.01,viewmat,projmat};

        let (means3d_storage, means3d_layout) = means3d?.storage_and_layout();
        let (scales_storage, scales_layout) = scales?.storage_and_layout();
        let (quats_storage, quats_layout) = quats?.storage_and_layout();
        let (viewmat_storage, viewmat_layout) = viewmat?.storage_and_layout();
        let (projmat_storage, projmat_layout) = projmat.storage_and_layout();


        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = c.dummy_fwd(means3d_storage, means3d_layout, scales_storage, scales_layout, quats_storage, quats_layout)?;
        Ok(())
    } */
}
