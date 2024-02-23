use candle::Tensor;
use candle::Result;
use candle::Storage

/// Projette des gaussiennes 3D dans un espace 2D en utilisant les paramètres spécifiés.
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
/// ou si la projection ne peut pas être réalisée pour une autre raison.

pub fn ProjectGaussians(num_points : int,
    means3d : &candle::Tensor,
    scales: &candle::Tensor,
    glob_scale : float,
    quats : &candle::Tensor,
    viewmat : &candle::Tensor,
    projmat : &candle::Tensor,
    fx : float,
    fy : float,
    cx : float,
    cy : float,
    img_height: uint,
    img_width : uint,
    tile_bounds: &(i32, i32, i32),
    clip_thresh: float) -> Result<(candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor)>
{   
    let max = 0;
    let A = [means3d,scales,quats]
    //Idee pour stack :
    //mais c'est à revoir, pb de dimension
    for arg in &A.iter(){
        if arg.rank() > max{
            max = arg.rank();
        }
    }
    for arg in &A.iter(){
        for 0..(max - arg.rank()){
            arg.unsqueeze(0);
        }
    }
    let tensor_in = stack(&A,1);
    let c = ProjectGaussians{glob_scale,fx,fy,cx,cy,img_height,img_width,tile_bounds,clip_thresh,viewmat,projmat};
    
    //reecriture de la fonction apply_op1_arc de tensor.rs pour bypass le fonctionnement normal
    let (storage_cov3d, shape_cov3d, storage_xys, shape_xys, storage_depth, shape_depth, storage_radii, shape_radii, storage_conics, shape_conics, storage_compensation, shape_compensation, storage_num_tiles_hit, shape_num_tiles_hit) = c.fwd((means3d.storage())?,(scales.storage())?,(quats.storage())?,(viewmat.storage())?,(projmat.storage())?)?;

   
    let tensor_cov3d = from_storage(storage_cov3d, shape_cov3d);
    let tensor_xys = from_storage(storage_xys, shape_xys);
    let tensor_depth = from_storage(storage_depth, shape_depth);
    let tensor_radii = from_storage(storage_radii, shape_radii);
    let tensor_conics = from_storage(storage_conics, shape_conics);
    let tensor_compensation = from_storage(storage_compensation, shape_compensation);
    let tensor_num_tiles_hit = from_storage(storage_num_tiles_hit, shape_num_tiles_hit);
    

    tensortot = cat(&[tensor_cov3d,tensor_xys,tensor_depth,tensor_radii,tensor_conics,tensor_compensation,tensor_num_tiles_hit],1);

    //réécriture de from_storage et copy
    shape = tensortot.shape();
    let storage = tensortot.storage().try_clone(self.layout())?;
    let dtype = storage.dtype();
    let device = storage.device();
    let op = BackpropOp::new1(tensor_in, |s| Op::CustomOp1(s, c.clone()));
    let tensor_out = Tensor::Tensor_ {
        id: TensorId::new(),
        storage: Arc::new(RwLock::new(storage)),
        layout: Layout::contiguous(shape),
        Option::None,
        op,
        dtype,
        device,
    }

    //la backpropagation va marcher puisque narrow qui va associer l'opération Backprop narrow aux tenseurs de sorte à ce que les gradients des tenseurs
    //de sortie reforme un tenseur unique de ces gradient, qui pourra etre rentré dans le backward de ProjectGaussian (le struct) qu'on re splitera pour donner au kernel cuda
    //et les gradients de seront remis dans 1 gradient dans bckward de ProjectGaussian(le struct) qui sera ensuite re-split dans les bons tenseur par la backpropagation grace à l'opération Cat

    let cov3d = tensor_out.narrow(1,0,6);
    let xys = tensor_out.narrow(1,6,2);
    let depth = tensor_out.narrow(1,8,1);
    let radii = tensor_out.narrow(1,9,1);
    let conics = tensor_out.narrow(1,10,3);
    let compensation = tensor_out.narrow(1,13,1);
    let num_tiles_hit = tensor_out.narrow(1,14,1);

    
    
    Ok((cov3d,xys,depth,radii,conics,compensation,num_tiles_hit))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_for_project() -> (candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor,candle::Tensor){
        
        let means3d = candle::Tensor::from_data(&[1.0,0.0,0.0,0.0,1.0,0.0], &candle::Shape::from_dims(&[2,3]));
        //means3d = [[1.0,0.0,0.0],
        //           [0.0,1.0,0.0]]
        
        let scales = candle::Tensor::from_data(&[1.0,1.0,1.0,1.0,1.0,1.0], &candle::Shape::from_dims(&[2,3]));
        //scales = [[1.0,1.0,1.0],
        //          [1.0,1.0,1.0]]
        
        let quats = candle::Tensor::from_data(&[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0], &candle::Shape::from_dims(&[2,4]));
        //quats = [[1.0,1.0,1.0,1.0],
        //         [0.0,0.0,0.0,0.0]]
        
        let viewmat = candle::Tensor::from_data(&[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0], &candle::Shape::from_dims(&[4,4]));
        //viewmat = [[1.0,0.0,0.0,0.0],
        //           [0.0,1.0,0.0,0.0],
        //           [0.0,0.0,1.0,0.0],
        //           [0.0,0.0,0.0,1.0]]
      
        let projmat = candle::Tensor::from_data(&[1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0], &candle::Shape::from_dims(&[4,4]));
        //projmat = [[1.0,0.0,0.0,0.0],
        //           [0.0,1.0,0.0,0.0],
        //           [0.0,0.0,1.0,0.0],
        //           [0.0,0.0,0.0,1.0]]
        
        let glob_scale = 1.0;
        let fx = 1.0;
        let fy = 1.0;
        let cx = 1.0;
        let cy = 1.0;
        let img_height = 1;
        let img_width = 1;
        let tile_bounds = (1,1,1);
        let clip_thresh = 1.0;
        (means3d,scales,quats,viewmat,projmat,glob_scale,fx,fy,cx,cy,img_height,img_width,tile_bounds,clip_thresh)
    }


    fn projection_matrix(fx: float, fy: float, W: uint, H: uint, n: float, f: float) -> candle::Tensor{
        let projmat = candle::Tensor::from_data(&[2.0 * fx / W, 0.0, 0.0, 0.0, 0.0, 2.0 * fy / H, 0.0, 0.0, 0.0, 0.0, (f + n) / (f - n), -2.0 * f * n / (f - n), 0.0, 0.0, 1.0, 0.0], &candle::Shape::from_dims(&[4,4]));
        projmat
    }
    
    fn check_close(a: &candle::Tensor, b: &candle::Tensor, atol: float, rtol: float){
        let diff = a.sub(b);
        let diff = diff.abs();
        let diff = diff.detach();
        let max = diff.max();
        let mean = diff.mean();
        assert!(max <= atol,
                "La valeur max de la différence est supérieure à la tolérance : max = {}, atol = {}", 
                max, 
                atol);
        assert!(mean <= rtol,
                "La valeur moyenne de la différence est supérieure à la tolérance : mean = {}, rtol = {}", 
                mean, 
                rtol);

    }

    #[test]
    fn test_project_gaussians(){
        let (means3d,scales,quats,viewmat,projmat,glob_scale,fx,fy,cx,cy,img_height,img_width,tile_bounds,clip_thresh) = setup_for_project();
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = ProjectGaussians(num_points, means3d, scales, glob_scale, quats, viewmat, projmat, fx, fy, cx, cy, img_height, img_width, tile_bounds, clip_thresh);
        
    }

    #[test]
    fn full_test_project_gaussians_forward(){
        let num_points = 100;
        let means3d = candle::Tensor::randn((num_points, 3), &candle::Device::cuda(0), true);
        let scales = candle::Tensor::rand((num_points, 3), &candle::Device::cuda(0)) + 0.2;
        let glob_scale = 1.0;
        let quats = candle::Tensor::randn((num_points, 4), &candle::Device::cuda(0));
        let quats = quats / quats.norm(candle::Norm::L2, &[1], true);
        let H = 512;
        let W = 512;
        let cx = W / 2;
        let cy = H / 2;
        let fx = W / 2;
        let fy = W / 2;
        let clip_thresh = 0.01;
        let viewmat = candle::Tensor::from_data(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0], &candle::Shape::from_dims(&[4,4]));
        let projmat = projection_matrix(fx, fy, W, H);
        let fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = ProjectGaussians(num_points, means3d, scales, glob_scale, quats, viewmat, projmat, fx, fy, cx, cy, img_height, img_width, tile_bounds, clip_thresh);
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
    fn test_dummy_fwd() -> Result<(), Box<dyn std::error::Error>>{
        let means3d = candle::Tensor::randn((100, 3), &candle::Device::cuda(0), true);
        let scales = candle::Tensor::rand((100, 3), &candle::Device::cuda(0)) + 0.2;
        let glob_scale = 1.0;
        let quats = candle::Tensor::randn((100, 4), &candle::Device::cuda(0));
        let quats = quats / quats.norm(candle::Norm::L2, &[1], true);
        let viewmat = candle::Tensor::from_data(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 1.0], &candle::Shape::from_dims(&[4,4]));
        let projmat = projection_matrix(1.0, 1.0, 512, 512);
        let fullmat = projmat.matmul(&viewmat);
        let BLOCK_X = 16;
        let BLOCK_Y = 16;
        let tile_bounds = ((512 + BLOCK_X - 1) / BLOCK_X, (512 + BLOCK_Y - 1) / BLOCK_Y, 1);
        let c = ProjectGaussians{glob_scale,1.0,1.0,1.0,1.0,512,512,(1,1,1),0.01,viewmat,projmat};
        let (cov3d, xys, depths, radii, conics, compensation, num_tiles_hit) = c.dummy_fwd(means3d.storage()?, scales.storage()?, quats.storage()?, viewmat.storage()?, projmat.storage()?)?;
        0k(())
    }

}