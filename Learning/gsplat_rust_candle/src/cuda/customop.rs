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
/// * `quats` - Un tensor contenant les quaternions.
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
    let (storage_cov3d, shape_cov3d, storage_xys, shape_xys, storage_depth, shape_depth, storage_radii, shape_radii, storage_conics, shape_conics, storage_compensation, shape_compensation, storage_num_tiles_hit, shape_num_tiles_hit) = c.fwd((means3d.storage())?,(scales.storage())?,(quats.storage())?,(viewmat.storage())?,(projmat.storage())?);

   
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