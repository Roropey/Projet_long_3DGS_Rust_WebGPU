//ET SI ?? : https://github.com/huggingface/candle/blob/main/candle-examples/examples/custom-ops/main.rs

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
    }

}


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
    //mais c'est à revoir, pb de dimension, et voir comment faire pour unstack après, car on est sur les storages...
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
    let tensor_in = stack(&A,0);
    let c = ProjectGaussians{glob_scale,fx,fy,cx,cy,img_height,img_width,tile_bounds,clip_thresh,viewmat,projmat};
    
    //reecriture de la fonction apply_op1_arc de tensor.rs pour bypass le fonctionnement normal
    let (storage, shape) = c.fwd((means3d.storage())?,(scale.storage())?,(quats.storage())?);
    let op = BackpropOp::new1(tensor_in, |s| Op::CustomOp1(s, c.clone()));
    let tensor_out = Ok(from_storage(storage, shape, op, false));

    //A FAIRE : Decomposer tensor_out en les sortie
    //la backpropagation va marcher puisque chunk utilise narrow qui va associer l'opération Backprop narrow aux tenseurs de sorte à ce que les gradients des tenseurs
    //de sortie reforme un tenseur unique de ces gradient, qui pourra etre rentré dans le backward de ProjectGaussian (le struct) qu'on re splitera pour donner au kernel cuda
    //et les gradients de seront remis dans 1 gradient dans bckward de ProjectGaussian(le struct) qui sera ensuite re-split dans les bons tenseur par la backpropagation grace à l'opération Cat
}