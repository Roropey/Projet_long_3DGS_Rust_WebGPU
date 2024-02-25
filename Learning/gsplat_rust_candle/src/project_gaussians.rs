use candle::Tensor;
use candle_core as candle;

mod cuda::customop.rs;


fn project_gaussian(
    means3d: Tensor,
    scales: Tensor,
    glob_scale: f32,
    quats: Tensor,
    viewmat: Tensor,
    projmat: Tensor,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    img_height: isize,
    img_width: isize,
    tile_bounds: (isize,isize,isize),
    clip_thresh: Option<f32> // Put value in Some(...), if none, put None
) -> (Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor){
    /*This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (f32): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (f32): focal length x.
       fy (f32): focal length y.
       cx (f32): principal point x.
       cy (f32): principal point y.
       img_height (isize): height of the rendered image.
       img_width (isize): width of the rendered image.
       tile_bounds ((isize,isize,isize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
       clip_thresh (f32): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    */
    let clip_thresh = clip_thresh.unwrap_or(0.01);
    _ProjectGaussians.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        viewmat.contiguous(),
        projmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        tile_bounds,
        clip_thresh,
    )
    // Besoin de définir la classe qui se base sur torch.autograd.Fonction...

}