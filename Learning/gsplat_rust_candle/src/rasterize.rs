use candle::{CustomOp2, Tensor};
use candle_core as candle;


use crate::utils;

fn rasterize_gaussians(
    xys: &Tensor,
    depths: &Tensor,
    radii: &Tensor,
    conics: &Tensor,
    num_tiles_hit: &Tensor,
    colors: &Tensor,
    opacity: &Tensor,
    img_height: isize,
    img_width: isize,
    block_width: isize,
    background: Option<&Tensor>, // When use, put Some(...) and not use, put None
    return_alpha: Option<bool> // When use, put Some(...), if not, put None
) -> Tensor {
    /*Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (isize): height of the rendered image.
        img_width (isize): width of the rendered image.
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    */
    if colors.dtype() == candle::DType::U8 {
        colors = colors.float() / 255; // Pas sûr que ça fonctionne
    }
    let background = background.unwrap_or(Tensor::ones(colors.shape().dims()[colors.shape().rank()-1], candle::DType::F64, colors.device()).unwrap());
    let return_alpha = return_alpha.unwrap_or(false);
    assert!(background.shape().dims()[0] == colors.shape().dims()[colors.shape().rank()-1], "Incorrect shape of background color tensor, expected shape {}",colors.shape().dims()[colors.shape().rank()-1]);
    assert!(xys.shape().rank()==2 && xys.shape().dims()[1] == 2, "xys, must have dimensions (N,2)");
    assert!(colors.shape().rank() == 2, "colors must have dimensions (N,D)");
    crate::cuda::customop::RasterizeGaussians(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        img_height,
        img_width,
        block_width,
        Some(background),
        Some(return_alpha),
    )
    // Besoin de définir la classe qui se base sur torch.autograd.Fonction...
    

}
