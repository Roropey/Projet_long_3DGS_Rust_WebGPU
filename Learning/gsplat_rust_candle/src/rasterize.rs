use candle::Tensor;
use candle_core as candle;

use cuda;
use utils::{bin_and_sort_gaussians, compute_cumulative_intersects};

fn rasterize_gaussians(
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    conics: Tensor,
    num_tiles_hit: Tensor,
    colors: Tensor,
    opacity: Tensor,
    img_height: isize,
    img_width: isize,
    background: Tensor, // When use, put Some(...) and not use, put None
    return_alpha: bool // When use, put Some(...), if not, put None
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
    if colors.dtype() == u8 {
        colors = colors.float() / 255; // Pas sûr que ça fonctionne
    }
    background = background.unwrap_or(Tensor::ones(colors.shape().dims()[colors.shape().rank()-1], f32, colors.device()));
    assert!(background.shape().dims()[0] == colors.shape().dims()[colors.shape().rank()-1], "Incorrect shape of background color tensor, expected shape {}",colors.shape().dims()[colors.shape().rank()-1]);
    assert!(xys.shape().rank()==2 && xys.shape().dims()[1] != 2, "xys, must have dimensions (N,2)");
    assert!(colors.shape().rank() == 2, "colors must have dimensions (N,D)");
    _RasterizeGaussians.apply(
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        img_height,
        img_width,
        background.contiguous(),
        return_alpha,
    )
    // Besoin de définir la classe qui se base sur torch.autograd.Fonction...
    

}