use candle::{CustomOp2, Tensor};
use candle_core as candle;


use crate::utils;

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
    block_width: isize,
    background: Option<Tensor>, // When use, put Some(...) and not use, put None
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
        block_width,
        Some(background.contiguous()),
        Some(return_alpha),
    )
    // Besoin de définir la classe qui se base sur torch.autograd.Fonction...
    

}

struct _RasterizeGaussians(isize,isize,isize,bool); //Ptt ajouter + élément comme dimensions des tensors (D par exemple pour colors)

impl CustomOp2 for _RasterizeGaussians{
    // Proposition :
    // Arg 1 : concat des tensors dependant des gaussiennes (de taille (N,X) où N est pareil pour toutes), en gros tout sauf background...
    // Arg 2 : background car dim diff (voir background comme élément de la struct si non différentiable)
    // C'est une proposition, mais ptt abandonner au vu des nouveaux pushs sur gsplat et de l'ensemble...
    // A voir comment le projection gaussians est géré pour peut-être appliquer la même
    fn name(&self) -> &'static str {
        "rasterize-gaussians"
    }
    fn cpu_fwd(
            &self,
            s1: &candle::CpuStorage,
            l1: &candle::Layout,
            s2: &candle::CpuStorage,
            l2: &candle::Layout,
        ) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
        
    }
    fn cuda_fwd(
            &self,
            _: &candle::CudaStorage,
            _: &candle::Layout,
            _: &candle::CudaStorage,
            _: &candle::Layout,
        ) -> candle::Result<(candle::CudaStorage, candle::Shape)> {
        
    }
    fn bwd(
            &self,
            _arg1: &Tensor,
            _arg2: &Tensor,
            _res: &Tensor,
            _grad_res: &Tensor,
        ) -> candle::Result<(Option<Tensor>, Option<Tensor>)> {
        
    }
}