use candle::Tensor;
use candle_core as candle;

use cuda;

fn num_sh_bases(degree:isize) -> isize{
    if degree == 0 {
        1
    } else if degree == 1 {
        4
    } else if degree == 2 {
        9
    } else if degree == 3 {
        16
    } else {
        25
    }
}

fn deg_from_sh(num_bases:isize) -> isize{
    if degree == 1 {
        0
    } else if degree == 4 {
        1
    } else if degree == 9 {
        2
    } else if degree == 16 {
        3
    } else if degree == 25 {
        4
    } else {
        assert!(false,"Invalid number of SH bases")
    }
}

fn spherical_harmonics(
    degrees_to_use:isize,
    viewdirs: Tensor,
    coeffs: Tensor
) -> Tensor {
    /*Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (isize): degree of SHs to use (<= total number available).
        viewdirs (candle::Tensor): viewing directions.
        coeffs (candle::Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    */
    assert!(coeffs.shape().dims()[coeffs.shape().rank()-2] >= num_sh_bases(degrees_to_use));
    _SphericalHarmonics.apply(degrees_to_use,viewdirs.contiguous(),coeffs.contiguous()) // Besoin de définir la classe qui se base sur torch.autograd.Fonction...
}