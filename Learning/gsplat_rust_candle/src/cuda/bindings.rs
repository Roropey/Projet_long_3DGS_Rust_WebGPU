#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("gsplat_rust_candle/src/cuda/csrc/bindings.h");

        type torchTensor;

        pub fn project_gaussians_forward(num_points : int,
            means3d : &torchTensor,
            scales: &torchTensor,
            glob_scale : float,
            quats : &torchTensor,
            viewmat : &torchTensor,
            projmat : &torchTensor,
            fx : float,
            fy : float,
            cx : float,
            cy : float,
            img_height: uint,
            img_width : uint,
            // a changer : const std::tuple<int, int, int> tile_bounds,
            clip_thresh: float) -> UniquePtr<torchTensor>;//ptetre sharedPtr
    }

    fn to_candle_tensor(UniquePtr<torchTensor> torch_tensor) -> candle::Tensor{
        tch_tensor = tch::Tensor::from_ptr(torch_tensor.as_ref());
        //candle_tensor =  qqchose cf : https://github.com/huggingface/candle/issues/973
    }

    fn to_torch_tensor(candle::Tensor candle_tensor) -> torchTensor{
        //tch_tensor = qqchose
        tch_tensor.as_ptr()
    }

}
pub fn project_gaussians_forward(num_points : int,
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
    // a changer : const std::tuple<int, int, int> tile_bounds,
    clip_thresh: float) -> candle::Tensor
{
    torch_tensor = ffi::project_gaussians_forward(num_points,
        to_torch_tensor(means3d),
        to_torch_tensor(scales),
        glob_scale,
        to_torch_tensor(quats),
        to_torch_tensor(viewmat),
        to_torch_tensor(projmat),
        fx : float,
        fy : float,
        cx : float,
        cy : float,
        img_height: uint,
        img_width : uint,
        // a changer : const std::tuple<int, int, int> tile_bounds,
        clip_thresh: float);
    to_candle_tensor(torch_tensor)
}