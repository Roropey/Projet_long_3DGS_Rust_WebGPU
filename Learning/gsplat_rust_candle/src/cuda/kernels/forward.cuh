
#include <cstdint>

// compute the 2d gaussian parameters from 3d gaussian parameters
extern "C" __global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float* __restrict__ projmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_width,
    const unsigned img_height,
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned tile_bounds_z,
    const float clip_thresh,
    float* __restrict__ covs3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    float* __restrict__ radii,
    float3* __restrict__ conics,
    float* __restrict__ compensation,
    int64_t* __restrict__ num_tiles_hit
);

// compute output color image from binned and sorted gaussians
extern "C" __global__ void rasterize_forward(
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned tile_bounds_z,
    const unsigned img_size_x,
    const unsigned img_size_y,
    const unsigned img_size_z,
    const int64_t* __restrict__ gaussian_ids_sorted,
    const float2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    float* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3* __restrict__ background
);

// compute output color image from binned and sorted gaussians
extern "C" __global__ void nd_rasterize_forward(
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned tile_bounds_z,
    const unsigned img_size_x,
    const unsigned img_size_y,
    const unsigned img_size_z,
    const unsigned channels,
    const int64_t* __restrict__ gaussian_ids_sorted,
    const float2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    float* __restrict__ final_index,
    float* __restrict__ out_img,
    const float* __restrict__ background
);

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ void project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 &cov2d,
    float &comp
);

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
);

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int64_t* __restrict__ cum_tiles_hit,
    const unsigned tile_bounds_x,
    const unsigned tile_bounds_y,
    const unsigned tile_bounds_z,
    int64_t* __restrict__ isect_ids,
    int64_t* __restrict__ gaussian_ids
);

extern "C" __global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, float2* __restrict__ tile_bins
);