use candle::Tensor;

pub mod cuda;

pub fn map_gaussian_to_intersects(
    num_points:isize,
    num_intersects:isize,
    xys: Tensor,
    depths:Tensor,
    radii:Tensor,
    cum_tiles_hit:Tensor,
    tile_bounds:(isize,isize,isize)
) -> (Tensor,Tensor){
    /* Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (isize): number of gaussians.
        num_intersects (isize): total number of tile intersections.
        xys (candle::Tensor): x,y locations of 2D gaussian projections.
        depths (candle::Tensor): z depth of gaussians.
        radii (candle::Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (candle::Tensor): list of cumulative tiles hit.
        tile_bounds ((isize,isize,isize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {candle::Tensor, candle::Tensor}:

        - **isect_ids** (candle::Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (candle::Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    */
    let (isect_ids, gaussian_ids) = cuda::map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds
    );
    return (isect_ids, gaussian_ids);

}

pub fn get_tile_bin_edges(
    num_intersects:isize,
    isect_ids_sorted: Tensor,
    tile_bounds:(isize,isize,isize)
) -> Tensor {
    /* Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (isize): total number of gaussian intersects.
        isect_ids_sorted (candle::Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds ((isize,isize,isize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (candle::Tensor): range of gaussians IDs hit per tile.
    */
    return cuda::get_tile_bin_edges(num_intersects,isect_ids_sorted.contiguous(),tile_bounds);
    // /!\ isect_ids_sorted tensor avec appel 
}

pub fn compute_cov2d_bounds(
    cov2d:Tensor
) -> (Tensor,Tensor){

    /*Computes bounds of 2D covariance matrix

    Args:
        cov2d (candle::Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {candle::Tensor, Tecandle::Tensornsor}:

        - **conic** (candle::Tensor): conic parameters for 2D gaussian.
        - **radii** (candle::Tensor): radii of 2D gaussian projections.
    */


    assert!(cov2d.shape().dims()[cov2d.shape().dims().len()-1]==3, "Execpected Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {}",tuple(cov2d.shape));
    let num_pts = cov2d.shape().dims()[0];
    assert!(num_pts>0);
    return cuda::compute_cov2d_bounds(num_pts,cov2d.contiguous());
}

pub fn compute_cumulative_intersects(
    num_tiles_hit:Tensor
) -> (isize,Tensor){
    /*Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    */
    let cum_tiles_hit = num_tiles_hit.cumsum(0);
    let num_intersects = cum_tiles_hit.get(cum_tiles_hit.shape().dims()[0]-1).unwrap().to_vec0::<isize>().unwrap();
    // suppose que cum_tiles_hit n'a qu'une dimension      cum_tiles_hit[-1].item();
    return (num_intersects,cum_tiles_hit);
}

pub fn bin_and_sort_gaussians(
    num_points:isize,
    num_intersects:isize,
    xys:Tensor,
    depths:Tensor,
    radii:Tensor,
    cum_tiles_hit:Tensor,
    tile_bounds:(isize,isize,isize)
) -> (Tensor,Tensor,Tensor,Tensor,Tensor){
    let (isect_ids, gaussian_ids )= map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    );
    let (isect_ids_sorted, sorted_indices) = torch.sort(isect_ids); // Pas encore trouver comment reproduire le sort, pistes sur https://github.com/huggingface/candle/issues/1359 et https://github.com/huggingface/candle/pull/1389/files
    let gaussian_ids_sorted = gaussian_ids.gather(sorted_indices,0);
    let tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds);
    return (isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins);
}