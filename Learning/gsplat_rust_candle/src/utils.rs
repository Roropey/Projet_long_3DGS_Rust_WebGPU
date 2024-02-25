use candle::{CpuStorage, Device, Layout, Shape, Tensor};
use candle_core as candle;


// Fonction copié sur https://github.com/huggingface/candle/pull/1389/files
// Argsort pour tensor à 1 dimension... A voir si ça marche dans notre cas
// Pas trop compris pourquoi la version cuda fonctionne avec le mm code que la version CPU malgré le changement de "device" (test réalisé)
struct ArgSort;
impl candle::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "arg-sort"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle::Result<(CpuStorage, Shape)> {
        //Verifier qu'une seule dimension (à modifier)
        if layout.shape().rank() != 1 {
            candle::bail!(
                "input should have a single dimension, got {:?}",
                layout.shape()
            )
        }

        let slice = storage.as_slice::<f32>()?;
        // Vérification contiguous
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        
        let mut dst = (0..src.len() as u32).collect::<Vec<u32>>();
        dst.sort_by(|&i, &j| src[i as usize].total_cmp(&src[j as usize]));
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }

    //Version cuda mal propre : passe par host pour faire le trie...
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> candle::Result<(candle::CudaStorage, Shape)> {
        //Verifier qu'une seule dimension (à modifier)
        if layout.shape().rank() != 1 {
            candle::bail!(
                "input should have a single dimension, got {:?}",
                layout.shape()
            )
        }
        // Récupère device
        let dev = storage.device.clone();
        //Récupère le slice mais en cuda
        let cuda_slice = storage.as_cuda_slice::<f32>()?;
        //Copie le slice sur le host (GPU -> CPU)
        let slice = dev.sync_reclaim(cuda_slice.clone()).unwrap();
        // Vérification contiguous
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = (0..src.len() as u32).collect::<Vec<u32>>(); // //
        dst.sort_by(|&i, &j| src[i as usize].total_cmp(&src[j as usize]));
        // Met le résultat sur le GPU
        let dst_cuda = dev.htod_copy(dst).unwrap();        
        let storage = candle::CudaStorage::wrap_cuda_slice(dst_cuda,dev);
        Ok((storage, layout.shape().clone()))
    }
}



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
    (isect_ids, gaussian_ids)

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
    cuda::get_tile_bin_edges(num_intersects,isect_ids_sorted.contiguous(),tile_bounds)
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


    assert!(cov2d.shape().dims()[cov2d.shape().rank()-1]==3, "Execpected Expected input cov2d to be of shape (*batch, 3) (upper triangular values)");
    let num_pts = cov2d.shape().dims()[0];
    assert!(num_pts>0);
    cuda::compute_cov2d_bounds(num_pts,cov2d.contiguous())
}

pub fn compute_cumulative_intersects(
    num_tiles_hit:Tensor
) -> (isize,Tensor){
    /*Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (candle::Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, candle::Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (candle::Tensor): a tensor of cumulated intersections (used for sorting).
    */
    let cum_tiles_hit = num_tiles_hit.cumsum(0).unwrap();
    let num_intersects = cum_tiles_hit.get(cum_tiles_hit.shape().dims()[0]-1).unwrap().to_vec0::<i64>().unwrap() as isize;
    // suppose que cum_tiles_hit n'a qu'une dimension      cum_tiles_hit[-1].item();
    (num_intersects,cum_tiles_hit)
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
    /*Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (isize): number of gaussians.
        num_intersects (isize): cumulative number of total gaussian intersections
        xys (candle::Tensor): x,y locations of 2D gaussian projections.
        depths (candle::Tensor): z depth of gaussians.
        radii (candle::Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (candle::Tensor): list of cumulative tiles hit.
        tile_bounds ((isize,isize,isize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {candle::Tensor, candle::Tensor, candle::Tensor, candle::Tensor, candle::Tensor}:

        - **isect_ids_unsorted** (candle::Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (candle::Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (candle::Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (candle::Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (candle::Tensor): range of gaussians hit per tile.
    */
    let (isect_ids, gaussian_ids )= map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    );
    let sorted_indices = isect_ids.apply_op1(ArgSort).unwrap();
    let isect_ids_sorted = isect_ids.gather(&sorted_indices,0).unwrap();
    //let (isect_ids_sorted, sorted_indices) = torch.sort(isect_ids); // pistes sur https://github.com/huggingface/candle/issues/1359 et https://github.com/huggingface/candle/pull/1389/files
    let gaussian_ids_sorted = gaussian_ids.gather(&sorted_indices,0).unwrap();
    let tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds);
    (isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins)
}