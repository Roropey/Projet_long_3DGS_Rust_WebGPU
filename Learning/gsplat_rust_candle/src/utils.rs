use candle::{CpuStorage, Device, Storage, Layout, Shape, Tensor, D, Result, tensor::from_storage, op::BackpropOp};
use candle::backend::{BackendDevice,BackendStorage};
use candle::cuda_backend::cudarc::driver::{LaunchAsync,LaunchConfig};
use candle::cuda_backend::{WrapErr,CudaDevice};
use candle_core as candle;
use std::ops::Not;
//#[cfg(feature = "cuda")]
use crate::cuda::cuda_kernels::{FORWARD, BINDINGS};

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

        let slice = storage.as_slice::<f32>().unwrap();
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
        let cuda_slice = storage.as_cuda_slice::<f32>().unwrap();
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

fn to_cuda_storage(storage: &Storage,layout : &Layout) -> Result<candle_core::CudaStorage> {
    match storage {
        Storage::Cuda(s) => Ok(s.try_clone(layout).unwrap()),
        _ => unreachable!(),
    }
}

pub fn map_gaussian_to_intersects(
    num_points:usize,
    num_intersects:usize,
    xys: &Tensor,
    depths:&Tensor,
    radii:&Tensor,
    cum_tiles_hit:&Tensor,
    tile_bounds:(usize,usize,usize),
    block_size: usize
) -> Result<(Tensor,Tensor)>{
    /* Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (usize): number of gaussians.
        num_intersects (usize): total number of tile intersections.
        xys (candle::Tensor): x,y locations of 2D gaussian projections.
        depths (candle::Tensor): z depth of gaussians.
        radii (candle::Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (candle::Tensor): list of cumulative tiles hit.
        tile_bounds ((usize,usize,usize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
        block_size (usize) : size of the block (addition made based on the most recent version of gsplat git but not use since no modification from the cuda files)

    Returns:
        A tuple of {candle::Tensor, candle::Tensor}:

        - **isect_ids** (candle::Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (candle::Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    */


    let (xys_storage,xys_layout) = xys.storage_and_layout();
    let xys_storage: candle::CudaStorage = to_cuda_storage(&xys_storage, &xys_layout).unwrap();
    let (depths_storage,depths_layout) = depths.storage_and_layout();
    let depths_storage: candle::CudaStorage = to_cuda_storage(&depths_storage, &depths_layout).unwrap();
    let (radii_storage,radii_layout) = radii.storage_and_layout();
    let radii_storage: candle::CudaStorage = to_cuda_storage(&radii_storage, &radii_layout).unwrap();
    let (cum_tiles_hit_storage,cum_tiles_hit_layout) = cum_tiles_hit.storage_and_layout();
    let cum_tiles_hit_storage: candle::CudaStorage = to_cuda_storage(&cum_tiles_hit_storage, &cum_tiles_hit_layout).unwrap();
    let dev_xys = xys_storage.device().clone();
    let dev_depths = depths_storage.device().clone();
    let dev_radii = radii_storage.device().clone();
    let dev_cum_tiles_hit = cum_tiles_hit_storage.device().clone();
    if (dev_xys.same_device(&dev_depths) && dev_xys.same_device(&dev_radii) && dev_xys.same_device(&dev_cum_tiles_hit)).not() {
        candle_core::bail!("all inputs must be on the same device")
        
    }
    let dev = dev_xys;

    let slice_xys = xys_storage.as_cuda_slice::<f32>().unwrap();
    let slice_xys = match xys_layout.contiguous_offsets() {
        None => candle_core::bail!("xys input has to be contiguous"),
        Some((o1, o2)) => slice_xys.slice(o1..o2),
    };

    let slice_depths = depths_storage.as_cuda_slice::<f32>().unwrap();
    let slice_depths = match depths_layout.contiguous_offsets() {
        None => candle_core::bail!("depths input has to be contiguous"),
        Some((o1, o2)) => slice_depths.slice(o1..o2),
    };

    let slice_radii = radii_storage.as_cuda_slice::<f32>().unwrap();
    let slice_radii = match radii_layout.contiguous_offsets() {
        None => candle_core::bail!("radii input has to be contiguous"),
        Some((o1, o2)) => slice_radii.slice(o1..o2),
    };

    let slice_cum_tiles_hit = cum_tiles_hit_storage.as_cuda_slice::<u32>().unwrap();
    let slice_cum_tiles_hit = match cum_tiles_hit_layout.contiguous_offsets() {
        None => candle_core::bail!("cum_tiles_hit input has to be contiguous"),
        Some((o1, o2)) => slice_cum_tiles_hit.slice(o1..o2),
    };

    let func = dev.get_or_load_func("map_gaussian_to_intersects",FORWARD).unwrap();
    let dst_isect_ids = unsafe { dev.alloc::<i64>(num_intersects as usize) }.w().unwrap();
    let dst_gaussian_ids = unsafe { dev.alloc::<i64>(num_intersects as usize) }.w().unwrap();
    let params =(
        num_points,
        &slice_xys,
        &slice_depths,
        &slice_radii,
        &slice_cum_tiles_hit,
        tile_bounds.0,
        tile_bounds.1,
        tile_bounds.2,
        &dst_isect_ids,
        &dst_gaussian_ids
    );
    let n_threads = 1024;
    let nb_block = (num_points + n_threads - 1 )/n_threads ;
    let cfg = LaunchConfig{
        grid_dim:(nb_block as u32,1,1),
        block_dim:(n_threads as u32,1,1),
        shared_mem_bytes: 0,
    };
    unsafe { func.launch(cfg, params) }.w().unwrap();
    let isect_ids_storage = candle::CudaStorage::wrap_cuda_slice(dst_isect_ids,dev.clone());
    let gaussian_ids_storage = candle::CudaStorage::wrap_cuda_slice(dst_gaussian_ids,dev);
    let isect_ids = from_storage(candle_core::Storage::Cuda(isect_ids_storage),Shape::from_dims(&[num_intersects as usize]), BackpropOp::none(),false);
    let gaussian_ids = from_storage(candle_core::Storage::Cuda(gaussian_ids_storage),Shape::from_dims(&[num_intersects as usize]), BackpropOp::none(),false);
    Ok((isect_ids, gaussian_ids))

}

pub fn get_tile_bin_edges(
    num_intersects:usize,
    isect_ids_sorted: &Tensor,
    tile_bounds:(usize,usize,usize)
) -> Result<Tensor> {
    /* Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (usize): total number of gaussian intersects.
        isect_ids_sorted (candle::Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds ((usize,usize,usize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (candle::Tensor): range of gaussians IDs hit per tile.
    */
    let (isect_ids_sorted_storage,isect_ids_sorted_layout) = isect_ids_sorted.storage_and_layout();
    let isect_ids_sorted_storage: candle::CudaStorage = to_cuda_storage(&isect_ids_sorted_storage, &isect_ids_sorted_layout).unwrap();
    let dev = isect_ids_sorted_storage.device().clone();
    let slice = isect_ids_sorted_storage.as_cuda_slice::<i64>().unwrap();
    let slice = match isect_ids_sorted_layout.contiguous_offsets() {
        None => candle_core::bail!("isect_ids_sorted input has to be contiguous"),
        Some((o1, o2)) => slice.slice(o1..o2),
    };
    let func = dev.get_or_load_func("get_tile_bin_edges",FORWARD).unwrap();
    let (tile_x,tile_y,_) = tile_bounds;
    let dst_tile_bins = unsafe { dev.alloc::<i64>((tile_x*tile_y*2) as usize) }.w().unwrap();
    let params = (num_intersects,&slice,&dst_tile_bins);
    let n_threads = 1024;
    let nb_block = (num_intersects + n_threads - 1 )/n_threads ;
    let cfg = LaunchConfig{
        grid_dim:(nb_block as u32,1,1),
        block_dim:(n_threads as u32,1,1),
        shared_mem_bytes: 0,
    };
    unsafe { func.launch(cfg, params) }.w().unwrap();
    let tile_bins_storage = candle::CudaStorage::wrap_cuda_slice(dst_tile_bins,dev);
    let tile_bins = from_storage(candle_core::Storage::Cuda(tile_bins_storage),Shape::from_dims(&[(tile_x*tile_y) as usize,2]), BackpropOp::none(),false);
    

    Ok(tile_bins)
}




pub fn compute_cov2d_bounds(
    cov2d:&Tensor
) -> Result<(Tensor,Tensor)>{

    /*Computes bounds of 2D covariance matrix

    Args:
        cov2d (candle::Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {candle::Tensor, Tecandle::Tensornsor}:

        - **conic** (candle::Tensor): conic parameters for 2D gaussian.
        - **radii** (candle::Tensor): radii of 2D gaussian projections.
    */


    assert!(cov2d.dim(D::Minus1).unwrap()==3, "Execpected Expected input cov2d to be of shape (*batch, 3) (upper triangular values)");
    let num_pts = cov2d.dim(0).unwrap();
    let cov2_dim1 = cov2d.dim(1).unwrap();
    assert!(num_pts>0);

    // Partie appel fonction CUDA/C++
    let (cov2d_storage, cov2d_layout) = cov2d.storage_and_layout();
    let cov2d_storage: candle::CudaStorage = to_cuda_storage(&cov2d_storage, &cov2d_layout).unwrap();
    let dev = cov2d_storage.device().clone();
    let slice_cov2d = cov2d_storage.as_cuda_slice::<f32>().unwrap();
    let slice_cov2d = match cov2d_layout.contiguous_offsets(){
        None => candle::bail!("cov 2d input has to be contiguous"),
        Some((o1, o2)) => slice_cov2d.slice(o1..o2),
    };
    let func = dev.get_or_load_func("compute_cov2d_bounds_kernel",BINDINGS).unwrap();
    let dst_conics = unsafe { dev.alloc::<f32>(num_pts*cov2_dim1) }.w().unwrap();
    let dst_radii = unsafe { dev.alloc::<f32>(num_pts) }.w().unwrap();
    let params = (
        num_pts as u32,
        &slice_cov2d,
        &dst_conics,
        &dst_radii
    );
    let n_threads = 1024;
    let nb_block = (num_pts + n_threads - 1 )/n_threads ;
    let cfg = LaunchConfig{
        grid_dim:(nb_block as u32,1,1),
        block_dim:(n_threads as u32,1,1),
        shared_mem_bytes: 0,
    };
    unsafe { func.launch(cfg, params) }.w().unwrap();
    let conics_storage = candle::CudaStorage::wrap_cuda_slice(dst_conics,dev.clone());
    let radii_storage = candle::CudaStorage::wrap_cuda_slice(dst_radii,dev);

    let conics = from_storage(candle_core::Storage::Cuda(conics_storage), Shape::from_dims(&[num_pts,cov2_dim1]), BackpropOp::none(),false);
    let radii = from_storage(candle_core::Storage::Cuda(radii_storage), Shape::from_dims(&[num_pts]), BackpropOp::none(),false);
    Ok((conics,radii))

}

pub fn compute_cumulative_intersects(
    num_tiles_hit:&Tensor
) -> Result<(usize,Tensor)>{
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
    let num_tiles_hit = num_tiles_hit.to_dtype(candle::DType::F64)?;
    let cum_tiles_hit = num_tiles_hit.cumsum(0).unwrap();
    let cum_tiles_hit = cum_tiles_hit.to_dtype(candle::DType::U32)?;
    println!("cum_tiles_hit info {:?}",cum_tiles_hit.get(cum_tiles_hit.dim(0).unwrap()-1).unwrap().to_vec1::<u32>().unwrap()[0]);
    let num_intersects = cum_tiles_hit.get(cum_tiles_hit.dim(0).unwrap()-1).unwrap().to_vec1::<u32>().unwrap()[0] as usize;
    // suppose que cum_tiles_hit n'a qu'une dimension      cum_tiles_hit[-1].item();
    Ok((num_intersects,cum_tiles_hit))
}

pub fn bin_and_sort_gaussians(
    num_points:usize,
    num_intersects:usize,
    xys:&Tensor,
    depths:&Tensor,
    radii:&Tensor,
    cum_tiles_hit:&Tensor,
    tile_bounds:(usize,usize,usize),
    block_size: usize
) -> Result<(Tensor,Tensor,Tensor,Tensor,Tensor)>{
    /*Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (usize): number of gaussians.
        num_intersects (usize): cumulative number of total gaussian intersections
        xys (candle::Tensor): x,y locations of 2D gaussian projections.
        depths (candle::Tensor): z depth of gaussians.
        radii (candle::Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (candle::Tensor): list of cumulative tiles hit.
        tile_bounds ((usize,usize,usize)): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {candle::Tensor, candle::Tensor, candle::Tensor, candle::Tensor, candle::Tensor}:

        - **isect_ids_unsorted** (candle::Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (candle::Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (candle::Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (candle::Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (candle::Tensor): range of gaussians hit per tile.
    */
    let (isect_ids, gaussian_ids )= map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds, block_size
    ).unwrap();
    let isect_ids = isect_ids.to_dtype(candle::DType::F32)?;
    let sorted_indices = isect_ids.apply_op1(ArgSort).unwrap();
    let isect_ids = isect_ids.to_dtype(candle::DType::I64)?;
    let isect_ids_sorted = isect_ids.gather(&sorted_indices,0).unwrap();
    //let (isect_ids_sorted, sorted_indices) = torch.sort(isect_ids); // pistes sur https://github.com/huggingface/candle/issues/1359 et https://github.com/huggingface/candle/pull/1389/files
    let gaussian_ids_sorted = gaussian_ids.gather(&sorted_indices,0).unwrap();
    let tile_bins = get_tile_bin_edges(num_intersects, &isect_ids_sorted, tile_bounds).unwrap();
    Ok((isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins))
}