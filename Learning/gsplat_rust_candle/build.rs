#![allow(unused)]
use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;
struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
    include_dirs: &'static [&'static str],
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "src/cuda/kernels/*.cu",
    rust_target: "src/cuda/cuda_kernels.rs",
    include_dirs: &["src/cuda/kernels/helpers.cuh","src/cuda/kernels/config.h","src/cuda/kernels/backward.cuh","src/cuda/kernels/forward.cuh","src/cuda/kernels/glm.hpp","src/cuda/kernels/gtc/type_ptr.hpp"],
}];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/kernels/forward.cu");
    println!("cargo:rerun-if-changed=cuda/kernels/backward.cu");
    println!("cargo:rerun-if-changed=cuda/kernels/forward.cuh");
    println!("cargo:rerun-if-changed=cuda/kernels/backward.cuh");

    #[cfg(feature = "cuda")]
    {
        for kdir in KERNEL_DIRS.iter() {
            let builder = bindgen_cuda::Builder::default()
            .kernel_paths_glob(kdir.kernel_glob)
            .arg("--verbose")
            .arg("-DWIN32_LEAN_AND_MEAN")
            .arg("--expt-relaxed-constexpr")
            .arg("-O3")
            .arg("--use_fast_math");
            println!("cargo:info={builder:?}");
            let bindings = builder.build_ptx().unwrap();
            bindings.write(kdir.rust_target).unwrap()
        }
    }
    Ok(())
}