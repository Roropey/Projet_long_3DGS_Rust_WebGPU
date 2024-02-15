//used by cargo to compile the rust code with c++

fn main() {
    cxx_build::bridge("src/cuda/bindings.rs")
        .file("src/cuda/csrc/bindings.cu")
        .cuda(true)
        .flag_if_supported("-std=c++20")
        .compile("gsplat_rust_candle");


    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=torch");
    
    println!("cargo:rerun-if-changed=src/cuda/bindings.rs");
    println!("cargo:rerun-if-changed=src/cuda/csrc/bindings.cu");
    println!("cargo:rerun-if-changed=src/cuda/csrc/bindings.h");
}