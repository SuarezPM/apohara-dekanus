// build.rs for airllm-kernels: invoke nvcc to compile .cu kernels into PTX
// for sm_75 (RTX 2060 SUPER). Mirrors vendor/candle-kernels/build.rs pattern
// using cudaforge to drive the compile + generate Rust FFI bindings.
//
// Phase 2b GPU unblock: produces PTX + extern "C" launchers consumed by
// cudarc at runtime.

use std::path::PathBuf;

fn main() {
    let kernels_dir = PathBuf::from("kernels");
    if !kernels_dir.exists() {
        eprintln!("airllm-kernels build: no kernels/ directory; skipping CUDA build");
        return;
    }

    // Re-run if any kernel changes
    println!("cargo:rerun-if-changed=kernels/");
    for entry in std::fs::read_dir(&kernels_dir).unwrap() {
        let entry = entry.unwrap();
        if let Some(ext) = entry.path().extension() {
            if ext == "cu" || ext == "cuh" {
                println!("cargo:rerun-if-changed={}", entry.path().display());
            }
        }
    }

    // Detect nvcc
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| {
        if std::path::Path::new("/opt/cuda/bin/nvcc").exists() {
            "/opt/cuda/bin/nvcc".to_string()
        } else {
            "nvcc".to_string()
        }
    });
    if std::process::Command::new(&nvcc).arg("--version").output().is_err() {
        eprintln!(
            "airllm-kernels build: nvcc not found at {}; skipping CUDA build \
             (CPU-only path will be used)",
            nvcc
        );
        return;
    }

    // Compile each .cu to PTX (sm_75)
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for entry in std::fs::read_dir(&kernels_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let ptx_path = out_dir.join(format!("{stem}.ptx"));
        eprintln!(
            "airllm-kernels build: compiling {} -> {}",
            path.display(),
            ptx_path.display()
        );
        let status = std::process::Command::new(&nvcc)
            .arg("-arch=sm_75")
            .arg("-ptx")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-std=c++17")
            .arg("-I").arg(&kernels_dir)
            .arg(&path)
            .arg("-o").arg(&ptx_path)
            .status();
        match status {
            Ok(s) if s.success() => {
                eprintln!("airllm-kernels build: OK {}", ptx_path.display());
            }
            Ok(s) => eprintln!("airllm-kernels build: nvcc exit {:?}", s.code()),
            Err(e) => eprintln!("airllm-kernels build: nvcc spawn err: {}", e),
        }
    }

    // Generate Rust FFI bindings (extern "C" declarations matching the
    // launcher signatures in each .cu file). For Phase 2b PoC this is
    // hand-written in src/ffi.rs; future versions can switch to bindgen.
    println!("cargo:rerun-if-changed=src/ffi.rs");
}
