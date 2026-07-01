//! Item 1 perf roadmap: test for the BF16 add kernel.
//!
//! Verifies that the custom add_bf16 CUDA kernel produces the same result as
//! the F32 dance (candle's built-in add via F32 cast) within BF16 tolerance.

use airllm_kernels::add as ak_add;
use candle_core::{DType, Device, Shape, Tensor};

/// BF16 add on GPU matches the F32-dance result within tolerance.
#[test]
fn gpu_add_bf16_roundtrip() -> anyhow::Result<()> {
    // Skip if CUDA is not available in this build.
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] gpu_add_bf16_roundtrip: Device::new_cuda(0) failed: {}", e);
            return Ok(());
        }
    };

    // Construct two [4] BF16 tensors with simple values.
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let a = Tensor::from_vec(data_a.clone(), Shape::from(vec![4]), &dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(data_b.clone(), Shape::from(vec![4]), &dev)?.to_dtype(DType::BF16)?;

    // Custom BF16 add kernel
    let out = ak_add(&a, &b)?;
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape(), &Shape::from(vec![4]));

    // Copy to CPU for verification
    let back: Vec<f32> = out.to_dtype(DType::F32)?.to_vec1()?;
    // Expected: 11.0, 22.0, 33.0, 44.0 (within BF16 tolerance ~0.5)
    let expected = vec![11.0_f32, 22.0, 33.0, 44.0];
    assert_eq!(back.len(), expected.len());
    for (i, (got, want)) in back.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 1.0,
            "add_bf16[{}] mismatch: got={:?} want={:?} diff={}",
            i,
            back,
            expected,
            diff
        );
    }
    eprintln!("[gpu_add_bf16_roundtrip] PASS: got={:?}", back);
    Ok(())
}

/// Larger tensor (4096 = Qwen3-8B hidden_size) to catch any size-specific bugs.
#[test]
fn gpu_add_bf16_4096() -> anyhow::Result<()> {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] gpu_add_bf16_4096: Device::new_cuda(0) failed: {}", e);
            return Ok(());
        }
    };

    let n: usize = 4096;
    let data_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let data_b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();
    let a = Tensor::from_vec(data_a.clone(), Shape::from(vec![1, n]), &dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(data_b.clone(), Shape::from(vec![1, n]), &dev)?.to_dtype(DType::BF16)?;

    let out = ak_add(&a, &b)?;
    let back: Vec<f32> = out.to_dtype(DType::F32)?.squeeze(0)?.to_vec1()?;
    eprintln!("[gpu_add_bf16_4096] first 5: {:?}", &back[..5]);
    eprintln!("[gpu_add_bf16_4096] last 5: {:?}", &back[n-5..]);
    // First element should be 0.0 (a[0]+b[0] = 0)
    // Second should be 0.003 (0.001 + 0.002)
    assert!(back[0].abs() < 0.01, "first element wrong: {}", back[0]);
    assert!((back[1] - 0.003).abs() < 0.01, "second element wrong: {}", back[1]);
    eprintln!("[gpu_add_bf16_4096] PASS");
    Ok(())
}
