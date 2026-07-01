//! Phase 1a of v2.2 plan: TDD test for the BF16 multiply kernel.

use airllm_kernels::mul as ak_mul;
use candle_core::{DType, Device, Shape, Tensor};

/// BF16 mul on GPU matches the F32-dance result within tolerance.
#[test]
fn gpu_mul_bf16_roundtrip() -> anyhow::Result<()> {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] gpu_mul_bf16_roundtrip: Device::new_cuda(0) failed: {}", e);
            return Ok(());
        }
    };

    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let a = Tensor::from_vec(data_a.clone(), Shape::from(vec![4]), &dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(data_b.clone(), Shape::from(vec![4]), &dev)?.to_dtype(DType::BF16)?;

    let out = ak_mul(&a, &b)?;
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape(), &Shape::from(vec![4]));

    let back: Vec<f32> = out.to_dtype(DType::F32)?.to_vec1()?;
    let expected = vec![10.0_f32, 40.0, 90.0, 160.0];
    assert_eq!(back.len(), expected.len());
    for (i, (got, want)) in back.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 1.0,
            "mul_bf16[{}] mismatch: got={:?} want={:?} diff={}",
            i,
            back,
            expected,
            diff
        );
    }
    eprintln!("[gpu_mul_bf16_roundtrip] PASS: got={:?}", back);
    Ok(())
}

/// Larger tensor (4096 = Qwen3-8B hidden_size).
#[test]
fn gpu_mul_bf16_4096() -> anyhow::Result<()> {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[skip] gpu_mul_bf16_4096: Device::new_cuda(0) failed: {}", e);
            return Ok(());
        }
    };

    let n: usize = 4096;
    let data_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let data_b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();
    let a = Tensor::from_vec(data_a.clone(), Shape::from(vec![1, n]), &dev)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(data_b.clone(), Shape::from(vec![1, n]), &dev)?.to_dtype(DType::BF16)?;

    let out = ak_mul(&a, &b)?;
    let back: Vec<f32> = out.to_dtype(DType::F32)?.squeeze(0)?.to_vec1()?;
    eprintln!("[gpu_mul_bf16_4096] first 5: {:?}", &back[..5]);
    // First element: 0.0 * 0.0 = 0.0
    // Second: 0.001 * 0.002 = 2e-6
    assert!(back[0].abs() < 0.01, "first element wrong: {}", back[0]);
    assert!((back[1] - 2e-6).abs() < 0.01, "second element wrong: {}", back[1]);
    eprintln!("[gpu_mul_bf16_4096] PASS");
    Ok(())
}
