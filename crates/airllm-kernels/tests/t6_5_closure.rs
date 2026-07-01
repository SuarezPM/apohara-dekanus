//! T6.5 closure: regression tests for airllm-kernels public API.
//!
//! These tests exercise the CPU path of `narrow`, `stack`, and `reshape`,
//! which delegate to candle-core's built-in implementations. They serve
//! as the regression guard for the public API surface after the T6.5
//! custom CUDA kernel wiring (F32 narrow + BF16 narrow specializations).
//!
//! The GPU path is exercised by `gpu_narrow_bf16_roundtrip` below, gated
//! on a working CUDA runtime. If the build has no GPU available, that
//! test returns `Ok(())` (skip) and the honest gap is filed in AUDIT.md.

use airllm_kernels::{narrow, reshape, stack};
use candle_core::{DType, Device, Shape, Tensor};

/// CPU regression: narrow a 2x3 tensor along dim=0 with start=1, length=1.
#[test]
fn cpu_narrow_2x3_dim0() -> anyhow::Result<()> {
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)?;
    let out = narrow(&t, 0, 1, 1)?;
    assert_eq!(out.shape(), &Shape::from(vec![1, 3]));
    let v: Vec<Vec<f32>> = out.to_vec2()?;
    assert_eq!(v, vec![vec![4.0, 5.0, 6.0]]);
    Ok(())
}

/// CPU regression: narrow a 2x3 tensor along dim=1 with start=0, length=2.
#[test]
fn cpu_narrow_2x3_dim1() -> anyhow::Result<()> {
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)?;
    let out = narrow(&t, 1, 0, 2)?;
    assert_eq!(out.shape(), &Shape::from(vec![2, 2]));
    let v: Vec<Vec<f32>> = out.to_vec2()?;
    assert_eq!(v, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
    Ok(())
}

/// CPU regression: stack two 1x2 tensors along dim=0 → 2x1x2.
///
/// Note: stacking 1x2 tensors along dim=0 preserves rank (the stack dim
/// is prepended with size N). For a 2x2 result, the inputs would need
/// to be rank-1 vectors of length 2.
#[test]
fn cpu_stack_two_1x2() -> anyhow::Result<()> {
    let a = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu)?;
    let b = Tensor::new(&[[3.0f32, 4.0]], &Device::Cpu)?;
    let out = stack(&[&a, &b], 0)?;
    assert_eq!(out.shape(), &Shape::from(vec![2, 1, 2]));
    let v: Vec<Vec<Vec<f32>>> = out.to_vec3()?;
    assert_eq!(v, vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]]);
    Ok(())
}

/// CPU regression: reshape 1x6 → 2x3.
#[test]
fn cpu_reshape_1x6_to_2x3() -> anyhow::Result<()> {
    let t = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]], &Device::Cpu)?;
    let out = reshape(&t, Shape::from(vec![2, 3]))?;
    assert_eq!(out.shape(), &Shape::from(vec![2, 3]));
    let v: Vec<Vec<f32>> = out.to_vec2()?;
    assert_eq!(v, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    Ok(())
}

/// Dtype guard: narrow_cuda on a non-F32/BF16 CPU tensor must either
/// succeed via candle's built-in CPU path (which handles all dtypes) or
/// return a clear error. The CPU path delegates to `t.narrow()` directly,
/// so this test pins that contract.
#[test]
fn cpu_narrow_dtype_agnostic() -> anyhow::Result<()> {
    let t = Tensor::zeros(&[4], DType::F64, &Device::Cpu)?;
    let out = narrow(&t, 0, 1, 2)?;
    assert_eq!(out.shape(), &Shape::from(vec![2]));
    let v: Vec<f64> = out.to_vec1()?;
    assert_eq!(v, vec![0.0, 0.0]);
    Ok(())
}

/// GPU smoke for narrow on BF16: requires CUDA. If CUDA is unavailable,
/// the test returns Ok (skip) and the honest gap is filed in AUDIT.md.
///
/// Roundtrip: create a 2x3 BF16 tensor on GPU, narrow to 1x3, copy to CPU,
/// assert it matches the expected row. This exercises the T6.5 BF16
/// specialization end-to-end.
#[test]
fn gpu_narrow_bf16_roundtrip() -> anyhow::Result<()> {
    // Skip if CUDA is not available in this build.
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "[skip] gpu_narrow_bf16_roundtrip: Device::new_cuda(0) failed: {}",
                e
            );
            return Ok(());
        }
    };

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data.clone(), Shape::from(vec![2, 3]), &dev)?.to_dtype(DType::BF16)?;
    let out = narrow(&t, 0, 1, 1)?;
    assert_eq!(out.dtype(), DType::BF16, "BF16 narrow must preserve dtype");
    assert_eq!(out.shape(), &Shape::from(vec![1, 3]));

    let back: Vec<Vec<f32>> = out.to_dtype(DType::F32)?.to_vec2()?;
    // The expected row (row 1, original) in f32, but BF16 has limited precision.
    // Compare with 0.5 absolute tolerance (BF16 has ~3 decimal digits of mantissa).
    let expected = vec![4.0_f32, 5.0, 6.0];
    assert_eq!(back.len(), 1, "shape should be 1x3, got {:?}", back);
    assert_eq!(back[0].len(), expected.len());
    for (got, want) in back[0].iter().zip(expected.iter()) {
        let diff = (got - want).abs();
        assert!(
            diff < 0.5,
            "BF16 narrow row mismatch: got={:?} want={:?} diff={}",
            back[0],
            expected,
            diff
        );
    }
    Ok(())
}
