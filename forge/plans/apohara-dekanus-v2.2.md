# apohara-dekanus v2.2 — PHASED Implementation Plan

> **Source**: Synthesized from Perplexity Deep Research output
> (`/home/thelinconx/Descargas/apohara-dekanus-v2.2.md`, 2122 lines, 2026-07-01).
> Detailed code patterns, papers, repos, honest gaps, and AUDIT.md
> entry templates live in that source document. This file is the
> **executable phased version** — what to do, in what order, and what
> each phase delivers. **NO time estimates** (per Pablo's instruction).
>
> **Repo**: https://github.com/SuarezPM/apohara-dekanus
> **Head at planning**: `8e2959c` (chore gitignore OMO/OMC state dirs)
> **Latest AUDIT entry at planning**: D0029 (BF16 add kernel works in
> model path after start_offset fix)
> **Plan author**: Sisyphus, based on Perplexity research synthesis
> **Plan date**: 2026-07-01

---

## Mission (verbatim from Perplexity, mission-driven)

Run a Qwen3.5-397B-A17B-class MoE inference engine natively in Rust,
on consumer hardware, at rates approaching usefulness for interactive
sessions. **No "viability analysis" — this is an engineering challenge.**
"Impossible" doesn't exist in our vocabulary; there are engineering
problems not yet solved.

The user (Pablo) is obsessive with honesty by-construction. Every
performance claim ships with a commit hash, an AUDIT.md entry, and a
reproducible test. **No marketing. No 500× fake benchmarks. Raw data.**

---

## Why phases, not dates

The user explicitly requested a plan structured as **phases with
definition-of-done**, not calendar weeks/months. Phases here mean:

- **A phase is a verifiable milestone.** It ends with a green benchmark
  or a green test that proves the previous bottleneck is solved.
- **A phase commits to main.** Each phase generates ≥1 AUDIT.md entries
  (D0030, D0031, ...) before merging.
- **A phase has a gate.** Either the gate passes (phase done) or it
  doesn't (phase reworked, not bypassed).
- **A phase may have sub-phases.** When natural, but each sub-phase is
  itself a verifiable milestone.

This is the opposite of "spend 4 weeks on Tier 1". If a phase takes
3 days, it takes 3 days. If it takes 3 months, it takes 3 months. The
structure is by deliverable, not by time.

---

## Phase ordering rationale

Perplexity's Tier 7 cross-reference table established:

```
T1 (Kernels)         → T2 (CUDA Graphs)     → T5 (Spec decode)
   ↓                       ↓                       ↓
T3 (MoE Stream)  ←  T4 (Quantization)  ←  (independent of T1, T2)
   ↓
T6 (Z3 Safety)     [parallel, no dependency]
   ↓
T7 (Synthesis / production integration)
```

This translates to phases with **dependency edges**, not time:

- **T1 is a hard prerequisite** for T2, T3, T4, T5. Kernels must exist
  before they can be captured in graphs, before MoE can run, before
  quantization can be benchmarked.
- **T2 depends on T1**, enables T5 (speculative decode needs captured
  graph to be replayable for draft model).
- **T3 depends on T1 + T4** (quantized experts are smaller, faster to
  SSD-stream).
- **T4 is independent** of T1 (can start as soon as T1's FP16 GEMM
  kernel template is proven).
- **T5 depends on T2 + T4** (graph replay + Q4 base model).
- **T6 is independent** of all inference work (Z3 proofs are
  orthogonal).
- **T7 is synthesis** — no new kernel/algorithm work, integration +
  benchmarks.

---

## Phase 0 — Bootstrap

**Goal**: Make the Perplexity plan an official v2.2 document + open
the AUDIT.md ledger for v2.2 work.

**Scope**:
- Copy this synthesized plan to `forge/plans/apohara-dekanus-v2.2.md`
- Open AUDIT.md entry D0030 — "v2.2 plan initial: phased roadmap
  synthesis from Perplexity research"
- Commit D0030 + this plan with the standard format
- Decide on tracking convention: `D0030-D0049` reserved for v2.2 phases
- Update README.md to link the v2.2 plan

**DoD**:
- [ ] `forge/plans/apohara-dekanus-v2.2.md` committed to main
- [ ] AUDIT.md has D0030 entry with honest summary
- [ ] README.md links to v2.2 plan
- [ ] One git commit with all 3 changes + tests passing (`cargo test
  --workspace` exit 0)

**AUDIT entries**: D0030

**Gate**: `git log --oneline -1` shows the v2.2 plan commit; AUDIT.md
contains D0030; `cargo build --workspace` returns 0.

---

## Phase 1 — Foundation: sm_75-first CUDA Kernels (Perplexity T1)

**Goal**: Close the 6 missing BF16 kernels (`mul`, `silu`, `cat`,
`reshape`, `affine`, `stack`) by generalizing the D0029 `start_offset`
pattern. Each kernel ships with: TDD test, model-path test, and an
AUDIT entry.

**Scope** (from Perplexity Tier 1):
- FP16 GEMM kernel for sm_75 (reference: `andylolu2/simpleGEMM`,
  CUTLASS `cutlass_simt_hgemm`, arXiv:2203.03341 Ootomo & Yokota)
- 6 element-wise BF16/FP16 kernels with `__nv_bfloat16` intrinsics
  and start_offset handling
- `affine` kernel (x * scale + bias) as hot-path op
- cudarc 0.19 `CudaSlice::slice()` returns `CudaView<'_, T>` — apply
  the D0029 pattern to all 6
- PTX intrinsics specific to sm_75:
  `mma.sync.aligned.m16n8k16.f16.f16.f16.f16`

**Sub-phases** (each is its own milestone):
- 1a: `mul_bf16` (simplest, gates the pattern)
- 1b: `add_bf16` (already done in D0029, ship as benchmark)
- 1c: `silu_bf16` (x * sigmoid(x))
- 1d: `cat_bf16` (concatenation)
- 1e: `reshape_bf16` (strided copy)
- 1f: `affine_bf16` (x * scale + bias)
- 1g: `stack_bf16` (concat along new dim)
- 1h: Model-path test that uses all 7 BF16 kernels in one forward
  pass on Qwen3-8B with --gpu

**DoD**:
- [ ] Each of 6 missing kernels has a TDD test in
  `crates/airllm-kernels/tests/` that passes in isolation
- [ ] `gpu_*_bf16_*_model_path` test added that wires all 6 through
  the dispatch shim on a real Qwen3-8B forward pass
- [ ] `dekanus-cli generate --model models/Qwen3-8B --gpu --n 5` runs
  clean and produces coherent tokens
- [ ] tok/s measurement recorded in AUDIT entry for each kernel
- [ ] 1 AUDIT entry per kernel: D0031-D0036 (6 entries) + D0037 for
  the model-path integration test
- [ ] All commits use the standard format: imperative summary +
  bullets + tests evidence

**AUDIT entries**: D0031, D0032, D0033, D0034, D0035, D0036, D0037

**Gate**: `cargo test -p airllm-kernels --release` is green; model-path
test passes; tok/s ≥ current 0.50 baseline (any improvement is a
bonus, the gate is "no regression").

**Honest expectation** (from Perplexity Tier 1):
- 20-30% latency reduction from eliminating ~18 redundant F32 cast
  launches per token
- Honest ceiling: ~0.65 tok/s on 8B without CUDA Graphs

**Dependencies**: none (Phase 0 only).

**Cross-references**:
- Required by: Phase 2 (CUDA Graphs need stable kernels), Phase 3
  (Quantization needs FP16 GEMM template), Phase 5 (MoE SSD needs
  efficient memory access).

---

## Phase 2 — CUDA Graphs in Rust (Perplexity T2)

**Goal**: Capture the autoregressive decode loop into a
`cudarc 0.19` `CudaGraph` with bucket-based shape variants. Real
1.1-1.4× decode throughput gain + 5-10× jitter reduction.

**Scope** (from Perplexity Tier 2):
- cudarc 0.19 API verified: `CudaStream::begin_capture(RELAXED)`,
  `end_capture(AUTO_FREE_ON_LAUNCH)`, `CudaGraph::launch()`,
  `CudaGraph::upload()`
- Pre-allocate all device tensors before capture (D0029-stable
  address pattern)
- Bucket-based capture: graphs for seq_len 1, 2, 4, 8, 16, 32, 64,
  128
- Replay path: pre-launch D2D async copies (`cuMemcpyDtoDAsync_v2`),
  `cuMemsetD32Async` for device scalars (NEVER
  `cuMemcpyHtoDAsync` with stack var — stack lifetime race)
- Warmup phase before capture (FlashAttention allocates internal
  buffers during first launch)
- Memory pool isolation via `cuDeviceSetMemPool` (prevents process
  frees from touching graph memory)

**Reference**: `ml-rust/boostr` `src/inference/decode_graph.rs` L1-L550
(the gold standard). Port the `DeviceScalars` pattern + the
`pre_replay_and_launch` skeleton.

**DoD**:
- [ ] New crate `dekanus-graph` (or extend `airllm-core`) with
  `cuda_graphs.rs` wrapper
- [ ] `cuda_graphs_parity_test.rs`: bit-exact identical output to
  eager mode for the same input
- [ ] `dekanus-cli generate --model models/Qwen3-8B --gpu --n 50`:
  tok/s ≥ 1.0× of baseline (Phase 1), latency p99 < 1.5× of p50
  (jitter reduced)
- [ ] 1-2 AUDIT entries documenting measured speedup and p99/p50
  ratio

**AUDIT entries**: D0038 (capture), D0039 (bucket-based replay + warmup)

**Gate**: `cargo test -p dekanus-graph --release` green; `dekanus
generate --gpu --n 50` tok/s ≥ 1.1× Phase 1; latency p99 < 1.5× p50.

**Honest expectation**:
- 1.1-1.4× decode throughput (NOT 10× — CUDA Graphs eliminate launch
  overhead, not memory bandwidth)
- 5-10× jitter reduction (the real value)

**Dependencies**: Phase 1 (kernels must be stable before capture).

**Cross-references**:
- Required by: Phase 4 (Speculative decoding needs graph replay for
  draft model).

---

## Phase 3 — Quantization + Mixed Precision (Perplexity T4)

**Goal**: Memory budget tier. Q4_K_M reduces Qwen3.5-397B from ~794
GB to ~219 GB, which fits comfortably on NVMe but requires a
dequant-to-FP16 fused kernel for sm_75 (no BF16 Tensor Cores).

**Scope** (from Perplexity Tier 4):
- Quantization decision matrix: Q4_K_M for FFN, FP16 for shared
  experts + attention output, INT8 (SmoothQuant) for QKV
- Dequant-to-FP16 fused kernel for sm_75 (uses FP16 Tensor Cores,
  NOT BF16)
- AWQ scale selection (activation-aware): arXiv:2306.00978
- SmoothQuant (activation outliers): arXiv:2211.10438
- AWQ calibration data selection
- Qwen3-8B Q4 model-path test (coherent output, 4-6 tok/s target)

**Quantization decision matrix** (from Perplexity Tier 4):
- Q4_K_M (FFN) + FP16 (shared expert + attn output) + INT8 (QKV
  projections via SmoothQuant)

**Sub-phases**:
- 3a: AWQ scale selection pipeline (offline calibration)
- 3b: Q4_K_M dequant-to-FP16 kernel (sm_75-tuned)
- 3c: SmoothQuant INT8 dequant kernel for QKV
- 3d: Model-path test on Qwen3-8B Q4 with --gpu
- 3e: tok/s benchmark

**DoD**:
- [ ] New crate `dekanus-quant-kv` (or rename) with real Q4_K_M
  dequant implementation
- [ ] `q4_k_m_dequant_test.rs`: numerical correctness vs F32
  reference within 1% PPL
- [ ] `dekanus-cli generate --model models/Qwen3-8B --gpu --quant q4
  --n 20`: coherent output, 2-3× tok/s vs Phase 2 baseline
- [ ] 1-2 AUDIT entries documenting measured tok/s and numerical
  parity

**AUDIT entries**: D0040 (Q4 dequant), D0041 (AWQ calibration), D0042
(Q4 model-path)

**Gate**: `dekanus-cli generate --gpu --quant q4 --n 20` produces
coherent output; tok/s ≥ 2× Phase 2; numerical parity within 1% of
F32 reference.

**Honest expectation**:
- Q4_K_M 8B: 4-6 tok/s on RTX 2060 SUPER
- Q4_K_M 35B: 1-2 tok/s with SSD streaming
- Q4_K_M 397B: 0.5-1.5 tok/s batched offline

**Dependencies**: Phase 1 (FP16 dequant kernel template).

**Cross-references**:
- Required by: Phase 4 (Speculative decode needs Q4 base rate), Phase
  5 (MoE SSD with quantized experts).
- Independent of: Phase 2 (CUDA Graphs).

---

## Phase 4 — Speculative Decoding + MTP (Perplexity T5)

**Goal**: 1.5-2× additional throughput once base rate reaches ~1
tok/s. Use Qwen3.5 native MTP heads (no auxiliary model needed).

**Scope** (from Perplexity Tier 5):
- Qwen3.5-397B MTP head loading (`nextn.*` tensors from GGUF)
- EAGLE-3 compatibility decision (auxiliary layers must be
  GatedAttention, not GatedDeltaNet — see SGLang PR #25408)
- Speculative decode loop: draft N tokens, verify in 1 forward
- Acceptance rate logging
- llama.cpp MTP integration pattern (--spec-type draft-mtp
  --spec-draft-n-max 6) achieved 1.8-2.2× speedup
- EAGLE-3 (arXiv:2503.01840) for 6.5× ideal conditions (alternative
  path, more complex)

**Sub-phases**:
- 4a: MTP head loader (load `nextn.*` from GGUF)
- 4b: Speculative decode loop (draft + verify)
- 4c: Acceptance rate logging + AUDIT measurement
- 4d: EAGLE-3 alternative (only if MTP insufficient)

**DoD**:
- [ ] New crate `dekanus-spec` (or extend `airllm-core`) with MTP
  support
- [ ] `mtp_acceptance_test.rs`: measure acceptance rate on first 100
  tokens of Qwen3-8B with --gpu
- [ ] `dekanus-cli generate --model models/Qwen3-8B --gpu --spec mtp
  --n 50`: tok/s ≥ 1.5× Phase 3 baseline, acceptance rate logged
- [ ] 1-2 AUDIT entries

**AUDIT entries**: D0043 (MTP head loading), D0044 (MTP acceptance
rate)

**Gate**: tok/s ≥ 1.5× Phase 3 with MTP; acceptance rate > 0.6 (else
fall back to EAGLE-3 in 4d).

**Honest expectation**:
- Native MTP: 1.5-2× speedup (llama.cpp reference: 1.8-2.2×)
- EAGLE-3: up to 6.5× in ideal conditions (complex to implement)
- Qwen3.5-397B with 60 layers (3 GatedDeltaNet + 1 GatedAttention
  pattern): MTP auxiliary only on GatedAttention layers

**Dependencies**: Phase 2 (graph replay) + Phase 3 (Q4 base rate).

**Cross-references**:
- Independent of: Phase 5 (MoE SSD), Phase 6 (Z3).

---

## Phase 5 — MoE SSD Streaming (Perplexity T3)

**Goal**: Existential tier for 397B. Without SSD streaming, 397B is
physically impossible on 8GB VRAM (34 GB/forward at FP16).

**Scope** (from Perplexity Tier 3):
- 3-stage prefetch queue: disk → RAM pinned → GPU
- Router-first top-k SSD stream: only load selected experts
- PagedAttention for stable-address KV cache (port from boostr)
- MoE-Infinity (arXiv:2401.14361) expert activation tracing
- PreScope (arXiv:2509.23638): LLaPor + PreSched + AsyncIO
- DuoServe-MoE (arXiv:2509.07379): prefill/decode phase separation
- LayerKV (arXiv:2410.00428): layer-wise KV cache eviction
- glommio re-enable for full NVMe throughput

**Sub-phases**:
- 5a: 3-stage prefetch queue (disk → RAM → GPU)
- 5b: Router-first top-k expert SSD stream
- 5c: PagedAttention port from boostr
- 5d: 35B model-path test (Qwen3.5-35B-A3B SSD streaming)
- 5e: 397B model-path test (batched offline)

**DoD**:
- [ ] New crate `dekanus-moe-stream` with full SSD streaming engine
- [ ] `moe_stream_test.rs`: 35B Q4 model-path test, tok/s ≥ 2 with SSD
- [ ] `dekanus-cli generate --model models/Qwen3.5-35B-A3B --gpu
  --quant q4 --n 10`: coherent output, tok/s ≥ 2 (vs 0 if no SSD
  streaming)
- [ ] 1-2 AUDIT entries

**AUDIT entries**: D0046 (MoeStreamEngine), D0047 (Double-buffer
prefetch), D0048 (35B benchmark)

**Gate**: 35B SSD streaming tok/s ≥ 2; coherent output verified.

**Honest expectation**:
- 35B Q4 with SSD: 2-3 tok/s (3.5 GB/s NVMe bandwidth bound)
- 397B Q4 with SSD: 0.5-1.5 tok/s (highly workload-dependent)
- Batched offline: 5-10× improvement over single-stream

**Dependencies**: Phase 1 (efficient memory access) + Phase 3
(quantized experts).

**Cross-references**:
- Independent of: Phase 4 (Speculative), Phase 6 (Z3).

---

## Phase 6 — Multi-Agent Safety (Z3 Proofs) (Perplexity T6)

**Goal**: Replace the `Inv15::prove() -> true` hardcoded stub with a
real Z3-based proof for INV-10 through INV-15. **Independent of
inference work — can run in parallel.**

**Scope** (from Perplexity Tier 6):
- Z3 Rust crate API: `Solver`, `Config`, `Sort`, `Int`, `Bool`,
  `forall_const`, `Solver::check_sat`
- 6 invariants: INV-10 (token budget), INV-11 (KV cache bound),
  INV-12 (expert routing safety), INV-13 (layer consistency),
  INV-14 (cache_salt isolation), INV-15 (memory ordering)
- Performance: incremental solving, push/pop, scoped contexts
- Negative tests: violations must be detected
- Reference: `/home/thelinconx/Mi_Universo/Mundo_Apohara/Apohara_Context_Forge/paper/inv15_paper.tex`
  (draft LaTeX, may need to be uploaded as gist for verification)

**Sub-phases**:
- 6a: Z3 Rust crate integration test (simple invariant as smoke)
- 6b: INV-10 (token budget) proof
- 6c: INV-11 (KV cache bound) proof
- 6d: INV-12 (expert routing safety) proof
- 6e: INV-13 (layer consistency) proof
- 6f: INV-14 (cache_salt isolation) proof
- 6g: INV-15 (memory ordering) proof
- 6h: Full invariant suite negative tests

**DoD**:
- [ ] Replace `Inv15::prove() -> true` stub in
  `crates/dekanus-romy/src/invariants.rs` with real Z3
- [ ] 6 invariants proven UNSAT (no counterexample) at startup
- [ ] Negative test suite: each invariant must detect injected
  violations
- [ ] 1 AUDIT entry per invariant + 1 entry for the full suite

**AUDIT entries**: D0049 (Inv15 Z3 proof), D0050 (full invariant suite)

**Gate**: All 6 invariants UNSAT; negative test suite green;
`Inv15::prove()` runtime < 100ms (cached, not in hot path).

**Honest expectation**:
- Z3 cold solve: 10-100ms per invariant
- Cached: <1ms (deterministic, same model)
- Hot path impact: 0 (Z3 runs at startup, not per-token)

**Dependencies**: none (orthogonal to inference work).

**Cross-references**:
- Independent of: All other phases.

---

## Phase 7 — Production Integration (Perplexity T7 synthesis)

**Goal**: Integrate all phases, run full benchmarks, ship v2.0.

**Scope**:
- Integrate Phase 1-6 outputs into a single `dekanus-cli` build
- Run full benchmark suite: Qwen3-8B, Qwen3.5-35B-A3B, Qwen3.5-397B-A17B
- Generate final performance report
- v2.0 release: reproducible builds, AUDIT clean, public release
- Update README with v2.0 capabilities

**Sub-phases**:
- 7a: Integration testing (all phases together)
- 7b: Full benchmark suite (8B, 35B, 397B)
- 7c: v2.0 release tag + reproducible build
- 7d: Public release + paper writeup

**DoD**:
- [ ] `dekanus-cli` binary runs all 3 model classes
- [ ] Full benchmark report in AUDIT.md
- [ ] v2.0 git tag signed
- [ ] Public release notes published

**AUDIT entries**: D0051+ (post-v2.0 entries as needed)

**Gate**: v2.0 shipped, reproducible from clean checkout.

**Honest expectation**:
- 8B: ≥ 5 tok/s (Phase 1-3-4 combined)
- 35B: ≥ 2 tok/s (Phase 5)
- 397B: 0.5-1.5 tok/s batched (Phase 5 ceiling)

**Dependencies**: All previous phases.

**Cross-references**: Terminal phase.

---

## Cross-reference index

| Phase | Depends on | Required by |
|---|---|---|
| 0 (Bootstrap) | none | 1-7 |
| 1 (Kernels) | 0 | 2, 3, 5 |
| 2 (CUDA Graphs) | 1 | 4 |
| 3 (Quantization) | 1 | 4, 5 |
| 4 (Spec decode) | 2, 3 | 7 |
| 5 (MoE SSD) | 1, 3 | 7 |
| 6 (Z3 Safety) | 0 (independent) | 7 |
| 7 (Production) | 1-6 | — (terminal) |

---

## Honest performance trajectory (bandwidth math, no claims)

```
Phase 0 (D0029 baseline):  0.50 tok/s on 8B, 0 tok/s on 35B/397B
Phase 1 (T1 kernels):     0.50-0.65 tok/s on 8B (no regression gate)
Phase 2 (+ CUDA Graphs):  0.55-0.90 tok/s on 8B (1.1-1.4× phase 1)
Phase 3 (+ Q4_K_M):       2-6 tok/s on 8B Q4 (Q4 reduces memory 4×)
Phase 4 (+ MTP):          3-12 tok/s on 8B Q4 MTP (1.5-2× phase 3)
Phase 5 (+ SSD stream):   2-3 tok/s on 35B Q4 (NVMe bandwidth bound)
Phase 6 (+ Z3):           same as phase 5 (orthogonal, no perf delta)
Phase 7 (integrated):     397B batched: 0.5-1.5 tok/s; 8B: 5-10 tok/s
```

**The 500x myth is not in this trajectory.** Every phase uses
bandwidth math, not marketing claims. If a phase over-delivers, it's
honest data; if it under-delivers, that's an AUDIT entry to revise
the plan.

---

## AUDIT.md entry reservation

| Entry | Subject |
|---|---|
| D0030 | v2.2 plan initial: phased roadmap synthesis from Perplexity research |
| D0031 | mul_bf16 kernel + test |
| D0032 | silu_bf16 kernel + test |
| D0033 | cat_bf16 kernel + test |
| D0034 | reshape_bf16 kernel + test |
| D0035 | affine_bf16 kernel + test |
| D0036 | stack_bf16 kernel + test |
| D0037 | Model-path integration test (all 7 BF16 kernels) |
| D0038 | CUDA Graph capture |
| D0039 | Bucket-based CUDA Graph replay + warmup |
| D0040 | Q4_K_M dequant-to-FP16 kernel |
| D0041 | AWQ scale calibration |
| D0042 | Q4 model-path test (Qwen3-8B Q4) |
| D0043 | Native MTP head loading |
| D0044 | MTP acceptance rate logging |
| D0045 | 8B benchmark (≥5 tok/s gate) |
| D0046 | MoeStreamEngine first run (35B) |
| D0047 | Double-buffer prefetch |
| D0048 | 35B benchmark (≥2 tok/s gate) |
| D0049 | Inv15 real Z3 proof |
| D0050 | Full invariant suite (INV-10 to INV-15) |
| D0051+ | Post-v2.0 production entries |

---

## Reference source

The complete 2122-line Perplexity Deep Research output lives at
`/home/thelinconx/Descargas/apohara-dekanus-v2.2.md` (also accessible
to the repo via `forge/plans/apohara-dekanus-v2.2-PERPLEXITY-SOURCE.md`
once we add it). It contains:

- **Tier 1-6 detail**: ~200-300 lines each of code patterns, paper
  references, performance estimates, honest gaps
- **Appendix A**: 5 discoveries (paiml/aprender, guoqingbao/attention.rs,
  PreScope, DuoServe-MoE, cool-japan/oxicuda)
- **Appendix B**: 8 honest gaps (arxiv ID corrections, unverified
  claims, etc.)

This phased file is the **executable summary**. When a phase starts,
the implementer reads the corresponding Tier in the Perplexity source
for full context.

---

## Decision matrix (locked at v2.2 publication)

| Decision | Choice | Rationale |
|---|---|---|
| Tensor backend | candle 0.11 + cudarc 0.19 | Working, all shims proven, v2.0+ fork OK |
| Quantization | Q4_K_M (FFN) + FP16 (shared expert + attn output) + INT8 (QKV) | Memory budget vs quality |
| MoE strategy | Router-first top-k SSD stream | Activates ~1-3 GB experts per forward |
| Speculative | Native MTP (Qwen3.5 ships with it) | Lower complexity than EAGLE-3, 1.5-2× speedup |
| Safety proof | Z3 (rust-z3 crate) | Production-grade SMT solver, <100ms cold per invariant |
| Hardware target | RTX 2060 SUPER sm_75 | $400 budget, FP16 Tensor Cores only |

**No alternatives to revisit** during v2.2 implementation. Changes
require a D-entry and rationale.

---

*No fabrication. No 500× speedups. Bandwidth math for every claim.*
*Every phase has a gate. Every gate has an AUDIT entry.*
