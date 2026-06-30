# Apohara-DeKanus

**Spiritual successor to airllm. Sparse activation + layer-streaming + Z3-proven safety on consumer GPUs.**

> Decānus (latín tardío, del griego bizantino *dekanós*) = "jefe de diez".
> Los *decans* dividen el zodíaco en 36 porciones de 10°.
> Meta-meta del proyecto: **dividir el LLM frontier en pedazos operables en 8GB de VRAM.**

## Status

**Phase 0 — Genesis** (esta release).

- Workspace Rust skeleton con 8 crates
- AUDIT.md honest ledger (carry verbatim desde Apohara Context Forge)
- Sin benchmarks reales todavía (Phase 1+)

## Hardware target

- **GPU**: NVIDIA RTX 2060 SUPER 8GB (sm_75 / Turing)
- **CPU**: Ryzen 5 3600 (Zen 2, sin NUMA, sin AMX)
- **RAM**: 16GB DDR4 (medido real: 46Gi, headroom extra)
- **Disco**: NVMe Gen3 (3.5 GB/s peak, 2.5 GB/s sustained)

## Roadmap (Phases)

| Phase | Entregable | Target tok/s | Failure mode |
|---|---|---|---|
| 0 | repo + skeleton | n/a | n/a |
| 1 | Qwen3-8B smoke | ≥35 in-VRAM | <20 = bug architecture |
| 2 | Qwen3-30B-A3B | 30-40 | <20 = MoE wiring roto |
| 3 | Qwen3-Coder-Next (~80B/3B) | 15-20 | <10 = layer-stream roto |
| 4 | dekanus-selective (ASET) | sparsity metrics | abstracción rota |
| 5 | Ornith-1.0-397B-A17B | **12-17.5** | <8 = honest stop |
| 6 | v1.0 release | reproducible + AUDIT clean | sin release = seguimos |

## Workspace layout

```
crates/
├── airllm-core/          # Layer-streaming engine (glommio + cudarc + candle)
├── dekanus-cli/          # CLI binary (dekanus run / doctor / info)
├── dekanus-selective/    # SelectivePolicy trait (ASET port) — game changer
├── dekanus-quant-kv/     # KV cache quantization (FWHT 3-bit + Lloyd-Max)
├── dekanus-llmlingua2/   # Prompt compression (BERT-base INT8)
├── dekanus-rag/          # TurboVec-RAG codec_v8
├── dekanus-romy/         # Multi-agent safety (cache_salt + Z3 INV-15)
└── audit-honesty/        # AUDIT.md ledger primitives
```

## Honest-by-construction

Esta proyecto hereda la cultura honest-by-construction de [Apohara Context Forge](https://github.com/SuarezPM/Apohara_Context_Forge):

- **AUDIT.md** = ledger público de claims (204 KB desde CF)
- **check_honesty.sh** = guard de CI que rechaza PRs con claims sin evidencia
- **honesty.yml** = GitHub workflow que ejecuta el guard

Cada claim de velocidad en README o paper debe incluir:
- Commit SHA
- Hardware fingerprint (GPU + CPU + RAM + NVMe)
- Model SHA (config.json SHA-256)
- `active_params_per_token` (sparsity ratio, equivalente Nanite de "triángulos evaluados por pixel")
- Profiler dump (`ncu` o `tokio-console`)

## Build

```bash
cargo build --release --workspace
cargo test --workspace
./target/release/dekanus info
```

## License

Apache 2.0. Ver [LICENSE](LICENSE).

## Provenance

- Fork conceptual: [lyogavin/airllm](https://github.com/lyogavin/airllm) (Apache 2.0)
- Stack base: [Apohara Context Forge](https://github.com/SuarezPM/Apohara_Context_Forge) (Apache 2.0)
- Selective primitive inspiration: ASET (ACL Findings 2026), MoE-Spec (arXiv 2602.16052)
- Rust inference: candle (sm_75 survivor), cudarc 0.19.8, glommio 0.9