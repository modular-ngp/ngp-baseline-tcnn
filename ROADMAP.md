# Minimal Instant-NGP Roadmap

## Guiding Principles
- Core implementation stays lean: only functionality required for Instant-NGP training, no logging layers, GUI hooks, or ancillary tooling.
- All data ingestion relies on the external `dataset` hostpack API; no dataset preprocessing logic lives in this repository.
- Tests are implemented after the core pipeline is complete; they remain decoupled and cover every critical computation path.

## Phase 0 — Core Skeleton (Week 0-1)
- Establish minimal CMake targets for the core library and a single executable entry point.
- Implement essential math utilities in `src/core` (vectors, matrices, transforms) as header-only modules with constexpr-friendly operations.
- Provide a compact configuration loader: a hand-written JSON subset parser that supports the exact structures needed for training parameters and file paths.

## Phase 1 — Hostpack Integration (Week 1-2)
- Create `HostpackLoader` wrapping `dataset::open_hostpack`, exposing cameras, frames, AABB, and color space in normalized formats ready for GPU upload.
- Implement CPU buffers that prepare ray origins/directions and per-frame metadata using only the math primitives from Phase 0.
- Define deterministic device upload helpers that mirror hostpack stride/pixel metadata and expose raw device pointers to downstream stages.

## Phase 2 — Encoding & Network Core (Week 2-3)
- Configure `tiny-cuda-nn` multi-resolution hash encoding and minimal density/color MLP heads, parameterized strictly through the internal config loader.
- Implement occupancy grid maintenance (bitfield plus exponential moving average) coupled to the scene bounds from the hostpack.
- Add checkpoint persistence using `tiny-cuda-nn` snapshot APIs alongside compact metadata files written by the internal JSON writer.

## Phase 3 — Training Pipeline (Week 3-4)
- Build the full training step: ray sampling, stratified marching, importance resampling, forward evaluation, loss accumulation, and backward pass via `tiny-cuda-nn`.
- Support mixed precision and CUDA graph capture only where it directly improves core training throughput; omit optional features.
- Expose minimal CLI parameters (hostpack path, config path, training duration) parsed by the core configuration utilities.

## Phase 4 — Rendering & Output (Week 4-5)
- Implement evaluation rendering that reuses the training sampling path to produce RGB frames for validation cameras.
- Compute PSNR/SSIM using hand-written routines (no third-party dependencies) and write outputs to a fixed directory structure.
- Add optional model export limited to density grid serialization required for downstream use; skip non-essential formats.

## Phase 5 — Test Suite (Week 5-6)
- Introduce a separate test target covering math utilities, configuration parsing, hostpack integration, occupancy updates, training step invariants, and renderer output checks.
- Use deterministic seeds and lightweight fixtures derived from small hostpack samples to validate critical checkpoints and functions.
- Integrate tests with `ctest`; ensure the core executable remains untouched by test scaffolding.
