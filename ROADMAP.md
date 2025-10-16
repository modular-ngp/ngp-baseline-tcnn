# Minimal Instant-NGP Roadmap

## Guiding Principles
- Core implementation stays lean: only functionality required for Instant-NGP training, no logging layers, GUI hooks, or ancillary tooling.
- All data ingestion relies on the external `dataset` hostpack API; no dataset preprocessing logic lives in this repository.
- Tests are implemented after the core pipeline is complete; they remain decoupled and cover every critical computation path.

## Phase 0 — Core Skeleton (Week 0-1, Complete)
- CMake now defines the `instantngp_core` static library and a minimalist `ngp-baseline-tcnn` executable, both compiled with strict C++23 flags.
- Header-only math utilities in `include/instantngp/core/math.hpp` supply constexpr-friendly vectors, transforms, and composition helpers.
- A bespoke JSON subset parser (`include/instantngp/core/config.hpp`, `src/core/config.cpp`) parses configuration documents without external dependencies.

## Phase 1 — Hostpack Integration (Week 1-2, Complete)
- Implemented `instantngp::io::HostpackLoader` (`include/instantngp/io/hostpack_loader.hpp`, `src/io/hostpack_loader.cpp`) that wraps the `dataset` API, normalizes camera intrinsics/extrinsics, exposes scene bounds, and surfaces frame/image metadata.
- Added `RayBatch` generation using the core math primitives to precompute deterministic ray origins and directions per frame.
- Introduced `core::DeviceBuffer` (`include/instantngp/core/device_buffer.hpp`) to manage CUDA memory and provide stride-aware image uploads that mirror hostpack layout.

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
