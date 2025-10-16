# Repository Guidelines

## Project Structure & Module Organization
- `CMakeLists.txt`: defines the C++23 target `ngp-baseline-tcnn` and pulls `tiny-cuda-nn` via `FetchContent`. Update GPU or dependency settings here.
- `main.cpp`: placeholder entry point; extend with training loops, kernel setup, and CLI wiring.
- `data/nerf-synthetic/`: staging area for NeRF sample assets. Keep raw datasets here and commit only lightweight metadata.
- `cmake/`: reserved for custom modules or toolchain files; add CUDA checks or presets as they appear.
- Build outputs belong in `cmake-build-*` or a local `build/` directory—never commit generated binaries.

## Build, Test, and Development Commands
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`: configure the project and download `tiny-cuda-nn`. Rerun after changing dependencies or CUDA flags.
- `cmake --build build --config RelWithDebInfo -j`: compile the executable with CUDA support; append `--target <name>` if multiple binaries are added.
- `ctest --test-dir build --output-on-failure`: execute registered host or GPU tests. Add new suites with `add_test`.
- `cmake --build build --target clean`: clear compiled artifacts while preserving the CMake cache.

## Coding Style & Naming Conventions
- Stick to modern C++23 idioms, 4-space indentation, `snake_case` for functions, `CamelCase` for types, and `SCREAMING_SNAKE_CASE` for constants.
- Prefer RAII wrappers and `std::span`/`std::expected` when managing CUDA buffers or results.
- Run `clang-format -i main.cpp` (LLVM style) before committing; introduce a `.clang-format` once shared style choices solidify.

## Testing Guidelines
- Collect GPU-focused tests under `tests/` (create as needed) with fixtures in `tests/data`.
- Name test files `<component>_test.cpp`, register executables with `add_executable`, and expose them via `add_test`.
- Target smoke coverage for kernel launches plus host-side regression checks; document required devices or compute capability in the test header.

## Commit & Pull Request Guidelines
- Write imperative, ≤72-character subjects (e.g., `Add NeRF dataset loader`) and include context and GPU requirements in the body.
- Reference issues with `Fixes #ID` and attach before/after metrics or screenshots for training quality.
- PRs should provide a purpose summary, recent build/test evidence (`ctest` output), and call out any new data assets or CUDA flags.

## CUDA & Dependency Notes
- Requires CUDA 11.8+ and a compute capability ≥7.0 GPU; verify availability with `nvidia-smi`.
- `FetchContent` clones `tiny-cuda-nn` during configure; run `cmake --build build --target tiny-cuda-nn` to refresh the dependency when updating.
