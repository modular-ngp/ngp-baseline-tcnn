# Repository Guidelines

## Project Structure & Module Organization
- `CMakeLists.txt`: configures the C++23 toolchain, pulls `tiny-cuda-nn` plus `dataset`, locates CUDA 13, and builds `instantngp_core` with the `ngp-baseline-tcnn` executable.
- `include/instantngp/core/`: header-only primitives (`math.hpp`, `config.hpp`, `device_buffer.hpp`) covering math, JSON parsing, and CUDA memory helpers.
- `include/instantngp/io/hostpack_loader.hpp` + `src/io/hostpack_loader.cpp`: wrap the hostpack reader, normalize camera metadata, emit ray batches, and handle stride-aware image uploads.
- `src/core/config.cpp`: bespoke JSON parser; extend only when new configuration fields are required.
- `src/main.cpp`: stub entry point for configuration loading and optional hostpack probing.
- `cmake/` and `data/`: staging areas for future helpers or sample assets; keep build outputs outside version control.

## Build, Test, and Development Commands
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`: configure and download dependencies. Rerun after editing CMake or dependency pins.
- `cmake --build build --config RelWithDebInfo -j`: compile the library and executable. Append `--target instantngp_core` for focused builds.
- `cmake --build build --target clean`: drop compiled outputs without touching the cache; run before changing compilers.
- Tests arrive later in the roadmap; toggle them with `-DINSTANTNGP_BUILD_TESTS=ON` once the suite lands.

## Coding Style & Naming Conventions
- Favor modern C++23 with 4-space indentation. Use `CamelCase` for types (`Vec3`, `Mat3x4`, `HostpackLoader`), `snake_case` for functions, and `SCREAMING_SNAKE_CASE` for constants.
- Keep headers self-contained and minimal; prefer `constexpr`/`noexcept` helpers and avoid logging or utilities unrelated to the Instant-NGP pipeline.
- When extending the JSON parser or hostpack loader, add only the syntax or metadata required by the pipeline and guard it with focused tests.

## Testing Guidelines
- Tests reside under `tests/` (added later) and build as standalone executables linked against `instantngp_core`. Keep each file focused on one subsystem (math, config parsing, hostpack loader, future training loops).
- Name tests `<area>_test.cpp` and register them via `add_test`. Use deterministic fixtures and compact hostpacks to validate AABB normalization, ray batches, and CUDA uploads.
- Target comprehensive coverage on math primitives, JSON parsing branches, hostpack translation, and future core routines while keeping production code free from test-only switches.

## Commit & Pull Request Guidelines
- Use imperative commit subjects ≤72 characters (e.g., `Implement JSON parser skeleton`) with concise bodies describing the change.
- Reference issues with `Fixes #ID` when relevant and enumerate newly introduced configuration fields or math helpers.
- Pull requests should state rationale, verification steps (build, lint, tests), and any follow-up work expected post-merge.
