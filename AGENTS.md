# Repository Guidelines

## Project Structure & Module Organization
- `CMakeLists.txt`: sets up the C++23 toolchain, fetches `tiny-cuda-nn` and the external `dataset` hostpack reader, and defines `instantngp_core` plus the `ngp-baseline-tcnn` executable.
- `include/instantngp/`: header-only APIs. `core/math.hpp` keeps lean vector/matrix helpers; `core/config.hpp` declares the JSON document interfaces.
- `src/core/config.cpp`: concrete JSON subset parser. Extend only for syntax required by Instant-NGP configs.
- `src/main.cpp`: minimal entry point that will orchestrate training. Keep non-core concerns out of this file.
- `cmake/` and `data/`: placeholders for future toolchains and sample assets. Generated outputs stay in `cmake-build-*` or `build/` and remain untracked.

## Build, Test, and Development Commands
- `cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`: configure and download dependencies. Rerun after editing CMake or dependency pins.
- `cmake --build build --config RelWithDebInfo -j`: compile the library and executable. Append `--target instantngp_core` for focused builds.
- `cmake --build build --target clean`: drop compiled outputs without touching the cache; run before changing compilers.
- Tests arrive later in the roadmap; toggle them with `-DINSTANTNGP_BUILD_TESTS=ON` once the suite lands.

## Coding Style & Naming Conventions
- Stick to modern C++23 idioms with 4-space indentation. Use `CamelCase` for types (`Vec3`, `Mat3x4`), `snake_case` for functions, and `SCREAMING_SNAKE_CASE` for constants.
- Keep headers self-contained; prefer `constexpr`/`noexcept` math helpers. Do not add logging or utilities unrelated to the Instant-NGP pipeline.
- When expanding the JSON parser, implement only the syntax you actively consume and cover it with targeted tests.

## Testing Guidelines
- Tests reside under `tests/` (created later) and compile to standalone executables linked against `instantngp_core`. Each test file should focus on a single subsystem (math, config parsing, future training steps).
- Name tests `<area>_test.cpp` and register them through CMake’s `add_test`. Use deterministic fixtures and hostpack samples to validate critical checkpoints.
- Aim for exhaustive coverage on math primitives, JSON parsing branches, and future core training routines. Keep the production code free from macros or flags used only by tests.

## Commit & Pull Request Guidelines
- Use imperative commit subjects ≤72 characters (e.g., `Implement JSON parser skeleton`) with concise bodies describing the change.
- Reference issues with `Fixes #ID` when relevant and enumerate newly introduced configuration fields or math helpers.
- Pull requests should state rationale, verification steps (build, lint, tests), and any follow-up work expected post-merge.
