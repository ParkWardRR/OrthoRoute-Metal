# OrthoRoute-Metal — Local CI with OrbStack

## Quick Start

```bash
# Run default CI (lint + tests, native)
./ci/run.sh

# Run everything (lint + tests + Metal build + packages)
./ci/run.sh all
```

## Available Commands

| Command | What it does |
|---------|-------------|
| `./ci/run.sh` | Default: lint + tests (native, fastest) |
| `./ci/run.sh test` | Run pytest only |
| `./ci/run.sh lint` | Run flake8 only |
| `./ci/run.sh typecheck` | Run mypy only |
| `./ci/run.sh metal` | Build Rust/Metal backend (macOS only) |
| `./ci/run.sh package` | Build KiCad plugin ZIPs |
| `./ci/run.sh docker` | Run lint+typecheck+tests in OrbStack container |
| `./ci/run.sh all` | Run everything (native + metal + packages) |

## Requirements

- **Native mode** (default): Python 3.10+, pytest, flake8
- **Container mode** (`docker`): [OrbStack](https://orbstack.dev) or Docker
- **Metal build** (`metal`): macOS + Rust 1.70+

## How It Works

### Native Mode (default)
Runs pytest, flake8, and mypy directly on your host machine. Fastest option — no container startup overhead.

### Container Mode (`docker`)
Builds a lightweight Python 3.12 Docker image and runs all checks inside it. Useful for:
- Reproducible environment
- Testing against a clean dependency set
- CI on machines without Python installed

Uses OrbStack if available (faster than Docker Desktop on macOS), falls back to standard Docker.

### Metal Build (`metal`)
Runs `cargo build --release` and `cargo test` in the `metal/` directory. Only works on macOS with Apple Silicon (requires Metal framework).

## Output

The CI runner reports pass/fail for each step and a summary:

```
═══════════════════════════════════════════════════════════════
  CI Summary
═══════════════════════════════════════════════════════════════

  Passed:  4
  Failed:  0
  Skipped: 0
  Time:    0m 12s

[OK] CI PASSED — all checks green ✓
```

Exit code is 0 on success, 1 on any failure.
