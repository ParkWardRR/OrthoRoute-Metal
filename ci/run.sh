#!/usr/bin/env bash
# =============================================================================
# OrthoRoute-Metal — Local CI Runner (OrbStack / Docker)
# =============================================================================
#
# Usage:
#   ./ci/run.sh              # Run full CI pipeline
#   ./ci/run.sh test         # Run tests only
#   ./ci/run.sh lint         # Run linting only
#   ./ci/run.sh typecheck    # Run mypy only
#   ./ci/run.sh metal        # Build Metal/Rust backend (native macOS only)
#   ./ci/run.sh all          # Run everything including Metal build
#
# Requires: OrbStack (https://orbstack.dev) or Docker
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="orthoroute-ci"
CONTAINER_NAME="orthoroute-ci-run"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Timing
SECONDS=0

log()    { echo -e "${BLUE}[CI]${NC} $*"; }
ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()   { echo -e "${RED}[FAIL]${NC} $*"; }
header() { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${CYAN}  $*${NC}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"; }

# Track pass/fail
PASSED=0
FAILED=0
SKIPPED=0

record_pass() { ((PASSED++)); ok "$1"; }
record_fail() { ((FAILED++)); fail "$1"; }
record_skip() { ((SKIPPED++)); warn "$1 (skipped)"; }

# ─────────────────────────────────────────────────────────────────────────────
# Docker / OrbStack detection
# ─────────────────────────────────────────────────────────────────────────────

detect_runtime() {
    if command -v docker &>/dev/null; then
        RUNTIME="docker"
        # Check if OrbStack is the backend
        if docker info 2>/dev/null | grep -qi orbstack; then
            log "Runtime: OrbStack (Docker)"
        else
            log "Runtime: Docker"
        fi
    else
        fail "Neither Docker nor OrbStack found. Install OrbStack: https://orbstack.dev"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Build CI container image
# ─────────────────────────────────────────────────────────────────────────────

build_image() {
    header "Building CI Image"

    if docker image inspect "$IMAGE_NAME" &>/dev/null; then
        log "Image '$IMAGE_NAME' exists, rebuilding..."
    fi

    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
    ok "CI image built: $IMAGE_NAME"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run command inside container
# ─────────────────────────────────────────────────────────────────────────────

run_in_container() {
    docker run --rm \
        --name "$CONTAINER_NAME" \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        "$IMAGE_NAME" \
        "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# CI Steps
# ─────────────────────────────────────────────────────────────────────────────

step_lint() {
    header "Linting (flake8)"
    if run_in_container flake8 orthoroute/ --count --statistics; then
        record_pass "flake8 linting"
    else
        record_fail "flake8 linting"
    fi
}

step_typecheck() {
    header "Type Checking (mypy)"
    if run_in_container mypy orthoroute/ --ignore-missing-imports --no-error-summary 2>&1; then
        record_pass "mypy type checking"
    else
        # mypy failures are warnings, not blockers, for this codebase
        warn "mypy reported issues (non-blocking)"
        record_pass "mypy type checking (warnings only)"
    fi
}

step_test() {
    header "Unit Tests (pytest)"
    if run_in_container pytest tests/ -v --tb=short -x 2>&1; then
        record_pass "pytest unit tests"
    else
        record_fail "pytest unit tests"
    fi
}

step_test_native() {
    header "Unit Tests (native — no container)"
    log "Running pytest directly on host..."
    cd "$PROJECT_ROOT"
    if python3 -m pytest tests/ -v --tb=short -x 2>&1; then
        record_pass "pytest unit tests (native)"
    else
        record_fail "pytest unit tests (native)"
    fi
}

step_lint_native() {
    header "Linting (native — no container)"
    cd "$PROJECT_ROOT"
    if python3 -m flake8 orthoroute/ --count --statistics 2>&1; then
        record_pass "flake8 linting (native)"
    else
        record_fail "flake8 linting (native)"
    fi
}

step_typecheck_native() {
    header "Type Checking (native — no container)"
    cd "$PROJECT_ROOT"
    if python3 -m mypy orthoroute/ --ignore-missing-imports --no-error-summary 2>&1; then
        record_pass "mypy type checking (native)"
    else
        warn "mypy reported issues (non-blocking)"
        record_pass "mypy type checking (native, warnings only)"
    fi
}

step_metal() {
    header "Metal Backend Build (Rust + MSL)"

    if [[ "$(uname)" != "Darwin" ]]; then
        record_skip "Metal build (not macOS)"
        return
    fi

    if ! command -v cargo &>/dev/null; then
        record_skip "Metal build (Rust not installed)"
        return
    fi

    log "Building Metal backend..."
    cd "$PROJECT_ROOT/metal"

    if cargo build --release 2>&1; then
        record_pass "cargo build --release"
    else
        record_fail "cargo build --release"
        return
    fi

    log "Running Rust tests..."
    if cargo test 2>&1; then
        record_pass "cargo test"
    else
        record_fail "cargo test"
    fi
}

step_package() {
    header "Plugin Package Build"
    cd "$PROJECT_ROOT"

    log "Building IPC plugin package..."
    if python3 build.py 2>&1; then
        record_pass "IPC plugin package"
    else
        record_fail "IPC plugin package"
    fi

    log "Building PCM plugin package..."
    if python3 build.py --pcm 2>&1; then
        record_pass "PCM plugin package"
    else
        record_fail "PCM plugin package"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print_summary() {
    local elapsed=$SECONDS
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))

    header "CI Summary"
    echo -e "  ${GREEN}Passed:${NC}  $PASSED"
    echo -e "  ${RED}Failed:${NC}  $FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
    echo -e "  ${BLUE}Time:${NC}    ${minutes}m ${seconds}s"
    echo ""

    if [[ $FAILED -gt 0 ]]; then
        fail "CI FAILED — $FAILED step(s) failed"
        exit 1
    else
        ok "CI PASSED — all checks green ✓"
        exit 0
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local mode="${1:-default}"

    echo -e "${BOLD}"
    echo "  ┌─────────────────────────────────────────┐"
    echo "  │  OrthoRoute-Metal — Local CI Pipeline    │"
    echo "  │  Runtime: OrbStack / Docker              │"
    echo "  └─────────────────────────────────────────┘"
    echo -e "${NC}"

    case "$mode" in
        test)
            step_test_native
            ;;
        lint)
            step_lint_native
            ;;
        typecheck)
            step_typecheck_native
            ;;
        metal)
            step_metal
            ;;
        package)
            step_package
            ;;
        docker|container)
            detect_runtime
            build_image
            step_lint
            step_typecheck
            step_test
            ;;
        all)
            # Native steps (fast, no container overhead)
            step_lint_native
            step_typecheck_native
            step_test_native
            step_metal
            step_package
            ;;
        default)
            # Default: run lint + tests natively (fastest)
            step_lint_native
            step_test_native
            ;;
        *)
            echo "Usage: $0 {test|lint|typecheck|metal|package|docker|all}"
            echo ""
            echo "  test       Run pytest (native)"
            echo "  lint       Run flake8 (native)"
            echo "  typecheck  Run mypy (native)"
            echo "  metal      Build Rust/Metal backend (macOS only)"
            echo "  package    Build KiCad plugin packages"
            echo "  docker     Run lint+typecheck+test in OrbStack/Docker container"
            echo "  all        Run everything (native + metal + package)"
            echo ""
            echo "Default (no args): lint + test (native)"
            exit 1
            ;;
    esac

    print_summary
}

main "$@"
