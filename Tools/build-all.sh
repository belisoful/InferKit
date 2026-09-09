#!/bin/bash
#
# Builds all three packages in one command. They are separate SwiftPM packages by design — the core
# carries no dependencies and a macOS 11 / iOS 14 / tvOS 14 floor, while MLX needs Apple Silicon and
# macOS 14 / iOS 16 — so "building the repo" means building each in turn, not one combined target.
#
#   Tools/build-all.sh          # build
#   Tools/build-all.sh --test   # build and test (MLX gets its Metal library from mlx-metallib.sh first)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST=false
[ "${1:-}" = "--test" ] && TEST=true

for package in "." "InferKitMLX" "InferKitFoundationModels"; do
    name="$([ "$package" = "." ] && echo InferKit || echo "$package")"
    echo "==> $name"
    ( cd "$ROOT/$package" && swift build )
    if $TEST; then
        if [ "$package" = "InferKitMLX" ]; then
            # SwiftPM compiles no Metal shaders; without the library every MLX-dependent test skips.
            ( cd "$ROOT/$package" && swift build --build-tests && "$ROOT/Tools/mlx-metallib.sh" \
                && swift test 2>&1 | grep -E "Executed .* tests, with" | tail -1 )
        else
            ( cd "$ROOT/$package" && swift test 2>&1 | grep -E "Executed .* tests, with" | tail -1 )
        fi
    fi
done
echo "==> all three packages built"
