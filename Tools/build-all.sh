#!/bin/bash
#
# Builds all three packages in one command. They are separate SwiftPM packages by design — the core
# carries no dependencies and a macOS 11 / iOS 14 / tvOS 14 floor, while MLX needs Apple Silicon and
# macOS 14 / iOS 16 — so "building the repo" means building each in turn, not one combined target.
#
#   Tools/build-all.sh          # build
#   Tools/build-all.sh --test   # build and test (MLX tests go through xcodebuild for the metallib)
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
            # MLX evaluation needs the Metal library only Xcode's build system bundles.
            ( cd "$ROOT/$package" && xcodebuild test -scheme InferKitMLX \
                -destination 'platform=macOS' -skipPackagePluginValidation 2>&1 \
                | grep -E "Executed .* tests, with|\*\* TEST" | tail -2 )
        else
            ( cd "$ROOT/$package" && swift test 2>&1 | grep -E "Executed .* tests, with" | tail -1 )
        fi
    fi
done
echo "==> all three packages built"
