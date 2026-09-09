#!/bin/bash
#
# Places MLX's Metal library where a SwiftPM test run finds it, so `swift test` on InferKitMLX runs
# the MLX-dependent tests instead of skipping them.
#
# SwiftPM cannot compile Metal shaders, so a plain `swift build` produces no `default.metallib` and MLX
# aborts the process at its first evaluation. MLX's loader looks first for `mlx.metallib` beside the
# binary it is linked into, which under `swift test` is the package's test bundle executable. This
# script compiles mlx-swift's kernels with the Metal toolchain (the same nine sources Xcode's build
# system compiles into `mlx-swift_Cmlx.bundle`) and copies the result there.
#
#   Tools/mlx-metallib.sh              # debug configuration, what `swift test` uses
#   Tools/mlx-metallib.sh --release    # release configuration, for `swift test -c release`
#
# A `swift build` relink leaves the file in place; re-run after `swift package clean` or an mlx-swift
# bump. Requires the Metal toolchain (`xcrun metal`), which ships with Xcode.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$ROOT/InferKitMLX"
CONFIG=debug
[ "${1:-}" = "--release" ] && CONFIG=release

KERNELS="$PACKAGE/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
BUNDLE="$PACKAGE/.build/$CONFIG/InferKitMLXPackageTests.xctest"
if [ ! -d "$KERNELS" ] || [ ! -d "$BUNDLE/Contents/MacOS" ]; then
    echo "==> building the test bundle: swift build --build-tests -c $CONFIG"
    ( cd "$PACKAGE" && swift build --build-tests -c "$CONFIG" )
fi
[ -d "$KERNELS" ] || { echo "mlx-swift kernels not found at $KERNELS" >&2; exit 1; }
[ -d "$BUNDLE/Contents/MacOS" ] || { echo "test bundle not found at $BUNDLE" >&2; exit 1; }

WORK="$PACKAGE/.build/mlx-metallib"
LIB="$WORK/mlx.metallib"
mkdir -p "$WORK"
if [ -f "$LIB" ] && [ -z "$(find "$KERNELS" -newer "$LIB" -print -quit)" ]; then
    echo "==> $LIB is up to date"
else
    echo "==> compiling the MLX kernels from $KERNELS"
    AIRS=()
    while IFS= read -r -d '' source; do
        air="$WORK/$(basename "$source" .metal).air"
        # The flags are mlx's own (mlx/backend/metal/kernels/CMakeLists.txt) at mlx-swift's macOS floor.
        xcrun -sdk macosx metal -x metal -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
            -mmacosx-version-min=14.0 -c "$source" -I "$KERNELS" -o "$air"
        AIRS+=("$air")
    done < <(find "$KERNELS" -name '*.metal' -print0 | sort -z)
    xcrun -sdk macosx metallib "${AIRS[@]}" -o "$LIB"
fi
cp "$LIB" "$BUNDLE/Contents/MacOS/mlx.metallib"
echo "==> placed $BUNDLE/Contents/MacOS/mlx.metallib"
