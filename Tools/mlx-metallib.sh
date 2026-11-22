#!/bin/bash
#
# Makes MLX's Metal library available to a SwiftPM test run of InferKitMLX, and guards the library
# that run loads.
#
#   Tools/mlx-metallib.sh                  # debug configuration, what `swift test` uses
#   Tools/mlx-metallib.sh --release        # release configuration, for `swift test -c release`
#   Tools/mlx-metallib.sh --verify         # rebuild with SwiftPM's recipe, compare with its library
#   Tools/mlx-metallib.sh --package <dir>  # a package other than this checkout's InferKitMLX
#
# MLX's loader (mlx/backend/metal/device.cpp) takes the first library it finds, in this order:
# `mlx.metallib` beside the binary MLX is linked into, `Resources/mlx.metallib` beside it, then
# `default.metallib` in an `mlx-swift_Cmlx.bundle`. Under `swift test` that binary is the test
# bundle's executable in `Contents/MacOS`.
#
# - Every test bundle carries SwiftPM's `mlx-swift_Cmlx.bundle/.../default.metallib` (Xcode 27) →
#   the script places nothing.
# - A bundle carries a colocated `mlx.metallib` that differs from its `default.metallib` → the
#   script names it and exits 1. MLX loads the colocated file first, so the run uses other kernels
#   and other numbers.
# - The bundle has no `default.metallib` (an older SwiftPM, one test bundle per package) → the
#   script compiles mlx-swift's kernels with SwiftPM's recipe (Tools/mlx-metallib-recipe.sh) and
#   copies the result to `Contents/MacOS/mlx.metallib`. It refuses when the mlx-swift checkout is
#   not the revision Package.resolved pins or has local changes, and when a SwiftPM-built library
#   exists in the build directory and the result differs from it.
#
# A `swift build` relink leaves a placed file in place; re-run after `swift package clean` or an
# mlx-swift bump. Requires the Metal toolchain (`xcrun metal`).
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=Tools/mlx-metallib-recipe.sh
source "$ROOT/Tools/mlx-metallib-recipe.sh"

PACKAGE="$ROOT/InferKitMLX"
CONFIG=debug
VERIFY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --release) CONFIG=release ;;
        --verify) VERIFY=1 ;;
        --package) shift; PACKAGE="$(cd "${1:?--package needs a directory}" && pwd)" ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

PRODUCTS="$PACKAGE/.build/$CONFIG"
CHECKOUT="$PACKAGE/.build/checkouts/mlx-swift"
KERNELS="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
# The compiled library embeds each source's path, so the kernels compile from their physical path,
# the one SwiftPM compiles from.
if [ -d "$KERNELS" ]; then
    KERNELS="$(cd -P "$KERNELS" && pwd -P)"
fi
BUNDLE="$PRODUCTS/InferKitMLXPackageTests.xctest"
SWIFTPM_LIBRARY="Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
WORK="$PACKAGE/.build/mlx-metallib"

sha() {
    shasum -a 256 "$1" | cut -d' ' -f1
}

test_bundles() {
    shopt -s nullglob
    local bundle
    for bundle in "$PRODUCTS"/*.xctest; do
        echo "$bundle"
    done
}

bundled_by_swiftpm() {
    local bundle found=0
    while IFS= read -r bundle; do
        [ -f "$bundle/$SWIFTPM_LIBRARY" ] || return 1
        found=1
    done < <(test_bundles)
    [ "$found" -eq 1 ]
}

# Prints a SwiftPM-built default.metallib for this configuration when the build directory has one.
reference_library() {
    local candidate
    for candidate in "$PRODUCTS/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" \
        "$PRODUCTS"/*.xctest/"$SWIFTPM_LIBRARY"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Reports every colocated mlx.metallib that would load ahead of its bundle's default.metallib.
# Returns 1 when one differs from it.
check_shadowing() {
    local bundle colocated status=0
    while IFS= read -r bundle; do
        [ -f "$bundle/$SWIFTPM_LIBRARY" ] || continue
        for colocated in "$bundle/Contents/MacOS/mlx.metallib" \
            "$bundle/Contents/MacOS/Resources/mlx.metallib"; do
            [ -f "$colocated" ] || continue
            if [ "$(sha "$colocated")" = "$(sha "$bundle/$SWIFTPM_LIBRARY")" ]; then
                echo "==> note: $colocated is byte-identical to the bundle's default.metallib" >&2
                continue
            fi
            echo "error: $colocated shadows $bundle/$SWIFTPM_LIBRARY" >&2
            echo "       MLX loads the colocated library first, so this bundle runs other" >&2
            echo "       kernels than SwiftPM built and its parity figures move. Move the" >&2
            echo "       file out of the bundle." >&2
            status=1
        done
    done < <(test_bundles)
    return "$status"
}

# Refuses a checkout whose kernels are not the ones Package.resolved pins.
check_checkout() {
    local pinned head
    pinned="$(python3 -I -c '
import json, sys
pins = json.load(open(sys.argv[1]))["pins"]
print(next(p["state"]["revision"] for p in pins if p["identity"] == "mlx-swift"))
' "$PACKAGE/Package.resolved")"
    head="$(git -C "$CHECKOUT" rev-parse HEAD)"
    if [ "$head" != "$pinned" ]; then
        echo "error: $CHECKOUT is at $head; Package.resolved pins $pinned" >&2
        return 1
    fi
    if [ -n "$(git -C "$CHECKOUT" status --porcelain)" ]; then
        echo "error: $CHECKOUT has local changes; its kernels are not revision $pinned" >&2
        return 1
    fi
    echo "$pinned"
}

# Builds $WORK/mlx.metallib with SwiftPM's recipe, reusing it while its stamp matches.
build_library() {
    local deployment stamp
    deployment="$(mlx_metal_deployment_target "$CHECKOUT")"
    stamp="$(mlx_metallib_stamp "$KERNELS" "$deployment")"
    mkdir -p "$WORK"
    if [ -f "$WORK/mlx.metallib" ] && [ -f "$WORK/stamp" ] \
        && [ "$(cat "$WORK/stamp")" = "$stamp" ]; then
        echo "==> $WORK/mlx.metallib is up to date"
        return 0
    fi
    echo "==> compiling the MLX kernels from $KERNELS with SwiftPM's recipe"
    mlx_build_metallib "$KERNELS" "$deployment" "$WORK" "$WORK/mlx.metallib.partial"
    mv "$WORK/mlx.metallib.partial" "$WORK/mlx.metallib"
    printf '%s\n' "$stamp" > "$WORK/stamp"
}

# Returns 1 when a SwiftPM-built library exists and the built one differs from it.
compare_with_reference() {
    local reference
    if ! reference="$(reference_library)"; then
        echo "==> no SwiftPM-built default.metallib in $PRODUCTS to compare with"
        return 0
    fi
    if [ "$(sha "$WORK/mlx.metallib")" != "$(sha "$reference")" ]; then
        echo "error: $WORK/mlx.metallib differs from SwiftPM's $reference" >&2
        echo "       $(sha "$WORK/mlx.metallib") against $(sha "$reference")" >&2
        return 1
    fi
    echo "==> $WORK/mlx.metallib is byte-identical to $reference"
}

check_shadowing

if [ "$VERIFY" -eq 1 ]; then
    [ -d "$KERNELS" ] || { echo "mlx-swift kernels not found at $KERNELS" >&2; exit 1; }
    if ! reference_library > /dev/null; then
        echo "no SwiftPM-built default.metallib in $PRODUCTS to verify against" >&2
        exit 1
    fi
    revision="$(check_checkout)"
    echo "==> mlx-swift $revision, $(mlx_metal_compiler_version)"
    build_library
    compare_with_reference
    exit 0
fi

if [ ! -d "$BUNDLE/Contents/MacOS" ] && bundled_by_swiftpm; then
    echo "==> every test bundle in $PRODUCTS already carries default.metallib; nothing to place"
    exit 0
fi
if [ ! -d "$KERNELS" ] || [ ! -d "$BUNDLE/Contents/MacOS" ]; then
    echo "==> building the test bundle: swift build --build-tests -c $CONFIG"
    ( cd "$PACKAGE" && swift build --build-tests -c "$CONFIG" )
    if [ -d "$KERNELS" ]; then
        KERNELS="$(cd -P "$KERNELS" && pwd -P)"
    fi
fi
if [ ! -d "$BUNDLE/Contents/MacOS" ] && bundled_by_swiftpm; then
    check_shadowing
    echo "==> every test bundle in $PRODUCTS already carries default.metallib; nothing to place"
    exit 0
fi
[ -d "$KERNELS" ] || { echo "mlx-swift kernels not found at $KERNELS" >&2; exit 1; }
[ -d "$BUNDLE/Contents/MacOS" ] || { echo "test bundle not found at $BUNDLE" >&2; exit 1; }

revision="$(check_checkout)"
echo "==> mlx-swift $revision, $(mlx_metal_compiler_version)"
build_library
compare_with_reference
cp "$WORK/mlx.metallib" "$BUNDLE/Contents/MacOS/mlx.metallib"
echo "==> placed $BUNDLE/Contents/MacOS/mlx.metallib"
