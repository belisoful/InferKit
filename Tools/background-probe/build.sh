#!/bin/bash
#
# Builds inferkit-background-probe in release and places MLX's Metal library beside it, into
# InferKitMLX/.build/background-probe/. The kernels compile with the flags Tools/mlx-metallib.sh uses.
#
#   Tools/background-probe/build.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE="$ROOT/InferKitMLX"
STAGE="$PACKAGE/.build/background-probe"
( cd "$PACKAGE" && swift build -c release --product inferkit-background-probe )
KERNELS="$PACKAGE/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
WORK="$PACKAGE/.build/mlx-metallib"
LIB="$WORK/mlx.metallib"
mkdir -p "$WORK" "$STAGE"
if [ ! -f "$LIB" ] || [ -n "$(find "$KERNELS" -newer "$LIB" -print -quit)" ]; then
    AIRS=()
    while IFS= read -r -d '' source; do
        air="$WORK/$(basename "$source" .metal).air"
        xcrun -sdk macosx metal -x metal -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
            -mmacosx-version-min=14.0 -c "$source" -I "$KERNELS" -o "$air"
        AIRS+=("$air")
    done < <(find "$KERNELS" -name '*.metal' -print0 | sort -z)
    xcrun -sdk macosx metallib "${AIRS[@]}" -o "$LIB"
fi
cp "$PACKAGE/.build/release/inferkit-background-probe" "$STAGE/"
cp "$LIB" "$STAGE/mlx.metallib"
codesign --force --sign - "$STAGE/inferkit-background-probe"
echo "==> staged $STAGE: inferkit-background-probe, mlx.metallib"
