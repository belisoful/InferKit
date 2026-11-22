#!/bin/bash
#
# Builds inferkit-background-probe in release from the package beside this script and places MLX's
# Metal library beside it, into Tools/background-probe/.build/stage/. The kernels compile with
# SwiftPM's recipe (Tools/mlx-metallib-recipe.sh), which Tools/mlx-metallib.sh also uses.
#
#   Tools/background-probe/build.sh
#
set -euo pipefail
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=Tools/mlx-metallib-recipe.sh
source "$PACKAGE/../mlx-metallib-recipe.sh"
STAGE="$PACKAGE/.build/stage"
( cd "$PACKAGE" && swift build -c release --product inferkit-background-probe )
CHECKOUT="$PACKAGE/.build/checkouts/mlx-swift"
KERNELS="$CHECKOUT/Source/Cmlx/mlx-generated/metal"
WORK="$PACKAGE/.build/mlx-metallib"
LIB="$WORK/mlx.metallib"
DEPLOYMENT="$(mlx_metal_deployment_target "$CHECKOUT")"
STAMP="$(mlx_metallib_stamp "$KERNELS" "$DEPLOYMENT")"
mkdir -p "$WORK" "$STAGE"
if [ ! -f "$LIB" ] || [ ! -f "$WORK/stamp" ] || [ "$(cat "$WORK/stamp")" != "$STAMP" ]; then
    mlx_build_metallib "$KERNELS" "$DEPLOYMENT" "$WORK" "$LIB.partial"
    mv "$LIB.partial" "$LIB"
    printf '%s\n' "$STAMP" > "$WORK/stamp"
fi
cp "$PACKAGE/.build/release/inferkit-background-probe" "$STAGE/"
cp "$LIB" "$STAGE/mlx.metallib"
codesign --force --sign - "$STAGE/inferkit-background-probe"
echo "==> staged $STAGE: inferkit-background-probe, mlx.metallib"
