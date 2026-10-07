#!/bin/bash
#
# Builds DocC documentation for InferKit.
#
# The core is pure Objective-C. DocC needs a symbol graph, and neither Xcode's `docbuild` nor the
# swift-docc-plugin extracts one for a pure-Objective-C SwiftPM library target, so the core path uses
# `clang -extract-api` on the public headers (which emits symbols only for the input files, excluding
# the SDK) and feeds the result to `docc convert` together with the `InferKit.docc` catalog.
#
# The three Swift companions (InferKitFoundationModels, InferKitMLX, InferKitAppleSwift) are Swift
# targets, each with its own `.docc` catalog. InferKitFoundationModels and InferKitAppleSwift build with
# the swift-docc-plugin (`swift package generate-documentation`). The plugin extracts a symbol graph
# for every dependency, and on mlx-swift's C++ Cmlx target `clang -extract-api` parses C++ headers as
# C and fails, so InferKitMLX extracts its own module's graph with `swift-symbolgraph-extract` and
# converts it with `docc convert`, which yields the pages the plugin does.
#
# Usage:
#   Tools/docc/build.sh [output-dir]        # core only (default: ./.docc-build/InferKit.doccarchive)
#   Tools/docc/build.sh --preview           # build the core then serve locally with `docc preview`
#   Tools/docc/build.sh --companion <name>  # build one companion (InferKitFoundationModels | InferKitMLX | InferKitAppleSwift)
#   Tools/docc/build.sh --all               # core + all three companions
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Builds InferKitMLX's DocC from the symbol graph of its own module, extracted after a build.
build_mlx_from_its_module() {
    local out="$1"
    local dir="$ROOT/InferKitMLX"
    echo "==> Building InferKitMLX DocC (swift-symbolgraph-extract)"
    ( cd "$dir" && swift build --target InferKitMLX )
    local bin maps
    bin="$(cd "$dir" && swift build --show-bin-path)"
    maps="$(dirname "$(dirname "$bin")")/Intermediates.noindex/GeneratedModuleMaps"
    if [ ! -f "$maps/InferKit.modulemap" ]; then
        echo "error: no InferKit.modulemap under $maps" >&2
        exit 1
    fi
    # The C targets the module imports (Cmlx, _NumericsShims) are found through their own module maps.
    local clang_maps=()
    while IFS= read -r map; do
        clang_maps+=(-Xcc "-fmodule-map-file=$map" -Xcc "-I$(dirname "$map")")
    done < <(find "$dir/.build/checkouts" -path '*/include/module.modulemap')
    (
        graphs="$(mktemp -d)"
        trap 'rm -rf "$graphs"' EXIT
        xcrun swift-symbolgraph-extract -module-name InferKitMLX \
            -target arm64-apple-macosx14.0 -sdk "$(xcrun --show-sdk-path)" \
            -I "$bin" -F "$bin" \
            -Xcc "-I$ROOT/Sources/InferKit/include" -Xcc "-fmodule-map-file=$maps/InferKit.modulemap" \
            ${clang_maps[@]+"${clang_maps[@]}"} \
            -minimum-access-level public -output-dir "$graphs"
        rm -rf "$out"
        xcrun docc convert "$dir/Sources/InferKitMLX/InferKitMLX.docc" \
            --fallback-display-name InferKitMLX \
            --fallback-bundle-identifier InferKitMLX \
            --additional-symbol-graph-dir "$graphs" \
            --emit-lmdb-index \
            --output-path "$out"
    )
}

# Builds one Swift companion package's DocC.
build_companion() {
    local pkg="$1"
    local out="$ROOT/.docc-build/$pkg.doccarchive"
    mkdir -p "$ROOT/.docc-build"
    if [ "$pkg" = "InferKitMLX" ]; then
        build_mlx_from_its_module "$out"
    else
        echo "==> Building $pkg DocC (swift-docc-plugin)"
        ( cd "$ROOT/$pkg" && swift package --allow-writing-to-directory "$out" \
            generate-documentation --target "$pkg" --output-path "$out" )
    fi
    local pages
    pages="$(find "$out/data/documentation" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    echo "==> Built $out ($pages documentation pages)"
}

case "${1:-}" in
    --companion)
        build_companion "${2:?usage: --companion <InferKitFoundationModels|InferKitMLX|InferKitAppleSwift>}"
        exit 0
        ;;
    --all)
        BUILD_ALL=1
        ;;
esac

OUTPUT="${1:-.docc-build/InferKit.doccarchive}"
PREVIEW=0
case "${1:-}" in
    --preview) PREVIEW=1; OUTPUT=".docc-build/InferKit.doccarchive" ;;
    --all)     OUTPUT=".docc-build/InferKit.doccarchive" ;;
esac

SDK="$(xcrun --show-sdk-path)"
GRAPH_DIR="$(mktemp -d)"
trap 'rm -rf "$GRAPH_DIR"' EXIT

echo "==> Extracting the Objective-C symbol graph"
xcrun clang -extract-api --product-name=InferKit -x objective-c-header \
    -target arm64-apple-macos11.0 -isysroot "$SDK" \
    -I Sources/InferKit/include -I Sources/InferKit/include/InferKit \
    Sources/InferKit/include/InferKit/*.h \
    -o "$GRAPH_DIR/InferKit.symbols.json"

echo "==> Converting the DocC catalog"
mkdir -p "$(dirname "$OUTPUT")"
rm -rf "$OUTPUT"

if [ "$PREVIEW" = "1" ]; then
    exec xcrun docc preview Sources/InferKit/InferKit.docc \
        --fallback-display-name InferKit \
        --fallback-bundle-identifier org.inferkit.InferKit \
        --additional-symbol-graph-dir "$GRAPH_DIR"
fi

xcrun docc convert Sources/InferKit/InferKit.docc \
    --fallback-display-name InferKit \
    --fallback-bundle-identifier org.inferkit.InferKit \
    --additional-symbol-graph-dir "$GRAPH_DIR" \
    --output-path "$OUTPUT"

PAGES="$(find "$OUTPUT/data/documentation" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "==> Built $OUTPUT ($PAGES documentation pages)"

if [ "${BUILD_ALL:-0}" = "1" ]; then
    build_companion InferKitFoundationModels
    build_companion InferKitMLX
    build_companion InferKitAppleSwift
fi
