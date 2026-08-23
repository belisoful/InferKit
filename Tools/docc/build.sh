#!/bin/bash
#
# Builds DocC documentation for InferKit.
#
# The core is pure Objective-C. DocC needs a symbol graph, and neither Xcode's `docbuild` nor the
# swift-docc-plugin extracts one for a pure-Objective-C SwiftPM library target, so the core path uses
# `clang -extract-api` on the public headers (which emits symbols only for the input files, excluding
# the SDK) and feeds the result to `docc convert` together with the `InferKit.docc` catalog.
#
# The two Swift companions (InferKitFoundationModels, InferKitMLX) are Swift targets, so the
# swift-docc-plugin extracts their symbol graphs. Each carries its own `.docc` catalog and builds with
# `swift package generate-documentation`.
#
# Usage:
#   Tools/docc/build.sh [output-dir]        # core only (default: ./.docc-build/InferKit.doccarchive)
#   Tools/docc/build.sh --preview           # build the core then serve locally with `docc preview`
#   Tools/docc/build.sh --companion <name>  # build one companion (InferKitFoundationModels | InferKitMLX)
#   Tools/docc/build.sh --all               # core + both companions
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Builds one Swift companion package's DocC via the swift-docc-plugin.
build_companion() {
    local pkg="$1"
    local out="$ROOT/.docc-build/$pkg.doccarchive"
    echo "==> Building $pkg DocC (swift-docc-plugin)"
    mkdir -p "$ROOT/.docc-build"
    ( cd "$ROOT/$pkg" && swift package --allow-writing-to-directory "$out" \
        generate-documentation --target "$pkg" --output-path "$out" )
    local pages
    pages="$(find "$out/data/documentation" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
    echo "==> Built $out ($pages documentation pages)"
}

case "${1:-}" in
    --companion)
        build_companion "${2:?usage: --companion <InferKitFoundationModels|InferKitMLX>}"
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
fi
