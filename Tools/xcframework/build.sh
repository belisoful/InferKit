#!/bin/bash
#
# Builds a distributable InferKit.xcframework (macOS + iOS device + iOS simulator).
#
# Why this is not just `swift build`: a SwiftPM library target emits object files and a module, never a
# standalone binary — the library is materialized only when a consumer links it. `xcodebuild archive`
# on a package scheme emits one merged relocatable object, which `libtool -static` turns into the
# static library `-create-xcframework` wants. Public headers come from the include directory, so the
# framework's `#import <InferKit/NFKFoo.h>` resolves the same way it does through SwiftPM.
#
#   Tools/xcframework/build.sh              # -> .xcframework-build/InferKit.xcframework
#   Tools/xcframework/build.sh --output DIR
#
# THE CORE ONLY. InferKitMLX has its own script, `build-mlx.sh`, because it packages differently: its
# graph is absorbed into one binary and the Metal library has to travel with it.
# InferKitFoundationModels is a thin Swift wrapper over a macOS 26 / iOS 26 system framework. It can be
# packaged (add BUILD_LIBRARY_FOR_DISTRIBUTION and ship the .swiftmodule in each slice rather than
# headers), but it is small enough that SwiftPM is simpler.
#
# Requires: Xcode command line tools.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/.xcframework-build"
[ "${1:-}" = "--output" ] && OUTPUT="$2"

WORK="$OUTPUT/intermediates"
# Clean only this script's own outputs: build-mlx.sh writes its artifacts into the same directory,
# and removing the whole thing discards a twenty-minute MLX build alongside a twenty-second core one.
rm -rf "$OUTPUT/InferKit.xcframework" "$OUTPUT/InferKit.xcframework.zip" "$WORK"
mkdir -p "$WORK"

slice() {                       # slice <destination> <tag>
    local destination="$1" tag="$2"
    echo "==> archiving $tag"
    xcodebuild archive \
        -workspace "$ROOT/InferKit.xcworkspace" \
        -scheme InferKit \
        -destination "generic/platform=$destination" \
        -archivePath "$WORK/$tag" \
        -derivedDataPath "$WORK/dd-$tag" \
        SKIP_INSTALL=NO \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        > "$WORK/$tag.log" 2>&1 || { tail -20 "$WORK/$tag.log"; exit 1; }

    local object
    object="$(find "$WORK/$tag.xcarchive" -name 'InferKit.o' | head -1)"
    [ -n "$object" ] || { echo "no InferKit.o in the $tag archive"; exit 1; }
    mkdir -p "$WORK/$tag-lib"
    # Apple's libtool, not GNU's, which shadows it on a Homebrew PATH and rejects -static.
    xcrun libtool -static -o "$WORK/$tag-lib/libInferKit.a" "$object"
}

slice "macOS"          macos
slice "iOS"            ios
slice "iOS Simulator"  iossim

echo "==> assembling the xcframework"
xcodebuild -create-xcframework \
    -library "$WORK/macos-lib/libInferKit.a"  -headers "$ROOT/Sources/InferKit/include/InferKit" \
    -library "$WORK/ios-lib/libInferKit.a"    -headers "$ROOT/Sources/InferKit/include/InferKit" \
    -library "$WORK/iossim-lib/libInferKit.a" -headers "$ROOT/Sources/InferKit/include/InferKit" \
    -output "$OUTPUT/InferKit.xcframework" > "$WORK/create.log" 2>&1 \
    || { tail -20 "$WORK/create.log"; exit 1; }

rm -rf "$WORK"
echo "==> $OUTPUT/InferKit.xcframework"
/usr/libexec/PlistBuddy -c "Print :AvailableLibraries" "$OUTPUT/InferKit.xcframework/Info.plist" \
    | grep -E "LibraryIdentifier" | sed 's/^ */    /'
