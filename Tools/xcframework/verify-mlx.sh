#!/bin/bash
#
# Links a consumer against each InferKitMLX xcframework's macOS slice and RUNS it, so a packaging
# mistake fails here rather than at a consumer's first inference.
#
# Linking is not the check. A binary that links and cannot find `default.metallib` throws
# "Failed to load the default metallib" at the first array evaluation, which is why each probe runs a
# real model on the GPU instead of only constructing one.
#
#   Tools/xcframework/verify-mlx.sh [.xcframework-build]
#
# Called by build-mlx.sh --verify. Both probes are Objective-C: that is the surface a binary artifact
# vends. A Swift consumer needs MLX's own module interfaces, which is what SwiftPM is for.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT/.xcframework-build}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TARGET=arm64-apple-macos14.0

# Only a macOS slice can be executed here, and only a variant that was built can be checked. An
# iOS-only or single-variant build is a legitimate request, so those cases report and pass rather
# than failing: what is absent was never claimed.
STATIC_SLICE="$(find "$OUTPUT/InferKitMLX.xcframework" -maxdepth 1 -type d -name 'macos*' 2>/dev/null | head -1 || true)"
DYNAMIC_SLICE="$(find "$OUTPUT/InferKitMLXDynamic.xcframework" -maxdepth 1 -type d -name 'macos*' 2>/dev/null | head -1 || true)"

if [ -z "$STATIC_SLICE" ] && [ -z "$DYNAMIC_SLICE" ]; then
    if [ -d "$OUTPUT/InferKitMLX.xcframework" ] || [ -d "$OUTPUT/InferKitMLXDynamic.xcframework" ]; then
        echo "    skipped: no macOS slice was built, and an iOS binary cannot run here"
        exit 0
    fi
    echo "    FAILED — no xcframework found in $OUTPUT"; exit 1
fi

cat > "$WORK/probe.m" <<'PROBE'
@import InferKit;
@import InferKitMLX;
#import <Foundation/Foundation.h>

int main(void) {
	@autoreleasepool {
		NSError *error = nil;
		id<NFKInferenceBackend> backend = [NFKMLXZeroDCE backendWithWeightsURL:nil error:&error];
		if (backend == nil) { NSLog(@"FAIL build: %@", error); return 1; }

		size_t side = 32;
		NSMutableData *bytes = [NSMutableData dataWithLength:side * side * 4];
		memset(bytes.mutableBytes, 96, bytes.length);
		CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)bytes);
		CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
		CGImageRef image = CGImageCreate(side, side, 8, 32, side * 4, space,
										 kCGImageAlphaNoneSkipLast, provider, NULL, false,
										 kCGRenderingIntentDefault);
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }];
		NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
		if (result == nil) { NSLog(@"FAIL run: %@", error); return 1; }
		if ([result outputForKey:NFKOutputImage] == nil) { NSLog(@"FAIL: no output"); return 1; }
		// One copy of the core means an object made here and one made inside a backend are the same
		// class; two copies would pass every other check and fail this one.
		if (![request isKindOfClass:NFKInferenceRequest.class]) { NSLog(@"FAIL: duplicated classes"); return 1; }
		printf("OK\n");
	}
	return 0;
}
PROBE

report() {                              # report <name> <output>
    if [ "$2" = OK ]; then
        echo "    $1: evaluated on the GPU"
    else
        echo "    $1: FAILED — $2"; exit 1
    fi
}

SWIFT_LIB="$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx"

# ---- static: the artifact alone, the Metal library taken from inside it ----
if [ -n "$STATIC_SLICE" ]; then
mkdir -p "$WORK/static-run"
xcrun clang -fobjc-arc -fmodules -target "$TARGET" -w \
    -I "$STATIC_SLICE/Headers" -fmodule-map-file="$STATIC_SLICE/Headers/module.modulemap" \
    "$WORK/probe.m" "$STATIC_SLICE/libInferKitMLX.a" \
    -framework Metal -framework Accelerate -framework Foundation -framework CoreGraphics -lc++ \
    -L "$SWIFT_LIB" -L /usr/lib/swift -Xlinker -rpath -Xlinker /usr/lib/swift \
    -Xlinker -dead_strip -o "$WORK/static-run/probe" 2>&1 | grep -viE "warning|swiftCompatibility" || true
[ -x "$WORK/static-run/probe" ] || { echo "    static: FAILED — did not link"; exit 1; }
# From inside the slice, which is the copy a consumer makes: an xcframework is a build-time
# container, so a static consumer places the bundle in their own product themselves.
cp -R "$STATIC_SLICE/mlx-swift_Cmlx.bundle" "$WORK/static-run/"
report "static (Objective-C)" "$(cd "$WORK/static-run" && ./probe 2>&1 | tail -1)"
fi

# ---- dynamic: the framework alone, Metal library inside it ----
# Objective-C, because that is what this variant is for. A SWIFT consumer of the dynamic framework
# additionally needs MLX's module interfaces and the C module maps beneath them (`Cmlx`,
# `_NumericsShims`), which a binary artifact does not carry; that consumer wants SwiftPM.
if [ -n "$DYNAMIC_SLICE" ]; then
mkdir -p "$WORK/dynamic-run"
xcrun clang -fobjc-arc -fmodules -target "$TARGET" -w \
    -F "$DYNAMIC_SLICE" -framework InferKitMLX \
    -I "$OUTPUT/CoreHeaders" -fmodule-map-file="$OUTPUT/CoreHeaders/module.modulemap" \
    "$WORK/probe.m" -framework Foundation -framework CoreGraphics \
    -L "$SWIFT_LIB" -L /usr/lib/swift -Xlinker -rpath -Xlinker /usr/lib/swift \
    -Xlinker -rpath -Xlinker "$DYNAMIC_SLICE" \
    -o "$WORK/dynamic-run/probe" 2>&1 | grep -viE "warning|swiftCompatibility" || true
[ -x "$WORK/dynamic-run/probe" ] || { echo "    dynamic: FAILED — did not link"; exit 1; }
# Nothing is copied beside it: the Metal library has to come from inside the framework.
report "dynamic (Objective-C)" "$("$WORK/dynamic-run/probe" 2>&1 | tail -1)"
fi
