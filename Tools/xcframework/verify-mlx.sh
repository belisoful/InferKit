#!/bin/bash
#
# Links a consumer against each InferKitMLX xcframework's macOS slice and RUNS it, so a packaging
# mistake fails here rather than at a consumer's first inference.
#
# Linking is not the check. A binary that links and cannot find `default.metallib` throws
# "Failed to load the default metallib" at the first array evaluation, which is why each probe runs a
# real model on the GPU instead of only constructing one.
#
#   Tools/xcframework/verify-mlx.sh [xcframework-build]
#
# Called by build-mlx.sh --verify. Each variant gets an Objective-C probe. The dynamic framework also
# gets two Swift probes: one that imports InferKitMLXObjC and loads no Swift interface of InferKitMLX,
# and one compiled from the module interfaces alone, because a missing C module beneath them (`Cmlx`,
# `_NumericsShims`) fails only a Swift consumer. The dynamic framework must load the Metal library it
# carries, `Resources/mlx.metallib`, even when the host ships an `mlx-swift_Cmlx.bundle` of its own, so
# the InferKitMLXObjC probe runs beside a decoy bundle and checks which library MLX resolves.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:-$ROOT/xcframework-build}"
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
echo "    static: a one-model consumer links to $(du -h "$WORK/static-run/probe" | awk '{print $1}') dead-stripped"
report "static (Objective-C)" "$(cd "$WORK/static-run" && ./probe 2>&1 | tail -1)"
fi

# ---- dynamic: what the framework must carry ----
FRAMEWORK="$DYNAMIC_SLICE/InferKitMLX.framework"
if [ -n "$DYNAMIC_SLICE" ]; then
for required in Resources/mlx.metallib Resources/Info.plist Modules/module.modulemap \
        Headers/InferKitMLX-Swift.h Headers/InferKit/module.modulemap Headers/InferKit/InferKit.h \
        Headers/InferKitMLXObjC/module.modulemap Headers/InferKitMLXObjC/InferKitMLXObjC.h; do
    [ -e "$FRAMEWORK/$required" ] || { echo "    dynamic: FAILED — no $required in the framework"; exit 1; }
done
# MLX's loader reaches Resources/default.metallib only after any mlx-swift_Cmlx.bundle the host ships.
if [ -e "$FRAMEWORK/Resources/default.metallib" ]; then
    echo "    dynamic: FAILED — the framework carries Resources/default.metallib, which a host's bundle shadows"; exit 1
fi
echo "    dynamic: Resources/mlx.metallib, and the InferKit and InferKitMLXObjC modules, are inside"
fi

# ---- dynamic, Objective-C: the framework alone, the Metal library and the core module inside it ----
if [ -n "$DYNAMIC_SLICE" ]; then
mkdir -p "$WORK/dynamic-run"
xcrun clang -fobjc-arc -fmodules -target "$TARGET" -w \
    -F "$DYNAMIC_SLICE" -framework InferKitMLX -I "$FRAMEWORK/Headers" \
    -fmodules-cache-path="$WORK/objc-module-cache" \
    "$WORK/probe.m" -framework Foundation -framework CoreGraphics \
    -L "$SWIFT_LIB" -L /usr/lib/swift -Xlinker -rpath -Xlinker /usr/lib/swift \
    -Xlinker -rpath -Xlinker "$DYNAMIC_SLICE" \
    -o "$WORK/dynamic-run/probe" 2>&1 | grep -viE "warning|swiftCompatibility" || true
[ -x "$WORK/dynamic-run/probe" ] || { echo "    dynamic: FAILED — did not link"; exit 1; }
echo "    dynamic: a one-model consumer links to $(du -h "$WORK/dynamic-run/probe" | awk '{print $1}') beside a $(du -h "$DYNAMIC_SLICE/InferKitMLX.framework/Versions/A/InferKitMLX" | awk '{print $1}') framework binary"
# Nothing is copied beside it: the Metal library has to come from inside the framework.
report "dynamic (Objective-C)" "$("$WORK/dynamic-run/probe" 2>&1 | tail -1)"
fi

# ---- dynamic, Swift through InferKitMLXObjC: the Objective-C interface only ----
# No `import InferKitMLX` and no `-framework` flag: the module map's `link framework` links it. The
# loaded-module trace proves no Swift interface of InferKitMLX or MLX was read, and the undefined
# symbols prove the probe binds to no Swift symbol of InferKitMLX, so it reaches the framework through
# Objective-C dispatch alone. This runs with or without --no-swift-interfaces.
if [ -n "$DYNAMIC_SLICE" ]; then
cat > "$WORK/objc-probe.swift" <<'OBJCPROBE'
import CoreGraphics
import Foundation
import InferKitMLXObjC

func fail(_ message: String) -> Never {
    print("FAIL: \(message)")
    exit(1)
}

guard let library = NFKMLXGPU.metalLibraryURL?.path,
      library.contains("InferKitMLX.framework/"), library.hasSuffix("/Resources/mlx.metallib") else {
    fail("MLX resolves \(NFKMLXGPU.metalLibraryURL?.path ?? "no Metal library"), not the framework's own")
}
NFKMLXReferenceModels.registerAll()
guard NFKMLXModelRegistry.registeredModelNames.contains("green-screen-keyer") else {
    fail("registerAll registered no green-screen-keyer")
}
guard !NFKModelCatalog.shared.descriptors.isEmpty else {
    fail("the shared catalog is empty")
}
do {
    // The Objective-C importer's name for +backendNamed:weightsURL:error:. The Swift module's own
    // spelling, backend(named:weightsURL:), is not part of the Objective-C interface.
    let backend = try NFKMLXModelRegistry.backendNamed("green-screen-keyer", weightsURL: nil)
    let side = 32
    var pixels = [UInt8](repeating: 0, count: side * side * 4)
    for index in stride(from: 0, to: pixels.count, by: 4) {
        pixels[index + 1] = 220
        pixels[index + 3] = 255
    }
    guard let context = CGContext(data: &pixels, width: side, height: side, bitsPerComponent: 8,
                                  bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let image = context.makeImage() else {
        fail("no test image")
    }
    let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
    guard result.output(forKey: NFKOutputImage) != nil else {
        fail("no output image")
    }
} catch {
    fail("\(error)")
}
print("OK")
OBJCPROBE
mkdir -p "$WORK/objc-swift-run/mlx-swift_Cmlx.bundle/Contents/Resources"
# A decoy with no valid shaders, where MLX looks third. The framework's own copy must come first.
printf 'not a Metal library' > "$WORK/objc-swift-run/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
xcrun swiftc -target "$TARGET" -F "$DYNAMIC_SLICE" -I "$FRAMEWORK/Headers" \
    -module-cache-path "$WORK/objc-swift-module-cache" \
    -emit-loaded-module-trace -emit-loaded-module-trace-path "$WORK/objc-swift-trace.json" \
    "$WORK/objc-probe.swift" -Xlinker -rpath -Xlinker "$DYNAMIC_SLICE" \
    -o "$WORK/objc-swift-run/probe" > "$WORK/objc-swift.log" 2>&1 || true
if [ ! -x "$WORK/objc-swift-run/probe" ]; then
    grep ": error:" "$WORK/objc-swift.log" | sort -u | head -5 | sed 's/^/      /' || true
    echo "    dynamic (Swift, InferKitMLXObjC): FAILED — did not compile"; exit 1
fi
if grep -qE 'InferKitMLX\.swiftmodule|/MLX\.swiftmodule|InferKitMLX\.swiftinterface' "$WORK/objc-swift-trace.json"; then
    echo "    dynamic (Swift, InferKitMLXObjC): FAILED — the probe loaded a Swift interface of InferKitMLX or MLX"; exit 1
fi
if xcrun nm -u "$WORK/objc-swift-run/probe" | grep -q '\$s11InferKitMLX'; then
    echo "    dynamic (Swift, InferKitMLXObjC): FAILED — the probe binds to Swift symbols of InferKitMLX"; exit 1
fi
report "dynamic (Swift, InferKitMLXObjC)" "$(cd "$WORK/objc-swift-run" && ./probe 2>&1 | tail -1)"
fi

# ---- dynamic, Swift: the interfaces and the C modules beneath them, from the framework alone ----
# The flags are the ones a consumer passes: `-F` for the framework, `-I` for its Modules directory (the
# dependency interfaces and the C modules), and `-I` for its Headers directory (the core's module). A
# private module cache makes the compiler build every interface from the shipped text instead of
# reusing an earlier result.
if [ -n "$DYNAMIC_SLICE" ]; then
if [ ! -d "$DYNAMIC_SLICE/InferKitMLX.framework/Modules/InferKitMLX.swiftmodule" ]; then
    # The C modules ship only beside the Swift interfaces, so their presence marks an interfaces build.
    if [ -d "$DYNAMIC_SLICE/InferKitMLX.framework/Modules/Cmlx" ]; then
        echo "    dynamic (Swift): FAILED — the framework carries the C modules but no InferKitMLX.swiftmodule"; exit 1
    fi
    echo "    dynamic (Swift): skipped, the framework was built with --no-swift-interfaces"
    exit 0
fi
cat > "$WORK/probe.swift" <<'SWIFTPROBE'
import Foundation
import InferKit
import InferKitMLX
import MLX

_ = NFKMLXModelRegistry.registeredModelNames
guard NFKMLXGPU.metalLibraryURL != nil else {
    print("FAIL: no Metal library")
    exit(1)
}
let sum = (MLXArray([1, 2, 3] as [Int32]) * 2).sum().item(Int32.self)
guard sum == 12 else {
    print("FAIL: evaluated \(sum), expected 12")
    exit(1)
}
print("OK")
SWIFTPROBE
mkdir -p "$WORK/swift-run"
xcrun swiftc -target "$TARGET" \
    -F "$DYNAMIC_SLICE" -I "$FRAMEWORK/Modules" -I "$FRAMEWORK/Headers" \
    -module-cache-path "$WORK/module-cache" \
    "$WORK/probe.swift" -framework InferKitMLX -Xlinker -rpath -Xlinker "$DYNAMIC_SLICE" \
    -o "$WORK/swift-run/probe" > "$WORK/swift.log" 2>&1 || true
if [ ! -x "$WORK/swift-run/probe" ]; then
    grep ": error:" "$WORK/swift.log" | sort -u | head -5 | sed 's/^/      /' || true
    echo "    dynamic (Swift): FAILED — did not compile"; exit 1
fi
report "dynamic (Swift)" "$("$WORK/swift-run/probe" 2>&1 | tail -1)"
fi
