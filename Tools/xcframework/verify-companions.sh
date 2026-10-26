#!/bin/bash
#
# Opens each companion framework the way a host with a lower floor does, and checks that it shares the
# host's one copy of the core.
#
# The frameworks are laid out as an app embeds them, InferKitMLX.framework and the companions side by
# side in Contents/Frameworks. Two probes run from Contents/MacOS:
#
#   - a C probe that links nothing and opens each companion with dlopen, so the companion's own run
#     path is what finds InferKitMLX.framework beside it;
#   - an Objective-C probe that links InferKitMLX.framework alone, as the host does, opens each
#     companion, adds its descriptors to a catalog by provider class name, and builds every engine
#     through the catalog. It checks that NFKModelDescriptor resolves to InferKitMLX.framework's image,
#     that each descriptor is that class, that each states its macOS 26 floor, and that Vision's lens
#     smudge engine answers an image.
#
#   Tools/xcframework/verify-companions.sh [--output DIR] [--against PATH] [--packages LIST]
#
# Called by build-companions.sh --verify. Only a macOS slice runs here; a build without one reports
# that and passes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/xcframework-build"
AGAINST=""
PACKAGES="foundation-models apple-swift"
while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2;;
        --against) AGAINST="$2"; shift 2;;
        --packages) PACKAGES="$(echo "$2" | tr ',' ' ')"; shift 2;;
        *) echo "unknown argument: $1"; exit 2;;
    esac
done
AGAINST="${AGAINST:-$OUTPUT/InferKitMLXDynamic.xcframework}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

MLX_SLICE="$(find "$AGAINST" -maxdepth 1 -type d -name 'macos*' 2>/dev/null | head -1 || true)"
[ -n "$MLX_SLICE" ] || { echo "    skipped: $AGAINST has no macOS slice, and an iOS binary cannot run here"; exit 0; }

APP="$WORK/Probe.app/Contents"
mkdir -p "$APP/MacOS" "$APP/Frameworks"
ditto "$MLX_SLICE/InferKitMLX.framework" "$APP/Frameworks/InferKitMLX.framework"
binaries=()
providers=()
for package in $PACKAGES; do
    case "$package" in
        foundation-models) module=InferKitFoundationModels; provider=NFKFoundationModelsDescriptors;;
        apple-swift)       module=InferKitAppleSwift;       provider=NFKAppleSwiftModelDescriptors;;
        *) echo "--packages takes foundation-models, apple-swift; not '$package'"; exit 2;;
    esac
    slice="$(find "$OUTPUT/$module.xcframework" -maxdepth 1 -type d -name 'macos*' 2>/dev/null | head -1 || true)"
    [ -n "$slice" ] || { echo "    skipped $module: no macOS slice was built"; continue; }
    ditto "$slice/$module.framework" "$APP/Frameworks/$module.framework"
    binaries+=("../Frameworks/$module.framework/$module")
    providers+=("$provider")
done
[ ${#binaries[@]} -gt 0 ] || exit 0

cat > "$WORK/open.c" <<'OPEN'
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
	for (int index = 1; index < argc; index++) {
		if (dlopen(argv[index], RTLD_NOW) == NULL) {
			printf("FAIL %s\n", dlerror());
			return 1;
		}
	}
	printf("OK\n");
	return 0;
}
OPEN

cat > "$WORK/host.m" <<'HOST'
@import InferKit;
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <dlfcn.h>

static NSString *ImageOf(const void *address) {
	Dl_info info;
	return dladdr(address, &info) != 0 && info.dli_fname != NULL ? @(info.dli_fname).lastPathComponent : @"?";
}

static CGImageRef GrayImage(size_t side) {
	NSMutableData *bytes = [NSMutableData dataWithLength:side * side * 4];
	memset(bytes.mutableBytes, 128, bytes.length);
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)bytes);
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	CGImageRef image = CGImageCreate(side, side, 8, 32, side * 4, space, kCGImageAlphaNoneSkipLast, provider,
									 NULL, false, kCGRenderingIntentDefault);
	CGColorSpaceRelease(space);
	CGDataProviderRelease(provider);
	return image;
}

// Arguments: pairs of a framework binary and its descriptor provider class.
int main(int argc, char **argv) {
	@autoreleasepool {
		NSString *core = ImageOf((__bridge const void *)NFKModelDescriptor.class);
		if (![core isEqualToString:@"InferKitMLX"]) { printf("FAIL the core resolves to %s\n", core.UTF8String); return 1; }
		NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
		for (int index = 1; index + 1 < argc; index += 2) {
			if (dlopen(argv[index], RTLD_NOW) == NULL) { printf("FAIL %s\n", dlerror()); return 1; }
			NSString *name = @(argv[index + 1]);
			Class provider = NSClassFromString(name);
			if (provider == Nil) { printf("FAIL no class %s\n", name.UTF8String); return 1; }
			NSString *image = ImageOf((__bridge const void *)provider);
			if (![@(argv[index]).lastPathComponent isEqualToString:image]) { printf("FAIL %s is in %s\n", name.UTF8String, image.UTF8String); return 1; }
			if (![ImageOf((__bridge const void *)NFKModelDescriptor.class) isEqualToString:core]) { printf("FAIL a second core loaded\n"); return 1; }
			if (![catalog addDescriptorsFromProviderClassName:name]) { printf("FAIL the catalog refused %s\n", name.UTF8String); return 1; }
			NSArray<NFKModelDescriptor *> *descriptors = [(Class<NFKModelDescriptorProvider>)provider modelDescriptors];
			if (descriptors.count == 0) { printf("FAIL %s lists no descriptors\n", name.UTF8String); return 1; }
			for (NFKModelDescriptor *descriptor in descriptors) {
				if (![descriptor isKindOfClass:NFKModelDescriptor.class]) { printf("FAIL duplicated classes\n"); return 1; }
				if (![descriptor.requirements.minimumOSVersions[@"macOS"] isEqualToString:@"26.0"]) {
					printf("FAIL %s states no macOS 26 floor\n", descriptor.identifier.UTF8String); return 1;
				}
				NSError *error = nil;
				id<NFKInferenceBackend> backend = [catalog backendForModelIdentifier:descriptor.identifier options:nil error:&error];
				if (backend == nil) { printf("FAIL build %s: %s\n", descriptor.identifier.UTF8String, error.localizedDescription.UTF8String); return 1; }
				printf("\n    %s: built %s from %s\n", descriptor.identifier.UTF8String, NSStringFromClass([(NSObject *)backend class]).UTF8String,
					   ImageOf((__bridge const void *)[(NSObject *)backend class]).UTF8String);
			}
		}
		if ([catalog descriptorWithIdentifier:@"vision-smudge"] != nil) {
			NSError *error = nil;
			id<NFKInferenceBackend> smudge = [catalog backendForModelIdentifier:@"vision-smudge" options:nil error:&error];
			CGImageRef image = GrayImage(256);
			NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }];
			NFKInferenceResult *result = [smudge runInferenceForRequest:request error:&error];
			CGImageRelease(image);
			if ([result outputForKey:NFKOutputClassifications] == nil) { printf("FAIL vision-smudge: %s\n", error.localizedDescription.UTF8String); return 1; }
			printf("\n    vision-smudge: answered an image\n");
		}
		printf("\nOK\n");
	}
	return 0;
}
HOST

xcrun clang -target arm64-apple-macos14.0 "$WORK/open.c" -o "$APP/MacOS/open" > "$WORK/open.log" 2>&1 \
    || { cat "$WORK/open.log"; echo "    FAILED — the C probe did not build"; exit 1; }
xcrun clang -target arm64-apple-macos14.0 -fobjc-arc -fmodules -fmodules-cache-path="$WORK/modules" \
    -F "$APP/Frameworks" -I "$APP/Frameworks/InferKitMLX.framework/Headers" \
    "$WORK/host.m" -framework InferKitMLX -framework Foundation -framework CoreGraphics \
    -Xlinker -rpath -Xlinker @executable_path/../Frameworks -o "$APP/MacOS/host" > "$WORK/host.log" 2>&1 \
    || { cat "$WORK/host.log"; echo "    FAILED — the host probe did not build"; exit 1; }

cd "$APP/MacOS"
opened="$(./open "${binaries[@]}" 2>&1 | tail -1 || true)"
[ "$opened" = OK ] || { echo "    FAILED — a companion did not open on its own run path: $opened"; exit 1; }
echo "    each companion opens on its own run path, InferKitMLX.framework beside it"
arguments=()
for index in "${!binaries[@]}"; do
    arguments+=("${binaries[$index]}" "${providers[$index]}")
done
# Vision logs its model loading to standard output without a newline, so each report line starts a new line.
./host "${arguments[@]}" > "$WORK/host-run.log" 2>&1 || true
grep '^    ' "$WORK/host-run.log" || true
if [ "$(tail -1 "$WORK/host-run.log")" != OK ]; then
    tail -3 "$WORK/host-run.log" | sed 's/^/    /'
    echo "    FAILED — the host probe"; exit 1
fi
echo "    one core: the companions' descriptors and backends are InferKitMLX.framework's classes"
