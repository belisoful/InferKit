#!/bin/bash
#
# Builds distributable InferKitMLX xcframeworks (macOS + iOS device + iOS simulator, arm64).
#
# Two artifacts come out of ONE archive per slice, because they differ only in how the objects are
# packaged. Nothing in Package.swift changes for distribution.
#
# Coverage instrumentation is turned OFF explicitly: the package's generated scheme leaves it on for the
# build action, and `___llvm_profile_runtime` then goes undefined when these objects are linked outside
# a test bundle.
#
# `xcodebuild build`, NOT `archive`. Archiving needs SKIP_INSTALL=NO to place products where they can be
# collected, and that installs every product in the graph — including mlx-swift's `encuda` command-line
# tool, whose install fails an iOS archive with "Multiple commands produce .../encuda". Building runs no
# install step, so the objects are collected from the intermediates instead. A second `type: .dynamic`
# product was also tried and abandoned; it made no difference, because the conflict is the tool.
#
#   InferKitMLX.xcframework         static  — libInferKitMLX.a + headers + a modulemap
#   InferKitMLXDynamic.xcframework  dynamic — InferKitMLX.framework, Metal library and modules inside
#   release/                        each macOS framework zipped, the xcframework zips, and release.json
#
# WHY THIS IS POSSIBLE AT ALL. MLX needs `default.metallib`, which SwiftPM delivers as a
# `mlx-swift_Cmlx.bundle` resource; a bare library carries no resources, which is why this companion
# went unpackaged. MLX's loader has four fallbacks (mlx/backend/metal/device.cpp), and `current_binary_dir()`
# is `dladdr` on MLX's own code, so it names whichever mach-O image MLX was linked into:
#
#   1. <binary dir>/mlx.metallib                      — beside the consumer's binary
#   2. <binary dir>/Resources/mlx.metallib            — the dynamic route: inside the framework
#   3. mlx-swift_Cmlx.bundle in the main or any loaded bundle  — the static route inside an app bundle
#   4. <binary dir>/Resources/default.metallib
#
# The dynamic framework satisfies (2) on its own, ahead of any mlx-swift_Cmlx.bundle the host app
# ships, so it always loads the shaders it was built with. A static consumer satisfies (3) by shipping the
# `mlx-swift_Cmlx.bundle` this script places inside each static slice, beside the library it belongs
# to. Delivering it inside the xcframework is only delivery: an xcframework is a build-time container
# that never ships, and Xcode copies nothing out of a static one, so the consumer still places the
# bundle in their own product. One artifact means it cannot arrive without its shaders.
#
# ARM64 ONLY. MLX requires Apple Silicon, so an x86_64 slice could link and never evaluate.
#
# WHY BOTH. Measured on a consumer using one model: static links to 13.8 MB, dynamic to 14.2 MB — the
# hoped-for dead-stripping win is about 3%, because Cmlx is one merged object and MLX's runtime is
# densely interconnected. So the choice is deployment mechanics, not size. Static needs no embedding
# or signing and vends both modules from one plain modulemap; dynamic is self-contained and shared
# between several consumers. Neither dominates, so both ship.
#
#   Tools/xcframework/build-mlx.sh                        # both variants, all three slices
#   Tools/xcframework/build-mlx.sh --output DIR
#   Tools/xcframework/build-mlx.sh --verify               # link and run a consumer against what was built
#   Tools/xcframework/build-mlx.sh --variant static       # static | dynamic | both (default both)
#   Tools/xcframework/build-mlx.sh --slices macos,ios     # macos | ios | iossim (default all three)
#   Tools/xcframework/build-mlx.sh --no-swift-interfaces  # dynamic only, ~13 MB a slice
#   Tools/xcframework/build-mlx.sh --no-release           # skip release/ (the zips and release.json)
#
# A build that writes release/ checks the version first, before compiling anything: release.sh's rule,
# inferkit_check_release_version in inferkit-version.sh. The frameworks' CFBundleShortVersionString and
# CFBundleVersion, release.json, and the zip names all take the one version inferkit_version reads. A
# `--no-release` build is a development build and skips the check.
#
# Each slice is a full compile of those 289 translation units, so `--slices` is the difference between
# roughly seven minutes and twenty. `--variant` and `--no-swift-interfaces` cost nothing to build and
# only trim what is written.
#
# Requires: Xcode command line tools.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=inferkit-version.sh
source "$ROOT/Tools/xcframework/inferkit-version.sh"
VERSION="$(inferkit_version "$ROOT")"
API_VERSION="$(inferkit_api_version "$ROOT")"
[ -n "$VERSION" ] && [ -n "$API_VERSION" ] || { echo "cannot read the version or API version from $ROOT"; exit 1; }
PACKAGE="$ROOT/InferKitMLX"
OUTPUT="$ROOT/xcframework-build"
VERIFY=no
VARIANT=both
INTERFACES=yes
RELEASE=yes
SLICES="macos ios iossim"
while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2;;
        --verify) VERIFY=yes; shift;;
        --variant) VARIANT="$2"; shift 2;;
        --slices) SLICES="$(echo "$2" | tr ',' ' ')"; shift 2;;
        --no-swift-interfaces) INTERFACES=no; shift;;
        --no-release) RELEASE=no; shift;;
        *) echo "unknown argument: $1"; exit 2;;
    esac
done

case "$VARIANT" in
    static|dynamic|both) ;;
    *) echo "--variant takes static, dynamic, or both"; exit 2;;
esac
for requested in $SLICES; do
    case "$requested" in
        macos|ios|iossim) ;;
        *) echo "--slices takes macos, ios, iossim; not '$requested'"; exit 2;;
    esac
done
wants() {                               # wants <static|dynamic>
    [ "$VARIANT" = both ] || [ "$VARIANT" = "$1" ]
}
if wants dynamic && [ "$RELEASE" = yes ]; then
    inferkit_check_release_version "$ROOT" || { echo "nothing built; pass --no-release for a development build"; exit 1; }
fi
building() {                            # building <slice tag>
    case " $SLICES " in *" $1 "*) return 0;; *) return 1;; esac
}

WORK="$OUTPUT/mlx-intermediates"
rm -rf "$WORK" "$OUTPUT/InferKitMLX.xcframework" "$OUTPUT/InferKitMLXDynamic.xcframework" \
    "$OUTPUT/release" \
    "$OUTPUT/CoreHeaders" "$OUTPUT/InferKit.modulemap" \
    "$OUTPUT/MetalResources"       # earlier layouts kept these beside the xcframework
mkdir -p "$WORK"

# The public headers a consumer compiles against: the core's, plus the generated Objective-C interface
# for everything `@objc` in InferKitMLX. Both modules are declared in one plain modulemap — a static
# library may do that, where clang requires every module inside a framework to be a framework module.
write_headers() {                       # write_headers <destination> <generated header> [framework]
    local destination="$1" generated="$2" kind="${3:-library}"
    rm -rf "$destination"
    mkdir -p "$destination"
    cp "$generated" "$destination/InferKitMLX-Swift.h"
    if [ "$kind" = framework ]; then
        # The framework carries the core's headers in Headers/InferKit (write_framework_modules).
        mkdir -p "$(dirname "$destination")/Modules"
        cat > "$(dirname "$destination")/Modules/module.modulemap" <<'FRAMEWORKMAP'
framework module InferKitMLX {
    header "InferKitMLX-Swift.h"
    export *
}
FRAMEWORKMAP
        return
    fi
    mkdir -p "$destination/InferKit"
    cp "$ROOT/Sources/InferKit/include/InferKit/"*.h "$ROOT/Sources/InferKitC/include/InferKitC/"*.h "$destination/InferKit/"
    cat > "$destination/module.modulemap" <<'MODULEMAP'
// The core travels inside this library rather than beside it, so there is exactly one copy of every
// NFK class: an object a consumer makes and one a backend makes are the same class.
// The plugin ABI and the C API are pure C, included on their own and never through the umbrella.
module InferKit {
    umbrella header "InferKit/InferKit.h"
    exclude header "InferKit/NFKInferKitC.h"
    exclude header "InferKit/NFKPluginABI.h"
    export *
    module * { export * }
}

module InferKitMLX {
    header "InferKitMLX-Swift.h"
    export *
}
MODULEMAP
}

# The two plain modules the dynamic framework carries in its Headers directory, which a consumer passes
# with `-I`. Clang looks up a module named X in `<search dir>/X/module.modulemap`, so each sits in a
# directory of its own name. Clang accepts only framework modules in a framework's own module map, and
# a map found through a search path is not bound by that rule. They live in Headers and not in Modules
# because clang also reads `<search dir>/module.modulemap`: through `-I Modules` it reads the
# framework's own map a second time and refuses the framework module from the wrong directory.
#
#   InferKit/         the core. Its headers sit beside the map, so `#import <InferKit/NFKFoo.h>`
#                     resolves through the same `-I`. This is the only core module a consumer of the
#                     framework needs; a second one on another search path redefines the module.
#   InferKitMLXObjC/  the Objective-C interface of InferKitMLX, for Swift clients that bind through
#                     Objective-C dispatch. It is the generated header with its
#                     `external_source_symbol` attribute switched off: that attribute tells Swift the
#                     declarations belong to the Swift module InferKitMLX, and Swift then imports every
#                     class with no members. The module never imports InferKitMLX, so a client that
#                     imports it loads no Swift interface of InferKitMLX or MLX.
write_framework_modules() {             # write_framework_modules <Headers directory> <generated header>
    local headers="$1" generated="$2"
    local objc="$headers/InferKitMLXObjC/InferKitMLXObjC.h"
    mkdir -p "$headers/InferKit" "$headers/InferKitMLXObjC"
    cp "$ROOT/Sources/InferKit/include/InferKit/"*.h "$ROOT/Sources/InferKitC/include/InferKitC/"*.h "$headers/InferKit/"
    cat > "$headers/InferKit/module.modulemap" <<'COREMAP'
module InferKit {
    umbrella header "InferKit.h"
    exclude header "NFKInferKitC.h"
    exclude header "NFKPluginABI.h"
    export *
    module * { export * }
}
COREMAP
    {
        echo "// The Objective-C interface of InferKitMLX $VERSION (API $API_VERSION), generated from InferKitMLX-Swift.h."
        sed 's/__has_attribute(external_source_symbol)/0/' "$generated"
    } > "$objc"
    # A generator that guards the attribute differently would leave it on, and Swift would import empty
    # classes again; the Swift probe in verify-mlx.sh catches that too, but only for a macOS build.
    if grep -q '__has_attribute(external_source_symbol)' "$objc" || ! grep -q 'external_source_symbol(language' "$objc"; then
        echo "the generated header no longer guards external_source_symbol the way InferKitMLXObjC expects"
        exit 1
    fi
    cat > "$headers/InferKitMLXObjC/module.modulemap" <<'OBJCMAP'
module InferKitMLXObjC {
    header "InferKitMLXObjC.h"
    link framework "InferKitMLX"
    export *
}
OBJCMAP
}

# The C modules the Swift interfaces import: `Cmlx` (from MLX, MLXFast, MLXRandom) and `_NumericsShims`
# (from swift-numerics' RealModule). SwiftPM finds them in the package checkouts; a binary consumer has
# only the framework, so they ship in its Modules directory. A Swift consumer already passes that
# directory with `-I` to reach the dependency `.swiftmodule`s, and clang looks up a module named X in
# `<search dir>/X/module.modulemap`, so these need no further flag. Cmlx's headers include each other
# as "mlx/c/array.h", resolved through the header search path, so its `mlx/` tree sits at the Modules
# root while its module map and umbrella header sit in `Cmlx/`.
write_c_modules() {                     # write_c_modules <Modules directory> <package checkouts>
    local modules="$1" checkouts="$2"
    local cmlx="$checkouts/mlx-swift/Source/Cmlx/include"
    local shims="$checkouts/swift-numerics/Sources/_NumericsShims/include"
    [ -f "$cmlx/module.modulemap" ] || { echo "no Cmlx module map at $cmlx"; exit 1; }
    [ -f "$shims/module.modulemap" ] || { echo "no _NumericsShims module map at $shims"; exit 1; }
    mkdir -p "$modules/Cmlx" "$modules/_NumericsShims"
    cp "$cmlx/module.modulemap" "$cmlx/mlx.h" "$modules/Cmlx/"
    cp -R "$cmlx/mlx" "$modules/"
    cp "$shims/"* "$modules/_NumericsShims/"
    # The checkouts are read-only, and the copies would be too.
    chmod -R u+w "$modules/Cmlx" "$modules/mlx" "$modules/_NumericsShims"
}

# A hand-linked framework needs its own Info.plist. `-create-xcframework` reads the identifier and the
# supported platform from it, and the loader reads the executable name.
write_framework_plist() {               # write_framework_plist <framework inner directory> <tag>
    local inner="$1" tag="$2" platform=MacOSX minimum=14.0
    case "$tag" in
        ios)    platform=iPhoneOS; minimum=17.0;;
        iossim) platform=iPhoneSimulator; minimum=17.0;;
    esac
    local plist="$inner/Resources/Info.plist"
    [ "$tag" = macos ] || plist="$inner/Info.plist"
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key><string>en</string>
	<key>CFBundleExecutable</key><string>InferKitMLX</string>
	<key>CFBundleIdentifier</key><string>com.inferkit.InferKitMLX</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>CFBundleName</key><string>InferKitMLX</string>
	<key>CFBundlePackageType</key><string>FMWK</string>
	<key>CFBundleShortVersionString</key><string>$VERSION</string>
	<key>CFBundleVersion</key><string>$VERSION</string>
	<key>NFKAPIVersion</key><string>$API_VERSION</string>
	<key>CFBundleSupportedPlatforms</key><array><string>$platform</string></array>
	<key>MinimumOSVersion</key><string>$minimum</string>
</dict>
</plist>
PLIST
}

slice() {               # slice <destination> <tag> <swift lib dir> <clang target> <configuration dir>
    local destination="$1" tag="$2"
    echo "==> building $tag (this compiles MLX's 289 C++ translation units)"
    ( cd "$PACKAGE" && xcodebuild build \
        -scheme InferKitMLX \
        -destination "generic/platform=$destination" \
        -configuration Release \
        -derivedDataPath "$WORK/dd-$tag" \
        -skipPackagePluginValidation \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        ARCHS=arm64 ONLY_ACTIVE_ARCH=NO \
        ENABLE_CODE_COVERAGE=NO CLANG_COVERAGE_MAPPING=NO SWIFT_ENABLE_CODE_COVERAGE=NO \
        ENABLE_TESTABILITY=NO \
        > "$WORK/$tag.log" 2>&1 ) || { tail -25 "$WORK/$tag.log"; exit 1; }

    local metallib generated
    # `|| true` so the explicit message below reports the problem: under `pipefail` a failing find
    # would otherwise take `set -e` and exit with nothing said.
    metallib="$(find "$WORK/dd-$tag/Build" -name default.metallib -path '*mlx-swift_Cmlx.bundle*' | head -1 || true)"
    generated="$(find "$WORK/dd-$tag/Build" -name 'InferKitMLX-Swift.h' | head -1 || true)"
    [ -n "$metallib" ] || { echo "no default.metallib in the $tag build"; exit 1; }
    [ -n "$generated" ] || { echo "no generated Objective-C header in the $tag build"; exit 1; }

    # ---- the static library: every target's object merged into one archive ----
    # The dynamic framework is linked FROM this library, so it is built whenever either variant is.
    mkdir -p "$WORK/$tag-static"
    # Apple's libtool, not GNU's, which shadows it on a Homebrew PATH and rejects -static. A target
    # whose objects are all inlined away contributes an empty member, which is a warning, not a fault.
    # Every target's objects, restricted to THIS platform's configuration directory. Building for iOS
    # also builds mlx-swift's command-line tool for the host, and those macOS objects sit in a plain
    # `Release/` beside the `Release-iphoneos/` ones; sweeping both in makes the linker reject the
    # library ("building for iOS, but linking in object file built for macOS"). The tool's own objects
    # are excluded outright, because an executable brings a `_main`.
    find "$WORK/dd-$tag/Build/Intermediates.noindex" -path "*/$5/*" -path '*/Objects-normal/*' -name '*.o' \
        | grep -v '/encuda.build/' > "$WORK/$tag-objects.txt"
    [ -s "$WORK/$tag-objects.txt" ] || { echo "no objects in the $tag build"; exit 1; }
    xcrun libtool -static -o "$WORK/$tag-static/libInferKitMLX.a" \
        -filelist "$WORK/$tag-objects.txt" 2>&1 | grep -v "has no symbols" || true
    # Debug information is most of the archive's bulk and none of its function.
    xcrun strip -S "$WORK/$tag-static/libInferKitMLX.a" 2>/dev/null || true
    if wants static; then write_headers "$WORK/$tag-headers" "$generated"; fi

    # ---- the Metal library a static consumer ships ----
    # Staged per slice here and copied into the assembled xcframework below, because
    # `-create-xcframework` builds the slice directories itself and takes only a library and headers.
    # The dynamic framework carries its own copy inside, so this is the static route only. The bundle
    # is the single form: it is what Copy Bundle Resources produces for an app, and it also resolves
    # beside a command-line binary, whose main bundle is its own directory.
    if wants static; then
        if [ "$tag" = macos ]; then
            mkdir -p "$WORK/$tag-metal/mlx-swift_Cmlx.bundle/Contents/Resources"
            cp "$metallib" "$WORK/$tag-metal/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
        else
            mkdir -p "$WORK/$tag-metal/mlx-swift_Cmlx.bundle"
            cp "$metallib" "$WORK/$tag-metal/mlx-swift_Cmlx.bundle/default.metallib"
        fi
    fi
    # ---- the dynamic framework, linked from those same objects ----
    if ! wants dynamic; then
        return
    fi
    local framework="$WORK/$tag-dynamic/InferKitMLX.framework"
    local inner="$framework"
    [ "$tag" = macos ] && inner="$framework/Versions/A"
    mkdir -p "$inner/Resources" "$inner/Modules"

    # `-all_load` and NO `-dead_strip`: `NFKDynamicBackend` resolves providers through
    # `NSClassFromString`, so a class with no static reference is still reachable and must survive.
    # It costs about 1.3 MB against a dead-stripped link, which is the right trade for a binary whose
    # discovery mechanism is by name.
    # `-isysroot` is not optional: without it clang links this slice against the macOS SDK's frameworks
    # whatever `-target` says, and ld rejects the Metal stub as built for the wrong platform.
    xcrun clang -dynamiclib -target "$4" -isysroot "$(xcrun --sdk "$3" --show-sdk-path)" \
        -Xlinker -all_load "$WORK/$tag-static/libInferKitMLX.a" \
        -framework Metal -framework Accelerate -framework Foundation -framework CoreGraphics -lc++ \
        -L "$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/$3" -L /usr/lib/swift \
        -Xlinker -install_name -Xlinker "@rpath/InferKitMLX.framework/InferKitMLX" \
        -o "$inner/InferKitMLX" 2>&1 | grep -viE "swiftCompatibility|warning" || true
    [ -f "$inner/InferKitMLX" ] || { echo "the $tag framework binary did not link"; exit 1; }
    xcrun strip -x -S "$inner/InferKitMLX" 2>/dev/null || true

    write_framework_plist "$inner" "$tag"
    # Strategy 2 of MLX's loader, ahead of any mlx-swift_Cmlx.bundle beside the host app.
    cp "$metallib" "$inner/Resources/mlx.metallib"
    # The headers an Objective-C consumer needs, and the interfaces a Swift one does.
    write_headers "$inner/Headers" "$generated" framework
    write_framework_modules "$inner/Headers" "$generated"
    # The Swift interfaces cost about 13 MB a slice and an Objective-C consumer reads none of them,
    # so `--no-swift-interfaces` drops them. Swift then reaches the framework through InferKitMLXObjC
    # only, hence the note in the summary.
    if [ "$INTERFACES" = yes ]; then
        cp -R "$WORK/dd-$tag/Build/Products/Release"*/*.swiftmodule "$inner/Modules/"
        [ -d "$inner/Modules/InferKitMLX.swiftmodule" ] || { echo "no InferKitMLX.swiftmodule in the $tag build"; exit 1; }
        write_c_modules "$inner/Modules" "$WORK/dd-$tag/SourcePackages/checkouts"
    fi
    if [ "$tag" = macos ]; then
        ln -sfn A "$framework/Versions/Current"
        for entry in InferKitMLX Resources Modules Headers; do
            ln -sfn "Versions/Current/$entry" "$framework/$entry"
        done
    fi

}

if building macos;  then slice "macOS"          macos   macosx          arm64-apple-macos14.0         Release; fi
if building ios;    then slice "iOS"            ios     iphoneos        arm64-apple-ios17.0           Release-iphoneos; fi
if building iossim; then slice "iOS Simulator"  iossim  iphonesimulator arm64-apple-ios17.0-simulator Release-iphonesimulator; fi

if wants static; then
    echo "==> assembling the static xcframework"
    arguments=()
    for tag in $SLICES; do
        arguments+=(-library "$WORK/$tag-static/libInferKitMLX.a" -headers "$WORK/$tag-headers")
    done
    xcodebuild -create-xcframework "${arguments[@]}" \
        -output "$OUTPUT/InferKitMLX.xcframework" > "$WORK/create-static.log" 2>&1 \
        || { tail -20 "$WORK/create-static.log"; exit 1; }
    # Xcode names the slice directories itself, so each one is matched back to the platform it holds.
    for slice in "$OUTPUT/InferKitMLX.xcframework"/*/; do
        case "$(basename "$slice")" in
            macos*)      tag=macos;;
            *simulator*) tag=iossim;;
            *)           tag=ios;;
        esac
        cp -R "$WORK/$tag-metal/mlx-swift_Cmlx.bundle" "$slice"
    done
fi

if wants dynamic; then
    echo "==> assembling the dynamic xcframework"
    arguments=()
    for tag in $SLICES; do
        arguments+=(-framework "$WORK/$tag-dynamic/InferKitMLX.framework")
    done
    xcodebuild -create-xcframework "${arguments[@]}" \
        -output "$OUTPUT/InferKitMLXDynamic.xcframework" > "$WORK/create-dynamic.log" 2>&1 \
        || { tail -20 "$WORK/create-dynamic.log"; exit 1; }
fi

if [ "$VERIFY" = yes ]; then
    echo "==> verifying that what was built actually evaluates on the GPU"
    "$ROOT/Tools/xcframework/verify-mlx.sh" "$OUTPUT" || exit 1
fi

if wants dynamic && [ "$RELEASE" = yes ]; then
    echo "==> writing the release folder"
    # The vendored mlx core version is read from this build's checkout, which the cleanup below removes.
    checkouts="$(find "$WORK" -maxdepth 3 -type d -path '*/SourcePackages/checkouts' | head -1 || true)"
    "$ROOT/Tools/xcframework/release.sh" --output "$OUTPUT" --checkouts "$checkouts" || exit 1
fi

rm -rf "$WORK"
echo
if wants static; then
    echo "==> $OUTPUT/InferKitMLX.xcframework          (static, $SLICES)"
    echo "    each slice carries the mlx-swift_Cmlx.bundle that slice's consumer ships"
fi
if wants dynamic; then
    echo "==> $OUTPUT/InferKitMLXDynamic.xcframework   (dynamic, $SLICES; InferKit $VERSION, API $API_VERSION)"
    echo "    Headers/ carries the InferKit and InferKitMLXObjC modules"
    if [ "$INTERFACES" = no ]; then
        echo "    carries NO Swift module interfaces: Swift imports InferKitMLXObjC only"
    fi
    if [ "$RELEASE" = yes ]; then
        echo "==> $OUTPUT/release                          (the zips and release.json)"
    fi
fi
for artifact in "$OUTPUT/InferKitMLX.xcframework" "$OUTPUT/InferKitMLXDynamic.xcframework"; do
    if [ -e "$artifact" ]; then du -sh "$artifact" | sed 's/^/    /'; fi
done
exit 0
