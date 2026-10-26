#!/bin/bash
#
# Builds the macOS 26 companions, InferKitFoundationModels and InferKitAppleSwift, as dynamic frameworks
# that link the core from InferKitMLX.framework instead of carrying their own copy.
#
#   InferKitFoundationModels.xcframework   dynamic — InferKitFoundationModels.framework
#   InferKitAppleSwift.xcframework         dynamic — InferKitAppleSwift.framework
#
# ONE CORE. Each companion package depends on the core's `InferKit` product, so a SwiftPM build of it
# compiles the core too. The framework binary is linked from the companion target's own objects alone,
# and every core symbol it uses stays undefined until the loader binds it to the core inside
# InferKitMLX.framework. A host that embeds InferKitMLX.framework and these companions holds exactly one
# copy of every NFK class, so a descriptor a companion makes is an NFKModelDescriptor to the host.
#
# THE FLOOR. Each framework's minimum is macOS 26 / iOS 26, the packages' floor, while InferKitMLX's is
# macOS 14 / iOS 17. A host with a lower floor embeds the companion and opens it with dlopen only on a
# system at or above it, then adds its descriptors by provider class name:
#
#   InferKitFoundationModels   NFKFoundationModelsDescriptors
#   InferKitAppleSwift         NFKAppleSwiftModelDescriptors
#
# Each framework's run path is the directory that holds it (@loader_path/../../.. on macOS, inside a
# versioned bundle; @loader_path/.. on iOS), so InferKitMLX.framework embedded beside it resolves.
#
#   Tools/xcframework/build-companions.sh                          # both, against xcframework-build/
#   Tools/xcframework/build-companions.sh --against PATH           # an InferKitMLXDynamic.xcframework
#   Tools/xcframework/build-companions.sh --packages apple-swift    # foundation-models | apple-swift
#   Tools/xcframework/build-companions.sh --slices macos           # default: every slice --against holds
#   Tools/xcframework/build-companions.sh --no-swift-interfaces
#   Tools/xcframework/build-companions.sh --verify                 # open each framework in a host
#   Tools/xcframework/build-companions.sh --output DIR
#
# The framework it links against comes from `build-mlx.sh --variant dynamic`, which `--companions`
# runs this script after. Run it from a checkout whose root folder is named InferKit: each package's
# `../` dependency takes the core's identity from that folder name.
#
# Requires: Xcode 26 or later (its macOS 26 / iOS 26 SDKs).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=inferkit-version.sh
source "$ROOT/Tools/xcframework/inferkit-version.sh"
VERSION="$(inferkit_version "$ROOT")"
API_VERSION="$(inferkit_api_version "$ROOT")"
[ -n "$VERSION" ] && [ -n "$API_VERSION" ] || { echo "cannot read the version or API version from $ROOT"; exit 1; }
OUTPUT="$ROOT/xcframework-build"
AGAINST=""
PACKAGES="foundation-models apple-swift"
SLICES=""
INTERFACES=yes
VERIFY=no
while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2;;
        --against) AGAINST="$2"; shift 2;;
        --packages) PACKAGES="$(echo "$2" | tr ',' ' ')"; shift 2;;
        --slices) SLICES="$(echo "$2" | tr ',' ' ')"; shift 2;;
        --no-swift-interfaces) INTERFACES=no; shift;;
        --verify) VERIFY=yes; shift;;
        *) echo "unknown argument: $1"; exit 2;;
    esac
done
AGAINST="${AGAINST:-$OUTPUT/InferKitMLXDynamic.xcframework}"
[ -d "$AGAINST" ] || { echo "no InferKitMLX dynamic xcframework at $AGAINST; run build-mlx.sh --variant dynamic first"; exit 1; }

# The InferKitMLX.framework of one slice, by the directory name -create-xcframework gave it.
mlx_slice() {                           # mlx_slice <tag>  →  the slice directory, or nothing
    local directory
    for directory in "$AGAINST"/*/; do
        directory="${directory%/}"
        case "$tag:$(basename "$directory")" in
            macos:macos-*|iossim:ios-*-simulator) echo "$directory"; return;;
            ios:ios-*) case "$(basename "$directory")" in *-simulator) ;; *) echo "$directory"; return;; esac;;
        esac
    done
}
if [ -z "$SLICES" ]; then
    for tag in macos ios iossim; do
        if [ -n "$(mlx_slice)" ]; then
            SLICES="$SLICES $tag"
        fi
    done
fi
for tag in $SLICES; do
    case "$tag" in
        macos|ios|iossim) ;;
        *) echo "--slices takes macos, ios, iossim; not '$tag'"; exit 2;;
    esac
    [ -d "$(mlx_slice)/InferKitMLX.framework" ] || { echo "$AGAINST has no $tag slice to link against"; exit 1; }
done
for package in $PACKAGES; do
    case "$package" in
        foundation-models|apple-swift) ;;
        *) echo "--packages takes foundation-models, apple-swift; not '$package'"; exit 2;;
    esac
done

module_of() {                           # module_of <package>
    case "$1" in
        foundation-models) echo InferKitFoundationModels;;
        apple-swift)       echo InferKitAppleSwift;;
    esac
}
# The system frameworks each package's target links. The Swift objects also autolink what they import.
frameworks_of() {                       # frameworks_of <package>
    case "$1" in
        foundation-models) echo "-framework FoundationModels";;
        apple-swift)       echo "-framework Speech -framework Vision -framework AVFoundation -framework Translation";;
    esac
}

WORK="$OUTPUT/companion-intermediates"
rm -rf "$WORK"
for package in $PACKAGES; do
    rm -rf "$OUTPUT/$(module_of "$package").xcframework"
done
mkdir -p "$WORK"

write_framework_plist() {               # write_framework_plist <plist> <module> <tag>
    local plist="$1" module="$2" tag="$3" platform=MacOSX
    case "$tag" in
        ios)    platform=iPhoneOS;;
        iossim) platform=iPhoneSimulator;;
    esac
    mkdir -p "$(dirname "$plist")"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key><string>en</string>
	<key>CFBundleExecutable</key><string>$module</string>
	<key>CFBundleIdentifier</key><string>com.inferkit.$module</string>
	<key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
	<key>CFBundleName</key><string>$module</string>
	<key>CFBundlePackageType</key><string>FMWK</string>
	<key>CFBundleShortVersionString</key><string>$VERSION</string>
	<key>CFBundleVersion</key><string>$VERSION</string>
	<key>NFKAPIVersion</key><string>$API_VERSION</string>
	<key>CFBundleSupportedPlatforms</key><array><string>$platform</string></array>
	<key>MinimumOSVersion</key><string>26.0</string>
</dict>
</plist>
PLIST
}

companion() {           # companion <package> <destination> <tag> <sdk> <clang target> <configuration dir>
    local package="$1" destination="$2" tag="$3" sdk="$4" target="$5" configuration="$6"
    local module directory derived
    module="$(module_of "$package")"
    directory="$ROOT/$module"
    derived="$WORK/dd-$package-$tag"
    echo "==> building $module for $tag"
    ( cd "$directory" && xcodebuild build \
        -scheme "$module" \
        -destination "generic/platform=$destination" \
        -configuration Release \
        -derivedDataPath "$derived" \
        -skipPackagePluginValidation \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        ARCHS=arm64 ONLY_ACTIVE_ARCH=NO \
        ENABLE_CODE_COVERAGE=NO CLANG_COVERAGE_MAPPING=NO SWIFT_ENABLE_CODE_COVERAGE=NO \
        ENABLE_TESTABILITY=NO \
        > "$WORK/$package-$tag.log" 2>&1 ) || { tail -25 "$WORK/$package-$tag.log"; exit 1; }

    # The companion target's objects only (Xcode names a package target's directory <target>-t.build), on
    # this platform's configuration directory. The core's objects sit in InferKit-t.build beside them and
    # are left out: InferKitMLX.framework supplies the core.
    find "$derived/Build/Intermediates.noindex" -path "*/$configuration/*" \
        -path "*/$module-t.build/Objects-normal/*" -name '*.o' > "$WORK/$package-$tag-objects.txt"
    [ -s "$WORK/$package-$tag-objects.txt" ] || { echo "no $module objects in the $tag build"; exit 1; }
    local generated
    generated="$(find "$derived/Build" -name "$module-Swift.h" | head -1 || true)"
    [ -n "$generated" ] || { echo "no generated Objective-C header for $module in the $tag build"; exit 1; }

    local against framework inner rpath
    against="$(mlx_slice)"
    framework="$WORK/$tag-$package/$module.framework"
    inner="$framework"
    rpath="@loader_path/.."
    if [ "$tag" = macos ]; then
        inner="$framework/Versions/A"
        rpath="@loader_path/../../.."
    fi
    mkdir -p "$inner/Headers" "$inner/Modules"

    # Undefined symbols are an error, so a core symbol InferKitMLX.framework does not export fails the
    # link here, not the host's dlopen.
    # shellcheck disable=SC2046
    xcrun clang -dynamiclib -target "$target" -isysroot "$(xcrun --sdk "$sdk" --show-sdk-path)" \
        -filelist "$WORK/$package-$tag-objects.txt" \
        -F "$against" -framework InferKitMLX \
        -framework Foundation $(frameworks_of "$package") \
        -L "$(xcode-select -p)/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/$sdk" -L /usr/lib/swift \
        -Xlinker -install_name -Xlinker "@rpath/$module.framework/$module" \
        -Xlinker -rpath -Xlinker "$rpath" \
        -o "$inner/$module" > "$WORK/$package-$tag-link.log" 2>&1 \
        || { cat "$WORK/$package-$tag-link.log"; echo "the $tag $module framework did not link"; exit 1; }
    xcrun strip -x -S "$inner/$module" 2>/dev/null || true

    # One copy of the core: the companion defines none of the core's classes, and binds them all to
    # InferKitMLX.framework.
    local core_classes duplicated
    core_classes="$(nm -gU "$against/InferKitMLX.framework/InferKitMLX" | awk '{print $3}' | grep -F 'OBJC_CLASS_$_' | sort || true)"
    duplicated="$(nm -gU "$inner/$module" | awk '{print $3}' | grep -F 'OBJC_CLASS_$_' | sort | comm -12 - <(echo "$core_classes") || true)"
    [ -z "$duplicated" ] || { echo "$module defines classes InferKitMLX.framework also defines:"; echo "$duplicated"; exit 1; }
    otool -L "$inner/$module" | grep -q "@rpath/InferKitMLX.framework/InferKitMLX" \
        || { echo "$module does not link InferKitMLX.framework"; exit 1; }

    if [ "$tag" = macos ]; then
        write_framework_plist "$inner/Resources/Info.plist" "$module" "$tag"
    else
        write_framework_plist "$inner/Info.plist" "$module" "$tag"
    fi
    # The generated header imports the core's module, which the host finds in InferKitMLX.framework's
    # Headers directory.
    cp "$generated" "$inner/Headers/$module-Swift.h"
    cat > "$inner/Modules/module.modulemap" <<FRAMEWORKMAP
framework module $module {
    header "$module-Swift.h"
    export *
}
FRAMEWORKMAP
    if [ "$INTERFACES" = yes ]; then
        cp -R "$derived/Build/Products/Release"*/"$module.swiftmodule" "$inner/Modules/"
        [ -d "$inner/Modules/$module.swiftmodule" ] || { echo "no $module.swiftmodule in the $tag build"; exit 1; }
    fi
    if [ "$tag" = macos ]; then
        ln -sfn A "$framework/Versions/Current"
        for entry in "$module" Resources Modules Headers; do
            ln -sfn "Versions/Current/$entry" "$framework/$entry"
        done
    fi
}

for package in $PACKAGES; do
    for tag in $SLICES; do
        case "$tag" in
            macos)  companion "$package" "macOS"         macos  macosx          arm64-apple-macos26.0         Release;;
            ios)    companion "$package" "iOS"           ios    iphoneos        arm64-apple-ios26.0           Release-iphoneos;;
            iossim) companion "$package" "iOS Simulator" iossim iphonesimulator arm64-apple-ios26.0-simulator Release-iphonesimulator;;
        esac
    done
    module="$(module_of "$package")"
    echo "==> assembling $module.xcframework"
    arguments=()
    for tag in $SLICES; do
        arguments+=(-framework "$WORK/$tag-$package/$module.framework")
    done
    xcodebuild -create-xcframework "${arguments[@]}" \
        -output "$OUTPUT/$module.xcframework" > "$WORK/create-$package.log" 2>&1 \
        || { tail -20 "$WORK/create-$package.log"; exit 1; }
done

if [ "$VERIFY" = yes ]; then
    echo "==> verifying that a host opens each companion and builds its engines"
    "$ROOT/Tools/xcframework/verify-companions.sh" --output "$OUTPUT" --against "$AGAINST" \
        --packages "$(echo $PACKAGES | tr ' ' ',')" || exit 1
fi

rm -rf "$WORK"
echo
for package in $PACKAGES; do
    module="$(module_of "$package")"
    echo "==> $OUTPUT/$module.xcframework   (dynamic, $SLICES; InferKit $VERSION, API $API_VERSION, macOS 26 / iOS 26)"
    du -sh "$OUTPUT/$module.xcframework" | sed 's/^/    /'
done
echo "    each links the core from InferKitMLX.framework; embed that framework beside it"
exit 0
