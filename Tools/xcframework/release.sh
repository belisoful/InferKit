#!/bin/bash
#
# Writes the release folder for a built InferKitMLX: the assets a release publishes, and release.json,
# the manifest an app reads to choose, download, and verify a version.
#
#   Tools/xcframework/release.sh [--output xcframework-build] [--checkouts DIR]
#
# build-mlx.sh runs it after a build that includes the dynamic variant. It reads what that build wrote
# in the output directory and writes `<output>/release/`:
#
#   InferKitMLX-<version>-<slice>.zip      each macOS slice's InferKitMLX.framework, the unit an app
#                                          installs into <root>/<version>/ beside release.json
#   InferKitMLXDynamic.xcframework.zip     the dynamic xcframework, for building against
#   InferKitMLX.xcframework.zip            the static xcframework, when it was built
#   release.json                           the manifest: versions, minimum systems, the commit, the
#                                          mlx-swift pin, and each asset's size and SHA-256
#
# An iOS app cannot load code it downloads, so only macOS slices become installable framework assets.
# The iOS slices ship inside the xcframework zips.
#
# `--checkouts` names the SwiftPM checkouts the build resolved, which is where the vendored mlx core
# version is read. Without it the script reads InferKitMLX/.build/checkouts, and records null when
# neither exists.
#
# The `signature` field is written empty and last, on its own line. The signing step fills it; the
# format is in Docs/agent-reference/distribution-and-packaging.md ("Release manifest").
#
# It refuses, before writing anything, a tree whose version is inconsistent: NFKInferKit.releaseVersion
# and the podspec's s.version must agree, a v* tag on HEAD must be v<version>, and the version must be
# newer than every other vX.Y.Z tag (inferkit_check_release_version in inferkit-version.sh).
#
# Requires: Xcode command line tools, python3.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/xcframework-build"
CHECKOUTS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2;;
        --checkouts) CHECKOUTS="$2"; shift 2;;
        *) echo "unknown argument: $1"; exit 2;;
    esac
done

# shellcheck source=inferkit-version.sh
source "$ROOT/Tools/xcframework/inferkit-version.sh"
VERSION="$(inferkit_version "$ROOT")"
API_VERSION="$(inferkit_api_version "$ROOT")"
[ -n "$VERSION" ] && [ -n "$API_VERSION" ] || { echo "cannot read the version or API version from $ROOT"; exit 1; }
inferkit_check_release_version "$ROOT" || { echo "no release written"; exit 1; }

DYNAMIC="$OUTPUT/InferKitMLXDynamic.xcframework"
STATIC="$OUTPUT/InferKitMLX.xcframework"
RELEASE="$OUTPUT/release"
[ -d "$DYNAMIC" ] || { echo "no InferKitMLXDynamic.xcframework in $OUTPUT; build it with build-mlx.sh"; exit 1; }
if [ -z "$CHECKOUTS" ] && [ -d "$ROOT/InferKitMLX/.build/checkouts" ]; then
    CHECKOUTS="$ROOT/InferKitMLX/.build/checkouts"
fi

rm -rf "$RELEASE"
mkdir -p "$RELEASE"

# ditto keeps the framework's symbolic links, which a zip made any other way can flatten.
for slice in "$DYNAMIC"/macos-*/; do
    [ -d "$slice/InferKitMLX.framework" ] || continue
    ditto -c -k --sequesterRsrc --keepParent "$slice/InferKitMLX.framework" \
        "$RELEASE/InferKitMLX-$VERSION-$(basename "$slice").zip"
done
ditto -c -k --sequesterRsrc --keepParent "$DYNAMIC" "$RELEASE/InferKitMLXDynamic.xcframework.zip"
if [ -d "$STATIC" ]; then
    ditto -c -k --sequesterRsrc --keepParent "$STATIC" "$RELEASE/InferKitMLX.xcframework.zip"
fi

COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "")"
LOCAL_CHANGES=false
if inferkit_tree_has_local_changes "$ROOT"; then
    LOCAL_CHANGES=true
fi

python3 -I - "$RELEASE" "$DYNAMIC" "$VERSION" "$API_VERSION" "$COMMIT" "$LOCAL_CHANGES" \
    "$ROOT/InferKitMLX/Package.resolved" "$CHECKOUTS" <<'PYTHON'
import hashlib, json, os, plistlib, re, sys

release, dynamic, version, api_version, commit, local_changes, resolved_path, checkouts = sys.argv[1:9]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

platform_names = {("macos", None): "macOS", ("ios", None): "iOS", ("ios", "simulator"): "iOS Simulator"}
minimums, slices = {}, {}
with open(os.path.join(dynamic, "Info.plist"), "rb") as handle:
    libraries = plistlib.load(handle)["AvailableLibraries"]
for library in libraries:
    identifier = library["LibraryIdentifier"]
    framework = os.path.join(dynamic, identifier, library["LibraryPath"])
    info = os.path.join(framework, "Resources", "Info.plist")
    if not os.path.exists(info):
        info = os.path.join(framework, "Info.plist")
    with open(info, "rb") as handle:
        minimum = plistlib.load(handle)["MinimumOSVersion"]
    platform = platform_names[(library["SupportedPlatform"], library.get("SupportedPlatformVariant"))]
    slices[identifier] = (platform, minimum)
    if platform != "iOS Simulator":
        minimums[platform] = minimum

mlx_swift = {"repository": None, "revision": None, "version": None, "mlxCoreVersion": None}
with open(resolved_path) as handle:
    for pin in json.load(handle)["pins"]:
        if pin["identity"] == "mlx-swift":
            mlx_swift["repository"] = pin["location"]
            mlx_swift["revision"] = pin["state"].get("revision")
            mlx_swift["version"] = pin["state"].get("version")
version_header = os.path.join(checkouts, "mlx-swift/Source/Cmlx/mlx/mlx/version.h") if checkouts else ""
if version_header and os.path.exists(version_header):
    text = open(version_header).read()
    parts = [re.search(r"#define MLX_VERSION_%s (\d+)" % name, text) for name in ("MAJOR", "MINOR", "PATCH")]
    if all(parts):
        mlx_swift["mlxCoreVersion"] = ".".join(part.group(1) for part in parts)

artifacts = []
for name in sorted(os.listdir(release)):
    path = os.path.join(release, name)
    entry = {"file": name}
    if name.startswith("InferKitMLX-"):
        identifier = name[len("InferKitMLX-%s-" % version):-len(".zip")]
        platform, minimum = slices[identifier]
        entry.update(kind="framework", platform=platform, libraryIdentifier=identifier,
                     minimumSystemVersion=minimum)
    else:
        entry.update(kind="xcframework", linkage="dynamic" if "Dynamic" in name else "static")
    entry.update(bytes=os.path.getsize(path), sha256=sha256(path))
    artifacts.append(entry)

manifest = {
    "kind": "inferkit.release",
    "schemaVersion": 1,
    "product": "InferKitMLX",
    "version": version,
    "apiVersion": api_version,
    "minimumSystemVersions": minimums,
    "commit": commit or None,
    "commitHasLocalChanges": local_changes == "true",
    "mlxSwift": mlx_swift,
    "artifacts": artifacts,
    "signature": "",
}
text = json.dumps(manifest, indent=2, ensure_ascii=False)
# The signing step finds this exact last line, so it must stay last and on its own line.
assert text.endswith('\n  "signature": ""\n}')
with open(os.path.join(release, "release.json"), "w") as handle:
    handle.write(text + "\n")
PYTHON

echo "    release.json: InferKitMLX $VERSION, API $API_VERSION"
for asset in "$RELEASE"/*.zip; do
    echo "    $(basename "$asset")  $(du -h "$asset" | awk '{print $1}')"
done
