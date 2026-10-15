#!/bin/bash
#
# Sourced by build-mlx.sh, release.sh, and Tools/abi/check-abi.sh. Reads the release version and the API
# version from a source tree, so an artifact records the versions of the sources it was built from.
#
#   inferkit_version <tree>       # "0.4.0", from +[NFKInferKit releaseVersion] in Sources/InferKit/NFKInferKit.m
#   inferkit_api_version <tree>   # "1.0", from NFK_API_VERSION_MAJOR / _MINOR in NFKAPIVersion.h

inferkit_version() {
    awk '/^\+ \(NSString \*\)releaseVersion$/ { found = 1 }
         found && /return @"/ { match($0, /@"[^"]*"/); print substr($0, RSTART + 2, RLENGTH - 3); exit }' \
        "$1/Sources/InferKit/NFKInferKit.m"
}

inferkit_api_version() {
    local header="$1/Sources/InferKitC/include/InferKitC/NFKAPIVersion.h" major minor
    # A tree from before the InferKitC target kept the macros in NFKInferKit.h.
    [ -f "$header" ] || header="$1/Sources/InferKit/include/InferKit/NFKInferKit.h"
    major="$(awk '/^#define NFK_API_VERSION_MAJOR / { print $3 }' "$header")"
    minor="$(awk '/^#define NFK_API_VERSION_MINOR / { print $3 }' "$header")"
    [ -n "$major" ] && [ -n "$minor" ] && echo "$major.$minor"
}
