#!/bin/bash
#
# Sourced by build-mlx.sh, release.sh, and Tools/abi/check-abi.sh. Reads the release version and the API
# version from a source tree, so an artifact records the versions of the sources it was built from.
#
#   inferkit_version <tree>                    # "0.4.0", from +[NFKInferKit releaseVersion] in NFKInferKit.m
#   inferkit_api_version <tree>                # "1.0", from NFK_API_VERSION_MAJOR / _MINOR in NFKAPIVersion.h
#   inferkit_podspec_version <tree>            # "0.4.0", from s.version in InferKit.podspec
#   inferkit_check_release_version <tree>      # prints why and returns 1 when the tree is not releasable
#   inferkit_tree_has_local_changes <tree>     # returns 0 when the tree differs from its commit
#
# A release version is X.Y.Z. inferkit_check_release_version refuses a tree when:
#   - s.version in InferKit.podspec differs from NFKInferKit.releaseVersion;
#   - the version is not X.Y.Z;
#   - the tree is not a git checkout;
#   - HEAD carries a v* tag other than v<version>;
#   - an existing vX.Y.Z tag on another commit is as new as the version or newer.
# A tag v<version> on HEAD is the release being built, so a rebuild of a tagged commit passes. The tags are
# the local ones: fetch the tags before a release build.

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

inferkit_podspec_version() {
    awk -F"'" '/^[[:space:]]*s\.version[[:space:]]*=/ { print $2; exit }' "$1/InferKit.podspec"
}

# Returns 0 when the X.Y.Z version $1 is newer than the X.Y.Z version $2.
inferkit_version_is_newer() {
    local newer older index
    IFS=. read -r -a newer <<< "$1"
    IFS=. read -r -a older <<< "$2"
    for index in 0 1 2; do
        if (( 10#${newer[index]} > 10#${older[index]} )); then return 0; fi
        if (( 10#${newer[index]} < 10#${older[index]} )); then return 1; fi
    done
    return 1
}

inferkit_check_release_version() {
    local tree="$1" version podspec head_tags tag
    version="$(inferkit_version "$tree")"
    podspec="$(inferkit_podspec_version "$tree")"
    if [ -z "$version" ]; then
        echo "cannot read NFKInferKit.releaseVersion from $tree"
        return 1
    fi
    if [ "$podspec" != "$version" ]; then
        echo "NFKInferKit.releaseVersion is $version but s.version in InferKit.podspec is ${podspec:-missing}; they move together"
        return 1
    fi
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "the release version $version is not X.Y.Z"
        return 1
    fi
    if ! git -C "$tree" rev-parse --verify --quiet HEAD > /dev/null; then
        echo "$tree is not a git checkout, and a release builds from a commit"
        return 1
    fi
    head_tags="$(git -C "$tree" tag --points-at HEAD --list 'v*')"
    for tag in $head_tags; do
        if [ "$tag" != "v$version" ]; then
            echo "HEAD carries the tag $tag, but the sources say $version"
            return 1
        fi
    done
    for tag in $(git -C "$tree" tag --list 'v*'); do
        if ! [[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then continue; fi
        if [ "$tag" = "v$version" ] && [ -n "$head_tags" ]; then continue; fi
        if ! inferkit_version_is_newer "$version" "${tag#v}"; then
            echo "the release version $version is not newer than the existing tag $tag; move the version in NFKInferKit.m and InferKit.podspec"
            return 1
        fi
    done
}

# Untracked files count because SwiftPM and the podspec glob them into the build; ignored files do not.
inferkit_tree_has_local_changes() {
    [ -n "$(git -C "$1" status --porcelain --untracked-files=all 2>/dev/null || true)" ]
}
