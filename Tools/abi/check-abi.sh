#!/bin/bash
#
# Checks that a built InferKitMLX.framework keeps the binary interface of a baseline release: a client
# compiled against the baseline's headers loads the new framework and finds every class, selector,
# protocol conformance, constant, and function the baseline declares. The expected interface is read
# from the baseline's public headers, so no list is maintained by hand.
#
#   Tools/abi/check-abi.sh --baseline current                  # self-check: the candidate's own headers
#   Tools/abi/check-abi.sh --baseline path/to/InferKitMLX.framework
#   Tools/abi/check-abi.sh --baseline v0.4.0                   # a git ref, built into a framework first
#   ... [--candidate path/to/InferKitMLX.framework] [--allow-unbumped] [--verbose] [--keep DIR]
#
# The candidate defaults to the macOS slice build-mlx.sh writes to xcframework-build/. Building one:
#   Tools/xcframework/build-mlx.sh --variant dynamic --slices macos --no-swift-interfaces
#
# What it checks, in order:
#
#   1. The surfaces. Tools/abi/abi-surface.py reads both sets of headers through clang and compares
#      them. A removal or a changed type, selector, value, or layout is incompatible and needs a new
#      API major version; an addition needs a new minor version. The API versions both sets of headers
#      state must follow that rule. `--allow-unbumped` reports a miss without failing, for a baseline
#      that is no tagged release.
#   2. The binary. A client compiled against the baseline headers loads the candidate with dlopen and
#      looks up each baseline class by its runtime name, its superclass, each protocol it adopts, and
#      each class and instance selector, property accessors included, then each exported constant and
#      function with dlsym. Skipped when the API major versions differ, since a new major version
#      promises nothing to the old one.
#   3. The constant values. With a baseline framework at hand, the client reads every string and
#      numeric constant from both frameworks, and a changed value fails: a client may have stored it.
#
# The first real baseline is the first tagged release. Until one exists, `--baseline current` checks
# that the framework implements what its own headers declare.
#
# A git ref is exported with `git archive` into a temporary directory and built with that tree's own
# build-mlx.sh, which takes about seven minutes and needs the mlx-swift checkout to resolve.
#
# Requires: Xcode command line tools, python3, an Apple Silicon Mac.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SURFACE="$ROOT/Tools/abi/abi-surface.py"
TARGET=arm64-apple-macos14.0
BASELINE=""
CANDIDATE=""
ALLOW_UNBUMPED=()
VERBOSE=()
KEEP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --baseline) BASELINE="$2"; shift 2;;
        --candidate) CANDIDATE="$2"; shift 2;;
        --allow-unbumped) ALLOW_UNBUMPED=(--allow-unbumped); shift;;
        --verbose) VERBOSE=(--verbose); shift;;
        --keep) KEEP="$2"; shift 2;;
        *) echo "unknown argument: $1"; exit 2;;
    esac
done
[ -n "$BASELINE" ] || { echo "--baseline takes current, a framework path, or a git ref"; exit 2; }

if [ -n "$KEEP" ]; then
    WORK="$KEEP"
    mkdir -p "$WORK"
else
    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
fi

if [ -z "$CANDIDATE" ]; then
    CANDIDATE="$(find "$ROOT/xcframework-build/InferKitMLXDynamic.xcframework" -maxdepth 2 \
        -type d -path '*/macos-*/InferKitMLX.framework' 2>/dev/null | head -1 || true)"
fi
[ -n "$CANDIDATE" ] && [ -e "$CANDIDATE/InferKitMLX" ] \
    || { echo "no candidate framework; build one or pass --candidate"; exit 2; }

# The headers a framework declares: the core's in Headers/InferKit, and InferKitMLX's generated
# Objective-C header in Headers. Releases before the InferKitMLXObjC module kept the same two paths.
header_root() {                         # header_root <framework>
    [ -f "$1/Headers/InferKit/InferKit.h" ] || { echo "no core headers in $1/Headers/InferKit" >&2; return 1; }
    echo "$1/Headers"
}

BASELINE_FRAMEWORK=""
case "$BASELINE" in
    current)
        BASELINE_FRAMEWORK="$CANDIDATE"
        echo "==> baseline: the candidate's own headers (self-check)"
        ;;
    *.framework|*.framework/)
        BASELINE_FRAMEWORK="${BASELINE%/}"
        [ -e "$BASELINE_FRAMEWORK/InferKitMLX" ] || { echo "no framework binary in $BASELINE_FRAMEWORK"; exit 2; }
        echo "==> baseline: $BASELINE_FRAMEWORK"
        ;;
    *)
        commit="$(git -C "$ROOT" rev-parse --verify --quiet "$BASELINE^{commit}" || true)"
        [ -n "$commit" ] || { echo "'$BASELINE' is neither current, a framework, nor a git ref"; exit 2; }
        echo "==> baseline: $BASELINE ($commit), building its framework (about seven minutes)"
        # SwiftPM resolves InferKitMLX's `../` dependency by the folder name, which must be InferKit.
        source_tree="$WORK/baseline-source/InferKit"
        mkdir -p "$source_tree"
        git -C "$ROOT" archive "$commit" | tar -x -C "$source_tree"
        flags=(--variant dynamic --slices macos --no-swift-interfaces --output "$WORK/baseline-build")
        if grep -q -- '--no-release' "$source_tree/Tools/xcframework/build-mlx.sh"; then
            flags+=(--no-release)
        fi
        nice -n 10 "$source_tree/Tools/xcframework/build-mlx.sh" "${flags[@]}" > "$WORK/baseline-build.log" 2>&1 \
            || { tail -20 "$WORK/baseline-build.log"; echo "the baseline did not build"; exit 1; }
        BASELINE_FRAMEWORK="$(find "$WORK/baseline-build/InferKitMLXDynamic.xcframework" -maxdepth 2 \
            -type d -path '*/macos-*/InferKitMLX.framework' | head -1)"
        ;;
esac

echo "==> reading the surfaces"
python3 "$SURFACE" surface --headers "$(header_root "$BASELINE_FRAMEWORK")" \
    --objc-header "$BASELINE_FRAMEWORK/Headers/InferKitMLX-Swift.h" --output "$WORK/baseline.json" \
    | sed 's/^    /    baseline: /'
python3 "$SURFACE" surface --headers "$(header_root "$CANDIDATE")" \
    --objc-header "$CANDIDATE/Headers/InferKitMLX-Swift.h" --output "$WORK/candidate.json" \
    | sed 's/^    /    candidate: /'

status=0
echo "==> comparing the surfaces"
python3 "$SURFACE" diff "$WORK/baseline.json" "$WORK/candidate.json" ${ALLOW_UNBUMPED[@]+"${ALLOW_UNBUMPED[@]}"} ${VERBOSE[@]+"${VERBOSE[@]}"} \
    || status=1

baseline_major="$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))["apiVersion"]; print(v.split(".")[0] if v else "")' "$WORK/baseline.json")"
candidate_major="$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))["apiVersion"]; print(v.split(".")[0] if v else "")' "$WORK/candidate.json")"
if [ -n "$baseline_major" ] && [ -n "$candidate_major" ] && [ "$baseline_major" != "$candidate_major" ]; then
    echo "==> the API major version changed ($baseline_major to $candidate_major): no binary check"
    exit "$status"
fi

echo "==> building a client against the baseline headers"
python3 "$SURFACE" headers --headers "$(header_root "$BASELINE_FRAMEWORK")" \
    --objc-header "$BASELINE_FRAMEWORK/Headers/InferKitMLX-Swift.h" --output "$WORK/client-headers"
python3 "$SURFACE" client "$WORK/baseline.json" --objc-header --output "$WORK/client.m"
xcrun clang -fobjc-arc -fmodules -fmodules-cache-path="$WORK/module-cache" -target "$TARGET" -w \
    -I "$WORK/client-headers" "$WORK/client.m" -framework Foundation -o "$WORK/client" \
    > "$WORK/client.log" 2>&1 || { head -20 "$WORK/client.log"; echo "    FAILED: the client did not compile"; exit 1; }

echo "==> loading the candidate"
"$WORK/client" "$CANDIDATE/InferKitMLX" "$WORK/baseline.json" "$WORK/candidate-values.json" || status=1

if [ "$(cd "$BASELINE_FRAMEWORK" && pwd -P)" != "$(cd "$CANDIDATE" && pwd -P)" ]; then
    echo "==> comparing constant values with the baseline framework"
    "$WORK/client" "$BASELINE_FRAMEWORK/InferKitMLX" "$WORK/baseline.json" "$WORK/baseline-values.json" > /dev/null || true
    python3 - "$WORK/baseline-values.json" "$WORK/candidate-values.json" <<'PYTHON' || status=1
import json, sys
baseline, candidate = (json.load(open(path)) for path in sys.argv[1:3])
changed = [name for name in sorted(baseline) if name in candidate and baseline[name] != candidate[name]]
for name in changed:
    print("    incompatible: constant %s changed from %r to %r" % (name, baseline[name], candidate[name]))
print("    %d constant values compared, %d changed" % (len(baseline), len(changed)))
sys.exit(1 if changed else 0)
PYTHON
fi

if [ "$status" -eq 0 ]; then
    echo "==> the candidate keeps the baseline's binary interface"
else
    echo "==> FAILED"
fi
exit "$status"
