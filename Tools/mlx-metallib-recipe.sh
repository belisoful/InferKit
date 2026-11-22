# shellcheck shell=bash
#
# SwiftPM's recipe for MLX's Metal library, sourced by Tools/mlx-metallib.sh and
# Tools/background-probe/build.sh.
#
# Xcode 27's SwiftPM compiles mlx-swift's kernels as a package resource with Xcode's Metal defaults,
# which include fast math (`MTL_FAST_MATH`). mlx's own CMake build and mlx-swift's Xcode project
# pass `-fno-fast-math` instead. The two libraries load the same kernels with different arithmetic,
# and the parity figures recorded in this repository come from SwiftPM's. These functions repeat
# SwiftPM's compile and link command lines, read from its build description
# (`.build/out/Intermediates.noindex/XCBuildData/*.xcbuilddata/manifest.json`, the
# `CompileMetalFile` and `MetalLink` commands of `mlx-swift_Cmlx`). Measured 2026-11-21 against
# mlx-swift 90194196 and Metal 32023.921: the result is byte-identical to SwiftPM's
# `default.metallib`.

MLX_METAL_COMPILE_FLAGS=(-fmetal-math-mode=fast -fmetal-math-fp32-functions=fast)

# Prints the macOS deployment target mlx-swift's Package.swift declares, which SwiftPM compiles its
# kernels for.
mlx_metal_deployment_target() {
    local manifest="$1/Package.swift" target
    target="$(sed -n -E \
        -e 's/.*\.macOS\("([0-9.]+)"\).*/\1/p' \
        -e 's/.*\.macOS\(\.v([0-9]+)\).*/\1.0/p' "$manifest" | head -1)"
    [ -n "$target" ] || { echo "no macOS platform in $manifest" >&2; return 1; }
    echo "$target"
}

# Prints the Metal compiler's version line, which identifies the toolchain a library came from.
mlx_metal_compiler_version() {
    xcrun -sdk macosx metal --version 2>&1 | head -1
}

# Prints one line per input that determines the library's bytes: the recipe, the compiler, the
# deployment target, and each kernel source's hash.
mlx_metallib_stamp() {
    local kernels="$1" deployment="$2" source
    echo "flags ${MLX_METAL_COMPILE_FLAGS[*]}"
    echo "compiler $(mlx_metal_compiler_version)"
    echo "sdk $(xcrun -sdk macosx --show-sdk-version)"
    echo "target air64-apple-macos$deployment"
    while IFS= read -r -d '' source; do
        echo "source ${source#"$kernels"/} $(shasum -a 256 "$source" | cut -d' ' -f1)"
    done < <(find "$kernels" -type f \( -name '*.metal' -o -name '*.h' \) -print0 | sort -z)
}

# Compiles every `.metal` file under $1 for macOS $2 into $3/*.air and links them into $4.
# Compiler diagnostics go to $3/compile.log; mlx's headers raise C++17 extension warnings that
# SwiftPM also leaves on.
mlx_build_metallib() {
    local kernels="$1" deployment="$2" work="$3" output="$4"
    local sdk source name log="$3/compile.log"
    local -a airs=() names=()
    sdk="$(xcrun -sdk macosx --show-sdk-path)"
    mkdir -p "$work"
    : > "$log"
    while IFS= read -r -d '' source; do
        names+=("$(basename "$source" .metal)")
    done < <(find "$kernels" -name '*.metal' -print0)
    [ "${#names[@]}" -gt 0 ] || { echo "no Metal sources under $kernels" >&2; return 1; }
    # SwiftPM names each AIR file after its source's basename and links them in basename order, so a
    # repeated basename would overwrite one kernel with another.
    if [ -n "$(printf '%s\n' "${names[@]}" | sort | uniq -d)" ]; then
        echo "two Metal sources under $kernels share a basename" >&2
        return 1
    fi
    while IFS= read -r name; do
        source="$(find "$kernels" -name "$name.metal" -print -quit)"
        xcrun -sdk macosx metal -c -target "air64-apple-macos$deployment" -isysroot "$sdk" \
            "${MLX_METAL_COMPILE_FLAGS[@]}" -o "$work/$name.air" "$source" >> "$log" 2>&1 \
            || { tail -20 "$log" >&2; return 1; }
        airs+=("$work/$name.air")
    done < <(printf '%s\n' "${names[@]}" | LC_ALL=C sort)
    xcrun -sdk macosx metal -target "air64-apple-macos$deployment" -o "$output" "${airs[@]}" \
        >> "$log" 2>&1 || { tail -20 "$log" >&2; return 1; }
}
