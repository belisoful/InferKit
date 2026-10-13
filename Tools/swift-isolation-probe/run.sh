#!/bin/sh
# Builds the core with SwiftPM, then compiles main.swift as Swift 6 with main-actor default isolation
# against it and runs it. Exit 0 means every probed block ran on InferKit's queues without an
# isolation trap; a trap exits with SIGTRAP (133). The output directory defaults to Local/.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-"$ROOT/Local/swift-isolation-probe"}
mkdir -p "$OUT/module"
cd "$ROOT"
swift build --product InferKit >/dev/null
BIN=$(swift build --show-bin-path)
printf 'module InferKit {\n\tumbrella header "%s"\n\texport *\n}\n' "$ROOT/Sources/InferKit/include/InferKit/InferKit.h" > "$OUT/module/module.modulemap"
swiftc -swift-version 6 -default-isolation MainActor \
	-I "$OUT/module" -Xcc -I"$ROOT/Sources/InferKit/include" \
	"$ROOT/Tools/swift-isolation-probe/main.swift" "$BIN/libInferKit.a" \
	-framework CoreML -framework Vision -framework VideoToolbox -framework Speech -framework SoundAnalysis \
	-framework NaturalLanguage -framework CoreVideo -framework CoreGraphics -framework ImageIO \
	-framework AVFoundation -framework Metal -framework IOSurface -framework Network -framework Security \
	-framework IOKit -o "$OUT/probe"
"$OUT/probe"
