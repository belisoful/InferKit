<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Distribution and packaging

## Distribution

InferKit ships as source through two channels that reference the same files:

- **Swift Package Manager** — `Package.swift`. The target's public API is `Sources/InferKit/include/`,
  so `#import <InferKit/NFKFoo.h>` resolves the same way against SwiftPM and the built framework.
  Quoted sibling imports (`#import "NFKFoo.h"`) resolve through the target's `headerSearchPath`
  entries (`include/InferKit` and `.`).
- **CocoaPods** — `InferKit.podspec`, a source pod. `source_files` compiles the same `.h`/`.m`;
  `public_header_files` marks `include/InferKit/*.h` as the public surface. The MLX companion is not
  a pod (MLX distributes through SwiftPM only), though it does package as an XCFramework — see
  "Packaging InferKitMLX as an XCFramework".

Keep the two in sync: a new source file is picked up by the SwiftPM glob automatically; the podspec
globs the same paths, so no per-file edit is needed there either. New public headers go in
`Sources/InferKit/include/InferKit/` and the umbrella `InferKit.h`.

`Tools/` ships in no distribution — it is developer and validation tooling, not the library. The
SwiftPM targets name `Sources/InferKit` / `InferKitMLX/Sources` as their paths, the podspec globs only
`Sources/InferKit/**/*.{h,m}`, and the XCFramework release assets are built binaries; none of them
compile or carry anything under `Tools/`. A consumer resolving the SwiftPM package clones the whole
repository (so `Tools/` lands on disk), but nothing there is built into the products or the release.
The `~30 Tools/*-to-safetensors/convert.py` scripts are therefore not required to consume the
library: `NFKMLXTorchFormat` reads a raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` natively (see `NFKMLXTorchCheckpoint` in
`mlx-weights-and-formats.md`), so a consumer needs no Python. The converters stay for two reasons
that are not consumer-facing — they are the **byte oracle** `NFKMLXTorchParityTests` holds the native
reader to, and the **offline path** to a portable, pickle-free safetensors — and the Python
reference-parity tooling (`Tools/reference-parity`, `Tools/validation-assets`) is the ground truth
every Swift port is measured against, which is irreducibly Python (torch / transformers / diffusers).
`Tools/README.md` records this boundary.

The version lives in three places that must move together on release: `s.version` in
`InferKit.podspec`, the string `NFKInferKit.version` returns (`Sources/InferKit/NFKInferKit.m`), and
the `vX.Y.Z` git tag. A test asserts the shape of `NFKInferKit.version`, not its value, so a stale
constant does not fail CI — bumping it is a release step. SwiftPM takes its version from the tag, so
`Package.swift` carries none.

## Packaging InferKitMLX as an XCFramework

`Tools/xcframework/build-mlx.sh` emits **two** artifacts from one archive per slice (arm64; macOS,
iOS device, iOS simulator):

- `InferKitMLX.xcframework` — static: `libInferKitMLX.a` (every target's object merged with `libtool`,
  the core included), headers, a modulemap, and the `mlx-swift_Cmlx.bundle` that slice's consumer
  ships. 47 MB a slice, 140 MB in all.
- `InferKitMLXDynamic.xcframework` — dynamic: `InferKitMLX.framework`, Metal library inside. A slice is
  a 15.5 MB binary, 3.6 MB of shaders, and 12 MB of Swift module interfaces an Objective-C consumer
  never reads; 96 MB in all.
- `CoreHeaders/` (the core's headers beside a modulemap) for Objective-C consumers of the dynamic one.

Either way a consumer's binary grows by about 14 MB plus the shaders.

What made it possible. MLX needs `default.metallib`, which SwiftPM delivers as a
`mlx-swift_Cmlx.bundle` resource, and a bare library carries no resources. MLX's loader
(`mlx/backend/metal/device.cpp`) tries four places, and `current_binary_dir()` is `dladdr` on MLX's
own code, so it names whichever mach-O image MLX was linked into:

1. `<binary dir>/mlx.metallib` — the static route, beside the consumer's binary.
2. `<binary dir>/Resources/mlx.metallib`.
3. `mlx-swift_Cmlx.bundle` in the main or any loaded bundle — the static route inside an app bundle,
   and the layout Copy Bundle Resources produces.
4. `<binary dir>/Resources/default.metallib` — commented in the reference as "if SwiftPM wrapped as a
   dynamic framework", which is exactly the dynamic route.

A dynamic framework satisfies (4) by itself. A static consumer satisfies (3) from the bundle inside
their slice. All three paths are measured by `verify-mlx.sh`, which links a consumer and **runs a
model**: a binary that links can still fail at the first array evaluation, which is the failure this
packaging exists to avoid.

The static xcframework carries its own shaders, and that is delivery only. An xcframework is a
build-time container that never ships, and Xcode copies nothing out of a static one, so the consumer
still places the bundle in their own product — measured, not assumed: a `binaryTarget` consuming a
slice with the bundle inside builds and then throws `Failed to load the default metallib`, and copying
the bundle beside the binary makes the same probe pass. What co-location buys is that the shaders
cannot arrive separately from the library, at 1.7 MB compressed against the two forms an earlier layout
shipped in a second asset. SwiftPM accepts the extra file in a slice without complaint. The earlier
separation was also justified here by a claim that a `binaryTarget` zip may hold nothing beside the
xcframework; that claim was never tested — SwiftPM requires `https` for a `binaryTarget` URL, so it
cannot be checked against a local server — and the question is now moot, because the zip holds exactly
one `.xcframework` either way.

Neither variant is smaller. A consumer using one model links to 13.8 MB static against 14.2 MB
dynamic — dead-stripping buys about 3%, because `Cmlx.o` is one merged object and MLX's runtime is
densely interconnected. Of the ~18 MB a slice weighs, MLX's C++ is 72% of the code and its shaders are
3.6 MB; InferKitMLX and the core together are about 320 KB. So the choice is deployment mechanics:

- Static needs no embedding and no code signing, and vends both modules from one plain modulemap.
- Dynamic is one self-contained drop-in and is shared between several consumers, but clang refuses a
  non-framework module inside a framework, so its `InferKit` module needs a modulemap passed with
  `-fmodule-map-file`. An Objective-C consumer is better served by the static variant.

Linking recipes and the release matrix live in `Docs/installation.md`, one per consumer shape, each
verified by building and running it: Xcode static and dynamic, SwiftPM `binaryTarget` for both variants
(SwiftPM wires the static one's headers and modulemap automatically; the dynamic one still needs
`CoreHeaders` passed through `-fmodule-map-file`), plug-in bundles, and a consumer's own static library.
The artifacts are not committed — this repository is source-distributed, so a consumer resolving it
clones its history. Three compressed release assets, one per variant carrying every slice: core 0.7 MB,
static 28 MB, dynamic 19 MB. The core's `build.sh` cleans only its own artifact — both scripts share
`.xcframework-build/`, and an `rm -rf` of the whole directory once discarded a twenty-minute MLX build
beside a twenty-second core one.

`--variant static|dynamic|both`, `--slices macos,ios,iossim`, and `--no-swift-interfaces` trim the
output. `--slices` also trims the build: each slice compiles those 289 translation units again, so it
is the difference between about seven minutes and twenty. `--no-swift-interfaces` applies only to the
dynamic framework — the static library never carried interfaces, and is Objective-C-only by
construction. `verify-mlx.sh` reports what it skipped rather than failing when a variant or the
macOS slice is absent; an iOS binary cannot be executed on the host.

Nothing in `Package.swift` changes for distribution. One `xcodebuild archive` of the ordinary
`InferKitMLX` product yields each target's merged object; `libtool` makes the static library from them
and `clang -dynamiclib` makes the framework binary from the same library. A second `type: .dynamic`
product was tried and abandoned: two library products in one package install the same dependency
objects, which fails an iOS archive with "Multiple commands produce .../ArgumentParser.o" (macOS
tolerated it). `BUILD_LIBRARY_FOR_DISTRIBUTION=YES` succeeds across the whole graph, C++ interop
included.

The dylib links `-all_load` and **without `-dead_strip`**: `NFKDynamicBackend` resolves providers
through `NSClassFromString`, so a class with no static reference is still reachable. It costs about
1.3 MB against a dead-stripped link, which is the right trade for a binary whose discovery mechanism
is by name.
