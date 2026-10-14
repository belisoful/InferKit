# Installation


### Swift Package Manager

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/belisoful/InferKit.git", from: "0.3.1")
```

then add `"InferKit"` to your target's dependencies. In Xcode, use File ▸ Add Package Dependencies
and enter the repository URL.

#### Adding a companion (InferKitMLX, InferKitFoundationModels, InferKitAppleSwift)

The companions are **separate packages that live in subdirectories of this repository**, each with its
own `Package.swift`. Nothing in the core references them — the dependency points the other way, so the
core's manifest never mentions MLX and opening the core in Xcode shows no MLX in its package graph.
That is what keeps the core's platform floor at macOS 11 / iOS 14 / tvOS 14 with no third-party
dependencies, while MLX needs Apple Silicon and macOS 14 / iOS 17.

Swift Package Manager cannot address a package that lives in a subdirectory of a remote repository,
so a companion is consumed **by path** from a local checkout (a clone, or a submodule):

```swift
dependencies: [
    // The companion declares the core as its own dependency, so both land in the graph.
    .package(path: "path/to/InferKit/InferKitMLX"),
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "InferKitMLX", package: "InferKitMLX"),
        .product(name: "InferKit", package: "InferKit"),
    ]),
]
```

In Xcode, drag the `InferKitMLX` folder into your project to add it as a local package, or open
`InferKitMLX/Package.swift` directly to work on the companion itself — that window is where MLX and
its dependencies appear.

Linking a companion is also what activates the core's optional capabilities: the core resolves a
provider class by name through `NSClassFromString`, so it never references MLX symbols and the feature
is simply unavailable when the companion is absent. See
[Dynamic backend discovery](companions.md#dynamic-backend-discovery-optional-engines).

### CocoaPods

Add to your `Podfile`:

```ruby
pod 'InferKit'
```

then run `pod install`. The `InferKitMLX` companion is not a pod, because MLX distributes through
SwiftPM. It does package as a binary — see below.

### Binary distribution (XCFramework)

The core builds into an XCFramework with `Tools/xcframework/build.sh`. The MLX companion has its own
script, which emits both linkage styles from one build:

```bash
Tools/xcframework/build-mlx.sh --verify
```

Three flags trim what gets built and written:

- `--variant static | dynamic | both` — stop producing the one you do not deploy. The compile is
  shared, so this costs no build time and halves the output.
- `--slices macos,ios,iossim` — each slice is a full compile of MLX's 289 translation units, so this
  is the difference between roughly seven minutes and twenty, and it scales the output the same way.
- `--no-swift-interfaces` — drops the Swift module interfaces, and the C modules they import, from
  the **dynamic** framework, about 13 MB a slice. The framework is then importable from Objective-C
  only. The static library never carried interfaces, so the flag does nothing for it.

`--verify` links a consumer against each artifact and runs a model on the GPU, and builds and runs a
Swift consumer against the dynamic framework's interfaces. A binary that links can still fail to find
its Metal library, so linking alone is not the check. A variant or macOS slice that was not built is
reported as skipped. A dynamic framework that carries the C modules but no `InferKitMLX.swiftmodule`
fails.

## Linking the binary artifacts

### Which artifact

| you are building | take | why |
| --- | --- | --- |
| an app, a framework, or a plug-in in Objective-C | `InferKitMLX.xcframework` (static) | nothing to embed or sign, and one modulemap vends both `InferKit` and `InferKitMLX` |
| an app that wants one self-contained drop-in | `InferKitMLXDynamic.xcframework` | the Metal library rides inside it |
| several plug-ins in one host | `InferKitMLXDynamic.xcframework` | they share one copy instead of one each |
| anything in Swift | `InferKitMLXDynamic.xcframework` | it carries the Swift module interfaces of InferKitMLX and MLX, and the C modules beneath them; the static library carries none |
| the core alone, no MLX | `InferKit.xcframework` | 4.7 MB, no MLX runtime at all |

Both MLX artifacts are arm64 only, with macOS, iOS device, and iOS simulator slices. **Do not add
`InferKit.xcframework` or the InferKit source package beside an MLX artifact**: the core travels inside
it, and a second copy gives you two of every `NFK` class, so an object your code makes and one a backend
makes fail `isKindOfClass:` against each other.

### The Metal library rule

MLX loads `default.metallib` at the first array evaluation, not at link time, so a mistake here builds
and links cleanly and then throws `Failed to load the default metallib` on first inference.
`NFKMLXGPU.metalLibraryURL` reports the library MLX will load, found the way its loader looks, or nil
when none is there, so a launch can check before the first evaluation.

The static xcframework carries the library inside each slice, so it arrives with the artifact and there
is one thing to download. Placing it is still yours to do: an xcframework is a build-time container that
never ships, and Xcode copies nothing out of a static one.

**It is required even if you never touch the GPU.** Selecting the CPU device does not avoid it: MLX
initializes its scheduler when the first stream is requested, and on Apple platforms that constructs the
Metal device, so `mlx_default_cpu_stream_new` itself throws `Failed to load the default metallib`.
Measured both ways — with the library absent, a probe pinned to `Device(cpu, 0)` aborts before its first
line of output; with it present, the same probe runs elementwise arithmetic, an `MLXNN` convolution
stack, and a full backend inference on the CPU.

- **Dynamic**: nothing to do. The library is inside `InferKitMLX.framework/Resources/`.
- **Static, app or plug-in**: drag `InferKitMLX.xcframework/<slice>/mlx-swift_Cmlx.bundle` into **Copy
  Bundle Resources**, taking the slice for the platform that target builds for.
- **Static, command-line tool**: put that same bundle beside the executable. A command-line binary's
  main bundle is its own directory, which is where MLX looks.

### An Xcode app or framework, static

1. Drag `InferKitMLX.xcframework` into the target's **Frameworks, Libraries, and Embedded Content**,
   set to **Do Not Embed** — a static library is linked in, not embedded.
2. Add the slice's `Headers` directory to **Header Search Paths**, and to **Other C Flags**:
   `-fmodule-map-file=$(SRCROOT)/path/to/InferKitMLX.xcframework/<slice>/Headers/module.modulemap`
3. Add `mlx-swift_Cmlx.bundle` to **Copy Bundle Resources**, per the rule above.
4. `@import InferKit;` and `@import InferKitMLX;`.

The linker needs `Metal`, `Accelerate`, `CoreGraphics`, and `libc++`; Xcode adds them from the
xcframework's own link metadata, and a hand-rolled link command must name them.

### An Xcode app or framework, dynamic

1. Drag `InferKitMLXDynamic.xcframework` in, set to **Embed & Sign**.
2. For Objective-C, add `CoreHeaders` to **Header Search Paths** and
   `-fmodule-map-file=$(SRCROOT)/path/to/CoreHeaders/module.modulemap` to **Other C Flags** — clang
   refuses a non-framework module inside a framework, so the core's module travels beside it.
3. For Swift, add `$(BUILT_PRODUCTS_DIR)/InferKitMLX.framework/Modules` and
   `$(SRCROOT)/path/to/CoreHeaders` to **Import Paths**. Xcode copies the slice's framework into the
   build products, so one path serves every platform.
4. `@import InferKit;` and `@import InferKitMLX;`, or in Swift `import InferKitMLX`, and `import MLX`
   to use MLX's types directly.

The framework's `Modules` directory holds the Swift interfaces of InferKitMLX and of every module it
imports (MLX, MLXNN, MLXFast, MLXRandom, MLXFFT, MLXOptimizers, swift-numerics), plus the two C modules
those interfaces import, `Cmlx` and `_NumericsShims`. A Swift consumer reaches all of them through that
one import path, and the core's module through `CoreHeaders`. From the command line:

```bash
swiftc -F InferKitMLXDynamic.xcframework/macos-arm64 \
       -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Modules \
       -I CoreHeaders \
       App.swift -framework InferKitMLX
```

Swift needs no `-fmodule-map-file`: it finds `CoreHeaders/module.modulemap` through the import path.
The framework must stay a single unit. Removing its Swift interfaces leaves only the generated
Objective-C header, whose classes Swift then imports with no members.

### Swift Package Manager, either variant

A `binaryTarget` consumes the xcframework directly. SwiftPM wires the static variant's headers and
modulemap for you, so nothing else is needed:

```swift
.binaryTarget(name: "InferKitMLX", path: "InferKitMLX.xcframework"),
.executableTarget(name: "App", dependencies: ["InferKitMLX"],
                  linkerSettings: [.linkedFramework("Metal"), .linkedFramework("Accelerate"),
                                   .linkedFramework("CoreGraphics")]),
```

The dynamic variant needs the core's module passed in, exactly as in Xcode:

```swift
.binaryTarget(name: "InferKitMLX", path: "InferKitMLXDynamic.xcframework"),
.executableTarget(name: "App", dependencies: ["InferKitMLX"],
                  cSettings: [.unsafeFlags(["-I", "CoreHeaders",
                                            "-fmodule-map-file=CoreHeaders/module.modulemap"])]),
```

A Swift target also needs the framework's `Modules` directory. SwiftPM does not resolve a relative
path in `unsafeFlags` against the package root, so the manifest builds absolute paths from its own
location. The slice path names one platform:

```swift
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path   // import Foundation
let slice = "\(root)/InferKitMLXDynamic.xcframework/macos-arm64"

.binaryTarget(name: "InferKitMLX", path: "InferKitMLXDynamic.xcframework"),
.executableTarget(name: "App", dependencies: ["InferKitMLX"],
                  swiftSettings: [.unsafeFlags(["-I", "\(slice)/InferKitMLX.framework/Modules",
                                                "-I", "\(root)/CoreHeaders"])],
                  linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"])]),
```

The `rpath` lets an executable target load the framework SwiftPM copies beside it.

SwiftPM does not copy the Metal bundle either. For an executable target, copy it beside the built
binary; for an app, it goes in the bundle. The rule above applies unchanged.

To consume a release asset rather than a local file, swap `path:` for `url:` and `checksum:`, which
`swift package compute-checksum InferKitMLX.xcframework.zip` prints.

### A plug-in bundle (FxPlug, an app extension)

Take the static variant. A plug-in is a bundle, so a dynamic framework has to be embedded inside it and
signed, and the host may already load another copy; the static library sidesteps both. Put the slice's
`mlx-swift_Cmlx.bundle` in the plug-in's own **Copy Bundle Resources** — MLX searches every loaded
bundle, not only the host's.

### Your own static library

A static library cannot link another one. Declare the dependency and let the final app or plug-in link
`libInferKitMLX.a` and ship the Metal library; your `.a` only needs the headers to compile against.

### CocoaPods

The core is a source pod (`pod 'InferKit'`). For the MLX artifacts, a podspec consuming the release
asset uses `vendored_frameworks` with an `http:` source, plus `resources` for the Metal bundle when you
take the static variant.

## Release assets

The build output is not committed — `xcframework-build/` is gitignored, and this repository is
distributed as source through SwiftPM and CocoaPods, so a consumer resolving the package clones its
history. A binary added per release would compound there permanently.

Three compressed assets, produced with
`ditto -c -k --sequesterRsrc --keepParent <artifact> <artifact>.zip`:

| asset | contents | size |
| --- | --- | --- |
| `InferKit.xcframework.zip` | the core alone, no MLX | 1.3 MB |
| `InferKitMLX.xcframework.zip` | static, three slices, Metal library inside each | 38 MB |
| `InferKitMLXDynamic.xcframework.zip` | dynamic, three slices | 25 MB |

**One asset per variant, carrying every slice**, rather than splitting by platform. An iOS developer
always needs two slices (device and simulator), so a platform split is two-way rather than three and
saves about 17 MB of a 27 MB download; and fewer assets means fewer checksums and no way for a consumer
to mix versions across separate downloads.

The static `.a` compresses about five to one, because a static archive is mostly zero-padding and
repeated symbol tables. **The Metal library ships inside the static xcframework rather than beside it.**
A static consumer who does not place it gets `Failed to load the default metallib` at the first
inference rather than a link error, and a separate download is one more thing to arrive without. Being
inside costs 1.7 MB compressed against the two forms an earlier layout shipped separately.

Publish `swift package compute-checksum` output for each zip alongside it, so a `binaryTarget` can
name a `url:` and `checksum:`.
