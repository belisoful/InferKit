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

The dynamic variant also writes `release/`: each macOS slice's framework zipped, the xcframework
zips, and `release.json`, the manifest an app reads to install and update releases (see
[Side-by-side releases](#side-by-side-releases)). `--no-release` skips it.

Every artifact carries the pure-C headers beside the core's: `NFKInferKitC.h` for a C, C++, or Rust
program that calls InferKit, and `NFKPluginABI.h` for a plugin library. The [C interface](c-api.md)
gives the compiler and linker lines.

Three flags trim what gets built and written:

- `--variant static | dynamic | both` — stop producing the one you do not deploy. The compile is
  shared, so this costs no build time and halves the output.
- `--slices macos,ios,iossim` — each slice is a full compile of MLX's 289 translation units, so this
  is the difference between roughly seven minutes and twenty, and it scales the output the same way.
- `--no-swift-interfaces` — drops the Swift module interfaces, and the C modules they import, from
  the **dynamic** framework, about 13 MB a slice. Objective-C then imports `InferKitMLX` and Swift
  imports `InferKitMLXObjC`. The static library never carried interfaces, so the flag does nothing
  for it.

`--verify` links a consumer against each artifact and runs a model on the GPU, and builds and runs
two Swift consumers against the dynamic framework: one through `InferKitMLXObjC`, beside a decoy
`mlx-swift_Cmlx.bundle` it must ignore, and one through the Swift interfaces. A binary that links can
still fail to find its Metal library, so linking alone is not the check. A variant or macOS slice
that was not built is reported as skipped. A dynamic framework that carries the C modules but no
`InferKitMLX.swiftmodule` fails.

## Linking the binary artifacts

### Which artifact

| you are building | take | why |
| --- | --- | --- |
| an app, a framework, or a plug-in in Objective-C | `InferKitMLX.xcframework` (static) | nothing to embed or sign, and one modulemap vends both `InferKit` and `InferKitMLX` |
| an app that wants one self-contained drop-in | `InferKitMLXDynamic.xcframework` | the Metal library and the core's module ride inside it |
| a macOS app that installs and updates InferKit apart from itself | `InferKitMLX-<version>-macos-arm64.zip` | one framework per release folder; see [Side-by-side releases](#side-by-side-releases) |
| several plug-ins in one host | `InferKitMLXDynamic.xcframework` | they share one copy instead of one each |
| anything in Swift | `InferKitMLXDynamic.xcframework` | it carries the `InferKitMLXObjC` module and the Swift module interfaces of InferKitMLX and MLX; the static library carries neither |
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

- **Dynamic**: nothing to do. The library is `InferKitMLX.framework/Resources/mlx.metallib`, and MLX
  finds it before any `mlx-swift_Cmlx.bundle` your app ships, so the framework always runs the shaders
  it was built with.
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
2. For Objective-C, add `$(BUILT_PRODUCTS_DIR)/InferKitMLX.framework/Headers` to **Header Search
   Paths**. The core's module, `Headers/InferKit`, is found there.
3. For Swift, add `$(BUILT_PRODUCTS_DIR)/InferKitMLX.framework/Headers` to **Import Paths**, and
   `$(BUILT_PRODUCTS_DIR)/InferKitMLX.framework/Modules` too to import the Swift interfaces. Xcode
   copies the slice's framework into the build products, so one path serves every platform.
4. `@import InferKit;` and `@import InferKitMLX;` in Objective-C. In Swift, `import InferKitMLXObjC`,
   or `import InferKitMLX` and `import MLX` to use the Swift interfaces (see below).

From the command line:

```bash
# Objective-C
clang -fobjc-arc -fmodules -F InferKitMLXDynamic.xcframework/macos-arm64 \
      -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Headers \
      App.m -framework InferKitMLX -framework Foundation

# Swift, the Objective-C interface: no -framework flag, the module links it
swiftc -F InferKitMLXDynamic.xcframework/macos-arm64 \
       -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Headers \
       App.swift

# Swift, the Swift interfaces
swiftc -F InferKitMLXDynamic.xcframework/macos-arm64 \
       -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Modules \
       -I InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework/Headers \
       App.swift -framework InferKitMLX
```

The core's module comes from the framework's `Headers` directory. An Objective-C target does not
pass `Modules`: clang also reads a search path's own `module.modulemap`, and `Modules/module.modulemap`
is the framework's, which clang then refuses from the wrong directory. Do not add a second `InferKit`
module, such as the `CoreHeaders` folder earlier builds shipped, on another search path: two maps for
one module fail with "redefinition of module".

### Swift: InferKitMLXObjC or InferKitMLX

The framework offers Swift two modules. Import one of them, never both: each declares the same
classes.

- **`import InferKitMLXObjC`** — the Objective-C interface of InferKitMLX, with the core re-exported.
  Calls go through Objective-C dispatch, which the [API version](#api-version) covers, so a client
  built against one release runs on a later release with the same API major version. It needs only
  `-I …/InferKitMLX.framework/Headers` and works with `--no-swift-interfaces`. Names follow the
  Objective-C importer: `+backendNamed:weightsURL:error:` is
  `NFKMLXModelRegistry.backendNamed(_:weightsURL:)`, and `+sharedCatalog` is `NFKModelCatalog.shared`.
- **`import InferKitMLX`** — the Swift interface, with every Swift-only building block (`MLXArray`,
  `Module`, the `*Configuration` structs). It exposes mlx-swift types, which change between mlx-swift
  releases, so the API version does not cover it, and a client that uses it is rebuilt with each
  release.

```swift
import InferKitMLXObjC

NFKMLXReferenceModels.registerAll()
let backend = try NFKMLXModelRegistry.backendNamed("green-screen-keyer", weightsURL: nil)
let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
let models = NFKModelCatalog.shared.visibleDescriptors
```

The framework's `Modules` directory holds the Swift interfaces of InferKitMLX and of every module it
imports (MLX, MLXNN, MLXFast, MLXRandom, MLXFFT, MLXOptimizers, swift-numerics), plus the two C modules
those interfaces import, `Cmlx` and `_NumericsShims`.

### Swift Package Manager, either variant

A `binaryTarget` consumes the xcframework directly. SwiftPM wires the static variant's headers and
modulemap for you, so nothing else is needed:

```swift
.binaryTarget(name: "InferKitMLX", path: "InferKitMLX.xcframework"),
.executableTarget(name: "App", dependencies: ["InferKitMLX"],
                  linkerSettings: [.linkedFramework("Metal"), .linkedFramework("Accelerate"),
                                   .linkedFramework("CoreGraphics")]),
```

The dynamic variant needs the framework's `Headers` directory for the core's module, and a Swift target
that imports the Swift interfaces needs its `Modules` directory too. SwiftPM does not resolve a
relative path in `unsafeFlags` against the package root, so the manifest builds absolute paths from
its own location. The slice path names one platform:

```swift
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path   // import Foundation
let framework = "\(root)/InferKitMLXDynamic.xcframework/macos-arm64/InferKitMLX.framework"

.binaryTarget(name: "InferKitMLX", path: "InferKitMLXDynamic.xcframework"),
.executableTarget(name: "App", dependencies: ["InferKitMLX"],
                  cSettings: [.unsafeFlags(["-I", "\(framework)/Headers"])],
                  swiftSettings: [.unsafeFlags(["-I", "\(framework)/Headers",
                                                "-I", "\(framework)/Modules"])],
                  linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path"])]),
```

A target that imports only `InferKitMLXObjC` leaves out the `Modules` path.

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

The compressed assets, produced with
`ditto -c -k --sequesterRsrc --keepParent <artifact> <artifact>.zip`; `build-mlx.sh` writes the MLX
ones and `release.json` into `release/`:

| asset | contents | size |
| --- | --- | --- |
| `InferKit.xcframework.zip` | the core alone, no MLX | 1.3 MB |
| `InferKitMLX.xcframework.zip` | static, three slices, Metal library inside each | 38 MB |
| `InferKitMLXDynamic.xcframework.zip` | dynamic, three slices | 25 MB |
| `InferKitMLX-<version>-macos-arm64.zip` | the macOS `InferKitMLX.framework`, the unit an app installs | |
| `release.json` | the manifest: versions, minimum systems, commit, mlx-swift pin, each zip's SHA-256 | |

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
name a `url:` and `checksum:`. It is the `sha256` `release.json` records for the zip.

## API version

`NFKInferKit.releaseVersion` is the release, such as `0.4.0`. `NFKInferKit.APIVersion` is the
binary interface, `"major.minor"`, with `APIMajorVersion` and `APIMinorVersion` as integers. Several
releases can share an API version. In Swift they read `NFKInferKit.releaseVersion`,
`NFKInferKit.apiVersion`, `apiMajorVersion`, and `apiMinorVersion`.

The promise covers the Objective-C interface and the C ABI: every class, selector, protocol,
constant, enumeration value, function, and structure the public headers declare, in the core and in
InferKitMLX's Objective-C interface (`@import InferKitMLX;` or Swift's `import InferKitMLXObjC`). It
does not cover the Swift interface of InferKitMLX (`import InferKitMLX`), whose public API exposes
mlx-swift types that change between mlx-swift releases.

- The **major** version changes only on an incompatible change: a declaration removed or renamed, a
  type, selector, constant value, enumeration value, or structure layout changed, a superclass or
  adopted protocol dropped, or a required method added to a protocol. The minor version resets to 0.
- The **minor** version counts additions: a new class, method, property, constant, function,
  enumeration case, protocol, or optional protocol method.
- A release that changes neither keeps the API version.

A client compiled against `NFK_API_VERSION_MAJOR` and `NFK_API_VERSION_MINOR` (macros in
`NFKInferKit.h`) runs on a library with the same major version and a minor version at least as high.
The API version starts at 1.0. InferKit 0.4.0 is the first release with an API version. Until it
ships, the API version stays 1.0 while the interface changes, and that release fixes what 1.0 names.

## Side-by-side releases

A macOS app can install InferKitMLX releases apart from itself and update them without being
rebuilt. Each release lives in its own folder under a root the app owns, such as
`~/Library/Application Support/<app>/InferKit`:

```
<root>/0.4.0/InferKitMLX.framework
<root>/0.4.0/release.json
<root>/0.5.0/InferKitMLX.framework
<root>/0.5.0/release.json
```

Installing a release: download `release.json`, verify its signature, download the framework zip it
names, check the zip's SHA-256 against the manifest, and unzip it with `ditto -x -k` into a new folder
named for the version, beside the manifest. Older folders stay until no process uses them.

Choosing one at launch:

1. Read every `<root>/*/release.json`.
2. Keep the releases whose `apiVersion` has the app's API major version and a minor version at least
   the app's, and whose `minimumSystemVersions.macOS` the machine meets.
3. Take the highest API minor version, then the highest `version`.
4. Load it, then check `NFKInferKit.APIVersion` against the manifest's.

The app's own executable must not link `InferKitMLX.framework`, or the system loads it at launch from
the run path instead. The app reaches InferKit one of two ways after loading a release:

- by name, through `NSClassFromString(@"NFKMLXModelRegistry")` and `dlsym` for constants, compiled
  against the headers of the release it was built with;
- through its own plug-in bundle or library that links `@rpath/InferKitMLX.framework/InferKitMLX` and
  uses InferKit directly, in Objective-C or through `import InferKitMLXObjC`. Load it after the
  framework: the system binds that install name to the framework already loaded, whatever the run
  path says.

A loaded framework cannot be unloaded, so a newer release takes effect at the next launch. iOS
apps cannot load code they download; they embed the framework at build time.

A loader that picks and loads the newest supported release, compiled against InferKit's headers and
linked to nothing from InferKit:

```objc
#import <Foundation/Foundation.h>
#import <InferKit/NFKInferKit.h>    // NFK_API_VERSION_MAJOR and _MINOR: the API this app was built against

static BOOL SystemMeets(NSString *minimum)
{
	NSArray<NSString *> *parts = [minimum componentsSeparatedByString:@"."];
	NSOperatingSystemVersion version = { parts[0].integerValue,
										 parts.count > 1 ? parts[1].integerValue : 0,
										 parts.count > 2 ? parts[2].integerValue : 0 };
	return [NSProcessInfo.processInfo isOperatingSystemAtLeastVersion:version];
}

/// The newest release under root this app supports, loaded; nil when there is none or it fails to load.
NSBundle *LoadNewestInferKit(NSURL *root, NSError **error)
{
	NSDictionary *chosen = nil;
	NSURL *chosenFolder = nil;
	NSArray<NSURL *> *folders = [NSFileManager.defaultManager contentsOfDirectoryAtURL:root
		includingPropertiesForKeys:nil options:NSDirectoryEnumerationSkipsHiddenFiles error:error];
	for (NSURL *folder in folders) {
		NSData *data = [NSData dataWithContentsOfURL:[folder URLByAppendingPathComponent:@"release.json"]];
		NSDictionary *manifest = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
		NSArray<NSString *> *api = [manifest[@"apiVersion"] componentsSeparatedByString:@"."];
		if (api.count != 2 || api[0].integerValue != NFK_API_VERSION_MAJOR
			|| api[1].integerValue < NFK_API_VERSION_MINOR) {
			continue;
		}
		NSString *minimum = manifest[@"minimumSystemVersions"][@"macOS"];
		if (minimum == nil || !SystemMeets(minimum)) {
			continue;
		}
		NSComparisonResult byAPI = [manifest[@"apiVersion"] compare:chosen[@"apiVersion"] options:NSNumericSearch];
		NSComparisonResult byRelease = [manifest[@"version"] compare:chosen[@"version"] options:NSNumericSearch];
		if (chosen == nil || byAPI == NSOrderedDescending
			|| (byAPI == NSOrderedSame && byRelease == NSOrderedDescending)) {
			chosen = manifest;
			chosenFolder = folder;
		}
	}
	if (chosen == nil) {
		return nil;
	}
	NSBundle *framework = [NSBundle bundleWithURL:[chosenFolder URLByAppendingPathComponent:@"InferKitMLX.framework"]];
	if (![framework loadAndReturnError:error]) {
		return nil;
	}
	// The framework stays loaded either way. A mismatch means the folder does not hold what its
	// manifest says, and the app must not use it.
	Class library = NSClassFromString(@"NFKInferKit");
	if (![[library APIVersion] isEqualToString:chosen[@"apiVersion"]]) {
		return nil;
	}
	return framework;
}
```

Verifying the manifest's signature is not shown: signing is not settled yet, and
`Docs/agent-reference/distribution-and-packaging.md` ("Release manifest") holds the proposal.
