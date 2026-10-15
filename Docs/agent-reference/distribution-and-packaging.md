<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Distribution and packaging

## Distribution

InferKit ships as source through two channels that reference the same files:

- **Swift Package Manager** — `Package.swift`. The target's public API is `Sources/InferKit/include/`,
  so `#import <InferKit/NFKFoo.h>` resolves the same way against SwiftPM and the built framework.
  Quoted sibling imports (`#import "NFKFoo.h"`) resolve through the target's `headerSearchPath`
  entries (`include/InferKit` and `.`). The core depends on `InferKitC` (`Sources/InferKitC`), the
  pure-C headers: the API version, the plugin ABI, and the C API. The package's second product is the
  macOS executable `inferkit-plugin-host`. `plugins.md` covers both.
- **CocoaPods** — `InferKit.podspec`, a source pod. `source_files` compiles the same `.h`/`.m`, and
  `InferKitC`'s `.h`/`.c`; `public_header_files` marks `include/InferKit/*.h` and
  `include/InferKitC/*.h` as the public surface. CocoaPods flattens both into `InferKit/`. The MLX companion is not
  a pod (MLX distributes through SwiftPM only), though it does package as an XCFramework — see
  "Packaging InferKitMLX as an XCFramework".

Keep the two in sync: a new source file is picked up by the SwiftPM glob automatically; the podspec
globs the same paths, so no per-file edit is needed there either. New public headers go in
`Sources/InferKit/include/InferKit/` and the umbrella `InferKit.h`.

`Tools/` ships in no distribution — it is developer and validation tooling, not the library. The
SwiftPM targets name `Sources/InferKit`, `Sources/InferKitC`, `Sources/inferkit-plugin-host`, and
`InferKitMLX/Sources` as their paths, the podspec globs only `Sources/InferKit` and `Sources/InferKitC`, and the XCFramework release assets are built binaries; none of them
compile or carry anything under `Tools/`. A consumer resolving the SwiftPM package clones the whole
repository (so `Tools/` lands on disk), but nothing there is built into the products or the release.
The 46 `Tools/*-to-safetensors` converters are therefore not required to consume the
library: `NFKMLXTorchFormat` reads a raw `.pth`/`.pt`/`.ckpt`/`.th`/HF `.bin` natively (see `NFKMLXTorchCheckpoint` in
`mlx-weights-and-formats.md`), so a consumer needs no Python. The converters stay for two reasons
that are not consumer-facing — they are the **byte oracle** `NFKMLXTorchParityTests` holds the native
reader to, and the **offline path** to a portable, pickle-free safetensors — and the Python
reference-parity tooling (`Tools/reference-parity`, `Tools/validation-assets`) is the ground truth
every Swift port is measured against, which is irreducibly Python (torch / transformers / diffusers).
`Tools/README.md` records this boundary.

The version lives in three places that must move together on release: `s.version` in
`InferKit.podspec`, the string `NFKInferKit.releaseVersion` returns
(`Sources/InferKit/NFKInferKit.m`), and the `vX.Y.Z` git tag. A test asserts the shape of
`NFKInferKit.releaseVersion`, not its value, so a stale constant does not fail CI — bumping it is a release step. SwiftPM takes its version from the tag, so
`Package.swift` carries none. `Tools/xcframework/inferkit-version.sh` reads the version and the API
version from the sources, so the framework's `Info.plist` and `release.json` record what the build
compiled.

## API version

`NFKInferKit.APIVersion` (`"major.minor"`) with `APIMajorVersion` and `APIMinorVersion` versions the
binary interface, apart from the release version. The numbers live in one place,
`NFK_API_VERSION_MAJOR` and `NFK_API_VERSION_MINOR` in the pure-C `NFKAPIVersion.h`
(`Sources/InferKitC/include/InferKitC/`), which `NFKInferKit.h` imports and the C API shares. The
class properties return the macros, so a client compiled against the headers records the interface it
expects, and the loaded library reports the interface it provides. The core's `build.sh` and
`build-mlx.sh` copy `NFKAPIVersion.h` beside the core's headers in each framework, where
`NFKInferKit.h` finds it as `<InferKit/NFKAPIVersion.h>`.

What the promise covers: the Objective-C interface and the C ABI of the core and of the `@objc`
interface the companions generate (`InferKitMLX-Swift.h`). That is every class by its runtime name,
superclass, adopted protocol, selector, property accessor, protocol method, exported constant and
its value, enumeration value, C function, and structure layout. The Swift interface of InferKitMLX is
outside it, because its public Swift API exposes mlx-swift types (`MLXArray`, `Module`) that change
between mlx-swift releases.

Which change moves which number:

- Removing or renaming any covered declaration → major.
- Changing a method's types or a function's signature → major.
- Changing a constant's value or an enumeration case's value → major. A client may have stored it.
- Changing a structure's fields → major.
- Changing a class's superclass or runtime name, or dropping an adopted protocol → major.
- Adding a required method to a protocol, or making an optional one required → major. A client's
  class that adopts the protocol lacks it.
- Making a required protocol method optional → major. A client calls it without checking.
- Adding a class, method, property, constant, function, enumeration case, protocol, or optional
  protocol method → minor.
- A major change resets the minor version to 0. A release that changes neither keeps the API
  version.
- Deprecating changes nothing. The removal that follows is the major change.

The starting point is 1.0. A loader compares numbers by one rule, same major and minor at least the
client's, and a 0.x API version would need a second rule because semantic versioning gives 0.x no
compatibility promise. The API version is independent of the release version, so it need not wait
for a 1.0 release. Before the first tagged release the API version stays 1.0 while the interface
changes, and the first tagged release fixes what 1.0 names. From then on `Tools/abi/check-abi.sh`
holds each release to the rule.

Swift names through the Objective-C interface follow the Objective-C importer, which can differ from
the Swift module's own spelling: `+backendNamed:weightsURL:error:` is
`NFKMLXModelRegistry.backendNamed(_:weightsURL:)` through `InferKitMLXObjC` and
`backend(named:weightsURL:)` through `InferKitMLX`. The promise covers the importer's spelling,
because it derives from the selector.

## Packaging InferKitMLX as an XCFramework

`Tools/xcframework/build-mlx.sh` emits **two** artifacts from one archive per slice (arm64; macOS,
iOS device, iOS simulator):

- `InferKitMLX.xcframework` — static: `libInferKitMLX.a` (every target's object merged with `libtool`,
  the core included), headers, a modulemap, and the `mlx-swift_Cmlx.bundle` that slice's consumer
  ships. 60 to 63 MB a slice, 183 MB in all.
- `InferKitMLXDynamic.xcframework` — dynamic: `InferKitMLX.framework`, Metal library inside. A slice is
  a 21 MB binary, 3.7 MB of shaders, and 19 MB of Swift module interfaces an Objective-C consumer
  never reads; 130 MB in all.
- `release/` — each macOS slice's framework zipped, the xcframework zips, and `release.json`
  (`Tools/xcframework/release.sh`, see "Release manifest"). `--no-release` skips it.

The dynamic framework is self-contained:

| path | holds |
| --- | --- |
| `InferKitMLX` | the binary, install name `@rpath/InferKitMLX.framework/InferKitMLX` |
| `Resources/mlx.metallib` | MLX's shaders |
| `Resources/Info.plist` | `CFBundleShortVersionString` (the release version) and `NFKAPIVersion` |
| `Headers/InferKitMLX-Swift.h` | the generated Objective-C interface, the framework module's header |
| `Headers/InferKit/` | the core's headers and its `module InferKit` map |
| `Headers/InferKitMLXObjC/` | the Objective-C-only module for Swift clients (see below) |
| `Modules/module.modulemap` | `framework module InferKitMLX` |
| `Modules/*.swiftmodule`, `Modules/Cmlx`, `Modules/mlx`, `Modules/_NumericsShims` | the Swift interfaces and the C modules beneath them, dropped by `--no-swift-interfaces` |

A consumer passes `-F <slice>` and `-I <slice>/InferKitMLX.framework/Headers`, and a Swift consumer of
the Swift interfaces also `-I <slice>/InferKitMLX.framework/Modules`. Clang looks up a module named X
in `<search dir>/X/module.modulemap`, and accepts a plain module there even inside a framework; only
the framework's own map is limited to framework modules. The plain modules sit in `Headers` and not
in `Modules`: clang also reads `<search dir>/module.modulemap`, and through `-I Modules` it read the
framework's own map a second time and refused the framework module as "built in directory
`InferKitMLX.framework` but now resides in `InferKitMLX.framework/Modules`". The framework carries the
only `InferKit` module its consumer needs. A second `InferKit` map on another search path redefines
the module, which is why the separate `CoreHeaders/` an earlier layout shipped beside the xcframework
is gone. Measured 2026-10-15 with a two-module stand-in: with the same module in two search paths, one
order compiles and the other fails with "redefinition of module".

A static consumer's binary grows by about 29 MB plus the shaders. A dynamic consumer ships the
21 MB framework binary, which carries them.

Run the script from a checkout whose root folder is named `InferKit`. SwiftPM takes a path dependency's
identity from its folder name, so InferKitMLX's `../` dependency on the core fails to resolve from a
differently named copy ("unknown package 'InferKit' in dependencies of target 'InferKitMLX'"), as the
workspace's `group:../InferKit` reference does.

What made it possible. MLX needs `default.metallib`, which SwiftPM delivers as a
`mlx-swift_Cmlx.bundle` resource, and a bare library carries no resources. MLX's loader
(`mlx/backend/metal/device.cpp`) tries four places, and `current_binary_dir()` is `dladdr` on MLX's
own code, so it names whichever mach-O image MLX was linked into:

1. `<binary dir>/mlx.metallib` — beside the consumer's binary.
2. `<binary dir>/Resources/mlx.metallib` — the dynamic route, inside the framework.
3. `mlx-swift_Cmlx.bundle` in the main or any loaded bundle — the static route inside an app bundle,
   and the layout Copy Bundle Resources produces.
4. `<binary dir>/Resources/default.metallib` — commented in the reference as "if SwiftPM wrapped as a
   dynamic framework".

A dynamic framework satisfies (2) by itself. It ships `Resources/mlx.metallib`, and (2) comes before
the host's bundles, so the framework loads the shaders it was built with whatever `mlx-swift_Cmlx.bundle`
the host app ships. With (4), which an earlier layout used, a host that shipped its own bundle (an app
that also links mlx-swift through SwiftPM, or an older release's static artifact) supplied the shaders
instead, built from an mlx core the framework's code was not compiled against. A static consumer satisfies (3)
from the bundle inside their slice. `verify-mlx.sh` measures each route, links a consumer, and **runs
a model**: a binary that links can still fail at the first array evaluation, which is the failure this
packaging exists to avoid. The InferKitMLXObjC probe runs beside a decoy `mlx-swift_Cmlx.bundle` holding
no valid library and checks that `NFKMLXGPU.metalLibraryURL` names the framework's
`Resources/mlx.metallib`; `NFKMLXGPU.metalLibraryURL` follows the loader's order, so the decoy would win
if the framework's copy came after it.

Nothing else in a loaded framework resolves through the host's main bundle. Neither package declares
SwiftPM resources, so there is no `Bundle.module` lookup; the model catalog's descriptors are compiled
code (`NFKMLXModelDescriptors`), not a resource file; and the framework's install name is
`@rpath/InferKitMLX.framework/InferKitMLX`, so it loads from wherever the host's run path or `dlopen`
points. The only main-bundle reads in the sources are MLX's loader (3) and `NFKMLXGPU.metalLibraryURL`,
which mirrors it. Checked 2026-10-15 by searching `Sources/` and `InferKitMLX/Sources/` for `Bundle.main`,
`Bundle.module`, `mainBundle`, `Bundle(for:`, `allBundles`, `allFrameworks`, `pathForResource`, and
`url(forResource:`, and the mlx-swift and swift-numerics checkouts for the same.

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

Neither variant is smaller. A consumer using one model links to 29 MB static, dead-stripped. A dynamic
consumer's own binary is tiny, 52 KB for the verification probe, beside the 21 MB framework binary it
ships. A static slice weighs 60 to 63 MB and a dynamic one 43 to 44 MB, and the shaders are 3.7 MB of
that. `Tools/xcframework/verify-mlx.sh` reports both link sizes, so these figures are re-measured
whenever a release is built. Measured 2026-09-15. So the choice is deployment mechanics:

- Static needs no embedding and no code signing, and vends both modules from one plain modulemap.
- Dynamic is one self-contained drop-in, carries its own `InferKit` module in `Headers/InferKit`, is
  shared between several consumers, and is the variant an app installs and updates apart from
  itself (see "Side-by-side installation").
- A Swift consumer takes the dynamic variant. The static library carries no Swift interfaces.

Linking recipes and the release matrix live in `Docs/installation.md`, one per consumer shape: Xcode
static and dynamic, SwiftPM `binaryTarget` for both variants (SwiftPM wires the static one's headers
and modulemap automatically; the dynamic one needs the framework's `Headers` directory, and a Swift
target that imports the Swift interfaces its `Modules` directory), plug-in bundles, a consumer's own
static library, and an app that loads a release from a folder.
The artifacts are not committed — this repository is source-distributed, so a consumer resolving it
clones its history. Three compressed release assets, one per variant carrying every slice: core 1.3 MB,
static 38 MB, dynamic 25 MB. The core's `build.sh` cleans only its own artifact — both scripts share
`xcframework-build/`, and an `rm -rf` of the whole directory once discarded a twenty-minute MLX build
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

### Swift consumers of the dynamic framework

The framework's `Modules` directory carries the `.swiftinterface` of InferKitMLX and of every Swift
module it imports (MLX, MLXNN, MLXFast, MLXRandom, MLXFFT, MLXOptimizers, swift-numerics). Those
interfaces import two C modules: `Cmlx` (MLX, MLXFast, MLXRandom) and `_NumericsShims` (RealModule).
SwiftPM supplies both from the package checkouts, so a framework that ships only the interfaces fails
a binary consumer with `no such module 'Cmlx'`. `build-mlx.sh` copies them from the build's
`SourcePackages/checkouts` into `Modules`:

- `Modules/Cmlx/` holds mlx-swift's `module.modulemap` and umbrella `mlx.h`.
- `Modules/mlx/c/` holds the mlx-c headers. They include each other as `"mlx/c/array.h"`, resolved
  through the header search path, so the tree sits at the `Modules` root.
- `Modules/_NumericsShims/` holds swift-numerics' module map and header.

The consumer passes `-I <slice>/InferKitMLX.framework/Modules` to reach the dependency interfaces, and
clang looks up a module named X in `<search dir>/X/module.modulemap`, so the C modules need no further
flag. The core's module comes from `-I <slice>/InferKitMLX.framework/Headers`. In Xcode the import
paths are `$(BUILT_PRODUCTS_DIR)/InferKitMLX.framework/Modules` and `…/Headers`, because Xcode copies
the slice's framework into the build products. SwiftPM rejects `$(BUILT_PRODUCTS_DIR)` in
`unsafeFlags` and runs the compiler outside the package root, so a relative `-I` does not resolve
there; the SwiftPM recipe builds absolute paths from `#filePath` and names the slice. Both recipes
were built and run on 2026-10-13 (Xcode 27, Swift 6.4) with the core module from the separate
`CoreHeaders/` an earlier layout shipped: `xcodebuild` with `SWIFT_INCLUDE_PATHS`, and `swift build`.
`verify-mlx.sh` builds the command-line form with `Headers` in its place.

`--no-swift-interfaces` drops the interfaces and the C modules together. `verify-mlx.sh` compiles and
runs a Swift probe (`import InferKitMLX`, `import MLX`, an evaluated `MLXArray`) against the dynamic
slice with a private module cache, so each interface is rebuilt from the shipped text.

### The Objective-C-only module, InferKitMLXObjC

`import InferKitMLXObjC` gives a Swift client the Objective-C interface of InferKitMLX and nothing of
its Swift interface. The client then binds through Objective-C dispatch, which the API version
covers, and a framework built against a newer mlx-swift still loads under it.
`Headers/InferKitMLXObjC/` holds `InferKitMLXObjC.h` and a map:

```
module InferKitMLXObjC {
    header "InferKitMLXObjC.h"
    link framework "InferKitMLX"
    export *
}
```

The header is `InferKitMLX-Swift.h` with `__has_attribute(external_source_symbol)` replaced by `0`. The
generated header wraps its declarations in `#pragma clang attribute push(__attribute__((external_source_symbol(language="Swift", defined_in="InferKitMLX", generated_declaration))))`,
which tells Swift the declarations belong to the Swift module InferKitMLX. Swift then imports each
class with no members, which is why importing the framework module from Swift without its interfaces
never worked. With the attribute off, the declarations are plain Objective-C. `build-mlx.sh` fails
when a generator guards the attribute differently. The module imports `InferKit` and re-exports it, so
one `import InferKitMLXObjC` reaches the core too, and it never imports the framework module, so
Swift loads no `InferKitMLX.swiftmodule` and no MLX interface. `link framework` autolinks the binary.

A Swift client imports `InferKitMLXObjC` or `InferKitMLX`, never both: each declares the same classes.
An Objective-C client keeps `@import InferKitMLX;`.

`verify-mlx.sh` compiles a Swift probe that imports only `InferKitMLXObjC`, passes `-F` and
`-I <framework>/Headers` and no `-framework`, calls `NFKMLXModelRegistry.registeredModelNames`,
`NFKMLXReferenceModels.registerAll()`, `NFKMLXModelRegistry.backendNamed(_:weightsURL:)`, and
`NFKModelCatalog.shared`, runs the green-screen keyer, and checks two things beyond its output: the
compiler's loaded-module trace names no `InferKitMLX` or `MLX` Swift module, and `nm -u` lists no
`$s11InferKitMLX` symbol. It runs with and without `--no-swift-interfaces`.

## Release manifest

`Tools/xcframework/release.sh` writes `<output>/release/`, and `build-mlx.sh` runs it after any build
that includes the dynamic variant:

| file | what it is |
| --- | --- |
| `InferKitMLX-<version>-macos-arm64.zip` | the macOS slice's `InferKitMLX.framework`, the unit an app installs |
| `InferKitMLXDynamic.xcframework.zip` | the dynamic xcframework, for building against |
| `InferKitMLX.xcframework.zip` | the static xcframework, when it was built |
| `release.json` | the manifest |

The zips come from `ditto -c -k --sequesterRsrc --keepParent`, which keeps a macOS framework's
symbolic links. Only macOS slices become installable framework assets: an iOS app may not load code
it downloads, so the iOS slices ship inside the xcframework zips and are embedded at build time.

`release.json`:

```json
{
  "kind": "inferkit.release",
  "schemaVersion": 1,
  "product": "InferKitMLX",
  "version": "0.4.0",
  "apiVersion": "1.0",
  "minimumSystemVersions": { "macOS": "14.0", "iOS": "17.0" },
  "commit": "<the InferKit commit>",
  "commitHasLocalChanges": false,
  "mlxSwift": {
    "repository": "https://github.com/ml-explore/mlx-swift",
    "revision": "901941965d82e4a216d4d117231d847d194c563d",
    "version": null,
    "mlxCoreVersion": "0.32.2"
  },
  "artifacts": [
    { "file": "InferKitMLX-0.4.0-macos-arm64.zip", "kind": "framework", "platform": "macOS",
      "libraryIdentifier": "macos-arm64", "minimumSystemVersion": "14.0", "bytes": 0, "sha256": "…" },
    { "file": "InferKitMLXDynamic.xcframework.zip", "kind": "xcframework", "linkage": "dynamic",
      "bytes": 0, "sha256": "…" }
  ],
  "signature": ""
}
```

- `version` and `apiVersion` come from the sources (`inferkit-version.sh`). The framework's
  `Info.plist` carries the same two values as `CFBundleShortVersionString` and `NFKAPIVersion`.
- `minimumSystemVersions` lists the platforms the build included.
- `commitHasLocalChanges` is true when tracked files differed from the commit. A published release has
  false.
- `mlxSwift` records the pin from `InferKitMLX/Package.resolved`; `version` is null while the pin is a
  revision. `mlxCoreVersion` is read from the build's own checkout (`mlx/version.h`), and is null when
  `release.sh` runs without one.
- `sha256` is the digest of the zip's bytes, the value `swift package compute-checksum` prints, so the
  xcframework entries double as a `binaryTarget` checksum.
- `signature` is written empty, as the last member, on its own line.

### The signing step (open)

Signing, notarization, and publishing wait for the developer. The manifest leaves room for a detached
signature with one rule: the signed bytes are the manifest's bytes as `release.sh` writes them, with the
`signature` value empty. A verifier restores those bytes by replacing the value of the last
`"signature": "…"` line with `""`, with no JSON canonicalization. The proposed value is
`ed25519:<base64 of the 64-byte signature>`.

Open question for the developer, which key signs what:

- **Ed25519 over the manifest.** The app pins the public key and verifies the manifest before it
  downloads anything, then checks the zip against the manifest's `sha256`. It authenticates the
  metadata an app decides with (`version`, `apiVersion`), and verifies on macOS 10.15 and later
  through CryptoKit's `Curve25519.Signing`. It does not satisfy the system's code-signing checks.
- **Developer ID codesign of the framework plus a notarization ticket.** A hardened app with library
  validation loads a framework from outside its bundle only when the framework is signed by the same
  team, unless it holds `com.apple.security.cs.disable-library-validation`. The signature covers the
  code on disk, not the manifest.
- The two protect different things, so the proposal is both: codesign and notarize the framework
  before zipping, then sign the manifest that names the zip's digest.

## Side-by-side installation

An app installs InferKitMLX releases apart from itself, each version in its own folder, and loads the
newest one it supports:

```
<root>/0.4.0/InferKitMLX.framework
<root>/0.4.0/release.json
<root>/0.5.0/InferKitMLX.framework
<root>/0.5.0/release.json
```

Choosing: read each `release.json`; keep the releases whose API major version equals the one the app
was built against (`NFK_API_VERSION_MAJOR`) and whose minor version is at least the app's
(`NFK_API_VERSION_MINOR`) and whose `minimumSystemVersions` the machine meets; take the highest API
minor version, then the highest release version. After loading, compare `NFKInferKit.APIMajorVersion`
and `APIMinorVersion` with the manifest's, since a framework cannot be unloaded and a mismatch must
stop the app from using it.

How the app's code reaches it. The app's executable does not link the framework, or dyld loads it at
launch from the run path. The app either reaches classes by name (`NSClassFromString`, `dlsym` for
constants), or keeps its InferKit code in a bundle or dylib that links
`@rpath/InferKitMLX.framework/InferKitMLX` and loads it after the framework: dyld binds an `@rpath`
install name to an image already loaded under that install name, whatever the run path. Measured
2026-10-15 on a built macOS slice: `release.sh`'s framework zip, unzipped with `ditto -x -k` into
`<root>/0.3.1/` beside its `release.json`, with decoy folders for API 2.0 and an older release. The
sample loader from `Docs/installation.md` picked 0.3.1 and loaded it through
`-[NSBundle loadAndReturnError:]`. A Swift dylib that imports `InferKitMLXObjC`, links
`@rpath/InferKitMLX.framework/InferKitMLX`, and has no run path to it then loaded with `dlopen`, ran
the green-screen keyer on the GPU, and `NFKMLXModelRegistry` resolved to the copy in `<root>/0.3.1`.
A newer release takes effect at the next launch. The consumer sample loader is in
`Docs/installation.md` ("Side-by-side releases").

## ABI baseline check

`Tools/abi/check-abi.sh --baseline <current | framework | git ref> [--candidate <framework>]` holds a
built framework to a baseline's binary interface. `Tools/abi/abi-surface.py` reads each set of headers
through clang's JSON syntax tree. The headers are copied without their module maps, so clang reads
them as text while Foundation arrives as a module and stays out of the dump: 46 MB of JSON, against
355 MB with Foundation read as text. The surface it writes lists:

- each class by its runtime name, taken from the `_OBJC_CLASS_$_` symbol clang names, and each
  protocol by the runtime name the generated header gives it. A Swift class or protocol without
  `@objc(Name)` carries a mangled runtime name, such as `_TtC11InferKitMLX12NFKMLXKokoro`, which
  `NSClassFromString(@"NFKMLXKokoro")` does not find and which changes with the Swift module or
  nesting. `diff` fails on any such name in the candidate, with or without `--allow-unbumped`. Eleven
  InferKitMLX classes had one until 2026-10-15; each now carries `@objc(Name)` equal to its Swift name.
- its superclass, adopted protocols, and every class and instance selector with its normalized
  signature, property accessors included. A method marked unavailable is left out.
- each protocol's required and optional selectors. Clang's JSON does not mark `@optional`, so the
  sections come from the `@optional` and `@required` markers in the header text before each method's
  offset.
- each exported constant and C function with its type, each enumeration case's value, and each
  structure's fields.
- the API version the headers state.

Types are compared without nullability, ownership qualifiers, or generic arguments, none of which
change the binary interface.

The script compares the two surfaces under the API version rule, then compiles a client against the
baseline headers that loads the candidate with `dlopen` and checks every class with `objc_getClass`,
superclass, conformance (with `@protocol` references compiled from the baseline headers), selector
with `class_respondsToSelector` (which consults the resolver and runs no `+initialize`), and constant
and function with `dlsym`. With a baseline framework at hand, it reads every string and numeric
constant from both frameworks and fails on a changed value. A git ref is exported with `git archive`
and built with that tree's own `build-mlx.sh`, about seven minutes.

The first real baseline is the first tagged release. Until then `--baseline current` checks that the
framework implements every declaration its headers make, and that no public class or protocol has a
mangled runtime name. On 2026-10-15 that self-check passed with
406 classes, 3697 selectors, 177 conformances, 241 constants, and 10 functions, none missing. Controls
confirmed each check fails when it should: a removed selector, a missing class, a changed superclass, a
dropped conformance, a missing constant and function, a required protocol method added, and an
enumeration value changed, each under an unchanged API version. `--baseline HEAD` against the same
build, with HEAD at the commit before the API version existed, built the baseline framework from the
ref, reported the three `NFKInferKit` API-version selectors as the only additions, found nothing
missing, and compared 241 constant values with none changed.

## The mlx-swift pin blocks shipping the companion

`InferKitMLX/Package.swift` pins `mlx-swift` by **revision**
(`901941965d82e4a216d4d117231d847d194c563d`, 2026-09-17, vendoring mlx core 0.32.2) rather than by
version. The reason is a defect rather than a preference: mlx core 0.31.1, which the newest tagged
release 0.31.6 vendors, computes the wrong GPU gradient after a fine-tune's first step, measured at
0 of 25 readings matching the CPU against 60 of 60 on core 0.32.2.

**SwiftPM accepts a revision requirement only in a root package.** A consumer that declares
`InferKitMLX` as a dependency cannot resolve it while that pin stands, so the companion is buildable
and testable here and not consumable elsewhere. The core package is unaffected: it does not depend on
mlx-swift.

What clears it: upstream tags an `mlx-swift` release carrying mlx core 0.32 or newer, and the pin
goes back to a version requirement (`from: "0.32.x"`). As of 2026-09-21 the newest tag is 0.31.6
(2026-07-02), so the block stands. When the tag lands:

1. Change the requirement in `InferKitMLX/Package.swift` back to `from:` with the new version, and
   delete the comment explaining the revision pin.
2. Resolve, and check the vendored core in
   `InferKitMLX/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/version.h`. The mlx-swift version and
   the mlx core version differ, and the core version is the one that matters.
3. Rebuild the Metal library with `Tools/mlx-metallib.sh`; a core bump changes the kernels.
4. Run the companion's full check, and specifically the fine-tuning gradient tests that motivated the
   revision pin, since a tagged release must carry the fix for the pin to move.
5. Record the new core version in `mlx-runtime-gotchas.md` if any kernel behavior moved with it.
