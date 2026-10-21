# Plugin examples

Each folder here is a plugin a plugin author copies as a starting point. Each one holds its sources, its
`inferkit-plugin.json` manifest, an `Info.plist` template, a README, and a `build.sh` that assembles the
`.inferkitplugin` bundle in the folder's `build/` and signs it ad hoc. None of them loads weights or
downloads anything. The folder ships in no distribution. [Docs/plugins.md](../Docs/plugins.md) is the
guide to every rule these examples follow.

| folder | `entry.kind` | what it shows | copy it for |
| --- | --- | --- | --- |
| [NewFamily-ObjC](NewFamily-ObjC/) | `objc` | A new model family, `color-palette`: two models, a per-model option, a plugin option, and the family contract other plugins implement. | A plugin that defines a kind of model of its own. |
| [ExistingFamily-ObjC](ExistingFamily-ObjC/) | `objc` | A model that joins the `color-palette` family with another algorithm, and an app that treats the members as interchangeable. | A plugin that offers another implementation of a family that exists. |
| [Filter-C](Filter-C/) | `c-abi` | An image filter against `NFKPluginABI.h` alone: request and result buffers, progress, cancellation, and who frees what. | A plugin in C, C++, Rust, or any language with a C interface. |
| [Tools-ObjC](Tools-ObjC/) | `objc` | Two tools and no models, each with a JSON Schema for its arguments. | Tools for an agent or for a server's MCP route. |
| [RemoteService-ObjC](RemoteService-ObjC/) | `objc` | A chat model served by an OpenAI-compatible server, through InferKit's `NFKRemoteBackend`, configured by plugin options. | A plugin that brings a hosted or local HTTP service into the catalog. |
| [Executable-Swift](Executable-Swift/) | `executable` | A Swift program that links InferKit and serves its own `NFKPlugin` with `NFKPluginService`. | A plugin that needs libraries the plugin host lacks, such as InferKitMLX. |

## Building

Each script runs from any directory and finds the InferKit source root two folders above itself:

```
PluginExamples/NewFamily-ObjC/build.sh
```

The script reads the plugin's identifier and version from the manifest and writes them into
`Info.plist`, so the manifest is the one place they are set. It writes over an earlier build in place.
It sets `umask 022`, since the loader refuses a bundle its group or others can write.

## Signing

The scripts sign for development:

```
codesign --force --sign - build/ColorPalette.inferkitplugin
codesign --verify --strict build/ColorPalette.inferkitplugin
```

A plugin for other people's machines is signed with a Developer ID and the hardened runtime instead:

```
codesign --force --timestamp --options runtime \
	--sign "Developer ID Application: <Team> (<TEAMID>)" build/ColorPalette.inferkitplugin
```

The signature seals `Contents/Resources`, so the manifest cannot change after signing. Sign after the
manifest and `Info.plist` are in place.

## Installing

An app keeps its plugins in a folder of its own, such as one inside its Application Support folder. The
folder, the bundle, and everything inside the bundle belong to the user and are writable by neither the
group nor others ([the ownership rule](../Docs/plugins.md#the-ownership-rule)). Installing a plugin is
copying its bundle there:

```
ditto build/ColorPalette.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/ColorPalette.inferkitplugin"
```

The app then lists the folder:

```objc
NFKPluginRegistry *plugins = NFKPluginRegistry.sharedRegistry;
NSArray<NSError *> *errors = nil;
NSArray<NFKPluginManifest *> *added = [plugins addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

Listing reads each manifest and loads no code. Each model joins `NFKModelCatalog.sharedCatalog` as
`<plugin id>/<model id>`, and each tool joins `plugins.tools`. A plugin's process starts when one of its
models first prepares or runs, or when one of its tools is called.

An `objc` or `c-abi` plugin runs in `inferkit-plugin-host`, which an app embeds
([Embedding the plugin host](../Docs/plugins.md#embedding-the-plugin-host)). During development the app
names the one the InferKit package builds:

```
swift build --product inferkit-plugin-host
```

```objc
plugins.hostExecutableURL = [NSURL fileURLWithPath:@"/path/to/InferKit/.build/debug/inferkit-plugin-host"];
```

An `executable` plugin is its own program and needs no host.

## Signature policies

The registry checks a plugin's signature when the plugin first launches or loads, under
`signaturePolicy`:

| policy | accepts | for |
| --- | --- | --- |
| `+[NFKPluginSignaturePolicy defaultPolicy]` | a valid signature that is not ad hoc | an app in its users' hands |
| `+[NFKPluginSignaturePolicy developmentPolicy]` | a valid signature, ad hoc included | the ad hoc builds these scripts make, on the developer's machine |
| `teamIdentifier` set | a signature by that team | an app that accepts its own team's plugins only |
| `requirementString` set | a signature that also satisfies a code requirement | an app with an allowlist of its own |

```objc
plugins.signaturePolicy = [NFKPluginSignaturePolicy developmentPolicy];
```

## Trust and the app's process

Every plugin runs in a process of its own by default. A plugin the app trusts, and whose manifest sets
`process.allowsInProcess`, runs in the app's process from its next load:

```objc
[plugins setTrusted:YES forPluginIdentifier:@"com.example.inferkit.color-palette"];
```

The five `objc` and `c-abi` examples allow it. The executable example cannot run in the app's process,
and trusting it makes its backends fail with `NFKPluginErrorNotTrusted`. Before trusting a plugin, the
app shows the user `NFKPluginInProcessWarningNoUnload`, `NFKPluginInProcessWarningCrashTakesDownApp`,
and `NFKPluginInProcessWarningFullAccess`. The registry keeps trust and option values in memory only, so
the app persists them and sets them again at launch.

## Names

The `NFK` prefix belongs to InferKit. A bundle that defines a class with that prefix is refused with
`NFKPluginErrorDuplicateRuntime`, so every class here carries a prefix of its own plugin's name. A
plugin identifier, a tool name, and a catalog identifier are each unique among the plugins an app lists:
two plugins that declare the same tool name cannot both be listed.
