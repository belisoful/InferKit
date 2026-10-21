# Text Statistics: an executable plugin in Swift

An executable plugin (`entry.kind` `executable`) written in Swift as a command-line program. The program
links InferKit and serves its own `NFKPlugin` object with InferKit's public `NFKPluginService`. Its one
model, `text-statistics`, counts the characters, words, sentences, lines, and paragraphs of a text and
estimates its reading time. It needs no weights.

This is the template for a plugin that needs libraries the plugin host lacks. The host,
`inferkit-plugin-host`, loads an `objc` or `c-abi` plugin into a process that holds InferKit alone. An
executable plugin is its own process, so it links whatever it needs.

| file | holds |
| --- | --- |
| `Sources/TextStatistics/main.swift` | the program: the plugin, its backend, its context, and the service |
| `Package.swift` | the SwiftPM package that builds the program, depending on InferKit by path |
| `inferkit-plugin.json` | the manifest |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds the program, assembles, and signs `build/TextStatistics.inferkitplugin` |

## How the program serves its plugin

The app launches the program with the plugin channel, a Unix socket, on descriptor 3, which the
environment names in `INFERKIT_PLUGIN_CHANNEL_FD`. The program:

1. reads its manifest from the bundle that holds it, with `NFKPluginManifest(bundleURL:)`;
2. makes its plugin object, `TextStatisticsPlugin`, with a context of its own (`StandaloneContext`);
3. makes `NFKPluginService(plugin:manifest:)` and calls `run(withChannelDescriptor:)` on descriptor 3.

The service performs the channel hand-off, answers every message of the plugin protocol, relays
progress, and returns when the app sends `shutdown` or closes the channel. The program then exits.

The service serves the plugin object it is handed, which the program makes before the app's hello
arrives, so the plugin's context carries no plugin option values. The manifest declares no plugin
options for that reason. A model's own options reach the plugin as they do in any kind: the build
options in `backend(forModelIdentifier:options:)`, and the request options in each request's
parameters.

## Build and sign

```
PluginExamples/Executable-Swift/build.sh
```

The script runs `swift build -c release` on the package beside it, copies the program into
`Contents/MacOS`, writes `Info.plist` and the manifest, and signs the bundle ad hoc. The package depends
on the InferKit package two folders up with `.package(path: "../..")`; a plugin kept in its own
repository names InferKit's URL and a version instead. InferKit links into the program statically.

- `swift build` builds for the machine's architecture, so the manifest lists `arm64`. A universal
  program builds with `--arch arm64 --arch x86_64`, and the manifest then lists both.
- The toolchain this example was built with, Swift 6.4 in Xcode 27, writes a deployment target of
  macOS 12.0 for a package that names macOS 11, so `Package.swift` and the manifest's
  `compatibility.minimumOS` name 12.0.
- For distribution, the program is signed with the hardened runtime:
  `codesign --force --timestamp --options runtime --sign "Developer ID Application: <Team> (<TEAMID>)"`.

## Install

```
ditto build/TextStatistics.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/TextStatistics.inferkitplugin"
```

```objc
NSArray<NSError *> *errors = nil;
[NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

The app needs no plugin host for this kind. The registry checks the running program's signature under
its policy when it starts.

## What the app sees

The catalog lists `com.example.inferkit.text-statistics/text-statistics`, family `text-statistics`,
with the option `wordsPerMinute` (238 by default). A request carries the text under `NFKInputPrompt`,
and the result carries the counts under `NFKOutputStructured`:

```objc
NSError *error = nil;
id<NFKInferenceBackend> statistics =
	[NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"com.example.inferkit.text-statistics/text-statistics"
													 options:@{ @"wordsPerMinute": @200 }
													   error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text }];
NSDictionary *counts = [statistics runInferenceForRequest:request error:&error].outputs[NFKOutputStructured];
// {"averageWordLength": 4.29, "characters": 93, "lines": 2, "paragraphs": 2, "readingSeconds": 6, "sentences": 3, "words": 17}
```

An executable plugin always runs in its own process. Its manifest sets `allowsInProcess` false, and a
trusted executable plugin's backends fail with `NFKPluginErrorNotTrusted`.

## An MLX family in a plugin

This is the form a plugin takes to run an MLX family: the program links InferKitMLX in its own process,
and its `NFKPlugin` hands out the companion's backends. The manifest lists the models, and their weights
stay outside the bundle, in a download cache, since verification hashes every file the signature seals.

InferKitMLX today pins mlx-swift by revision. The comment in `InferKitMLX/Package.swift` states that
SwiftPM allows a revision requirement only in a root package, so the pin cannot ship, and that a version
requirement replaces it once mlx-swift tags a release carrying MLX core 0.32.
