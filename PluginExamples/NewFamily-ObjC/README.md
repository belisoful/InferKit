# Color Palette: a new model family

An Objective-C plugin (`entry.kind` `objc`) that introduces the model family `color-palette`. Each model
reads an image and returns its dominant colors, found by k-means clustering over a sample of its pixels.
It needs no weights and no network.

- Two models: `palette-5` and `palette-8`, the five and the eight dominant colors.
- A per-model option, `samples`: how many pixels the clustering reads.
- A plugin option, `order`: whether each palette lists the most common color first or the darkest first.
- The family contract below, which another plugin implements to join the family.

| file | holds |
| --- | --- |
| `ColorPalettePlugin.m` | the principal class `ColorPalettePlugin` and the backend `ColorPaletteKMeansBackend` |
| `inferkit-plugin.json` | the manifest: the models, their option, and the plugin option |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds, assembles, and signs `build/ColorPalette.inferkitplugin` |

## The family contract

The catalog groups models by their `family` (`-[NFKModelCatalog descriptorsInFamily:]`). Every member of
`color-palette` reads and writes the same keys, so an app swaps one member for another without changing
its request. A member declares:

| | key | value |
| --- | --- | --- |
| family | `"family": "color-palette"` | |
| task | `"tasks": ["palette-extraction"]` | |
| input | `image` (`NFKInputImage`), required | a 32BGRA `CVPixelBuffer` |
| output | `structured` (`NFKOutputStructured`), required | an `NSDictionary`, below |
| request parameter | `samples`, an integer option | how many pixels the model reads, 4096 by default |

The structured output holds two fields:

- `colors`: an array, one dictionary per color, each with `hex` (`"#RRGGBB"`), `red`, `green`, and `blue`
  (integers from 0 to 255), and `share` (the fraction of the read pixels the color stands for, from 0
  to 1). The array holds at most the model's number of colors, and fewer when the image holds fewer
  distinct colors. It lists the most common color first, unless a member's own setting orders it
  another way.
- `sampledPixels`: the number of pixels read. A pixel whose alpha is below 128 is skipped.

```json
{"colors": [{"hex": "#C81E1E", "red": 200, "green": 30, "blue": 30, "share": 0.375},
            {"hex": "#FAFAFA", "red": 250, "green": 250, "blue": 250, "share": 0.25}],
 "sampledPixels": 4096}
```

A member in any language meets the same contract. A `c-abi` or `executable` member receives the image as
the wire coding's `pixelBuffer` entry and returns the structured value as a JSON object
([Docs/c-api.md](../../Docs/c-api.md)). [ExistingFamily-ObjC](../ExistingFamily-ObjC/) is a member
written with another algorithm.

## How the plugin is written

- `modelDescriptors` and `pluginOptions` return the manifest's models and options, read with
  `+[NFKPluginManifest manifestWithBundleURL:error:]` from `[NSBundle bundleForClass:self.class]`. The
  manifest is the one place they are declared, and the models the plugin describes at its first launch
  match the manifest's.
- `-[NFKPluginContext options]` carries only the values the app set. `optionValueForKey:` falls back to
  the manifest's default for the rest.
- `samples` arrives two ways. The catalog's build entry fills a request option into each request's
  parameters, and `-[NFKPluginRegistry backendForModelIdentifier:options:error:]` hands its options to
  the build. The backend reads the request's parameter first and the build's value second.
- The backend holds no state, so the manifest allows two runs at once (`maximumConcurrentRuns`).

## Build and sign

```
PluginExamples/NewFamily-ObjC/build.sh
```

The script compiles a universal (`arm64`, `x86_64`) bundle with `-undefined dynamic_lookup`, so every
InferKit symbol binds to the InferKit already in the process that loads it, writes `Info.plist` and the
manifest into the bundle, and signs it ad hoc with `codesign --force --sign -`. An ad hoc bundle loads
only under `+[NFKPluginSignaturePolicy developmentPolicy]`; [the index](../README.md#signing) gives the
Developer ID command.

## Install

Copy the bundle into the app's plugins folder, and list the folder:

```
ditto build/ColorPalette.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/ColorPalette.inferkitplugin"
```

```objc
NFKPluginRegistry *plugins = NFKPluginRegistry.sharedRegistry;
plugins.signaturePolicy = [NFKPluginSignaturePolicy developmentPolicy];
NSArray<NSError *> *errors = nil;
[plugins addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

## What the app sees

The catalog lists two models, each with engine `plugin`, family `color-palette`, task
`palette-extraction`, and the option `samples`:

- `com.example.inferkit.color-palette/palette-5`
- `com.example.inferkit.color-palette/palette-8`

The plugin's own setting `order` is a choice between `share` ("Most common first") and `lightness`
("Darkest first"). The app shows it in its settings and sets it by value:

```objc
NSError *error = nil;
[plugins setOptionValues:@{ @"order": @"lightness" } forPluginIdentifier:@"com.example.inferkit.color-palette" error:&error];
```

A run:

```objc
id<NFKInferenceBackend> palette =
	[NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"com.example.inferkit.color-palette/palette-8"
													 options:@{ @"samples": @8192 }
													   error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)pixelBuffer }];
NFKInferenceResult *result = [palette runInferenceForRequest:request error:&error];
NSArray<NSDictionary *> *colors = result.outputs[NFKOutputStructured][@"colors"];
```

The plugin's process starts with the first run. It exits once it has had no run and no backend for the
manifest's `idleExitSeconds`, 300.
