# Median-Cut Palette: joining a family

An Objective-C plugin (`entry.kind` `objc`) whose model joins a family that exists: `color-palette`,
which [NewFamily-ObjC](../NewFamily-ObjC/) introduces. The model `median-cut-8` meets the family's
contract with another algorithm. Median cut sorts the sampled colors along their widest channel,
splits them at the median, and repeats on the box with the widest range until it holds eight boxes. Each
box's mean is one palette color.

| file | holds |
| --- | --- |
| `MedianCutPalettePlugin.m` | the principal class `MedianCutPalettePlugin` and the backend `MedianCutPaletteBackend` |
| `inferkit-plugin.json` | the manifest: one model in the `color-palette` family |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds, assembles, and signs `build/MedianCutPalette.inferkitplugin` |

## Joining the family

A model joins a family by declaring the family's name and its contract in its manifest entry. Nothing
else links the two plugins: either installs and runs without the other.

```json
{
  "id": "median-cut-8",
  "family": "color-palette",
  "tasks": ["palette-extraction"],
  "inputs": [{ "key": "image", "valueType": "image", "required": true }],
  "outputs": [{ "key": "structured", "valueType": "structured", "required": true }],
  "options": [{ "key": "samples", "kind": "integer", "default": 4096, "minimum": 64, "maximum": 262144 }]
}
```

The backend returns the contract's fields: `colors`, each with `hex`, `red`, `green`, `blue`, and
`share`, most common first, and `sampledPixels`. A member may add settings of its own. This one declares
no plugin options, and the `order` option of NewFamily-ObjC belongs to that plugin alone.

## Build and sign

```
PluginExamples/ExistingFamily-ObjC/build.sh
```

The script builds a universal bundle that binds to the InferKit of the process that loads it, writes its
`Info.plist` and manifest, and signs it ad hoc. [The index](../README.md) covers signing for
distribution and the signature policies.

## Install

```
ditto build/MedianCutPalette.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/MedianCutPalette.inferkitplugin"
```

```objc
NSArray<NSError *> *errors = nil;
[NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

## What the app sees

With both palette plugins installed, the catalog lists three members of `color-palette`:

- `com.example.inferkit.color-palette/palette-5`
- `com.example.inferkit.color-palette/palette-8`
- `com.example.inferkit.median-cut-palette/median-cut-8`

The app asks the catalog for the family and treats every member alike: one request, one option, one
result shape. A picker of palette methods is the list below, and a member a later plugin adds joins it
with no change to the app.

```objc
NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)pixelBuffer }];
for (NFKModelDescriptor *member in [catalog descriptorsInFamily:@"color-palette"]) {
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [catalog backendForModelIdentifier:member.identifier
																 options:@{ @"samples": @4096 }
																   error:&error];
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	NSArray<NSDictionary *> *colors = result.outputs[NFKOutputStructured][@"colors"];
	NSLog(@"%@: %@", member.displayName, [colors valueForKey:@"hex"]);
}
```

## Joining a family InferKit ships

A plugin joins one of InferKit's own families the same way: it declares the family's name, its task, and
the core keys the family's models read and write. A chat model declares the task `text-generation`, the
family of the architecture it runs, such as `qwen3`, and the core text keys: `prompt` (`NFKInputPrompt`)
and `messages` (`NFKInputMessages`) in, `text` (`NFKOutputText`) out.

```json
{
  "id": "qwen3-local",
  "family": "qwen3",
  "tasks": ["text-generation"],
  "inputs": [{ "key": "prompt", "valueType": "text" }, { "key": "messages", "valueType": "messages" }],
  "outputs": [{ "key": "text", "valueType": "text", "required": true }]
}
```

An app that lists `descriptorsForTask:@"text-generation"` or `descriptorsInFamily:@"qwen3"` then lists
the plugin's model beside InferKit's, and runs it with the request it sends any chat model. The plugin's
model keeps its own catalog identifier, `<plugin id>/qwen3-local`, so it replaces none of InferKit's.
[RemoteService-ObjC](../RemoteService-ObjC/) is a complete text-generation plugin.
