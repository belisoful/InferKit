# Everyday Tools: a tools-only plugin

An Objective-C plugin (`entry.kind` `objc`) that offers two tools and no models. A language model calls
them through an agent loop, and an MCP client calls them through a server.

- `convert_units` converts a measurement between units of one kind: length, mass, volume, or
  temperature, with Foundation's `NSMeasurement`.
- `calculate_date` adds days to a date (`add-days`) or counts the days between two dates
  (`days-between`), in the Gregorian calendar.
- The plugin option `decimalPlaces` rounds converted values, 4 by default.

| file | holds |
| --- | --- |
| `EverydayToolsPlugin.m` | the principal class `EverydayToolsPlugin` and each tool's code |
| `inferkit-plugin.json` | the manifest: each tool's name, description, and JSON Schema, and the plugin option |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds, assembles, and signs `build/EverydayTools.inferkitplugin` |

## The tools

The manifest declares each tool. An app lists the tools from the manifest without starting the plugin,
and a model reads the description and the schema to decide when and how to call one:

```json
{
  "name": "calculate_date",
  "description": "Calculates with calendar dates in the Gregorian calendar. …",
  "parameters": {
    "type": "object",
    "properties": {
      "operation": { "type": "string", "enum": ["add-days", "days-between"], "description": "What to calculate." },
      "date": { "type": "string", "description": "The first date, YYYY-MM-DD." },
      "days": { "type": "integer", "description": "For add-days: the days to add." },
      "otherDate": { "type": "string", "description": "For days-between: the second date, YYYY-MM-DD." }
    },
    "required": ["operation", "date"]
  }
}
```

The plugin's `tools` method builds one `NFKInferenceTool` per declaration in its own manifest and
attaches the handler of that name, so the declaration the app lists and the one the plugin runs are the
same. A handler returns an `NFKToolResult`: `text` for the model, and `outputs` for the app, here the
value under `NFKOutputStructured`. A handler that fails returns an error, which reaches the model as an
error result it can correct.

```
convert_units {"value": 10, "from": "km", "to": "mi"}                         → 10 km is 6.2137 mi
calculate_date {"operation": "add-days", "date": "2026-10-20", "days": 45}    → 2026-12-04, a Friday
calculate_date {"operation": "days-between", "date": "2026-10-20", "otherDate": "2027-01-01"}   → 73 days
```

The plugin has no models, so `modelDescriptors` is empty and `backendForModelIdentifier:options:error:`
answers `NFKPluginErrorUnknownModel`.

## Build and sign

```
PluginExamples/Tools-ObjC/build.sh
```

## Install

```
ditto build/EverydayTools.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/EverydayTools.inferkitplugin"
```

```objc
NSArray<NSError *> *errors = nil;
[NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

## What the app sees

The catalog lists no model for this plugin. `NFKPluginRegistry.sharedRegistry.tools` lists
`convert_units` and `calculate_date`, read from the manifest. Listing them starts nothing, and calling
one starts the plugin's process, or loads the plugin when the app trusts it. A tool name is unique among
the listed plugins, so a second plugin that declares `convert_units` is refused with
`NFKPluginErrorIdentifierInUse`.

An agent gets the tools with any chat backend that calls tools:

```objc
NSArray<NFKInferenceTool *> *tools = NFKPluginRegistry.sharedRegistry.tools;
NFKToolLoop *agent = [[NFKToolLoop alloc] initWithBackend:chatBackend tools:tools];
NSError *error = nil;
NFKInferenceRequest *question = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"How many miles is a 10 km run?" }];
NFKInferenceResult *answer = [agent runInferenceForRequest:question error:&error];
NSString *text = answer.outputs[NFKOutputText];
```

A server offers them to MCP clients at `POST /mcp`:

```objc
NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
for (NFKInferenceTool *tool in NFKPluginRegistry.sharedRegistry.tools) {
	[server addTool:tool];
}
```

The app sets the plugin option by its key. The plugin's running process restarts to receive it:

```objc
[NFKPluginRegistry.sharedRegistry setOptionValues:@{ @"decimalPlaces": @2 }
							  forPluginIdentifier:@"com.example.inferkit.everyday-tools"
											error:&error];
```
