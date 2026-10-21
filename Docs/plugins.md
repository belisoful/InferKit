# Plugins

A plugin brings models and tools to an app that uses InferKit, without being linked into the app. The
app lists a plugin's models in its model catalog from the plugin's manifest, and builds and runs them as
any other `NFKInferenceBackend`. Plugins are macOS only. `NFKPluginManifest` reads a manifest on every
platform.

[PluginExamples/](../PluginExamples/README.md) in the source tree holds a plugin of each kind to copy: a
new model family and a member of it, a C filter, a tools-only plugin, a remote chat model, and a Swift
executable. Each has a `build.sh` that assembles and signs its bundle.

- [Kinds](#kinds)
- [The bundle](#the-bundle)
- [The manifest](#the-manifest)
- [An Objective-C or Swift plugin](#an-objective-c-or-swift-plugin)
- [A C, C++, or Rust plugin](#a-c-c-or-rust-plugin)
- [An executable plugin](#an-executable-plugin)
- [Signing](#signing)
- [The ownership rule](#the-ownership-rule)
- [Listing plugins and building their models](#listing-plugins-and-building-their-models)
- [Plugin options](#plugin-options)
- [Versions and reloading](#versions-and-reloading)
- [Idle exit and lazy hosting](#idle-exit-and-lazy-hosting)
- [Trust and the app's process](#trust-and-the-apps-process)
- [Embedding the plugin host](#embedding-the-plugin-host)
- [Sandboxed and unsandboxed apps](#sandboxed-and-unsandboxed-apps)
- [Errors](#errors)

## Kinds

A plugin runs in a process of its own by default. Listing it reads its manifest and loads no code. Its
process starts when one of its models first prepares or runs, and exits once it is idle.

| `entry.kind` | what the bundle holds | what runs it |
| --- | --- | --- |
| `objc` | an Objective-C or Swift bundle whose principal class adopts `NFKPlugin` | `inferkit-plugin-host`, or the app's process when trusted |
| `c-abi` | a library that exports `nfk_plugin_entry` from `NFKPluginABI.h` | `inferkit-plugin-host`, or the app's process when trusted |
| `executable` | a program that speaks the plugin protocol itself | the program, always in its own process |

`inferkit-plugin-host` is an executable product of the InferKit package. It loads one `objc` or `c-abi`
plugin and serves its models to the app over a socket. An app that serves those kinds embeds it
([Embedding the plugin host](#embedding-the-plugin-host)).

## The bundle

A plugin is a `.inferkitplugin` bundle:

```
Upscaler.inferkitplugin/
  Contents/
    Info.plist
    MacOS/Upscaler                   the bundle, library, or program
    Resources/inferkit-plugin.json   the manifest
```

- `Info.plist`: `CFBundleIdentifier` equals the manifest's `id`, `CFBundleShortVersionString` equals its
  `version`, `CFBundleExecutable` names the file in `Contents/MacOS`, and `CFBundlePackageType` is `BNDL`.
  An `objc` plugin also names its `NSPrincipalClass`.
- The code signature seals `Contents/Resources`, so the manifest cannot change after signing.
- Verification hashes every file the seal covers on each launch. A plugin keeps large weights outside
  its bundle, in a download cache such as `NFKHFHub`'s.

## The manifest

`Contents/Resources/inferkit-plugin.json` describes the plugin without running it:

```json
{
  "schemaVersion": 1,
  "kind": "inferkit.plugin",
  "id": "com.example.upscaler",
  "version": "1.2.0",
  "displayName": "Example Upscaler",
  "summary": "Enlarges an image four times.",
  "compatibility": {
    "inferkitAPI": "1.0",
    "pluginABI": "1.0",
    "protocol": "1.0",
    "minimumOS": { "macOS": "13.0" },
    "architectures": ["arm64"]
  },
  "entry": { "kind": "c-abi", "executable": "Upscaler", "symbol": "nfk_plugin_entry" },
  "process": { "idleExitSeconds": 300, "allowsInProcess": true, "maximumConcurrentRuns": 1 },
  "options": [{ "key": "tileSize", "kind": "integer", "default": 512 }],
  "models": [{
    "id": "upscale",
    "displayName": "Upscale ×4",
    "family": "example-upscaler",
    "tasks": ["upscaling"],
    "inputs": [{ "key": "image", "valueType": "image", "required": true }],
    "outputs": [{ "key": "image", "valueType": "image", "required": true }],
    "options": [{ "key": "strength", "kind": "number", "default": 0.5, "minimum": 0, "maximum": 1 }]
  }],
  "tools": [{
    "name": "describe_upscaler",
    "description": "Says which scales the upscaler supports.",
    "parameters": { "type": "object", "properties": {} }
  }]
}
```

| field | required | meaning |
| --- | --- | --- |
| `schemaVersion`, `kind` | yes | `1` and `"inferkit.plugin"`. |
| `id` | yes | The plugin's identifier: letters, digits, `.`, `-`, and `_`. |
| `version` | yes | The plugin's version. |
| `displayName`, `summary` | no | What a user interface shows. `displayName` defaults to `id`. |
| `compatibility.inferkitAPI` | yes | The InferKit API version the plugin needs, `"major.minor"`. A host runs it with the same major version and at least that minor version (`NFKInferKit.APIVersion`). |
| `compatibility.pluginABI` | `c-abi` | The C plugin ABI version the library was built against. |
| `compatibility.protocol` | no | The plugin protocol version, `"1.0"` by default. |
| `compatibility.minimumOS`, `compatibility.architectures` | no | The macOS version and the architectures the plugin runs on. |
| `entry.kind` | yes | `objc`, `c-abi`, or `executable`. |
| `entry.executable` | no | The file in `Contents/MacOS`. `CFBundleExecutable` by default. Either name is a relative path with no `.` or `..` component, and the file it names, with symbolic links resolved, lies in the bundle's `Contents/MacOS`; otherwise the bundle is refused with `NFKPluginErrorManifestMismatch` before any code loads or launches. |
| `entry.principalClass` | no | An `objc` plugin's principal class. `NSPrincipalClass` by default. |
| `entry.symbol` | no | A `c-abi` plugin's entry function. `nfk_plugin_entry` by default. |
| `entry.arguments` | no | An `executable` plugin's arguments. |
| `process.idleExitSeconds` | no | How long an idle process runs before it exits. 300 by default. |
| `process.allowsInProcess` | no | Whether the plugin may run in the app's process when the app trusts it. `false` by default. |
| `process.maximumConcurrentRuns` | no | The runs one backend takes at once. 1 by default. |
| `options` | no | The plugin's own settings, as [model options](model-catalog.md#option). |
| `models` | no | The models, as [model catalog descriptors](model-catalog.md#descriptor). The engine is `"plugin"`, and an identifier holds no `/`. |
| `tools` | no | The tools: `name`, `description`, and a JSON Schema `parameters` object. |
| `extensions` | no | Fields of the plugin's own, kept as they are. |

The models the plugin's code describes at its first launch include every model the manifest lists.

## An Objective-C or Swift plugin

The principal class adopts `NFKPlugin`. Its backends are ordinary `NFKInferenceBackend`s:

```objc
@interface UpscalerPlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@end

@implementation UpscalerPlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
    self = [super init];
    if (self != nil) {
        _context = context;   // options, a temporary directory, the host's API version, the log
    }
    return self;
}

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
    NFKModelDescriptor *upscale = [NFKModelDescriptor descriptorWithDictionary:@{
        @"id": @"upscale", @"family": @"example-upscaler", @"engine": @"plugin",
        @"inputs": @[ @{ @"key": NFKInputImage, @"required": @YES } ],
        @"outputs": @[ @{ @"key": NFKOutputImage, @"required": @YES } ] } error:NULL];
    return @[ upscale ];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
                                                      options:(NSDictionary<NSString *, id> *)options
                                                        error:(NSError **)error
{
    return [[UpscalerBackend alloc] initWithOptions:options];
}

@end
```

- `tools`, `pluginOptions`, and `willUnload` are optional.
- A backend that implements `submitInferenceJobForRequest:` reports progress and partial results, and
  cancelling the app's job cancels it. `unload`, `isLoaded`, `residentByteCount`, and `peakByteCount`
  reach the app.
- The bundle binds to the InferKit already loaded in the process that loads it, so it embeds no copy of
  InferKit, and no class in it carries the `NFK` prefix. A bundle that defines an `NFK` class is refused
  with `NFKPluginErrorDuplicateRuntime`.
- `context.runsInHostProcess` is YES when the plugin runs in the app's process.

Build it from the InferKit source root:

```
xcrun clang -bundle -fobjc-arc -undefined dynamic_lookup \
    -I Sources/InferKit/include -I Sources/InferKitC/include \
    -arch arm64 -mmacosx-version-min=13.0 \
    -framework Foundation -framework CoreVideo \
    -o Upscaler.inferkitplugin/Contents/MacOS/Upscaler UpscalerPlugin.m
```

A Swift plugin's principal class is an `@objc` class that adopts `NFKPlugin`, and its bundle follows
the same rules.

## A C, C++, or Rust plugin

The library includes `NFKPluginABI.h` alone and links nothing from InferKit. It exports one function,
which checks the host's ABI version, keeps the host's function table, and returns its own:

```c
#include <InferKitC/NFKPluginABI.h>

static nfk_plugin table;
static const nfk_plugin_host *upscaler_host;   /* valid for the plugin's life */

NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t host_abi_version, const nfk_plugin_host *host,
                                                     const nfk_plugin **out_plugin)
{
    if ((host_abi_version >> 16) != NFK_PLUGIN_ABI_VERSION_MAJOR) {
        return NFK_PLUGIN_STATUS_INCOMPATIBLE;
    }
    upscaler_host = host;
    table.struct_size = sizeof table;
    table.abi_version = NFK_PLUGIN_ABI_VERSION;
    table.context = NULL;                     /* passed back to every function */
    table.describe = upscaler_describe;       /* {"models": [...], "tools": [...], "options": [...]} */
    table.create_backend = upscaler_create_backend;
    table.run = upscaler_run;                 /* request {"inputs", "parameters"} → result {"outputs"} */
    table.destroy_backend = upscaler_destroy_backend;
    table.free_payload = upscaler_free_payload;
    table.free_string = upscaler_free_string;
    table.shutdown = upscaler_shutdown;
    *out_plugin = &table;
    return NFK_PLUGIN_STATUS_OK;
}
```

- Requests and results are payloads: a JSON object in InferKit's tagged wire form and the buffers its
  `{"$nfk": "buffer", "index": n}` entries name. A large image arrives as a buffer of pixel rows.
- Whoever allocates frees. A request is valid until `run` returns. The host frees a result with
  `free_payload` and a string with `free_string`.
- `run` reports progress with `host->report_progress`, polls `host->is_cancelled`, and writes result files
  to `host->output_directory`. `cancel`, `unload`, `describe_backend`, and `call_tool` are optional.
- A later minor version of the ABI only appends fields. A reader uses a field only when its offset plus
  its size is at most `struct_size`. InferKit accepts any `struct_size` at least the structure's size at
  ABI 1.0, so a plugin built against a later minor version loads.
- An array of buffers strides by its first element's `struct_size`. A plugin reads a request's buffer
  `i` at `(const uint8_t *)buffers + i * buffers[0].struct_size`.

Build it:

```
xcrun clang -dynamiclib -std=c99 -fvisibility=hidden -I Sources/InferKitC/include \
    -arch arm64 -mmacosx-version-min=13.0 -install_name @rpath/Upscaler \
    -o Upscaler.inferkitplugin/Contents/MacOS/Upscaler upscaler.c
```

A C++ library includes the same header, which carries `extern "C"` guards. A Rust crate builds a
`cdylib` whose `#[no_mangle] pub extern "C" fn nfk_plugin_entry` follows the header's layout.

## An executable plugin

An `executable` plugin is a program in any language. The app launches it with a Unix socket on
descriptor 3 (`INFERKIT_PLUGIN_CHANNEL_FD=3`) and speaks the plugin protocol over it:

1. The program creates a socket pair, sends one end to the app on descriptor 3 as the frame
   `{"type": "channel"}` with that descriptor attached, and speaks on the other end. It keeps both open
   until descriptor 3 reads end of file.
2. Each frame is a 12-byte header (the magic `0x504B464E` "NFKP", the JSON length, the descriptor count,
   and the flags, little-endian) and a UTF-8 JSON object with a `"type"`. Descriptors travel with the
   frame's first byte as `SCM_RIGHTS`, exactly as many as the header counts. A frame that carries more
   or fewer ends the channel with `NFKPluginErrorProtocolViolation`, and the app closes every descriptor
   it received.
3. The app sends `hello`, `describe`, `build`, `prepare`, `run`, `unload`, `release`, `status`,
   `callTool`, `ping`, and `shutdown`, each with an integer `id`, and the program answers each with
   `{"type": "reply", "id": n}` carrying `"result"` or `"error"`. During a run the program sends
   `progress`. `cancel` names a run, and that run answers `"cancelled": true`.

The sample in `Tests/PluginFixtures/executable/sample_executable.m` is a complete program that uses
Foundation alone. An executable plugin always runs in its own process.

A program that links InferKit serves its plugin with `NFKPluginService` instead of speaking the protocol
itself. `-initWithManifest:pluginFactory:` makes the plugin when the app's hello arrives, with the context
an `objc` plugin receives, its option values included, and `-runWithChannelDescriptor:` serves it on
descriptor 3.

## Signing

Sign the bundle after its manifest and `Info.plist` are in place:

- Distribution: `codesign --force --timestamp --options runtime --sign "Developer ID Application: <Team>
  (<TEAMID>)" Upscaler.inferkitplugin`.
- Development: `codesign --force --sign - Upscaler.inferkitplugin`. This ad hoc signature loads only under
  the development policy.
- Check: `codesign --verify --strict Upscaler.inferkitplugin`.

An app chooses which signatures it accepts with `NFKPluginSignaturePolicy`:

| policy | accepts |
| --- | --- |
| `+defaultPolicy` | a valid signature that is not ad hoc |
| `+developmentPolicy` | a valid signature, ad hoc included, for building plugins on the developer's machine |
| `teamIdentifier` set | a signature by that team, proved by the certificate chain (`NFKPluginErrorTeamMismatch` otherwise). A team identifier is ten uppercase letters and digits; another value refuses every signer. |
| `requirementString` set | a signature that also satisfies that code requirement |

The registry checks the signature when a plugin first launches or loads, and caches a success by the
code's hash and its files, so a later edit to the bundle is checked again.

The registry's `signaturePolicy` covers every plugin, and a plugin can have its own:
`setSignaturePolicy:forPluginIdentifier:` sets one, and `addPluginsInDirectoryURL:signaturePolicy:errors:`
gives each plugin of a folder the folder's. An app keeps installed plugins under the default policy
pinned to their team and a folder of local builds under the development policy:

```objc
NFKPluginRegistry *registry = NFKPluginRegistry.sharedRegistry;
[registry addPluginsInDirectoryURL:installedPluginsURL errors:NULL];
[registry addPluginsInDirectoryURL:developerPluginsURL
                   signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy errors:NULL];
```

`+[NFKPluginSignaturePolicy signingTeamIdentifierOfCodeAtURL:error:]` reads the team that signed a bundle
without loading it, so an app records the team at install and refuses an update another team signed.
It answers nil, with no error, for a bundle signed ad hoc or not at all.

## The ownership rule

A plugin is refused when another account could change it between its signature check and its load. The
folder that holds the bundle, which is the plugins folder an app passes to
`addPluginsInDirectoryURL:errors:`, the bundle, and everything inside the bundle must each belong to the
user or to root and be writable by neither the group nor others. A plugin that breaks the rule fails
with `NFKPluginErrorLoadFailed`, naming the path and the reason ("is writable by its group"). A folder
the app makes inside its Application Support folder meets the rule.

## Listing plugins and building their models

`NFKPluginRegistry` lists plugins in a model catalog. `sharedRegistry` lists them in
`NFKModelCatalog.sharedCatalog`:

```objc
NFKPluginRegistry *plugins = NFKPluginRegistry.sharedRegistry;
NSArray<NSError *> *errors = nil;
for (NFKPluginManifest *manifest in [plugins addPluginsInDirectoryURL:pluginsFolder errors:&errors]) {
    for (NFKModelDescriptor *model in manifest.models) {
        // the catalog lists it as "com.example.upscaler/upscale"
    }
}
NSError *error = nil;
id<NFKInferenceBackend> upscaler =
    [NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"com.example.upscaler/upscale"
                                                     options:@{ @"strength": @0.8 }
                                                       error:&error];
```

- Each model joins the catalog as `<plugin id>/<model id>`, with engine `"plugin"`, `registryName` the
  model's own identifier, and `extensions.plugin` holding `{"id", "version", "kind"}`.
- A plugin cannot replace a model the catalog already lists: such a plugin fails with
  `NFKPluginErrorIdentifierInUse` and lists none of its models. Two bundles with one plugin identifier
  are refused the same way.
- The catalog's build entry checks the options against the descriptor, the machine against its
  requirements, and the memory estimate against the budget, as it does for any model. The load options
  reach the plugin's build, and the request options become defaults the returned `NFKDefaultedBackend`
  fills into each request.
- `-[NFKPluginRegistry backendForModelIdentifier:options:error:]` returns the `NFKPluginBackend` itself,
  with the options passed through unchecked.
- An `NFKPluginBackend` answers its keys and model information from the manifest. Its process starts
  when it prepares or runs.
- `tools` lists every plugin's tools from the manifests. Calling one starts its plugin. A tool name is
  unique across the listed plugins: a plugin that declares a tool another listed plugin declares fails
  with `NFKPluginErrorIdentifierInUse`, naming the tool and the plugin that holds it. A reload of the same
  plugin keeps its own names.
- `removePluginWithIdentifier:` removes the plugin's models. Its process stops once no backend holds it.

## Plugin options

The manifest's top-level `options` are the plugin's own settings. The app sets their values:

```objc
NSError *error = nil;
if (![registry setOptionValues:@{ @"tileSize": @"256" } forPluginIdentifier:@"com.example.upscaler" error:&error]) {
    NSArray<NFKModelOptionProblem *> *problems = error.userInfo[NFKModelOptionProblemsKey];
}
NSDictionary *values = [registry optionValuesForPluginIdentifier:@"com.example.upscaler"];   // @{ @"tileSize": @256 }
```

- The values resolve against the manifest's options as model options do: a string converts to the
  option's kind, and a choice takes its value, written as text or as the value itself, not its title. A value that does not resolve fails with
  `NFKModelCatalogErrorInvalidOptions` in `NFKModelCatalogErrorDomain`, its problems under
  `NFKModelOptionProblemsKey` with the model options' codes, and changes nothing.
- The plugin reads the values from `-[NFKPluginContext options]`, and a C plugin from its host table's
  `options_json`, with the manifest's `default` for each option the app left unset. An executable plugin
  reads them from the context its `NFKPluginService` hands the factory of
  `initWithManifest:pluginFactory:`. A plugin's process receives them at its next start, and a running process restarts to
  receive them.
- A plugin already loaded into the app's process keeps the values it was loaded with until the app quits.
- The registry keeps the values in memory, through a reload or a removal. The app persists them and sets
  them again at launch, after it adds the plugin.
- A reload, or adding the plugin again, checks each value against the new manifest. A key the manifest no
  longer declares is dropped, and so is a value that no longer resolves, such as one above a lowered
  maximum. The rest reach the plugin's next start. The app reads `optionValuesForPluginIdentifier:` after
  a reload to persist what remains.

## Versions and reloading

`reloadPluginWithIdentifier:error:` reads the manifest from the bundle again, after the app replaced the
bundle with a new version. The catalog's descriptors follow the new manifest. A running process
restarts:

1. New requests wait, in order.
2. Runs in flight finish on the old version, up to `NFKPluginProcess.restartDrainTimeout` (120 s), and are
   cancelled after it.
3. The old process stops, and the new version starts and serves the waiting requests.

A backend made before the reload builds its model again on the new version before its next run. A
process that is not running starts the new version on its next request.

`reloadPluginWithIdentifier:error:` returns once the catalog follows the new manifest, while the restart
is still under way. `reloadPluginWithIdentifier:completionHandler:` reports the end of the restart:

```objc
[registry reloadPluginWithIdentifier:@"com.example.upscaler" completionHandler:^(NSError *error) {
    // nil once the new version is ready, or at once when no process runs; else why the reload or restart failed
}];
```

## Idle exit and lazy hosting

A plugin's process exits once it has had no run, no request, and no live backend for the manifest's
`idleExitSeconds`. Memory pressure shortens the interval as it shortens `NFKInferenceServer`'s idle
unload. The next request starts the process again.

A server hosts a plugin's model lazily. The model is listed at once and built on its first request:

```objc
NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
server.idleUnloadInterval = 600;
[server addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
    return [NFKPluginRegistry.sharedRegistry backendForModelIdentifier:@"com.example.upscaler/upscale"
                                                               options:nil error:error];
} forModelName:@"upscaler"];
```

`addModelWithIdentifier:options:catalog:forModelName:error:` hosts it through the catalog's build entry
instead. The server's idle unload releases the backend, and the plugin's process then exits after its
own idle interval.

## Status, start, and stop

The registry reports each plugin's status, whatever started it: a backend's run, a tool call, a
server's request, or the app.

- `statusForPluginIdentifier:` → `NFKPluginStatus`: `processState` (not running, starting, ready,
  restarting, failed), `failureError`, `processIdentifier`, `launchDate`, the `version` it runs, and its
  `isolation`. A trusted plugin reports ready once its code is in the app's process.
- `statusHandler` and `NFKPluginRegistryStatusDidChangeNotification` deliver every change on a
  background queue, a signature refused at a start included.
- `startPluginWithIdentifier:error:` (or `…completionHandler:`) starts the plugin ahead of its first
  request and runs its signature checks then, so a refusal shows when the user loads the plugin.
- `stopPluginWithIdentifier:completionHandler:` stops it now: requests made from the call on wait, the
  runs in flight get `restartDrainTimeout` to finish, and the process ends. The next request starts it
  again. A plugin in the app's process cannot stop (`NFKPluginErrorInProcessUnloadable`).

```objc
registry.statusHandler = ^(NFKPluginStatus *status) {
    // status.processState, status.failureError, status.launchDate: on a background queue
};
NSError *error = nil;
if (![registry startPluginWithIdentifier:@"com.example.upscaler" error:&error]) {
    // a refused signature, or a process that does not start
}
[registry stopPluginWithIdentifier:@"com.example.upscaler" completionHandler:^(NSError *error) { }];
```

## Trust and the app's process

A plugin the app trusts runs in the app's process: `[registry setTrusted:YES
forPluginIdentifier:@"com.example.upscaler"]`. Its code then loads into the app on the plugin's next
load, after its signature passes the registry's policy. Before trusting a plugin, the app shows the user
what that means:

| constant | text |
| --- | --- |
| `NFKPluginInProcessWarningNoUnload` | A plugin that runs in the app's process stays loaded until the app quits. |
| `NFKPluginInProcessWarningCrashTakesDownApp` | A crash in a plugin that runs in the app's process ends the app. |
| `NFKPluginInProcessWarningFullAccess` | A plugin that runs in the app's process has every access the app has. |

- The registry keeps trust in memory. The app persists the choice and sets it again at launch.
- A trusted plugin runs in the app's process only when its manifest sets `allowsInProcess`. An
  `executable` plugin never does. Either is refused with `NFKPluginErrorNotTrusted`, its backends and
  its tools alike.
- The plugin sees the same protocol, manifest, and options either way. `NFKPluginBackend.isolation`
  says where it runs.
- Code in the app's process stays until the app quits. A reload to another version of such a plugin
  fails with `NFKPluginErrorLoadFailed` until then.
- An app that loads trusted `objc` plugins keeps InferKit's symbols exported, since the plugin binds to
  them: link the InferKit dynamic framework, or set `STRIP_STYLE = non-global` on a static link.

## Embedding the plugin host

An app that serves `objc` or `c-abi` plugins in their own processes embeds `inferkit-plugin-host`:

1. Build it: `swift build -c release --product inferkit-plugin-host`.
2. Copy it to `MyApp.app/Contents/Helpers/inferkit-plugin-host`.
3. Sign it with its entitlements, before the app, so the app's seal covers the signed helper:

   ```
   codesign --force --timestamp --options runtime \
       --entitlements Sources/inferkit-plugin-host/Entitlements/inferkit-plugin-host.entitlements \
       --sign "Developer ID Application: <Team> (<TEAMID>)" \
       MyApp.app/Contents/Helpers/inferkit-plugin-host
   ```

4. Sign the app last.

The registry finds the host in `Contents/Helpers`, then among the main bundle's auxiliary executables,
then beside the main executable, then beside the image that contains InferKit.
`NFKPluginRegistry.hostExecutableURL` names another. The registry checks the host's signature under its
policy pinned to the app's own team, since the app's team signs the host, and without the policy's
requirement, which names the plugins' signer. An app that names no team, and a policy that accepts ad hoc
signatures, pin no team on the host.

`inferkit-plugin-host.entitlements` carries `com.apple.security.cs.disable-library-validation`, so the
hardened host loads plugin code another team signed.

An `objc` plugin binds by name to the InferKit in the process that loads it, so that process keeps every
public InferKit constant and function, the ones it never uses itself included. A release link
dead-strips those, so the host calls `NFKPluginRetainPublicInterface()` at launch, which references
them all. An app that links InferKit statically and trusts plugins to run in its process calls it once
too. An app that embeds InferKit as a dynamic framework keeps every public symbol already.

## Sandboxed and unsandboxed apps

| | unsandboxed app | sandboxed app |
| --- | --- | --- |
| host entitlements | `inferkit-plugin-host.entitlements` | `inferkit-plugin-host-sandboxed.entitlements` (`app-sandbox`, `inherit`) |
| plugin process | outside any sandbox | inherits the app's sandbox |
| `objc`, `c-abi` from the app's team | yes | yes |
| `objc`, `c-abi` from another team | yes | re-signed by the app's team, or shipped as `executable` |
| `executable` | yes | yes |

A helper that inherits the sandbox carries only those two entitlements, so the hardened runtime's
library validation admits only code the app's own team signed. A sandboxed app that serves another
team's plugin code either re-signs the plugin with its own team or takes the plugin as an `executable`,
which runs its own signed program.

A sandboxed app sets the registry's `sharedMemoryPrefix` to its app group, followed by a slash:

```objc
NFKPluginRegistry.sharedRegistry.sharedMemoryPrefix = @"ABCDE12345.ik/";   // the app group "ABCDE12345.ik"
```

- Every plugin process takes the prefix at its next start. Its hello offers the `file-descriptors`
  capability and the prefix.
- Large values cross in shared memory named under the app group, which the sandbox allows both
  processes. A region that cannot be made falls back to an unlinked temporary file.
- Files cross as open descriptors beside their paths when the plugin accepts the capability, so a
  plugin reads a file the app opened without access to its path.
- A shared-memory name holds at most 31 characters, and the prefix leaves the rest for random
  characters. With fewer than 10 left, every region takes the temporary-file fallback, so the app group
  that carries plugin traffic has a short identifier: "ABCDE12345.ik/" leaves 17.
- An unsandboxed app leaves the prefix nil, the default, and hello offers no capability.

## What the completeness test proves

`NFKPluginCompletenessTests` builds one plugin of each kind (`objc`, `c-abi`, `executable`) from
`Tests/PluginFixtures/` and walks each end to end through `NFKPluginRegistry`:

- The catalog lists all three from their manifests with no plugin code loaded and no process running.
- Each kind, in its own process, inverts a 64 × 64 image and 4096 floats, which cross inline, and a
  256 × 256 image and 65536 floats, which cross as shared-memory attachments. An `NFKDetection` crosses
  to the plugin and back as tagged JSON.
- Each kind reports progress with partial results, cancels a run and keeps serving, answers a tool call,
  unloads its model, and exits once idle with no backend left.
- Each kind restarts to a second version while a run is in flight; the run finishes on the first version
  and the next run reaches the second.
- The trusted `objc` and `c-abi` plugins give in the app's process the outputs they give in their own; an
  `executable` plugin is refused there.
- Each kind refuses a bundle edited after signing and a signature from a team other than the pinned one.
- The C API builds a plugin model with `nfk_backend_create("<plugin id>/<model id>", …)` and runs it
  through the plugin's process.
- A server that hosts each kind with `addBackendProvider:forModelName:` and unloads it with
  `unloadIdleModels` ends the plugin's process.

## Errors

Every plugin error is in `NFKPluginErrorDomain`:

| code | when |
| --- | --- |
| `MalformedManifest`, `UnsupportedManifestVersion`, `ManifestMismatch` | the manifest is malformed, newer than this release reads, or disagrees with its bundle |
| `Incompatible` | the plugin needs an API, ABI, protocol, system version, or architecture this host lacks |
| `SignatureInvalid`, `TeamMismatch` | the signature fails the policy |
| `LoadFailed` | the code does not load, or the bundle breaks the ownership rule |
| `DuplicateRuntime` | an `objc` plugin carries its own copy of InferKit |
| `LaunchFailed`, `HandshakeFailed`, `TimedOut` | the process does not start, greet, or answer in time |
| `ProcessExited`, `Stopped` | the process ended before the operation finished, during its hello included, with its exit status or signal and the end of its standard error |
| `UnknownModel`, `UnknownTool`, `UnknownPlugin` | nothing is listed under the identifier |
| `IdentifierInUse` | the catalog already lists one of the plugin's catalog identifiers, another bundle holds its plugin identifier, or another plugin declares one of its tools |
| `NotTrusted` | the plugin cannot run in the app's process |
| `InProcessUnloadable` | a stop asked for a plugin loaded into the app's process |

A plugin's own error reaches the app with its domain and code.
