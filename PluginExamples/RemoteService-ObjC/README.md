# Remote Chat: a model served over HTTP

An Objective-C plugin (`entry.kind` `objc`) that serves a chat model from an OpenAI-compatible HTTP
server: a hosted API, or a local server such as llama.cpp, LM Studio, Ollama, or an
`NFKInferenceServer`. The backend is InferKit's own `NFKRemoteBackend`, which speaks the
chat-completions route. The plugin's options name the server, so an app shows them in its settings.

| file | holds |
| --- | --- |
| `RemoteChatPlugin.m` | the principal class `RemoteChatPlugin`, which configures an `NFKRemoteBackend` |
| `inferkit-plugin.json` | the manifest: one `text-generation` model and the three plugin options |
| `Info.plist.in` | the bundle's `Info.plist`, which `build.sh` fills from the manifest |
| `build.sh` | builds, assembles, and signs `build/RemoteChat.inferkitplugin` |

## The model and its options

The manifest declares one model, `chat`, with task `text-generation` and the core text keys:

| | key | value |
| --- | --- | --- |
| input | `prompt` (`NFKInputPrompt`) | the prompt, wrapped as one user message |
| input | `messages` (`NFKInputMessages`) | an OpenAI-style conversation, used as it is |
| output | `text` (`NFKOutputText`), required | the reply |
| request option | `temperature` (`NFKParameterTemperature`) | 0 to 2 |
| request option | `maxTokens` (`NFKParameterMaxTokens`) | at least 1 |

The plugin options configure the server:

| key | kind | default | meaning |
| --- | --- | --- | --- |
| `baseURL` | text | `http://127.0.0.1:8080/v1/` | the server's base URL; the backend appends `chat/completions` |
| `model` | text | empty | the model name the request body carries |
| `apiKey` | text | empty | the bearer token, or empty for a server that takes none |

The backend is `NFKRemoteBackend`, so the rest of its contract reaches the plugin's model too: images
under `NFKInputImage` ride on the last user turn, `NFKParameterTools` becomes the server's function tools,
and `submitInferenceJobForRequest:` streams the reply, which reaches the app as partial results under
`text`.

## Build and sign

```
PluginExamples/RemoteService-ObjC/build.sh
```

## Install

```
ditto build/RemoteChat.inferkitplugin "$HOME/Library/Application Support/MyApp/Plugins/RemoteChat.inferkitplugin"
```

```objc
NSArray<NSError *> *errors = nil;
[NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

## What the app sees

The catalog lists `com.example.inferkit.remote-chat/chat` with task `text-generation`, so an app that
lists `descriptorsForTask:@"text-generation"` lists it beside its other chat models. The manifest's
`options` give the settings form: one text field per option, with its title, help, and default.

The app sets the values, and keeps them itself, since the registry holds them in memory only. An API key
belongs in the keychain, and the app sets it again at launch:

```objc
NSError *error = nil;
BOOL set = [NFKPluginRegistry.sharedRegistry setOptionValues:@{
	@"baseURL": @"http://127.0.0.1:11434/v1/",
	@"model": @"llama3.2",
	@"apiKey": keyFromKeychain,
} forPluginIdentifier:@"com.example.inferkit.remote-chat" error:&error];
```

A running plugin process restarts to receive new values. A plugin the app trusts to run in its own
process keeps the values it loaded with until the app quits.

A run:

```objc
id<NFKInferenceBackend> chat = [NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"com.example.inferkit.remote-chat/chat"
																				 options:@{ @"temperature": @0.7 }
																				   error:&error];
NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Name three primary colors." }];
NFKInferenceResult *result = [chat runInferenceForRequest:request error:&error];
NSString *reply = result.outputs[NFKOutputText];
```

A `baseURL` that is not an `http` or `https` URL fails the build with `kNFKError_InferenceNotReady`.

## Network access

The plugin's process makes the HTTP requests. In a sandboxed app the plugin's process inherits the
app's sandbox ([Sandboxed and unsandboxed apps](../../Docs/plugins.md#sandboxed-and-unsandboxed-apps)),
so the app carries `com.apple.security.network.client`.
