# Plugins

List the plugins an app installs in the model catalog, and run their models in a process of their own or
in the app's.

## Overview

A plugin is a `.inferkitplugin` bundle: an Objective-C bundle whose principal class adopts ``NFKPlugin``,
a C library that exports `nfk_plugin_entry`, or a program that speaks the plugin protocol. Its manifest,
read by ``NFKPluginManifest`` without loading code, lists its models as catalog descriptors and its
tools. Plugins are macOS only. The consumer guide, `Docs/plugins.md`, covers writing, signing, and
embedding.

### Listing plugins

``NFKPluginRegistry`` reads each plugin's manifest and lists its models in a catalog as
`<plugin id>/<model id>`, with engine `plugin`, the model's own identifier as its registry name, and
`extensions.plugin` naming the plugin's identifier, version, and kind. A plugin cannot replace a model
the catalog already lists. A bundle that another account could change is refused.

```objc
NSArray<NSError *> *errors = nil;
NSArray<NFKPluginManifest *> *added =
    [NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:pluginsFolder errors:&errors];
```

### Building and running

The registry registers the `plugin` engine's builder, so
``NFKModelCatalog/backendForModelIdentifier:options:error:`` builds a plugin's model after it checks the
options, the machine, and the memory estimate. The backend is an ``NFKPluginBackend``. Its keys come from
the manifest, and its plugin's ``NFKPluginProcess`` starts when it first prepares or runs. The process
exits after its idle interval once no backend holds it, so a server that hosts the model with
``NFKInferenceServer/addBackendProvider:forModelName:`` and unloads it when idle also ends the process.

``NFKPluginRegistry/reloadPluginWithIdentifier:error:`` reads a new version from disk and restarts a
running process with it: runs in flight finish, and new requests wait for the new version.
``NFKPluginRegistry/reloadPluginWithIdentifier:completionHandler:`` reports when the restart ends.

``NFKPluginRegistry/setOptionValues:forPluginIdentifier:error:`` sets a plugin's own options, checked
against its manifest as model options are, and the plugin reads them from ``NFKPluginContext/options``.
A sandboxed app sets ``NFKPluginRegistry/sharedMemoryPrefix`` to its app group, so large values cross in
app-group shared memory.

### Trust

``NFKPluginRegistry/setTrusted:forPluginIdentifier:`` runs a plugin in the app's process from its next
load, through ``NFKPluginLoader``. The app shows ``NFKPluginInProcessWarningNoUnload``,
``NFKPluginInProcessWarningCrashTakesDownApp``, and ``NFKPluginInProcessWarningFullAccess`` first, and
persists the choice itself. An `executable` plugin, or one whose manifest does not allow it, is refused
with ``NFKPluginError/NFKPluginErrorNotTrusted``.

## Topics

### The registry

- ``NFKPluginRegistry``
- ``NFKPluginInProcessWarningNoUnload``
- ``NFKPluginInProcessWarningCrashTakesDownApp``
- ``NFKPluginInProcessWarningFullAccess``

### Running

- ``NFKPluginBackend``
- ``NFKPluginIsolation``
- ``NFKPluginProcess``
- ``NFKPluginProcessState``
- ``NFKPluginSignaturePolicy``
- ``NFKPluginLoader``
- ``NFKLoadedPlugin``

### Writing a plugin

- <doc:CPluginInterface>
- ``NFKPlugin``
- ``NFKPluginContext``
- ``NFKPluginManifest``
- ``NFKPluginEntryKind``
- ``NFKPluginService``
- ``NFKPluginLogLevel``
- ``NFKPluginError``
