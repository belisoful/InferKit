# The C plugin interface

Write a plugin library in C, C++, Rust, or any language with a C foreign-function interface.

## Overview

`NFKPluginABI.h` is the one header a `c-abi` plugin includes, in the `InferKitC` target. The library
links nothing from InferKit and exports one function, ``nfk_plugin_entry``. InferKit calls it with the
host's function table, ``nfk_plugin_host``, and receives the plugin's, ``nfk_plugin``. Every later call
goes through the two tables. The interface carries its own version, ``NFK_PLUGIN_ABI_VERSION``, apart
from the API version of the Objective-C and C interfaces.

```c
#include <InferKitC/NFKPluginABI.h>

static nfk_plugin table;
static const nfk_plugin_host *upscaler_host;

NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t host_abi_version, const nfk_plugin_host *host,
                                                     const nfk_plugin **out_plugin)
{
    if ((host_abi_version >> 16) != NFK_PLUGIN_ABI_VERSION_MAJOR) {
        return NFK_PLUGIN_STATUS_INCOMPATIBLE;
    }
    upscaler_host = host;
    table.struct_size = sizeof table;
    table.abi_version = NFK_PLUGIN_ABI_VERSION;
    table.describe = upscaler_describe;
    table.create_backend = upscaler_create_backend;
    table.run = upscaler_run;
    table.destroy_backend = upscaler_destroy_backend;
    table.free_payload = upscaler_free_payload;
    table.free_string = upscaler_free_string;
    table.shutdown = upscaler_shutdown;
    *out_plugin = &table;
    return NFK_PLUGIN_STATUS_OK;
}
```

The rules both sides follow:

- A minor version only appends structure fields. A reader uses a field only when its offset plus its size
  is at most the structure's `struct_size`, and walks an array of buffers with its first element's
  `struct_size` as the stride.
- A NULL optional function pointer means the operation is unsupported.
- Whoever allocates frees. A request stays valid until `run` returns; the host frees a result with
  `free_payload` and a string with `free_string`; a partial result passed to `report_progress` is valid
  during that call.
- A request and a result are an ``nfk_plugin_payload``: InferKit's tagged JSON and the buffers its
  `{"$nfk": "buffer", "index": i}` entries name. InferKit's value classes, such as a detection, cross as
  JSON objects tagged by class in both directions.

The same plugin runs in its own process under the plugin host or, when the app trusts it, in the app's
process. <doc:Plugins> describes listing, trust, and hosting.

## Topics

### Entering the plugin

- ``nfk_plugin_entry``
- ``nfk_plugin_entry_function``
- ``NFK_PLUGIN_ENTRY_SYMBOL``
- ``NFK_PLUGIN_EXPORT``

### Versions

- ``NFK_PLUGIN_ABI_VERSION``
- ``NFK_PLUGIN_ABI_VERSION_MAJOR``
- ``NFK_PLUGIN_ABI_VERSION_MINOR``

### The two tables

- ``nfk_plugin``
- ``nfk_plugin_host``

### Payloads

- ``nfk_plugin_payload``
- ``nfk_plugin_buffer``
- ``nfk_plugin_buffer_kind``
- ``nfk_plugin_buffer_kind_code``

### Status and logging

- ``nfk_plugin_status``
- ``nfk_plugin_status_code``
- ``nfk_plugin_log_level``
- ``nfk_plugin_log_level_code``
