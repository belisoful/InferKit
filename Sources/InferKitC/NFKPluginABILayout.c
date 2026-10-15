/*
 *  NFKPluginABILayout.c
 *  InferKit
 *
 *  Fixes the layout of every structure in the C headers. A change that moves a field fails the build,
 *  since a plugin or a program compiled against the earlier layout would read the wrong bytes.
 */

#include <stddef.h>

#include "include/InferKitC/NFKPluginABI.h"
#include "include/InferKitC/NFKInferKitC.h"

#if defined(__LP64__) && __LP64__

_Static_assert(NFK_PLUGIN_ABI_VERSION == 0x00010000, "the plugin ABI version is 1.0");

_Static_assert(offsetof(nfk_plugin_buffer, struct_size) == 0, "nfk_plugin_buffer.struct_size");
_Static_assert(offsetof(nfk_plugin_buffer, kind) == 4, "nfk_plugin_buffer.kind");
_Static_assert(offsetof(nfk_plugin_buffer, bytes) == 8, "nfk_plugin_buffer.bytes");
_Static_assert(offsetof(nfk_plugin_buffer, length) == 16, "nfk_plugin_buffer.length");
_Static_assert(offsetof(nfk_plugin_buffer, surface) == 24, "nfk_plugin_buffer.surface");
_Static_assert(offsetof(nfk_plugin_buffer, path) == 32, "nfk_plugin_buffer.path");
_Static_assert(offsetof(nfk_plugin_buffer, descriptor) == 40, "nfk_plugin_buffer.descriptor");
_Static_assert(offsetof(nfk_plugin_buffer, reserved) == 44, "nfk_plugin_buffer.reserved");
_Static_assert(sizeof(nfk_plugin_buffer) == 48, "nfk_plugin_buffer size");
_Static_assert(sizeof(nfk_plugin_buffer) % _Alignof(nfk_plugin_buffer) == 0,
			   "an array of nfk_plugin_buffer strides by struct_size, which keeps every element aligned");

_Static_assert(offsetof(nfk_plugin_payload, struct_size) == 0, "nfk_plugin_payload.struct_size");
_Static_assert(offsetof(nfk_plugin_payload, buffer_count) == 4, "nfk_plugin_payload.buffer_count");
_Static_assert(offsetof(nfk_plugin_payload, json) == 8, "nfk_plugin_payload.json");
_Static_assert(offsetof(nfk_plugin_payload, buffers) == 16, "nfk_plugin_payload.buffers");
_Static_assert(sizeof(nfk_plugin_payload) == 24, "nfk_plugin_payload size");

_Static_assert(offsetof(nfk_plugin_host, struct_size) == 0, "nfk_plugin_host.struct_size");
_Static_assert(offsetof(nfk_plugin_host, abi_version) == 4, "nfk_plugin_host.abi_version");
_Static_assert(offsetof(nfk_plugin_host, context) == 8, "nfk_plugin_host.context");
_Static_assert(offsetof(nfk_plugin_host, inferkit_api_version) == 16, "nfk_plugin_host.inferkit_api_version");
_Static_assert(offsetof(nfk_plugin_host, reserved) == 20, "nfk_plugin_host.reserved");
_Static_assert(offsetof(nfk_plugin_host, options_json) == 24, "nfk_plugin_host.options_json");
_Static_assert(offsetof(nfk_plugin_host, temporary_directory) == 32, "nfk_plugin_host.temporary_directory");
_Static_assert(offsetof(nfk_plugin_host, log) == 40, "nfk_plugin_host.log");
_Static_assert(offsetof(nfk_plugin_host, report_progress) == 48, "nfk_plugin_host.report_progress");
_Static_assert(offsetof(nfk_plugin_host, is_cancelled) == 56, "nfk_plugin_host.is_cancelled");
_Static_assert(offsetof(nfk_plugin_host, output_directory) == 64, "nfk_plugin_host.output_directory");
_Static_assert(sizeof(nfk_plugin_host) == 72, "nfk_plugin_host size");

_Static_assert(offsetof(nfk_plugin, struct_size) == 0, "nfk_plugin.struct_size");
_Static_assert(offsetof(nfk_plugin, abi_version) == 4, "nfk_plugin.abi_version");
_Static_assert(offsetof(nfk_plugin, context) == 8, "nfk_plugin.context");
_Static_assert(offsetof(nfk_plugin, describe) == 16, "nfk_plugin.describe");
_Static_assert(offsetof(nfk_plugin, create_backend) == 24, "nfk_plugin.create_backend");
_Static_assert(offsetof(nfk_plugin, describe_backend) == 32, "nfk_plugin.describe_backend");
_Static_assert(offsetof(nfk_plugin, run) == 40, "nfk_plugin.run");
_Static_assert(offsetof(nfk_plugin, cancel) == 48, "nfk_plugin.cancel");
_Static_assert(offsetof(nfk_plugin, unload) == 56, "nfk_plugin.unload");
_Static_assert(offsetof(nfk_plugin, destroy_backend) == 64, "nfk_plugin.destroy_backend");
_Static_assert(offsetof(nfk_plugin, call_tool) == 72, "nfk_plugin.call_tool");
_Static_assert(offsetof(nfk_plugin, free_payload) == 80, "nfk_plugin.free_payload");
_Static_assert(offsetof(nfk_plugin, free_string) == 88, "nfk_plugin.free_string");
_Static_assert(offsetof(nfk_plugin, shutdown) == 96, "nfk_plugin.shutdown");
_Static_assert(sizeof(nfk_plugin) == 104, "nfk_plugin size");

_Static_assert(offsetof(nfk_buffer, struct_size) == offsetof(nfk_plugin_buffer, struct_size), "nfk_buffer.struct_size");
_Static_assert(offsetof(nfk_buffer, kind) == offsetof(nfk_plugin_buffer, kind), "nfk_buffer.kind");
_Static_assert(offsetof(nfk_buffer, bytes) == offsetof(nfk_plugin_buffer, bytes), "nfk_buffer.bytes");
_Static_assert(offsetof(nfk_buffer, length) == offsetof(nfk_plugin_buffer, length), "nfk_buffer.length");
_Static_assert(offsetof(nfk_buffer, surface) == offsetof(nfk_plugin_buffer, surface), "nfk_buffer.surface");
_Static_assert(offsetof(nfk_buffer, path) == offsetof(nfk_plugin_buffer, path), "nfk_buffer.path");
_Static_assert(offsetof(nfk_buffer, descriptor) == offsetof(nfk_plugin_buffer, descriptor), "nfk_buffer.descriptor");
_Static_assert(offsetof(nfk_buffer, reserved) == offsetof(nfk_plugin_buffer, reserved), "nfk_buffer.reserved");
_Static_assert(sizeof(nfk_buffer) == sizeof(nfk_plugin_buffer), "nfk_buffer size");

_Static_assert(offsetof(nfk_payload, struct_size) == offsetof(nfk_plugin_payload, struct_size), "nfk_payload.struct_size");
_Static_assert(offsetof(nfk_payload, buffer_count) == offsetof(nfk_plugin_payload, buffer_count), "nfk_payload.buffer_count");
_Static_assert(offsetof(nfk_payload, json) == offsetof(nfk_plugin_payload, json), "nfk_payload.json");
_Static_assert(offsetof(nfk_payload, buffers) == offsetof(nfk_plugin_payload, buffers), "nfk_payload.buffers");
_Static_assert(sizeof(nfk_payload) == sizeof(nfk_plugin_payload), "nfk_payload size");

#endif

/*! The plugin ABI version this library was built with. It gives the target's object a symbol, without
	which libtool warns that the object has none. Hidden, so it is no part of the exported interface. */
__attribute__((visibility("hidden"))) const uint32_t NFKPluginABILayoutVersion = NFK_PLUGIN_ABI_VERSION;
