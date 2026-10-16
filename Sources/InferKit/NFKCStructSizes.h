//
//  NFKCStructSizes.h
//  InferKit
//
//  The smallest struct_size each structure of the C headers may carry, and the test for a field a later
//  minor version appends. A reader compares struct_size with these fixed sizes, never with sizeof, so a
//  library built against minor N reads every writer built against minor 0 through N.
//

#import <stddef.h>

#if __has_include(<InferKitC/NFKPluginABI.h>)
#import <InferKitC/NFKInferKitC.h>
#import <InferKitC/NFKPluginABI.h>
#elif __has_include(<InferKit/NFKPluginABI.h>)
#import <InferKit/NFKInferKitC.h>
#import <InferKit/NFKPluginABI.h>
#else
#import "../InferKitC/include/InferKitC/NFKInferKitC.h"
#import "../InferKitC/include/InferKitC/NFKPluginABI.h"
#endif

/*! The byte just past field of type: its offset plus its size. */
#define NFKCStructFieldEnd(type, field) (offsetof(type, field) + sizeof(((type *)0)->field))

/*! Whether a structure its writer sized as object->struct_size holds field. A field a later minor
	version appends is read only when this holds. */
#define NFKCStructCovers(object, type, field) ((size_t)(object)->struct_size >= NFKCStructFieldEnd(type, field))

/*! nfk_buffer and nfk_plugin_buffer at version 1.0, through reserved. */
static const size_t NFKCBufferSize1_0 = NFKCStructFieldEnd(nfk_plugin_buffer, reserved);

/*! nfk_payload and nfk_plugin_payload at version 1.0, through buffers. */
static const size_t NFKCPayloadSize1_0 = NFKCStructFieldEnd(nfk_plugin_payload, buffers);

/*! nfk_plugin at plugin ABI 1.0, through shutdown. */
static const size_t NFKCPluginTableSize1_0 = NFKCStructFieldEnd(nfk_plugin, shutdown);

_Static_assert(NFKCStructFieldEnd(nfk_buffer, reserved) == NFKCStructFieldEnd(nfk_plugin_buffer, reserved),
			   "nfk_buffer and nfk_plugin_buffer share the 1.0 layout");
_Static_assert(NFKCStructFieldEnd(nfk_payload, buffers) == NFKCStructFieldEnd(nfk_plugin_payload, buffers),
			   "nfk_payload and nfk_plugin_payload share the 1.0 layout");
