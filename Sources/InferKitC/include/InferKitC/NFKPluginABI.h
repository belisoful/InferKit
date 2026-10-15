/*
 *  NFKPluginABI.h
 *  InferKit
 *
 *  The C interface a plugin library implements. A plugin written in C, C++, Rust, or any language
 *  with a C foreign-function interface includes only this header and links nothing from InferKit.
 */

#ifndef NFKPluginABI_h
#define NFKPluginABI_h

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
	@header		NFKPluginABI.h
	@abstract	The C interface between InferKit and a plugin library.
	@discussion	InferKit loads the library, finds @c nfk_plugin_entry, and calls it with the host's function
				table. The entry returns the plugin's function table. Every later call goes through the two
				tables. The interface carries its own version, @c NFK_PLUGIN_ABI_VERSION_MAJOR and
				@c NFK_PLUGIN_ABI_VERSION_MINOR, apart from the API version of the Objective-C interface.

				Rules both sides follow:
				- A different major version on the other side → the entry returns
				  @c NFK_PLUGIN_STATUS_INCOMPATIBLE.
				- A minor version only appends fields to a structure. A reader uses a field only when
				  @c offsetof(field) + @c sizeof(field) is at most the structure's @c struct_size.
				- An array of buffers is walked with its first element's @c struct_size as the stride,
				  so a reader built against an earlier minor version steps over the fields it does not
				  know. Every element of one array has the same @c struct_size.
				- An optional function pointer that is NULL means the operation is unsupported.
				- Whoever allocates frees. A request payload stays valid until @c run returns. The host
				  frees a plugin's result through @c free_payload and a plugin's string through
				  @c free_string. A partial result passed to @c report_progress is valid during that call.
				- @c run is serialized per backend unless the manifest allows concurrent runs.
				  @c cancel may be called from any thread. Polling @c is_cancelled is enough to honor
				  a cancellation.
				- Nothing calls the host after @c shutdown returns.
				- An error is a UTF-8 JSON object {"message": …, "domain": …, "code": …}; the domain
				  and code are optional.

				A payload's JSON is InferKit's tagged wire form. A value larger than the inline limit
				arrives as {"$nfk": "buffer", "index": i, "as": …} and its bytes are @c buffers[i].
				Docs/agent-reference/plugins.md specifies the form. Introduced in InferKit 0.4.0.
*/

/*! The plugin interface's major version. A plugin built against another major version is refused.
	Introduced in InferKit 0.4.0. */
#define NFK_PLUGIN_ABI_VERSION_MAJOR 1

/*! The plugin interface's minor version. Each minor version appends fields to the structures.
	Introduced in InferKit 0.4.0. */
#define NFK_PLUGIN_ABI_VERSION_MINOR 0

/*! The plugin interface version as one number: the major version in the high 16 bits and the minor
	version in the low 16 bits. Introduced in InferKit 0.4.0. */
#define NFK_PLUGIN_ABI_VERSION ((NFK_PLUGIN_ABI_VERSION_MAJOR << 16) | NFK_PLUGIN_ABI_VERSION_MINOR)

/*! The name of the function InferKit looks up in a plugin library. Introduced in InferKit 0.4.0. */
#define NFK_PLUGIN_ENTRY_SYMBOL "nfk_plugin_entry"

/*! Marks a function the plugin library exports. Introduced in InferKit 0.4.0. */
#if defined(__GNUC__) || defined(__clang__)
#define NFK_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define NFK_PLUGIN_EXPORT
#endif

/*! The outcome of a call. Introduced in InferKit 0.4.0. */
typedef int32_t nfk_plugin_status;

/*!
	@enum		nfk_plugin_status_code
	@abstract	The values of nfk_plugin_status.
	@constant	NFK_PLUGIN_STATUS_OK The call succeeded.
	@constant	NFK_PLUGIN_STATUS_INCOMPATIBLE The other side's major version differs.
	@constant	NFK_PLUGIN_STATUS_INVALID_ARGUMENT An argument is malformed.
	@constant	NFK_PLUGIN_STATUS_MISSING_INPUT The request lacks an input the model needs.
	@constant	NFK_PLUGIN_STATUS_UNSUPPORTED The plugin does not support the operation or the value.
	@constant	NFK_PLUGIN_STATUS_NOT_READY The backend cannot run yet.
	@constant	NFK_PLUGIN_STATUS_FAILED The operation failed. The error JSON says why.
	@constant	NFK_PLUGIN_STATUS_CANCELLED The run ended because it was cancelled.
	@constant	NFK_PLUGIN_STATUS_NO_MEMORY An allocation failed.
	@constant	NFK_PLUGIN_STATUS_UNKNOWN_MODEL The plugin has no model with the identifier.
	@constant	NFK_PLUGIN_STATUS_UNKNOWN_TOOL The plugin has no tool with the name.
	Introduced in InferKit 0.4.0.
*/
enum nfk_plugin_status_code {
	NFK_PLUGIN_STATUS_OK = 0,
	NFK_PLUGIN_STATUS_INCOMPATIBLE = 1,
	NFK_PLUGIN_STATUS_INVALID_ARGUMENT = 2,
	NFK_PLUGIN_STATUS_MISSING_INPUT = 3,
	NFK_PLUGIN_STATUS_UNSUPPORTED = 4,
	NFK_PLUGIN_STATUS_NOT_READY = 5,
	NFK_PLUGIN_STATUS_FAILED = 6,
	NFK_PLUGIN_STATUS_CANCELLED = 7,
	NFK_PLUGIN_STATUS_NO_MEMORY = 8,
	NFK_PLUGIN_STATUS_UNKNOWN_MODEL = 9,
	NFK_PLUGIN_STATUS_UNKNOWN_TOOL = 10
};

/*! The severity of a log message. Introduced in InferKit 0.4.0. */
typedef int32_t nfk_plugin_log_level;

/*!
	@enum		nfk_plugin_log_level_code
	@abstract	The values of nfk_plugin_log_level.
	Introduced in InferKit 0.4.0.
*/
enum nfk_plugin_log_level_code {
	NFK_PLUGIN_LOG_DEBUG = 0,
	NFK_PLUGIN_LOG_INFO = 1,
	NFK_PLUGIN_LOG_WARNING = 2,
	NFK_PLUGIN_LOG_ERROR = 3
};

/*! What a buffer holds. Introduced in InferKit 0.4.0. */
typedef int32_t nfk_plugin_buffer_kind;

/*!
	@enum		nfk_plugin_buffer_kind_code
	@abstract	The values of nfk_plugin_buffer_kind.
	@constant	NFK_PLUGIN_BUFFER_BYTES @c bytes and @c length name memory.
	@constant	NFK_PLUGIN_BUFFER_SURFACE @c surface is an IOSurfaceRef.
	@constant	NFK_PLUGIN_BUFFER_FILE @c path names a file, or @c descriptor is an open file.
	Introduced in InferKit 0.4.0.
*/
enum nfk_plugin_buffer_kind_code {
	NFK_PLUGIN_BUFFER_BYTES = 0,
	NFK_PLUGIN_BUFFER_SURFACE = 1,
	NFK_PLUGIN_BUFFER_FILE = 2
};

/*!
	@struct		nfk_plugin_buffer
	@abstract	One attachment of a payload.
	@field		struct_size The structure's size as its writer compiled it.
	@field		kind What the buffer holds.
	@field		bytes The memory, for NFK_PLUGIN_BUFFER_BYTES.
	@field		length The length of @c bytes in bytes.
	@field		surface An IOSurfaceRef, for NFK_PLUGIN_BUFFER_SURFACE.
	@field		path A UTF-8 file path, for NFK_PLUGIN_BUFFER_FILE, or NULL.
	@field		descriptor An open file descriptor, or -1.
	@field		reserved Zero.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_plugin_buffer {
	uint32_t struct_size;
	nfk_plugin_buffer_kind kind;
	const void *bytes;
	uint64_t length;
	void *surface;
	const char *path;
	int32_t descriptor;
	uint32_t reserved;
} nfk_plugin_buffer;

/*!
	@struct		nfk_plugin_payload
	@abstract	A request or a result: the tagged JSON and the buffers its attachments index.
	@field		struct_size The structure's size as its writer compiled it.
	@field		buffer_count The number of elements of @c buffers.
	@field		json A UTF-8 JSON object in InferKit's tagged wire form.
	@field		buffers The attachments, or NULL when @c buffer_count is 0. Element @c i starts
				@c i times @c buffers[0].struct_size bytes after the first.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_plugin_payload {
	uint32_t struct_size;
	uint32_t buffer_count;
	const char *json;
	const nfk_plugin_buffer *buffers;
} nfk_plugin_payload;

/*! A backend the plugin made. Opaque to the host. Introduced in InferKit 0.4.0. */
typedef struct nfk_plugin_backend nfk_plugin_backend;

/*! One run in progress. Opaque to the plugin. Introduced in InferKit 0.4.0. */
typedef struct nfk_plugin_run nfk_plugin_run;

/*!
	@struct		nfk_plugin_host
	@abstract	The functions and values InferKit gives a plugin.
	@field		struct_size The structure's size as the host compiled it.
	@field		abi_version The host's NFK_PLUGIN_ABI_VERSION.
	@field		context Passed back as the first argument of every function here.
	@field		inferkit_api_version The host library's NFK_API_VERSION.
	@field		reserved Zero.
	@field		options_json The plugin's options as a UTF-8 JSON object.
	@field		temporary_directory A UTF-8 path to a directory the plugin may write.
	@field		log Writes a message to the host's log.
	@field		report_progress Reports a run's progress: a fraction from 0 to 1, or -1 when unknown,
				and an optional partial result valid during the call.
	@field		is_cancelled Returns nonzero once the run is cancelled.
	@field		output_directory Returns a UTF-8 path to the directory a run writes its result files
				to, valid until the run returns.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_plugin_host {
	uint32_t struct_size;
	uint32_t abi_version;
	void *context;
	uint32_t inferkit_api_version;
	uint32_t reserved;
	const char *options_json;
	const char *temporary_directory;
	void (*log)(void *context, nfk_plugin_log_level level, const char *message);
	void (*report_progress)(void *context, nfk_plugin_run *run, double fraction, const nfk_plugin_payload *partial);
	int32_t (*is_cancelled)(void *context, nfk_plugin_run *run);
	const char *(*output_directory)(void *context, nfk_plugin_run *run);
} nfk_plugin_host;

/*!
	@struct		nfk_plugin
	@abstract	The functions a plugin gives InferKit.
	@field		struct_size The structure's size as the plugin compiled it.
	@field		abi_version The plugin's NFK_PLUGIN_ABI_VERSION.
	@field		context Passed back as the first argument of every function here.
	@field		describe Writes a JSON object {"models": […], "tools": […], "options": […]} to
				@c *out_json, freed with @c free_string.
	@field		create_backend Makes a backend for a model with a JSON object of options.
	@field		describe_backend Optional. Writes a JSON object {"inputs", "parameters", "outputs",
				"modelInfo", "loaded", "residentBytes", "peakBytes"} for a backend.
	@field		run Runs a request and writes the result to @c *out_result, freed with
				@c free_payload.
	@field		cancel Optional. Asks a run in progress to end.
	@field		unload Optional. Frees a backend's model and keeps what reloading it needs.
	@field		destroy_backend Frees a backend.
	@field		call_tool Optional. Runs a tool with a JSON object of arguments and writes a JSON tool
				result to @c *out_result_json.
	@field		free_payload Frees a payload the plugin returned.
	@field		free_string Frees a string the plugin returned.
	@field		shutdown Frees the plugin. Nothing calls the plugin afterward.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_plugin {
	uint32_t struct_size;
	uint32_t abi_version;
	void *context;
	nfk_plugin_status (*describe)(void *context, char **out_json);
	nfk_plugin_status (*create_backend)(void *context, const char *model_id, const char *options_json,
										nfk_plugin_backend **out_backend, char **out_error_json);
	nfk_plugin_status (*describe_backend)(void *context, nfk_plugin_backend *backend, char **out_json);
	nfk_plugin_status (*run)(void *context, nfk_plugin_backend *backend, nfk_plugin_run *run,
							 const nfk_plugin_payload *request, nfk_plugin_payload **out_result,
							 char **out_error_json);
	void (*cancel)(void *context, nfk_plugin_backend *backend, nfk_plugin_run *run);
	void (*unload)(void *context, nfk_plugin_backend *backend);
	void (*destroy_backend)(void *context, nfk_plugin_backend *backend);
	nfk_plugin_status (*call_tool)(void *context, const char *tool_name, const char *arguments_json,
								   char **out_result_json, char **out_error_json);
	void (*free_payload)(void *context, nfk_plugin_payload *payload);
	void (*free_string)(void *context, char *string);
	void (*shutdown)(void *context);
} nfk_plugin;

/*! The type of nfk_plugin_entry, for a host that looks it up with dlsym. Introduced in InferKit 0.4.0. */
typedef nfk_plugin_status (*nfk_plugin_entry_function)(uint32_t host_abi_version, const nfk_plugin_host *host,
													   const nfk_plugin **out_plugin);

/*!
	@function	nfk_plugin_entry
	@abstract	The function a plugin library exports under NFK_PLUGIN_ENTRY_SYMBOL.
	@discussion	It checks @c host_abi_version, keeps @c host for the plugin's life, and writes its function
				table to @c *out_plugin. The table stays valid until @c shutdown returns. A different major
				version returns NFK_PLUGIN_STATUS_INCOMPATIBLE. Introduced in InferKit 0.4.0.
*/
NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t host_abi_version, const nfk_plugin_host *host,
													 const nfk_plugin **out_plugin);

#ifdef __cplusplus
}
#endif

#endif /* NFKPluginABI_h */
