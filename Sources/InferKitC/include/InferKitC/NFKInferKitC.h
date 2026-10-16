/*
 *  NFKInferKitC.h
 *  InferKit
 *
 *  The C interface a program calls InferKit through: build a backend from the model catalog, run it,
 *  follow its progress, and cancel it.
 */

#ifndef NFKInferKitC_h
#define NFKInferKitC_h

#include <stddef.h>
#include <stdint.h>

#include "NFKAPIVersion.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
	@header		NFKInferKitC.h
	@abstract	InferKit's C interface for programs written in C, C++, Rust, or another language with a C
				foreign-function interface.
	@discussion	The interface is covered by the API version (@c NFK_API_VERSION_MAJOR,
				@c NFK_API_VERSION_MINOR, @c nfk_api_version()). Docs/c-api.md is the guide.

				Rules:
				- A request's @c nfk_payload structure, its JSON, and its @c buffers array are read during
				  the call that takes them. The memory each buffer points to stays valid until
				  @c nfk_backend_run returns, until the completion of @c nfk_backend_submit fires, or
				  until @c nfk_job_free returns when it comes first. InferKit wraps that memory without
				  copying.
				- A structure's @c struct_size is at least its size at version 1.0. A field a later minor
				  version appends is read only when the writer's @c struct_size covers it.
				- Everything InferKit returns is freed with its own @c *_free function. Each @c *_free
				  function accepts NULL.
				- A function with an @c out_error parameter sets @c *out_error to an error the caller frees
				  when it fails and to NULL when it succeeds. @c out_error may be NULL.
				- Callbacks run on a serial queue of the job's own at user-initiated quality of service,
				  one at a time and in order. @c nfk_backend_submit never calls them from inside itself.
				- A completion fires exactly once, unless @c nfk_job_free comes first. After
				  @c nfk_job_free returns, no callback runs.
				- No function lets an Objective-C exception escape. An exception becomes an error.
				- Every function is safe to call from any thread.

				A payload's JSON is InferKit's tagged wire form, {"inputs": {…}, "parameters": {…},
				"outputModality": n}. A value with bytes travels inline, such as
				{"$nfk": "data", "base64": …}, or as a buffer: {"$nfk": "buffer", "index": i, "as": kind,
				"length": n, …} with its bytes in @c buffers[i]. A result carries every value with bytes
				as a buffer, and InferKit's value classes (detections, keypoints, classifications, …) as
				JSON objects tagged by class, such as {"$nfk": "detection", "boundingBox": {…}}.
				Introduced in InferKit 0.4.0.
*/

/*! What a buffer holds. Introduced in InferKit 0.4.0. */
typedef int32_t nfk_buffer_kind;

/*!
	@enum		nfk_buffer_kind_code
	@abstract	The values of nfk_buffer_kind.
	@constant	NFK_BUFFER_BYTES @c bytes and @c length name memory.
	@constant	NFK_BUFFER_SURFACE @c surface is an IOSurfaceRef, read while locked read-only. @c length,
				when not 0, limits the bytes read.
	@constant	NFK_BUFFER_FILE @c path names a file, or @c descriptor is an open regular file. A file
				named by path is mapped; a descriptor's file is read into memory.
	Introduced in InferKit 0.4.0.
*/
enum nfk_buffer_kind_code {
	NFK_BUFFER_BYTES = 0,
	NFK_BUFFER_SURFACE = 1,
	NFK_BUFFER_FILE = 2
};

/*!
	@struct		nfk_buffer
	@abstract	One attachment of a payload. The same layout as nfk_plugin_buffer.
	@discussion	An array of buffers strides by its first element's @c struct_size, so a caller built
				against a later minor version, whose buffers are longer, is read correctly.
	@field		struct_size The structure's size as its writer compiled it: sizeof(nfk_buffer).
	@field		kind What the buffer holds.
	@field		bytes The memory, for NFK_BUFFER_BYTES.
	@field		length The length of @c bytes in bytes.
	@field		surface An IOSurfaceRef, for NFK_BUFFER_SURFACE.
	@field		path A UTF-8 file path, for NFK_BUFFER_FILE, or NULL.
	@field		descriptor An open file descriptor, or -1.
	@field		reserved Zero.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_buffer {
	uint32_t struct_size;
	nfk_buffer_kind kind;
	const void *bytes;
	uint64_t length;
	void *surface;
	const char *path;
	int32_t descriptor;
	uint32_t reserved;
} nfk_buffer;

/*!
	@struct		nfk_payload
	@abstract	A request: the tagged JSON and the buffers its attachments index. The same layout as
				nfk_plugin_payload.
	@field		struct_size The structure's size as its writer compiled it: sizeof(nfk_payload).
	@field		buffer_count The number of elements of @c buffers.
	@field		json A UTF-8 JSON object {"inputs": {…}, "parameters": {…}, "outputModality": n}.
	@field		buffers The attachments, or NULL when @c buffer_count is 0.
	Introduced in InferKit 0.4.0.
*/
typedef struct nfk_payload {
	uint32_t struct_size;
	uint32_t buffer_count;
	const char *json;
	const nfk_buffer *buffers;
} nfk_payload;

/*! A backend built from the catalog. Introduced in InferKit 0.4.0. */
typedef struct nfk_backend nfk_backend;

/*! An asynchronous run. Introduced in InferKit 0.4.0. */
typedef struct nfk_job nfk_job;

/*! A run's outputs: tagged JSON and the buffers it indexes. Introduced in InferKit 0.4.0. */
typedef struct nfk_result nfk_result;

/*! A failure: a domain, a code, and a message. Introduced in InferKit 0.4.0. */
typedef struct nfk_error nfk_error;

/*! Receives a run's progress, a fraction from 0 to 1 or -1 when unknown, and the run's latest partial
	result when it has one, or NULL. The partial is valid during the call and is not freed by the
	receiver. Introduced in InferKit 0.4.0. */
typedef void (*nfk_progress_callback)(void *context, double fraction, const nfk_result *partial);

/*! Receives a run's end: a result or an error, never both. The receiver owns each and frees it with
	nfk_result_free or nfk_error_free. Introduced in InferKit 0.4.0. */
typedef void (*nfk_completion_callback)(void *context, nfk_result *result, nfk_error *error);

/*! The library's API version: the major version in the high 16 bits and the minor version in the low 16
	bits. Compare it with NFK_API_VERSION. Introduced in InferKit 0.4.0. */
uint32_t nfk_api_version(void);

/*! The library's release version, such as "0.4.0". Static storage; not freed. Introduced in InferKit 0.4.0. */
const char *nfk_release_version(void);

/*! The shared model catalog as a manifest, a UTF-8 JSON object freed with nfk_string_free. NULL with an
	error when it cannot be written. Introduced in InferKit 0.4.0. */
char *nfk_catalog_copy_json(nfk_error **out_error);

/*! Adds every .inferkitplugin bundle directly inside a directory to the shared catalog by reading their
	manifests, and returns the number added. A bundle that fails is skipped, and out_error receives the
	first such failure while the count holds the others. Returns -1 with an error when path is NULL, and
	on iOS and tvOS, which load no plugins. Introduced in InferKit 0.4.0. */
int32_t nfk_plugins_add_directory(const char *path, nfk_error **out_error);

/*! Builds a backend for a catalog identifier, or "model@preset", with a JSON object of options, or NULL
	with an error. options_json may be NULL or empty for no options. The errors are the catalog's, in
	NFKModelCatalogErrorDomain: UnknownModel (5), InvalidOptions (4, also for options that are not a JSON
	object), Unavailable (7), DoesNotFit (8), NoBuilder (6). Free the backend with nfk_backend_free.
	Introduced in InferKit 0.4.0. */
nfk_backend *nfk_backend_create(const char *catalog_identifier, const char *options_json, nfk_error **out_error);

/*! A backend's description, a UTF-8 JSON object {"identifier", "ready", "loaded", "residentBytes",
	"peakBytes", "inputs", "parameters", "outputs", "modelInfo"} freed with nfk_string_free. The key lists
	appear when the backend declares them, and modelInfo keeps only what JSON carries. NULL for a NULL
	backend. Introduced in InferKit 0.4.0. */
char *nfk_backend_copy_description(nfk_backend *backend);

/*! Loads a backend's model ahead of its first run and blocks the calling thread until it is loaded.
	Returns 1, or 0 with an error. A backend with nothing to load returns 1. Introduced in InferKit 0.4.0. */
int32_t nfk_backend_prepare(nfk_backend *backend, nfk_error **out_error);

/*! Frees a backend's model and keeps what reloading it needs. A run in flight finishes first. The next
	run loads the model again. Introduced in InferKit 0.4.0. */
void nfk_backend_unload(nfk_backend *backend);

/*! Frees a backend. A job submitted to it holds it until the job's run ends. Introduced in InferKit 0.4.0. */
void nfk_backend_free(nfk_backend *backend);

/*! Runs a request and blocks the calling thread until it ends. Returns the result, or NULL with an
	error. A request InferKit cannot read fails in NFKInferenceErrorDomain with
	kNFKError_InferenceMissingInput (2). Introduced in InferKit 0.4.0. */
nfk_result *nfk_backend_run(nfk_backend *backend, const nfk_payload *request, nfk_error **out_error);

/*! Starts a run and returns at once. progress may be NULL. completion fires exactly once, on the job's
	queue, also for a request InferKit cannot read. Returns NULL, and starts nothing, when backend,
	request, or completion is NULL. A callback may start on another thread before this function returns.
	Free the job with nfk_job_free. Introduced in InferKit 0.4.0. */
nfk_job *nfk_backend_submit(nfk_backend *backend, const nfk_payload *request, nfk_progress_callback progress,
							nfk_completion_callback completion, void *context);

/*! Asks a run to end. Unless the completion already fired, it fires with a cancellation error
	(NSCocoaErrorDomain, 3072) once the run no longer holds the request's buffers. It never fires inside
	this call. Introduced in InferKit 0.4.0. */
void nfk_job_cancel(nfk_job *job);

/*! Frees a job. A run that has not completed is cancelled, and its completion never fires. nfk_job_free
	then waits until the run no longer reads the request's buffers, so the caller may free them once it
	returns: at once for a run that already returned, and when the backend stops for a run it is still
	in. A backend that does not check for cancellation holds it until the run ends, and InferKit logs a
	fault every 10 seconds of that wait. After it returns no callback runs: it waits for a callback in
	progress on another thread. Called from inside the job's progress callback, it waits the same way
	and that callback is the job's last. Called from inside the completion, it returns at once. It is
	not called on a thread the backend's run waits for. Introduced in InferKit 0.4.0. */
void nfk_job_free(nfk_job *job);

/*! A result's tagged JSON, {"outputs": {…}}, valid until nfk_result_free. Introduced in InferKit 0.4.0. */
const char *nfk_result_json(const nfk_result *result);

/*! The number of buffers a result's JSON indexes. Introduced in InferKit 0.4.0. */
uint32_t nfk_result_buffer_count(const nfk_result *result);

/*! A result's buffer at index, an NFK_BUFFER_BYTES buffer valid until nfk_result_free, or NULL past the
	end. Introduced in InferKit 0.4.0. */
const nfk_buffer *nfk_result_buffer(const nfk_result *result, uint32_t index);

/*! Frees a result. Introduced in InferKit 0.4.0. */
void nfk_result_free(nfk_result *result);

/*! An error's domain, such as "NFKModelCatalogErrorDomain", valid until nfk_error_free. Introduced in
	InferKit 0.4.0. */
const char *nfk_error_domain(const nfk_error *error);

/*! An error's code within its domain. Introduced in InferKit 0.4.0. */
int64_t nfk_error_code(const nfk_error *error);

/*! An error's message, valid until nfk_error_free. Introduced in InferKit 0.4.0. */
const char *nfk_error_message(const nfk_error *error);

/*! An error's details, a UTF-8 JSON object of its user info freed with nfk_string_free, or NULL for a NULL
	error. An error with no details is {}, since the message is not repeated. Each entry keeps its user-info
	key:
	- "NFKModelOptionProblems" → [{"key", "code", "message", "value"}], the options a build refused;
	- "NFKModelMemoryEstimate", "NFKModelMemoryFittingEstimate" → a memory estimate's fields;
	- "NFKModelMemoryFittingOptions" → the options nearest the requested ones that fit;
	- "NSUnderlyingError" → {"domain", "code", "message", "details"}, the cause;
	- strings, numbers, and JSON values → themselves.
	A value of any other type is left out. Introduced in InferKit 0.4.0. */
char *nfk_error_copy_details_json(const nfk_error *error);

/*! Frees an error. Introduced in InferKit 0.4.0. */
void nfk_error_free(nfk_error *error);

/*! Frees a string InferKit returned. Introduced in InferKit 0.4.0. */
void nfk_string_free(char *string);

#ifdef __cplusplus
}
#endif

#endif /* NFKInferKitC_h */
