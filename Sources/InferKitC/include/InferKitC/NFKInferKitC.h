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
				@c NFK_API_VERSION_MINOR, @c nfk_api_version()).

				Rules:
				- Request buffers the caller passes stay valid until @c nfk_backend_run returns, or until
				  the completion of @c nfk_backend_submit fires. InferKit wraps them without copying.
				- Everything InferKit returns is freed with its own @c *_free function.
				- Callbacks run on a user-initiated queue. @c nfk_backend_submit never calls them before
				  it returns.
				- A completion fires exactly once. After @c nfk_job_free returns, no callback starts.
				- No function lets an Objective-C exception escape. An exception becomes an error.

				A payload's JSON is InferKit's tagged wire form. A value larger than the inline limit
				travels as {"$nfk": "buffer", "index": i, "as": …} with its bytes in @c buffers[i].
				Introduced in InferKit 0.4.0.
*/

/*! What a buffer holds. Introduced in InferKit 0.4.0. */
typedef int32_t nfk_buffer_kind;

/*!
	@enum		nfk_buffer_kind_code
	@abstract	The values of nfk_buffer_kind.
	@constant	NFK_BUFFER_BYTES @c bytes and @c length name memory.
	@constant	NFK_BUFFER_SURFACE @c surface is an IOSurfaceRef.
	@constant	NFK_BUFFER_FILE @c path names a file, or @c descriptor is an open file.
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
	@field		struct_size The structure's size as its writer compiled it.
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
	@field		struct_size The structure's size as its writer compiled it.
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

/*! Receives a run's progress, a fraction from 0 to 1 or -1 when unknown, and an optional partial result
	valid during the call. Introduced in InferKit 0.4.0. */
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

/*! Adds every plugin in a directory to the catalog by reading their manifests. Returns the number added,
	or -1 with an error. Introduced in InferKit 0.4.0. */
int32_t nfk_plugins_add_directory(const char *path, nfk_error **out_error);

/*! Builds a backend for a catalog identifier with a JSON object of options, or NULL with an error. Free
	it with nfk_backend_free. Introduced in InferKit 0.4.0. */
nfk_backend *nfk_backend_create(const char *catalog_identifier, const char *options_json, nfk_error **out_error);

/*! A backend's description, a UTF-8 JSON object {"identifier", "ready", "loaded", "inputs", "parameters",
	"outputs", "modelInfo"} freed with nfk_string_free. Introduced in InferKit 0.4.0. */
char *nfk_backend_copy_description(nfk_backend *backend);

/*! Frees a backend's model and keeps what reloading it needs. Introduced in InferKit 0.4.0. */
void nfk_backend_unload(nfk_backend *backend);

/*! Frees a backend. Introduced in InferKit 0.4.0. */
void nfk_backend_free(nfk_backend *backend);

/*! Runs a request and blocks until it ends. Returns the result, or NULL with an error. Introduced in
	InferKit 0.4.0. */
nfk_result *nfk_backend_run(nfk_backend *backend, const nfk_payload *request, nfk_error **out_error);

/*! Starts a run and returns at once. progress may be NULL. completion fires exactly once. Free the job
	with nfk_job_free. Introduced in InferKit 0.4.0. */
nfk_job *nfk_backend_submit(nfk_backend *backend, const nfk_payload *request, nfk_progress_callback progress,
							nfk_completion_callback completion, void *context);

/*! Asks a run to end. Its completion fires with a cancellation error unless it already fired. Introduced
	in InferKit 0.4.0. */
void nfk_job_cancel(nfk_job *job);

/*! Frees a job. No callback starts after it returns. Introduced in InferKit 0.4.0. */
void nfk_job_free(nfk_job *job);

/*! A result's tagged JSON, {"outputs": {…}}, valid until nfk_result_free. Introduced in InferKit 0.4.0. */
const char *nfk_result_json(const nfk_result *result);

/*! The number of buffers a result's JSON indexes. Introduced in InferKit 0.4.0. */
uint32_t nfk_result_buffer_count(const nfk_result *result);

/*! A result's buffer at index, valid until nfk_result_free, or NULL past the end. Introduced in
	InferKit 0.4.0. */
const nfk_buffer *nfk_result_buffer(const nfk_result *result, uint32_t index);

/*! Frees a result. Introduced in InferKit 0.4.0. */
void nfk_result_free(nfk_result *result);

/*! An error's domain, valid until nfk_error_free. Introduced in InferKit 0.4.0. */
const char *nfk_error_domain(const nfk_error *error);

/*! An error's code. Introduced in InferKit 0.4.0. */
int64_t nfk_error_code(const nfk_error *error);

/*! An error's message, valid until nfk_error_free. Introduced in InferKit 0.4.0. */
const char *nfk_error_message(const nfk_error *error);

/*! Frees an error. Introduced in InferKit 0.4.0. */
void nfk_error_free(nfk_error *error);

/*! Frees a string InferKit returned. Introduced in InferKit 0.4.0. */
void nfk_string_free(char *string);

#ifdef __cplusplus
}
#endif

#endif /* NFKInferKitC_h */
