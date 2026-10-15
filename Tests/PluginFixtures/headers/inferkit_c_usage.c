/*
 *  inferkit_c_usage.c
 *  InferKit plugin fixtures
 *
 *  Uses every declaration of NFKInferKitC.h and NFKAPIVersion.h. NFKCHeaderTests compiles it with
 *  -fsyntax-only as C99 and as C++11, so a declaration that only one language accepts fails the check.
 *  It is never linked.
 */

#include <InferKitC/NFKInferKitC.h>

#include <string.h>

#if NFK_API_VERSION != ((NFK_API_VERSION_MAJOR << 16) | NFK_API_VERSION_MINOR)
#error "NFK_API_VERSION packs the major and minor versions"
#endif

static void usage_progress(void *context, double fraction, const nfk_result *partial)
{
	(void)context;
	(void)fraction;
	(void)nfk_result_json(partial);
}

static void usage_completion(void *context, nfk_result *result, nfk_error *error)
{
	(void)context;
	if (result != NULL) {
		nfk_result_free(result);
	}
	if (error != NULL) {
		nfk_error_free(error);
	}
}

int usage_main(void);

int usage_main(void)
{
	nfk_progress_callback progress = usage_progress;
	nfk_completion_callback completion = usage_completion;
	nfk_buffer_kind kinds[3] = { NFK_BUFFER_BYTES, NFK_BUFFER_SURFACE, NFK_BUFFER_FILE };
	unsigned char bytes[4] = { 1, 2, 3, 4 };
	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = kinds[0];
	buffer.bytes = bytes;
	buffer.length = sizeof bytes;
	buffer.surface = NULL;
	buffer.path = NULL;
	buffer.descriptor = -1;
	buffer.reserved = 0;
	nfk_payload request;
	memset(&request, 0, sizeof request);
	request.struct_size = (uint32_t)sizeof request;
	request.buffer_count = 1;
	request.json = "{\"inputs\":{},\"parameters\":{},\"outputModality\":0}";
	request.buffers = &buffer;

	nfk_error *error = NULL;
	uint32_t version = nfk_api_version();
	const char *release = nfk_release_version();
	char *catalog = nfk_catalog_copy_json(&error);
	int32_t added = nfk_plugins_add_directory("/tmp", &error);
	nfk_backend *backend = nfk_backend_create("com.example.plugin/model", "{}", &error);
	char *description = nfk_backend_copy_description(backend);
	int32_t prepared = nfk_backend_prepare(backend, &error);
	nfk_result *result = nfk_backend_run(backend, &request, &error);
	const char *json = nfk_result_json(result);
	uint32_t count = nfk_result_buffer_count(result);
	const nfk_buffer *first = nfk_result_buffer(result, 0);
	nfk_job *job = nfk_backend_submit(backend, &request, progress, completion, NULL);
	nfk_job_cancel(job);
	nfk_job_free(job);
	nfk_backend_unload(backend);
	nfk_backend_free(backend);
	nfk_result_free(result);
	const char *domain = nfk_error_domain(error);
	int64_t code = nfk_error_code(error);
	const char *message = nfk_error_message(error);
	char *details = nfk_error_copy_details_json(error);
	nfk_error_free(error);
	nfk_string_free(details);
	nfk_string_free(description);
	nfk_string_free(catalog);
	return (int)(version + (uint32_t)added + (uint32_t)prepared + count + (uint32_t)code + (uint32_t)kinds[1] + (uint32_t)kinds[2])
		+ (release != NULL) + (json != NULL) + (first != NULL) + (domain != NULL) + (message != NULL);
}
