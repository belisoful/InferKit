/*
 *  inferkit_c_consumer.c
 *  InferKit plugin fixtures
 *
 *  A C99 program that calls InferKit through NFKInferKitC.h alone. NFKInferKitCTests builds it under
 *  -std=c99 -pedantic -Werror as a bundle whose InferKit symbols bind to the test process, loads it, and
 *  calls inferkit_c_consumer_run with the test's echo model. It returns 0, or 1 with the failed step in
 *  report.
 */

#include <InferKitC/NFKInferKitC.h>

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int inferkit_c_consumer_run(const char *catalog_identifier, char *report, size_t report_length);

typedef struct consumer_completion {
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	int progress_count;
	int completion_count;
	int succeeded;
} consumer_completion;

static void consumer_progress(void *context, double fraction, const nfk_result *partial)
{
	consumer_completion *state = (consumer_completion *)context;
	(void)fraction;
	(void)partial;
	pthread_mutex_lock(&state->mutex);
	state->progress_count += 1;
	pthread_mutex_unlock(&state->mutex);
}

static void consumer_completed(void *context, nfk_result *result, nfk_error *error)
{
	consumer_completion *state = (consumer_completion *)context;
	pthread_mutex_lock(&state->mutex);
	state->completion_count += 1;
	state->succeeded = result != NULL && error == NULL && strstr(nfk_result_json(result), "c:submitted") != NULL;
	pthread_cond_broadcast(&state->condition);
	pthread_mutex_unlock(&state->mutex);
	nfk_result_free(result);
	nfk_error_free(error);
}

static int consumer_fail(char *report, size_t report_length, const char *step, nfk_error *error)
{
	snprintf(report, report_length, "%s: %s", step, error != NULL ? nfk_error_message(error) : "no error");
	nfk_error_free(error);
	return 1;
}

static int consumer_run_json(nfk_backend *backend, char *report, size_t report_length)
{
	nfk_payload request;
	nfk_error *error = NULL;
	nfk_result *result;
	int found;
	memset(&request, 0, sizeof request);
	request.struct_size = (uint32_t)sizeof request;
	request.json = "{\"inputs\": {\"prompt\": \"hello\"}, \"parameters\": {}, \"outputModality\": 0}";
	result = nfk_backend_run(backend, &request, &error);
	if (result == NULL) {
		return consumer_fail(report, report_length, "run with JSON", error);
	}
	found = strstr(nfk_result_json(result), "c:hello") != NULL;
	nfk_result_free(result);
	return found ? 0 : consumer_fail(report, report_length, "the JSON result names no c:hello", NULL);
}

static int consumer_run_floats(nfk_backend *backend, char *report, size_t report_length)
{
	float values[256];
	nfk_buffer buffer;
	nfk_payload request;
	nfk_error *error = NULL;
	nfk_result *result;
	const nfk_buffer *output;
	const float *inverted;
	size_t index;
	int matches = 1;
	for (index = 0; index < 256; index++) {
		values[index] = (float)index / 255.0f;
	}
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_BYTES;
	buffer.bytes = values;
	buffer.length = sizeof values;
	buffer.descriptor = -1;
	memset(&request, 0, sizeof request);
	request.struct_size = (uint32_t)sizeof request;
	request.buffer_count = 1;
	request.buffers = &buffer;
	request.json = "{\"inputs\": {\"values\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"float32Array\", "
				   "\"length\": 1024, \"count\": 256}}}";
	result = nfk_backend_run(backend, &request, &error);
	if (result == NULL) {
		return consumer_fail(report, report_length, "run with a float buffer", error);
	}
	output = nfk_result_buffer(result, 0);
	if (output == NULL || output->length != sizeof values) {
		nfk_result_free(result);
		return consumer_fail(report, report_length, "the result carries no float buffer", NULL);
	}
	inverted = (const float *)output->bytes;
	for (index = 0; index < 256; index++) {
		matches = matches && inverted[index] == 1.0f - values[index];
	}
	nfk_result_free(result);
	return matches ? 0 : consumer_fail(report, report_length, "the floats are not inverted", NULL);
}

static int consumer_submit(nfk_backend *backend, char *report, size_t report_length)
{
	consumer_completion state;
	nfk_payload request;
	nfk_job *job;
	struct timespec deadline;
	int waited = 0;
	int succeeded;
	memset(&state, 0, sizeof state);
	pthread_mutex_init(&state.mutex, NULL);
	pthread_cond_init(&state.condition, NULL);
	memset(&request, 0, sizeof request);
	request.struct_size = (uint32_t)sizeof request;
	request.json = "{\"inputs\": {\"prompt\": \"submitted\"}}";
	job = nfk_backend_submit(backend, &request, consumer_progress, consumer_completed, &state);
	if (job == NULL) {
		return consumer_fail(report, report_length, "submit", NULL);
	}
	deadline.tv_sec = time(NULL) + 30;
	deadline.tv_nsec = 0;
	pthread_mutex_lock(&state.mutex);
	while (state.completion_count == 0 && waited == 0) {
		waited = pthread_cond_timedwait(&state.condition, &state.mutex, &deadline);
	}
	succeeded = state.completion_count == 1 && state.succeeded;
	pthread_mutex_unlock(&state.mutex);
	nfk_job_free(job);
	pthread_cond_destroy(&state.condition);
	pthread_mutex_destroy(&state.mutex);
	return succeeded ? 0 : consumer_fail(report, report_length, "the submitted run did not complete with c:submitted", NULL);
}

int inferkit_c_consumer_run(const char *catalog_identifier, char *report, size_t report_length)
{
	nfk_error *error = NULL;
	nfk_backend *backend;
	char *description;
	int status;
	if (nfk_api_version() >> 16 != NFK_API_VERSION_MAJOR) {
		return consumer_fail(report, report_length, "another API major version", NULL);
	}
	backend = nfk_backend_create(catalog_identifier, "{\"mode\": \"c\"}", &error);
	if (backend == NULL) {
		return consumer_fail(report, report_length, "create", error);
	}
	description = nfk_backend_copy_description(backend);
	status = description != NULL && strstr(description, "\"identifier\"") != NULL ? 0
		: consumer_fail(report, report_length, "describe", NULL);
	nfk_string_free(description);
	if (status == 0) {
		status = consumer_run_json(backend, report, report_length);
	}
	if (status == 0) {
		status = consumer_run_floats(backend, report, report_length);
	}
	if (status == 0) {
		status = consumer_submit(backend, report, report_length);
	}
	nfk_backend_free(backend);
	if (status == 0) {
		snprintf(report, report_length, "ok");
	}
	return status;
}
