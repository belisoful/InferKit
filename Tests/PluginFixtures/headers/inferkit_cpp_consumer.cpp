/*
 *  inferkit_cpp_consumer.cpp
 *  InferKit plugin fixtures
 *
 *  A C++11 program that calls InferKit through NFKInferKitC.h, owning each handle with std::unique_ptr.
 *  NFKInferKitCTests builds it under -std=c++11 -pedantic -Werror as a bundle whose InferKit symbols
 *  bind to the test process, loads it, and calls inferkit_cpp_consumer_run with the test's echo model.
 *  It returns 0, or 1 with the failed step in report.
 */

#include <InferKitC/NFKInferKitC.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {

struct backend_deleter {
	void operator()(nfk_backend *backend) const { nfk_backend_free(backend); }
};

struct result_deleter {
	void operator()(nfk_result *result) const { nfk_result_free(result); }
};

struct error_deleter {
	void operator()(nfk_error *error) const { nfk_error_free(error); }
};

struct job_deleter {
	void operator()(nfk_job *job) const { nfk_job_free(job); }
};

struct string_deleter {
	void operator()(char *string) const { nfk_string_free(string); }
};

typedef std::unique_ptr<nfk_backend, backend_deleter> backend_ptr;
typedef std::unique_ptr<nfk_result, result_deleter> result_ptr;
typedef std::unique_ptr<nfk_error, error_deleter> error_ptr;
typedef std::unique_ptr<nfk_job, job_deleter> job_ptr;
typedef std::unique_ptr<char, string_deleter> string_ptr;

struct completion_state {
	std::mutex mutex;
	std::condition_variable condition;
	int completions = 0;
	std::string json;
};

void on_completion(void *context, nfk_result *result, nfk_error *error)
{
	completion_state *state = static_cast<completion_state *>(context);
	result_ptr owned_result(result);
	error_ptr owned_error(error);
	std::lock_guard<std::mutex> lock(state->mutex);
	state->completions += 1;
	state->json = owned_result ? nfk_result_json(owned_result.get()) : "";
	state->condition.notify_all();
}

int fail(char *report, size_t report_length, const char *step, const error_ptr &error)
{
	std::snprintf(report, report_length, "%s: %s", step, error ? nfk_error_message(error.get()) : "no error");
	return 1;
}

nfk_payload payload_with_json(const char *json)
{
	nfk_payload request;
	std::memset(&request, 0, sizeof request);
	request.struct_size = static_cast<uint32_t>(sizeof request);
	request.json = json;
	return request;
}

}

extern "C" int inferkit_cpp_consumer_run(const char *catalog_identifier, char *report, size_t report_length);

extern "C" int inferkit_cpp_consumer_run(const char *catalog_identifier, char *report, size_t report_length)
{
	nfk_error *raw_error = nullptr;
	backend_ptr backend(nfk_backend_create(catalog_identifier, "{\"mode\": \"cpp\"}", &raw_error));
	error_ptr create_error(raw_error);
	if (!backend) {
		return fail(report, report_length, "create", create_error);
	}
	string_ptr description(nfk_backend_copy_description(backend.get()));
	if (!description || std::strstr(description.get(), "\"identifier\"") == nullptr) {
		return fail(report, report_length, "describe", error_ptr());
	}

	nfk_payload request = payload_with_json("{\"inputs\": {\"prompt\": \"hello\"}}");
	raw_error = nullptr;
	result_ptr result(nfk_backend_run(backend.get(), &request, &raw_error));
	error_ptr run_error(raw_error);
	if (!result || std::string(nfk_result_json(result.get())).find("cpp:hello") == std::string::npos) {
		return fail(report, report_length, "run", run_error);
	}

	completion_state state;
	nfk_payload submitted = payload_with_json("{\"inputs\": {\"prompt\": \"submitted\"}}");
	job_ptr job(nfk_backend_submit(backend.get(), &submitted, nullptr, on_completion, &state));
	if (!job) {
		return fail(report, report_length, "submit", error_ptr());
	}
	bool completed;
	std::string json;
	{
		std::unique_lock<std::mutex> lock(state.mutex);
		completed = state.condition.wait_for(lock, std::chrono::seconds(30), [&state] { return state.completions > 0; });
		json = state.json;
	}
	job.reset();
	if (!completed || state.completions != 1 || json.find("cpp:submitted") == std::string::npos) {
		return fail(report, report_length, "the submitted run did not complete with cpp:submitted", error_ptr());
	}
	std::snprintf(report, report_length, "ok");
	return 0;
}
