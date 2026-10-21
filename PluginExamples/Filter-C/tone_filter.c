/*
 *  tone_filter.c
 *  InferKit plugin examples
 *
 *  A plugin written against NFKPluginABI.h alone, in C99, linking nothing from InferKit. It serves two
 *  image filters, "sepia" and "grayscale". Each reads a 32BGRA pixel buffer under "image", blends the
 *  filtered color into the original by the "strength" option (0 keeps the image, 1 filters it fully),
 *  keeps alpha, and returns the filtered image under "image".
 *
 *  What the file shows:
 *  - the request payload: the tagged JSON and the buffers its {"$nfk": "buffer"} entries index, or the
 *    base64 of a small inline image;
 *  - the result payload: a buffer the plugin allocates and the host frees through free_payload;
 *  - progress through host->report_progress and cancellation by polling host->is_cancelled;
 *  - strings the plugin returns, which the host frees through free_string.
 */

#include <InferKitC/NFKPluginABI.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TONE_FORMAT_32BGRA 1111970369.0
#define TONE_MAXIMUM_DEPTH 32
#define TONE_PROGRESS_STEPS 16

/* Whether a structure its writer sized as struct_size holds field. A later minor version of the ABI
   appends fields, so a reader checks before it reads one. */
#define TONE_HAS_FIELD(object, type, field) \
	((size_t)(object)->struct_size >= offsetof(type, field) + sizeof(((type *)0)->field))

#pragma mark JSON

/* A small JSON reader, enough for the request objects the host sends. */

typedef enum tone_json_kind {
	TONE_JSON_NULL,
	TONE_JSON_BOOLEAN,
	TONE_JSON_NUMBER,
	TONE_JSON_STRING,
	TONE_JSON_ARRAY,
	TONE_JSON_OBJECT
} tone_json_kind;

typedef struct tone_json {
	tone_json_kind kind;
	double number;
	char *string;
	size_t count;
	char **keys;
	struct tone_json **items;
} tone_json;

static void json_free(tone_json *value)
{
	if (value == NULL) {
		return;
	}
	for (size_t index = 0; index < value->count; index++) {
		json_free(value->items[index]);
		if (value->keys != NULL) {
			free(value->keys[index]);
		}
	}
	free(value->items);
	free(value->keys);
	free(value->string);
	free(value);
}

static void json_skip_space(const char **cursor)
{
	while (**cursor == ' ' || **cursor == '\t' || **cursor == '\n' || **cursor == '\r') {
		(*cursor)++;
	}
}

static int json_hex_digit(char character)
{
	if (character >= '0' && character <= '9') {
		return character - '0';
	}
	if (character >= 'a' && character <= 'f') {
		return character - 'a' + 10;
	}
	if (character >= 'A' && character <= 'F') {
		return character - 'A' + 10;
	}
	return -1;
}

/* Writes a Basic Multilingual Plane scalar as UTF-8 and returns the bytes written. */
static size_t json_write_utf8(char *out, unsigned scalar)
{
	if (scalar < 0x80) {
		out[0] = (char)scalar;
		return 1;
	}
	if (scalar < 0x800) {
		out[0] = (char)(0xC0 | (scalar >> 6));
		out[1] = (char)(0x80 | (scalar & 0x3F));
		return 2;
	}
	out[0] = (char)(0xE0 | (scalar >> 12));
	out[1] = (char)(0x80 | ((scalar >> 6) & 0x3F));
	out[2] = (char)(0x80 | (scalar & 0x3F));
	return 3;
}

/* Reads a string at the cursor. A \u escape outside the Basic Multilingual Plane is written as its two
   surrogates, three bytes each; the keys and base64 text the host sends hold none. */
static char *json_read_string(const char **cursor)
{
	if (**cursor != '"') {
		return NULL;
	}
	const char *start = ++(*cursor);
	size_t capacity = 1;
	while (**cursor != '"' && **cursor != '\0') {
		*cursor += **cursor == '\\' && (*cursor)[1] != '\0' ? 2 : 1;
		capacity += 3;
	}
	if (**cursor != '"') {
		return NULL;
	}
	char *string = malloc(capacity);
	if (string == NULL) {
		return NULL;
	}
	size_t length = 0;
	for (const char *at = start; at < *cursor; at++) {
		if (*at != '\\') {
			string[length++] = *at;
			continue;
		}
		at++;
		const char *simple = strchr("\"\\/bfnrt", *at);
		if (simple != NULL && *at != '\0') {
			string[length++] = "\"\\/\b\f\n\r\t"[simple - "\"\\/bfnrt"];
			continue;
		}
		unsigned scalar = 0;
		for (int digit = 1; *at == 'u' && digit <= 4; digit++) {
			int value = at + digit < *cursor ? json_hex_digit(at[digit]) : -1;
			if (value < 0) {
				free(string);
				return NULL;
			}
			scalar = scalar << 4 | (unsigned)value;
		}
		if (*at != 'u') {
			free(string);
			return NULL;
		}
		length += json_write_utf8(string + length, scalar);
		at += 4;
	}
	string[length] = '\0';
	(*cursor)++;
	return string;
}

static tone_json *json_read_value(const char **cursor, int depth);

static int json_append(tone_json *container, char *key, tone_json *item)
{
	tone_json **items = realloc(container->items, (container->count + 1) * sizeof *items);
	if (items == NULL) {
		return 0;
	}
	container->items = items;
	if (container->kind == TONE_JSON_OBJECT) {
		char **keys = realloc(container->keys, (container->count + 1) * sizeof *keys);
		if (keys == NULL) {
			return 0;
		}
		container->keys = keys;
		keys[container->count] = key;
	}
	items[container->count++] = item;
	return 1;
}

static int json_read_members(const char **cursor, tone_json *container, char closing, int depth)
{
	json_skip_space(cursor);
	if (**cursor == closing) {
		(*cursor)++;
		return 1;
	}
	for (;;) {
		char *key = NULL;
		if (container->kind == TONE_JSON_OBJECT) {
			json_skip_space(cursor);
			key = json_read_string(cursor);
			json_skip_space(cursor);
			if (key == NULL || **cursor != ':') {
				free(key);
				return 0;
			}
			(*cursor)++;
		}
		tone_json *item = json_read_value(cursor, depth + 1);
		if (item == NULL || !json_append(container, key, item)) {
			free(key);
			json_free(item);
			return 0;
		}
		json_skip_space(cursor);
		if (**cursor == closing) {
			(*cursor)++;
			return 1;
		}
		if (**cursor != ',') {
			return 0;
		}
		(*cursor)++;
	}
}

static tone_json *json_read_value(const char **cursor, int depth)
{
	json_skip_space(cursor);
	if (depth > TONE_MAXIMUM_DEPTH) {
		return NULL;
	}
	tone_json *value = calloc(1, sizeof *value);
	if (value == NULL) {
		return NULL;
	}
	int read = 0;
	char first = **cursor;
	if (first == '{' || first == '[') {
		value->kind = first == '{' ? TONE_JSON_OBJECT : TONE_JSON_ARRAY;
		(*cursor)++;
		read = json_read_members(cursor, value, first == '{' ? '}' : ']', depth);
	} else if (first == '"') {
		value->kind = TONE_JSON_STRING;
		value->string = json_read_string(cursor);
		read = value->string != NULL;
	} else if (strncmp(*cursor, "true", 4) == 0 || strncmp(*cursor, "false", 5) == 0) {
		value->kind = TONE_JSON_BOOLEAN;
		value->number = first == 't';
		*cursor += first == 't' ? 4 : 5;
		read = 1;
	} else if (strncmp(*cursor, "null", 4) == 0) {
		*cursor += 4;
		read = 1;
	} else {
		char *end = NULL;
		value->kind = TONE_JSON_NUMBER;
		value->number = strtod(*cursor, &end);
		read = end != *cursor;
		*cursor = end;
	}
	if (!read) {
		json_free(value);
		return NULL;
	}
	return value;
}

static tone_json *json_parse(const char *text)
{
	if (text == NULL) {
		return NULL;
	}
	tone_json *value = json_read_value(&text, 0);
	json_skip_space(&text);
	if (value != NULL && *text != '\0') {
		json_free(value);
		return NULL;
	}
	return value;
}

static const tone_json *json_member(const tone_json *object, const char *key)
{
	if (object == NULL || object->kind != TONE_JSON_OBJECT) {
		return NULL;
	}
	for (size_t index = 0; index < object->count; index++) {
		if (strcmp(object->keys[index], key) == 0) {
			return object->items[index];
		}
	}
	return NULL;
}

static const char *json_string_member(const tone_json *object, const char *key)
{
	const tone_json *member = json_member(object, key);
	return member != NULL && member->kind == TONE_JSON_STRING ? member->string : NULL;
}

static double json_number_member(const tone_json *object, const char *key, double fallback)
{
	const tone_json *member = json_member(object, key);
	return member != NULL && member->kind == TONE_JSON_NUMBER ? member->number : fallback;
}

#pragma mark Base64

static int base64_value(char character)
{
	static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	const char *found = character != '\0' ? strchr(alphabet, character) : NULL;
	return found != NULL ? (int)(found - alphabet) : -1;
}

/* Decodes base64 into a malloc'd buffer, or returns NULL when the text is malformed. */
static uint8_t *base64_decode(const char *text, size_t *out_length)
{
	size_t length = strlen(text);
	uint8_t *bytes = length % 4 == 0 ? malloc(length / 4 * 3 + 1) : NULL;
	if (bytes == NULL) {
		return NULL;
	}
	size_t written = 0;
	for (size_t index = 0; index < length; index += 4) {
		unsigned group = 0;
		int padding = 0;
		for (int offset = 0; offset < 4; offset++) {
			char character = text[index + offset];
			int value = character == '=' ? 0 : base64_value(character);
			if (value < 0 || (character == '=' && index + 4 != length)) {
				free(bytes);
				return NULL;
			}
			padding += character == '=';
			group = group << 6 | (unsigned)value;
		}
		bytes[written++] = (uint8_t)(group >> 16);
		if (padding < 2) {
			bytes[written++] = (uint8_t)(group >> 8);
		}
		if (padding < 1) {
			bytes[written++] = (uint8_t)group;
		}
	}
	*out_length = written;
	return bytes;
}

#pragma mark Errors and strings

/* Every string the plugin hands the host is malloc'd here and freed by tone_free_string. */
static char *tone_copy_string(const char *string)
{
	size_t length = strlen(string) + 1;
	char *copy = malloc(length);
	if (copy != NULL) {
		memcpy(copy, string, length);
	}
	return copy;
}

/* The message is one of this file's constant strings, already escaped for JSON. */
static nfk_plugin_status tone_fail(char **out_error_json, nfk_plugin_status status, const char *message, long code)
{
	if (out_error_json != NULL) {
		char text[512];
		snprintf(text, sizeof text, "{\"message\":\"%s\",\"domain\":\"NFKInferenceErrorDomain\",\"code\":%ld}", message, code);
		*out_error_json = tone_copy_string(text);
	}
	return status;
}

#pragma mark Plugin state

typedef struct tone_instance {
	nfk_plugin table;
	const nfk_plugin_host *host;
} tone_instance;

typedef enum tone_model {
	TONE_MODEL_SEPIA,
	TONE_MODEL_GRAYSCALE
} tone_model;

struct nfk_plugin_backend {
	tone_model model;
	double strength;
};

/* A result and everything it owns, in one allocation tone_free_payload frees. */
typedef struct tone_result {
	nfk_plugin_payload payload;
	nfk_plugin_buffer buffer;
	uint8_t *pixels;
	char *json;
} tone_result;

/* The image a request carries: its geometry, and its bytes, which owned holds when they were decoded
   from inline base64. */
typedef struct tone_image {
	size_t width;
	size_t height;
	size_t bytes_per_row;
	const uint8_t *pixels;
	uint8_t *owned;
} tone_image;

static const char tone_description[] =
	"{\"models\":["
		"{\"id\":\"sepia\",\"displayName\":\"Sepia\",\"family\":\"tone-filter\",\"engine\":\"plugin\","
		"\"summary\":\"Tones an image in sepia.\",\"tasks\":[\"image-to-image\"],"
		"\"inputs\":[{\"key\":\"image\",\"valueType\":\"image\",\"required\":true}],"
		"\"outputs\":[{\"key\":\"image\",\"valueType\":\"image\",\"required\":true}],"
		"\"options\":[{\"key\":\"strength\",\"kind\":\"number\",\"title\":\"Strength\","
		"\"help\":\"0 keeps the image as it is, and 1 tones it fully.\",\"default\":1,\"minimum\":0,\"maximum\":1,\"step\":0.05}]},"
		"{\"id\":\"grayscale\",\"displayName\":\"Grayscale\",\"family\":\"tone-filter\",\"engine\":\"plugin\","
		"\"summary\":\"Turns an image to shades of gray.\",\"tasks\":[\"image-to-image\"],"
		"\"inputs\":[{\"key\":\"image\",\"valueType\":\"image\",\"required\":true}],"
		"\"outputs\":[{\"key\":\"image\",\"valueType\":\"image\",\"required\":true}],"
		"\"options\":[{\"key\":\"strength\",\"kind\":\"number\",\"title\":\"Strength\","
		"\"help\":\"0 keeps the image as it is, and 1 removes its color fully.\",\"default\":1,\"minimum\":0,\"maximum\":1,\"step\":0.05}]}"
	"],\"tools\":[],\"options\":[]}";

#pragma mark Reading the image

/* The request's buffer at index. The array strides by its first element's struct_size, which a later
   minor version of the ABI lengthens. */
static const nfk_plugin_buffer *tone_request_buffer(const nfk_plugin_payload *request, double index)
{
	if (index < 0 || request->buffers == NULL || index >= request->buffer_count
		|| !TONE_HAS_FIELD(&request->buffers[0], nfk_plugin_buffer, reserved)) {
		return NULL;
	}
	size_t stride = request->buffers[0].struct_size;
	return (const nfk_plugin_buffer *)((const uint8_t *)request->buffers + stride * (size_t)index);
}

/* Reads the 32BGRA image under "image", attached as a buffer or inline as base64. */
static nfk_plugin_status tone_read_image(const tone_json *entry, const nfk_plugin_payload *request, tone_image *image,
										 char **out_error_json)
{
	const char *tag = json_string_member(entry, "$nfk");
	const char *kind = json_string_member(entry, "as");
	int attached = tag != NULL && strcmp(tag, "buffer") == 0 && kind != NULL && strcmp(kind, "pixelBuffer") == 0;
	int inline_image = tag != NULL && strcmp(tag, "pixelBuffer") == 0;
	const tone_json *planes = json_member(entry, "planes");
	const tone_json *plane = planes != NULL && planes->kind == TONE_JSON_ARRAY && planes->count == 1 ? planes->items[0] : NULL;
	double width = json_number_member(entry, "width", 0);
	double height = json_number_member(entry, "height", 0);
	double bytes_per_row = json_number_member(plane, "bytesPerRow", 0);
	double rows = json_number_member(plane, "rows", 0);
	double offset = json_number_member(plane, "offset", 0);
	if ((!attached && !inline_image) || json_number_member(entry, "format", 0) != TONE_FORMAT_32BGRA
		|| width < 1 || height < 1 || bytes_per_row < width * 4 || rows < height || offset < 0) {
		return tone_fail(out_error_json, NFK_PLUGIN_STATUS_UNSUPPORTED, "the filter reads a 32BGRA pixel buffer under \\\"image\\\"", 4);
	}
	const uint8_t *bytes = NULL;
	size_t length = 0;
	if (attached) {
		const nfk_plugin_buffer *buffer = tone_request_buffer(request, json_number_member(entry, "index", -1));
		if (buffer == NULL || buffer->kind != NFK_PLUGIN_BUFFER_BYTES || buffer->bytes == NULL) {
			return tone_fail(out_error_json, NFK_PLUGIN_STATUS_UNSUPPORTED, "the image's buffer is missing or is not memory", 4);
		}
		bytes = buffer->bytes;
		length = (size_t)buffer->length;
	} else {
		const char *base64 = json_string_member(plane, "base64");
		image->owned = base64 != NULL ? base64_decode(base64, &length) : NULL;
		bytes = image->owned;
	}
	if (bytes == NULL || (size_t)offset + (size_t)bytes_per_row * (size_t)height > length) {
		free(image->owned);
		image->owned = NULL;
		return tone_fail(out_error_json, NFK_PLUGIN_STATUS_INVALID_ARGUMENT, "the image's bytes are missing or short", 2);
	}
	image->width = (size_t)width;
	image->height = (size_t)height;
	image->bytes_per_row = (size_t)bytes_per_row;
	image->pixels = bytes + (size_t)offset;
	return NFK_PLUGIN_STATUS_OK;
}

#pragma mark Filtering

static uint8_t tone_clamp(double value)
{
	return value <= 0 ? 0 : (value >= 255 ? 255 : (uint8_t)(value + 0.5));
}

/* Filters one 32BGRA pixel, blended into the original by strength. */
static void tone_filter_pixel(tone_model model, double strength, const uint8_t *from, uint8_t *to)
{
	double blue = from[0];
	double green = from[1];
	double red = from[2];
	double filtered[3];
	if (model == TONE_MODEL_SEPIA) {
		filtered[0] = 0.272 * red + 0.534 * green + 0.131 * blue;
		filtered[1] = 0.349 * red + 0.686 * green + 0.168 * blue;
		filtered[2] = 0.393 * red + 0.769 * green + 0.189 * blue;
	} else {
		double luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
		filtered[0] = luma;
		filtered[1] = luma;
		filtered[2] = luma;
	}
	for (int channel = 0; channel < 3; channel++) {
		to[channel] = tone_clamp(from[channel] + strength * (filtered[channel] - from[channel]));
	}
	to[3] = from[3];
}

/* Filters the image row by row into pixels. Between bands of rows it reports progress and checks for
   cancellation. Returns 0 when the run was cancelled. */
static int tone_filter_rows(const tone_instance *instance, const nfk_plugin_backend *backend, nfk_plugin_run *run,
							const tone_image *image, double strength, uint8_t *pixels)
{
	const nfk_plugin_host *host = instance->host;
	size_t band = image->height / TONE_PROGRESS_STEPS > 0 ? image->height / TONE_PROGRESS_STEPS : 1;
	for (size_t row = 0; row < image->height; row++) {
		if (row % band == 0) {
			if (host->is_cancelled != NULL && host->is_cancelled(host->context, run) != 0) {
				return 0;
			}
			if (host->report_progress != NULL) {
				host->report_progress(host->context, run, (double)row / (double)image->height, NULL);
			}
		}
		const uint8_t *from = image->pixels + row * image->bytes_per_row;
		uint8_t *to = pixels + row * image->bytes_per_row;
		for (size_t column = 0; column < image->width; column++) {
			tone_filter_pixel(backend->model, strength, from + column * 4, to + column * 4);
		}
	}
	return 1;
}

#pragma mark The function table

static nfk_plugin_status tone_describe(void *context, char **out_json)
{
	*out_json = tone_copy_string(tone_description);
	return *out_json != NULL ? NFK_PLUGIN_STATUS_OK : NFK_PLUGIN_STATUS_NO_MEMORY;
}

static nfk_plugin_status tone_create_backend(void *context, const char *model_id, const char *options_json,
											 nfk_plugin_backend **out_backend, char **out_error_json)
{
	*out_backend = NULL;
	int sepia = model_id != NULL && strcmp(model_id, "sepia") == 0;
	int grayscale = model_id != NULL && strcmp(model_id, "grayscale") == 0;
	if (!sepia && !grayscale) {
		return tone_fail(out_error_json, NFK_PLUGIN_STATUS_UNKNOWN_MODEL, "the plugin has no model with that identifier", 14);
	}
	nfk_plugin_backend *backend = calloc(1, sizeof *backend);
	if (backend == NULL) {
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	tone_json *options = json_parse(options_json);
	backend->model = sepia ? TONE_MODEL_SEPIA : TONE_MODEL_GRAYSCALE;
	backend->strength = json_number_member(options, "strength", 1.0);
	json_free(options);
	*out_backend = backend;
	return NFK_PLUGIN_STATUS_OK;
}

static void tone_free_payload(void *context, nfk_plugin_payload *payload)
{
	tone_result *result = (tone_result *)payload;
	if (result == NULL) {
		return;
	}
	free(result->pixels);
	free(result->json);
	free(result);
}

/* Writes the filtered pixels as the result's one buffer, in the same geometry as the request's image. */
static tone_result *tone_make_result(const tone_image *image, uint8_t *pixels)
{
	tone_result *result = calloc(1, sizeof *result);
	char json[512];
	size_t length = image->bytes_per_row * image->height;
	int written = snprintf(json, sizeof json,
						   "{\"outputs\":{\"image\":{\"$nfk\":\"buffer\",\"as\":\"pixelBuffer\",\"index\":0,\"length\":%zu,"
						   "\"format\":%.0f,\"width\":%zu,\"height\":%zu,\"planes\":[{\"offset\":0,\"bytesPerRow\":%zu,\"rows\":%zu}]}}}",
						   length, TONE_FORMAT_32BGRA, image->width, image->height, image->bytes_per_row, image->height);
	if (result == NULL || written < 0 || (size_t)written >= sizeof json || (result->json = tone_copy_string(json)) == NULL) {
		free(result);
		return NULL;
	}
	result->pixels = pixels;
	result->buffer.struct_size = (uint32_t)sizeof result->buffer;
	result->buffer.kind = NFK_PLUGIN_BUFFER_BYTES;
	result->buffer.bytes = pixels;
	result->buffer.length = length;
	result->buffer.descriptor = -1;
	result->payload.struct_size = (uint32_t)sizeof result->payload;
	result->payload.buffer_count = 1;
	result->payload.json = result->json;
	result->payload.buffers = &result->buffer;
	return result;
}

static nfk_plugin_status tone_run(void *context, nfk_plugin_backend *backend, nfk_plugin_run *run,
								  const nfk_plugin_payload *request, nfk_plugin_payload **out_result, char **out_error_json)
{
	const tone_instance *instance = context;
	*out_result = NULL;
	tone_json *body = json_parse(request->json);
	const tone_json *entry = json_member(json_member(body, "inputs"), "image");
	if (entry == NULL) {
		json_free(body);
		return tone_fail(out_error_json, NFK_PLUGIN_STATUS_MISSING_INPUT, "the filter needs an image under \\\"image\\\"", 2);
	}
	double strength = json_number_member(json_member(body, "parameters"), "strength", backend->strength);
	strength = strength < 0 ? 0 : (strength > 1 ? 1 : strength);
	tone_image image = { 0 };
	nfk_plugin_status status = tone_read_image(entry, request, &image, out_error_json);
	json_free(body);
	if (status != NFK_PLUGIN_STATUS_OK) {
		return status;
	}
	uint8_t *pixels = calloc(image.height, image.bytes_per_row);
	if (pixels == NULL) {
		free(image.owned);
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	int finished = tone_filter_rows(instance, backend, run, &image, strength, pixels);
	free(image.owned);
	if (!finished) {
		free(pixels);
		return NFK_PLUGIN_STATUS_CANCELLED;
	}
	tone_result *result = tone_make_result(&image, pixels);
	if (result == NULL) {
		free(pixels);
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	*out_result = &result->payload;
	return NFK_PLUGIN_STATUS_OK;
}

static void tone_destroy_backend(void *context, nfk_plugin_backend *backend)
{
	free(backend);
}

static void tone_free_string(void *context, char *string)
{
	free(string);
}

static void tone_shutdown(void *context)
{
	free(context);
}

NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t host_abi_version, const nfk_plugin_host *host,
													 const nfk_plugin **out_plugin)
{
	if (out_plugin == NULL) {
		return NFK_PLUGIN_STATUS_INVALID_ARGUMENT;
	}
	*out_plugin = NULL;
	if ((host_abi_version >> 16) != NFK_PLUGIN_ABI_VERSION_MAJOR) {
		return NFK_PLUGIN_STATUS_INCOMPATIBLE;
	}
	if (host == NULL || !TONE_HAS_FIELD(host, nfk_plugin_host, output_directory)) {
		return NFK_PLUGIN_STATUS_INVALID_ARGUMENT;
	}
	tone_instance *instance = calloc(1, sizeof *instance);
	if (instance == NULL) {
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	instance->host = host;
	nfk_plugin *table = &instance->table;
	table->struct_size = (uint32_t)sizeof *table;
	table->abi_version = NFK_PLUGIN_ABI_VERSION;
	table->context = instance;
	table->describe = tone_describe;
	table->create_backend = tone_create_backend;
	table->run = tone_run;
	table->destroy_backend = tone_destroy_backend;
	table->free_payload = tone_free_payload;
	table->free_string = tone_free_string;
	table->shutdown = tone_shutdown;
	*out_plugin = table;
	return NFK_PLUGIN_STATUS_OK;
}
