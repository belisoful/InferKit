/*
 *  sample_plugin.c
 *  InferKit plugin fixtures
 *
 *  A plugin written against NFKPluginABI.h alone, in C99, linking nothing from InferKit. It serves two
 *  models and one tool:
 *  - "invert" inverts a 32BGRA image under "image" (each color byte becomes 255 minus itself, alpha
 *    kept) and a float32 or float64 array under "values" (each element becomes 1 minus itself).
 *    A detection under "region", in the wire coding's tagged JSON, comes back under "detections", its
 *    box mirrored left to right (x becomes 1 - x - width) and its confidence replaced by 1 minus itself.
 *  - "slow" counts "steps" steps of "stepMilliseconds" each, reports progress with a partial result
 *    {"step": n} after each, and stops when the run is cancelled. "exitAtStep" ends the process at that
 *    step, which only a plugin running in a process of its own survives.
 *  - "echo" returns the "text" argument after the plugin's "echoPrefix" option.
 *  Every result carries "pluginVersion". Every allocation is counted, and
 *  sample_plugin_live_allocations() reads the count, so a test can prove the plugin frees what it
 *  allocates.
 */

#include <InferKitC/NFKPluginABI.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef SAMPLE_PLUGIN_VERSION
#define SAMPLE_PLUGIN_VERSION "1.0.0"
#endif

#define SAMPLE_FORMAT_32BGRA 1111970369.0
#define SAMPLE_WEIGHT_BYTES 4096u
#define SAMPLE_MAXIMUM_DEPTH 64

/* Whether a structure its writer sized as struct_size holds field. */
#define SAMPLE_HAS_FIELD(object, type, field) \
	((size_t)(object)->struct_size >= offsetof(type, field) + sizeof(((type *)0)->field))

#pragma mark Counted allocation

static int64_t sample_live_allocation_count;

static void *sample_allocate(size_t size)
{
	void *memory = calloc(1, size > 0 ? size : 1);
	if (memory != NULL) {
		__atomic_add_fetch(&sample_live_allocation_count, 1, __ATOMIC_SEQ_CST);
	}
	return memory;
}

static void *sample_reallocate(void *memory, size_t size)
{
	void *grown = realloc(memory, size > 0 ? size : 1);
	if (grown != NULL && memory == NULL) {
		__atomic_add_fetch(&sample_live_allocation_count, 1, __ATOMIC_SEQ_CST);
	}
	return grown;
}

static void sample_free(void *memory)
{
	if (memory == NULL) {
		return;
	}
	__atomic_sub_fetch(&sample_live_allocation_count, 1, __ATOMIC_SEQ_CST);
	free(memory);
}

static char *sample_copy_string(const char *string)
{
	size_t length = strlen(string);
	char *copy = sample_allocate(length + 1);
	if (copy != NULL) {
		memcpy(copy, string, length + 1);
	}
	return copy;
}

/* The number of the plugin's allocations not yet freed. */
NFK_PLUGIN_EXPORT int64_t sample_plugin_live_allocations(void)
{
	return __atomic_load_n(&sample_live_allocation_count, __ATOMIC_SEQ_CST);
}

#pragma mark Text

typedef struct sample_text {
	char *bytes;
	size_t length;
	size_t capacity;
	int failed;
} sample_text;

static void text_append(sample_text *text, const char *bytes, size_t length)
{
	if (text->failed) {
		return;
	}
	if (text->length + length + 1 > text->capacity) {
		size_t capacity = text->capacity > 0 ? text->capacity : 256;
		while (capacity < text->length + length + 1) {
			capacity *= 2;
		}
		char *grown = sample_reallocate(text->bytes, capacity);
		if (grown == NULL) {
			text->failed = 1;
			return;
		}
		text->bytes = grown;
		text->capacity = capacity;
	}
	memcpy(text->bytes + text->length, bytes, length);
	text->length += length;
	text->bytes[text->length] = '\0';
}

static void text_append_string(sample_text *text, const char *string)
{
	text_append(text, string, strlen(string));
}

static void text_append_format(sample_text *text, const char *format, ...)
{
	char buffer[128];
	va_list arguments;
	va_start(arguments, format);
	int length = vsnprintf(buffer, sizeof buffer, format, arguments);
	va_end(arguments);
	if (length < 0 || (size_t)length >= sizeof buffer) {
		text->failed = 1;
		return;
	}
	text_append(text, buffer, (size_t)length);
}

static void text_append_json_string(sample_text *text, const char *string)
{
	text_append(text, "\"", 1);
	for (const unsigned char *cursor = (const unsigned char *)string; *cursor != '\0'; cursor++) {
		if (*cursor == '"' || *cursor == '\\') {
			char escaped[2] = { '\\', (char)*cursor };
			text_append(text, escaped, 2);
		} else if (*cursor < 0x20) {
			text_append_format(text, "\\u%04x", *cursor);
		} else {
			text_append(text, (const char *)cursor, 1);
		}
	}
	text_append(text, "\"", 1);
}

/* Hands the text's bytes to the caller, or frees them and returns NULL when an append failed. */
static char *text_take(sample_text *text)
{
	if (text->failed || text->bytes == NULL) {
		sample_free(text->bytes);
		text->bytes = NULL;
		return NULL;
	}
	char *bytes = text->bytes;
	text->bytes = NULL;
	return bytes;
}

static char *sample_error_json(const char *message, const char *domain, long code)
{
	sample_text text = { 0 };
	text_append_string(&text, "{\"message\":");
	text_append_json_string(&text, message);
	text_append_string(&text, ",\"domain\":");
	text_append_json_string(&text, domain);
	text_append_format(&text, ",\"code\":%ld}", code);
	return text_take(&text);
}

static nfk_plugin_status sample_fail(char **out_error_json, nfk_plugin_status status, const char *message,
									 const char *domain, long code)
{
	if (out_error_json != NULL) {
		*out_error_json = sample_error_json(message, domain, code);
	}
	return status;
}

#pragma mark Base64

static const char sample_base64_alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void base64_append(sample_text *text, const uint8_t *bytes, size_t length)
{
	for (size_t index = 0; index < length; index += 3) {
		uint32_t group = (uint32_t)bytes[index] << 16;
		size_t remaining = length - index;
		if (remaining > 1) {
			group |= (uint32_t)bytes[index + 1] << 8;
		}
		if (remaining > 2) {
			group |= bytes[index + 2];
		}
		char quad[4] = {
			sample_base64_alphabet[(group >> 18) & 0x3F],
			sample_base64_alphabet[(group >> 12) & 0x3F],
			remaining > 1 ? sample_base64_alphabet[(group >> 6) & 0x3F] : '=',
			remaining > 2 ? sample_base64_alphabet[group & 0x3F] : '=',
		};
		text_append(text, quad, 4);
	}
}

static int base64_value(char character)
{
	const char *found = character != '\0' ? strchr(sample_base64_alphabet, character) : NULL;
	return found != NULL ? (int)(found - sample_base64_alphabet) : -1;
}

/* Decodes base64 into a counted allocation, or returns NULL when the text is malformed. */
static uint8_t *base64_decode(const char *string, size_t *out_length)
{
	size_t length = strlen(string);
	if (length % 4 != 0) {
		return NULL;
	}
	uint8_t *bytes = sample_allocate(length / 4 * 3);
	if (bytes == NULL) {
		return NULL;
	}
	size_t written = 0;
	for (size_t index = 0; index < length; index += 4) {
		int values[4];
		int padding = 0;
		for (int offset = 0; offset < 4; offset++) {
			char character = string[index + offset];
			values[offset] = character == '=' ? 0 : base64_value(character);
			padding += character == '=';
			if (values[offset] < 0 || (character == '=' && index + 4 != length)) {
				sample_free(bytes);
				return NULL;
			}
		}
		uint32_t group = ((uint32_t)values[0] << 18) | ((uint32_t)values[1] << 12) | ((uint32_t)values[2] << 6) | (uint32_t)values[3];
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

#pragma mark JSON

typedef enum sample_json_kind {
	SAMPLE_JSON_NULL,
	SAMPLE_JSON_BOOLEAN,
	SAMPLE_JSON_NUMBER,
	SAMPLE_JSON_STRING,
	SAMPLE_JSON_ARRAY,
	SAMPLE_JSON_OBJECT
} sample_json_kind;

typedef struct sample_json {
	sample_json_kind kind;
	double number;
	char *string;
	size_t count;
	char **keys;
	struct sample_json **items;
} sample_json;

typedef struct sample_json_reader {
	const char *cursor;
	int depth;
} sample_json_reader;

static void json_free(sample_json *value)
{
	if (value == NULL) {
		return;
	}
	for (size_t index = 0; index < value->count; index++) {
		json_free(value->items[index]);
		if (value->keys != NULL) {
			sample_free(value->keys[index]);
		}
	}
	sample_free(value->items);
	sample_free(value->keys);
	sample_free(value->string);
	sample_free(value);
}

static void json_skip_space(sample_json_reader *reader)
{
	while (*reader->cursor == ' ' || *reader->cursor == '\t' || *reader->cursor == '\n' || *reader->cursor == '\r') {
		reader->cursor++;
	}
}

static int json_hex4(const char *cursor, uint32_t *out_value)
{
	uint32_t value = 0;
	for (int index = 0; index < 4; index++) {
		char character = cursor[index];
		int digit = character >= '0' && character <= '9' ? character - '0'
			: character >= 'a' && character <= 'f' ? character - 'a' + 10
			: character >= 'A' && character <= 'F' ? character - 'A' + 10 : -1;
		if (digit < 0) {
			return 0;
		}
		value = value << 4 | (uint32_t)digit;
	}
	*out_value = value;
	return 1;
}

static void json_append_utf8(sample_text *text, uint32_t scalar)
{
	char bytes[4];
	size_t length;
	if (scalar < 0x80) {
		bytes[0] = (char)scalar;
		length = 1;
	} else if (scalar < 0x800) {
		bytes[0] = (char)(0xC0 | (scalar >> 6));
		bytes[1] = (char)(0x80 | (scalar & 0x3F));
		length = 2;
	} else if (scalar < 0x10000) {
		bytes[0] = (char)(0xE0 | (scalar >> 12));
		bytes[1] = (char)(0x80 | ((scalar >> 6) & 0x3F));
		bytes[2] = (char)(0x80 | (scalar & 0x3F));
		length = 3;
	} else {
		bytes[0] = (char)(0xF0 | (scalar >> 18));
		bytes[1] = (char)(0x80 | ((scalar >> 12) & 0x3F));
		bytes[2] = (char)(0x80 | ((scalar >> 6) & 0x3F));
		bytes[3] = (char)(0x80 | (scalar & 0x3F));
		length = 4;
	}
	text_append(text, bytes, length);
}

/* Reads the escape after a backslash at cursor into text, and returns the cursor after it, or NULL. */
static const char *json_read_escape(const char *cursor, sample_text *text)
{
	const char *simple = "\"\\/bfnrt";
	const char *replacement = "\"\\/\b\f\n\r\t";
	const char *found = *cursor != '\0' ? strchr(simple, *cursor) : NULL;
	if (found != NULL) {
		text_append(text, replacement + (found - simple), 1);
		return cursor + 1;
	}
	uint32_t scalar = 0;
	if (*cursor != 'u' || !json_hex4(cursor + 1, &scalar)) {
		return NULL;
	}
	cursor += 5;
	if (scalar >= 0xD800 && scalar <= 0xDBFF) {
		uint32_t low = 0;
		if (cursor[0] != '\\' || cursor[1] != 'u' || !json_hex4(cursor + 2, &low) || low < 0xDC00 || low > 0xDFFF) {
			return NULL;
		}
		scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00);
		cursor += 6;
	}
	json_append_utf8(text, scalar);
	return cursor;
}

static char *json_read_string(sample_json_reader *reader)
{
	if (*reader->cursor != '"') {
		return NULL;
	}
	reader->cursor++;
	sample_text text = { 0 };
	text_append(&text, "", 0);
	while (*reader->cursor != '"') {
		if (*reader->cursor == '\0' || (unsigned char)*reader->cursor < 0x20) {
			text.failed = 1;
			break;
		}
		if (*reader->cursor == '\\') {
			const char *next = json_read_escape(reader->cursor + 1, &text);
			if (next == NULL) {
				text.failed = 1;
				break;
			}
			reader->cursor = next;
			continue;
		}
		text_append(&text, reader->cursor, 1);
		reader->cursor++;
	}
	if (!text.failed) {
		reader->cursor++;
	}
	return text_take(&text);
}

static sample_json *json_read_value(sample_json_reader *reader);

static int json_add(sample_json *container, char *key, sample_json *item)
{
	sample_json **items = sample_reallocate(container->items, (container->count + 1) * sizeof *items);
	if (items == NULL) {
		return 0;
	}
	container->items = items;
	if (container->kind == SAMPLE_JSON_OBJECT) {
		char **keys = sample_reallocate(container->keys, (container->count + 1) * sizeof *keys);
		if (keys == NULL) {
			return 0;
		}
		container->keys = keys;
		keys[container->count] = key;
	}
	items[container->count++] = item;
	return 1;
}

/* Reads the members of an array or an object after its opening bracket. */
static int json_read_members(sample_json_reader *reader, sample_json *container, char closing)
{
	json_skip_space(reader);
	if (*reader->cursor == closing) {
		reader->cursor++;
		return 1;
	}
	for (;;) {
		char *key = NULL;
		if (container->kind == SAMPLE_JSON_OBJECT) {
			json_skip_space(reader);
			key = json_read_string(reader);
			json_skip_space(reader);
			if (key == NULL || *reader->cursor != ':') {
				sample_free(key);
				return 0;
			}
			reader->cursor++;
		}
		sample_json *item = json_read_value(reader);
		if (item == NULL || !json_add(container, key, item)) {
			sample_free(key);
			json_free(item);
			return 0;
		}
		json_skip_space(reader);
		if (*reader->cursor == closing) {
			reader->cursor++;
			return 1;
		}
		if (*reader->cursor != ',') {
			return 0;
		}
		reader->cursor++;
	}
}

static int json_read_literal(sample_json_reader *reader, const char *literal)
{
	size_t length = strlen(literal);
	if (strncmp(reader->cursor, literal, length) != 0) {
		return 0;
	}
	reader->cursor += length;
	return 1;
}

static int json_read_number(sample_json_reader *reader, double *out_number)
{
	char digits[64];
	size_t length = 0;
	while (strchr("+-0123456789.eE", *reader->cursor) != NULL && *reader->cursor != '\0' && length + 1 < sizeof digits) {
		digits[length++] = *reader->cursor++;
	}
	digits[length] = '\0';
	char *end = NULL;
	*out_number = strtod(digits, &end);
	return length > 0 && end == digits + length;
}

static sample_json *json_read_value(sample_json_reader *reader)
{
	json_skip_space(reader);
	if (reader->depth >= SAMPLE_MAXIMUM_DEPTH) {
		return NULL;
	}
	sample_json *value = sample_allocate(sizeof *value);
	if (value == NULL) {
		return NULL;
	}
	int succeeded;
	char first = *reader->cursor;
	if (first == '{' || first == '[') {
		value->kind = first == '{' ? SAMPLE_JSON_OBJECT : SAMPLE_JSON_ARRAY;
		reader->cursor++;
		reader->depth++;
		succeeded = json_read_members(reader, value, first == '{' ? '}' : ']');
		reader->depth--;
	} else if (first == '"') {
		value->kind = SAMPLE_JSON_STRING;
		value->string = json_read_string(reader);
		succeeded = value->string != NULL;
	} else if (first == 't' || first == 'f') {
		value->kind = SAMPLE_JSON_BOOLEAN;
		value->number = first == 't';
		succeeded = json_read_literal(reader, first == 't' ? "true" : "false");
	} else if (first == 'n') {
		value->kind = SAMPLE_JSON_NULL;
		succeeded = json_read_literal(reader, "null");
	} else {
		value->kind = SAMPLE_JSON_NUMBER;
		succeeded = json_read_number(reader, &value->number);
	}
	if (!succeeded) {
		json_free(value);
		return NULL;
	}
	return value;
}

/* Parses a whole JSON text, or returns NULL when it is malformed. */
static sample_json *json_parse(const char *text)
{
	if (text == NULL) {
		return NULL;
	}
	sample_json_reader reader = { text, 0 };
	sample_json *value = json_read_value(&reader);
	json_skip_space(&reader);
	if (value != NULL && *reader.cursor != '\0') {
		json_free(value);
		return NULL;
	}
	return value;
}

static const sample_json *json_member(const sample_json *object, const char *key)
{
	if (object == NULL || object->kind != SAMPLE_JSON_OBJECT) {
		return NULL;
	}
	for (size_t index = 0; index < object->count; index++) {
		if (strcmp(object->keys[index], key) == 0) {
			return object->items[index];
		}
	}
	return NULL;
}

static const char *json_string_member(const sample_json *object, const char *key)
{
	const sample_json *member = json_member(object, key);
	return member != NULL && member->kind == SAMPLE_JSON_STRING ? member->string : NULL;
}

static double json_number_member(const sample_json *object, const char *key, double fallback)
{
	const sample_json *member = json_member(object, key);
	return member != NULL && member->kind == SAMPLE_JSON_NUMBER ? member->number : fallback;
}

static const sample_json *json_item(const sample_json *array, size_t index)
{
	return array != NULL && array->kind == SAMPLE_JSON_ARRAY && index < array->count ? array->items[index] : NULL;
}

#pragma mark Plugin state

typedef struct sample_instance {
	nfk_plugin table;
	const nfk_plugin_host *host;
	char *echo_prefix;
} sample_instance;

typedef enum sample_model {
	SAMPLE_MODEL_INVERT,
	SAMPLE_MODEL_SLOW
} sample_model;

struct nfk_plugin_backend {
	sample_model model;
	void *weights;
	nfk_plugin_run *current_run;
	int32_t cancel_requested;
};

/* A result payload and everything it owns, in one allocation the host frees through free_payload. */
typedef struct sample_result {
	nfk_plugin_payload payload;
	nfk_plugin_buffer buffers[2];
	void *owned[2];
	char *json;
} sample_result;

static void sample_log(const sample_instance *instance, nfk_plugin_log_level level, const char *message)
{
	const nfk_plugin_host *host = instance->host;
	if (SAMPLE_HAS_FIELD(host, nfk_plugin_host, log) && host->log != NULL) {
		host->log(host->context, level, message);
	}
}

static int sample_is_cancelled(const sample_instance *instance, nfk_plugin_backend *backend, nfk_plugin_run *run)
{
	const nfk_plugin_host *host = instance->host;
	if (__atomic_load_n(&backend->cancel_requested, __ATOMIC_SEQ_CST)) {
		return 1;
	}
	return SAMPLE_HAS_FIELD(host, nfk_plugin_host, is_cancelled) && host->is_cancelled != NULL
		&& host->is_cancelled(host->context, run) != 0;
}

static void sample_report_progress(const sample_instance *instance, nfk_plugin_run *run, double fraction, const char *partial_json)
{
	const nfk_plugin_host *host = instance->host;
	if (!SAMPLE_HAS_FIELD(host, nfk_plugin_host, report_progress) || host->report_progress == NULL) {
		return;
	}
	nfk_plugin_payload partial;
	memset(&partial, 0, sizeof partial);
	partial.struct_size = (uint32_t)sizeof partial;
	partial.json = partial_json;
	host->report_progress(host->context, run, fraction, partial_json != NULL ? &partial : NULL);
}

static void sample_load(nfk_plugin_backend *backend)
{
	if (backend->weights == NULL) {
		backend->weights = sample_allocate(SAMPLE_WEIGHT_BYTES);
	}
}

static void sample_sleep_milliseconds(long milliseconds)
{
	struct timespec interval = { milliseconds / 1000, (milliseconds % 1000) * 1000000L };
	while (nanosleep(&interval, &interval) != 0) {
	}
}

#pragma mark Attachments

/* Where a tagged value's bytes are: decoded from base64 into owned, or a buffer of the request. */
typedef struct sample_bytes {
	const uint8_t *bytes;
	size_t length;
	uint8_t *owned;
	int attached;
} sample_bytes;

static int sample_attached_bytes(const sample_json *entry, const nfk_plugin_payload *request, sample_bytes *out_bytes)
{
	double index = json_number_member(entry, "index", -1);
	if (index < 0 || request->buffers == NULL || index >= request->buffer_count) {
		return 0;
	}
	/* The array strides by its first element's struct_size, which a later minor version lengthens. */
	size_t stride = request->buffers[0].struct_size;
	if (!SAMPLE_HAS_FIELD(&request->buffers[0], nfk_plugin_buffer, reserved)) {
		return 0;
	}
	const nfk_plugin_buffer *buffer = (const nfk_plugin_buffer *)((const uint8_t *)request->buffers + stride * (size_t)index);
	if (buffer->kind != NFK_PLUGIN_BUFFER_BYTES || buffer->bytes == NULL) {
		return 0;
	}
	out_bytes->bytes = buffer->bytes;
	out_bytes->length = (size_t)buffer->length;
	out_bytes->attached = 1;
	return 1;
}

static int sample_inline_bytes(const char *base64, sample_bytes *out_bytes)
{
	size_t length = 0;
	uint8_t *bytes = base64 != NULL ? base64_decode(base64, &length) : NULL;
	if (bytes == NULL) {
		return 0;
	}
	out_bytes->bytes = bytes;
	out_bytes->owned = bytes;
	out_bytes->length = length;
	return 1;
}

/* Adds a result attachment the result owns, and returns its index. */
static size_t sample_result_attach(sample_result *result, uint8_t *bytes, size_t length)
{
	size_t index = result->payload.buffer_count++;
	nfk_plugin_buffer *buffer = &result->buffers[index];
	buffer->struct_size = (uint32_t)sizeof *buffer;
	buffer->kind = NFK_PLUGIN_BUFFER_BYTES;
	buffer->bytes = bytes;
	buffer->length = length;
	buffer->descriptor = -1;
	result->owned[index] = bytes;
	return index;
}

#pragma mark Invert

static int sample_invert_image(const sample_json *entry, const nfk_plugin_payload *request, sample_result *result,
							   sample_text *outputs, char **out_error_json)
{
	const char *tag = json_string_member(entry, "$nfk");
	const char *kind = json_string_member(entry, "as");
	int attached = tag != NULL && strcmp(tag, "buffer") == 0 && kind != NULL && strcmp(kind, "pixelBuffer") == 0;
	int inline_buffer = tag != NULL && strcmp(tag, "pixelBuffer") == 0;
	const sample_json *plane = json_item(json_member(entry, "planes"), 0);
	double width = json_number_member(entry, "width", 0);
	double height = json_number_member(entry, "height", 0);
	double bytes_per_row = json_number_member(plane, "bytesPerRow", 0);
	double rows = json_number_member(plane, "rows", 0);
	double offset = json_number_member(plane, "offset", 0);
	if ((!attached && !inline_buffer) || json_number_member(entry, "format", 0) != SAMPLE_FORMAT_32BGRA
		|| width < 1 || height < 1 || bytes_per_row < width * 4 || rows < height || offset < 0) {
		*out_error_json = sample_error_json("invert takes a 32BGRA pixel buffer under \"image\"", "NFKInferenceErrorDomain", 4);
		return 0;
	}
	sample_bytes source = { 0 };
	int found = attached ? sample_attached_bytes(entry, request, &source) : sample_inline_bytes(json_string_member(plane, "base64"), &source);
	size_t length = (size_t)bytes_per_row * (size_t)rows;
	if (!found || (size_t)offset + length > source.length) {
		sample_free(source.owned);
		*out_error_json = sample_error_json("the image's bytes are missing or short", "NFKInferenceErrorDomain", 2);
		return 0;
	}
	uint8_t *inverted = sample_allocate(length);
	if (inverted == NULL) {
		sample_free(source.owned);
		return 0;
	}
	const uint8_t *pixels = source.bytes + (size_t)offset;
	for (size_t row = 0; row < (size_t)rows; row++) {
		for (size_t column = 0; column < (size_t)bytes_per_row; column++) {
			size_t at = row * (size_t)bytes_per_row + column;
			int color = column < (size_t)width * 4 && column % 4 != 3;
			inverted[at] = color ? (uint8_t)(255 - pixels[at]) : pixels[at];
		}
	}
	sample_free(source.owned);
	text_append_format(outputs, "\"image\":{\"format\":%.0f,\"width\":%.0f,\"height\":%.0f,", SAMPLE_FORMAT_32BGRA, width, height);
	if (attached) {
		size_t index = sample_result_attach(result, inverted, length);
		text_append_format(outputs, "\"$nfk\":\"buffer\",\"as\":\"pixelBuffer\",\"index\":%zu,\"length\":%zu,", index, length);
		text_append_format(outputs, "\"planes\":[{\"offset\":0,\"bytesPerRow\":%.0f,\"rows\":%.0f}]}", bytes_per_row, rows);
		return 1;
	}
	text_append_format(outputs, "\"$nfk\":\"pixelBuffer\",\"planes\":[{\"bytesPerRow\":%.0f,\"rows\":%.0f,\"base64\":\"", bytes_per_row, rows);
	base64_append(outputs, inverted, length);
	text_append_string(outputs, "\"}]}");
	sample_free(inverted);
	return 1;
}

static int sample_invert_values(const sample_json *entry, const nfk_plugin_payload *request, sample_result *result,
								sample_text *outputs, char **out_error_json)
{
	const char *tag = json_string_member(entry, "$nfk");
	int attached = tag != NULL && strcmp(tag, "buffer") == 0;
	const char *kind = attached ? json_string_member(entry, "as") : tag;
	int singles = kind != NULL && strcmp(kind, "float32Array") == 0;
	int doubles = kind != NULL && strcmp(kind, "float64Array") == 0;
	sample_bytes source = { 0 };
	int found = (singles || doubles)
		&& (attached ? sample_attached_bytes(entry, request, &source) : sample_inline_bytes(json_string_member(entry, "base64"), &source));
	size_t element_size = singles ? sizeof(float) : sizeof(double);
	if (!found || source.length % element_size != 0) {
		sample_free(source.owned);
		*out_error_json = sample_error_json("invert takes a float32 or float64 array under \"values\"", "NFKInferenceErrorDomain", 4);
		return 0;
	}
	size_t count = source.length / element_size;
	uint8_t *inverted = sample_allocate(source.length);
	if (inverted == NULL) {
		sample_free(source.owned);
		return 0;
	}
	for (size_t index = 0; index < count; index++) {
		if (singles) {
			float value;
			memcpy(&value, source.bytes + index * element_size, sizeof value);
			value = 1.0f - value;
			memcpy(inverted + index * element_size, &value, sizeof value);
		} else {
			double value;
			memcpy(&value, source.bytes + index * element_size, sizeof value);
			value = 1.0 - value;
			memcpy(inverted + index * element_size, &value, sizeof value);
		}
	}
	size_t length = source.length;
	sample_free(source.owned);
	if (attached) {
		size_t index = sample_result_attach(result, inverted, length);
		text_append_format(outputs, "\"values\":{\"$nfk\":\"buffer\",\"as\":\"%s\",\"index\":%zu,\"length\":%zu,\"count\":%zu}",
						   kind, index, length, count);
		return 1;
	}
	text_append_format(outputs, "\"values\":{\"$nfk\":\"%s\",\"base64\":\"", kind);
	base64_append(outputs, inverted, length);
	text_append_string(outputs, "\"}");
	sample_free(inverted);
	return 1;
}

/* A detection under "region" comes back under "detections", its box mirrored left to right and its
   confidence replaced by 1 minus itself; the label and class index are kept. */
static int sample_invert_region(const sample_json *entry, sample_text *outputs, char **out_error_json)
{
	const char *tag = json_string_member(entry, "$nfk");
	const sample_json *box = json_member(entry, "boundingBox");
	if (tag == NULL || strcmp(tag, "detection") != 0 || box == NULL || box->kind != SAMPLE_JSON_OBJECT) {
		*out_error_json = sample_error_json("invert takes a detection with a boundingBox under \"region\"", "NFKInferenceErrorDomain", 4);
		return 0;
	}
	double x = json_number_member(box, "x", 0);
	double y = json_number_member(box, "y", 0);
	double width = json_number_member(box, "width", 0);
	double height = json_number_member(box, "height", 0);
	text_append_string(outputs, "\"detections\":[{\"$nfk\":\"detection\"");
	const char *label = json_string_member(entry, "label");
	if (label != NULL) {
		text_append_string(outputs, ",\"label\":");
		text_append_json_string(outputs, label);
	}
	text_append_format(outputs, ",\"classIndex\":%.0f,\"confidence\":%.17g", json_number_member(entry, "classIndex", -1),
					   1.0 - json_number_member(entry, "confidence", 0));
	text_append_format(outputs, ",\"boundingBox\":{\"x\":%.17g,\"y\":%.17g,\"width\":%.17g,\"height\":%.17g}}]",
					   1.0 - x - width, y, width, height);
	return 1;
}

static nfk_plugin_status sample_run_invert(const sample_json *inputs, const nfk_plugin_payload *request,
										   sample_result *result, sample_text *outputs, char **out_error_json)
{
	const sample_json *image = json_member(inputs, "image");
	const sample_json *values = json_member(inputs, "values");
	const sample_json *region = json_member(inputs, "region");
	if (image == NULL && values == NULL && region == NULL) {
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_MISSING_INPUT, "invert needs \"image\", \"values\", or \"region\"",
						   "NFKInferenceErrorDomain", 2);
	}
	if (image != NULL && !sample_invert_image(image, request, result, outputs, out_error_json)) {
		return *out_error_json != NULL ? NFK_PLUGIN_STATUS_UNSUPPORTED : NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	if (image != NULL && values != NULL) {
		text_append_string(outputs, ",");
	}
	if (values != NULL && !sample_invert_values(values, request, result, outputs, out_error_json)) {
		return *out_error_json != NULL ? NFK_PLUGIN_STATUS_UNSUPPORTED : NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	if (region != NULL && (image != NULL || values != NULL)) {
		text_append_string(outputs, ",");
	}
	if (region != NULL && !sample_invert_region(region, outputs, out_error_json)) {
		return NFK_PLUGIN_STATUS_UNSUPPORTED;
	}
	return NFK_PLUGIN_STATUS_OK;
}

#pragma mark Slow

static nfk_plugin_status sample_run_slow(const sample_instance *instance, nfk_plugin_backend *backend, nfk_plugin_run *run,
										 const sample_json *parameters, sample_text *outputs, char **out_error_json)
{
	long steps = (long)json_number_member(parameters, "steps", 10);
	long step_milliseconds = (long)json_number_member(parameters, "stepMilliseconds", 20);
	long exit_at_step = (long)json_number_member(parameters, "exitAtStep", 0);
	if (steps < 1 || step_milliseconds < 0) {
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_INVALID_ARGUMENT, "steps is at least 1 and stepMilliseconds at least 0",
						   "NFKInferenceErrorDomain", 4);
	}
	for (long step = 1; step <= steps; step++) {
		sample_sleep_milliseconds(step_milliseconds);
		if (sample_is_cancelled(instance, backend, run)) {
			return sample_fail(out_error_json, NFK_PLUGIN_STATUS_CANCELLED, "the run was cancelled", "NSCocoaErrorDomain", 3072);
		}
		if (step == exit_at_step) {
			_exit(70);
		}
		char partial[64];
		snprintf(partial, sizeof partial, "{\"outputs\":{\"step\":%ld}}", step);
		sample_report_progress(instance, run, (double)step / (double)steps, partial);
	}
	text_append_format(outputs, "\"steps\":%ld", steps);
	return NFK_PLUGIN_STATUS_OK;
}

#pragma mark The function table

static const char sample_description[] =
	"{\"models\":["
		"{\"id\":\"invert\",\"displayName\":\"Invert\",\"family\":\"sample-invert\","
		"\"summary\":\"Inverts a 32BGRA image, a float array, or a detection's box.\",\"engine\":\"plugin\",\"tasks\":[\"image-to-image\"],"
		"\"inputs\":[{\"key\":\"image\",\"valueType\":\"image\"},{\"key\":\"values\",\"valueType\":\"embedding\"},"
		"{\"key\":\"region\",\"valueType\":\"json\"}],"
		"\"outputs\":[{\"key\":\"image\",\"valueType\":\"image\"},{\"key\":\"values\",\"valueType\":\"embedding\"},"
		"{\"key\":\"detections\",\"valueType\":\"detections\"},"
		"{\"key\":\"pluginVersion\",\"valueType\":\"text\",\"required\":true}]},"
		"{\"id\":\"slow\",\"displayName\":\"Slow\",\"family\":\"sample-slow\","
		"\"summary\":\"Counts steps, reporting progress after each, until it finishes or is cancelled.\",\"engine\":\"plugin\","
		"\"outputs\":[{\"key\":\"steps\",\"valueType\":\"integer\",\"required\":true},"
		"{\"key\":\"pluginVersion\",\"valueType\":\"text\",\"required\":true}],"
		"\"options\":[{\"key\":\"steps\",\"kind\":\"integer\",\"default\":10,\"minimum\":1},"
		"{\"key\":\"stepMilliseconds\",\"kind\":\"integer\",\"default\":20,\"minimum\":0},"
		"{\"key\":\"exitAtStep\",\"kind\":\"integer\",\"minimum\":1,"
		"\"help\":\"Ends the plugin's process at this step. Only for a plugin in a process of its own.\"}]}"
	"],"
	"\"tools\":[{\"name\":\"echo\",\"description\":\"Returns the text it is given, after the echoPrefix option.\","
		"\"parameters\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}}],"
	"\"options\":[{\"key\":\"echoPrefix\",\"kind\":\"text\",\"default\":\"\",\"title\":\"Echo prefix\"}]}";

static nfk_plugin_status sample_describe(void *context, char **out_json)
{
	*out_json = sample_copy_string(sample_description);
	return *out_json != NULL ? NFK_PLUGIN_STATUS_OK : NFK_PLUGIN_STATUS_NO_MEMORY;
}

static nfk_plugin_status sample_create_backend(void *context, const char *model_id, const char *options_json,
											   nfk_plugin_backend **out_backend, char **out_error_json)
{
	*out_backend = NULL;
	int invert = model_id != NULL && strcmp(model_id, "invert") == 0;
	int slow = model_id != NULL && strcmp(model_id, "slow") == 0;
	if (!invert && !slow) {
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_UNKNOWN_MODEL, "the plugin has no model with that identifier",
						   "NFKPluginErrorDomain", 14);
	}
	nfk_plugin_backend *backend = sample_allocate(sizeof *backend);
	if (backend == NULL) {
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	backend->model = invert ? SAMPLE_MODEL_INVERT : SAMPLE_MODEL_SLOW;
	*out_backend = backend;
	return NFK_PLUGIN_STATUS_OK;
}

static nfk_plugin_status sample_describe_backend(void *context, nfk_plugin_backend *backend, char **out_json)
{
	int invert = backend->model == SAMPLE_MODEL_INVERT;
	unsigned resident = backend->weights != NULL ? SAMPLE_WEIGHT_BYTES : 0;
	sample_text text = { 0 };
	text_append_string(&text, invert ? "{\"inputs\":[\"image\",\"region\",\"values\"],\"parameters\":[],\"outputs\":[\"detections\",\"image\",\"values\",\"pluginVersion\"],"
								  : "{\"inputs\":[],\"parameters\":[\"steps\",\"stepMilliseconds\",\"exitAtStep\"],\"outputs\":[\"steps\",\"pluginVersion\"],");
	text_append_format(&text, "\"modelInfo\":{\"architecture\":\"%s\"},", invert ? "sample-invert" : "sample-slow");
	text_append_format(&text, "\"loaded\":%s,\"residentBytes\":%u,\"peakBytes\":%u}", resident > 0 ? "true" : "false", resident, SAMPLE_WEIGHT_BYTES);
	*out_json = text_take(&text);
	return *out_json != NULL ? NFK_PLUGIN_STATUS_OK : NFK_PLUGIN_STATUS_NO_MEMORY;
}

static nfk_plugin_status sample_run(void *context, nfk_plugin_backend *backend, nfk_plugin_run *run,
									const nfk_plugin_payload *request, nfk_plugin_payload **out_result, char **out_error_json)
{
	const sample_instance *instance = context;
	if (request == NULL || out_result == NULL || out_error_json == NULL) {
		return NFK_PLUGIN_STATUS_INVALID_ARGUMENT;
	}
	*out_result = NULL;
	*out_error_json = NULL;
	sample_json *body = json_parse(request->json);
	const sample_json *inputs = json_member(body, "inputs");
	if (inputs == NULL || inputs->kind != SAMPLE_JSON_OBJECT) {
		json_free(body);
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_INVALID_ARGUMENT, "the request carries no inputs object",
						   "NFKInferenceErrorDomain", 2);
	}
	sample_result *result = sample_allocate(sizeof *result);
	if (result == NULL) {
		json_free(body);
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	result->payload.struct_size = (uint32_t)sizeof result->payload;
	result->payload.buffers = result->buffers;
	sample_load(backend);
	__atomic_store_n(&backend->cancel_requested, 0, __ATOMIC_SEQ_CST);
	__atomic_store_n(&backend->current_run, run, __ATOMIC_SEQ_CST);
	sample_text outputs = { 0 };
	text_append_string(&outputs, "{\"outputs\":{");
	nfk_plugin_status status = backend->model == SAMPLE_MODEL_INVERT
		? sample_run_invert(inputs, request, result, &outputs, out_error_json)
		: sample_run_slow(instance, backend, run, json_member(body, "parameters"), &outputs, out_error_json);
	__atomic_store_n(&backend->current_run, NULL, __ATOMIC_SEQ_CST);
	json_free(body);
	text_append_string(&outputs, ",\"pluginVersion\":\"" SAMPLE_PLUGIN_VERSION "\"}}");
	result->json = text_take(&outputs);
	result->payload.json = result->json;
	if (status == NFK_PLUGIN_STATUS_OK && result->json == NULL) {
		status = NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	if (status != NFK_PLUGIN_STATUS_OK) {
		instance->table.free_payload(context, &result->payload);
		return status;
	}
	*out_result = &result->payload;
	return NFK_PLUGIN_STATUS_OK;
}

static void sample_cancel(void *context, nfk_plugin_backend *backend, nfk_plugin_run *run)
{
	if (__atomic_load_n(&backend->current_run, __ATOMIC_SEQ_CST) == run) {
		__atomic_store_n(&backend->cancel_requested, 1, __ATOMIC_SEQ_CST);
	}
}

static void sample_unload(void *context, nfk_plugin_backend *backend)
{
	sample_free(backend->weights);
	backend->weights = NULL;
}

static void sample_destroy_backend(void *context, nfk_plugin_backend *backend)
{
	if (backend == NULL) {
		return;
	}
	sample_free(backend->weights);
	sample_free(backend);
}

static nfk_plugin_status sample_call_tool(void *context, const char *tool_name, const char *arguments_json,
										  char **out_result_json, char **out_error_json)
{
	const sample_instance *instance = context;
	*out_result_json = NULL;
	if (tool_name == NULL || strcmp(tool_name, "echo") != 0) {
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_UNKNOWN_TOOL, "the plugin has no tool with that name",
						   "NFKPluginErrorDomain", 15);
	}
	sample_json *arguments = json_parse(arguments_json);
	const char *echoed = json_string_member(arguments, "text");
	if (echoed == NULL) {
		json_free(arguments);
		return sample_fail(out_error_json, NFK_PLUGIN_STATUS_INVALID_ARGUMENT, "echo needs a string \"text\"",
						   "NFKInferenceErrorDomain", 2);
	}
	sample_text text = { 0 };
	text_append_string(&text, "{\"$nfk\":\"toolResult\",\"text\":");
	sample_text joined = { 0 };
	text_append_string(&joined, instance->echo_prefix != NULL ? instance->echo_prefix : "");
	text_append_string(&joined, echoed);
	char *combined = text_take(&joined);
	text_append_json_string(&text, combined != NULL ? combined : "");
	sample_free(combined);
	text_append_string(&text, ",\"isError\":false,\"images\":[],\"outputs\":{}}");
	json_free(arguments);
	*out_result_json = text_take(&text);
	return *out_result_json != NULL ? NFK_PLUGIN_STATUS_OK : NFK_PLUGIN_STATUS_NO_MEMORY;
}

static void sample_free_payload(void *context, nfk_plugin_payload *payload)
{
	sample_result *result = (sample_result *)payload;
	if (result == NULL) {
		return;
	}
	for (size_t index = 0; index < 2; index++) {
		sample_free(result->owned[index]);
	}
	sample_free(result->json);
	sample_free(result);
}

static void sample_free_string(void *context, char *string)
{
	sample_free(string);
}

static void sample_shutdown(void *context)
{
	sample_instance *instance = context;
	sample_log(instance, NFK_PLUGIN_LOG_INFO, "sample plugin " SAMPLE_PLUGIN_VERSION " shut down");
	sample_free(instance->echo_prefix);
	sample_free(instance);
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
	if (host == NULL || !SAMPLE_HAS_FIELD(host, nfk_plugin_host, options_json)) {
		return NFK_PLUGIN_STATUS_INVALID_ARGUMENT;
	}
	sample_instance *instance = sample_allocate(sizeof *instance);
	if (instance == NULL) {
		return NFK_PLUGIN_STATUS_NO_MEMORY;
	}
	sample_json *options = json_parse(host->options_json);
	const char *prefix = json_string_member(options, "echoPrefix");
	instance->echo_prefix = prefix != NULL ? sample_copy_string(prefix) : NULL;
	json_free(options);
	instance->host = host;
	nfk_plugin *table = &instance->table;
	table->struct_size = (uint32_t)sizeof *table;
	table->abi_version = NFK_PLUGIN_ABI_VERSION;
	table->context = instance;
	table->describe = sample_describe;
	table->create_backend = sample_create_backend;
	table->describe_backend = sample_describe_backend;
	table->run = sample_run;
	table->cancel = sample_cancel;
	table->unload = sample_unload;
	table->destroy_backend = sample_destroy_backend;
	table->call_tool = sample_call_tool;
	table->free_payload = sample_free_payload;
	table->free_string = sample_free_string;
	table->shutdown = sample_shutdown;
	sample_log(instance, NFK_PLUGIN_LOG_INFO, "sample plugin " SAMPLE_PLUGIN_VERSION " started");
	*out_plugin = table;
	return NFK_PLUGIN_STATUS_OK;
}
