//
//  NFKRemoteEmbeddingBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteEmbeddingBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"
#import <InferKit/NFKRemoteFileStore.h>
#import "NFKRemoteTransportPrivate.h"
#import "NFKRemoteBatchPrivate.h"

/*! A vector a service wrote as base64, decoded by the encoding_format the request named: OpenAI's
	base64 is little-endian float32, Perplexity's base64_binary is packed bits, least significant bit
	first, each a 0 or 1, and Perplexity's base64_int8, which is also its default, is signed int8. */
static NSArray<NSNumber *> * _Nullable NFKRemoteEmbeddingDecodedVector(NSString *encoded, NSString * _Nullable format)
{
	NSData *data = [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters];
	if (data == nil) {
		return nil;
	}
	const uint8_t *bytes = data.bytes;
	NSMutableArray<NSNumber *> *vector = [NSMutableArray array];
	if ([format isEqualToString:@"base64"]) {
		for (NSUInteger offset = 0; offset + sizeof(float) <= data.length; offset += sizeof(float)) {
			uint32_t bits = OSReadLittleInt32(bytes, offset);
			float value;
			memcpy(&value, &bits, sizeof(value));
			[vector addObject:@(value)];
		}
	} else if ([format isEqualToString:@"base64_binary"]) {
		for (NSUInteger index = 0; index < data.length * 8; index++) {
			[vector addObject:@((bytes[index / 8] >> (index % 8)) & 1)];
		}
	} else {
		for (NSUInteger index = 0; index < data.length; index++) {
			[vector addObject:@((int8_t)bytes[index])];
		}
	}
	return vector;
}

@implementation NFKRemoteEmbeddingBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteEmbeddingBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSArray<NSString *> *serving = @[ @"openai", @"gemini", @"mistral", @"together", @"openrouter", @"perplexity",
									  @"ollama", @"lmstudio", @"llamacpp", @"vllm", @"inferkit" ];
	if (![serving containsObject:provider.identifier]) {
		return nil;
	}
	NFKRemoteEmbeddingBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"embeddings"]];
	if ([provider.identifier isEqualToString:@"gemini"]) {
		backend.endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"models"];
		backend.apiStyle = NFKRemoteEmbeddingAPIStyleGeminiNative;
	}
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		backend.session = pinned;
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 60.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote-embedding";
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:@[ NFKOutputEmbedding, NFKRemoteBackendRawKey ]];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSString *text = [self textForRequest:request];
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:YES error:outError];
	if (attachments == nil) {
		return nil;
	}
	if (text == nil && attachments.isEmpty) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries neither text nor media" error:outError];
	}
	id input = [self inputForText:text attachments:attachments error:outError];
	if (input == nil) {
		return nil;
	}
	BOOL contextualized = self.isContextualizedModel && [input isKindOfClass:NSString.class];
	NSDictionary *body = [self responseForInput:contextualized ? @[ @[ input ] ] : input parameters:request.parameters
											URL:contextualized ? self.contextualizedEndpointURL : self.endpointURL error:outError];
	if (body == nil) {
		return nil;
	}
	NSString *format = request.parameters[@"encoding_format"];
	NSArray<NSArray<NSNumber *> *> *vectors = contextualized ? [self documentVectorsInBody:body encoding:format error:outError].firstObject
															 : [self vectorsInBody:body encoding:format error:outError];
	if (vectors == nil) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: vectors.firstObject,
													NFKRemoteBackendRawKey: body }];
}

- (nullable NSArray<NSArray<NSNumber *> *> *)embeddingsForTexts:(NSArray<NSString *> *)texts
														  error:(NSError * _Nullable *)outError
{
	if (texts.count == 0) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"no texts to embed" error:outError];
		return nil;
	}
	NSDictionary *body = [self responseForInput:texts parameters:nil URL:self.endpointURL error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray<NSArray<NSNumber *> *> *vectors = [self vectorsInBody:body encoding:nil error:outError];
	if (vectors != nil && vectors.count != texts.count) {
		[self failWithCode:kNFKError_InferenceBackendFailure
					reason:[NSString stringWithFormat:@"%lu texts were sent and %lu vectors came back",
							(unsigned long)texts.count, (unsigned long)vectors.count]
					 error:outError];
		return nil;
	}
	return vectors;
}

- (nullable NSArray<NSArray<NSArray<NSNumber *> *> *> *)contextualizedEmbeddingsForDocuments:(NSArray<NSArray<NSString *> *> *)documents
																						 error:(NSError * _Nullable *)outError
{
	if (documents.count == 0) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"no documents to embed" error:outError];
		return nil;
	}
	NSDictionary *body = [self responseForInput:documents parameters:nil URL:self.contextualizedEndpointURL error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray<NSArray<NSArray<NSNumber *> *> *> *vectors = [self documentVectorsInBody:body encoding:nil error:outError];
	if (vectors != nil && vectors.count != documents.count) {
		[self failWithCode:kNFKError_InferenceBackendFailure
					reason:[NSString stringWithFormat:@"%lu documents were sent and %lu came back",
							(unsigned long)documents.count, (unsigned long)vectors.count]
					 error:outError];
		return nil;
	}
	return vectors;
}

- (BOOL)isContextualizedModel
{
	return [self.modelName hasPrefix:@"pplx-embed-context"] || [self.modelName hasPrefix:@"perplexity/pplx-embed-context"];
}

- (nullable NSURL *)contextualizedEndpointURL
{
	return [self.endpointURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"contextualizedembeddings"];
}

#pragma mark Request and response

// Text alone goes as it stands. Media goes as OpenRouter's content parts, or, on the Gemini style,
// as the native parts that style always sends.
- (nullable id)inputForText:(nullable NSString *)text attachments:(NFKRemoteAttachments *)attachments error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteEmbeddingAPIStyleGeminiNative) {
		NSMutableArray *parts = [NSMutableArray array];
		if (text.length > 0) {
			[parts addObject:@{ @"text": text }];
		}
		for (NSData *png in attachments.imagePNGs) {
			[parts addObject:[self inlinePart:png mimeType:@"image/png"]];
		}
		if (attachments.audioData != nil) {
			[parts addObject:[self inlinePart:attachments.audioData mimeType:[@"audio/" stringByAppendingString:attachments.audioFormat ?: @"wav"]]];
		}
		if (attachments.videoData != nil) {
			[parts addObject:[self inlinePart:attachments.videoData mimeType:[@"video/" stringByAppendingString:attachments.videoFormat ?: @"mp4"]]];
		}
		for (NSDictionary *document in attachments.documents) {
			NFKRemoteFile *file = document[@"fileReference"];
			if (file != nil) {
				[parts addObject:@{ @"file_data": @{ @"mime_type": file.mimeType ?: @"application/pdf",
													 @"file_uri": file.uri.absoluteString ?: file.identifier } }];
				continue;
			}
			[parts addObject:[self inlinePart:document[@"data"] mimeType:document[@"mediaType"] ?: @"application/pdf"]];
		}
		return @{ @"parts": parts };
	}
	if (attachments.isEmpty) {
		return text;
	}
	if (attachments.videoData != nil || attachments.documents.count > 0) {
		[self failWithCode:kNFKError_InferenceUnsupported
					reason:@"this embeddings service reads text, images, and audio; video and documents need Gemini's native style" error:outError];
		return nil;
	}
	NSMutableArray *parts = [NSMutableArray array];
	if (text.length > 0) {
		[parts addObject:@{ @"type": @"text", @"text": text }];
	}
	for (NSData *png in attachments.imagePNGs) {
		NSString *dataURL = [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
		[parts addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": dataURL } }];
	}
	if (attachments.audioData != nil) {
		[parts addObject:@{ @"type": @"input_audio", @"input_audio": @{ @"data": [attachments.audioData base64EncodedStringWithOptions:0],
																		@"format": attachments.audioFormat ?: @"wav" } }];
	}
	return @[ @{ @"content": parts } ];
}

- (NSDictionary *)inlinePart:(NSData *)data mimeType:(NSString *)mimeType
{
	return @{ @"inline_data": @{ @"mime_type": mimeType, @"data": [data base64EncodedStringWithOptions:0] } };
}

// Gemini names the model in the path, embeds one content per call and several through
// batchEmbedContents, and spells two of OpenAI's fields its own way.
- (NSMutableURLRequest *)geminiRequestForInput:(id)input parameters:(nullable NSDictionary<NSString *, id> *)parameters
{
	NSMutableDictionary<NSString *, id> *options = [NSMutableDictionary dictionary];
	for (NSString *key in parameters) {
		NSString *name = [key isEqualToString:@"dimensions"] ? @"outputDimensionality"
					   : [key isEqualToString:@"task_type"] ? @"taskType" : key;
		options[name] = parameters[key];
	}
	NSString *model = [@"models/" stringByAppendingString:self.modelName ?: @""];
	NSDictionary *body = nil;
	NSString *action = nil;
	if ([input isKindOfClass:NSArray.class]) {
		NSMutableArray *requests = [NSMutableArray array];
		for (NSString *text in input) {
			NSMutableDictionary *each = [NSMutableDictionary dictionaryWithDictionary:options];
			each[@"model"] = model;
			each[@"content"] = @{ @"parts": @[ @{ @"text": text } ] };
			[requests addObject:each];
		}
		body = @{ @"requests": requests };
		action = @":batchEmbedContents";
	} else {
		NSMutableDictionary *single = [NSMutableDictionary dictionaryWithDictionary:options];
		single[@"content"] = [input isKindOfClass:NSString.class] ? @{ @"parts": @[ @{ @"text": input } ] } : input;
		body = single;
		action = @":embedContent";
	}
	NSURL *url = [self.endpointURL URLByAppendingPathComponent:[(self.modelName ?: @"") stringByAppendingString:action]];
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	if (self.apiKey.length > 0) {
		[urlRequest setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}
	return urlRequest;
}

- (nullable NSString *)textForRequest:(NFKInferenceRequest *)request
{
	NSString *prompt = request.prompt;
	if (prompt.length > 0) {
		return prompt;
	}
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	for (NSDictionary *message in request.messages) {
		if ([message isKindOfClass:NSDictionary.class] && [message[@"content"] isKindOfClass:NSString.class]) {
			[pieces addObject:message[@"content"]];
		}
	}
	return pieces.count > 0 ? [pieces componentsJoinedByString:@"\n"] : nil;
}

- (nullable NSDictionary *)responseForInput:(id)input
								 parameters:(nullable NSDictionary<NSString *, id> *)parameters
										URL:(nullable NSURL *)url
									  error:(NSError * _Nullable *)outError
{
	if (url == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
		return nil;
	}
	NSMutableURLRequest *urlRequest = nil;
	if (self.apiStyle == NFKRemoteEmbeddingAPIStyleGeminiNative) {
		urlRequest = [self geminiRequestForInput:input parameters:parameters];
	} else {
		NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
		if (self.modelName.length > 0) {
			body[@"model"] = self.modelName;
		}
		body[@"input"] = input;
		// Parameters fold into the body so a caller sets dimensions, encoding_format, and similar.
		for (NSString *key in parameters) {
			body[key] = parameters[key];
		}
		NSError *encodeError = nil;
		NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
		if (payload == nil) {
			if (outError != NULL) { *outError = encodeError; }
			return nil;
		}
		urlRequest = [NSMutableURLRequest requestWithURL:url];
		urlRequest.HTTPBody = payload;
		[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	}
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	if (data == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object" error:outError];
		return nil;
	}
	return object;
}

// The envelope is data[] of {index, embedding}; the vectors are ordered by index, not by position.
// Gemini's is embedding.values for one content and embeddings[].values for a batch, in order.
/*! Contextualized embeddings: data holds one entry per document, each with a data entry per chunk,
	both ordered here by their index. */
- (nullable NSArray<NSArray<NSArray<NSNumber *> *> *> *)documentVectorsInBody:(NSDictionary *)body
																	 encoding:(nullable NSString *)format
																		error:(NSError * _Nullable *)outError
{
	NSArray *documents = [body[@"data"] isKindOfClass:NSArray.class] ? body[@"data"] : nil;
	if (documents.count == 0) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no embeddings" error:outError];
		return nil;
	}
	NSMutableArray<NSArray<NSArray<NSNumber *> *> *> *vectors = [NSMutableArray array];
	for (NSDictionary *document in [self entriesOrderedByIndex:documents]) {
		NSArray<NSArray<NSNumber *> *> *chunks = [document isKindOfClass:NSDictionary.class]
			? [self vectorsInBody:document encoding:format error:outError] : nil;
		if (chunks == nil) {
			return nil;
		}
		[vectors addObject:chunks];
	}
	return vectors;
}

- (NSArray *)entriesOrderedByIndex:(NSArray *)entries
{
	return [entries sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
		NSNumber *left = [a isKindOfClass:NSDictionary.class] ? a[@"index"] : nil;
		NSNumber *right = [b isKindOfClass:NSDictionary.class] ? b[@"index"] : nil;
		return [(left ?: @0) compare:(right ?: @0)];
	}];
}

- (nullable NSArray<NSArray<NSNumber *> *> *)vectorsInBody:(NSDictionary *)body
												  encoding:(nullable NSString *)format
													 error:(NSError * _Nullable *)outError
{
	if ([body[@"embedding"] isKindOfClass:NSDictionary.class] && [body[@"embedding"][@"values"] isKindOfClass:NSArray.class]) {
		return @[ body[@"embedding"][@"values"] ];
	}
	if ([body[@"embeddings"] isKindOfClass:NSArray.class]) {
		NSMutableArray<NSArray<NSNumber *> *> *vectors = [NSMutableArray array];
		for (NSDictionary *entry in body[@"embeddings"]) {
			NSArray *values = [entry isKindOfClass:NSDictionary.class] ? entry[@"values"] : nil;
			if (![values isKindOfClass:NSArray.class]) {
				[self failWithCode:kNFKError_InferenceBackendFailure reason:@"an embedding entry carries no vector" error:outError];
				return nil;
			}
			[vectors addObject:values];
		}
		return vectors;
	}
	NSArray *entries = body[@"data"];
	if (![entries isKindOfClass:NSArray.class] || entries.count == 0) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no embeddings" error:outError];
		return nil;
	}
	NSMutableArray<NSArray<NSNumber *> *> *vectors = [NSMutableArray array];
	for (NSDictionary *entry in [self entriesOrderedByIndex:entries]) {
		id vector = [entry isKindOfClass:NSDictionary.class] ? entry[@"embedding"] : nil;
		if ([vector isKindOfClass:NSString.class]) {
			vector = NFKRemoteEmbeddingDecodedVector(vector, [format isKindOfClass:NSString.class] ? format : nil);
		}
		if (![vector isKindOfClass:NSArray.class]) {
			[self failWithCode:kNFKError_InferenceBackendFailure reason:@"an embedding entry carries no vector" error:outError];
			return nil;
		}
		[vectors addObject:vector];
	}
	return vectors;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable NFKInferenceResult *)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end

#pragma mark - Batches

@implementation NFKRemoteEmbeddingBackend (NFKRemoteBatch)

- (nullable NSString *)batchModelName
{
	return self.modelName;
}

// A batched embedding is one text: the media parts and the contextualized route are runs of their own.
// Gemini's native style batches the embedContent request it sends for one text.
- (nullable NSURLRequest *)batchURLRequestForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	if (self.isContextualizedModel) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"a batch embeds no contextualized documents" error:outError];
		return nil;
	}
	NSString *text = [self textForRequest:request];
	if (text == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries no text" error:outError];
		return nil;
	}
	if (self.apiStyle == NFKRemoteEmbeddingAPIStyleGeminiNative) {
		return [self geminiRequestForInput:text parameters:request.parameters];
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionaryWithDictionary:request.parameters];
	body[@"model"] = self.modelName;
	body[@"input"] = text;
	NSData *payload = [NSJSONSerialization isValidJSONObject:body] ? [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL] : nil;
	if (payload == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"the parameters are not JSON-serializable" error:outError];
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL ?: [NSURL URLWithString:@"embeddings"]];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.HTTPBody = payload;
	return urlRequest;
}

- (nullable NFKInferenceResult *)batchResultForReply:(NSDictionary *)reply
											 request:(NFKInferenceRequest *)request
											   error:(NSError * _Nullable *)outError
{
	NSArray<NSArray<NSNumber *> *> *vectors = [self vectorsInBody:reply encoding:request.parameters[@"encoding_format"] error:outError];
	return vectors != nil ? [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: vectors.firstObject, NFKRemoteBackendRawKey: reply }] : nil;
}

@end
