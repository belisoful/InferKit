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
	NSArray<NSString *> *serving = @[ @"openai", @"gemini", @"mistral", @"together", @"openrouter",
									  @"ollama", @"lmstudio", @"llamacpp", @"vllm" ];
	if (![serving containsObject:provider.identifier]) {
		return nil;
	}
	NFKRemoteEmbeddingBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"embeddings"]];
	if ([provider.identifier isEqualToString:@"gemini"]) {
		backend.endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"models"];
		backend.apiStyle = NFKRemoteEmbeddingAPIStyleGeminiNative;
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
	NSDictionary *body = [self responseForInput:input parameters:request.parameters error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray<NSArray<NSNumber *> *> *vectors = [self vectorsInBody:body error:outError];
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
	NSDictionary *body = [self responseForInput:texts parameters:nil error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray<NSArray<NSNumber *> *> *vectors = [self vectorsInBody:body error:outError];
	if (vectors != nil && vectors.count != texts.count) {
		[self failWithCode:kNFKError_InferenceBackendFailure
					reason:[NSString stringWithFormat:@"%lu texts were sent and %lu vectors came back",
							(unsigned long)texts.count, (unsigned long)vectors.count]
					 error:outError];
		return nil;
	}
	return vectors;
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
									  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
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
		urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
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
- (nullable NSArray<NSArray<NSNumber *> *> *)vectorsInBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
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
	NSArray *ordered = [entries sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
		NSNumber *left = [a isKindOfClass:NSDictionary.class] ? a[@"index"] : nil;
		NSNumber *right = [b isKindOfClass:NSDictionary.class] ? b[@"index"] : nil;
		return [(left ?: @0) compare:(right ?: @0)];
	}];
	NSMutableArray<NSArray<NSNumber *> *> *vectors = [NSMutableArray array];
	for (NSDictionary *entry in ordered) {
		NSArray *vector = [entry isKindOfClass:NSDictionary.class] ? entry[@"embedding"] : nil;
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
