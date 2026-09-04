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
	if (provider.apiStyle != NFKRemoteAPIStyleOpenAIChat) {
		return nil;
	}
	NFKRemoteEmbeddingBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"embeddings"]];
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
	if (text == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries neither a prompt nor messages" error:outError];
	}
	NSDictionary *body = [self responseForInput:text parameters:request.parameters error:outError];
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

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

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
- (nullable NSArray<NSArray<NSNumber *> *> *)vectorsInBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
{
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
