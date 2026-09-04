//
//  NFKRemoteModerationBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteModerationBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteModerationBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteModerationBackend *backend = [[self alloc] init];
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
	NFKRemoteModerationBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"moderations"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 30.0;
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
	return @"remote-moderation";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSString *text = [self textForRequest:request];
	id image = [request inputForKey:NFKInputImage];
	if (text == nil && image == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries neither text nor an image" error:outError];
	}

	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	// Text alone goes as a string; with an image the input is the multimodal parts list.
	if (image == nil) {
		body[@"input"] = text;
	} else {
		NSString *dataURL = [NFKImageCoding dataURLForImage:image];
		if (dataURL == nil) {
			return [self failWithCode:kNFKError_InferenceMissingInput
							   reason:@"the image under NFKInputImage is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture" error:outError];
		}
		NSMutableArray *parts = [NSMutableArray array];
		if (text != nil) {
			[parts addObject:@{ @"type": @"text", @"text": text }];
		}
		[parts addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": dataURL } }];
		body[@"input"] = parts;
	}
	for (NSString *key in request.parameters) {
		body[key] = request.parameters[key];
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
	NSArray *results = [object isKindOfClass:NSDictionary.class] ? object[@"results"] : nil;
	NSDictionary *verdict = [results isKindOfClass:NSArray.class] && results.count > 0 ? results.firstObject : nil;
	if (![verdict isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no moderation result" error:outError];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputClassifications: [self classificationsInVerdict:verdict],
													NFKOutputStructured: verdict }];
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

// Each category's score is its confidence; the categories keep the index the service listed them
// in, and the list is ordered most confident first.
- (NSArray<NFKClassification *> *)classificationsInVerdict:(NSDictionary *)verdict
{
	NSDictionary *scores = [verdict[@"category_scores"] isKindOfClass:NSDictionary.class] ? verdict[@"category_scores"] : @{};
	NSArray<NSString *> *names = [scores.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray array];
	[names enumerateObjectsUsingBlock:^(NSString *name, NSUInteger index, BOOL *stop) {
		NSNumber *score = scores[name];
		if ([score isKindOfClass:NSNumber.class]) {
			[classifications addObject:[NFKClassification classificationWithLabel:name classIndex:(NSInteger)index confidence:score.doubleValue]];
		}
	}];
	[classifications sortUsingComparator:^NSComparisonResult(NFKClassification *a, NFKClassification *b) {
		return a.confidence > b.confidence ? NSOrderedAscending : a.confidence < b.confidence ? NSOrderedDescending : NSOrderedSame;
	}];
	return classifications;
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
