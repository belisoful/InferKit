//
//  NFKRemoteClassifierBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteClassifierBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteClassifierBackend

@synthesize session = _session;

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NFKRemoteClassifierBackend *backend = [[self alloc] init];
	if ([provider.identifier isEqualToString:@"mistral"]) {
		backend.endpointURL = [provider URLForPath:@"classifications"];
		backend.apiStyle = NFKRemoteClassifierAPIStyleMistral;
	} else if ([provider.identifier isEqualToString:@"vllm"]) {
		backend.endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"classify"];
		backend.apiStyle = NFKRemoteClassifierAPIStyleVLLM;
	} else {
		return nil;
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
	return @"remote-classifier";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;
	NSURL *url = self.endpointURL;
	BOOL conversation = self.apiStyle == NFKRemoteClassifierAPIStyleMistral && request.prompt.length == 0 && request.messages.count > 0;
	if (conversation) {
		url = [[self.endpointURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"chat"] URLByAppendingPathComponent:@"classifications"];
		body[@"input"] = @{ @"messages": request.messages };
	} else if (request.prompt.length > 0) {
		body[@"input"] = request.prompt;
	} else {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages" error:outError];
	}
	for (NSString *key in request.parameters) {
		body[key] = request.parameters[key];
	}

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the classification call failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![reply isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object" error:outError];
	}
	NSArray<NFKClassification *> *classifications = self.apiStyle == NFKRemoteClassifierAPIStyleVLLM
		? [self vLLMClassificationsIn:reply] : [self mistralClassificationsIn:reply];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputClassifications: classifications, NFKRemoteBackendRawKey: reply }];
}

#pragma mark Response

// results[0] maps each target to {scores: {label: score}}; with one target the label stands alone.
- (NSArray<NFKClassification *> *)mistralClassificationsIn:(NSDictionary *)reply
{
	NSArray *results = [reply[@"results"] isKindOfClass:NSArray.class] ? reply[@"results"] : @[];
	NSDictionary *first = [results.firstObject isKindOfClass:NSDictionary.class] ? results.firstObject : @{};
	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray array];
	BOOL severalTargets = first.count > 1;
	for (NSString *target in [first.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSDictionary *scores = [first[target] isKindOfClass:NSDictionary.class] && [first[target][@"scores"] isKindOfClass:NSDictionary.class]
			? first[target][@"scores"] : @{};
		NSArray<NSString *> *labels = [scores.allKeys sortedArrayUsingSelector:@selector(compare:)];
		[labels enumerateObjectsUsingBlock:^(NSString *label, NSUInteger index, BOOL *stop) {
			NSNumber *score = scores[label];
			if ([score isKindOfClass:NSNumber.class]) {
				NSString *name = severalTargets ? [NSString stringWithFormat:@"%@/%@", target, label] : label;
				[classifications addObject:[NFKClassification classificationWithLabel:name classIndex:(NSInteger)index confidence:score.doubleValue]];
			}
		}];
	}
	return [self sorted:classifications];
}

// data[0].probs is one probability per class index; the label names the top class only.
- (NSArray<NFKClassification *> *)vLLMClassificationsIn:(NSDictionary *)reply
{
	NSArray *data = [reply[@"data"] isKindOfClass:NSArray.class] ? reply[@"data"] : @[];
	NSDictionary *first = [data.firstObject isKindOfClass:NSDictionary.class] ? data.firstObject : @{};
	NSArray *probabilities = [first[@"probs"] isKindOfClass:NSArray.class] ? first[@"probs"] : @[];
	NSString *topLabel = [first[@"label"] isKindOfClass:NSString.class] ? first[@"label"] : nil;
	__block NSUInteger topIndex = NSNotFound;
	__block double topProbability = -1;
	[probabilities enumerateObjectsUsingBlock:^(NSNumber *probability, NSUInteger index, BOOL *stop) {
		if ([probability isKindOfClass:NSNumber.class] && probability.doubleValue > topProbability) {
			topProbability = probability.doubleValue;
			topIndex = index;
		}
	}];
	NSMutableArray<NFKClassification *> *classifications = [NSMutableArray array];
	[probabilities enumerateObjectsUsingBlock:^(NSNumber *probability, NSUInteger index, BOOL *stop) {
		if ([probability isKindOfClass:NSNumber.class]) {
			NSString *label = index == topIndex ? topLabel : nil;
			[classifications addObject:[NFKClassification classificationWithLabel:label classIndex:(NSInteger)index confidence:probability.doubleValue]];
		}
	}];
	return [self sorted:classifications];
}

- (NSArray<NFKClassification *> *)sorted:(NSMutableArray<NFKClassification *> *)classifications
{
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
