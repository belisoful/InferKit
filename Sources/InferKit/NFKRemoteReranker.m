//
//  NFKRemoteReranker.m
//  InferKit
//

#import <InferKit/NFKRemoteReranker.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteReranker

@synthesize session = _session;

+ (instancetype)rerankerWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteReranker *reranker = [[self alloc] init];
	reranker.endpointURL = endpointURL;
	return reranker;
}

+ (nullable instancetype)rerankerForProvider:(NFKRemoteProvider *)provider
									  apiKey:(nullable NSString *)apiKey
								   modelName:(nullable NSString *)modelName
{
	if (provider.apiStyle != NFKRemoteAPIStyleOpenAIChat) {
		return nil;
	}
	NFKRemoteReranker *reranker = [self rerankerWithEndpointURL:[provider URLForPath:@"rerank"]];
	reranker.apiKey = apiKey;
	reranker.modelName = modelName;
	return reranker;
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

#pragma mark Scoring

// The reply lists {index, relevance_score} in relevance order; the scores are put back in the
// documents' order so the ith score belongs to the ith document.
- (nullable NSArray<NSNumber *> *)scoresForQuery:(NSString *)query
									   documents:(NSArray<NSString *> *)documents
										   error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	if (documents.count == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"no documents to rank" error:outError];
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"query"] = query;
	body[@"documents"] = documents;
	body[@"top_n"] = @(documents.count);
	body[@"return_documents"] = @NO;

	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.endpointURL];
	request.HTTPMethod = @"POST";
	request.timeoutInterval = self.timeout;
	request.HTTPBody = payload;
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
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
	if (![results isKindOfClass:NSArray.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no results" error:outError];
	}
	NSMutableArray<NSNumber *> *scores = [NSMutableArray array];
	for (NSUInteger i = 0; i < documents.count; i++) {
		[scores addObject:@0];
	}
	NSUInteger placed = 0;
	for (NSDictionary *entry in results) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSNumber *index = entry[@"index"];
		NSNumber *score = entry[@"relevance_score"] ?: entry[@"score"];
		if ([index isKindOfClass:NSNumber.class] && [score isKindOfClass:NSNumber.class]
			&& index.integerValue >= 0 && (NSUInteger)index.integerValue < documents.count) {
			scores[(NSUInteger)index.integerValue] = score;
			placed++;
		}
	}
	if (placed == 0) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response scored none of the documents" error:outError];
	}
	return scores;
}

- (nullable NSArray<NSNumber *> *)rankedIndicesForQuery:(NSString *)query
											  documents:(NSArray<NSString *> *)documents
												  error:(NSError * _Nullable *)outError
{
	NSArray<NSNumber *> *scores = [self scoresForQuery:query documents:documents error:outError];
	if (scores == nil) {
		return nil;
	}
	NSMutableArray<NSNumber *> *indices = [NSMutableArray array];
	for (NSUInteger i = 0; i < scores.count; i++) {
		[indices addObject:@(i)];
	}
	[indices sortUsingComparator:^NSComparisonResult(NSNumber *a, NSNumber *b) {
		return [scores[b.unsignedIntegerValue] compare:scores[a.unsignedIntegerValue]];
	}];
	return indices;
}

- (nullable NSNumber *)scoreForQuery:(NSString *)query
							document:(NSString *)document
							   error:(NSError * _Nullable *)outError
{
	return [self scoresForQuery:query documents:@[ document ] error:outError].firstObject;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable id)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
