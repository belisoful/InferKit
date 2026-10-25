//
//  NFKRemoteWebSearch.m
//  InferKit
//

#import <InferKit/NFKRemoteWebSearch.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteTransportPrivate.h"

@interface NFKWebSearchResult ()
@property (nonatomic, copy, readwrite) NSString *title;
@property (nonatomic, copy, readwrite) NSURL *URL;
@property (nonatomic, copy, readwrite) NSString *snippet;
@property (nonatomic, copy, readwrite, nullable) NSString *date;
@property (nonatomic, copy, readwrite, nullable) NSString *lastUpdated;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKWebSearchResult

+ (nullable instancetype)resultWithEntry:(NSDictionary<NSString *, id> *)entry
{
	if (![entry isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *title = [entry[@"title"] isKindOfClass:NSString.class] ? entry[@"title"] : nil;
	NSURL *url = [entry[@"url"] isKindOfClass:NSString.class] ? [NSURL URLWithString:entry[@"url"]] : nil;
	NSString *snippet = [entry[@"snippet"] isKindOfClass:NSString.class] ? entry[@"snippet"] : nil;
	if (title == nil || url == nil || snippet == nil) {
		return nil;
	}
	NFKWebSearchResult *result = [[self alloc] initPrivate];
	result.title = title;
	result.URL = url;
	result.snippet = snippet;
	result.date = [entry[@"date"] isKindOfClass:NSString.class] ? entry[@"date"] : nil;
	result.lastUpdated = [entry[@"last_updated"] isKindOfClass:NSString.class] ? entry[@"last_updated"] : nil;
	result.raw = entry;
	return result;
}

- (instancetype)initPrivate
{
	return [super init];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

@end

@implementation NFKRemoteWebSearch

@synthesize session = _session;

+ (instancetype)webSearchWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteWebSearch *search = [[self alloc] init];
	search.endpointURL = endpointURL;
	return search;
}

// Perplexity serves search at its host's root, beside the /v1 base.
+ (nullable instancetype)webSearchForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	if (![provider.identifier isEqualToString:@"perplexity"]) {
		return nil;
	}
	NFKRemoteWebSearch *search = [self webSearchWithEndpointURL:[provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"search"]];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		search.session = pinned;
	}
	search.apiKey = apiKey;
	return search;
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

#pragma mark Searching

- (nullable NSArray<NFKWebSearchResult *> *)resultsForQuery:(NSString *)query error:(NSError * _Nullable *)outError
{
	return [self resultsForQuery:query parameters:nil error:outError];
}

- (nullable NSArray<NFKWebSearchResult *> *)resultsForQuery:(id)query
												 parameters:(nullable NSDictionary<NSString *, id> *)parameters
													  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	BOOL queries = [query isKindOfClass:NSArray.class] && [query count] > 0;
	if (!queries && !([query isKindOfClass:NSString.class] && [query length] > 0)) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the search names no query" error:outError];
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionaryWithDictionary:parameters ?: @{}];
	body[@"query"] = query;
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization isValidJSONObject:body] ? [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError] : nil;
	if (payload == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the parameters are not JSON-serializable" error:outError];
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
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the search failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	NSArray *entries = [reply isKindOfClass:NSDictionary.class] && [reply[@"results"] isKindOfClass:NSArray.class] ? reply[@"results"] : nil;
	if (entries == nil) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no results" error:outError];
	}
	NSMutableArray<NFKWebSearchResult *> *results = [NSMutableArray array];
	for (id entry in [self flattened:entries]) {
		NFKWebSearchResult *result = [NFKWebSearchResult resultWithEntry:entry];
		if (result != nil) {
			[results addObject:result];
		}
	}
	return results;
}

/*! The entries in order, with a list nested per query taken in its place. Perplexity's reference
	answers one flat list for several queries; a nested one is read the same way. */
- (NSArray *)flattened:(NSArray *)entries
{
	NSMutableArray *flat = [NSMutableArray array];
	for (id entry in entries) {
		if ([entry isKindOfClass:NSArray.class]) {
			[flat addObjectsFromArray:entry];
		} else {
			[flat addObject:entry];
		}
	}
	return flat;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (nullable id)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
