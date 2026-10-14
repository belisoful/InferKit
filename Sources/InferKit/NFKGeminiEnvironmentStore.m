//
//  NFKGeminiEnvironmentStore.m
//  InferKit
//

#import <InferKit/NFKGeminiEnvironmentStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteTransportPrivate.h"

@implementation NFKGeminiEnvironmentFile

- (instancetype)initWithEntry:(NSDictionary<NSString *, id> *)entry
{
	self = [super init];
	if (self != nil) {
		NSString *path = [entry[@"path"] isKindOfClass:NSString.class] ? entry[@"path"] : @"";
		_path = [path copy];
		_name = [([entry[@"name"] isKindOfClass:NSString.class] ? entry[@"name"] : path.lastPathComponent) copy];
		_directory = [entry[@"type"] isEqual:@"DIRECTORY"];
		// size_bytes is an int64 written as a string, the way Google's JSON spells 64-bit integers.
		id size = entry[@"size_bytes"];
		_size = [size isKindOfClass:NSString.class] || [size isKindOfClass:NSNumber.class] ? [size longLongValue] : 0;
		_mimeType = [([entry[@"mime_type"] isKindOfClass:NSString.class] ? entry[@"mime_type"] : nil) copy];
		_raw = [entry copy];
	}
	return self;
}

@end

/*! Whether name is one path component that stays where it is put: non-empty, without a slash, and
	neither . nor .., which the URL would resolve up the tree to another resource under the same key. */
static BOOL NFKGeminiEnvironmentNameIsSafe(NSString * _Nullable name)
{
	return name.length > 0 && ![name containsString:@"/"] && ![name isEqualToString:@"."] && ![name isEqualToString:@".."];
}

static NSString * const NFKGeminiEnvironmentURLFailure =
	@"no endpoint URL is set, the environment id or a component of the path is empty, '.', '..', or carries a slash, "
	@"or a write's endpoint has no API version segment to put upload/ before";

/*! Whether a path component names an API version (v1, v1beta, v2alpha1), the segment Google's upload/
	prefix goes before. */
static BOOL NFKGeminiEnvironmentIsVersion(NSString *component)
{
	return [component rangeOfString:@"^v[0-9]+[a-z0-9]*$" options:NSRegularExpressionSearch].location != NSNotFound;
}

@implementation NFKGeminiEnvironmentStore

@synthesize session = _session;

+ (nullable instancetype)environmentStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	if (![provider.identifier isEqualToString:@"gemini"]) {
		return nil;
	}
	NFKGeminiEnvironmentStore *store = [[self alloc] init];
	// The preset's base is the OpenAI layer under v1beta; the environments collection is its sibling.
	store.endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"environments"];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		store.session = pinned;
	}
	store.apiKey = apiKey;
	return store;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/environments"];
		_timeout = 120.0;
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

#pragma mark Environments

- (nullable NSArray<NSDictionary<NSString *, id> *> *)environmentsWithError:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"no endpoint URL is set" error:outError];
	}
	NSMutableArray<NSDictionary *> *environments = [NSMutableArray array];
	NSMutableSet<NSString *> *seenTokens = [NSMutableSet set];
	NSString *pageToken = nil;
	do {
		NSDictionary *query = pageToken.length > 0 ? @{ @"page_token": pageToken } : nil;
		NSDictionary *page = [self JSONForRequest:[self requestTo:[self URLOf:self.endpointURL query:query] method:@"GET" body:nil contentType:nil]
											error:outError];
		if (page == nil) {
			return nil;
		}
		for (NSDictionary *entry in [page[@"environments"] isKindOfClass:NSArray.class] ? page[@"environments"] : @[]) {
			if ([entry isKindOfClass:NSDictionary.class]) {
				[environments addObject:entry];
			}
		}
		pageToken = [page[@"next_page_token"] isKindOfClass:NSString.class] ? page[@"next_page_token"] : nil;
		// A service that hands back a token it already gave would otherwise loop forever.
		if (pageToken.length > 0 && [seenTokens containsObject:pageToken]) {
			break;
		}
		if (pageToken.length > 0) {
			[seenTokens addObject:pageToken];
		}
	} while (pageToken.length > 0);
	return environments;
}

- (BOOL)deleteEnvironment:(NSString *)environmentIdentifier error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil || !NFKGeminiEnvironmentNameIsSafe(environmentIdentifier)) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"no endpoint URL is set, or the environment id is empty or not a single path component" error:outError];
		return NO;
	}
	NSURL *url = [self.endpointURL URLByAppendingPathComponent:environmentIdentifier];
	return [self JSONForRequest:[self requestTo:url method:@"DELETE" body:nil contentType:nil] error:outError] != nil;
}

#pragma mark Files

- (nullable NSArray<NFKGeminiEnvironmentFile *> *)filesInEnvironment:(NSString *)environmentIdentifier
															  atPath:(nullable NSString *)path
														   recursive:(BOOL)recursive
															   error:(NSError * _Nullable *)outError
{
	NSURL *url = [self URLForPath:path ?: @"" environment:environmentIdentifier query:recursive ? @{ @"recursive": @"true" } : nil upload:NO];
	if (url == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:NFKGeminiEnvironmentURLFailure error:outError];
	}
	NSDictionary *reply = [self JSONForRequest:[self requestTo:url method:@"GET" body:nil contentType:nil] error:outError];
	return reply == nil ? nil : [self filesInReply:reply];
}

- (nullable NSData *)contentsOfFileAtPath:(NSString *)path
							inEnvironment:(NSString *)environmentIdentifier
								recursive:(BOOL)recursive
									error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *query = [NSMutableDictionary dictionaryWithObject:@"media" forKey:@"alt"];
	if (recursive) {
		query[@"recursive"] = @"true";
	}
	NSURL *url = [self URLForPath:path environment:environmentIdentifier query:query upload:NO];
	if (url == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:NFKGeminiEnvironmentURLFailure error:outError];
	}
	return [self dataForRequest:[self requestTo:url method:@"GET" body:nil contentType:nil] error:outError];
}

- (nullable NSArray<NFKGeminiEnvironmentFile *> *)writeData:(NSData *)data
													 toPath:(NSString *)path
											  inEnvironment:(NSString *)environmentIdentifier
												  overwrite:(BOOL)overwrite
													extract:(BOOL)extract
													  error:(NSError * _Nullable *)outError
{
	if (path.length == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"a write needs a path in the environment" error:outError];
	}
	NSMutableDictionary *query = [NSMutableDictionary dictionary];
	if (overwrite) {
		query[@"overwrite"] = @"true";
	}
	if (extract) {
		query[@"extract"] = @"true";
	}
	NSURL *url = [self URLForPath:path environment:environmentIdentifier query:query upload:YES];
	if (url == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:NFKGeminiEnvironmentURLFailure error:outError];
	}
	NSString *contentType = extract ? @"application/x-tar" : @"application/octet-stream";
	NSDictionary *reply = [self JSONForRequest:[self requestTo:url method:@"PUT" body:data contentType:contentType] error:outError];
	return reply == nil ? nil : [self filesInReply:reply];
}

- (NSArray<NFKGeminiEnvironmentFile *> *)filesInReply:(NSDictionary *)reply
{
	NSMutableArray<NFKGeminiEnvironmentFile *> *files = [NSMutableArray array];
	for (NSDictionary *entry in [reply[@"files"] isKindOfClass:NSArray.class] ? reply[@"files"] : @[]) {
		if ([entry isKindOfClass:NSDictionary.class]) {
			[files addObject:[[NFKGeminiEnvironmentFile alloc] initWithEntry:entry]];
		}
	}
	return files;
}

#pragma mark The wire

/*! The environment's files at path, split at its slashes, each component percent-encoded by
	URLByAppendingPathComponent:. A nil or empty path is the environment's root. nil when the id or a
	component is not one safe name (see NFKGeminiEnvironmentNameIsSafe), so "a//b" and "/" fail too.
	Uploads go to the collection under Google's upload/ segment, which sits before the API version:
	…/upload/v1beta/environments for …/v1beta/environments, at the root or under a proxy's path; an
	endpoint whose parent segment is no version has nowhere to put it, and a write to it is nil. */
- (nullable NSURL *)URLForPath:(nullable NSString *)path
				   environment:(NSString *)environmentIdentifier
						 query:(nullable NSDictionary<NSString *, NSString *> *)query
						upload:(BOOL)upload
{
	NSURL *url = self.endpointURL;
	if (url == nil || !NFKGeminiEnvironmentNameIsSafe(environmentIdentifier)) {
		return nil;
	}
	if (upload) {
		NSURL *version = url.URLByDeletingLastPathComponent;
		if (!NFKGeminiEnvironmentIsVersion(version.lastPathComponent)) {
			return nil;
		}
		url = [[[version.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"upload"]
				URLByAppendingPathComponent:version.lastPathComponent] URLByAppendingPathComponent:url.lastPathComponent];
	}
	url = [[url URLByAppendingPathComponent:environmentIdentifier] URLByAppendingPathComponent:@"files"];
	for (NSString *component in path.length > 0 ? [path componentsSeparatedByString:@"/"] : @[]) {
		if (!NFKGeminiEnvironmentNameIsSafe(component)) {
			return nil;
		}
		url = [url URLByAppendingPathComponent:component];
	}
	return [self URLOf:url query:query];
}

- (nullable NSURL *)URLOf:(NSURL *)url query:(nullable NSDictionary<NSString *, NSString *> *)query
{
	NSURLComponents *components = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
	for (NSString *name in [query.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		[items addObject:[NSURLQueryItem queryItemWithName:name value:query[name]]];
	}
	components.queryItems = items.count > 0 ? items : nil;
	return components.URL;
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method body:(nullable NSData *)body contentType:(nullable NSString *)contentType
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	request.HTTPBody = body;
	if (contentType != nil) {
		[request setValue:contentType forHTTPHeaderField:@"Content-Type"];
	}
	if (self.apiKey.length > 0) {
		[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}
	return request;
}

- (nullable NSData *)dataForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
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
	return data;
}

/*! The reply as a JSON object; an empty body (a deletion) reads as an empty object. */
- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSData *data = [self dataForRequest:request error:outError];
	if (data == nil) {
		return nil;
	}
	if (data.length == 0) {
		return @{};
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![reply isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the reply is not a JSON object" error:outError];
	}
	return reply;
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
