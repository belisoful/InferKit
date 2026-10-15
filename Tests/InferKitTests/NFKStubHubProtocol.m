//
//  NFKStubHubProtocol.m
//  InferKitTests
//

#import "NFKStubHubProtocol.h"
#import <CommonCrypto/CommonCrypto.h>

@implementation NFKStubHubResponse

+ (instancetype)responseWithStatus:(NSInteger)status body:(nullable NSData *)body
{
	NFKStubHubResponse *response = [[NFKStubHubResponse alloc] init];
	response.status = status;
	response.body = body ?: [NSData data];
	response.headers = @{};
	response.dropsAfterByteCount = -1;
	return response;
}

@end

static NSMutableDictionary<NSString *, NFKStubHubResponse *> *NFKStubHubResponses;
static NSMutableArray<NSURLRequest *> *NFKStubHubRequests;
static NSMutableSet<NSString *> *NFKStubHubDroppedPaths;

@implementation NFKStubHubProtocol

+ (void)initialize
{
	if (self == NFKStubHubProtocol.class) {
		NFKStubHubResponses = [NSMutableDictionary dictionary];
		NFKStubHubRequests = [NSMutableArray array];
		NFKStubHubDroppedPaths = [NSMutableSet set];
	}
}

+ (NSURL *)endpointURL
{
	return [NSURL URLWithString:@"http://hub.test"];
}

+ (NSURLSession *)session
{
	NSURLSessionConfiguration *configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];
	configuration.protocolClasses = @[ NFKStubHubProtocol.class ];
	return [NSURLSession sessionWithConfiguration:configuration];
}

+ (void)reset
{
	@synchronized (self) {
		[NFKStubHubResponses removeAllObjects];
		[NFKStubHubRequests removeAllObjects];
		[NFKStubHubDroppedPaths removeAllObjects];
	}
}

+ (void)setResponse:(NFKStubHubResponse *)response forPath:(NSString *)path
{
	@synchronized (self) {
		NFKStubHubResponses[path] = response;
	}
}

+ (NSString *)sha256OfData:(NSData *)data
{
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
	NSMutableString *hex = [NSMutableString string];
	for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
		[hex appendFormat:@"%02x", digest[i]];
	}
	return hex;
}

+ (NSDictionary<NSString *, NFKStubHubResponse *> *)serveRepo:(NSString *)repo
													 revision:(NSString *)revision
														files:(NSDictionary<NSString *, NSData *> *)files
													 lfsPaths:(NSSet<NSString *> *)lfsPaths
{
	NSMutableArray *listing = [NSMutableArray array];
	NSMutableDictionary<NSString *, NFKStubHubResponse *> *responses = [NSMutableDictionary dictionary];
	for (NSString *path in [files.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSData *data = files[path];
		NSString *sha = [self sha256OfData:data];
		NSMutableDictionary *entry = [@{ @"type": @"file", @"path": path, @"size": @(data.length),
										 @"oid": [sha substringToIndex:40] } mutableCopy];
		if ([lfsPaths containsObject:path]) {
			entry[@"lfs"] = @{ @"oid": sha, @"size": @(data.length), @"pointerSize": @134 };
		}
		[listing addObject:entry];
		NFKStubHubResponse *response = [NFKStubHubResponse responseWithStatus:200 body:data];
		response.headers = @{ @"ETag": [NSString stringWithFormat:@"\"%@\"", sha] };
		[self setResponse:response forPath:[NSString stringWithFormat:@"/%@/resolve/%@/%@", repo, revision, path]];
		responses[path] = response;
	}
	[listing addObject:@{ @"type": @"directory", @"path": @"folder", @"oid": @"0" }];
	NSData *tree = [NSJSONSerialization dataWithJSONObject:listing options:0 error:NULL];
	[self setResponse:[NFKStubHubResponse responseWithStatus:200 body:tree]
			  forPath:[NSString stringWithFormat:@"/api/models/%@/tree/%@", repo, revision]];
	return responses;
}

+ (NSArray<NSURLRequest *> *)requests
{
	@synchronized (self) {
		return [NFKStubHubRequests copy];
	}
}

+ (NSArray<NSURLRequest *> *)requestsForPath:(NSString *)path
{
	return [self.requests filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(NSURLRequest *request, NSDictionary *bindings) {
		return [request.URL.path isEqualToString:path];
	}]];
}

+ (BOOL)canInitWithRequest:(NSURLRequest *)request
{
	return [request.URL.host isEqualToString:@"hub.test"];
}

+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request
{
	return request;
}

- (void)startLoading
{
	NSString *path = self.request.URL.path;
	NFKStubHubResponse *script = nil;
	BOOL drops = NO;
	@synchronized (NFKStubHubProtocol.class) {
		[NFKStubHubRequests addObject:self.request];
		script = NFKStubHubResponses[path];
		if (script.dropsAfterByteCount >= 0 && ![NFKStubHubDroppedPaths containsObject:path]) {
			[NFKStubHubDroppedPaths addObject:path];
			drops = YES;
		}
	}
	if (script == nil) {
		script = [NFKStubHubResponse responseWithStatus:404 body:[@"not found" dataUsingEncoding:NSUTF8StringEncoding]];
	}

	NSInteger status = script.status;
	NSData *body = script.body;
	NSMutableDictionary<NSString *, NSString *> *headers = [script.headers mutableCopy];
	NSString *range = [self.request valueForHTTPHeaderField:@"Range"];
	long long start = 0;
	if (status == 200 && range != nil && sscanf(range.UTF8String, "bytes=%lld-", &start) == 1 && start < (long long)body.length) {
		status = 206;
		headers[@"Content-Range"] = [NSString stringWithFormat:@"bytes %lld-%lu/%lu", start,
									 (unsigned long)body.length - 1, (unsigned long)body.length];
		body = [body subdataWithRange:NSMakeRange((NSUInteger)start, body.length - (NSUInteger)start)];
	}
	headers[@"Content-Length"] = [NSString stringWithFormat:@"%lu", (unsigned long)body.length];
	headers[@"Accept-Ranges"] = @"bytes";
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:self.request.URL
															  statusCode:status
															 HTTPVersion:@"HTTP/1.1"
															headerFields:headers];
	[self.client URLProtocol:self didReceiveResponse:response cacheStoragePolicy:NSURLCacheStorageNotAllowed];
	if ([self.request.HTTPMethod isEqualToString:@"HEAD"]) {
		[self.client URLProtocolDidFinishLoading:self];
		return;
	}

	if (drops) {
		NSUInteger count = MIN((NSUInteger)script.dropsAfterByteCount, body.length);
		[self.client URLProtocol:self didLoadData:[body subdataWithRange:NSMakeRange(0, count)]];
		[self.client URLProtocol:self didFailWithError:[NSError errorWithDomain:NSURLErrorDomain
																		   code:NSURLErrorNetworkConnectionLost
																	   userInfo:nil]];
		return;
	}
	NSUInteger chunk = script.chunkSize > 0 ? script.chunkSize : MAX(body.length, (NSUInteger)1);
	for (NSUInteger offset = 0; offset < body.length; offset += chunk) {
		NSUInteger length = MIN(chunk, body.length - offset);
		[self.client URLProtocol:self didLoadData:[body subdataWithRange:NSMakeRange(offset, length)]];
		if (script.stallsAfterFirstChunk) {
			return;
		}
	}
	[self.client URLProtocolDidFinishLoading:self];
}

- (void)stopLoading
{
}

@end
