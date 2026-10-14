//
//  NFKServerRoute.m
//  InferKit
//

#import "NFKServerRoutePrivate.h"
#import <InferKit/NFKInferenceServer.h>

@interface NFKServerRouteRequest ()
@property (nonatomic, copy, readwrite) NSString *method;
@property (nonatomic, copy, readwrite) NSString *path;
@property (nonatomic, copy, readwrite) NSString *pathPrefix;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSString *> *query;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSString *> *headers;
@property (nonatomic, copy, readwrite) NSData *body;
@property (nonatomic, assign, readwrite) BOOL fromLoopback;
@property (nonatomic, copy, readwrite) NSString *clientAddress;
@property (nonatomic, copy, readwrite, nullable) NSString *keyName;
- (instancetype)initPrivately;
@end

@implementation NFKServerRouteRequest

+ (instancetype)requestWithHTTPRequest:(NFKHTTPRequest *)request pathPrefix:(NSString *)pathPrefix
{
	NFKServerRouteRequest *routed = [[self alloc] initPrivately];
	routed.method = request.method;
	routed.path = request.path;
	routed.pathPrefix = pathPrefix;
	routed.query = request.query;
	routed.headers = request.headers;
	routed.body = request.body;
	routed.fromLoopback = request.fromLoopback;
	routed.clientAddress = request.clientAddress;
	routed.keyName = request.keyName;
	return routed;
}

- (instancetype)initPrivately
{
	return [super init];
}

- (nullable NSString *)valueForHeader:(NSString *)name
{
	return self.headers[name.lowercaseString];
}

@end

@interface NFKServerRouteResponse ()
@property (nonatomic, strong) NFKHTTPResponse *reply;
- (instancetype)initPrivately;
@end

@implementation NFKServerRouteResponse

+ (instancetype)responseWithHTTPResponse:(NFKHTTPResponse *)response
{
	NFKServerRouteResponse *writer = [[self alloc] initPrivately];
	writer.reply = response;
	return writer;
}

- (instancetype)initPrivately
{
	return [super init];
}

- (BOOL)headersSent
{
	return self.reply.headersSent;
}

- (BOOL)isFinished
{
	return self.reply.finished;
}

- (BOOL)isDisconnected
{
	return self.reply.disconnected;
}

- (nullable void (^)(void))disconnectHandler
{
	return self.reply.disconnectHandler;
}

- (void)setDisconnectHandler:(nullable void (^)(void))disconnectHandler
{
	self.reply.disconnectHandler = disconnectHandler;
}

- (void)sendStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers body:(nullable NSData *)body
{
	[self.reply sendStatus:status headers:headers body:body];
}

- (void)sendStatus:(NSInteger)status JSONObject:(id)object headers:(nullable NSDictionary<NSString *, NSString *> *)headers
{
	[self.reply sendStatus:status JSONObject:object headers:headers];
}

- (void)beginStreamWithStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers
{
	[self.reply beginStreamWithStatus:status headers:headers];
}

- (void)sendStreamData:(NSData *)data
{
	[self.reply sendStreamData:data];
}

- (void)sendEventData:(NSString *)payload
{
	[self.reply sendEventData:payload];
}

- (void)endStream
{
	[self.reply endStream];
}

@end

static BOOL NFKServerRouteRefuse(NSString *reason, NSError * _Nullable * _Nullable error)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceServerErrorDomain code:NFKInferenceServerErrorInvalidConfiguration
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

NSString * _Nullable NFKServerRouteNormalizedPrefix(NSString *pathPrefix, NSError * _Nullable * _Nullable error)
{
	if (![pathPrefix hasPrefix:@"/"]) {
		NFKServerRouteRefuse([NSString stringWithFormat:@"the route prefix \"%@\" does not start with /", pathPrefix], error);
		return nil;
	}
	NSString *trimmed = [pathPrefix hasSuffix:@"/"] ? [pathPrefix substringToIndex:pathPrefix.length - 1] : pathPrefix;
	NSArray<NSString *> *segments = [[trimmed substringFromIndex:MIN(trimmed.length, 1)] componentsSeparatedByString:@"/"];
	for (NSString *segment in segments) {
		if (segment.length == 0 || [segment isEqualToString:@"."] || [segment isEqualToString:@".."]) {
			NFKServerRouteRefuse([NSString stringWithFormat:@"the route prefix \"%@\" has an empty, \".\", or \"..\" segment; the root is the server's", pathPrefix], error);
			return nil;
		}
	}
	NSString *first = segments.firstObject.lowercaseString;
	if ([first isEqualToString:@"v1"] || [first isEqualToString:@"mcp"]) {
		NFKServerRouteRefuse([NSString stringWithFormat:@"the route prefix \"%@\" lies under /%@, which the server keeps for itself", pathPrefix, first], error);
		return nil;
	}
	return [trimmed stringByAppendingString:@"/"];
}

BOOL NFKServerRoutePathMatchesPrefix(NSString *path, NSString *prefix)
{
	return [path hasPrefix:prefix] || [path isEqualToString:[prefix substringToIndex:prefix.length - 1]];
}
