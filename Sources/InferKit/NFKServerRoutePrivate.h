//
//  NFKServerRoutePrivate.h
//  InferKit
//

#import <InferKit/NFKServerRoute.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

@interface NFKServerRouteRequest (NFKPrivate)

/*! The app route's view of a parsed request matched under pathPrefix. */
+ (instancetype)requestWithHTTPRequest:(NFKHTTPRequest *)request pathPrefix:(NSString *)pathPrefix;

@end

@interface NFKServerRouteResponse (NFKPrivate)

/*! A writer that forwards to the listener's reply. */
+ (instancetype)responseWithHTTPResponse:(NFKHTTPResponse *)response;

@end

/*! pathPrefix with one leading and one trailing slash, or nil with an error when it is malformed or
	names a route the server keeps for itself (/v1 or /mcp). */
NSString * _Nullable NFKServerRouteNormalizedPrefix(NSString *pathPrefix, NSError * _Nullable * _Nullable error);

/*! Whether path lies under the normalized prefix: the prefix without its trailing slash, or anything
	below it. */
BOOL NFKServerRoutePathMatchesPrefix(NSString *path, NSString *prefix);

NS_ASSUME_NONNULL_END
