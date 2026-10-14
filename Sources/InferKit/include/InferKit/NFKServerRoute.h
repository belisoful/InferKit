//
//  NFKServerRoute.h
//  InferKit
//

#ifndef NFKServerRoute_h
#define NFKServerRoute_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKServerRouteRequest
	@abstract   One HTTP request an app route receives from NFKInferenceServer.
	@discussion The server parsed it and checked its key before the route's handler sees it. The object
				is immutable. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKServerRouteRequest : NSObject

/*! The method, uppercased: GET, POST, PUT, DELETE, and so on. */
@property (nonatomic, readonly, copy) NSString *method;

/*! The percent-decoded path, without the query, as the client sent it. */
@property (nonatomic, readonly, copy) NSString *path;

/*! The prefix of the route that matched, as routePathPrefixes lists it, such as "/myapp/". */
@property (nonatomic, readonly, copy) NSString *pathPrefix;

/*! The query's fields, percent-decoded. A field named twice keeps its last value. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *query;

/*! The header fields, keyed by lowercased name. A field sent twice is joined with ", ". */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *headers;

/*! The body, empty when the request carried none. A chunked body arrives whole. */
@property (nonatomic, readonly, copy) NSData *body;

/*! Whether the client connected from this machine. */
@property (nonatomic, readonly) BOOL fromLoopback;

/*! The client's IP address as text, such as "192.168.1.20" or "::1". An IPv4 client reached over IPv6
	reads as its IPv4 address. */
@property (nonatomic, readonly, copy) NSString *clientAddress;

/*! The name of the server key the request presented, or nil when it presented none the server holds,
	as a client exempt from the key check need not. */
@property (nonatomic, readonly, copy, nullable) NSString *keyName;

/*! A header field's value, the name matched in any case, or nil. */
- (nullable NSString *)valueForHeader:(NSString *)name;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKServerRouteResponse
	@abstract   The writer for one app-route request's reply.
	@discussion Every method may be called from any thread, at once or later, and the writes reach the
				connection in call order. A reply is whole (sendStatus:headers:body: or
				sendStatus:JSONObject:headers:) or streamed with chunked transfer encoding
				(beginStreamWithStatus:headers:, sendStreamData: or sendEventData:, endStream). Only the
				first reply counts; later calls do nothing. A reply that is never sent holds the
				connection until the client goes away. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKServerRouteResponse : NSObject

/*! Whether the reply's head has been written. */
@property (nonatomic, readonly) BOOL headersSent;

/*! Whether the reply has ended. */
@property (nonatomic, readonly, getter=isFinished) BOOL finished;

/*! Whether the client went away before the reply ended. */
@property (nonatomic, readonly, getter=isDisconnected) BOOL disconnected;

/*! Runs at most once, on a background queue, when the client goes away before the reply ends, so a
	long reply can stop its work. */
@property (nonatomic, copy, nullable) void (^ NS_SWIFT_SENDABLE disconnectHandler)(void);

/*! Sends a whole reply. The server writes Content-Length, Connection, and Server itself. */
- (void)sendStatus:(NSInteger)status
		   headers:(nullable NSDictionary<NSString *, NSString *> *)headers
			  body:(nullable NSData *)body;

/*! Sends object as a JSON reply with Content-Type application/json. */
- (void)sendStatus:(NSInteger)status
		JSONObject:(id)object
		   headers:(nullable NSDictionary<NSString *, NSString *> *)headers;

/*! Writes a streamed reply's head; the body follows in sendStreamData: and sendEventData: calls. */
- (void)beginStreamWithStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers;

/*! Writes one chunk of a streamed reply. */
- (void)sendStreamData:(NSData *)data;

/*! Writes one server-sent event: "data: <payload>" and the blank line that ends it. */
- (void)sendEventData:(NSString *)payload;

/*! Ends a streamed reply. */
- (void)endStream;

- (instancetype)init NS_UNAVAILABLE;

@end

/*! Answers one app-route request: it reads request and replies through response, at once or later from
	any thread. A server calls it on a background queue at user-initiated quality of service. Introduced
	in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKServerRouteHandler)(NFKServerRouteRequest *request, NFKServerRouteResponse *response);

NS_ASSUME_NONNULL_END

#endif /* NFKServerRoute_h */
