//
//  NFKHTTPServer.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <Security/Security.h>

NS_ASSUME_NONNULL_BEGIN

/*! One HTTP request as the listener parsed it. Header names are lowercased. */
@interface NFKHTTPRequest : NSObject
@property (nonatomic, copy, readonly) NSString *method;
/*! The percent-decoded path, without the query. */
@property (nonatomic, copy, readonly) NSString *path;
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSString *> *query;
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSString *> *headers;
@property (nonatomic, copy, readonly) NSData *body;
/*! Whether the peer's address is a loopback address, which is to say the client runs on this machine. */
@property (nonatomic, readonly) BOOL fromLoopback;
- (nullable NSString *)valueForHeader:(NSString *)name;
@end

/*! One part of a multipart/form-data body. */
@interface NFKHTTPFormPart : NSObject
@property (nonatomic, copy, readonly) NSString *name;
@property (nonatomic, copy, readonly, nullable) NSString *filename;
@property (nonatomic, copy, readonly, nullable) NSString *contentType;
@property (nonatomic, copy, readonly) NSData *data;
/*! The part's bytes as UTF-8 text. */
@property (nonatomic, copy, readonly) NSString *stringValue;
@end

/*! The parts of a multipart/form-data request, or nil when the body is not one or is malformed. */
NSArray<NFKHTTPFormPart *> * _Nullable NFKHTTPFormParts(NFKHTTPRequest *request);

/*!
	@class      NFKHTTPResponse
	@abstract   The writer for one request's reply.
	@discussion Every method may be called from any thread; the writes reach the connection in call
				order. A reply is either whole (sendStatus:headers:body:) or streamed with chunked
				transfer encoding (beginStream…, sendStreamData:, endStream). disconnectHandler runs
				at most once, when the client goes away before the reply ends.
*/
@interface NFKHTTPResponse : NSObject
@property (nonatomic, readonly) BOOL headersSent;
@property (nonatomic, readonly) BOOL finished;
@property (nonatomic, readonly) BOOL disconnected;
@property (nonatomic, copy, nullable) void (^disconnectHandler)(void);
- (void)sendStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers body:(nullable NSData *)body;
- (void)sendStatus:(NSInteger)status JSONObject:(id)object headers:(nullable NSDictionary<NSString *, NSString *> *)headers;
- (void)beginStreamWithStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers;
- (void)sendStreamData:(NSData *)data;
/*! A server-sent event: "data: <payload>" and the blank line that ends it. */
- (void)sendEventData:(NSString *)payload;
- (void)endStream;
@end

typedef void (^NFKHTTPRequestHandler)(NFKHTTPRequest *request, NFKHTTPResponse *response);

/*!
	@class      NFKHTTPListener
	@abstract   An HTTP/1.1 server over Network.framework.
	@discussion Keeps connections alive, answers Expect: 100-continue, reads Content-Length and chunked
				request bodies, and serves one request at a time per connection. A connection whose
				client closes it while a reply is pending fires that reply's disconnectHandler, which
				is how a served run learns to stop. The handler runs on the connection's queue and
				must not block it.
*/
@interface NFKHTTPListener : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithHandler:(NFKHTTPRequestHandler)handler NS_DESIGNATED_INITIALIZER;

@property (nonatomic, assign) uint16_t port;
@property (nonatomic, assign) BOOL loopbackOnly;
/*! A SecIdentityRef; when set, the listener speaks TLS with it. */
@property (nonatomic, strong, nullable) id TLSIdentity;
/*! The Bonjour service type to advertise under, or nil for none. */
@property (nonatomic, copy, nullable) NSString *serviceType;
/*! The advertised name; nil lets the system use the computer's name. */
@property (nonatomic, copy, nullable) NSString *serviceName;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, NSString *> *TXTRecord;
@property (nonatomic, assign) NSUInteger maximumBodyBytes;
@property (nonatomic, assign) NSTimeInterval idleTimeout;

/*! The port the listener is bound to once started. */
@property (nonatomic, readonly) uint16_t boundPort;

/*! Binds and starts listening; blocks until the listener is ready or has failed. */
- (BOOL)startWithError:(NSError * _Nullable *)outError;
/*! Stops listening and closes every connection. */
- (void)stop;
@end

NS_ASSUME_NONNULL_END
