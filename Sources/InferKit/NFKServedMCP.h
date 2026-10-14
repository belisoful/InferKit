//
//  NFKServedMCP.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKServerObservation.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

/*! The MCP protocol versions the server speaks, newest first. */
FOUNDATION_EXPORT NSArray<NSString *> *NFKMCPProtocolVersions(void);

/*! A tool's result as MCP content: its text, each image it shows the model, the audio file it
	generated, and its video as an embedded resource, with isError and any structured output. */
FOUNDATION_EXPORT NSDictionary<NSString *, id> *NFKMCPCallResult(NFKToolResult *result);

/*!
	@class      NFKServedMCPSessionStore
	@abstract   The MCP sessions a server issued, their server-to-client streams, and their expiry.
	@discussion Every property and method may be used from any thread. An initialize issues a session
				that waits, unlisted and unreported, for the client's first request carrying its id;
				that request starts it. An issued session whose client never sends the id back is
				discarded after 60 seconds, or idleTimeout when shorter, and at most 256 wait at once,
				so issuing one more discards the oldest.

				A timer on the store's serial queue pings each open stream every pingInterval and ends
				a session whose client left missedPingLimit pings unanswered, or one with no stream, no
				request in flight, and no activity for idleTimeout. The timer runs only while a session
				is active or waits for its client.
*/
@interface NFKServedMCPSessionStore : NSObject

/*! Seconds without activity after which a session with no stream ends; 0 never ends one. */
@property (atomic, assign) NSTimeInterval idleTimeout;

/*! Seconds between pings on an open stream; 0 sends none. */
@property (atomic, assign) NSTimeInterval pingInterval;

/*! Unanswered pings after which a session ends. */
@property (atomic, assign) NSUInteger missedPingLimit;

/*! Called off the store's lock, in order, with a snapshot when a session starts (ended NO) or ends. */
@property (atomic, copy, nullable) void (^eventHandler)(NFKServerMCPSession *session, BOOL ended);

/*! A snapshot of each active session, oldest first. An issued session that waits for its client is
	not listed. */
- (NSArray<NFKServerMCPSession *> *)sessions;

/*! Ends the active session; NO when no active session has the identifier. */
- (BOOL)endSessionWithIdentifier:(NSString *)identifier reason:(NFKServerMCPSessionEndReason)reason;

/*! Ends every session started with the key named keyName and discards the issued ones it presented. */
- (void)endSessionsWithKeyName:(NSString *)keyName reason:(NFKServerMCPSessionEndReason)reason;

/*! Ends every session and discards every issued one. */
- (void)endAllSessionsWithReason:(NFKServerMCPSessionEndReason)reason;

/*! Applies a changed idleTimeout or pingInterval to the timer, starting or stopping it as the change
	requires. */
- (void)settingsChanged;

@end

/*!
	@class      NFKServedMCP
	@abstract   The Model Context Protocol endpoint, answered from the server's side.
	@discussion Streamable HTTP. A POST carries one JSON-RPC message or a batch of them and is answered
				with JSON; a message without an id (a notification, or a response to the server) is
				acknowledged with 202. initialize, ping, tools/list, and tools/call are answered; any
				other method is not found.

				Sessions: initialize issues an Mcp-Session-Id in its reply, and the session starts at
				the client's first request that carries it. A request that carries one names an active
				or issued session of the same key name, or is answered 404, which tells the client to
				initialize again. A request without one is served statelessly. GET with a session opens
				the server-to-client event stream, which carries the server's pings; DELETE with a
				session ends it. GET without a session is 405.

				An Origin header naming another host is refused with 403, which keeps a web page from
				reaching a local server through DNS rebinding. A tools/call runs off the connection's
				queue and answers when the tool finishes.
*/
@interface NFKServedMCP : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (void)serveRequest:(NFKHTTPRequest *)request
			response:(NFKHTTPResponse *)response
			   tools:(NSArray<NFKInferenceTool *> *)tools
		instructions:(nullable NSString *)instructions
			sessions:(NFKServedMCPSessionStore *)sessions;

@end

NS_ASSUME_NONNULL_END
