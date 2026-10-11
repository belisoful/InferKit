//
//  NFKServedMCP.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

/*! The MCP protocol versions the server speaks, newest first. */
FOUNDATION_EXPORT NSArray<NSString *> *NFKMCPProtocolVersions(void);

/*! A tool's result as MCP content: its text, each image it shows the model, the audio file it
	generated, and its video as an embedded resource, with isError and any structured output. */
FOUNDATION_EXPORT NSDictionary<NSString *, id> *NFKMCPCallResult(NFKToolResult *result);

/*!
	@class      NFKServedMCP
	@abstract   The Model Context Protocol endpoint, answered from the server's side.
	@discussion Streamable HTTP, stateless: a POST carries one JSON-RPC message or a batch of them and
				is answered with JSON; a message without an id (a notification, or a response to the
				server) is acknowledged with 202. initialize, ping, tools/list, and tools/call are
				answered; any other method is not found. The server issues no session and opens no
				stream of its own, so GET and DELETE are 405. An Origin header naming another host is
				refused with 403, which keeps a web page from reaching a local server through DNS
				rebinding. A tools/call runs off the connection's queue and answers when the tool
				finishes.
*/
@interface NFKServedMCP : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (void)serveRequest:(NFKHTTPRequest *)request
			response:(NFKHTTPResponse *)response
			   tools:(NSArray<NFKInferenceTool *> *)tools
		instructions:(nullable NSString *)instructions;

@end

NS_ASSUME_NONNULL_END
