//
//  NFKMCPClient.h
//  InferKit
//

#ifndef NFKMCPClient_h
#define NFKMCPClient_h

#import <Foundation/Foundation.h>
#import "NFKInferenceTool.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKMCPClient
	@abstract   Reaches the tools a Model Context Protocol server serves, as NFKInferenceTool.
	@discussion The client speaks MCP's Streamable HTTP transport to one endpoint: it initializes,
				lists the server's tools, and calls them, reading a reply sent as JSON or as an
				event stream. toolsWithError: returns each tool the server lists as an
				NFKInferenceTool, so NFKToolLoop offers a remote server's tools to any model, local
				or remote, beside its own.

				A call's MCP content becomes an NFKToolResult:
				- text content joins into the text the model reads.
				- image content becomes the images the model is shown, and the first image also rides
				  under NFKOutputImage.
				- audio content is written to a file and rides as an NFKAudioAsset under NFKOutputAudio.
				- a video resource rides as an NFKVideoAsset under NFKOutputVideo.
				- structured content rides under NFKOutputStructured.
				A result the server marks isError is an error result. A protocol error (an unknown
				tool, a malformed call) returns nil with an error whose description is the server's
				message.

				apiKey goes out as a Bearer token, which NFKInferenceServer and most hosted MCP
				servers read. The client keeps the session identifier a server assigns and sends it
				with each later request. Calls block and are serialized, so run them off the main
				thread. Introduced in InferKit 0.4.0.
*/
@interface NFKMCPClient : NSObject

/*! The server's MCP endpoint, such as http://localhost:8765/mcp. */
@property (nonatomic, readonly, copy) NSURL *endpointURL;

/*! The key sent as a Bearer token, or nil for none. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! Header fields sent with every request, for a server that reads its credentials elsewhere. */
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *additionalHeaders;

/*! How long one request waits for its reply, in seconds. Defaults to 600, since a tool that
	generates media takes minutes. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The server's name and version from its initialize reply, or nil before connecting. */
@property (nonatomic, readonly, copy, nullable) NSDictionary<NSString *, id> *serverInfo;

/*! What the server tells a client's model about itself, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *instructions;

/*! The protocol version the server agreed to, or nil before connecting. */
@property (nonatomic, readonly, copy, nullable) NSString *protocolVersion;

- (instancetype)initWithEndpointURL:(NSURL *)endpointURL apiKey:(nullable NSString *)apiKey NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! Initializes the session. The other calls connect first when they need to. */
- (BOOL)connectWithError:(NSError * _Nullable *)error;

/*! Every tool the server lists, across every page, as NFKInferenceTool. */
- (nullable NSArray<NFKInferenceTool *> *)toolsWithError:(NSError * _Nullable *)error;

/*! Calls the tool named name with arguments and returns its result. */
- (nullable NFKToolResult *)resultOfToolNamed:(NSString *)name
									arguments:(NSDictionary<NSString *, id> *)arguments
										error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(resultOfTool(named:arguments:));

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMCPClient_h */
