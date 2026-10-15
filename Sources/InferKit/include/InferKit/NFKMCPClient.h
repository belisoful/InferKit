//
//  NFKMCPClient.h
//  InferKit
//

#ifndef NFKMCPClient_h
#define NFKMCPClient_h

#import <Foundation/Foundation.h>
#import "NFKInferenceTool.h"
#import "NFKSubprocess.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKMCPTransport
	@abstract   How an NFKMCPClient reaches its server.
	@constant   NFKMCPTransportStreamableHTTP JSON-RPC posted to one HTTP endpoint, answered as JSON
				or as an event stream.
	@constant   NFKMCPTransportStdio JSON-RPC exchanged one message per line with a server the client
				launches as a subprocess. macOS only.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKMCPTransport) {
	NFKMCPTransportStreamableHTTP = 0,
	NFKMCPTransportStdio = 1,
};

/*!
	@class      NFKMCPClient
	@abstract   Reaches the tools a Model Context Protocol server serves, as NFKInferenceTool.
	@discussion The client speaks MCP over either of the protocol's transports, with one API:
				- Streamable HTTP: initWithEndpointURL:apiKey: names the server's endpoint.
				- stdio: initWithCommand:arguments:environment:workingDirectoryURL: names a program the
				  client launches, the way `claude mcp add --transport stdio` registers one. macOS only.
				The client initializes, lists the server's tools, and calls them. toolsWithError:
				returns each tool the server lists as an NFKInferenceTool, so NFKToolLoop offers a
				server's tools to any model, local or remote, beside its own.

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

				Over HTTP, apiKey goes out as a Bearer token, which NFKInferenceServer and most hosted
				MCP servers read, and the client keeps the session identifier a server assigns and
				sends it with each later request.

				Over stdio, the first call launches the server through serverProcess, an NFKSubprocess
				an application configures before that call: its restart policy, its stop timeouts,
				and its standard error, which is the server's log. The client answers the server's
				ping and refuses the server's other requests with JSON-RPC error -32601. apiKey and
				additionalHeaders do not apply.

				A stdio server that exits stays down:
				- every pending call returns kNFKError_RemoteUnreachable naming the exit status or
				  signal, and so does every later call.
				- connectWithError: launches the server again.
				- with NFKSubprocessRestartPolicyOnCrash, serverProcess relaunches a crashed server
				  itself, and the next call initializes a new session with it.
				- after disconnect, the next call launches the server again.
				serverProcess's stateHandler reports each exit.

				A stdio server inherits only HOME, LOGNAME, PATH, SHELL, TERM, USER, TMPDIR, LANG, and
				LC_ALL from this process, then the environment given at initialization, as the MCP
				reference SDKs pass it. An MCP server is often a third party's program, and an
				application's environment can hold its own credentials. inheritsFullEnvironment passes
				the whole environment instead. NFKSubprocess on its own inherits the whole
				environment, for a child the application itself ships.

				Calls block and are serialized, so run them off the main thread. Every request waits
				for its reply up to a deadline: timeout for a tool call, protocolTimeout for the rest.
				A request past its deadline returns kNFKError_RemoteUnreachable, and over stdio the
				client also sends notifications/cancelled. Introduced in InferKit 0.4.0.
*/
@interface NFKMCPClient : NSObject

/*! How the client reaches its server. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly) NFKMCPTransport transport;

/*! The server's MCP endpoint, such as http://localhost:8765/mcp, or nil over stdio. */
@property (nonatomic, readonly, copy, nullable) NSURL *endpointURL;

/*! The stdio server's process, or nil over HTTP. Configure it before the first call. The client
	reads its standard output and state through hooks of its own, so the process's lineHandler and
	stateHandler stay free for an application. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, strong, nullable) NFKSubprocess *serverProcess API_UNAVAILABLE(ios, tvos, watchos);

/*! The variables a stdio server inherits from this process by default: HOME, LOGNAME, PATH, SHELL,
	TERM, USER, TMPDIR, LANG, and LC_ALL. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *defaultInheritedEnvironmentVariables;

/*! Whether a stdio server inherits this process's whole environment. NO passes only
	defaultInheritedEnvironmentVariables, through serverProcess's inheritedEnvironmentVariables.
	Defaults to NO. Set it before the first call. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) BOOL inheritsFullEnvironment API_UNAVAILABLE(ios, tvos, watchos);

/*! The key sent as a Bearer token over HTTP, or nil for none. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! Header fields sent with every HTTP request, for a server that reads its credentials elsewhere. */
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *additionalHeaders;

/*! How long a tool call waits for its reply, in seconds. Defaults to 600, since a tool that
	generates media takes minutes. A value that is not positive uses 600. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! How long every other request waits for its reply, in seconds: initialize, ping, and tools/list.
	Defaults to 30. A value that is not positive uses 30. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval protocolTimeout;

/*! The server's name and version from its initialize reply, or nil before connecting. */
@property (nonatomic, readonly, copy, nullable) NSDictionary<NSString *, id> *serverInfo;

/*! What the server tells a client's model about itself, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *instructions;

/*! The protocol version the server agreed to, or nil before connecting. */
@property (nonatomic, readonly, copy, nullable) NSString *protocolVersion;

/*! A client of the server at endpointURL over Streamable HTTP. */
- (instancetype)initWithEndpointURL:(NSURL *)endpointURL apiKey:(nullable NSString *)apiKey NS_DESIGNATED_INITIALIZER;

/*!
	@abstract   A client of a server it launches and speaks to over stdio.
	@discussion command is an absolute path, or a name found on the server's PATH or in the
				directories NFKAgentCLI searches, and arguments follow it, with no shell between.
				environment adds to the variables the server inherits and overrides those of the same
				name; set serverProcess's inheritsEnvironment to NO to pass environment alone.
				workingDirectoryURL nil starts the server in this process's working directory.
				Nothing launches until the first call.
				Introduced in InferKit 0.4.0.
*/
- (instancetype)initWithCommand:(NSString *)command
					  arguments:(NSArray<NSString *> *)arguments
					environment:(nullable NSDictionary<NSString *, NSString *> *)environment
			workingDirectoryURL:(nullable NSURL *)workingDirectoryURL NS_DESIGNATED_INITIALIZER API_UNAVAILABLE(ios, tvos, watchos);

- (instancetype)init NS_UNAVAILABLE;

/*! Initializes the session, launching a stdio server that is not running. The other calls connect
	first when they need to, and launch a stdio server only on the first call or after disconnect. */
- (BOOL)connectWithError:(NSError * _Nullable *)error;

/*! Sends ping and returns once the server answers. Introduced in InferKit 0.4.0. */
- (BOOL)pingWithError:(NSError * _Nullable *)error;

/*! Every tool the server lists, across every page, as NFKInferenceTool. */
- (nullable NSArray<NFKInferenceTool *> *)toolsWithError:(NSError * _Nullable *)error;

/*! Calls the tool named name with arguments and returns its result. */
- (nullable NFKToolResult *)resultOfToolNamed:(NSString *)name
									arguments:(NSDictionary<NSString *, id> *)arguments
										error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(resultOfTool(named:arguments:));

/*! Ends the session. Over HTTP, a session the server assigned is deleted. Over stdio, the server is
	stopped and waited for, by closing its input, then SIGTERM, then SIGKILL, per serverProcess's
	timeouts. A later call connects again, launching a stdio server. Introduced in InferKit 0.4.0. */
- (void)disconnect;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMCPClient_h */
