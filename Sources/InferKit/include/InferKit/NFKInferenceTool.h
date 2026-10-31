//
//  NFKInferenceTool.h
//  InferKit
//

#ifndef NFKInferenceTool_h
#define NFKInferenceTool_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKToolResult
	@abstract   What a tool returns: the text the model reads, the images it is shown, and the outputs
				the application keeps.
	@discussion A model reads only text and images. text is what it learns from the call, so it states
				what happened and what came back. images are PNG files the model is shown beside the
				text where its provider takes images in a tool result. outputs are what the tool
				produced for the application, keyed by NFKOutput* (the NFKAudioAsset a speech tool
				generated under NFKOutputAudio), which reach the caller through the tool loop's
				NFKOutputToolInvocations and never reach the model.

				An error result (isError YES) tells the model the call failed and why, so the model can
				correct its arguments or answer without the tool. Introduced in InferKit 0.4.0.
*/
@interface NFKToolResult : NSObject

/*! What the model reads. */
@property (nonatomic, readonly, copy) NSString *text;

/*! PNG images the model is shown with the text. */
@property (nonatomic, readonly, copy) NSArray<NSData *> *imagePNGs;

/*! What the tool produced for the application, keyed by NFKOutput*. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *outputs;

/*! YES when the call failed and text says why. */
@property (nonatomic, readonly) BOOL isError;

+ (instancetype)resultWithText:(NSString *)text;

+ (instancetype)resultWithText:(NSString *)text
					   outputs:(nullable NSDictionary<NSString *, id> *)outputs
					 imagePNGs:(nullable NSArray<NSData *> *)imagePNGs;

/*! A failed call, with text telling the model what went wrong. */
+ (instancetype)errorResultWithText:(NSString *)text;

- (instancetype)initWithText:(NSString *)text
					 outputs:(nullable NSDictionary<NSString *, id> *)outputs
				   imagePNGs:(nullable NSArray<NSData *> *)imagePNGs
					 isError:(BOOL)isError NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKToolCaller
	@abstract   The MCP client whose tools/call reached a tool through an NFKInferenceServer's /mcp.
	@discussion The server builds one for every tools/call it serves and passes it to the tool's caller
				handler, start handler, or resultForArguments:caller:error:. A tool invoked anywhere
				else (a tool loop, invokeWithArguments: directly) receives nil.

				sessionIdentifier names the MCP session the request carried in Mcp-Session-Id, and
				clientName and clientVersion come from that session's initialize. A request without a
				session carries nil in all three. keyName is the name of the server key the request
				presented, which a client cannot choose. headers holds the request's HTTP headers by
				lowercase name, without the ones that carry a credential (authorization,
				proxy-authorization, cookie, x-api-key, x-goog-api-key). Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKToolCaller : NSObject

/*! The Mcp-Session-Id of the active session the request carried, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *sessionIdentifier;

/*! clientInfo.name from the session's initialize, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *clientName;

/*! clientInfo.version from the session's initialize, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *clientVersion;

/*! The name of the server key the request presented, or nil when the server takes none. */
@property (nonatomic, readonly, copy, nullable) NSString *keyName;

/*! The IP address the request came from. */
@property (nonatomic, readonly, copy) NSString *clientAddress;

/*! The request's HTTP headers by lowercase name, without the headers that carry a credential. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *headers;

- (instancetype)initWithSessionIdentifier:(nullable NSString *)sessionIdentifier
							   clientName:(nullable NSString *)clientName
							clientVersion:(nullable NSString *)clientVersion
								  keyName:(nullable NSString *)keyName
							clientAddress:(NSString *)clientAddress
								  headers:(nullable NSDictionary<NSString *, NSString *> *)headers NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The value of the header named name, matched without regard to case, or nil. */
- (nullable NSString *)valueForHeader:(NSString *)name;

@end

/*!
	@typedef    NFKToolHandler
	@abstract   Runs a tool on the model's arguments and returns its result, or nil with an error.
	@discussion The arguments are the parsed JSON object the model wrote. A returned error reaches the
				model as an error result carrying its localized description. It runs on the thread that
				invokes the tool: a tool loop's thread, or a server's background queue for an MCP call.
*/
typedef NFKToolResult * _Nullable (^ NS_SWIFT_SENDABLE NFKToolHandler)(NSDictionary<NSString *, id> *arguments,
													NSError * _Nullable * _Nullable error);

/*!
	@typedef    NFKToolCallerHandler
	@abstract   Runs a tool as NFKToolHandler does, with the MCP client that called it, or nil.
	@discussion Introduced in InferKit 0.4.0.
*/
typedef NFKToolResult * _Nullable (^ NS_SWIFT_SENDABLE NFKToolCallerHandler)(NSDictionary<NSString *, id> *arguments,
														  NFKToolCaller * _Nullable caller,
														  NSError * _Nullable * _Nullable error);

/*!
	@typedef    NFKToolStartHandler
	@abstract   Starts a tool without holding the calling thread and returns the block that cancels it.
	@discussion The arguments passed the schema check. The handler returns at once and calls
				completion exactly once, from any thread, with the tool's result; an error result
				reports a failure. The returned block, when not nil, cancels the call: the server runs
				it when the client leaves before the result, and the handler still calls completion.
				Introduced in InferKit 0.4.0.
*/
typedef dispatch_block_t _Nullable (^ NS_SWIFT_SENDABLE NFKToolStartHandler)(NSDictionary<NSString *, id> *arguments,
															  NFKToolCaller * _Nullable caller,
															  void (^ NS_SWIFT_SENDABLE completion)(NFKToolResult *result));

/*!
	@class      NFKInferenceTool
	@abstract   A tool a language model can call: a name, a description, a JSON Schema for its
				arguments, and the code that runs it.
	@discussion A tool is offered to a model through its declaration, the NFKParameterTools entry
				{name, description, parameters}. The model reads the description to decide whether and
				when to call the tool, so the description states what the tool does, when to use it,
				what each argument means, and what comes back. The schema's property descriptions carry
				the per-argument detail.

				NFKToolLoop offers a set of tools to any backend that calls tools, runs each call, and
				returns the result to the model. NFKInferenceServer lists the tools registered on it
				to MCP clients. A tool is built from a handler block, or by subclassing and overriding
				resultForArguments:error:; NFKBackendTool makes any inference backend a tool.

				A tool called through a server's /mcp learns its caller, an NFKToolCaller: a tool built
				with a caller handler or a start handler receives it, and so does a subclass that
				overrides resultForArguments:caller:error: or startWithArguments:caller:completion:.
				Everywhere else the caller is nil. A tool built with a start handler holds no thread
				while it waits, so a call that waits minutes, such as one that waits for a message,
				starts that way.

				invokeWithArguments: is the entry point a caller uses. It checks the arguments against
				the schema's required properties and top-level types, runs the tool, and turns a
				failure into an error result, so it never fails. A tool runs synchronously on the
				caller's thread, and one tool may be invoked from several threads at once by a server.
				Introduced in InferKit 0.4.0.
*/
@interface NFKInferenceTool : NSObject

/*! The name the model calls the tool by: letters, digits, underscores, and hyphens, at most 64
	characters, which every provider accepts. */
@property (nonatomic, readonly, copy) NSString *name;

/*! What the tool does, when to use it, and what it returns, written for the model. */
@property (nonatomic, readonly, copy) NSString *toolDescription;

/*! The JSON Schema object for the arguments: {"type": "object", "properties": {…}, "required": […]}. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *parameters;

/*! The tool as an NFKParameterTools entry: {name, description, parameters}. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *declaration;

/*! A tool that runs handler. */
+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
					 handler:(NFKToolHandler)handler;

/*! A tool that runs handler with its caller. Introduced in InferKit 0.4.0. */
+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
			   callerHandler:(NFKToolCallerHandler)handler;

/*! A tool that starts with handler and holds no thread while it runs. invokeWithArguments: and
	resultForArguments:caller:error: wait for its completion on the calling thread. Introduced in
	InferKit 0.4.0. */
+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
				startHandler:(NFKToolStartHandler)handler;

/*! The initializer a subclass calls; the subclass overrides resultForArguments:error:. */
- (instancetype)initWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*!
	@method     resultForArguments:error:
	@abstract   Runs the tool. A subclass overrides it; the base class runs the handler.
	@discussion Called with arguments that passed the schema check. Returns nil with an error when the
				call fails; invokeWithArguments: reports that error to the model.
*/
- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments
										 error:(NSError * _Nullable * _Nullable)error;

/*!
	@method     resultForArguments:caller:error:
	@abstract   Runs the tool for a caller. A subclass that reads its caller overrides it.
	@discussion The base class runs the caller handler, or the start handler and waits for its
				completion, or else calls resultForArguments:error:, so a subclass that overrides only
				that method runs unchanged. Introduced in InferKit 0.4.0.
*/
- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments
										caller:(nullable NFKToolCaller *)caller
										 error:(NSError * _Nullable * _Nullable)error;

/*!
	@method     invokeWithArguments:
	@abstract   Checks the arguments, runs the tool, and returns its result or an error result.
	@discussion A missing required property, or a top-level property whose JSON type differs from its
				schema's, returns an error result naming each problem without running the tool. The tool
				runs with no caller.
*/
- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments;

/*!
	@method     invokeWithArguments:caller:
	@abstract   Checks the arguments and runs the tool for caller, as invokeWithArguments: does.
	@discussion Introduced in InferKit 0.4.0.
*/
- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments caller:(nullable NFKToolCaller *)caller;

/*!
	@method     startWithArguments:caller:completion:
	@abstract   Checks the arguments, starts the tool, and calls completion once with its result.
	@discussion A server's /mcp starts every call this way. A tool built with a start handler, or a
				subclass that overrides this method, returns at once, calls completion later from any
				thread, and returns the block that cancels the call, or nil. Every other tool runs
				invokeWithArguments:caller: on the calling thread, calls completion before returning,
				and returns nil. Arguments that fail the schema check complete with the error result
				invokeWithArguments: gives. Introduced in InferKit 0.4.0.
*/
- (nullable dispatch_block_t)startWithArguments:(NSDictionary<NSString *, id> *)arguments
										 caller:(nullable NFKToolCaller *)caller
									 completion:(void (^ NS_SWIFT_SENDABLE)(NFKToolResult *result))completion;

/*! The problems with arguments under the schema, or an empty array when there are none. */
- (NSArray<NSString *> *)problemsWithArguments:(NSDictionary<NSString *, id> *)arguments;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceTool_h */
