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
	@typedef    NFKToolHandler
	@abstract   Runs a tool on the model's arguments and returns its result, or nil with an error.
	@discussion The arguments are the parsed JSON object the model wrote. A returned error reaches the
				model as an error result carrying its localized description. It runs on the thread that
				invokes the tool: a tool loop's thread, or a server's background queue for an MCP call.
*/
typedef NFKToolResult * _Nullable (^ NS_SWIFT_SENDABLE NFKToolHandler)(NSDictionary<NSString *, id> *arguments,
													NSError * _Nullable * _Nullable error);

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
	@method     invokeWithArguments:
	@abstract   Checks the arguments, runs the tool, and returns its result or an error result.
	@discussion A missing required property, or a top-level property whose JSON type differs from its
				schema's, returns an error result naming each problem without running the tool.
*/
- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments;

/*! The problems with arguments under the schema, or an empty array when there are none. */
- (NSArray<NSString *> *)problemsWithArguments:(NSDictionary<NSString *, id> *)arguments;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceTool_h */
