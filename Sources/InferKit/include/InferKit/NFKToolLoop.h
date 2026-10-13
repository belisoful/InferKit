//
//  NFKToolLoop.h
//  InferKit
//

#ifndef NFKToolLoop_h
#define NFKToolLoop_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"
#import "NFKInferenceTool.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKToolInvocation
	@abstract   One tool call a tool loop ran: the call the model made and the tool's result.
	@discussion The tool loop returns them in order under NFKOutputToolInvocations. result.outputs
				carries what the tool produced for the application, such as the NFKAudioAsset a speech
				tool generated. Introduced in InferKit 0.4.0.
*/
@interface NFKToolInvocation : NSObject

/*! The call's identifier, which its answer in the conversation names. */
@property (nonatomic, readonly, copy) NSString *callIdentifier;

/*! The tool the model called. */
@property (nonatomic, readonly, copy) NSString *toolName;

/*! The arguments the model wrote, parsed. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *arguments;

/*! The tool's result, an error result when the call failed or was declined. */
@property (nonatomic, readonly, strong) NFKToolResult *result;

/*! How long the tool ran, in seconds. */
@property (nonatomic, readonly) NSTimeInterval durationSeconds;

- (instancetype)initWithCallIdentifier:(NSString *)callIdentifier
							  toolName:(NSString *)toolName
							 arguments:(NSDictionary<NSString *, id> *)arguments
								result:(NFKToolResult *)result
					   durationSeconds:(NSTimeInterval)durationSeconds NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

/*! Decides whether a call runs. A declined call reaches the model as an error result. */
typedef BOOL (^ NS_SWIFT_SENDABLE NFKToolCallApproval)(NSString *toolName, NSDictionary<NSString *, id> *arguments);

/*!
	@class      NFKToolLoop
	@abstract   Runs a conversation in which a model calls tools: it offers the tools, runs each call,
				returns the results to the model, and repeats until the model answers.
	@discussion The model is any backend that returns NFKOutputToolCalls: a remote chat model
				(NFKRemoteBackend, NFKAnthropicBackend, NFKRemoteResponsesBackend,
				NFKGeminiInteractionsBackend, NFKGeminiGenerateContentBackend), or an on-device
				language model. The tools are any
				NFKInferenceTool, so one model reaches another through NFKBackendTool:

				- A remote Claude model with a local Qwen model as a tool.
				- A local Qwen model with a speech generator as a tool.

				Each turn sends the conversation and the tools' declarations, appends the model's turn
				(NFKToolMessages assistantMessageForResult:), runs the calls in order, and appends each
				result as a tool turn. The loop ends when a turn makes no call, or makes one to a tool it
				does not hold, which the caller runs (the request's own NFKParameterTools entries are
				offered beside the loop's tools for that purpose). After maximumTurns model turns that
				still call tools, the last turn's calls are returned unrun with
				NFKOutputToolTurnLimitReached.

				The loop is itself a backend. Its result is the final turn's result with three more
				outputs: NFKOutputMessages (the whole conversation, to continue it),
				NFKOutputToolInvocations (each call and its result), and NFKOutputUsage summed over the
				turns. Because it adopts NFKInferenceBackend, a tool loop is served by
				NFKInferenceServer, wrapped as a tool in turn, or balanced like any other backend.

				A failing tool does not end the loop: its error reaches the model as an error result,
				so the model can correct its arguments or answer without it. A failing model turn ends
				the loop with that error. The calls of one turn run one after another on the loop's
				thread. submitInferenceJobForRequest: runs the loop at user-initiated quality of
				service, reports each model turn's streamed partial results, and cancels between steps
				and inside a model turn. Introduced in InferKit 0.4.0.
*/
@interface NFKToolLoop : NSObject <NFKInferenceBackend>

/*! The model the loop converses with. */
@property (nonatomic, readonly, strong) id<NFKInferenceBackend> backend;

/*! The tools the model is offered, by name. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

/*! The most model turns one run takes. Defaults to 8. */
@property (nonatomic) NSUInteger maximumTurns;

/*! Decides whether each call runs, on the loop's thread; nil runs every call. */
@property (nonatomic, copy, nullable) NFKToolCallApproval approvalHandler;

/*! Invoked after each tool runs, on the loop's thread, so an application acts on a result (plays the
	audio a speech tool generated) before the model answers. */
@property (nonatomic, copy, nullable) void (^ NS_SWIFT_SENDABLE invocationHandler)(NFKToolInvocation *invocation);

- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
						  tools:(NSArray<NFKInferenceTool *> *)tools NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The tool named name, or nil. */
- (nullable NFKInferenceTool *)toolNamed:(NSString *)name;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKToolLoop_h */
