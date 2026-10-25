//
//  NFKMistralConversationsBackend.h
//  InferKit
//

#ifndef NFKMistralConversationsBackend_h
#define NFKMistralConversationsBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKMistralConversationsBackend
	@abstract   An inference backend that calls Mistral's Conversations API (POST /v1/conversations).
	@discussion The Conversations API runs a model, or a saved agent named by agentIdentifier, with
				Mistral's built-in tools, and keeps the conversation on the service. Its built-in tools
				reach nothing on chat completions: web search (web_search, web_search_premium), the
				code interpreter, image generation, search over document libraries, and custom
				connectors.

				The request reads NFKInputPrompt or NFKInputMessages (a leading system turn becomes
				instructions; a tool conversation's calls and answers become function.call and
				function.result entries), with NFKInputImage and NFKInputImages as image_url chunks on
				the last user turn. NFKParameterTools become function tools; an entry with a type passes
				through, which is how a caller asks for {type: web_search}, {type: code_interpreter},
				{type: image_generation}, or {type: document_library, library_ids}. The sampling keys,
				NFKParameterMaxTokens, NFKParameterStopSequences, NFKParameterSeed,
				NFKParameterReasoningEffort, and NFKParameterJSONSchema go under completion_args. Every
				other parameter goes out under its own name (store, handoff_execution, guardrails,
				metadata).

				NFKParameterPreviousResponseIdentifier names a conversation to continue (POST
				/v1/conversations/{id}), whose model, instructions, and tools stay as they began, so the
				request carries only the new turns; with a from_entry_id parameter the conversation
				restarts from that entry (POST /v1/conversations/{id}/restart).

				The reply's text comes back under NFKOutputText, its thinking under NFKOutputReasoning,
				function calls under NFKOutputToolCalls, tool_reference chunks under
				NFKOutputCitations, each tool.execution and agent.handoff entry and each tool_file
				chunk under NFKOutputServerToolResults, an image the image_generation tool wrote under
				NFKOutputImage (and NFKOutputImages) when downloadsImages is on, a parsed schema reply
				under NFKOutputStructured, the token counts under NFKOutputUsage, and the conversation's
				id under NFKOutputResponseIdentifier. submitInferenceJobForRequest: streams the text and
				the thinking into the job's partialResult. runInferenceForRequest: blocks; run it off the
				render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKMistralConversationsBackend : NSObject <NFKInferenceBackend>

/*! The conversations endpoint, for example https://api.mistral.ai/v1/conversations. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model a new conversation runs. A conversation needs it or agentIdentifier. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! A saved agent a new conversation runs instead of modelName, sent as agent_id. */
@property (nonatomic, copy, nullable) NSString *agentIdentifier;

/*! Whether a tool_file chunk naming an image is downloaded from the Files API and returned under
	NFKOutputImage. Defaults to YES. */
@property (nonatomic, assign) BOOL downloadsImages;

/*! The request timeout in seconds. Defaults to 600; a tool run can take minutes. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A backend pointed at the provider's conversations endpoint: mistral serves one, and every other
	preset returns nil. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Streams the reply into the job's partialResult; cancelling the job closes the connection. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! Every entry of a conversation the service keeps, in order (GET /v1/conversations/{id}/history),
	or nil with an error. Blocks. */
- (nullable NSArray<NSDictionary<NSString *, id> *> *)historyOfConversationWithIdentifier:(NSString *)identifier
																				   error:(NSError * _Nullable *)outError;

/*! Deletes a conversation the service keeps. Blocks. */
- (BOOL)deleteConversationWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! The bytes of a file a tool wrote (GET /v1/files/{id}/content beside the endpoint), or nil with an
	error. Blocks. */
- (nullable NSData *)contentsOfFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and returns
	the block that cancels the request. */
- (void (^ NS_SWIFT_SENDABLE)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^ NS_SWIFT_SENDABLE)(NSString *line))lineHandler
			  completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMistralConversationsBackend_h */
