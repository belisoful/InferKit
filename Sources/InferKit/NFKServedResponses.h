//
//  NFKServedResponses.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceServer;

/*!
	@class      NFKServedResponses
	@abstract   OpenAI's Responses API, read and written from the server's side.
	@discussion The request method inverts what NFKRemoteResponsesBackend sends, so a conversation that
				crosses from it, or from Codex, arrives at the hosted backend in the contract's OpenAI
				chat shape under NFKInputMessages: instructions and the system and developer turns
				before the first other turn become one system turn, function_call items an assistant
				turn's tool_calls, and function_call_output items tool turns. A custom tool (free-form
				input) reaches the backend as a function taking one string argument, input, and its
				call comes back as a custom_tool_call. Built-in tools (web search, file search, a
				local shell) are not offered, since the server runs none. Reasoning items are not
				replayed. Images arrive as pixel buffers and files as their bytes; a file_id or a
				remote URL is refused.

				The server keeps no responses: store is false in every reply, and previous_response_id
				or a conversation is refused, so each request carries its whole conversation, as
				Codex sends it.

				A reply is completed, or incomplete with reason max_output_tokens when the output tokens
				the backend reports reach max_output_tokens.
*/
@interface NFKServedResponses : NSObject

- (instancetype)init NS_UNAVAILABLE;

#pragma mark Routes

/*! Serves POST /v1/responses on server's models: the response object, or its event stream when the
	body sets stream. */
+ (void)serveResponses:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server;

/*! Serves POST /v1/responses/input_tokens: {object: "response.input_tokens", input_tokens} from the
	hosted backend's tokenCountForRequest:error:. */
+ (void)serveInputTokens:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server;

#pragma mark Wire

/*! The request a Responses body asks for, and the names of its custom tools. */
+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								  customToolNames:(NSSet<NSString *> * _Nullable * _Nullable)outCustomToolNames
											error:(NSError * _Nullable *)outError;

/*! The response object answering body with result. */
+ (NSDictionary<NSString *, id> *)responseForResult:(NFKInferenceResult *)result
											   body:(NSDictionary *)body
										 identifier:(NSString *)identifier
										  createdAt:(NSInteger)createdAt
											  model:(NSString *)model
									customToolNames:(NSSet<NSString *> *)customToolNames;

/*! A new response identifier, resp_ and 32 hexadecimal digits. */
+ (NSString *)newResponseIdentifier;

@end

/*!
	@class      NFKServedResponsesStream
	@abstract   Writes one reply as the Responses API's server-sent events.
	@discussion The stream opens with response.created and response.in_progress. Each output item
				opens with response.output_item.added and closes with response.output_item.done: a
				reasoning item grows by response.reasoning_summary_text.delta, a message by
				response.output_text.delta, a function call by response.function_call_arguments.delta,
				and a custom tool call by response.custom_tool_call_input.delta. response.completed (or
				response.incomplete) carries the whole response, and response.failed a failure. Every
				event carries a sequence_number. A partial result's text and reasoning are cumulative,
				and only what extends the text already sent goes out. Not thread-safe: the caller
				serializes the calls.
*/
@interface NFKServedResponsesStream : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithBody:(NSDictionary *)body
				  identifier:(NSString *)identifier
				   createdAt:(NSInteger)createdAt
					   model:(NSString *)model
			 customToolNames:(NSSet<NSString *> *)customToolNames
				eventHandler:(void (^)(NSString *name, NSDictionary<NSString *, id> *data))eventHandler NS_DESIGNATED_INITIALIZER;

/*! Whether response.created has gone out, after which a failure is a response.failed event. */
@property (nonatomic, readonly) BOOL started;

/*! Sends what the partial adds to the reasoning and the text sent so far. */
- (void)appendPartial:(NFKInferenceResult *)partial;

/*! Sends the rest of the reply, its calls, and response.completed or response.incomplete. */
- (void)finishWithResult:(NFKInferenceResult *)result;

/*! Sends response.failed; the caller then ends the stream. */
- (void)failWithError:(NSError *)error;

@end

NS_ASSUME_NONNULL_END
