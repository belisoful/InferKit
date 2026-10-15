//
//  NFKServedAnthropic.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceServer;

/*!
	@class      NFKServedAnthropic
	@abstract   Anthropic's Messages API, read and written from the server's side.
	@discussion The request method inverts what NFKAnthropicBackend sends, so a conversation that
				crosses from it, or from Claude Code, arrives at the hosted backend in the contract's
				OpenAI chat shape under NFKInputMessages: the system field becomes a system turn,
				tool_use blocks an assistant turn's tool_calls, and tool_result blocks tool turns
				whose name, error flag, and images ride under NFKMessageMetadataKey. Images arrive
				under NFKInputImage and NFKInputImages, documents under NFKInputDocument and
				NFKInputDocuments, and a lone user turn also as NFKInputPrompt. Media must be inline
				base64: the server fetches nothing for a client and keeps no files.

				A tool_choice that forces one tool is structured output: the tool's input schema
				becomes NFKParameterJSONSchema, and the reply is a tool_use block of that tool whose
				input is the structured result.

				The reply methods write the Message object and its event stream. The stop reason is
				tool_use when the reply calls a tool, max_tokens when the output tokens the backend
				reports reach max_tokens, and end_turn otherwise. input_tokens excludes the tokens a
				cache served, which cache_read_input_tokens carries, as the API counts them.
*/
@interface NFKServedAnthropic : NSObject

- (instancetype)init NS_UNAVAILABLE;

#pragma mark Routes

/*! Serves POST /v1/messages on server's models: the Message, or its event stream when the body sets
	stream. */
+ (void)serveMessages:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server;

/*! Serves POST /v1/messages/count_tokens: {input_tokens} from the hosted backend's
	tokenCountForRequest:error:. */
+ (void)serveTokenCount:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server;

/*! The model list in Anthropic's shape: {data: [{type: "model", id, display_name, created_at}],
	has_more, first_id, last_id}, created_at the server's start as an RFC 3339 date. */
+ (NSDictionary<NSString *, id> *)modelListForNames:(NSArray<NSString *> *)names createdAt:(NSInteger)createdAt;

/*! One model in Anthropic's shape. */
+ (NSDictionary<NSString *, id> *)modelEntryForName:(NSString *)name createdAt:(NSInteger)createdAt;

#pragma mark Wire

/*! The request a Messages body asks for, and the name of the tool its tool_choice forces. */
+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								   forcedToolName:(NSString * _Nullable * _Nullable)outForcedToolName
											error:(NSError * _Nullable *)outError;

/*! The Message object answering request with result. */
+ (NSDictionary<NSString *, id> *)messageForResult:(NFKInferenceResult *)result
										   request:(NFKInferenceRequest *)request
										identifier:(NSString *)identifier
											 model:(NSString *)model
									forcedToolName:(nullable NSString *)forcedToolName;

/*! The error body the API returns for status: {type: "error", error: {type, message}}, with the
	InferKit domain and code beside the message so an InferKit client reads the exact code. */
+ (NSDictionary<NSString *, id> *)errorObjectForError:(NSError *)error status:(NSInteger)status;

/*! A new message identifier, msg_ and 32 hexadecimal digits. */
+ (NSString *)newMessageIdentifier;

@end

/*!
	@class      NFKServedAnthropicStream
	@abstract   Writes one reply as the Messages API's server-sent events.
	@discussion The first event is message_start; each content block then opens with
				content_block_start, grows by content_block_delta (thinking_delta, text_delta, or
				input_json_delta), and closes with content_block_stop; message_delta carries the stop
				reason and the usage, and message_stop ends the reply. A partial result's text and
				reasoning are cumulative, and only what extends the text already sent goes out. Not
				thread-safe: the caller serializes the calls.
*/
@interface NFKServedAnthropicStream : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithRequest:(NFKInferenceRequest *)request
					 identifier:(NSString *)identifier
						  model:(NSString *)model
				 forcedToolName:(nullable NSString *)forcedToolName
				   eventHandler:(void (^)(NSString *name, NSDictionary<NSString *, id> *data))eventHandler NS_DESIGNATED_INITIALIZER;

/*! Whether message_start has gone out, after which a failure is an error event. */
@property (nonatomic, readonly) BOOL started;

/*! Sends what the partial adds to the reasoning and the text sent so far. */
- (void)appendPartial:(NFKInferenceResult *)partial;

/*! Sends the rest of the reply, its tool uses, message_delta, and message_stop. */
- (void)finishWithResult:(NFKInferenceResult *)result;

/*! Sends an error event; the caller then ends the stream. */
- (void)failWithError:(NSError *)error status:(NSInteger)status;

@end

NS_ASSUME_NONNULL_END
