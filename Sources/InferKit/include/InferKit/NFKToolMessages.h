//
//  NFKToolMessages.h
//  InferKit
//

#ifndef NFKToolMessages_h
#define NFKToolMessages_h

#import <Foundation/Foundation.h>

@class NFKInferenceResult;
@class NFKToolResult;

NS_ASSUME_NONNULL_BEGIN

/*! The message key under which InferKit keeps what a conversation needs beside the OpenAI chat
	fields (NSDictionary): a backend's own record of an assistant turn, and a tool result's name,
	error flag, and images. Every backend removes it before a message goes on the wire.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKMessageMetadataKey;

/*!
	@class      NFKToolMessages
	@abstract   Builds the conversation turns of a tool call: the assistant turn that made it and the
				tool turn that answers it.
	@discussion A conversation under NFKInputMessages keeps the OpenAI chat shape for tool use, which
				every chat backend reads:

				- the assistant turn: {role: "assistant", content, tool_calls: [{id, type: "function",
				  function: {name, arguments}}]}, arguments a JSON string.
				- each answer: {role: "tool", tool_call_id, content}, one per call, after the
				  assistant turn.

				Each backend converts the turns to its provider's form (Anthropic tool_use and
				tool_result blocks, Responses function_call items, Gemini function_call and
				function_result steps, a local model's chat template). What the chat shape cannot carry
				rides under NFKMessageMetadataKey: the provider's own record of the assistant turn, which
				the backend that wrote it replays exactly, and a result's tool name, error flag, and
				images, which a provider that takes them receives. NFKToolLoop builds every turn through
				this class; a caller that runs its own loop uses it the same way.
				Introduced in InferKit 0.4.0.
*/
@interface NFKToolMessages : NSObject

/*! The reply as the assistant turn to append: the backend's NFKOutputAssistantMessage when it wrote
	one, otherwise a turn built from the text and NFKOutputToolCalls. A call without an identifier is
	given one, so its answer can name it. */
+ (NSDictionary<NSString *, id> *)assistantMessageForResult:(NFKInferenceResult *)result;

/*! The calls an assistant turn makes, in the NFKOutputToolCalls shape: {id, name, arguments (the
	parsed object), argumentsJSON}. A call whose arguments are not a JSON object carries an empty
	object under arguments and the model's text under argumentsJSON. */
+ (NSArray<NSDictionary<NSString *, id> *> *)toolCallsOfAssistantMessage:(NSDictionary<NSString *, id> *)message;

/*! The tool turn answering the call with identifier callIdentifier to the tool named toolName. */
+ (NSDictionary<NSString *, id> *)messageForToolCallIdentifier:(NSString *)callIdentifier
														 name:(NSString *)toolName
													   result:(NFKToolResult *)result;

/*!
	@method     assistantMessageWithText:toolCalls:record:format:source:
	@abstract   The assistant turn a backend returns under NFKOutputAssistantMessage.
	@discussion The chat fields come from text and toolCalls (the NFKOutputToolCalls shape). record is
				the backend's own form of the turn, kept under NFKMessageMetadataKey with format and
				source, and recordOfMessage:format:source: returns it to the same backend, which replays
				it; nil keeps no record. A backend outside the core uses this so its turns read the same
				way as the core's.
*/
+ (NSDictionary<NSString *, id> *)assistantMessageWithText:(nullable NSString *)text
												 toolCalls:(nullable NSArray<NSDictionary<NSString *, id> *> *)toolCalls
													record:(nullable id)record
													format:(nullable NSString *)format
													source:(nullable NSString *)source;

/*! A backend's record of an assistant turn, when the turn carries one written in format by source
	(any source when source is nil), otherwise nil. */
+ (nullable id)recordOfMessage:(NSDictionary<NSString *, id> *)message
						format:(NSString *)format
						source:(nullable NSString *)source;

/*! The name of the tool a tool turn answers: its own, or the name of the call it answers in
	conversation. Nil when neither says. */
+ (nullable NSString *)toolNameOfMessage:(NSDictionary<NSString *, id> *)message
						  inConversation:(NSArray<NSDictionary<NSString *, id> *> *)conversation;

/*! YES when a tool turn reports a failed call. */
+ (BOOL)isErrorToolMessage:(NSDictionary<NSString *, id> *)message;

/*! A tool turn's text: its content string, or its text parts joined. */
+ (NSString *)textOfToolMessage:(NSDictionary<NSString *, id> *)message;

/*! The PNG images a tool turn shows the model. */
+ (NSArray<NSData *> *)imagePNGsOfToolMessage:(NSDictionary<NSString *, id> *)message;

/*! The messages with NFKMessageMetadataKey removed, for a caller that sends them through its own
	client. */
+ (NSArray<NSDictionary<NSString *, id> *> *)messagesWithoutMetadata:(NSArray<NSDictionary<NSString *, id> *> *)messages;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKToolMessages_h */
