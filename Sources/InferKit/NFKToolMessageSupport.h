//
//  NFKToolMessageSupport.h
//  InferKit
//
//  The parts of a tool conversation the chat backends read while converting it to their provider's
//  form. Private to the core.
//

#ifndef NFKToolMessageSupport_h
#define NFKToolMessageSupport_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The provider formats an assistant turn's record is kept in. */
extern NSString * const NFKProviderTurnFormatOpenAIChat;
extern NSString * const NFKProviderTurnFormatAnthropicMessages;
extern NSString * const NFKProviderTurnFormatOpenAIResponses;
extern NSString * const NFKProviderTurnFormatGeminiInteractions;
extern NSString * const NFKProviderTurnFormatGeminiGenerateContent;
extern NSString * const NFKProviderTurnFormatMistralConversations;

/*! The assistant turn: the chat fields from text and calls (NFKOutputToolCalls shape), and the
	provider's record under NFKMessageMetadataKey when turn is non-nil. source names the service the
	record came from, so only that service replays it. */
NSDictionary<NSString *, id> *NFKAssistantMessage(NSString * _Nullable text,
												  NSArray<NSDictionary *> * _Nullable toolCalls,
												  NSString * _Nullable format,
												  NSString * _Nullable source,
												  id _Nullable turn);

/*! The provider's record of an assistant turn when it was written in format by source, otherwise nil. */
id _Nullable NFKMessageProviderTurn(NSDictionary *message, NSString *format, NSString * _Nullable source);

/*! The message without NFKMessageMetadataKey. */
NSDictionary<NSString *, id> *NFKMessageWithoutMetadata(NSDictionary *message);

/*! YES for a {role: "tool"} answer. */
BOOL NFKMessageIsToolResult(NSDictionary *message);

/*! A tool answer's parts: its call identifier, tool name (nil when unknown), text, error flag, and
	the PNG images the model is shown. */
NSString *NFKToolMessageCallIdentifier(NSDictionary *message);
NSString * _Nullable NFKToolMessageName(NSDictionary *message);
NSString *NFKToolMessageText(NSDictionary *message);
BOOL NFKToolMessageIsError(NSDictionary *message);
NSArray<NSData *> *NFKToolMessageImagePNGs(NSDictionary *message);

/*! A tool answer's name, found from the call it answers when the answer does not carry one. */
NSString * _Nullable NFKToolMessageNameIn(NSDictionary *message, NSArray *conversation);

NS_ASSUME_NONNULL_END

#endif /* NFKToolMessageSupport_h */
