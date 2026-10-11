//
//  NFKToolMessages.m
//  InferKit
//

#import "NFKToolMessages.h"
#import "NFKToolMessageSupport.h"
#import "NFKInferenceTool.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceKeys.h"

NSString * const NFKMessageMetadataKey = @"inferkit";

NSString * const NFKProviderTurnFormatOpenAIChat			= @"openai-chat";
NSString * const NFKProviderTurnFormatAnthropicMessages	= @"anthropic-messages";
NSString * const NFKProviderTurnFormatOpenAIResponses		= @"openai-responses";
NSString * const NFKProviderTurnFormatGeminiInteractions	= @"gemini-interactions";

static NSString * const NFKMetadataFormatKey	= @"format";
static NSString * const NFKMetadataSourceKey	= @"source";
static NSString * const NFKMetadataTurnKey		= @"turn";
static NSString * const NFKMetadataToolNameKey	= @"toolName";
static NSString * const NFKMetadataIsErrorKey	= @"isError";
static NSString * const NFKMetadataImagesKey	= @"images";

static NSDictionary *NFKMessageMetadata(NSDictionary *message)
{
	id metadata = [message isKindOfClass:NSDictionary.class] ? message[NFKMessageMetadataKey] : nil;
	return [metadata isKindOfClass:NSDictionary.class] ? metadata : @{};
}

static NSString *NFKJSONText(id object)
{
	if (![NSJSONSerialization isValidJSONObject:object]) {
		return @"{}";
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

// One call in the chat wire form, given an identifier when the backend reported none so its answer
// can name it.
static NSDictionary *NFKWireToolCall(NSDictionary *call)
{
	NSString *identifier = [call[@"id"] isKindOfClass:NSString.class] && [call[@"id"] length] > 0
		? call[@"id"] : [@"call_" stringByAppendingString:[NSUUID.UUID.UUIDString stringByReplacingOccurrencesOfString:@"-" withString:@""]];
	NSString *arguments = [call[@"argumentsJSON"] isKindOfClass:NSString.class] ? call[@"argumentsJSON"]
		: NFKJSONText([call[@"arguments"] isKindOfClass:NSDictionary.class] ? call[@"arguments"] : @{});
	NSString *name = [call[@"name"] isKindOfClass:NSString.class] ? call[@"name"] : @"";
	return @{ @"id": identifier, @"type": @"function", @"function": @{ @"name": name, @"arguments": arguments } };
}

NSDictionary<NSString *, id> *NFKAssistantMessage(NSString *text, NSArray<NSDictionary *> *toolCalls,
												  NSString *format, NSString *source, id turn)
{
	NSMutableDictionary *message = [NSMutableDictionary dictionaryWithObject:@"assistant" forKey:@"role"];
	NSMutableArray *wireCalls = [NSMutableArray array];
	for (NSDictionary *call in toolCalls ?: @[]) {
		if ([call isKindOfClass:NSDictionary.class]) {
			[wireCalls addObject:NFKWireToolCall(call)];
		}
	}
	message[@"content"] = text.length > 0 ? text : (wireCalls.count > 0 ? (id)NSNull.null : @"");
	if (wireCalls.count > 0) {
		message[@"tool_calls"] = wireCalls;
	}
	if (turn != nil && format != nil) {
		NSMutableDictionary *metadata = [NSMutableDictionary dictionaryWithDictionary:@{ NFKMetadataFormatKey: format, NFKMetadataTurnKey: turn }];
		metadata[NFKMetadataSourceKey] = source;
		message[NFKMessageMetadataKey] = metadata;
	}
	return message;
}

id NFKMessageProviderTurn(NSDictionary *message, NSString *format, NSString *source)
{
	NSDictionary *metadata = NFKMessageMetadata(message);
	if (![metadata[NFKMetadataFormatKey] isEqual:format]) {
		return nil;
	}
	id recorded = metadata[NFKMetadataSourceKey];
	if (recorded != nil && source != nil && ![recorded isEqual:source]) {
		return nil;
	}
	return metadata[NFKMetadataTurnKey];
}

NSDictionary<NSString *, id> *NFKMessageWithoutMetadata(NSDictionary *message)
{
	if (![message isKindOfClass:NSDictionary.class] || message[NFKMessageMetadataKey] == nil) {
		return message;
	}
	NSMutableDictionary *stripped = [message mutableCopy];
	[stripped removeObjectForKey:NFKMessageMetadataKey];
	return stripped;
}

BOOL NFKMessageIsToolResult(NSDictionary *message)
{
	return [message isKindOfClass:NSDictionary.class] && [message[@"role"] isEqual:@"tool"];
}

NSString *NFKToolMessageCallIdentifier(NSDictionary *message)
{
	return [message[@"tool_call_id"] isKindOfClass:NSString.class] ? message[@"tool_call_id"] : @"";
}

NSString *NFKToolMessageName(NSDictionary *message)
{
	id name = NFKMessageMetadata(message)[NFKMetadataToolNameKey] ?: message[@"name"];
	return [name isKindOfClass:NSString.class] ? name : nil;
}

NSString *NFKToolMessageNameIn(NSDictionary *message, NSArray *conversation)
{
	NSString *name = NFKToolMessageName(message);
	if (name != nil) {
		return name;
	}
	NSString *identifier = NFKToolMessageCallIdentifier(message);
	for (NSDictionary *turn in conversation) {
		if (![turn isKindOfClass:NSDictionary.class] || ![turn[@"tool_calls"] isKindOfClass:NSArray.class]) {
			continue;
		}
		for (NSDictionary *call in turn[@"tool_calls"]) {
			if ([call isKindOfClass:NSDictionary.class] && [call[@"id"] isEqual:identifier]) {
				id function = call[@"function"];
				return [function isKindOfClass:NSDictionary.class] && [function[@"name"] isKindOfClass:NSString.class] ? function[@"name"] : nil;
			}
		}
	}
	return nil;
}

NSString *NFKToolMessageText(NSDictionary *message)
{
	id content = message[@"content"];
	if ([content isKindOfClass:NSString.class]) {
		return content;
	}
	if (![content isKindOfClass:NSArray.class]) {
		return @"";
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	for (NSDictionary *part in content) {
		if ([part isKindOfClass:NSDictionary.class] && [part[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:part[@"text"]];
		}
	}
	return [texts componentsJoinedByString:@"\n"];
}

BOOL NFKToolMessageIsError(NSDictionary *message)
{
	return [NFKMessageMetadata(message)[NFKMetadataIsErrorKey] boolValue];
}

NSArray<NSData *> *NFKToolMessageImagePNGs(NSDictionary *message)
{
	NSMutableArray<NSData *> *images = [NSMutableArray array];
	id encoded = NFKMessageMetadata(message)[NFKMetadataImagesKey];
	for (NSString *base64 in [encoded isKindOfClass:NSArray.class] ? encoded : @[]) {
		NSData *png = [base64 isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:base64 options:0] : nil;
		if (png != nil) {
			[images addObject:png];
		}
	}
	return images;
}

@implementation NFKToolMessages

+ (NSDictionary<NSString *, id> *)assistantMessageForResult:(NFKInferenceResult *)result
{
	NSDictionary *written = [result outputForKey:NFKOutputAssistantMessage];
	if ([written isKindOfClass:NSDictionary.class]) {
		return written;
	}
	return NFKAssistantMessage(result.text, result.toolCalls, nil, nil, nil);
}

+ (NSArray<NSDictionary<NSString *, id> *> *)toolCallsOfAssistantMessage:(NSDictionary<NSString *, id> *)message
{
	NSMutableArray<NSDictionary *> *calls = [NSMutableArray array];
	NSArray *wireCalls = [message[@"tool_calls"] isKindOfClass:NSArray.class] ? message[@"tool_calls"] : @[];
	for (NSDictionary *wireCall in wireCalls) {
		NSDictionary *function = [wireCall isKindOfClass:NSDictionary.class] && [wireCall[@"function"] isKindOfClass:NSDictionary.class]
			? wireCall[@"function"] : nil;
		if (![function[@"name"] isKindOfClass:NSString.class]) {
			continue;
		}
		id arguments = function[@"arguments"];
		NSString *argumentsJSON = [arguments isKindOfClass:NSString.class] ? arguments : NFKJSONText(arguments ?: @{});
		id parsed = [arguments isKindOfClass:NSDictionary.class] ? arguments
			: [NSJSONSerialization JSONObjectWithData:[argumentsJSON dataUsingEncoding:NSUTF8StringEncoding] ?: [NSData data] options:0 error:NULL];
		[calls addObject:@{ @"id": [wireCall[@"id"] isKindOfClass:NSString.class] ? wireCall[@"id"] : @"",
							@"name": function[@"name"],
							@"arguments": [parsed isKindOfClass:NSDictionary.class] ? parsed : @{},
							@"argumentsJSON": argumentsJSON }];
	}
	return calls;
}

+ (NSDictionary<NSString *, id> *)messageForToolCallIdentifier:(NSString *)callIdentifier
														 name:(NSString *)toolName
													   result:(NFKToolResult *)result
{
	NSMutableDictionary *metadata = [NSMutableDictionary dictionaryWithObject:toolName forKey:NFKMetadataToolNameKey];
	if (result.isError) {
		metadata[NFKMetadataIsErrorKey] = @YES;
	}
	if (result.imagePNGs.count > 0) {
		NSMutableArray<NSString *> *images = [NSMutableArray arrayWithCapacity:result.imagePNGs.count];
		for (NSData *png in result.imagePNGs) {
			[images addObject:[png base64EncodedStringWithOptions:0]];
		}
		metadata[NFKMetadataImagesKey] = images;
	}
	return @{ @"role": @"tool", @"tool_call_id": callIdentifier, @"content": result.text, NFKMessageMetadataKey: metadata };
}

+ (NSDictionary<NSString *, id> *)assistantMessageWithText:(nullable NSString *)text
												 toolCalls:(nullable NSArray<NSDictionary<NSString *, id> *> *)toolCalls
													record:(nullable id)record
													format:(nullable NSString *)format
													source:(nullable NSString *)source
{
	return NFKAssistantMessage(text, toolCalls, format, source, record);
}

+ (nullable id)recordOfMessage:(NSDictionary<NSString *, id> *)message format:(NSString *)format source:(nullable NSString *)source
{
	return NFKMessageProviderTurn(message, format, source);
}

+ (nullable NSString *)toolNameOfMessage:(NSDictionary<NSString *, id> *)message
						  inConversation:(NSArray<NSDictionary<NSString *, id> *> *)conversation
{
	return NFKToolMessageNameIn(message, conversation);
}

+ (BOOL)isErrorToolMessage:(NSDictionary<NSString *, id> *)message
{
	return NFKToolMessageIsError(message);
}

+ (NSString *)textOfToolMessage:(NSDictionary<NSString *, id> *)message
{
	return NFKToolMessageText(message);
}

+ (NSArray<NSData *> *)imagePNGsOfToolMessage:(NSDictionary<NSString *, id> *)message
{
	return NFKToolMessageImagePNGs(message);
}

+ (NSArray<NSDictionary<NSString *, id> *> *)messagesWithoutMetadata:(NSArray<NSDictionary<NSString *, id> *> *)messages
{
	NSMutableArray<NSDictionary *> *stripped = [NSMutableArray arrayWithCapacity:messages.count];
	for (NSDictionary *message in messages) {
		[stripped addObject:NFKMessageWithoutMetadata(message)];
	}
	return stripped;
}

@end
