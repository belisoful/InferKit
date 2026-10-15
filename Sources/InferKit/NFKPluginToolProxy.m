//
//  NFKPluginToolProxy.m
//  InferKit
//

#import "NFKPluginToolProxy.h"

#if TARGET_OS_OSX

#import "NFKPluginChannel.h"
#import "NFKPluginProcessPrivate.h"
#import "NFKPluginWire.h"

@implementation NFKPluginToolProxy

- (nullable instancetype)initWithProcess:(NFKPluginProcess *)process declaration:(NSDictionary<NSString *, id> *)declaration
{
	NSString *name = [declaration[@"name"] isKindOfClass:NSString.class] ? declaration[@"name"] : nil;
	if (name.length == 0) {
		return nil;
	}
	NSString *summary = [declaration[@"description"] isKindOfClass:NSString.class] ? declaration[@"description"] : @"";
	NSDictionary *parameters = [declaration[@"parameters"] isKindOfClass:NSDictionary.class] ? declaration[@"parameters"] : @{ @"type": @"object" };
	self = [super initWithName:name description:summary parameters:parameters];
	if (self != nil) {
		_process = process;
	}
	return self;
}

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	__block NSDictionary<NSString *, id> *reply = nil;
	__block NSArray<NFKPluginDescriptor *> *replyDescriptors = @[];
	__block NSError *failure = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	NSDictionary *message = @{ @"type": NFKPluginMessageCallTool, @"tool": self.name, @"arguments": arguments ?: @{} };
	BOOL sent = [self.process performWithReservation:nil block:^BOOL(NFKPluginConnection *connection, NSError **blockError) {
		return [connection sendRequest:message descriptors:nil progressHandler:nil
						  replyHandler:^(NSDictionary<NSString *, id> * _Nullable received, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
			reply = received;
			replyDescriptors = descriptors;
			failure = received != nil ? NFKPluginErrorFromReply(received) : replyError;
			dispatch_semaphore_signal(done);
		} error:blockError] != 0;
	} error:&failure];
	if (sent) {
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	}
	if (!sent || failure != nil) {
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	id result = reply[@"result"];
	id value = [result isKindOfClass:NSDictionary.class] ? result[@"toolResult"] : nil;
	return [self toolResultFromJSONValue:value descriptors:replyDescriptors error:error];
}

/*! A toolResult in the wire coding's tagged form, a {text, isError} object, or a string. */
- (nullable NFKToolResult *)toolResultFromJSONValue:(nullable id)value
										descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
											  error:(NSError **)error
{
	if ([value isKindOfClass:NSString.class]) {
		return [NFKToolResult resultWithText:value];
	}
	if (![value isKindOfClass:NSDictionary.class]) {
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation, [NSString stringWithFormat:@"the plugin's %@ answered without a toolResult", self.name]);
		return nil;
	}
	if (value[@"$nfk"] != nil) {
		NFKPluginAttachmentReader *reader = [[NFKPluginAttachmentReader alloc] initWithDescriptors:descriptors];
		NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
		decoder.attachmentSource = reader;
		id decoded = [decoder valueFromJSONValue:value key:@"toolResult" error:error];
		if ([decoded isKindOfClass:NFKToolResult.class]) {
			return decoded;
		}
		if (decoded != nil) {
			NFKPluginSetError(error, NFKPluginErrorProtocolViolation, [NSString stringWithFormat:@"the plugin's %@ answered with a value that is not a tool result", self.name]);
		}
		return nil;
	}
	NSString *text = [value[@"text"] isKindOfClass:NSString.class] ? value[@"text"] : @"";
	BOOL isError = [value[@"isError"] isKindOfClass:NSNumber.class] && [value[@"isError"] boolValue];
	return [[NFKToolResult alloc] initWithText:text outputs:nil imagePNGs:nil isError:isError];
}

@end

#endif
