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
	return [self resultForArguments:arguments caller:nil error:error];
}

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments
										caller:(nullable NFKToolCaller *)caller
										 error:(NSError **)error
{
	__block NFKToolResult *result = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[self sendCallWithArguments:arguments caller:caller replyHandler:^(NFKToolResult *received, NSError *callError) {
		result = received;
		failure = callError;
		dispatch_semaphore_signal(done);
	}];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	if (result == nil && error != NULL) {
		*error = failure;
	}
	return result;
}

// The plugin runs the call while no thread here waits; only starting a stopped process holds this one.
- (nullable dispatch_block_t)startWithArguments:(NSDictionary<NSString *, id> *)arguments
										 caller:(nullable NFKToolCaller *)caller
									 completion:(void (^)(NFKToolResult *result))completion
{
	NSArray<NSString *> *problems = [self problemsWithArguments:arguments];
	if (problems.count > 0) {
		completion([NFKToolResult errorResultWithText:[NSString stringWithFormat:@"The call to %@ was not run: %@.",
													   self.name, [problems componentsJoinedByString:@"; "]]]);
		return nil;
	}
	NSString *name = self.name;
	[self sendCallWithArguments:arguments caller:caller replyHandler:^(NFKToolResult *received, NSError *failure) {
		NSString *reason = failure.localizedDescription ?: @"the tool reported no reason";
		completion(received ?: [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"%@ failed: %@", name, reason]]);
	}];
	return nil;
}

/*! Sends callTool, with the caller when there is one, and calls replyHandler once with the plugin's
	result, or nil and the error. */
- (void)sendCallWithArguments:(NSDictionary<NSString *, id> *)arguments
					   caller:(nullable NFKToolCaller *)caller
				 replyHandler:(void (^)(NFKToolResult * _Nullable result, NSError * _Nullable error))replyHandler
{
	NSMutableDictionary *message = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": NFKPluginMessageCallTool, @"tool": self.name,
																					@"arguments": arguments ?: @{} }];
	message[@"caller"] = caller != nil ? NFKPluginCallerJSONObject(caller) : nil;
	NSError *failure = nil;
	BOOL sent = [self.process performWithReservation:nil block:^BOOL(NFKPluginConnection *connection, NSError **blockError) {
		return [connection sendRequest:message descriptors:nil progressHandler:nil
						  replyHandler:^(NSDictionary<NSString *, id> * _Nullable received, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
			NSError *replyFailure = received != nil ? NFKPluginErrorFromReply(received) : replyError;
			if (replyFailure != nil) {
				replyHandler(nil, replyFailure);
				return;
			}
			id result = received[@"result"];
			id value = [result isKindOfClass:NSDictionary.class] ? result[@"toolResult"] : nil;
			NSError *decodeError = nil;
			NFKToolResult *decoded = [self toolResultFromJSONValue:value descriptors:descriptors error:&decodeError];
			replyHandler(decoded, decodeError);
		} error:blockError] != 0;
	} error:&failure];
	if (!sent) {
		replyHandler(nil, failure);
	}
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
