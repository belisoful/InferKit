//
//  NFKPluginTestPeer.m
//  InferKitTests
//

#import "NFKPluginTestPeer.h"

#if TARGET_OS_OSX

#import <unistd.h>

BOOL NFKPluginTestWaitUntil(NSTimeInterval timeout, BOOL (^condition)(void))
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow < 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return YES;
}

NFKPluginManifest *NFKPluginTestManifest(NSString *version, NSTimeInterval idleExitSeconds)
{
	NSDictionary *model = @{
		@"id": @"echo",
		@"displayName": @"Echo",
		@"family": @"echo",
		@"engine": @"plugin",
		@"tasks": @[ @"text-generation" ],
		@"inputs": @[ @{ @"key": NFKInputPrompt, @"required": @YES } ],
		@"outputs": @[ @{ @"key": NFKOutputText, @"required": @YES } ],
		@"parameters": @[
			@{ @"key": NFKParameterTemperature, @"kind": @"number", @"stage": @"request" },
			@{ @"key": @"size", @"kind": @"text", @"stage": @"load" },
		],
	};
	NSDictionary *manifest = @{
		@"schemaVersion": @1,
		@"kind": @"inferkit.plugin",
		@"id": @"com.example.echo",
		@"version": version,
		@"displayName": @"Echo",
		@"summary": @"Echoes its prompt.",
		@"compatibility": @{ @"inferkitAPI": @"1.0", @"pluginABI": @"1.0", @"protocol": @"1.0",
							 @"minimumOS": @{ @"macOS": @"11.0" }, @"architectures": @[ @"arm64", @"x86_64" ] },
		@"entry": @{ @"kind": @"objc", @"principalClass": @"EchoPlugin" },
		@"process": @{ @"idleExitSeconds": @(idleExitSeconds), @"allowsInProcess": @NO, @"maximumConcurrentRuns": @1 },
		@"options": @[],
		@"models": @[ model ],
		@"tools": @[ @{ @"name": @"shout", @"description": @"Shouts its text.",
						@"parameters": @{ @"type": @"object", @"properties": @{ @"text": @{ @"type": @"string" } } } } ],
	};
	NSData *data = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL];
	NSError *error = nil;
	NFKPluginManifest *parsed = [NFKPluginManifest manifestWithJSONData:data error:&error];
	NSCAssert(parsed != nil, @"the test manifest reads: %@", error);
	return parsed;
}

@interface NFKPluginTestPeer ()
@property (atomic, readwrite) BOOL stopped;
@property (atomic, readwrite) pid_t auditedProcessIdentifier;
@property (atomic, strong, nullable) NFKPluginChannel *channel;
@property (nonatomic, strong) NSMutableArray<NSDictionary<NSString *, id> *> *messages;
@property (atomic, copy) NSArray<NSDictionary<NSString *, id> *> *manifestModels;
@property (atomic, assign) NSInteger nextHandle;
@end

@implementation NFKPluginTestPeer

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_protocol = @{ @"major": @1, @"minor": @0 };
		_handlers = @{};
		_recentStandardError = @"";
		_messages = [NSMutableArray array];
		_manifestModels = @[];
		_nextHandle = 100;
	}
	return self;
}

#pragma mark Launcher

- (BOOL)launchPluginProcess:(NFKPluginProcess *)process channelDescriptor:(int)descriptor error:(NSError **)error
{
	if (self.failsLaunch) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NSPOSIXErrorDomain code:ENOENT userInfo:@{ NSLocalizedDescriptionKey: @"the test launch fails" }];
		}
		return NO;
	}
	NFKPluginManifest *manifest = process.manifest;
	self.pluginIdentifier = self.pluginIdentifier ?: manifest.identifier;
	self.pluginVersion = self.pluginVersion ?: manifest.version;
	NSMutableArray *models = [NSMutableArray array];
	for (NFKModelDescriptor *model in manifest.models) {
		[models addObject:model.dictionaryRepresentation];
	}
	self.manifestModels = models;
	NFKPluginChannel *channel = [NFKPluginChannel channelByHandingOffDescriptor:dup(descriptor) error:error];
	if (channel == nil) {
		return NO;
	}
	channel.messageHandler = ^(NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors) {
		[self receiveMessage:message descriptors:descriptors];
	};
	__weak NFKPluginChannel *weakChannel = channel;
	channel.closeHandler = ^(NSError * _Nullable closeError) {
		weakChannel.messageHandler = nil;
	};
	self.channel = channel;
	[channel resume];
	return YES;
}

- (void)stopWithCompletionHandler:(void (^)(void))completionHandler
{
	self.stopped = YES;
	[self.channel close];
	if (self.terminationReason == NFKSubprocessTerminationReasonNone) {
		self.terminationReason = NFKSubprocessTerminationReasonExit;
		self.terminationStatus = 0;
	}
	completionHandler();
}

- (pid_t)processIdentifier
{
	return 0;
}

- (BOOL)validatePeerWithAuditToken:(audit_token_t)auditToken process:(NFKPluginProcess *)process error:(NSError **)error
{
	self.auditedProcessIdentifier = (pid_t)auditToken.val[5];
	return YES;
}

- (void)crashWithSignal:(int)signal standardError:(NSString *)standardError
{
	self.terminationReason = NFKSubprocessTerminationReasonSignal;
	self.terminationStatus = signal;
	self.recentStandardError = standardError;
	[self.channel close];
}

#pragma mark Recording

- (NSArray<NSDictionary<NSString *, id> *> *)receivedMessages
{
	@synchronized (self.messages) {
		return [self.messages copy];
	}
}

- (NSArray<NSString *> *)receivedTypes
{
	return [self.receivedMessages valueForKey:@"type"];
}

- (NSArray<NSDictionary<NSString *, id> *> *)receivedMessagesOfType:(NSString *)type
{
	return [self.receivedMessages filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"type == %@", type]];
}

- (nullable NSDictionary<NSString *, id> *)waitForMessageOfType:(NSString *)type timeout:(NSTimeInterval)timeout
{
	NFKPluginTestWaitUntil(timeout, ^BOOL {
		return [self receivedMessagesOfType:type].count > 0;
	});
	return [self receivedMessagesOfType:type].firstObject;
}

#pragma mark Answering

- (void)receiveMessage:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	@synchronized (self.messages) {
		[self.messages addObject:message];
	}
	NSString *type = message[@"type"];
	NFKPluginTestPeerHandler handler = self.handlers[type];
	if (handler != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			handler(self, message, descriptors);
		});
		return;
	}
	[self answerMessage:message descriptors:descriptors];
}

- (void)answerMessage:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NSString *type = message[@"type"];
	if ([type isEqualToString:NFKPluginMessageHello]) {
		[self replyTo:message result:@{ @"protocol": self.protocol,
										 @"plugin": @{ @"id": self.pluginIdentifier ?: @"", @"version": self.pluginVersion ?: @"" },
										 @"inferkitAPI": @"1.0", @"capabilities": @[], @"pid": @(getpid()) }];
	} else if ([type isEqualToString:NFKPluginMessageDescribe]) {
		[self replyTo:message result:@{ @"models": [self describedModels], @"tools": @[], @"options": @[] }];
	} else if ([type isEqualToString:NFKPluginMessageBuild]) {
		NSInteger handle = 0;
		@synchronized (self) {
			self.nextHandle += 1;
			handle = self.nextHandle;
		}
		[self replyTo:message result:@{ @"backend": @(handle), @"ready": @YES, @"loaded": @NO,
										 @"modelInfo": @{ NFKModelInfoParameterCount: @7 } }];
	} else if ([type isEqualToString:NFKPluginMessagePrepare]) {
		[self replyTo:message result:@{ @"loaded": @YES, @"residentBytes": @1000, @"peakBytes": @2000 }];
	} else if ([type isEqualToString:NFKPluginMessageStatus]) {
		[self replyTo:message result:@{ @"loaded": @YES, @"residentBytes": @1000, @"peakBytes": @2000, @"backendStatus": @{ @"queued": @0 } }];
	} else if ([type isEqualToString:NFKPluginMessageRun]) {
		NFKInferenceRequest *request = [self requestOfRun:message descriptors:descriptors];
		NSString *text = [NSString stringWithFormat:@"%lu:%@", (unsigned long)self.launchNumber, request.prompt ?: @""];
		[self replyTo:message inferenceResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: text }]];
	} else if ([type isEqualToString:NFKPluginMessageUnload]) {
		[self replyTo:message result:@{}];
		[self sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": @"unloaded", @"backend": message[@"backend"] ?: NSNull.null }];
	} else if ([type isEqualToString:NFKPluginMessageCallTool]) {
		[self replyTo:message result:@{ @"toolResult": @{ @"text": [NSString stringWithFormat:@"%@ called", message[@"tool"]], @"isError": @NO } }];
	} else if ([type isEqualToString:NFKPluginMessageShutdown]) {
		[self replyTo:message result:@{}];
		[self.channel close];
	} else if (message[@"id"] != nil) {
		[self replyTo:message result:@{}];
	}
}

- (NSArray *)describedModels
{
	NSArray<NSString *> *identifiers = self.describedModelIdentifiers;
	if (identifiers == nil) {
		return self.manifestModels;
	}
	NSMutableArray *models = [NSMutableArray array];
	for (NSString *identifier in identifiers) {
		[models addObject:@{ @"id": identifier }];
	}
	return models;
}

#pragma mark Sending

- (void)sendMessage:(NSDictionary<NSString *, id> *)message
{
	[self.channel sendMessage:message error:NULL];
}

- (void)replyTo:(NSDictionary<NSString *, id> *)request result:(id)result
{
	[self sendMessage:@{ @"type": NFKPluginMessageReply, @"id": request[@"id"], @"result": result }];
}

- (void)replyTo:(NSDictionary<NSString *, id> *)request error:(NSError *)error
{
	[self sendMessage:@{ @"type": NFKPluginMessageReply, @"id": request[@"id"], @"error": [NFKInferenceWireCoding JSONObjectForError:error][@"error"] }];
}

- (void)replyCancelledTo:(NSDictionary<NSString *, id> *)request
{
	[self sendMessage:@{ @"type": NFKPluginMessageReply, @"id": request[@"id"], @"cancelled": @YES }];
}

- (void)replyTo:(NSDictionary<NSString *, id> *)request inferenceResult:(NFKInferenceResult *)result
{
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSink = writer;
	NSError *error = nil;
	id encoded = [coder JSONObjectForResult:result error:&error];
	if (encoded == nil) {
		[self replyTo:request error:error];
		return;
	}
	[self.channel sendMessage:@{ @"type": NFKPluginMessageReply, @"id": request[@"id"], @"result": encoded }
				  descriptors:writer.descriptors error:NULL];
}

- (void)sendProgress:(double)progress partial:(nullable NFKInferenceResult *)partial forRequest:(NSDictionary<NSString *, id> *)request
{
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	NSMutableDictionary *message = [NSMutableDictionary dictionaryWithDictionary:@{ @"type": NFKPluginMessageProgress, @"id": request[@"id"], @"progress": @(progress) }];
	if (partial != nil) {
		NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
		coder.attachmentSink = writer;
		message[@"partial"] = [coder JSONObjectForResult:partial error:NULL];
	}
	[self.channel sendMessage:message descriptors:writer.descriptors error:NULL];
}

- (nullable NFKInferenceRequest *)requestOfRun:(NSDictionary<NSString *, id> *)run descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NFKPluginAttachmentReader *reader = [[NFKPluginAttachmentReader alloc] initWithDescriptors:descriptors];
	reader.copiesFileRegions = NO;
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSource = reader;
	return [coder requestFromJSONObject:run[@"request"] error:NULL];
}

@end

#pragma mark - Harness

@interface NFKPluginTestHarness ()
@property (nonatomic, strong) NSMutableArray<NFKPluginTestPeer *> *launched;
@end

@implementation NFKPluginTestHarness

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_launched = [NSMutableArray array];
	}
	return self;
}

- (NSArray<NFKPluginTestPeer *> *)peers
{
	@synchronized (self.launched) {
		return [self.launched copy];
	}
}

- (NFKPluginLauncherFactory)factory
{
	return ^id<NFKPluginLauncher>(NFKPluginProcess *process) {
		NFKPluginTestPeer *peer = [[NFKPluginTestPeer alloc] init];
		@synchronized (self.launched) {
			[self.launched addObject:peer];
			peer.launchNumber = self.launched.count;
		}
		void (^configure)(NFKPluginTestPeer *) = self.configure;
		if (configure != nil) {
			configure(peer);
		}
		return peer;
	};
}

@end

#endif
