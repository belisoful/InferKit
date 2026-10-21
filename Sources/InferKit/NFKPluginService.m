//
//  NFKPluginService.m
//  InferKit
//

#import "NFKPluginService.h"
#import "NFKPluginServicePrivate.h"
#import "NFKPluginLoaderPrivate.h"
#import "NFKPluginManifestPrivate.h"
#import "NFKPluginCAdapter.h"
#import "NFKPluginChannel.h"
#import "NFKPluginWire.h"
#import "NFKPluginSignaturePolicy.h"
#import "NFKInferenceWireCoding.h"
#import "NFKInferenceTool.h"
#import "NFKInferKit.h"
#import "NFKErrors.h"
#import "NFKAudioAsset.h"
#import "NFKVideoAsset.h"

#if TARGET_OS_OSX

#import <fcntl.h>
#import <sysexits.h>
#import <unistd.h>

/*! The capability every service answers hello with: large values cross as shared-memory attachments. */
static NSString * const NFKPluginCapabilityAttachments = @"attachments";


static const NSTimeInterval NFKPluginServiceDefaultDrainTimeout = 5;

static NSError *NFKPluginServiceCancellation(void)
{
	return [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError
						   userInfo:@{ NSLocalizedDescriptionKey: @"the run was cancelled" }];
}

static BOOL NFKPluginServiceIsCancellation(NSError * _Nullable error)
{
	return error.code == NSUserCancelledError && [error.domain isEqualToString:NSCocoaErrorDomain];
}

/*! value when it is of class, else nil. */
static id _Nullable NFKPluginServiceField(id _Nullable value, Class class)
{
	return [value isKindOfClass:class] ? value : nil;
}

/*! The part of value JSON carries: strings, finite numbers, null, and arrays and string-keyed dictionaries
	of them. Anything else is left out. */
static id _Nullable NFKPluginServiceJSONSafe(id _Nullable value)
{
	if ([value isKindOfClass:NSString.class] || value == NSNull.null) {
		return value;
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return isfinite([value doubleValue]) ? value : nil;
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *safe = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			id kept = NFKPluginServiceJSONSafe(element);
			if (kept != nil) {
				[safe addObject:kept];
			}
		}
		return safe;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *safe = [NSMutableDictionary dictionaryWithCapacity:[value count]];
		for (id key in value) {
			id kept = [key isKindOfClass:NSString.class] ? NFKPluginServiceJSONSafe(value[key]) : nil;
			if (kept != nil) {
				safe[key] = kept;
			}
		}
		return safe;
	}
	return nil;
}

#pragma mark - Served backend and run

/*! A backend the host built, under the handle the service gave it. */
@interface NFKPluginServedBackend : NSObject
@property (nonatomic, strong) NSNumber *handle;
@property (nonatomic, strong) id<NFKInferenceBackend> backend;
@property (nonatomic, copy) NSString *modelIdentifier;
@property (nonatomic, strong) NFKPluginRunSlots *slots;
@property (atomic, assign) BOOL lastLoaded;
@end

@implementation NFKPluginServedBackend
@end

/*! One run the host asked for. It is answered once: with its result, its error, or cancelled. */
@interface NFKPluginServedRun : NSObject
@property (nonatomic, strong) NSNumber *identifier;
@property (nonatomic, strong) NFKPluginServedBackend *served;
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;
@property (atomic, strong, nullable) NFKInferenceJob *job;
@property (atomic, assign) BOOL cancelled;
@property (nonatomic, assign) BOOL replied;
@end

@implementation NFKPluginServedRun

/*! Claims the run's one reply. Returns NO when it was already given. */
- (BOOL)claimReply
{
	@synchronized (self) {
		if (self.replied) {
			return NO;
		}
		self.replied = YES;
		return YES;
	}
}

@end

#pragma mark - Service

@interface NFKPluginService ()
@property (nonatomic, readwrite, strong, nullable) id<NFKPlugin> plugin;
/*! The context a factory's plugin received, whose temporary directory goes when serving ends. */
@property (atomic, strong, nullable) NFKPluginHostContext *factoryContext;
@property (nonatomic, readwrite, copy) NFKPluginManifest *manifest;
@property (nonatomic, copy, nullable) NFKPluginServiceLoader loader;
@property (atomic, strong, nullable) NFKPluginChannel *channel;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NFKPluginServedBackend *> *backends;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NFKPluginServedRun *> *runs;
@property (nonatomic, assign) NSInteger lastHandle;
@property (atomic, assign) BOOL helloDone;
@property (atomic, assign) BOOL serving;
@property (atomic, assign) NSUInteger inlineByteLimit;
@property (atomic, assign) BOOL passesFilesAsDescriptors;
@property (atomic, copy, nullable) NSString *sharedMemoryPrefix;
@property (nonatomic, strong) dispatch_queue_t workQueue;
@property (atomic, strong, nullable) NSError *verificationError;
@end

@implementation NFKPluginService

- (instancetype)initWithPlugin:(id<NFKPlugin>)plugin manifest:(NFKPluginManifest *)manifest
{
	self = [super init];
	if (self != nil) {
		[self setUpWithManifest:manifest];
		_plugin = plugin;
	}
	return self;
}

- (instancetype)initWithManifest:(NFKPluginManifest *)manifest
				   pluginFactory:(id<NFKPlugin> _Nullable (^)(id<NFKPluginContext> context, NSError **error))factory
{
	__block __weak NFKPluginService *weakService = nil;
	NFKPluginManifest *described = [manifest copy];
	self = [self initWithManifest:described loader:^id<NFKPlugin>(NSDictionary<NSString *, id> *options, NFKPluginLogHandler logHandler,
																 NSError **error) {
		NFKPluginHostContext *context = [[NFKPluginHostContext alloc] initWithPluginIdentifier:described.identifier
																					   options:[described optionValuesWithDefaults:options]
																			 runsInHostProcess:NO logHandler:logHandler];
		id<NFKPlugin> plugin = factory(context, error);
		if (plugin == nil) {
			[context removeTemporaryDirectory];
			return nil;
		}
		weakService.factoryContext = context;
		return plugin;
	}];
	weakService = self;
	return self;
}

- (instancetype)initWithManifest:(NFKPluginManifest *)manifest loader:(NFKPluginServiceLoader)loader
{
	self = [super init];
	if (self != nil) {
		[self setUpWithManifest:manifest];
		_loader = [loader copy];
	}
	return self;
}

- (void)setUpWithManifest:(NFKPluginManifest *)manifest
{
	self.manifest = manifest;
	self.backends = [NSMutableDictionary dictionary];
	self.runs = [NSMutableDictionary dictionary];
	self.inlineByteLimit = NFKPluginDefaultInlineByteLimit;
	self.drainTimeout = NFKPluginServiceDefaultDrainTimeout;
	dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0);
	self.workQueue = dispatch_queue_create("com.inferkit.plugin-service.work", attributes);
}

#pragma mark Serving

- (BOOL)runWithChannelDescriptor:(int)descriptor error:(NSError **)error
{
	@synchronized (self) {
		if (self.serving) {
			close(descriptor);
			return NFKPluginSetError(error, NFKPluginErrorProtocolViolation, @"the service already serves a channel");
		}
		self.serving = YES;
	}
	NFKPluginChannel *channel = [NFKPluginChannel channelByHandingOffDescriptor:descriptor error:error];
	if (channel == nil) {
		self.serving = NO;
		return NO;
	}
	dispatch_semaphore_t ended = dispatch_semaphore_create(0);
	__block NSError *closeError = nil;
	__weak NFKPluginService *weakSelf = self;
	channel.messageHandler = ^(NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors) {
		[weakSelf handleMessage:message descriptors:descriptors];
	};
	channel.closeHandler = ^(NSError *closingError) {
		closeError = closingError;
		dispatch_semaphore_signal(ended);
	};
	self.channel = channel;
	[self verifyBeforeServing];
	[channel resume];
	dispatch_semaphore_wait(ended, DISPATCH_TIME_FOREVER);
#if DEBUG
	NFKPluginDebugTrace(@"service %p: its channel ended (%@); draining the runs", self, closeError);
#endif
	[self finishServing];
#if DEBUG
	NFKPluginDebugTrace(@"service %p: runs drained and backends released", self);
#endif
	channel.messageHandler = nil;
	channel.closeHandler = nil;
	self.channel = nil;
	self.serving = NO;
	[self.factoryContext removeTemporaryDirectory];
	if (closeError != nil) {
		if (error != NULL) {
			*error = closeError;
		}
		return NO;
	}
	return YES;
}

/*! Runs the verifier, if any, and tells the host it has finished, so the host times hello from here. */
- (void)verifyBeforeServing
{
	NSError * _Nullable (^verifier)(void) = self.verifier;
	if (verifier == nil) {
		return;
	}
	self.verificationError = verifier();
	[self.channel sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": NFKPluginEventVerified } error:NULL];
}

/*! Cancels the runs still in flight, waits up to drainTimeout for them to end, and releases every
	backend. */
- (void)finishServing
{
	NSArray<NFKPluginServedRun *> *runs = nil;
	@synchronized (self) {
		runs = self.runs.allValues;
	}
	for (NFKPluginServedRun *run in runs) {
		[self cancelRun:run];
	}
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:self.drainTimeout];
	while (deadline.timeIntervalSinceNow > 0) {
		@synchronized (self) {
			if (self.runs.count == 0) {
				break;
			}
		}
		[NSThread sleepForTimeInterval:0.02];
	}
	@synchronized (self) {
		[self.runs removeAllObjects];
		[self.backends removeAllObjects];
	}
}

#pragma mark Dispatch

// Runs on the channel's handler queue.
- (void)handleMessage:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NSString *type = message[@"type"];
	if ([type isEqualToString:NFKPluginMessageCancel]) {
		[self cancelRunWithIdentifier:NFKPluginServiceField(message[@"id"], NSNumber.class)];
		return;
	}
	NSNumber *identifier = NFKPluginServiceField(message[@"id"], NSNumber.class);
	if (identifier == nil || [type isEqualToString:NFKPluginMessageReply]) {
		return;
	}
	NSArray<NSString *> *beforeHello = @[ NFKPluginMessageHello, NFKPluginMessagePing, NFKPluginMessageShutdown ];
	if (!self.helloDone && [self handlesType:type] && ![beforeHello containsObject:type]) {
		[self reply:identifier failure:NFKPluginErrorProtocolViolation reason:[NSString stringWithFormat:@"\"%@\" came before hello", type]];
		return;
	}
	if ([type isEqualToString:NFKPluginMessageHello]) {
		[self handleHello:message];
	} else if ([type isEqualToString:NFKPluginMessageDescribe]) {
		[self handleDescribe:message];
	} else if ([type isEqualToString:NFKPluginMessageBuild]) {
		[self handleBuild:message];
	} else if ([type isEqualToString:NFKPluginMessagePrepare]) {
		[self handlePrepare:message];
	} else if ([type isEqualToString:NFKPluginMessageRun]) {
		[self handleRun:message descriptors:descriptors];
	} else if ([type isEqualToString:NFKPluginMessageUnload]) {
		[self handleUnload:message];
	} else if ([type isEqualToString:NFKPluginMessageRelease]) {
		[self handleRelease:message];
	} else if ([type isEqualToString:NFKPluginMessageStatus]) {
		[self handleStatus:message];
	} else if ([type isEqualToString:NFKPluginMessageCallTool]) {
		[self handleCallTool:message descriptors:descriptors];
	} else if ([type isEqualToString:NFKPluginMessagePing]) {
		[self reply:identifier result:@{} descriptors:nil];
	} else if ([type isEqualToString:NFKPluginMessageShutdown]) {
		[self handleShutdown:message];
	} else {
		[self reply:identifier failure:NFKPluginErrorProtocolUnsupported
			 reason:[NSString stringWithFormat:@"the plugin does not support the message \"%@\"", type]];
	}
}

- (BOOL)handlesType:(NSString *)type
{
	NSArray<NSString *> *types = @[ NFKPluginMessageHello, NFKPluginMessageDescribe, NFKPluginMessageBuild, NFKPluginMessagePrepare,
									NFKPluginMessageRun, NFKPluginMessageUnload, NFKPluginMessageRelease, NFKPluginMessageStatus,
									NFKPluginMessageCallTool, NFKPluginMessagePing, NFKPluginMessageShutdown ];
	return [types containsObject:type];
}

#pragma mark Handshake and description

- (void)handleHello:(NSDictionary<NSString *, id> *)message
{
	NSNumber *identifier = message[@"id"];
	if (self.helloDone) {
		[self reply:identifier failure:NFKPluginErrorProtocolViolation reason:@"hello came twice"];
		return;
	}
	NSDictionary *protocol = NFKPluginServiceField(message[@"protocol"], NSDictionary.class);
	NSNumber *major = NFKPluginServiceField(protocol[@"major"], NSNumber.class);
	NSNumber *minor = NFKPluginServiceField(protocol[@"minor"], NSNumber.class);
	if (major == nil || major.integerValue != NFKPluginProtocolMajorVersion) {
		[self reply:identifier failure:NFKPluginErrorProtocolUnsupported
			 reason:[NSString stringWithFormat:@"the host speaks plugin protocol %@.%@, and the plugin speaks %ld.%ld",
					 major ?: @"?", minor ?: @"?", (long)NFKPluginProtocolMajorVersion, (long)NFKPluginProtocolMinorVersion]];
		return;
	}
	NSNumber *inlineBytes = NFKPluginServiceField(NFKPluginServiceField(message[@"limits"], NSDictionary.class)[@"inlineBytes"], NSNumber.class);
	if (inlineBytes.longLongValue > 0) {
		self.inlineByteLimit = (NSUInteger)inlineBytes.longLongValue;
	}
	NSArray *offered = NFKPluginServiceField(message[@"capabilities"], NSArray.class) ?: @[];
	NSMutableArray<NSString *> *accepted = [NSMutableArray arrayWithObject:NFKPluginCapabilityAttachments];
	if ([offered containsObject:NFKPluginCapabilityFileDescriptors]) {
		[accepted addObject:NFKPluginCapabilityFileDescriptors];
		self.passesFilesAsDescriptors = YES;
	}
	self.sharedMemoryPrefix = NFKPluginServiceField(message[@"sharedMemoryPrefix"], NSString.class);
	if (self.verificationError != nil) {
		[self reply:identifier error:self.verificationError];
		return;
	}
	if (self.plugin == nil && ![self loadPluginWithOptions:NFKPluginServiceField(message[@"options"], NSDictionary.class) ?: @{} identifier:identifier]) {
		return;
	}
	self.helloDone = YES;
	NSInteger negotiatedMinor = MIN(minor.integerValue, NFKPluginProtocolMinorVersion);
	[self reply:identifier result:@{ @"protocol": @{ @"major": @(NFKPluginProtocolMajorVersion), @"minor": @(MAX(negotiatedMinor, 0)) },
									 @"plugin": @{ @"id": self.manifest.identifier, @"version": self.manifest.version },
									 @"inferkitAPI": NFKInferKit.APIVersion,
									 @"capabilities": accepted,
									 @"pid": @(getpid()) }
	descriptors:nil];
}

- (BOOL)loadPluginWithOptions:(NSDictionary<NSString *, id> *)options identifier:(NSNumber *)identifier
{
	NFKPluginServiceLoader loader = self.loader;
	if (loader == nil) {
		[self reply:identifier failure:NFKPluginErrorLoadFailed reason:@"the service has no plugin"];
		return NO;
	}
	__weak NFKPluginService *weakSelf = self;
	NSError *loadError = nil;
	id<NFKPlugin> plugin = loader(options, ^(NFKPluginLogLevel level, NSString *text) {
		[weakSelf sendLogMessage:text level:level];
	}, &loadError);
	if (plugin == nil) {
		NSError *failure = loadError;
		if (failure == nil) {
			NFKPluginSetError(&failure, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"the plugin %@ did not load", self.manifest.identifier]);
		}
		[self reply:identifier error:failure];
		return NO;
	}
	self.plugin = plugin;
	return YES;
}

- (void)handleDescribe:(NSDictionary<NSString *, id> *)message
{
	NSNumber *identifier = message[@"id"];
	id<NFKPlugin> plugin = self.plugin;
	dispatch_async(self.workQueue, ^{
		NSMutableArray *models = [NSMutableArray array];
		for (NFKModelDescriptor *model in [plugin modelDescriptors]) {
			[models addObject:model.dictionaryRepresentation];
		}
		NSMutableArray *tools = [NSMutableArray array];
		for (NFKInferenceTool *tool in [plugin respondsToSelector:@selector(tools)] ? [plugin tools] : @[]) {
			[tools addObject:tool.declaration];
		}
		NSMutableArray *options = [NSMutableArray array];
		for (NFKModelOption *option in [plugin respondsToSelector:@selector(pluginOptions)] ? [plugin pluginOptions] : @[]) {
			[options addObject:option.dictionaryRepresentation];
		}
		[self reply:identifier result:@{ @"models": NFKPluginServiceJSONSafe(models), @"tools": NFKPluginServiceJSONSafe(tools),
										 @"options": NFKPluginServiceJSONSafe(options) } descriptors:nil];
	});
}

#pragma mark Backends

- (nullable NFKPluginServedBackend *)servedBackendForMessage:(NSDictionary<NSString *, id> *)message
{
	NSNumber *handle = NFKPluginServiceField(message[@"backend"], NSNumber.class);
	NFKPluginServedBackend *served = nil;
	@synchronized (self) {
		served = handle != nil ? self.backends[handle] : nil;
	}
	if (served == nil) {
		[self reply:message[@"id"] failure:NFKPluginErrorProtocolViolation
			 reason:[NSString stringWithFormat:@"no backend has the handle %@", message[@"backend"] ?: @"(none)"]];
	}
	return served;
}

- (void)handleBuild:(NSDictionary<NSString *, id> *)message
{
	NSNumber *identifier = message[@"id"];
	NSString *model = NFKPluginServiceField(message[@"model"], NSString.class);
	NSDictionary *options = NFKPluginServiceField(message[@"options"], NSDictionary.class) ?: @{};
	if (model == nil || [self.manifest modelWithIdentifier:model] == nil) {
		[self reply:identifier failure:NFKPluginErrorUnknownModel
			 reason:[NSString stringWithFormat:@"the plugin %@ lists no model \"%@\"", self.manifest.identifier, model ?: @"(none)"]];
		return;
	}
	id<NFKPlugin> plugin = self.plugin;
	dispatch_async(self.workQueue, ^{
		NSError *buildError = nil;
		id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:model options:options error:&buildError];
		if (backend == nil) {
			if (buildError == nil) {
				NFKPluginSetError(&buildError, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"the plugin made no backend for \"%@\"", model]);
			}
			[self reply:identifier error:buildError];
			return;
		}
		NFKPluginServedBackend *served = [[NFKPluginServedBackend alloc] init];
		served.backend = backend;
		served.modelIdentifier = model;
		served.slots = [[NFKPluginRunSlots alloc] initWithLimit:self.manifest.maximumConcurrentRuns];
		served.lastLoaded = NFKInferenceIsLoaded(backend);
		@synchronized (self) {
			self.lastHandle += 1;
			served.handle = @(self.lastHandle);
			self.backends[served.handle] = served;
		}
		[self reply:identifier result:[self buildResultOfServedBackend:served] descriptors:nil];
	});
}

- (NSDictionary<NSString *, id> *)buildResultOfServedBackend:(NFKPluginServedBackend *)served
{
	id<NFKInferenceBackend> backend = served.backend;
	NSMutableDictionary<NSString *, id> *result = [NSMutableDictionary dictionary];
	result[@"backend"] = served.handle;
	result[@"ready"] = @(backend.isReady);
	result[@"loaded"] = @(served.lastLoaded);
	if ([backend respondsToSelector:@selector(supportedInputKeys)]) {
		result[@"inputs"] = [backend.supportedInputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedParameterKeys)]) {
		result[@"parameters"] = [backend.supportedParameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedOutputKeys)]) {
		result[@"outputs"] = [backend.supportedOutputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(modelInfo)]) {
		result[@"modelInfo"] = NFKPluginServiceJSONSafe(backend.modelInfo) ?: @{};
	}
	return result;
}

- (void)handlePrepare:(NSDictionary<NSString *, id> *)message
{
	NSNumber *identifier = message[@"id"];
	NFKPluginServedBackend *served = [self servedBackendForMessage:message];
	if (served == nil) {
		return;
	}
	dispatch_async(self.workQueue, ^{
		NSError *prepareError = nil;
		BOOL prepared = NFKInferencePrepare(served.backend, &prepareError);
		[self reportLoadStateOfServedBackend:served];
		if (!prepared) {
			[self reply:identifier error:prepareError ?: [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceNotReady
																		  userInfo:@{ NSLocalizedDescriptionKey: @"the backend did not prepare" }]];
			return;
		}
		[self reply:identifier result:[self memoryOfServedBackend:served] descriptors:nil];
	});
}

- (NSDictionary<NSString *, id> *)memoryOfServedBackend:(NFKPluginServedBackend *)served
{
	return @{ @"loaded": @(NFKInferenceIsLoaded(served.backend)),
			  @"residentBytes": @(NFKInferenceResidentByteCount(served.backend)),
			  @"peakBytes": @(NFKInferencePeakByteCount(served.backend)) };
}

- (void)handleUnload:(NSDictionary<NSString *, id> *)message
{
	NSNumber *identifier = message[@"id"];
	NFKPluginServedBackend *served = [self servedBackendForMessage:message];
	if (served == nil) {
		return;
	}
	dispatch_async(self.workQueue, ^{
		NFKInferenceUnload(served.backend);
		[self reportLoadStateOfServedBackend:served];
		[self reply:identifier result:@{} descriptors:nil];
	});
}

- (void)handleRelease:(NSDictionary<NSString *, id> *)message
{
	NSNumber *handle = NFKPluginServiceField(message[@"backend"], NSNumber.class);
	@synchronized (self) {
		if (handle != nil) {
			[self.backends removeObjectForKey:handle];
		}
	}
	[self reply:message[@"id"] result:@{} descriptors:nil];
}

- (void)handleStatus:(NSDictionary<NSString *, id> *)message
{
	NFKPluginServedBackend *served = [self servedBackendForMessage:message];
	if (served == nil) {
		return;
	}
	id<NFKInferenceBackend> backend = served.backend;
	NSMutableDictionary<NSString *, id> *result = [[self memoryOfServedBackend:served] mutableCopy];
	result[@"backendStatus"] = [backend respondsToSelector:@selector(backendStatus)] ? (NFKPluginServiceJSONSafe(backend.backendStatus) ?: @{}) : @{};
	[self reply:message[@"id"] result:result descriptors:nil];
}

/*! Sends a load event when the backend's isLoaded differs from what the host was last told. */
- (void)reportLoadStateOfServedBackend:(NFKPluginServedBackend *)served
{
	BOOL loaded = NFKInferenceIsLoaded(served.backend);
	BOOL changed = NO;
	@synchronized (served) {
		changed = loaded != served.lastLoaded;
		served.lastLoaded = loaded;
	}
	if (!changed) {
		return;
	}
	[self.channel sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": loaded ? @"loaded" : @"unloaded", @"backend": served.handle,
								 @"residentBytes": @(NFKInferenceResidentByteCount(served.backend)),
								 @"peakBytes": @(NFKInferencePeakByteCount(served.backend)) } error:NULL];
}

#pragma mark Runs

- (void)handleRun:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NSNumber *identifier = message[@"id"];
	NFKPluginServedBackend *served = [self servedBackendForMessage:message];
	if (served == nil) {
		return;
	}
	NSString *outputPath = NFKPluginServiceField(message[@"outputDirectory"], NSString.class);
	NSURL *outputDirectory = outputPath.isAbsolutePath ? [NSURL fileURLWithPath:outputPath isDirectory:YES] : nil;
	NFKPluginAttachmentReader *reader = [[NFKPluginAttachmentReader alloc] initWithDescriptors:descriptors];
	reader.copiesFileRegions = NO;
	reader.temporaryDirectoryURL = outputDirectory;
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:outputDirectory];
	decoder.attachmentSource = reader;
	NSError *decodeError = nil;
	NFKInferenceRequest *request = [decoder requestFromJSONObject:message[@"request"] error:&decodeError];
	if (request == nil) {
		[self reply:identifier error:decodeError];
		return;
	}
	NFKPluginServedRun *run = [[NFKPluginServedRun alloc] init];
	run.identifier = identifier;
	run.served = served;
	run.outputDirectoryURL = outputDirectory;
	@synchronized (self) {
		if (self.runs[identifier] != nil) {
			run = nil;
		} else {
			self.runs[identifier] = run;
		}
	}
	if (run == nil) {
		[self reply:identifier failure:NFKPluginErrorProtocolViolation reason:[NSString stringWithFormat:@"a run with the id %@ is in flight", identifier]];
		return;
	}
	dispatch_queue_t workQueue = self.workQueue;
	[served.slots admit:^{
		dispatch_async(workQueue, ^{
			[self executeRun:run request:request];
		});
	}];
}

// Runs on the work queue, holding one of the backend's run slots until finishRun:result:error:. A job's
// run ends in its completion handler, so a running job holds no thread.
- (void)executeRun:(NFKPluginServedRun *)run request:(NFKInferenceRequest *)request
{
	if (run.cancelled) {
		[self finishRun:run result:nil error:NFKPluginServiceCancellation()];
		return;
	}
	NFKInferenceJob *job = [self jobForRun:run request:request];
	if (job == nil) {
		NSError *error = nil;
		NFKInferenceResult *result = [run.served.backend runInferenceForRequest:request error:&error];
		[self finishRun:run result:result error:error];
		return;
	}
	__weak NFKPluginService *weakSelf = self;
	__weak NFKPluginServedRun *weakRun = run;
	job.progressHandler = ^(NFKInferenceJob *progressed) {
		NFKPluginServedRun *current = weakRun;
		if (current != nil) {
			[weakSelf sendProgressOfRun:current job:progressed];
		}
	};
	job.completionHandler = ^(NFKInferenceJob *done) {
		BOOL succeeded = done.status == NFKInferenceJobStatusSucceeded && done.result != nil;
		NSError *error = done.status == NFKInferenceJobStatusCancelled ? NFKPluginServiceCancellation() : done.error;
		[self finishRun:run result:succeeded ? done.result : nil error:succeeded ? nil : error];
	};
	BOOL cancelNow = NO;
	@synchronized (run) {
		run.job = job;
		cancelNow = run.cancelled;
	}
	if (cancelNow) {
		[job cancel];
	}
}

/*! The one end of an admitted run: frees its slot for the next waiting run and answers it. */
- (void)finishRun:(NFKPluginServedRun *)run result:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	run.job = nil;
	[run.served.slots finish];
	if (run.cancelled) {
		result = nil;
		error = NFKPluginServiceCancellation();
	}
	[self endRun:run result:result error:error];
}

/*! The job an asynchronous backend starts for the run, or nil for a backend that only runs
	synchronously. */
- (nullable NFKInferenceJob *)jobForRun:(NFKPluginServedRun *)run request:(NFKInferenceRequest *)request
{
	id<NFKInferenceBackend> backend = run.served.backend;
	if (run.outputDirectoryURL != nil && [backend conformsToProtocol:@protocol(NFKPluginOutputDirectoryBackend)]) {
		return [(id<NFKPluginOutputDirectoryBackend>)backend submitInferenceJobForRequest:request outputDirectoryURL:run.outputDirectoryURL];
	}
	if ([backend respondsToSelector:@selector(submitInferenceJobForRequest:)]) {
		return [backend submitInferenceJobForRequest:request];
	}
	return nil;
}

- (void)sendProgressOfRun:(NFKPluginServedRun *)run job:(NFKInferenceJob *)job
{
	if (run.cancelled) {
		return;
	}
	double progress = job.progress;
	NSMutableDictionary<NSString *, id> *message = [NSMutableDictionary dictionary];
	message[@"type"] = NFKPluginMessageProgress;
	message[@"id"] = run.identifier;
	message[@"progress"] = @(isfinite(progress) && progress >= 0 ? MIN(progress, 1.0) : -1.0);
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = nil;
	NFKInferenceResult *partial = job.partialResult;
	if (partial != nil) {
		writer = [self attachmentWriter];
		id encoded = [[self encoderWithWriter:writer] JSONObjectForResult:[self result:partial placingFilesIn:run.outputDirectoryURL] error:NULL];
		message[@"partial"] = encoded;
	}
	[self.channel sendMessage:message descriptors:writer.descriptors error:NULL];
}

- (void)endRun:(NFKPluginServedRun *)run result:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	@synchronized (self) {
		if (self.runs[run.identifier] == run) {
			[self.runs removeObjectForKey:run.identifier];
		}
	}
	if ([run claimReply]) {
		if (result != nil) {
			[self replyResult:result toRun:run];
		} else if (NFKPluginServiceIsCancellation(error)) {
			[self replyCancelled:run.identifier];
		} else {
			[self reply:run.identifier error:error ?: [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
																	userInfo:@{ NSLocalizedDescriptionKey: @"the run failed" }]];
		}
	}
	[self reportLoadStateOfServedBackend:run.served];
}

- (void)replyResult:(NFKInferenceResult *)result toRun:(NFKPluginServedRun *)run
{
	NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [self attachmentWriter];
	NSError *encodeError = nil;
	id encoded = [[self encoderWithWriter:writer] JSONObjectForResult:[self result:result placingFilesIn:run.outputDirectoryURL] error:&encodeError];
	if (encoded == nil) {
		[self reply:run.identifier error:encodeError];
		return;
	}
	[self reply:run.identifier result:encoded descriptors:writer.descriptors];
}

- (void)cancelRunWithIdentifier:(nullable NSNumber *)identifier
{
	NFKPluginServedRun *run = nil;
	@synchronized (self) {
		run = identifier != nil ? self.runs[identifier] : nil;
	}
	if (run != nil) {
		[self cancelRun:run];
	}
}

/*! Cancels a run. An asynchronous run ends through its job, which then answers cancelled. A run that
	waits for a slot or runs synchronously cannot be interrupted, so it is answered cancelled now and its
	result is dropped. */
- (void)cancelRun:(NFKPluginServedRun *)run
{
	NFKInferenceJob *job = nil;
	@synchronized (run) {
		run.cancelled = YES;
		job = run.job;
	}
	if (job != nil) {
		[job cancel];
	} else if ([run claimReply]) {
		[self replyCancelled:run.identifier];
	}
}

#pragma mark Result files

/*! The result with every file value that lies outside directory copied into it, so the host, which
	reads only its run's directory, receives it. */
- (NFKInferenceResult *)result:(NFKInferenceResult *)result placingFilesIn:(nullable NSURL *)directory
{
	if (directory == nil) {
		return result;
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithCapacity:result.outputs.count];
	[result.outputs enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
		outputs[key] = [self value:value placingFilesIn:directory];
	}];
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (id)value:(id)value placingFilesIn:(NSURL *)directory
{
	if ([value isKindOfClass:NSURL.class] && [value isFileURL]) {
		return [self fileURL:value placedIn:directory];
	}
	if ([value isKindOfClass:NFKAudioAsset.class] && [value fileURL] != nil) {
		NFKAudioAsset *asset = value;
		return [NFKAudioAsset audioAssetWithFileURL:[self fileURL:asset.fileURL placedIn:directory] durationSeconds:asset.durationSeconds
										 sampleRate:asset.sampleRate channelCount:asset.channelCount];
	}
	if ([value isKindOfClass:NFKVideoAsset.class] && [value fileURL] != nil) {
		NFKVideoAsset *asset = value;
		return [NFKVideoAsset videoAssetWithFileURL:[self fileURL:asset.fileURL placedIn:directory] durationSeconds:asset.durationSeconds
									framesPerSecond:asset.framesPerSecond dimensions:asset.dimensions];
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *placed = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			[placed addObject:[self value:element placingFilesIn:directory]];
		}
		return placed;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *placed = [NSMutableDictionary dictionaryWithCapacity:[value count]];
		[(NSDictionary *)value enumerateKeysAndObjectsUsingBlock:^(id key, id element, BOOL *stop) {
			placed[key] = [self value:element placingFilesIn:directory];
		}];
		return placed;
	}
	return value;
}

- (NSURL *)fileURL:(NSURL *)url placedIn:(NSURL *)directory
{
	NSString *inside = [directory.URLByStandardizingPath.path stringByAppendingString:@"/"];
	if ([url.URLByStandardizingPath.path hasPrefix:inside]) {
		return url;
	}
	NSURL *target = [directory URLByAppendingPathComponent:[NSString stringWithFormat:@"%@-%@", NSUUID.UUID.UUIDString, url.lastPathComponent]];
	return [NSFileManager.defaultManager copyItemAtURL:url toURL:target error:NULL] ? target : url;
}

#pragma mark Tools

- (void)handleCallTool:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NSNumber *identifier = message[@"id"];
	NSString *name = NFKPluginServiceField(message[@"tool"], NSString.class);
	NSDictionary *arguments = NFKPluginServiceField(message[@"arguments"], NSDictionary.class) ?: @{};
	id<NFKPlugin> plugin = self.plugin;
	NFKInferenceTool *tool = nil;
	for (NFKInferenceTool *candidate in [plugin respondsToSelector:@selector(tools)] ? [plugin tools] : @[]) {
		if ([candidate.name isEqualToString:name ?: @""]) {
			tool = candidate;
			break;
		}
	}
	if (tool == nil) {
		[self reply:identifier failure:NFKPluginErrorUnknownTool
			 reason:[NSString stringWithFormat:@"the plugin %@ has no tool \"%@\"", self.manifest.identifier, name ?: @"(none)"]];
		return;
	}
	dispatch_async(self.workQueue, ^{
		NFKToolResult *result = [tool invokeWithArguments:arguments];
		NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [self attachmentWriter];
		NSError *encodeError = nil;
		id encoded = [[self encoderWithWriter:writer] JSONValueForValue:result key:@"toolResult" error:&encodeError];
		if (encoded == nil) {
			[self reply:identifier error:encodeError];
			return;
		}
		[self reply:identifier result:@{ @"toolResult": encoded } descriptors:writer.descriptors];
	});
}

#pragma mark Shutdown and logs

- (void)handleShutdown:(NSDictionary<NSString *, id> *)message
{
	NSArray<NFKPluginServedRun *> *runs = nil;
	@synchronized (self) {
		runs = self.runs.allValues;
	}
	for (NFKPluginServedRun *run in runs) {
		[self cancelRun:run];
	}
	[self reply:message[@"id"] result:@{} descriptors:nil];
#if DEBUG
	NFKPluginDebugTrace(@"service %p answered shutdown and closes its channel %@", self, self.channel);
#endif
	[self.channel close];
}

- (void)sendLogMessage:(NSString *)message level:(NFKPluginLogLevel)level
{
	NFKPluginChannel *channel = self.channel;
	if (channel != nil && !channel.isClosed
		&& [channel sendMessage:@{ @"type": NFKPluginMessageLog, @"level": @(level), @"message": message } error:NULL]) {
		return;
	}
	fprintf(stderr, "%s: %s\n", self.manifest.identifier.UTF8String, message.UTF8String);
}

#pragma mark Coding and replies

- (NFKPluginAttachmentWriter *)attachmentWriter
{
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	writer.namePrefix = self.sharedMemoryPrefix;
	writer.passesFilesAsDescriptors = self.passesFilesAsDescriptors;
	return writer;
}

- (NFKInferenceWireCoding *)encoderWithWriter:(NFKPluginAttachmentWriter *)writer
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSink = writer;
	coder.inlineByteLimit = self.inlineByteLimit;
	coder.encodesValueObjectsAsJSON = YES;
	return coder;
}

- (void)reply:(NSNumber *)identifier result:(id)result descriptors:(nullable NSArray<NSNumber *> *)descriptors
{
	NSError *sendError = nil;
	BOOL sent = [self.channel sendMessage:@{ @"type": NFKPluginMessageReply, @"id": identifier, @"result": result }
							  descriptors:descriptors error:&sendError];
	if (!sent && sendError.code == NFKPluginErrorProtocolViolation && [sendError.domain isEqualToString:NFKPluginErrorDomain]) {
		[self reply:identifier error:sendError];
	}
}

- (void)reply:(NSNumber *)identifier error:(NSError *)error
{
	NSDictionary *body = [NFKInferenceWireCoding JSONObjectForError:error][@"error"];
	[self.channel sendMessage:@{ @"type": NFKPluginMessageReply, @"id": identifier, @"error": body } error:NULL];
}

- (void)reply:(NSNumber *)identifier failure:(NFKPluginError)code reason:(NSString *)reason
{
	NSError *error = nil;
	NFKPluginSetError(&error, code, reason);
	[self reply:identifier error:error];
}

- (void)replyCancelled:(NSNumber *)identifier
{
	[self.channel sendMessage:@{ @"type": NFKPluginMessageReply, @"id": identifier, @"cancelled": @YES } error:NULL];
}

@end

#pragma mark - Plugin host

static const char * const NFKPluginHostUsage =
	"usage: inferkit-plugin-host --bundle <path> [--team-id <team>] [--requirement <requirement>] [--allow-ad-hoc [--allow-unsigned]]\n";

/*! The parsed arguments of the plugin host. */
typedef struct {
	const char *bundlePath;
	const char *teamIdentifier;
	const char *requirement;
	BOOL allowsAdHoc;
	BOOL allowsUnsigned;
} NFKPluginHostArguments;

static BOOL NFKPluginHostParseArguments(int argc, char **argv, NFKPluginHostArguments *arguments)
{
	*arguments = (NFKPluginHostArguments){ NULL, NULL, NULL, NO, NO };
	for (int index = 1; index < argc; index++) {
		const char *argument = argv[index];
		if (argument == NULL) {
			return NO;
		}
		if (strcmp(argument, "--bundle") == 0 && index + 1 < argc) {
			arguments->bundlePath = argv[++index];
		} else if (strcmp(argument, "--team-id") == 0 && index + 1 < argc) {
			arguments->teamIdentifier = argv[++index];
		} else if (strcmp(argument, "--requirement") == 0 && index + 1 < argc) {
			arguments->requirement = argv[++index];
		} else if (strcmp(argument, "--allow-ad-hoc") == 0) {
			arguments->allowsAdHoc = YES;
		} else if (strcmp(argument, "--allow-unsigned") == 0) {
			arguments->allowsUnsigned = YES;
		} else {
			return NO;
		}
	}
	return arguments->bundlePath != NULL;
}

/*! The policy the arguments name. --allow-unsigned is for local development and needs the development
	policy, --allow-ad-hoc. Returns nil with a reason otherwise. */
static NFKPluginSignaturePolicy * _Nullable NFKPluginHostPolicy(const NFKPluginHostArguments *arguments, NSString * _Nullable * _Nonnull reason)
{
	if (arguments->allowsUnsigned && !arguments->allowsAdHoc) {
		*reason = @"--allow-unsigned is for local development and needs --allow-ad-hoc";
		return nil;
	}
	NFKPluginSignaturePolicy *policy = arguments->allowsAdHoc ? NFKPluginSignaturePolicy.developmentPolicy : NFKPluginSignaturePolicy.defaultPolicy;
	policy.requiresValidSignature = !arguments->allowsUnsigned;
	policy.teamIdentifier = arguments->teamIdentifier != NULL ? [NSString stringWithUTF8String:arguments->teamIdentifier] : nil;
	policy.requirementString = arguments->requirement != NULL ? [NSString stringWithUTF8String:arguments->requirement] : nil;
	return policy;
}

/*! The channel descriptor INFERKIT_PLUGIN_CHANNEL_FD names, else 3, or -1 when it is not open. */
static int NFKPluginHostChannelDescriptor(void)
{
	const char *named = getenv(NFKPluginChannelEnvironmentVariable.UTF8String);
	int descriptor = named != NULL ? atoi(named) : NFKPluginChannelDescriptorNumber;
	return descriptor >= 0 && fcntl(descriptor, F_GETFD) != -1 ? descriptor : -1;
}

int NFKPluginHostMain(int argc, char **argv)
{
	@autoreleasepool {
		NFKPluginHostArguments arguments;
		NSString *reason = nil;
		NFKPluginSignaturePolicy *policy = NFKPluginHostParseArguments(argc, argv, &arguments) ? NFKPluginHostPolicy(&arguments, &reason) : nil;
		if (policy == nil) {
			if (reason != nil) {
				fprintf(stderr, "inferkit-plugin-host: %s\n", reason.UTF8String);
			}
			fputs(NFKPluginHostUsage, stderr);
			return EX_USAGE;
		}
		int descriptor = NFKPluginHostChannelDescriptor();
		if (descriptor < 0) {
			fprintf(stderr, "inferkit-plugin-host: no plugin channel is open on descriptor %d\n", NFKPluginChannelDescriptorNumber);
			return EX_OSERR;
		}
		NSURL *bundleURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:arguments.bundlePath] isDirectory:YES];
		NSError *error = nil;
		NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:bundleURL error:&error];
		if (manifest == nil) {
			fprintf(stderr, "inferkit-plugin-host: %s\n", error.localizedDescription.UTF8String);
			return EX_DATAERR;
		}
		__block NFKLoadedPlugin *loaded = nil;
		NFKPluginService *service = [[NFKPluginService alloc] initWithManifest:manifest loader:^id<NFKPlugin>(NSDictionary<NSString *, id> *options,
																											 NFKPluginLogHandler logHandler,
																											 NSError **loadError) {
			// The loader checks ownership and the signature again immediately before the code loads; the
			// verifier's success is cached, so the second check hashes nothing.
			NFKLoadedPlugin *plugin = [NFKPluginLoader loadPluginAtURL:bundleURL signaturePolicy:policy options:options
													 runsInHostProcess:NO logHandler:logHandler error:loadError];
			if (plugin != nil && ![plugin.manifest isEqual:manifest]) {
				NFKPluginSetError(loadError, NFKPluginErrorManifestMismatch,
								  [NSString stringWithFormat:@"the plugin %@ changed on disk after the host read its manifest", manifest.identifier]);
				return nil;
			}
			loaded = plugin;
			return plugin.plugin;
		}];
		service.verifier = ^NSError *{
			NSError *verificationError = nil;
			if (!NFKPluginCheckBundleOwnership(bundleURL, &verificationError) || ![policy validateCodeAtURL:bundleURL error:&verificationError]) {
				return verificationError;
			}
			return nil;
		};
		BOOL orderly = [service runWithChannelDescriptor:descriptor error:&error];
#if DEBUG
		NFKPluginDebugTrace(@"plugin host: the service returned %@; unloading the plugin", orderly ? @"orderly" : error);
#endif
		if ([loaded.plugin respondsToSelector:@selector(willUnload)]) {
			[loaded.plugin willUnload];
		}
		[loaded.context removeTemporaryDirectory];
#if DEBUG
		NFKPluginDebugTrace(@"plugin host: returning from main with status %d", orderly ? 0 : EX_PROTOCOL);
#endif
		if (!orderly) {
			fprintf(stderr, "inferkit-plugin-host: %s\n", error.localizedDescription.UTF8String);
			return EX_PROTOCOL;
		}
		return 0;
	}
}

#endif
