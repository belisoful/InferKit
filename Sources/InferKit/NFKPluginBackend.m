//
//  NFKPluginBackend.m
//  InferKit
//

#import "NFKPluginBackend.h"
#import "NFKPluginProcess.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import "NFKPluginChannel.h"
#import "NFKPluginProcessPrivate.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKModelVocabulary.h>
#import <objc/runtime.h>

static const void *NFKPluginRunDirectoryKey = &NFKPluginRunDirectoryKey;

/*! How often backendStatus asks the plugin for a fresh status, at most, in seconds. */
static const NSTimeInterval NFKPluginStatusRefreshInterval = 1;

id _Nullable NFKPluginResultFileOwner(NFKInferenceResult *result)
{
	return objc_getAssociatedObject(result, NFKPluginRunDirectoryKey);
}

#pragma mark - Run directory

/*! The directory the host makes for one run's output files. It is removed when the object is
	released, and every result decoded from the run holds it. */
@interface NFKPluginRunDirectory : NSObject
@property (nonatomic, readonly, copy) NSURL *URL;
@end

@implementation NFKPluginRunDirectory

+ (nullable instancetype)directoryWithError:(NSError **)error
{
	NSURL *parent = [[[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES]
					 URLByAppendingPathComponent:@"plugin-runs" isDirectory:YES];
	NSURL *directory = [parent URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES
												 attributes:@{ NSFilePosixPermissions: @0700 } error:error]) {
		return nil;
	}
	return [[self alloc] initWithURL:directory.URLByResolvingSymlinksInPath];
}

- (instancetype)initWithURL:(NSURL *)URL
{
	self = [super init];
	if (self != nil) {
		_URL = [URL copy];
	}
	return self;
}

- (void)dealloc
{
	[NSFileManager.defaultManager removeItemAtURL:_URL error:NULL];
}

@end

#pragma mark - Run state

/*! What a cancel needs to reach a run: the connection it went out on and its request identifier. */
@interface NFKPluginRunState : NSObject
@property (nonatomic, assign) BOOL cancelled;
@property (nonatomic, strong, nullable) NFKPluginConnection *connection;
@property (nonatomic, assign) NSInteger identifier;
@end

@implementation NFKPluginRunState

- (void)cancel
{
	NFKPluginConnection *connection = nil;
	NSInteger identifier = 0;
	@synchronized (self) {
		self.cancelled = YES;
		connection = self.connection;
		identifier = self.identifier;
	}
	[self sendCancelOnConnection:connection identifier:identifier];
}

- (void)startedOnConnection:(NFKPluginConnection *)connection identifier:(NSInteger)identifier
{
	BOOL cancelled = NO;
	@synchronized (self) {
		self.connection = connection;
		self.identifier = identifier;
		cancelled = self.cancelled;
	}
	if (cancelled) {
		[self sendCancelOnConnection:connection identifier:identifier];
	}
}

- (BOOL)isCancelled
{
	@synchronized (self) {
		return self.cancelled;
	}
}

- (void)sendCancelOnConnection:(nullable NFKPluginConnection *)connection identifier:(NSInteger)identifier
{
	if (connection != nil && identifier != 0) {
		[connection sendMessage:@{ @"type": NFKPluginMessageCancel, @"id": @(identifier) } error:NULL];
	}
}

@end

#pragma mark - Backend

@interface NFKPluginBackend ()
@property (nonatomic, readwrite, copy) NSString *pluginIdentifier;
@property (nonatomic, readwrite, copy) NSString *modelIdentifier;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *options;
@property (nonatomic, readwrite, copy) NFKModelDescriptor *descriptor;
@property (nonatomic, readwrite) NFKPluginIsolation isolation;
@property (nonatomic, readwrite, strong, nullable) NFKPluginProcess *process;
@property (nonatomic, strong, nullable) id<NFKInferenceBackend> pluginBackend;
@property (nonatomic, strong) NSLock *buildLock;
@property (atomic, readwrite) NSUInteger builtGeneration;
@property (atomic, strong, nullable) id handle;
@property (atomic, strong, nullable) NFKPluginConnection *builtConnection;
@property (atomic, assign) BOOL loaded;
@property (atomic, assign) NSInteger residentBytes;
@property (atomic, assign) NSInteger peakBytes;
@property (atomic, copy) NSDictionary<NSString *, id> *builtModelInfo;
@property (atomic, copy) NSDictionary<NSString *, id> *lastStatus;
@property (atomic, assign) NSTimeInterval lastStatusRequest;
@end

@implementation NFKPluginBackend

- (instancetype)initWithProcess:(NFKPluginProcess *)process
					 descriptor:(NFKModelDescriptor *)descriptor
						options:(NSDictionary<NSString *, id> *)options
{
	self = [self initWithPluginIdentifier:process.manifest.identifier descriptor:descriptor options:options];
	if (self != nil) {
		_process = process;
		_isolation = NFKPluginIsolationProcess;
	}
	return self;
}

- (instancetype)initWithPluginIdentifier:(NSString *)pluginIdentifier
							  descriptor:(NFKModelDescriptor *)descriptor
								 options:(NSDictionary<NSString *, id> *)options
						   pluginBackend:(id<NFKInferenceBackend>)pluginBackend
{
	self = [self initWithPluginIdentifier:pluginIdentifier descriptor:descriptor options:options];
	if (self != nil) {
		_pluginBackend = pluginBackend;
		_isolation = NFKPluginIsolationInProcess;
	}
	return self;
}

- (instancetype)initWithPluginIdentifier:(NSString *)pluginIdentifier
							  descriptor:(NFKModelDescriptor *)descriptor
								 options:(NSDictionary<NSString *, id> *)options
{
	self = [super init];
	if (self != nil) {
		_pluginIdentifier = [pluginIdentifier copy];
		_modelIdentifier = [descriptor.identifier copy];
		_descriptor = [descriptor copy];
		_options = [options copy];
		_buildLock = [[NSLock alloc] init];
		_builtModelInfo = @{};
		_lastStatus = @{};
	}
	return self;
}

- (void)dealloc
{
	if (_builtConnection != nil && _handle != nil) {
		[_builtConnection releaseBackendHandle:_handle];
	}
}

#pragma mark Identity and keys

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return [NSString stringWithFormat:@"plugin:%@/%@", self.pluginIdentifier, self.modelIdentifier];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:[self.descriptor.inputs valueForKey:@"key"]];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:[self.descriptor.outputs valueForKey:@"key"]];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithArray:[self.descriptor.requestOptions valueForKey:@"key"]];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	NSMutableDictionary<NSString *, id> *info = [NSMutableDictionary dictionary];
	info[NFKModelInfoArchitecture] = self.descriptor.family;
	NFKModelVariant *variant = self.descriptor.defaultVariant;
	info[NFKModelInfoResidentBytes] = variant.residentByteCount;
	info[NFKModelInfoPeakBytes] = variant.peakByteCount;
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		if ([pluginBackend respondsToSelector:@selector(modelInfo)]) {
			[info addEntriesFromDictionary:pluginBackend.modelInfo];
		}
		return info;
	}
	[info addEntriesFromDictionary:self.builtModelInfo];
	if (self.isLoaded && self.residentBytes > 0) {
		info[NFKModelInfoResidentBytes] = @(self.residentBytes);
	}
	if (self.peakBytes > 0) {
		info[NFKModelInfoPeakBytes] = @(self.peakBytes);
	}
	return info;
}

#pragma mark Residency

- (BOOL)isLoaded
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return NFKInferenceIsLoaded(pluginBackend);
	}
	NFKPluginConnection *connection = self.builtConnection;
	return self.loaded && connection != nil && !connection.closed;
}

- (NSInteger)residentByteCount
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return NFKInferenceResidentByteCount(pluginBackend);
	}
	return self.isLoaded ? self.residentBytes : 0;
}

- (NSInteger)peakByteCount
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return NFKInferencePeakByteCount(pluginBackend);
	}
	return self.peakBytes;
}

- (void)unload
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		NFKInferenceUnload(pluginBackend);
		return;
	}
	NFKPluginConnection *connection = self.process.readyConnection;
	id handle = connection != nil && self.builtConnection == connection ? self.handle : nil;
	if (handle == nil) {
		return;
	}
	[connection sendRequest:@{ @"type": NFKPluginMessageUnload, @"backend": handle } descriptors:nil progressHandler:nil
			   replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable error) {
	} error:NULL];
}

- (NSDictionary<NSString *, id> *)backendStatus
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return [pluginBackend respondsToSelector:@selector(backendStatus)] ? pluginBackend.backendStatus : @{};
	}
	[self refreshStatus];
	return self.lastStatus;
}

/*! Asks the running plugin for this backend's status, at most once per NFKPluginStatusRefreshInterval.
	Starts nothing and waits for nothing. */
- (void)refreshStatus
{
	NSTimeInterval now = NSProcessInfo.processInfo.systemUptime;
	NFKPluginConnection *connection = self.process.readyConnection;
	id handle = connection != nil && self.builtConnection == connection ? self.handle : nil;
	if (handle == nil || now - self.lastStatusRequest < NFKPluginStatusRefreshInterval) {
		return;
	}
	self.lastStatusRequest = now;
	__weak NFKPluginBackend *weakSelf = self;
	[connection sendRequest:@{ @"type": NFKPluginMessageStatus, @"backend": handle } descriptors:nil progressHandler:nil
			   replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable error) {
		if (reply != nil && NFKPluginErrorFromReply(reply) == nil) {
			[weakSelf applyStatus:reply[@"result"]];
		}
	} error:NULL];
}

/*! Takes loaded, residentBytes, peakBytes, and backendStatus from a prepare or status result. */
- (void)applyStatus:(nullable id)status
{
	if (![status isKindOfClass:NSDictionary.class]) {
		return;
	}
	if ([status[@"loaded"] isKindOfClass:NSNumber.class]) {
		self.loaded = [status[@"loaded"] boolValue];
	}
	if ([status[@"residentBytes"] isKindOfClass:NSNumber.class]) {
		self.residentBytes = [status[@"residentBytes"] integerValue];
	}
	[self recordPeakFrom:status];
	if ([status[@"backendStatus"] isKindOfClass:NSDictionary.class]) {
		self.lastStatus = status[@"backendStatus"];
	}
}

- (void)handleEvent:(NSDictionary<NSString *, id> *)event
{
	NSString *name = [event[@"event"] isKindOfClass:NSString.class] ? event[@"event"] : @"";
	if ([name isEqualToString:@"loaded"]) {
		self.loaded = YES;
		if ([event[@"residentBytes"] isKindOfClass:NSNumber.class]) {
			self.residentBytes = [event[@"residentBytes"] integerValue];
		}
		[self recordPeakFrom:event];
	} else if ([name isEqualToString:@"unloaded"]) {
		self.loaded = NO;
		self.residentBytes = 0;
		[self recordPeakFrom:event];
	}
}

/*! Raises the peak to a message's peakBytes and to the resident bytes now known. The peak is the most the
	backend has held, so a message that omits the field, or reports less after an unload, keeps it. */
- (void)recordPeakFrom:(NSDictionary<NSString *, id> *)message
{
	NSInteger reported = [message[@"peakBytes"] isKindOfClass:NSNumber.class] ? [message[@"peakBytes"] integerValue] : 0;
	self.peakBytes = MAX(self.peakBytes, MAX(reported, self.residentBytes));
}

#pragma mark Building

/*! The plugin's handle for this backend on connection, built there when the backend has none from that
	process. Runs in the request's turn. */
- (nullable id)handleOnConnection:(NFKPluginConnection *)connection error:(NSError **)error
{
	[self.buildLock lock];
	if (self.builtConnection == connection && self.handle != nil) {
		id handle = self.handle;
		[self.buildLock unlock];
		return handle;
	}
	id result = [self buildOnConnection:connection error:error];
	id handle = [result isKindOfClass:NSDictionary.class] ? result[@"backend"] : nil;
	if (result != nil && (handle == nil || handle == NSNull.null)) {
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"the plugin built %@ without naming a backend", self.modelIdentifier]);
		handle = nil;
	}
	if (handle == nil) {
		[self.buildLock unlock];
		return nil;
	}
	NFKPluginConnection *previous = self.builtConnection;
	id previousHandle = self.handle;
	if (previous != nil && previousHandle != nil) {
		[previous releaseBackendHandle:previousHandle];
	}
	self.handle = handle;
	self.builtConnection = connection;
	self.builtGeneration = connection.generation;
	self.loaded = [result[@"loaded"] isKindOfClass:NSNumber.class] && [result[@"loaded"] boolValue];
	self.residentBytes = self.loaded && [result[@"residentBytes"] isKindOfClass:NSNumber.class] ? [result[@"residentBytes"] integerValue] : 0;
	[self recordPeakFrom:result];
	self.builtModelInfo = [result[@"modelInfo"] isKindOfClass:NSDictionary.class] ? result[@"modelInfo"] : @{};
	[connection registerBackend:self handle:handle];
	[self.buildLock unlock];
	return handle;
}

/*! Sends build and waits for its result up to the process's buildTimeout. A build that answers after the
	timeout has its backend released, since no request uses it. */
- (nullable id)buildOnConnection:(NFKPluginConnection *)connection error:(NSError **)error
{
	NSDictionary *message = @{ @"type": NFKPluginMessageBuild, @"model": self.modelIdentifier, @"options": self.options };
	NSObject *lock = [[NSObject alloc] init];
	__block BOOL abandoned = NO;
	__block id result = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t answered = dispatch_semaphore_create(0);
	__weak NFKPluginConnection *weakConnection = connection;
	NSInteger identifier = [connection sendRequest:message descriptors:nil progressHandler:nil
									  replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
		NSError *replyFailure = reply != nil ? NFKPluginErrorFromReply(reply) : replyError;
		id replyResult = replyFailure == nil ? (reply[@"result"] ?: NSNull.null) : nil;
		BOOL late = NO;
		@synchronized (lock) {
			late = abandoned;
			result = replyResult;
			failure = replyFailure;
		}
		id handle = [replyResult isKindOfClass:NSDictionary.class] ? replyResult[@"backend"] : nil;
		if (late && handle != nil && handle != NSNull.null) {
			[weakConnection sendRequest:@{ @"type": NFKPluginMessageRelease, @"backend": handle } descriptors:nil progressHandler:nil
						   replyHandler:^(NSDictionary<NSString *, id> * _Nullable releaseReply, NSArray<NFKPluginDescriptor *> *releaseDescriptors, NSError * _Nullable releaseError) {
			} error:NULL];
		}
		dispatch_semaphore_signal(answered);
	} error:error];
	if (identifier == 0) {
		return nil;
	}
	NSTimeInterval timeout = self.process.buildTimeout;
	dispatch_time_t deadline = timeout > 0 ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)) : DISPATCH_TIME_FOREVER;
	BOOL timedOut = dispatch_semaphore_wait(answered, deadline) != 0;
	@synchronized (lock) {
		// A reply that raced the deadline is kept.
		if (timedOut && result == nil && failure == nil) {
			abandoned = YES;
			[connection detachRequest:identifier];
			NFKPluginSetError(error, NFKPluginErrorTimedOut,
							  [NSString stringWithFormat:@"plugin %@ did not build model %@ within %g seconds", self.pluginIdentifier,
							   self.modelIdentifier, timeout]);
			return nil;
		}
		if (result == nil && error != NULL) {
			*error = failure;
		}
		return result;
	}
}

- (BOOL)prepareWithError:(NSError **)error
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return NFKInferencePrepare(pluginBackend, error);
	}
	NSObject *lock = [[NSObject alloc] init];
	__block BOOL answered = NO;
	__block NSError *failure = nil;
	__block NSInteger identifier = 0;
	__block NFKPluginConnection *sentOn = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	__weak NFKPluginBackend *weakSelf = self;
	NSError *sendError = nil;
	BOOL sent = [self.process performWithReservation:nil block:^BOOL(NFKPluginConnection *connection, NSError **blockError) {
		id handle = [self handleOnConnection:connection error:blockError];
		if (handle == nil) {
			return NO;
		}
		NSInteger sentIdentifier = [connection sendRequest:@{ @"type": NFKPluginMessagePrepare, @"backend": handle } descriptors:nil progressHandler:nil
											  replyHandler:^(NSDictionary<NSString *, id> * _Nullable message, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
			NSError *replyFailure = message != nil ? NFKPluginErrorFromReply(message) : replyError;
			// A late answer still reports the load.
			if (replyFailure == nil) {
				[weakSelf applyStatus:message[@"result"]];
			}
			@synchronized (lock) {
				answered = YES;
				failure = replyFailure;
			}
			dispatch_semaphore_signal(done);
		} error:blockError];
		if (sentIdentifier == 0) {
			return NO;
		}
		@synchronized (lock) {
			identifier = sentIdentifier;
			sentOn = connection;
		}
		return YES;
	} error:&sendError];
	if (!sent) {
		if (error != NULL) {
			*error = sendError;
		}
		return NO;
	}
	NSTimeInterval timeout = self.process.buildTimeout;
	dispatch_time_t deadline = timeout > 0 ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)) : DISPATCH_TIME_FOREVER;
	BOOL timedOut = dispatch_semaphore_wait(done, deadline) != 0;
	@synchronized (lock) {
		if (timedOut && !answered) {
			[sentOn detachRequest:identifier];
			return NFKPluginSetError(error, NFKPluginErrorTimedOut,
									 [NSString stringWithFormat:@"plugin %@ did not prepare model %@ within %g seconds", self.pluginIdentifier,
									  self.modelIdentifier, timeout]);
		}
		if (failure != nil) {
			if (error != NULL) {
				*error = failure;
			}
			return NO;
		}
	}
	return YES;
}

#pragma mark Running

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return [pluginBackend runInferenceForRequest:request error:error];
	}
	__block NFKInferenceResult *result = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	NSError *sendError = nil;
	BOOL (^send)(NFKPluginConnection *, NSError **) = [self runBlockForRequest:request streaming:NO job:nil state:[[NFKPluginRunState alloc] init]
																	 completion:^(NFKInferenceResult * _Nullable runResult, NSError * _Nullable runError) {
		result = runResult;
		failure = runError;
		dispatch_semaphore_signal(done);
	} error:&sendError];
	if (send != nil && [self.process performWithReservation:nil block:send error:&sendError]) {
		dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	} else {
		failure = sendError;
	}
	if (result == nil && error != NULL) {
		*error = failure;
	}
	return result;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	id<NFKInferenceBackend> pluginBackend = self.pluginBackend;
	if (pluginBackend != nil) {
		return NFKInferenceSubmit(pluginBackend, request, NULL);
	}
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NFKPluginRunState *state = [[NFKPluginRunState alloc] init];
	job.cancellationHandler = ^{
		[state cancel];
	};
	[job reportProgress:-1.0];
	void (^completion)(NFKInferenceResult *, NSError *) = ^(NFKInferenceResult * _Nullable result, NSError * _Nullable runError) {
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:runError];
		}
	};
	NSError *failure = nil;
	BOOL (^send)(NFKPluginConnection *, NSError **) = [self runBlockForRequest:request streaming:YES job:job state:state
																	 completion:completion error:&failure];
	if (send == nil) {
		completion(nil, failure);
		return job;
	}
	[self.process performAsynchronouslyWithBlock:send completion:^(BOOL performed, NSError * _Nullable error) {
		if (!performed) {
			completion(nil, error);
		}
	}];
	return job;
}

/*! The block that sends a run in its turn, or nil with an error when the run's directory cannot be made.
	completion is called once with the run's result or error, unless the block fails to send, which the
	caller reports. The request's large values travel as attachments, and the plugin writes its result
	files into a run directory that lives as long as the results decoded from it. */
- (nullable BOOL (^)(NFKPluginConnection *, NSError **))runBlockForRequest:(NFKInferenceRequest *)request
																  streaming:(BOOL)streaming
																		job:(nullable NFKInferenceJob *)job
																	  state:(NFKPluginRunState *)state
																 completion:(void (^)(NFKInferenceResult * _Nullable result, NSError * _Nullable error))completion
																	  error:(NSError **)error
{
	NFKPluginRunDirectory *directory = [NFKPluginRunDirectory directoryWithError:error];
	if (directory == nil) {
		return nil;
	}
	return ^BOOL(NFKPluginConnection *connection, NSError **blockError) {
		if (state.isCancelled) {
			if (blockError != NULL) {
				*blockError = NFKPluginCancellationError();
			}
			return NO;
		}
		id handle = [self handleOnConnection:connection error:blockError];
		if (handle == nil) {
			return NO;
		}
		NFKPluginAttachmentWriter *writer NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginAttachmentWriter alloc] init];
		NSString *prefix = self.process.sharedMemoryPrefix;
		NSArray *accepted = [connection.helloReply[@"capabilities"] isKindOfClass:NSArray.class] ? connection.helloReply[@"capabilities"] : @[];
		writer.namePrefix = prefix;
		writer.passesFilesAsDescriptors = prefix != nil && [accepted containsObject:NFKPluginCapabilityFileDescriptors];
		NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:directory.URL];
		encoder.attachmentSink = writer;
		encoder.inlineByteLimit = connection.inlineByteLimit;
		encoder.encodesValueObjectsAsJSON = YES;
		id encoded = [encoder JSONObjectForRequest:request error:blockError];
		if (encoded == nil) {
			return NO;
		}
		NSDictionary *message = @{ @"type": NFKPluginMessageRun, @"backend": handle, @"request": encoded, @"stream": @(streaming),
								   @"outputDirectory": directory.URL.path };
		NFKPluginProgressHandler progress = nil;
		if (job != nil) {
			progress = ^(NSDictionary<NSString *, id> *update, NSArray<NFKPluginDescriptor *> *descriptors) {
				[self relayProgress:update descriptors:descriptors directory:directory job:job];
			};
		}
		NSInteger identifier = [connection sendRequest:message descriptors:writer.descriptors progressHandler:progress
										  replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
			NSError *runError = reply != nil ? NFKPluginErrorFromReply(reply) : replyError;
			NFKInferenceResult *result = nil;
			if (runError == nil) {
				result = [self resultFromJSONObject:reply[@"result"] descriptors:descriptors directory:directory error:&runError];
			}
			completion(result, result != nil ? nil : runError);
		} error:blockError];
		if (identifier == 0) {
			return NO;
		}
		[state startedOnConnection:connection identifier:identifier];
		return YES;
	};
}

- (void)relayProgress:(NSDictionary<NSString *, id> *)update
		  descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
			directory:(NFKPluginRunDirectory *)directory
				  job:(NFKInferenceJob *)job
{
	double progress = [update[@"progress"] isKindOfClass:NSNumber.class] ? [update[@"progress"] doubleValue] : -1.0;
	id partialObject = update[@"partial"];
	NFKInferenceResult *partial = nil;
	if (partialObject != nil && partialObject != NSNull.null) {
		partial = [self resultFromJSONObject:partialObject descriptors:descriptors directory:directory error:NULL];
	}
	[job reportProgress:progress partialResult:partial];
}

/*! Decodes a result or a partial with its frame's attachments. A file it names must lie in the run
	directory. */
- (nullable NFKInferenceResult *)resultFromJSONObject:(nullable id)object
										  descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
											directory:(NFKPluginRunDirectory *)directory
												error:(NSError **)error
{
	NFKPluginAttachmentReader *reader = [[NFKPluginAttachmentReader alloc] initWithDescriptors:descriptors];
	reader.copiesFileRegions = YES;
	reader.temporaryDirectoryURL = directory.URL;
	reader.allowedFileDirectoryURL = directory.URL;
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:directory.URL];
	decoder.attachmentSource = reader;
	NFKInferenceResult *result = [decoder resultFromJSONObject:object error:error];
	if (result != nil) {
		objc_setAssociatedObject(result, NFKPluginRunDirectoryKey, directory, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	}
	return result;
}

@end

#endif
