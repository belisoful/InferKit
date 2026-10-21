//
//  NFKPluginServiceTests.m
//  InferKitTests
//
//  The plugin side of the protocol, driven in process over a socket pair: the hand-off, hello, every
//  request type, progress with partial results, cancellation, attachments both ways, result files, tool
//  calls, unknown messages, shutdown, the load-on-hello mode the plugin host runs, and the plugin host
//  program itself.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"
#import "../../Sources/InferKit/NFKPluginChannel.h"
#import "../../Sources/InferKit/NFKPluginWire.h"
#import "../../Sources/InferKit/NFKPluginServicePrivate.h"
#import "../../Sources/InferKit/NFKInferenceWireCoding.h"
#import "../../Sources/InferKit/NFKSubprocessPrivate.h"
#import <mach/mach.h>
#import <unistd.h>

/*! The number of threads this process runs now. */
static NSInteger NFKServiceTestThreadCount(void)
{
	thread_act_array_t threads = NULL;
	mach_msg_type_number_t count = 0;
	if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) {
		return -1;
	}
	for (mach_msg_type_number_t index = 0; index < count; index++) {
		mach_port_deallocate(mach_task_self(), threads[index]);
	}
	vm_deallocate(mach_task_self(), (vm_address_t)threads, sizeof(thread_act_t) * count);
	return (NSInteger)count;
}

#pragma mark - Test plugin

/*! Echoes its prompt in upper case, returns an input "data" as it came, and with the parameter "file"
	writes its prompt to a file of its own and returns that file. An input "region" comes back as it came,
	with the class it arrived as under "regionClass". */
@interface NFKServiceTestEchoBackend : NSObject <NFKInferenceBackend>
@property (atomic, assign) BOOL loaded;
@end

@implementation NFKServiceTestEchoBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"service-test-echo";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObjects:NFKInputPrompt, @"data", nil];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObjects:NFKOutputText, @"data", @"file", nil];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoArchitecture: @"echo", @"unencodable": [NSDate date] };
}

- (NSDictionary<NSString *, id> *)backendStatus
{
	return @{ @"runs": @0 };
}

- (BOOL)prepareWithError:(NSError **)error
{
	self.loaded = YES;
	return YES;
}

- (BOOL)isLoaded
{
	return self.loaded;
}

- (NSInteger)residentByteCount
{
	return self.loaded ? 1024 : 0;
}

- (void)unload
{
	self.loaded = NO;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	NSString *prompt = request.inputs[NFKInputPrompt];
	id region = request.inputs[@"region"];
	if (region != nil) {
		return [NFKInferenceResult resultWithOutputs:@{ @"region": region, @"regionClass": NSStringFromClass([region class]) }];
	}
	if (prompt == nil && request.inputs[@"data"] == nil) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput
									 userInfo:@{ NSLocalizedDescriptionKey: @"the echo needs a prompt" }];
		}
		return nil;
	}
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	outputs[NFKOutputText] = prompt.uppercaseString;
	outputs[@"data"] = request.inputs[@"data"];
	if ([request.parameters[@"file"] boolValue]) {
		NSURL *file = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					   [NSString stringWithFormat:@"service-test-%@.txt", NSUUID.UUID.UUIDString]];
		[prompt writeToURL:file atomically:YES encoding:NSUTF8StringEncoding error:NULL];
		outputs[@"file"] = file;
	}
	self.loaded = YES;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

@end

/*! Counts "steps" steps of 10 ms as a job, reporting progress with a partial {"step": n} after each, and
	ends when the job is cancelled. */
@interface NFKServiceTestSlowBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKServiceTestSlowBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"service-test-slow";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{}];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSInteger steps = [request.parameters[@"steps"] integerValue];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		for (NSInteger step = 1; step <= steps; step++) {
			[NSThread sleepForTimeInterval:0.01];
			if (job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			[job reportProgress:(double)step / (double)steps partialResult:[NFKInferenceResult resultWithOutputs:@{ @"step": @(step) }]];
		}
		[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ @"steps": @(steps) }]];
	});
	return job;
}

@end

/*! Sleeps "milliseconds" synchronously and answers its prompt. */
@interface NFKServiceTestNapBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKServiceTestNapBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"service-test-nap";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	[NSThread sleepForTimeInterval:[request.parameters[@"milliseconds"] doubleValue] / 1000.0];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: request.inputs[NFKInputPrompt] ?: @"" }];
}

@end

@interface NFKServiceTestPlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@end

@implementation NFKServiceTestPlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	self = [super init];
	if (self != nil) {
		_context = context;
	}
	return self;
}

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return @[ [NFKModelDescriptor descriptorWithDictionary:@{ @"id": @"echo", @"family": @"echo", @"engine": @"plugin" } error:NULL],
			  [NFKModelDescriptor descriptorWithDictionary:@{ @"id": @"slow", @"family": @"slow", @"engine": @"plugin" } error:NULL] ];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if ([modelIdentifier isEqualToString:@"echo"]) {
		return [[NFKServiceTestEchoBackend alloc] init];
	}
	if ([modelIdentifier isEqualToString:@"slow"]) {
		return [[NFKServiceTestSlowBackend alloc] init];
	}
	if ([modelIdentifier isEqualToString:@"nap"]) {
		return [[NFKServiceTestNapBackend alloc] init];
	}
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel userInfo:nil];
	}
	return nil;
}

- (NSArray<NFKInferenceTool *> *)tools
{
	return @[ [NFKInferenceTool toolWithName:@"greet" description:@"Greets a name."
								  parameters:@{ @"type": @"object", @"properties": @{ @"name": @{ @"type": @"string" } }, @"required": @[ @"name" ] }
									 handler:^NFKToolResult *(NSDictionary<NSString *, id> *arguments, NSError **error) {
		return [NFKToolResult resultWithText:[NSString stringWithFormat:@"Hello, %@", arguments[@"name"]]];
	}] ];
}

- (NSArray<NFKModelOption *> *)pluginOptions
{
	return @[ [NFKModelOption optionWithKey:@"greeting" kind:NFKModelOptionKindText] ];
}

@end

#pragma mark - Tests

@interface NFKPluginServiceTests : XCTestCase
@property (nonatomic, strong, nullable) NFKPluginChannel *channel;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *messages;
@property (nonatomic, strong) NSMutableArray<NSArray<NFKPluginDescriptor *> *> *messageDescriptors;
@property (nonatomic, assign) BOOL channelClosed;
@property (nonatomic, strong, nullable) NSError *closeError;
@property (nonatomic, assign) NSInteger lastIdentifier;
@property (nonatomic, strong) dispatch_semaphore_t served;
@property (nonatomic, assign) BOOL servedOrderly;
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginServiceTests

- (void)setUp
{
	[super setUp];
	self.messages = [NSMutableArray array];
	self.messageDescriptors = [NSMutableArray array];
	self.temporaryURLs = [NSMutableArray array];
	self.served = dispatch_semaphore_create(0);
}

- (void)tearDown
{
	[self.channel close];
	self.channel = nil;
	for (NSURL *url in self.temporaryURLs) {
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}
	[super tearDown];
}

#pragma mark The public interface

/*! The repository's root, from this file's location. */
- (NSURL *)sourceRoot
{
	return [[NSURL fileURLWithPath:@__FILE__] URLByAppendingPathComponent:@"../../.."].URLByStandardizingPath;
}

// The table the plugin host keeps alive names every constant and function the public headers declare, so
// a public symbol added without regenerating it fails here rather than in a plugin that cannot bind.
- (void)testThePublicInterfaceTableIsCurrent
{
	NSURL *script = [[self sourceRoot] URLByAppendingPathComponent:@"Tools/plugin-host/public-interface.py"];
	NSError *error = nil;
	NSString *output = [NFKPluginFixtures runProgram:@"/usr/bin/python3" arguments:@[ @"-I", script.path, @"--check" ] error:&error];
	XCTAssertNotNil(output, @"%@", error);
}

// Every symbol the table names resolves in this image.
- (void)testRetainingThePublicInterfaceKeepsEverySymbol
{
	NSURL *table = [[self sourceRoot] URLByAppendingPathComponent:@"Sources/InferKit/NFKPluginPublicInterface.m"];
	NSString *text = [NSString stringWithContentsOfURL:table encoding:NSUTF8StringEncoding error:NULL];
	NSUInteger listed = [text componentsSeparatedByString:@"(const void *)&"].count - 1;
	XCTAssertGreaterThan(listed, (NSUInteger)300);
	XCTAssertEqual(NFKPluginRetainPublicInterface(), listed);
}

#pragma mark Harness

- (NFKPluginManifest *)manifest
{
	return [self manifestWithConcurrentRuns:2];
}

- (NFKPluginManifest *)manifestWithConcurrentRuns:(NSInteger)runs
{
	NSDictionary *object = @{ @"schemaVersion": @1, @"kind": @"inferkit.plugin", @"id": @"com.example.service-test", @"version": @"1.0.0",
							  @"compatibility": @{ @"inferkitAPI": @"1.0" }, @"entry": @{ @"kind": @"objc" },
							  @"process": @{ @"maximumConcurrentRuns": @(runs) },
							  @"models": @[ @{ @"id": @"echo", @"family": @"echo" }, @{ @"id": @"slow", @"family": @"slow" },
											@{ @"id": @"nap", @"family": @"nap" } ],
							  @"tools": @[ @{ @"name": @"greet" } ] };
	NSError *error = nil;
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithJSONData:[NSJSONSerialization dataWithJSONObject:object options:0 error:NULL] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	return manifest;
}

- (NFKPluginService *)testService
{
	return [[NFKPluginService alloc] initWithPlugin:[[NFKServiceTestPlugin alloc] initWithContext:(id<NFKPluginContext>)NSNull.null]
										   manifest:[self manifest]];
}

/*! Serves service on one end of a socket pair on a thread of its own and accepts the hand-off on the
	other, as a host does. */
- (void)startService:(NFKPluginService *)service
{
	int pluginEnd = -1;
	int hostEnd = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&pluginEnd second:&hostEnd error:&error], @"%@", error);
	[NSThread detachNewThreadWithBlock:^{
		NSError *serveError = nil;
		BOOL orderly = [service runWithChannelDescriptor:pluginEnd error:&serveError];
		@synchronized (self) {
			self.servedOrderly = orderly;
		}
		dispatch_semaphore_signal(self.served);
	}];
	NFKPluginChannel *channel = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:5 error:&error];
	XCTAssertNotNil(channel, @"%@", error);
	[self attachToChannel:channel];
}

- (void)attachToChannel:(NFKPluginChannel *)channel
{
	channel.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
		@synchronized (self) {
			[self.messages addObject:message];
			[self.messageDescriptors addObject:descriptors];
		}
	};
	channel.closeHandler = ^(NSError *error) {
		@synchronized (self) {
			self.channelClosed = YES;
			self.closeError = error;
		}
	};
	self.channel = channel;
	[channel resume];
}

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow <= 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.005];
	}
	return YES;
}

- (NSNumber *)send:(NSDictionary *)message descriptors:(nullable NSArray<NSNumber *> *)descriptors
{
	NSMutableDictionary *withIdentifier = [message mutableCopy];
	@synchronized (self) {
		self.lastIdentifier += 1;
		withIdentifier[@"id"] = @(self.lastIdentifier);
	}
	NSError *error = nil;
	BOOL sent = [self.channel sendMessage:withIdentifier descriptors:descriptors error:&error];
	@synchronized (self) {
		XCTAssertTrue(sent, @"%@; the channel closed with %@", error, self.closeError);
	}
	return withIdentifier[@"id"];
}

/*! The reply to identifier, waiting up to ten seconds. */
- (nullable NSDictionary *)replyTo:(NSNumber *)identifier descriptors:(NSArray<NFKPluginDescriptor *> * _Nullable * _Nullable)outDescriptors
{
	__block NSDictionary *reply = nil;
	__block NSArray<NFKPluginDescriptor *> *replyDescriptors = nil;
	[self waitUntil:^BOOL {
		@synchronized (self) {
			[self.messages enumerateObjectsUsingBlock:^(NSDictionary *message, NSUInteger index, BOOL *stop) {
				if ([message[@"type"] isEqual:@"reply"] && [message[@"id"] isEqual:identifier]) {
					reply = message;
					replyDescriptors = self.messageDescriptors[index];
					*stop = YES;
				}
			}];
		}
		return reply != nil;
	} timeout:10];
	@synchronized (self) {
		XCTAssertNotNil(reply, @"a reply to %@; closed: %d with %@; received: %@", identifier, self.channelClosed, self.closeError, self.messages);
	}
	if (outDescriptors != NULL && reply != nil) {
		*outDescriptors = replyDescriptors;
	}
	return reply;
}

- (nullable NSDictionary *)request:(NSDictionary *)message
{
	return [self replyTo:[self send:message descriptors:nil] descriptors:NULL];
}

- (NSArray<NSDictionary *> *)messagesOfType:(NSString *)type
{
	@synchronized (self) {
		return [self.messages filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"type == %@", type]];
	}
}

- (NSDictionary *)hello
{
	return @{ @"type": @"hello", @"protocol": @{ @"major": @1, @"minor": @4 }, @"inferkitAPI": @"1.0", @"pluginABI": @"1.0",
			  @"capabilities": @[ @"mach-surfaces" ], @"options": @{ @"greeting": @"Hi" }, @"limits": @{ @"inlineBytes": @65536 } };
}

- (NSNumber *)startAndBuild:(NSString *)model
{
	[self startService:[self testService]];
	XCTAssertNotNil([self request:[self hello]][@"result"]);
	NSDictionary *built = [self request:@{ @"type": @"build", @"model": model, @"options": @{} }];
	XCTAssertNotNil(built[@"result"], @"%@", built);
	return built[@"result"][@"backend"];
}

- (void)assertReply:(NSDictionary *)reply failsWithDomain:(NSString *)domain code:(NSInteger)code
{
	NSError *error = [NFKInferenceWireCoding errorFromJSONObject:@{ @"error": reply[@"error"] ?: @{} }];
	XCTAssertEqualObjects(error.domain, domain, @"%@", reply);
	XCTAssertEqual(error.code, code, @"%@", reply);
}

- (nullable NFKInferenceResult *)resultOfReply:(NSDictionary *)reply descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	XCTAssertNotNil(reply[@"result"], @"the run failed: %@", reply[@"error"]);
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSource = [[NFKPluginAttachmentReader alloc] initWithDescriptors:descriptors];
	NSError *error = nil;
	NFKInferenceResult *result = reply[@"result"] != nil ? [coder resultFromJSONObject:reply[@"result"] error:&error] : nil;
	XCTAssertNotNil(result, @"%@", error);
	return result;
}

#pragma mark Handshake

- (void)testHelloAnswersWithThePluginAndTheLowerMinorVersion
{
	[self startService:[self testService]];
	NSDictionary *reply = [self request:[self hello]];
	NSDictionary *result = reply[@"result"];
	XCTAssertEqualObjects(result[@"protocol"], (@{ @"major": @1, @"minor": @0 }));
	XCTAssertEqualObjects(result[@"plugin"], (@{ @"id": @"com.example.service-test", @"version": @"1.0.0" }));
	XCTAssertEqualObjects(result[@"inferkitAPI"], @"1.0");
	XCTAssertEqualObjects(result[@"capabilities"], @[ @"attachments" ], @"mach-surfaces stays reserved");
	XCTAssertEqualObjects(result[@"pid"], @(getpid()));
	[self assertReply:[self request:[self hello]] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolViolation];
}

- (void)testHelloOfAnotherMajorVersionIsRefused
{
	[self startService:[self testService]];
	NSMutableDictionary *hello = [[self hello] mutableCopy];
	hello[@"protocol"] = @{ @"major": @2, @"minor": @0 };
	[self assertReply:[self request:hello] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolUnsupported];
}

- (void)testARequestBeforeHelloIsRefused
{
	[self startService:[self testService]];
	[self assertReply:[self request:@{ @"type": @"describe" }] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolViolation];
	XCTAssertNotNil(([self request:@{ @"type": @"ping" }][@"result"]), @"ping answers before hello");
}

- (void)testDescribeListsModelsToolsAndOptions
{
	[self startService:[self testService]];
	[self request:[self hello]];
	NSDictionary *result = [self request:@{ @"type": @"describe" }][@"result"];
	XCTAssertEqualObjects([result[@"models"] valueForKey:@"id"], (@[ @"echo", @"slow" ]));
	XCTAssertEqualObjects([result[@"tools"] valueForKey:@"name"], @[ @"greet" ]);
	XCTAssertEqualObjects([result[@"options"] valueForKey:@"key"], @[ @"greeting" ]);
}

#pragma mark Backends

- (void)testBuildPrepareRunStatusUnloadAndRelease
{
	NSNumber *handle = [self startAndBuild:@"echo"];
	XCTAssertTrue([handle isKindOfClass:NSNumber.class], @"a handle is an integer");
	NSDictionary *built = [self messagesOfType:@"reply"].lastObject[@"result"];
	XCTAssertEqualObjects(built[@"inputs"], (@[ @"data", NFKInputPrompt ]));
	XCTAssertEqualObjects(built[@"modelInfo"], @{ NFKModelInfoArchitecture: @"echo" }, @"a value JSON cannot carry is left out");
	XCTAssertEqualObjects(built[@"loaded"], @NO);

	NSDictionary *prepared = [self request:@{ @"type": @"prepare", @"backend": handle }][@"result"];
	XCTAssertEqualObjects(prepared, (@{ @"loaded": @YES, @"residentBytes": @1024, @"peakBytes": @0 }));
	NSDictionary *loaded = [self messagesOfType:@"event"].lastObject;
	XCTAssertEqualObjects(loaded, (@{ @"type": @"event", @"event": @"loaded", @"backend": handle, @"residentBytes": @1024, @"peakBytes": @0 }));

	NSDictionary *ran = [self request:@{ @"type": @"run", @"backend": handle,
										 @"request": @{ @"inputs": @{ NFKInputPrompt: @"hello" }, @"parameters": @{} } }];
	XCTAssertEqualObjects(ran[@"result"][@"outputs"][NFKOutputText], @"HELLO");

	NSDictionary *status = [self request:@{ @"type": @"status", @"backend": handle }][@"result"];
	XCTAssertEqualObjects(status[@"backendStatus"], @{ @"runs": @0 });
	XCTAssertEqualObjects(status[@"loaded"], @YES);

	XCTAssertNotNil(([self request:@{ @"type": @"unload", @"backend": handle }][@"result"]));
	XCTAssertEqualObjects([self messagesOfType:@"event"].lastObject[@"event"], @"unloaded");

	XCTAssertNotNil(([self request:@{ @"type": @"release", @"backend": handle }][@"result"]));
	[self assertReply:[self request:@{ @"type": @"status", @"backend": handle }] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolViolation];
}

- (void)testAnUnknownModelIsRefusedAtBuild
{
	[self startService:[self testService]];
	[self request:[self hello]];
	[self assertReply:[self request:@{ @"type": @"build", @"model": @"absent" }] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel];
}

#pragma mark Runs

- (void)testARunsErrorCrossesWithItsDomainAndCode
{
	NSNumber *handle = [self startAndBuild:@"echo"];
	NSDictionary *reply = [self request:@{ @"type": @"run", @"backend": handle, @"request": @{ @"inputs": @{}, @"parameters": @{} } }];
	[self assertReply:reply failsWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput];
}

- (void)testARunSendsProgressWithPartialResultsBeforeItsResult
{
	NSNumber *handle = [self startAndBuild:@"slow"];
	NSNumber *identifier = [self send:@{ @"type": @"run", @"backend": handle,
										 @"request": @{ @"inputs": @{}, @"parameters": @{ @"steps": @3 } } } descriptors:nil];
	NSDictionary *reply = [self replyTo:identifier descriptors:NULL];
	XCTAssertEqualObjects(reply[@"result"][@"outputs"][@"steps"], @3);
	NSArray<NSDictionary *> *progress = [self messagesOfType:@"progress"];
	XCTAssertEqual(progress.count, 3u);
	XCTAssertEqualObjects(progress.firstObject[@"id"], identifier);
	XCTAssertEqualObjects(progress.lastObject[@"progress"], @1);
	XCTAssertEqualObjects(progress.lastObject[@"partial"][@"outputs"][@"step"], @3, @"a partial rides every progress message");
	@synchronized (self) {
		XCTAssertTrue([self.messages indexOfObject:progress.lastObject] < [self.messages indexOfObject:reply]);
	}
}

- (void)testCancelEndsARunWithCancelled
{
	NSNumber *handle = [self startAndBuild:@"slow"];
	NSNumber *identifier = [self send:@{ @"type": @"run", @"backend": handle,
										 @"request": @{ @"inputs": @{}, @"parameters": @{ @"steps": @100000 } } } descriptors:nil];
	XCTAssertTrue([self waitUntil:^BOOL { return [self messagesOfType:@"progress"].count > 0; } timeout:10]);
	XCTAssertTrue(([self.channel sendMessage:@{ @"type": @"cancel", @"id": identifier } error:NULL]));
	NSDictionary *reply = [self replyTo:identifier descriptors:NULL];
	XCTAssertEqualObjects(reply, (@{ @"type": @"reply", @"id": identifier, @"cancelled": @YES }));
	[NSThread sleepForTimeInterval:0.1];
	NSArray *replies = [[self messagesOfType:@"reply"] filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"id == %@", identifier]];
	XCTAssertEqual(replies.count, 1u, @"cancel gets no reply of its own");
}

- (void)testLargeValuesCrossAsAttachmentsBothWays
{
	NSNumber *handle = [self startAndBuild:@"echo"];
	NSMutableData *bytes = [NSMutableData dataWithLength:200 * 1024];
	for (NSUInteger index = 0; index < bytes.length; index++) {
		((uint8_t *)bytes.mutableBytes)[index] = (uint8_t)(index * 7);
	}
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSink = writer;
	NSError *error = nil;
	id request = [coder JSONObjectForRequest:[NFKInferenceRequest requestWithInputs:@{ @"data": bytes }] error:&error];
	XCTAssertEqualObjects(request[@"inputs"][@"data"][@"$nfk"], @"buffer", @"%@", error);
	NSNumber *identifier = [self send:@{ @"type": @"run", @"backend": handle, @"request": request } descriptors:writer.descriptors];

	NSArray<NFKPluginDescriptor *> *descriptors = nil;
	NSDictionary *reply = [self replyTo:identifier descriptors:&descriptors];
	XCTAssertEqualObjects(reply[@"result"][@"outputs"][@"data"][@"$nfk"], @"buffer", @"an attached input is answered attached: %@", reply);
	XCTAssertEqual(descriptors.count, 1u);
	XCTAssertEqualObjects([self resultOfReply:reply descriptors:descriptors].outputs[@"data"], bytes);

	NSData *small = [@"small" dataUsingEncoding:NSUTF8StringEncoding];
	identifier = [self send:@{ @"type": @"run", @"backend": handle,
							   @"request": [coder JSONObjectForRequest:[NFKInferenceRequest requestWithInputs:@{ @"data": small }] error:NULL] }
				descriptors:nil];
	reply = [self replyTo:identifier descriptors:&descriptors];
	XCTAssertEqualObjects(reply[@"result"][@"outputs"][@"data"][@"$nfk"], @"data", @"an inline input is answered inline");
	XCTAssertEqual(descriptors.count, 0u);
}

- (void)testAValueClassCrossesAsJSONBothWays
{
	NSNumber *handle = [self startAndBuild:@"echo"];
	NFKDetection *detection = [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.875
													  boundingBox:CGRectMake(0.25, 0.25, 0.5, 0.5) quadrilateral:nil];
	NFKInferenceWireCoding *host = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	host.attachmentSink = [[NFKPluginAttachmentWriter alloc] init];
	host.encodesValueObjectsAsJSON = YES;
	id request = [host JSONObjectForRequest:[NFKInferenceRequest requestWithInputs:@{ @"region": detection }] error:NULL];
	XCTAssertEqualObjects(request[@"inputs"][@"region"][@"$nfk"], @"detection");

	NSDictionary *reply = [self request:@{ @"type": @"run", @"backend": handle, @"request": request }];
	NSDictionary *outputs = reply[@"result"][@"outputs"];
	XCTAssertEqualObjects(outputs[@"regionClass"], @"NFKDetection", @"the plugin received a detection");
	NSDictionary *expected = @{ @"$nfk": @"detection", @"label": @"cat", @"classIndex": @3, @"confidence": @0.875,
								@"boundingBox": @{ @"x": @0.25, @"y": @0.25, @"width": @0.5, @"height": @0.5 } };
	XCTAssertEqualObjects(outputs[@"region"], expected, @"the service answers with the JSON form, never an archive");
	XCTAssertEqualObjects([self resultOfReply:reply descriptors:@[]].outputs[@"region"], detection);

	NFKInferenceWireCoding *archiving = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	id archived = [archiving JSONObjectForRequest:[NFKInferenceRequest requestWithInputs:@{ @"region": detection }] error:NULL];
	XCTAssertEqualObjects(archived[@"inputs"][@"region"][@"$nfk"], @"archive");
	reply = [self request:@{ @"type": @"run", @"backend": handle, @"request": archived }];
	XCTAssertEqualObjects(reply[@"result"][@"outputs"][@"regionClass"], @"NFKDetection", @"an archive still decodes");
	XCTAssertEqualObjects(reply[@"result"][@"outputs"][@"region"][@"$nfk"], @"detection");
}

- (void)testResultFilesLandInTheRunsOutputDirectory
{
	NSNumber *handle = [self startAndBuild:@"echo"];
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[NSString stringWithFormat:@"service-test-run-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSDictionary *reply = [self request:@{ @"type": @"run", @"backend": handle, @"outputDirectory": directory.path,
										   @"request": @{ @"inputs": @{ NFKInputPrompt: @"keep" }, @"parameters": @{ @"file": @YES } } }];
	NSString *path = reply[@"result"][@"outputs"][@"file"][@"path"];
	XCTAssertTrue([path hasPrefix:[directory.URLByStandardizingPath.path stringByAppendingString:@"/"]], @"%@", reply);
	XCTAssertEqualObjects([NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:NULL], @"keep");
}

- (void)testRunsWaitingForASlotHoldNoThreads
{
	NFKPluginService *service = [[NFKPluginService alloc] initWithPlugin:[[NFKServiceTestPlugin alloc] initWithContext:(id<NFKPluginContext>)NSNull.null]
																manifest:[self manifestWithConcurrentRuns:1]];
	[self startService:service];
	[self request:[self hello]];
	NSNumber *handle = [self request:@{ @"type": @"build", @"model": @"nap" }][@"result"][@"backend"];
	XCTAssertNotNil(handle);
	NSInteger baseline = NFKServiceTestThreadCount();
	NSMutableArray<NSNumber *> *identifiers = [NSMutableArray array];
	for (NSInteger index = 0; index < 64; index++) {
		NSDictionary *run = @{ @"type": @"run", @"backend": handle,
							   @"request": @{ @"inputs": @{ NFKInputPrompt: [NSString stringWithFormat:@"%ld", (long)index] },
											  @"parameters": @{ @"milliseconds": @10 } } };
		[identifiers addObject:[self send:run descriptors:nil]];
	}
	__block NSInteger most = baseline;
	XCTAssertTrue([self waitUntil:^BOOL {
		most = MAX(most, NFKServiceTestThreadCount());
		return [self messagesOfType:@"reply"].count >= 66;
	} timeout:30]);
	XCTAssertLessThanOrEqual(most - baseline, 16, @"64 runs queued on one slot ran on %ld threads more than the %ld before",
							 (long)(most - baseline), (long)baseline);
	for (NSUInteger index = 0; index < identifiers.count; index++) {
		NSDictionary *reply = [self replyTo:identifiers[index] descriptors:NULL];
		XCTAssertEqualObjects(reply[@"result"][@"outputs"][NFKOutputText], ([NSString stringWithFormat:@"%lu", (unsigned long)index]));
	}
}

#pragma mark Tools and other messages

- (void)testCallToolAnswersAToolResultAndRefusesAnUnknownTool
{
	[self startService:[self testService]];
	[self request:[self hello]];
	NSDictionary *reply = [self request:@{ @"type": @"callTool", @"tool": @"greet", @"arguments": @{ @"name": @"Ada" } }];
	NSDictionary *expected = @{ @"$nfk": @"toolResult", @"text": @"Hello, Ada", @"isError": @NO, @"images": @[], @"outputs": @{} };
	XCTAssertEqualObjects(reply[@"result"][@"toolResult"], expected);
	[self assertReply:[self request:@{ @"type": @"callTool", @"tool": @"absent", @"arguments": @{} }]
	  failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownTool];
}

- (void)testAnUnknownMessageIsUnsupportedWithAnIdentifierAndIgnoredWithout
{
	[self startService:[self testService]];
	[self request:[self hello]];
	[self assertReply:[self request:@{ @"type": @"frobnicate" }] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolUnsupported];
	XCTAssertTrue([self.channel sendMessage:@{ @"type": @"frobnicate" } error:NULL]);
	XCTAssertNotNil(([self request:@{ @"type": @"ping", @"unknownField": @1 }][@"result"]));
	XCTAssertEqual([self messagesOfType:@"reply"].count, 3u);
}

#pragma mark Ending

- (void)testShutdownEndsServingInOrder
{
	[self startService:[self testService]];
	[self request:[self hello]];
	XCTAssertNotNil(([self request:@{ @"type": @"shutdown" }][@"result"]));
	XCTAssertEqual(dispatch_semaphore_wait(self.served, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	@synchronized (self) {
		XCTAssertTrue(self.servedOrderly);
	}
	XCTAssertTrue([self waitUntil:^BOOL { @synchronized (self) { return self.channelClosed; } } timeout:5]);
}

- (void)testClosingTheChannelEndsServing
{
	[self startService:[self testService]];
	[self request:[self hello]];
	[self.channel close];
	XCTAssertEqual(dispatch_semaphore_wait(self.served, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)), 0);
	@synchronized (self) {
		XCTAssertTrue(self.servedOrderly);
	}
}

#pragma mark Loading on hello

- (void)testAServiceThatLoadsOnHelloPassesTheOptionsAndForwardsLogs
{
	__block NSDictionary *received = nil;
	NFKPluginService *service = [[NFKPluginService alloc] initWithManifest:[self manifest] loader:^id<NFKPlugin>(NSDictionary *options,
																												 NFKPluginLogHandler logHandler,
																												 NSError **error) {
		received = options;
		logHandler(NFKPluginLogLevelWarning, @"loaded");
		return [[NFKServiceTestPlugin alloc] initWithContext:(id<NFKPluginContext>)NSNull.null];
	}];
	XCTAssertNil(service.plugin, @"nothing loads before hello");
	[self startService:service];
	XCTAssertNotNil([self request:[self hello]][@"result"]);
	XCTAssertEqualObjects(received, @{ @"greeting": @"Hi" });
	XCTAssertNotNil(service.plugin);
	XCTAssertEqualObjects([self messagesOfType:@"log"].firstObject, (@{ @"type": @"log", @"level": @2, @"message": @"loaded" }));
}

- (void)testAServiceThatLoadsOnHelloAnswersTheLoadError
{
	NFKPluginService *service = [[NFKPluginService alloc] initWithManifest:[self manifest] loader:^id<NFKPlugin>(NSDictionary *options,
																												 NFKPluginLogHandler logHandler,
																												 NSError **error) {
		*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorDuplicateRuntime
								 userInfo:@{ NSLocalizedDescriptionKey: @"the plugin carries its own InferKit" }];
		return nil;
	}];
	[self startService:service];
	NSDictionary *reply = [self request:[self hello]];
	[self assertReply:reply failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorDuplicateRuntime];
	XCTAssertEqualObjects(reply[@"error"][@"message"], @"the plugin carries its own InferKit");
	[self assertReply:[self request:@{ @"type": @"describe" }] failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProtocolViolation];
}

#pragma mark C plugin

- (void)testACPluginServesThroughTheService
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindCABI version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	NFKLoadedPlugin *loaded = [NFKPluginLoader loadPluginAtURL:bundle signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error];
	XCTAssertNotNil(loaded, @"%@", error);
	[self startService:[[NFKPluginService alloc] initWithPlugin:loaded.plugin manifest:loaded.manifest]];
	[self request:[self hello]];
	NSDictionary *built = [self request:@{ @"type": @"build", @"model": NFKPluginFixtureInvertModel }];
	NSNumber *handle = built[@"result"][@"backend"];
	XCTAssertNotNil(handle, @"%@", built);

	NSMutableArray<NSNumber *> *values = [NSMutableArray array];
	for (NSUInteger index = 0; index < 30000; index++) {
		[values addObject:@(0.25f)];
	}
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSink = writer;
	id request = [coder JSONObjectForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: values }] error:&error];
	NSNumber *identifier = [self send:@{ @"type": @"run", @"backend": handle, @"request": request } descriptors:writer.descriptors];
	NSArray<NFKPluginDescriptor *> *descriptors = nil;
	NSDictionary *reply = [self replyTo:identifier descriptors:&descriptors];
	NFKInferenceResult *result = [self resultOfReply:reply descriptors:descriptors];
	NSArray *inverted = result.outputs[NFKPluginFixtureValuesKey];
	XCTAssertEqual(inverted.count, 30000u);
	XCTAssertEqualObjects(inverted.firstObject, @0.75);

	NSDictionary *tool = [self request:@{ @"type": @"callTool", @"tool": NFKPluginFixtureEchoTool, @"arguments": @{ @"text": @"through" } }];
	XCTAssertEqualObjects(tool[@"result"][@"toolResult"][@"text"], @"through");
	XCTAssertNotNil(([self request:@{ @"type": @"release", @"backend": handle }][@"result"]));
}

#pragma mark The plugin host

- (void)testThePluginHostServesABundleItVerifies
{
	NSURL *host = [NFKPluginFixtures hostExecutableURL];
	if (host == nil) {
		XCTSkip(@"the inferkit-plugin-host product is not built beside the tests");
	}
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindObjC version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	for (NSNumber *allowsAdHoc in @[ @NO, @YES ]) {
		int pluginEnd = -1;
		int hostEnd = -1;
		XCTAssertTrue([NFKPluginChannel createSocketPair:&pluginEnd second:&hostEnd error:&error], @"%@", error);
		NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithObjects:@"--bundle", bundle.path, nil];
		if (allowsAdHoc.boolValue) {
			[arguments addObject:@"--allow-ad-hoc"];
		}
		NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:host.path arguments:arguments];
		process.ownerInheritedDescriptors = @{ @3: @(pluginEnd) };
		process.environment = @{ @"INFERKIT_PLUGIN_CHANNEL_FD": @"3" };
		XCTAssertTrue([process startWithError:&error], @"%@", error);
		process.ownerInheritedDescriptors = nil;
		close(pluginEnd);
		NFKPluginChannel *channel = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:10 error:&error];
		XCTAssertNotNil(channel, @"%@ %@", error, process.recentStandardError);
		@synchronized (self) {
			[self.messages removeAllObjects];
			[self.messageDescriptors removeAllObjects];
		}
		[self attachToChannel:channel];
		NSDictionary *hello = [self request:[self hello]];
		NSDictionary *verified = [self messagesOfType:@"event"].firstObject;
		XCTAssertEqualObjects(verified[@"event"], @"verified", @"the host verifies after the hand-off and says so");
		@synchronized (self) {
			XCTAssertTrue([self.messages indexOfObject:verified] < [self.messages indexOfObject:hello], @"before it answers hello");
		}
		if (allowsAdHoc.boolValue) {
			XCTAssertEqualObjects(hello[@"result"][@"plugin"][@"id"], [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindObjC], @"%@", hello);
			XCTAssertEqualObjects(hello[@"result"][@"pid"], @(process.processIdentifier));
			XCTAssertEqual(channel.peerProcessIdentifier, process.processIdentifier, @"the host handed over a socket of its own");
		} else {
			[self assertReply:hello failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];
		}
		XCTAssertNotNil(([self request:@{ @"type": @"shutdown" }][@"result"]));
		pid_t pid = process.processIdentifier;
		if (![self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:10]) {
			XCTFail(@"the host outlived its shutdown by 10 s (ad hoc allowed: %@): %@", allowsAdHoc, [self diagnosisOfHost:process pid:pid]);
		}
		XCTAssertEqual(process.terminationReason, NFKSubprocessTerminationReasonExit);
		XCTAssertEqual(process.terminationStatus, 0, @"%@", process.recentStandardError);
		[channel close];
	}
	self.channel = nil;
}

/*! What a plugin host that has not exited is doing: its state, the end of its standard error, and every
	thread's stack, sampled for a second. */
- (NSString *)diagnosisOfHost:(NFKSubprocess *)process pid:(pid_t)pid
{
	NSString *identifier = [NSString stringWithFormat:@"%d", pid];
	NSString *status = [NFKPluginFixtures runProgram:@"/bin/ps" arguments:@[ @"-o", @"pid=,ppid=,stat=,etime=,command=", @"-p", identifier ]
											   error:NULL] ?: @"(not listed)";
	NSURL *report = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					 [NSString stringWithFormat:@"inferkit-plugin-host-%@.sample.txt", NSUUID.UUID.UUIDString]];
	NSError *sampleError = nil;
	NSString *stacks = [NFKPluginFixtures runProgram:@"/usr/bin/sample" arguments:@[ identifier, @"1", @"-mayDie", @"-file", report.path ]
											   error:&sampleError] != nil
		? [NSString stringWithContentsOfURL:report encoding:NSUTF8StringEncoding error:NULL] : sampleError.localizedDescription;
	[NSFileManager.defaultManager removeItemAtURL:report error:NULL];
	return [NSString stringWithFormat:@"subprocess state %ld; ps: %@\nstandard error:\n%@\nstacks:\n%@", (long)process.state, status,
			process.recentStandardError, stacks ?: @"(no report)"];
}

/*! Runs the plugin host with arguments on a fresh channel and returns the hello reply, or nil when the
	process exits first. */
- (nullable NSDictionary *)helloFromHostWithArguments:(NSArray<NSString *> *)arguments process:(NFKSubprocess * _Nullable * _Nonnull)outProcess
{
	NSURL *host = [NFKPluginFixtures hostExecutableURL];
	int pluginEnd = -1;
	int hostEnd = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&pluginEnd second:&hostEnd error:&error], @"%@", error);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:host.path arguments:arguments];
	process.ownerInheritedDescriptors = @{ @3: @(pluginEnd) };
	*outProcess = process;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	process.ownerInheritedDescriptors = nil;
	close(pluginEnd);
	NFKPluginChannel *channel = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:10 error:&error];
	if (channel == nil) {
		return nil;
	}
	@synchronized (self) {
		[self.messages removeAllObjects];
		[self.messageDescriptors removeAllObjects];
	}
	[self attachToChannel:channel];
	return [self request:[self hello]];
}

- (void)testThePluginHostRefusesABundleThatCarriesItsOwnInferKit
{
	if ([NFKPluginFixtures hostExecutableURL] == nil) {
		XCTSkip(@"the inferkit-plugin-host product is not built beside the tests");
	}
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures duplicateRuntimeBundleURLWithError:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	NFKSubprocess *process = nil;
	NSDictionary *hello = [self helloFromHostWithArguments:@[ @"--bundle", bundle.path, @"--allow-ad-hoc" ] process:&process];
	[self assertReply:hello failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorDuplicateRuntime];
	XCTAssertTrue([hello[@"error"][@"message"] containsString:@"NFKInferenceRequest"], @"%@", hello);
	XCTAssertNotNil(([self request:@{ @"type": @"shutdown" }][@"result"]));
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:10]);
	XCTAssertEqual(process.terminationStatus, 0, @"%@", process.recentStandardError);
	[self.channel close];
	self.channel = nil;
}

- (void)testThePluginHostEnforcesARequirementAndRefusesUnsignedAlone
{
	if ([NFKPluginFixtures hostExecutableURL] == nil) {
		XCTSkip(@"the inferkit-plugin-host product is not built beside the tests");
	}
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindObjC version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error);

	NFKSubprocess *process = nil;
	NSArray<NSString *> *pinned = @[ @"--bundle", bundle.path, @"--allow-ad-hoc", @"--requirement", @"identifier \"com.example.someone-else\"" ];
	NSDictionary *hello = [self helloFromHostWithArguments:pinned process:&process];
	[self assertReply:hello failsWithDomain:NFKPluginErrorDomain code:NFKPluginErrorTeamMismatch];
	XCTAssertNotNil(([self request:@{ @"type": @"shutdown" }][@"result"]));
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:10]);
	[self.channel close];

	NFKSubprocess *refused = [[NFKSubprocess alloc] initWithCommand:[NFKPluginFixtures hostExecutableURL].path
														  arguments:@[ @"--bundle", bundle.path, @"--allow-unsigned" ]];
	XCTAssertTrue([refused startWithError:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return refused.state == NFKSubprocessStateExited; } timeout:10]);
	XCTAssertEqual(refused.terminationStatus, 64, @"--allow-unsigned needs the development policy");
	XCTAssertTrue([refused.recentStandardError containsString:@"--allow-ad-hoc"], @"%@", refused.recentStandardError);
	self.channel = nil;
}

@end

#endif
