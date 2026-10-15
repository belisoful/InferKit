//
//  NFKPluginBackendTests.m
//  InferKitTests
//
//  A plugin's model as an inference backend, against an in-test peer: keys from the manifest with no
//  process, the lazy start, prepare, progress and partial results, cancel, errors that keep their
//  domain and code, residency through events, release, the rebuild after a restart, attachments with
//  the run directory, and the in-process form.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginTestPeer.h"

/*! A plugin's own backend, for the in-process form. */
@interface NFKPluginInProcessEchoBackend : NSObject <NFKInferenceBackend>
@property (atomic, assign) BOOL unloaded;
@end

@implementation NFKPluginInProcessEchoBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"in-process-echo";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [@"in-process:" stringByAppendingString:request.prompt ?: @""] }];
}

- (void)unload
{
	self.unloaded = YES;
}

- (BOOL)isLoaded
{
	return !self.unloaded;
}

- (NSInteger)residentByteCount
{
	return self.unloaded ? 0 : 512;
}

@end

@interface NFKPluginBackendTests : XCTestCase
@property (nonatomic, strong) NFKPluginTestHarness *harness;
@property (nonatomic, strong) NFKPluginProcess *process;
@end

@implementation NFKPluginBackendTests

- (void)setUp
{
	self.continueAfterFailure = NO;
	self.harness = [[NFKPluginTestHarness alloc] init];
	self.process = [[NFKPluginProcess alloc] initWithManifest:NFKPluginTestManifest(@"1.0.0", 300) hostExecutableURL:nil];
	self.process.launcherFactory = self.harness.factory;
	self.process.memoryPressureProvider = ^double {
		return 0;
	};
}

- (void)tearDown
{
	XCTestExpectation *stopped = [self expectationWithDescription:@"stopped"];
	[self.process stopWithCompletionHandler:^{
		[stopped fulfill];
	}];
	[self waitForExpectations:@[ stopped ] timeout:20];
}

- (NFKPluginBackend *)backend
{
	NSError *error = nil;
	NFKPluginBackend *backend = [self.process backendForModelIdentifier:@"echo" options:@{ @"size": @"small" } error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	return backend;
}

- (NFKInferenceRequest *)requestWithPrompt:(NSString *)prompt
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: prompt }];
}

- (void)waitForJob:(NFKInferenceJob *)job
{
	XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
		return job.status >= NFKInferenceJobStatusSucceeded;
	}));
}

#pragma mark Description

- (void)testKeysAndModelInfoComeFromTheManifestWithoutAProcess
{
	NFKPluginBackend *backend = self.backend;
	XCTAssertEqualObjects(backend.backendIdentifier, @"plugin:com.example.echo/echo");
	XCTAssertEqualObjects(backend.pluginIdentifier, @"com.example.echo");
	XCTAssertEqualObjects(backend.modelIdentifier, @"echo");
	XCTAssertEqualObjects(backend.options, (@{ @"size": @"small" }));
	XCTAssertEqual(backend.isolation, NFKPluginIsolationProcess);
	XCTAssertEqual(backend.process, self.process);
	XCTAssertTrue(backend.isReady);
	XCTAssertEqualObjects(backend.supportedInputKeys, [NSSet setWithObject:NFKInputPrompt]);
	XCTAssertEqualObjects(backend.supportedOutputKeys, [NSSet setWithObject:NFKOutputText]);
	XCTAssertEqualObjects(backend.supportedParameterKeys, [NSSet setWithObject:NFKParameterTemperature], @"a load option is not a request parameter");
	XCTAssertEqualObjects(backend.modelInfo[NFKModelInfoArchitecture], @"echo");
	XCTAssertFalse(backend.isLoaded);
	XCTAssertEqual(backend.residentByteCount, 0);
	XCTAssertEqual(self.process.state, NFKPluginProcessStateNotRunning);
	XCTAssertEqual(self.harness.peers.count, 0u);
}

#pragma mark Running

- (void)testTheFirstRunStartsTheProcess
{
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithPrompt:@"hello"] error:&error];
	XCTAssertEqualObjects(result.text, @"1:hello", @"%@", error);
	XCTAssertEqual(self.process.state, NFKPluginProcessStateReady);
	NFKPluginTestPeer *peer = self.harness.peers.firstObject;
	XCTAssertEqualObjects(peer.receivedTypes, (@[ NFKPluginMessageHello, NFKPluginMessageDescribe, NFKPluginMessageBuild, NFKPluginMessageRun ]));
	NSDictionary *build = [peer receivedMessagesOfType:NFKPluginMessageBuild].firstObject;
	XCTAssertEqualObjects(build[@"model"], @"echo");
	XCTAssertEqualObjects(build[@"options"], (@{ @"size": @"small" }));
	NSDictionary *run = [peer receivedMessagesOfType:NFKPluginMessageRun].firstObject;
	XCTAssertEqualObjects(run[@"backend"], @101);
	XCTAssertEqualObjects(run[@"stream"], @NO);
	XCTAssertEqualObjects(backend.modelInfo[NFKModelInfoParameterCount], @7, @"the build's model information joins the manifest's");

	XCTAssertEqualObjects([backend runInferenceForRequest:[self requestWithPrompt:@"again"] error:&error].text, @"1:again", @"%@", error);
	XCTAssertEqual([peer receivedMessagesOfType:NFKPluginMessageBuild].count, 1u, @"one build serves every run of a process");
}

- (void)testPrepareStartsTheProcessAndReportsTheLoad
{
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	XCTAssertTrue([backend prepareWithError:&error], @"%@", error);
	XCTAssertTrue(backend.isLoaded);
	XCTAssertEqual(backend.residentByteCount, 1000);
	XCTAssertEqual(backend.peakByteCount, 2000);
	XCTAssertEqualObjects(self.harness.peers.firstObject.receivedTypes.lastObject, NFKPluginMessagePrepare);
}

- (void)testAPreparationThatOutlastsTheBuildTimeoutFailsWithTimedOut
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessagePrepare: ^(NFKPluginTestPeer *slow, NSDictionary *message, NSArray *descriptors) {
			[NSThread sleepForTimeInterval:1.0];
			[slow replyTo:message result:@{ @"loaded": @YES, @"residentBytes": @3000, @"peakBytes": @4000 }];
		} };
	};
	self.process.buildTimeout = 0.3;
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	XCTAssertFalse([backend prepareWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
	XCTAssertEqual(error.code, NFKPluginErrorTimedOut);
	XCTAssertTrue([error.localizedDescription containsString:@"echo"], @"%@", error);
	XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
		return backend.isLoaded;
	}), @"the late preparation's load is recorded");
	XCTAssertEqual(backend.residentByteCount, 3000);
	XCTAssertEqualObjects([backend runInferenceForRequest:[self requestWithPrompt:@"after"] error:&error].text, @"1:after",
						  @"the timed-out preparation does not hold the process: %@", error);
}

- (void)testASubmittedJobRelaysProgressAndPartialResults
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *running, NSDictionary *message, NSArray *descriptors) {
			[running sendProgress:0.25 partial:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"par" }] forRequest:message];
			[running sendProgress:0.75 partial:nil forRequest:message];
			[running replyTo:message inferenceResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"partial and whole" }]];
		} };
	};
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self requestWithPrompt:@"stream"]];
	NSMutableArray<NSNumber *> *progress = [NSMutableArray array];
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	job.progressHandler = ^(NFKInferenceJob *updated) {
		@synchronized (progress) {
			[progress addObject:@(updated.progress)];
			if (updated.partialResult.text != nil) {
				[partials addObject:updated.partialResult.text];
			}
		}
	};
	[self waitForJob:job];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"partial and whole");
	@synchronized (progress) {
		XCTAssertTrue([progress containsObject:@0.25], @"%@", progress);
		XCTAssertTrue([progress containsObject:@0.75], @"%@", progress);
		XCTAssertEqualObjects(partials.firstObject, @"par");
	}
	NSDictionary *run = [self.harness.peers.firstObject receivedMessagesOfType:NFKPluginMessageRun].firstObject;
	XCTAssertEqualObjects(run[@"stream"], @YES);
}

- (void)testCancellingAJobCancelsTheRunInThePlugin
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *running, NSDictionary *message, NSArray *descriptors) {
			// The run answers only when the host cancels it.
		}, NFKPluginMessageCancel: ^(NFKPluginTestPeer *running, NSDictionary *message, NSArray *descriptors) {
			[running replyCancelledTo:message];
		} };
	};
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self requestWithPrompt:@"long"]];
	NFKPluginTestPeer *peer = nil;
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return self.harness.peers.count == 1;
	}));
	peer = self.harness.peers.firstObject;
	NSDictionary *run = [peer waitForMessageOfType:NFKPluginMessageRun timeout:5];
	XCTAssertNotNil(run);
	[job cancel];
	NSDictionary *cancel = [peer waitForMessageOfType:NFKPluginMessageCancel timeout:5];
	XCTAssertEqualObjects(cancel[@"id"], run[@"id"]);
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
}

- (void)testAPluginErrorKeepsItsDomainAndCode
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *failing, NSDictionary *message, NSArray *descriptors) {
			[failing replyTo:message error:[NSError errorWithDomain:@"com.example.echo" code:42
														   userInfo:@{ NSLocalizedDescriptionKey: @"the echo is out of breath", @"attempt": @3 }]];
		} };
	};
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[self requestWithPrompt:@"fail"] error:&error]);
	XCTAssertEqualObjects(error.domain, @"com.example.echo");
	XCTAssertEqual(error.code, 42);
	XCTAssertEqualObjects(error.localizedDescription, @"the echo is out of breath");
	XCTAssertEqualObjects(error.userInfo[@"attempt"], @3);

	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self requestWithPrompt:@"fail"]];
	[self waitForJob:job];
	XCTAssertEqualObjects(job.error.domain, @"com.example.echo");
	XCTAssertEqual(job.error.code, 42);
}

#pragma mark Residency

- (void)testEventsDriveIsLoadedAndResidentBytes
{
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	XCTAssertNotNil([backend runInferenceForRequest:[self requestWithPrompt:@"load"] error:&error], @"%@", error);
	XCTAssertFalse(backend.isLoaded, @"the build reported the model unloaded");
	NFKPluginTestPeer *peer = self.harness.peers.firstObject;
	NSDictionary *run = [peer receivedMessagesOfType:NFKPluginMessageRun].firstObject;

	[peer sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": @"loaded", @"backend": run[@"backend"], @"residentBytes": @4096, @"peakBytes": @8192 }];
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return backend.isLoaded;
	}));
	XCTAssertEqual(backend.residentByteCount, 4096);
	XCTAssertEqual(backend.peakByteCount, 8192);
	XCTAssertEqualObjects(backend.modelInfo[NFKModelInfoResidentBytes], @4096);

	[backend unload];
	NSDictionary *unload = [peer waitForMessageOfType:NFKPluginMessageUnload timeout:5];
	XCTAssertEqualObjects(unload[@"backend"], run[@"backend"]);
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return !backend.isLoaded;
	}));
	XCTAssertEqual(backend.residentByteCount, 0);
	XCTAssertEqual(backend.peakByteCount, 8192, @"the peak stays after an unload");
	XCTAssertEqual(self.process.state, NFKPluginProcessStateReady, @"an unload keeps the process");
}

- (void)testAMessageWithoutAPeakKeepsTheMeasuredPeak
{
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	XCTAssertNotNil([backend runInferenceForRequest:[self requestWithPrompt:@"load"] error:&error], @"%@", error);
	NFKPluginTestPeer *peer = self.harness.peers.firstObject;
	id handle = [peer receivedMessagesOfType:NFKPluginMessageRun].firstObject[@"backend"];

	[peer sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": @"loaded", @"backend": handle, @"residentBytes": @4096 }];
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return backend.isLoaded;
	}));
	XCTAssertEqual(backend.peakByteCount, 4096, @"a load without a peak raises the peak to what it holds");

	[peer sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": @"unloaded", @"backend": handle }];
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return !backend.isLoaded;
	}));
	XCTAssertEqual(backend.peakByteCount, 4096, @"an unload without a peak keeps it");

	[peer sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": @"loaded", @"backend": handle, @"residentBytes": @1024, @"peakBytes": @2048 }];
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return backend.isLoaded && backend.residentByteCount == 1024;
	}));
	XCTAssertEqual(backend.peakByteCount, 4096, @"a lower reported peak keeps the highest one measured");
}

- (void)testAnUnloadWithoutAProcessStartsNothing
{
	NFKPluginBackend *backend = self.backend;
	[backend unload];
	XCTAssertEqual(self.harness.peers.count, 0u);
	XCTAssertFalse(NFKInferenceIsLoaded(backend));
}

- (void)testReleasingTheBackendReleasesItsPluginBackendAndLetsTheProcessIdle
{
	self.process.idleExitInterval = 0.3;
	id handle = nil;
	@autoreleasepool {
		NFKPluginBackend *backend = self.backend;
		NSError *error = nil;
		XCTAssertNotNil([backend runInferenceForRequest:[self requestWithPrompt:@"keep"] error:&error], @"%@", error);
		handle = [self.harness.peers.firstObject receivedMessagesOfType:NFKPluginMessageRun].firstObject[@"backend"];
		[NSThread sleepForTimeInterval:0.8];
		XCTAssertEqual(self.process.state, NFKPluginProcessStateReady, @"a live backend keeps the process");
	}
	NSDictionary *released = [self.harness.peers.firstObject waitForMessageOfType:NFKPluginMessageRelease timeout:5];
	XCTAssertEqualObjects(released[@"backend"], handle);
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return self.process.state == NFKPluginProcessStateNotRunning;
	}));
}

- (void)testABackendBuildsAgainAfterTheProcessRestarts
{
	NFKPluginBackend *backend = self.backend;
	NSError *error = nil;
	XCTAssertEqualObjects([backend runInferenceForRequest:[self requestWithPrompt:@"before"] error:&error].text, @"1:before", @"%@", error);
	XCTAssertTrue([backend prepareWithError:&error], @"%@", error);
	XCTAssertTrue(backend.isLoaded);
	XCTAssertEqual(backend.builtGeneration, 1u);

	XCTestExpectation *restarted = [self expectationWithDescription:@"restarted"];
	[self.process restartWithManifest:NFKPluginTestManifest(@"1.1.0", 300) completionHandler:^(NSError *restartError) {
		XCTAssertNil(restartError);
		[restarted fulfill];
	}];
	[self waitForExpectations:@[ restarted ] timeout:30];
	XCTAssertFalse(backend.isLoaded, @"the old process's model is gone");

	XCTAssertEqualObjects([backend runInferenceForRequest:[self requestWithPrompt:@"after"] error:&error].text, @"2:after", @"%@", error);
	XCTAssertEqual(backend.builtGeneration, 2u);
	XCTAssertEqualObjects(self.harness.peers[1].receivedTypes,
						  (@[ NFKPluginMessageHello, NFKPluginMessageDescribe, NFKPluginMessageBuild, NFKPluginMessageRun ]));
}

#pragma mark Attachments

- (void)testLargeValuesTravelAsAttachmentsAndResultFilesLiveInTheRunDirectory
{
	NSMutableData *input = [NSMutableData dataWithLength:200 * 1024];
	arc4random_buf(input.mutableBytes, input.length);
	NSMutableData *output = [NSMutableData dataWithLength:100 * 1024];
	arc4random_buf(output.mutableBytes, output.length);
	__block NSData *received = nil;
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *running, NSDictionary *message, NSArray *descriptors) {
			received = [[running requestOfRun:message descriptors:descriptors] inputForKey:@"data"];
			NSURL *directory = [NSURL fileURLWithPath:message[@"outputDirectory"] isDirectory:YES];
			NSURL *file = [directory URLByAppendingPathComponent:@"answer.txt"];
			[@"written by the plugin" writeToURL:file atomically:YES encoding:NSUTF8StringEncoding error:NULL];
			[running replyTo:message inferenceResult:[NFKInferenceResult resultWithOutputs:@{ @"data": output, @"file": file }]];
		} };
	};
	NSString *directoryPath = nil;
	@autoreleasepool {
		NSError *error = nil;
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"bytes", @"data": input }];
		NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
		XCTAssertNotNil(result, @"%@", error);
		XCTAssertEqualObjects(received, input);

		NSDictionary *run = [self.harness.peers.firstObject receivedMessagesOfType:NFKPluginMessageRun].firstObject;
		XCTAssertEqualObjects(run[@"request"][@"inputs"][@"data"][@"$nfk"], @"buffer", @"the input left the JSON");
		directoryPath = run[@"outputDirectory"];
		XCTAssertEqualObjects([result outputForKey:@"data"], output);
		NSURL *file = [result outputForKey:@"file"];
		XCTAssertTrue([file.path hasPrefix:directoryPath], @"%@", file);
		XCTAssertEqualObjects([NSString stringWithContentsOfURL:file encoding:NSUTF8StringEncoding error:NULL], @"written by the plugin");
		XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:directoryPath]);
	}
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return ![NSFileManager.defaultManager fileExistsAtPath:directoryPath];
	}), @"the run directory goes with the result");
}

#pragma mark In process

- (void)testAnInProcessBackendForwardsToThePluginsBackend
{
	NFKPluginInProcessEchoBackend *own = [[NFKPluginInProcessEchoBackend alloc] init];
	NFKModelDescriptor *descriptor = self.process.manifest.models.firstObject;
	NFKPluginBackend *backend = [[NFKPluginBackend alloc] initWithPluginIdentifier:@"com.example.echo" descriptor:descriptor
																		   options:@{} pluginBackend:own];
	XCTAssertEqual(backend.isolation, NFKPluginIsolationInProcess);
	XCTAssertNil(backend.process);
	XCTAssertEqualObjects(backend.backendIdentifier, @"plugin:com.example.echo/echo");
	XCTAssertEqualObjects(backend.supportedInputKeys, [NSSet setWithObject:NFKInputPrompt]);
	NSError *error = nil;
	XCTAssertEqualObjects([backend runInferenceForRequest:[self requestWithPrompt:@"here"] error:&error].text, @"in-process:here", @"%@", error);
	XCTAssertTrue(backend.isLoaded);
	XCTAssertEqual(backend.residentByteCount, 512);
	[backend unload];
	XCTAssertTrue(own.unloaded);
	XCTAssertFalse(backend.isLoaded);

	NFKInferenceJob *job = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"job"]];
	[self waitForJob:job];
	XCTAssertEqualObjects(job.result.text, @"in-process:job");
	XCTAssertEqual(self.harness.peers.count, 0u);
}

@end

#endif
