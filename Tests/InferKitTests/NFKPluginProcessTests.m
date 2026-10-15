//
//  NFKPluginProcessTests.m
//  InferKitTests
//
//  The plugin supervisor against an in-test peer: its states, the hello checks, crashes and the crash
//  limit, restarts with a drain, idle exit and memory pressure, stop, the signature check before a
//  launch, tools, and the plugin's log.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginTestPeer.h"

@interface NFKPluginProcessTests : XCTestCase
@property (nonatomic, strong) NFKPluginTestHarness *harness;
@property (nonatomic, strong) NSMutableArray<NSNumber *> *states;
@end

@implementation NFKPluginProcessTests

- (void)setUp
{
	self.continueAfterFailure = NO;
	self.harness = [[NFKPluginTestHarness alloc] init];
	self.states = [NSMutableArray array];
}

- (NFKPluginProcess *)processWithIdleExit:(NSTimeInterval)idleExitSeconds
{
	NFKPluginProcess *process = [[NFKPluginProcess alloc] initWithManifest:NFKPluginTestManifest(@"1.0.0", idleExitSeconds) hostExecutableURL:nil];
	process.launcherFactory = self.harness.factory;
	process.memoryPressureProvider = ^double {
		return 0;
	};
	NSMutableArray<NSNumber *> *states = self.states;
	process.stateHandler = ^(NFKPluginProcess *changed, NFKPluginProcessState state) {
		@synchronized (states) {
			[states addObject:@(state)];
		}
	};
	return process;
}

- (NSArray<NSNumber *> *)recordedStates
{
	@synchronized (self.states) {
		return [self.states copy];
	}
}

- (NFKPluginBackend *)backendOfProcess:(NFKPluginProcess *)process
{
	NSError *error = nil;
	NFKPluginBackend *backend = [process backendForModelIdentifier:@"echo" options:@{} error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	return backend;
}

- (NFKInferenceRequest *)requestWithPrompt:(NSString *)prompt
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: prompt }];
}

- (void)waitForJob:(NFKInferenceJob *)job
{
	XCTAssertTrue(NFKPluginTestWaitUntil(10, ^BOOL {
		return job.status >= NFKInferenceJobStatusSucceeded;
	}));
}

#pragma mark Starting

- (void)testAStartPassesThroughStartingToReadyAfterTheHello
{
	NFKPluginProcess *process = [self processWithIdleExit:300];
	XCTAssertEqual(process.state, NFKPluginProcessStateNotRunning);
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
	XCTAssertEqual(process.generation, 1u);
	XCTAssertNil(process.failureError);
	XCTAssertTrue(NFKPluginTestWaitUntil(2, ^BOOL {
		return self.recordedStates.count == 2;
	}));
	XCTAssertEqualObjects(self.recordedStates, (@[ @(NFKPluginProcessStateStarting), @(NFKPluginProcessStateReady) ]));

	NFKPluginTestPeer *peer = self.harness.peers.firstObject;
	XCTAssertEqualObjects(peer.receivedTypes, (@[ NFKPluginMessageHello, NFKPluginMessageDescribe ]));
	NSDictionary *hello = peer.receivedMessages.firstObject;
	XCTAssertEqualObjects(hello[@"protocol"], (@{ @"major": @1, @"minor": @0 }));
	XCTAssertEqualObjects(hello[@"inferkitAPI"], NFKInferKit.APIVersion);
	XCTAssertEqualObjects(hello[@"limits"][@"inlineBytes"], @65536);
	XCTAssertEqual(peer.auditedProcessIdentifier, getpid(), @"the audit token comes from the socket the peer handed over");

	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertEqual(self.harness.peers.count, 1u, @"a Ready process is left alone");
}

- (void)testAHelloWithAnotherMajorVersionFailsWithItsReason
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.protocol = @{ @"major": @2, @"minor": @0 };
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
	XCTAssertEqual(error.code, NFKPluginErrorProtocolUnsupported);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
	XCTAssertEqual(process.failureError.code, NFKPluginErrorProtocolUnsupported);
	XCTAssertTrue([process.failureError.localizedDescription containsString:@"2.0"], @"%@", process.failureError);
	XCTAssertTrue(self.harness.peers.firstObject.stopped);
	XCTAssertTrue(NFKPluginTestWaitUntil(2, ^BOOL {
		return self.recordedStates.count == 2;
	}));
	XCTAssertEqualObjects(self.recordedStates, (@[ @(NFKPluginProcessStateStarting), @(NFKPluginProcessStateFailed) ]));
}

- (void)testHelloWaitsForTheHostsVerificationThenForItsOwnTimeout
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageHello: ^(NFKPluginTestPeer *greeted, NSDictionary *message, NSArray *descriptors) {
			[NSThread sleepForTimeInterval:0.4];
			[greeted sendMessage:@{ @"type": NFKPluginMessageEvent, @"event": NFKPluginEventVerified }];
			[NSThread sleepForTimeInterval:0.4];
			[greeted replyTo:message result:@{ @"protocol": @{ @"major": @1, @"minor": @0 },
											   @"plugin": @{ @"id": @"com.example.echo", @"version": @"1.0.0" },
											   @"inferkitAPI": @"1.0", @"capabilities": @[ @"attachments" ], @"pid": @(getpid()) }];
		} };
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	XCTAssertEqual(process.verificationTimeout, 120);
	process.handshakeTimeout = 0.6;
	process.verificationTimeout = 10;
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"0.8 s in all, but neither phase passes its own limit: %@", error);
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
}

- (void)testAVerificationThatOutlastsItsTimeoutFailsTheStart
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageHello: ^(NFKPluginTestPeer *greeted, NSDictionary *message, NSArray *descriptors) {
			[NSThread sleepForTimeInterval:2];
		} };
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	process.handshakeTimeout = 10;
	process.verificationTimeout = 0.3;
	NSDate *started = [NSDate date];
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertLessThan(-started.timeIntervalSinceNow, 1.5, @"the verification limit applies, not the handshake's");
	XCTAssertEqual(error.code, NFKPluginErrorHandshakeFailed);
	NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
	XCTAssertTrue([underlying.localizedDescription containsString:@"verification"], @"%@", error);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
}

- (void)testAHelloFromAnotherVersionFailsTheHandshake
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.pluginVersion = @"9.9.9";
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorHandshakeFailed);
	XCTAssertTrue([error.localizedDescription containsString:@"9.9.9"], @"%@", error);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
}

- (void)testAModelTheManifestListsMustBeDescribed
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.describedModelIdentifiers = @[ @"other" ];
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorUnknownModel);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
}

- (void)testAFailedLaunchLeavesTheProcessFailed
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.failsLaunch = YES;
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorLaunchFailed);
	XCTAssertEqualObjects([error.userInfo[NSUnderlyingErrorKey] domain], NSPOSIXErrorDomain);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
}

- (void)testAnUnsignedHostIsRefusedBeforeLaunch
{
	NSURL *host = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
				   [NSString stringWithFormat:@"unsigned-host-%@", NSUUID.UUID.UUIDString]];
	XCTAssertTrue([@"#!/bin/sh\nexit 0\n" writeToURL:host atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NFKPluginProcess *process = [[NFKPluginProcess alloc] initWithManifest:NFKPluginTestManifest(@"1.0.0", 300) hostExecutableURL:host];
	process.launcherFactory = self.harness.factory;
	XCTAssertTrue(process.signaturePolicy.requiresValidSignature);

	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
	XCTAssertEqual(error.code, NFKPluginErrorSignatureInvalid);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
	XCTAssertEqual(self.harness.peers.count, 0u, @"the refusal comes before any launch");
	[NSFileManager.defaultManager removeItemAtURL:host error:NULL];
}

#pragma mark Crashes

- (void)testACrashMidRunFailsTheRunAndTheNextRunRelaunches
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		if (peer.launchNumber == 1) {
			peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *crashing, NSDictionary *message, NSArray *descriptors) {
				[crashing crashWithSignal:SIGSEGV standardError:@"segmentation fault in the model"];
			} };
		}
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NFKPluginBackend *backend = [self backendOfProcess:process];

	NSError *error = nil;
	XCTAssertNil([backend runInferenceForRequest:[self requestWithPrompt:@"first"] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
	XCTAssertEqual(error.code, NFKPluginErrorProcessExited);
	XCTAssertTrue([error.localizedDescription containsString:@"signal 11"], @"%@", error);
	XCTAssertEqualObjects(error.userInfo[NSLocalizedFailureReasonErrorKey], @"segmentation fault in the model");
	XCTAssertEqual(process.state, NFKPluginProcessStateNotRunning);
	XCTAssertEqual(process.terminationReason, NFKSubprocessTerminationReasonSignal);
	XCTAssertEqual(process.terminationStatus, SIGSEGV);

	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithPrompt:@"second"] error:&error];
	XCTAssertEqualObjects(result.text, @"2:second", @"%@", error);
	XCTAssertEqual(self.harness.peers.count, 2u);
	XCTAssertEqual(process.generation, 2u);
	XCTAssertEqual(backend.builtGeneration, 2u);
	XCTAssertEqualObjects(self.harness.peers[1].receivedTypes,
						  (@[ NFKPluginMessageHello, NFKPluginMessageDescribe, NFKPluginMessageBuild, NFKPluginMessageRun ]),
						  @"the new process builds the backend again before the run");
}

- (void)testCrashesUpToTheLimitLeaveTheProcessFailedUntilAStart
{
	NFKPluginProcess *process = [self processWithIdleExit:300];
	process.crashLimit = 3;
	process.crashLimitInterval = 60;
	NSError *error = nil;
	for (NSUInteger crash = 1; crash <= 3; crash++) {
		XCTAssertTrue([process startWithError:&error], @"%@", error);
		[self.harness.peers.lastObject crashWithSignal:SIGABRT standardError:@"abort"];
		NFKPluginProcessState expected = crash < 3 ? NFKPluginProcessStateNotRunning : NFKPluginProcessStateFailed;
		XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
			return process.state == expected;
		}), @"crash %lu leaves state %ld", (unsigned long)crash, (long)process.state);
	}
	XCTAssertEqual(process.failureError.code, NFKPluginErrorProcessExited);
	XCTAssertTrue([process.failureError.localizedDescription containsString:@"3 times"], @"%@", process.failureError);

	NFKPluginBackend *backend = [self backendOfProcess:process];
	XCTAssertNil([backend runInferenceForRequest:[self requestWithPrompt:@"refused"] error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorProcessExited);
	XCTAssertEqual(self.harness.peers.count, 3u, @"a Failed process is not relaunched by a request");

	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
	XCTAssertNil(process.failureError);
	XCTAssertEqual(self.harness.peers.count, 4u);
}

#pragma mark Restart

- (void)testARestartDrainsTheRunInFlightAndHandsHeldRunsToTheNewProcessInOrder
{
	dispatch_semaphore_t releaseOldRun = dispatch_semaphore_create(0);
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		if (peer.launchNumber == 1) {
			peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *old, NSDictionary *message, NSArray *descriptors) {
				dispatch_semaphore_wait(releaseOldRun, DISPATCH_TIME_FOREVER);
				NFKInferenceRequest *request = [old requestOfRun:message descriptors:descriptors];
				[old replyTo:message inferenceResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [@"1:" stringByAppendingString:request.prompt] }]];
			} };
		}
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NFKPluginBackend *backend = [self backendOfProcess:process];
	NFKInferenceJob *inFlight = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"A"]];
	XCTAssertNotNil([self.harness.peers.firstObject waitForMessageOfType:NFKPluginMessageRun timeout:5]);

	XCTestExpectation *restarted = [self expectationWithDescription:@"restarted"];
	__block NSError *restartError = [NSError errorWithDomain:@"unset" code:0 userInfo:nil];
	[process restartWithManifest:NFKPluginTestManifest(@"2.0.0", 300) completionHandler:^(NSError *error) {
		restartError = error;
		[restarted fulfill];
	}];
	NFKInferenceJob *heldFirst = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"B"]];
	NFKInferenceJob *heldSecond = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"C"]];

	XCTAssertTrue(NFKPluginTestWaitUntil(2, ^BOOL {
		return process.state == NFKPluginProcessStateRestarting;
	}));
	[NSThread sleepForTimeInterval:0.3];
	XCTAssertEqual(self.harness.peers.count, 1u, @"the new process waits for the run in flight");
	XCTAssertEqual(heldFirst.status, NFKInferenceJobStatusRunning);

	dispatch_semaphore_signal(releaseOldRun);
	[self waitForExpectations:@[ restarted ] timeout:10];
	XCTAssertNil(restartError);
	[self waitForJob:inFlight];
	[self waitForJob:heldFirst];
	[self waitForJob:heldSecond];

	XCTAssertEqualObjects(inFlight.result.text, @"1:A", @"the old run finishes on the old process");
	XCTAssertEqualObjects(heldFirst.result.text, @"2:B", @"%@", heldFirst.error);
	XCTAssertEqualObjects(heldSecond.result.text, @"2:C", @"%@", heldSecond.error);
	XCTAssertTrue([self.harness.peers.firstObject.receivedTypes containsObject:NFKPluginMessageShutdown]);

	NFKPluginTestPeer *fresh = self.harness.peers[1];
	XCTAssertEqualObjects(fresh.receivedTypes, (@[ NFKPluginMessageHello, NFKPluginMessageDescribe, NFKPluginMessageBuild,
												   NFKPluginMessageRun, NFKPluginMessageRun ]));
	NSArray<NSDictionary *> *runs = [fresh receivedMessagesOfType:NFKPluginMessageRun];
	XCTAssertEqualObjects([fresh requestOfRun:runs[0] descriptors:@[]].prompt, @"B");
	XCTAssertEqualObjects([fresh requestOfRun:runs[1] descriptors:@[]].prompt, @"C");
	XCTAssertEqualObjects(process.manifest.version, @"2.0.0");
	XCTAssertEqual(process.generation, 2u);
	XCTAssertEqual(backend.builtGeneration, 2u);
	XCTAssertTrue([self.recordedStates containsObject:@(NFKPluginProcessStateRestarting)]);
	XCTAssertEqualObjects(self.recordedStates.lastObject, @(NFKPluginProcessStateReady));
}

- (void)testARestartCancelsRunsThatOutlastTheDrainTimeout
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		if (peer.launchNumber == 1) {
			peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *old, NSDictionary *message, NSArray *descriptors) {
				// The run answers only when the host cancels it.
			}, NFKPluginMessageCancel: ^(NFKPluginTestPeer *old, NSDictionary *message, NSArray *descriptors) {
				[old replyCancelledTo:message];
			} };
		}
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	process.restartDrainTimeout = 0.3;
	NFKPluginBackend *backend = [self backendOfProcess:process];
	NFKInferenceJob *stuck = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"stuck"]];
	NSDictionary *run = [self.harness.peers.firstObject waitForMessageOfType:NFKPluginMessageRun timeout:5];
	XCTAssertNotNil(run);

	XCTestExpectation *restarted = [self expectationWithDescription:@"restarted"];
	[process restartWithManifest:NFKPluginTestManifest(@"2.0.0", 300) completionHandler:^(NSError *error) {
		XCTAssertNil(error);
		[restarted fulfill];
	}];
	[self waitForExpectations:@[ restarted ] timeout:10];
	[self waitForJob:stuck];

	NSDictionary *cancel = [self.harness.peers.firstObject receivedMessagesOfType:NFKPluginMessageCancel].firstObject;
	XCTAssertEqualObjects(cancel[@"id"], run[@"id"]);
	XCTAssertEqual(stuck.status, NFKInferenceJobStatusFailed);
	XCTAssertEqualObjects(stuck.error.domain, NSCocoaErrorDomain);
	XCTAssertEqual(stuck.error.code, NSUserCancelledError);
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
}

- (void)testARestartThatCannotLaunchFailsTheHeldRequestsWithItsReason
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.failsLaunch = peer.launchNumber > 1;
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	NFKPluginBackend *backend = [self backendOfProcess:process];

	XCTestExpectation *restarted = [self expectationWithDescription:@"restarted"];
	[process restartWithManifest:NFKPluginTestManifest(@"2.0.0", 300) completionHandler:^(NSError *restartError) {
		XCTAssertEqual(restartError.code, NFKPluginErrorLaunchFailed);
		[restarted fulfill];
	}];
	NFKInferenceJob *held = [backend submitInferenceJobForRequest:[self requestWithPrompt:@"held"]];
	[self waitForExpectations:@[ restarted ] timeout:10];
	[self waitForJob:held];
	XCTAssertEqual(held.error.code, NFKPluginErrorLaunchFailed);
	XCTAssertEqual(process.state, NFKPluginProcessStateFailed);
}

#pragma mark Idle exit

- (void)testAnIdleProcessExitsAfterItsInterval
{
	NFKPluginProcess *process = [self processWithIdleExit:0.3];
	XCTAssertEqual(process.idleExitInterval, 0.3);
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return process.state == NFKPluginProcessStateNotRunning;
	}));
	NFKPluginTestPeer *peer = self.harness.peers.firstObject;
	XCTAssertTrue([peer.receivedTypes containsObject:NFKPluginMessageShutdown]);
	XCTAssertTrue(peer.stopped);

	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertEqual(self.harness.peers.count, 2u, @"the next request starts the process again");
}

- (void)testMemoryPressureShortensTheIdleExit
{
	NFKPluginProcess *process = [self processWithIdleExit:60];
	process.shortestIdleExitInterval = 0.2;
	__block double pressure = 0;
	process.memoryPressureProvider = ^double {
		return pressure;
	};
	XCTAssertEqual(process.effectiveIdleExitInterval, 60);
	pressure = 0.8;
	XCTAssertEqualWithAccuracy(process.effectiveIdleExitInterval, 60 - (60 - 0.2) * 0.5, 1e-9, @"linear between the onset and full pressure");
	pressure = 1;
	XCTAssertEqualWithAccuracy(process.effectiveIdleExitInterval, 0.2, 1e-9);
	XCTAssertEqual(process.idleExitInterval, 60, @"the configured interval is unchanged");

	pressure = 0;
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	[NSThread sleepForTimeInterval:0.6];
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
	pressure = 1;
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		return process.state == NFKPluginProcessStateNotRunning;
	}));
}

#pragma mark Stop

- (void)testAStopDuringARunFailsTheRunWithStopped
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageRun: ^(NFKPluginTestPeer *running, NSDictionary *message, NSArray *descriptors) {
			// The run never answers.
		} };
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NFKPluginBackend *backend = [self backendOfProcess:process];
	__block NSError *runError = nil;
	XCTestExpectation *ran = [self expectationWithDescription:@"run returned"];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[self requestWithPrompt:@"long"] error:&error]);
		runError = error;
		[ran fulfill];
	});
	XCTAssertNotNil([self.harness.peers.firstObject waitForMessageOfType:NFKPluginMessageRun timeout:5]);

	XCTestExpectation *stopped = [self expectationWithDescription:@"stopped"];
	[process stopWithCompletionHandler:^{
		[stopped fulfill];
	}];
	[self waitForExpectations:@[ ran, stopped ] timeout:10];
	XCTAssertEqualObjects(runError.domain, NFKPluginErrorDomain);
	XCTAssertEqual(runError.code, NFKPluginErrorStopped);
	XCTAssertEqual(process.state, NFKPluginProcessStateNotRunning);
	XCTAssertTrue([self.harness.peers.firstObject.receivedTypes containsObject:NFKPluginMessageShutdown]);
	XCTAssertTrue(self.harness.peers.firstObject.stopped);
}

#pragma mark Models and tools

- (void)testABackendForAnUnknownModelFailsWithoutALaunch
{
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertNil([process backendForModelIdentifier:@"missing" options:@{} error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorUnknownModel);
	XCTAssertEqual(self.harness.peers.count, 0u);
}

- (void)testToolsComeFromTheManifestAndRunInThePlugin
{
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSArray<NFKInferenceTool *> *tools = process.tools;
	XCTAssertEqual(tools.count, 1u);
	XCTAssertEqualObjects(tools.firstObject.name, @"shout");
	XCTAssertEqualObjects(tools.firstObject.toolDescription, @"Shouts its text.");
	XCTAssertEqual(self.harness.peers.count, 0u, @"listing tools starts nothing");

	NFKToolResult *result = [tools.firstObject invokeWithArguments:@{ @"text": @"hi" }];
	XCTAssertFalse(result.isError, @"%@", result.text);
	XCTAssertEqualObjects(result.text, @"shout called");
	NSDictionary *call = [self.harness.peers.firstObject receivedMessagesOfType:NFKPluginMessageCallTool].firstObject;
	XCTAssertEqualObjects(call[@"tool"], @"shout");
	XCTAssertEqualObjects(call[@"arguments"], (@{ @"text": @"hi" }));
}

- (void)testAToolErrorKeepsItsDomainAndCode
{
	self.harness.configure = ^(NFKPluginTestPeer *peer) {
		peer.handlers = @{ NFKPluginMessageCallTool: ^(NFKPluginTestPeer *failing, NSDictionary *message, NSArray *descriptors) {
			[failing replyTo:message error:[NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownTool
														   userInfo:@{ NSLocalizedDescriptionKey: @"no such tool" }]];
		} };
	};
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSError *error = nil;
	XCTAssertNil([process.tools.firstObject resultForArguments:@{ @"text": @"hi" } error:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
	XCTAssertEqual(error.code, NFKPluginErrorUnknownTool);
	XCTAssertEqualObjects(error.localizedDescription, @"no such tool");
}

- (void)testThePluginsLogReachesTheLogHandler
{
	NFKPluginProcess *process = [self processWithIdleExit:300];
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	process.logHandler = ^(NFKPluginProcess *logging, NFKPluginLogLevel level, NSString *message) {
		@synchronized (lines) {
			[lines addObject:[NSString stringWithFormat:@"%ld %@", (long)level, message]];
		}
	};
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	[self.harness.peers.firstObject sendMessage:@{ @"type": NFKPluginMessageLog, @"level": @2, @"message": @"weights are old" }];
	XCTAssertTrue(NFKPluginTestWaitUntil(5, ^BOOL {
		@synchronized (lines) {
			return [lines containsObject:@"2 weights are old"];
		}
	}));
}

@end

#endif
