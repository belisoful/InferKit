//
//  NFKSubprocessTests.m
//  InferKitTests
//
//  Launching and supervising a program: line framing, a write larger than the pipe, the restart policy's
//  limit, and the reaping of many short-lived children started at once. The programs are /bin/sh, cat,
//  and codesign, which ship with macOS. The stdio MCP client's tests cover the stop escalation, the
//  environment, and the working directory.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKSubprocessTests : XCTestCase
@end

@implementation NFKSubprocessTests

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow < 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.02];
	}
	return YES;
}

- (void)testASubprocessFramesItsOutputIntoLines
{
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh"
														  arguments:@[ @"-c", @"printf 'one\\r\\ntw'; printf 'o\\n\\nthree'" ]];
	process.restartPolicy = NFKSubprocessRestartPolicyOnCrash;
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	process.lineHandler = ^(NSData *line) {
		@synchronized (lines) {
			[lines addObject:[[NSString alloc] initWithData:line encoding:NSUTF8StringEncoding]];
		}
	};
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
	NSArray *received = nil;
	@synchronized (lines) {
		received = [lines copy];
	}
	NSArray *expected = @[ @"one", @"two", @"", @"three" ];
	XCTAssertEqualObjects(received, expected, @"output left without a newline arrives as a final line before the exit");
	XCTAssertEqual(process.terminationStatus, 0);
	XCTAssertFalse(process.isRestartPending, @"a clean exit is not a crash");
	XCTAssertEqual(process.restartCount, 0u);
}

- (void)testAWriteLargerThanThePipeWaitsForTheReader
{
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"cat" arguments:@[]];
	NSMutableData *echoed = [NSMutableData data];
	process.lineHandler = ^(NSData *line) {
		@synchronized (echoed) {
			[echoed appendData:line];
		}
	};
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"cat is found on PATH, %@", error);
	NSMutableData *line = [NSMutableData dataWithLength:1024 * 1024];
	memset(line.mutableBytes, 'x', line.length);
	XCTAssertTrue([process writeLine:line error:&error], @"%@", error);
	[process stop];
	XCTAssertEqual(process.terminationReason, NFKSubprocessTerminationReasonExit);
	XCTAssertEqual(process.terminationStatus, 0, @"cat exits on its input closing");
	@synchronized (echoed) {
		XCTAssertEqualObjects(echoed, line);
	}
	XCTAssertFalse([process writeData:line error:&error]);
	XCTAssertEqualObjects(error.domain, NFKSubprocessErrorDomain);
	XCTAssertEqual(error.code, NFKSubprocessErrorNotRunning);
}

- (void)testEveryShortLivedChildIsReapedWhenManyStartAtOnce
{
	// The kernel can report a child's exit before waitpid sees it. Before the subprocess retried the reap,
	// about 1 short codesign run in 800, started on several threads at once, stayed Running for good: 600
	// runs here catch that about half the time. With the retry, no run may stay.
	NSInteger threads = 6;
	NSInteger runsPerThread = 100;
	NSURL *folder = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					 [NSString stringWithFormat:@"subprocess-reap-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:folder withIntermediateDirectories:YES attributes:nil error:NULL]);
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:folder error:NULL];
	}];
	__block NSInteger stuck = 0;
	__block NSInteger failed = 0;
	dispatch_group_t group = dispatch_group_create();
	for (NSInteger thread = 0; thread < threads; thread++) {
		NSString *program = [folder URLByAppendingPathComponent:[NSString stringWithFormat:@"ls-%ld", (long)thread]].path;
		XCTAssertTrue([NSFileManager.defaultManager copyItemAtPath:@"/bin/ls" toPath:program error:NULL]);
		dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			for (NSInteger run = 0; run < runsPerThread; run++) {
				@autoreleasepool {
					NSArray<NSString *> *arguments = run % 2 == 0 ? @[ @"--force", @"--sign", @"-", program ] : @[ @"--verify", program ];
					NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/usr/bin/codesign" arguments:arguments];
					dispatch_semaphore_t ended = dispatch_semaphore_create(0);
					process.stateHandler = ^(NFKSubprocess *subprocess, NFKSubprocessState state) {
						if (state == NFKSubprocessStateExited || state == NFKSubprocessStateFailedToLaunch) {
							dispatch_semaphore_signal(ended);
						}
					};
					if (![process startWithError:NULL]) {
						@synchronized (self) {
							failed += 1;
						}
						continue;
					}
					if (dispatch_semaphore_wait(ended, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0) {
						@synchronized (self) {
							stuck += 1;
						}
						[process stop];
					}
				}
			}
		});
	}
	XCTAssertEqual(dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 600 * NSEC_PER_SEC)), 0, @"the runs end");
	@synchronized (self) {
		XCTAssertEqual(failed, 0);
		XCTAssertEqual(stuck, 0, @"every exited child is reaped and reported");
	}
}

- (void)testACrashingSubprocessRestartsUpToItsLimit
{
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"exit 7" ]];
	process.restartPolicy = NFKSubprocessRestartPolicyOnCrash;
	process.maximumRestarts = 2;
	process.restartDelay = 0.05;
	NSObject *guard = [[NSObject alloc] init];
	__block NSUInteger exits = 0;
	__block BOOL pendingAtLastExit = YES;
	process.stateHandler = ^(NFKSubprocess *subprocess, NFKSubprocessState state) {
		if (state == NFKSubprocessStateExited) {
			@synchronized (guard) {
				exits += 1;
				pendingAtLastExit = subprocess.isRestartPending;
			}
		}
	};
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL {
		@synchronized (guard) {
			return exits == 3;
		}
	} timeout:5], @"the first run and two restarts");
	[NSThread sleepForTimeInterval:0.3];
	@synchronized (guard) {
		XCTAssertEqual(exits, 3u, @"no restart past maximumRestarts");
		XCTAssertFalse(pendingAtLastExit);
	}
	XCTAssertEqual(process.restartCount, 2u);
	XCTAssertEqual(process.terminationStatus, 7);
	XCTAssertEqual(process.state, NFKSubprocessStateExited);
}

@end
