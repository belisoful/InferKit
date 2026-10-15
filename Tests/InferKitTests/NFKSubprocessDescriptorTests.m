//
//  NFKSubprocessDescriptorTests.m
//  InferKitTests
//
//  The owner hooks a plugin host uses: descriptors a child inherits beyond its standard streams, and
//  the quality of service it starts at.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "../../Sources/InferKit/NFKSubprocessPrivate.h"
#import <libproc.h>
#import <poll.h>
#import <sys/proc_info.h>
#import <unistd.h>

@interface NFKSubprocessDescriptorTests : XCTestCase
@end

@implementation NFKSubprocessDescriptorTests

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

/*! Everything readable from descriptor until every writer has closed it, or nil when a writer is
	still open after 10 seconds. */
- (nullable NSString *)readToEndOfDescriptor:(int)descriptor
{
	NSMutableData *data = [NSMutableData data];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	uint8_t buffer[256];
	ssize_t count = 0;
	do {
		struct pollfd readable = { .fd = descriptor, .events = POLLIN };
		int remaining = (int)MAX(0, deadline.timeIntervalSinceNow * 1000);
		if (poll(&readable, 1, remaining) <= 0) {
			close(descriptor);
			return nil;
		}
		count = read(descriptor, buffer, sizeof buffer);
		if (count > 0) {
			[data appendBytes:buffer length:(NSUInteger)count];
		}
	} while (count > 0);
	close(descriptor);
	return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

- (void)testAChildWritesToTheDescriptorItInherits
{
	int channel[2];
	XCTAssertEqual(pipe(channel), 0);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"echo ok >&3" ]];
	process.ownerInheritedDescriptors = @{ @3: @(channel[1]) };
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	process.ownerInheritedDescriptors = nil;
	close(channel[1]);
	XCTAssertEqualObjects([self readToEndOfDescriptor:channel[0]], @"ok\n");
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
	XCTAssertEqual(process.terminationStatus, 0, @"descriptor 3 was open in the child");
}

- (void)testInheritedDescriptorsLandOnTheirOwnNumbers
{
	int first[2];
	int second[2];
	XCTAssertEqual(pipe(first), 0);
	XCTAssertEqual(pipe(second), 0);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"echo three >&3; echo four >&4" ]];
	process.ownerInheritedDescriptors = @{ @3: @(first[1]), @4: @(second[1]) };
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	process.ownerInheritedDescriptors = nil;
	close(first[1]);
	close(second[1]);
	XCTAssertEqualObjects([self readToEndOfDescriptor:first[0]], @"three\n");
	XCTAssertEqualObjects([self readToEndOfDescriptor:second[0]], @"four\n");
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
}

- (void)testTheSubprocessOwnsItsCopiesOfInheritedDescriptors
{
	int channel[2];
	XCTAssertEqual(pipe(channel), 0);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"echo ok >&3" ]];
	process.ownerInheritedDescriptors = @{ @3: @(channel[1]) };
	close(channel[1]);
	XCTAssertNotEqualObjects(process.ownerInheritedDescriptors[@3], @(channel[1]), @"the subprocess holds a copy of its own");
	NSError *error = nil;
	for (NSUInteger launch = 0; launch < 2; launch++) {
		XCTAssertTrue([process startWithError:&error], @"a launch after the caller closed its descriptor, %@", error);
		XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
		XCTAssertEqual(process.terminationStatus, 0);
	}
	process.ownerInheritedDescriptors = nil;
	XCTAssertEqualObjects([self readToEndOfDescriptor:channel[0]], @"ok\nok\n", @"each launch wrote through its copy, and clearing the property closed the last one");
}

- (void)testOnlyTheNamedDescriptorsReachTheChild
{
	int stray[2];
	XCTAssertEqual(pipe(stray), 0);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"echo leaked >&3" ]];
	NSError *error = nil;
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	close(stray[1]);
	XCTAssertEqualObjects([self readToEndOfDescriptor:stray[0]], @"", @"a descriptor no one named stays out of the child");
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
	XCTAssertNotEqual(process.terminationStatus, 0, @"the shell finds descriptor 3 closed");
}

- (void)testAnInvalidInheritedDescriptorFailsTheLaunch
{
	int channel[2];
	XCTAssertEqual(pipe(channel), 0);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"true" ]];
	process.ownerInheritedDescriptors = @{ @2: @(channel[1]) };
	NSError *error = nil;
	XCTAssertFalse([process startWithError:&error], @"a standard stream is not an inherited descriptor");
	XCTAssertEqual(error.code, NFKSubprocessErrorLaunchFailed);
	XCTAssertEqual(((NSError *)error.userInfo[NSUnderlyingErrorKey]).code, EINVAL);

	close(channel[0]);
	close(channel[1]);
	// A number far above any open descriptor, so no other thread's open can reuse it.
	process.ownerInheritedDescriptors = @{ @3: @1000000 };
	XCTAssertFalse([process startWithError:&error], @"a source that is not open fails the launch");
	XCTAssertEqual(((NSError *)error.userInfo[NSUnderlyingErrorKey]).code, EBADF);
}

- (void)testTheChildStartsAtTheRequestedQualityOfService
{
	NFKSubprocess *background = [[NFKSubprocess alloc] initWithCommand:@"/bin/sleep" arguments:@[ @"5" ]];
	background.ownerQualityOfService = QOS_CLASS_BACKGROUND;
	NFKSubprocess *unspecified = [[NFKSubprocess alloc] initWithCommand:@"/bin/sleep" arguments:@[ @"5" ]];
	NSError *error = nil;
	XCTAssertTrue([background startWithError:&error], @"%@", error);
	XCTAssertTrue([unspecified startWithError:&error], @"%@", error);

	struct proc_taskinfo backgroundInfo = { 0 };
	struct proc_taskinfo unspecifiedInfo = { 0 };
	XCTAssertEqual(proc_pidinfo(background.processIdentifier, PROC_PIDTASKINFO, 0, &backgroundInfo, sizeof backgroundInfo),
				   (int)sizeof backgroundInfo);
	XCTAssertEqual(proc_pidinfo(unspecified.processIdentifier, PROC_PIDTASKINFO, 0, &unspecifiedInfo, sizeof unspecifiedInfo),
				   (int)sizeof unspecifiedInfo);
	// The background class runs at base priority 4; a child without a class starts at the default 31.
	XCTAssertLessThan(backgroundInfo.pti_priority, unspecifiedInfo.pti_priority);
	XCTAssertLessThanOrEqual(backgroundInfo.pti_priority, 4);
	[background stop];
	[unspecified stop];
}

@end

#endif
