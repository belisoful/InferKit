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

/*! Everything readable from descriptor until every writer has closed it. */
- (NSString *)readToEndOfDescriptor:(int)descriptor
{
	NSMutableData *data = [NSMutableData data];
	uint8_t buffer[256];
	ssize_t count = 0;
	while ((count = read(descriptor, buffer, sizeof buffer)) > 0) {
		[data appendBytes:buffer length:(NSUInteger)count];
	}
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
	close(first[1]);
	close(second[1]);
	XCTAssertEqualObjects([self readToEndOfDescriptor:first[0]], @"three\n");
	XCTAssertEqualObjects([self readToEndOfDescriptor:second[0]], @"four\n");
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
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
	XCTAssertEqual([error.userInfo[NSUnderlyingErrorKey] code], EINVAL);

	close(channel[0]);
	close(channel[1]);
	// A number far above any open descriptor, so no other thread's open can reuse it.
	process.ownerInheritedDescriptors = @{ @3: @1000000 };
	XCTAssertFalse([process startWithError:&error], @"a source that is not open fails the launch");
	XCTAssertEqual([error.userInfo[NSUnderlyingErrorKey] code], EBADF);
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
