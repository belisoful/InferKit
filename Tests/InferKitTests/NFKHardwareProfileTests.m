//
//  NFKHardwareProfileTests.m
//  NFKTests
//
//  The machine, as the toolkit sees it. Every reading here is supposed to degrade rather than throw,
//  so most of what these assert is that a missing value is a zero or an empty string and not a crash
//  — the failure mode that matters on a machine this was never run on.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKHardwareProfile.h>

@interface NFKHardwareProfileTests : XCTestCase
@end

@implementation NFKHardwareProfileTests

- (void)testTheProfileIsSharedAndReadOnce
{
	XCTAssertNotNil(NFKHardwareProfile.currentProfile);
	XCTAssertEqual(NFKHardwareProfile.currentProfile, NFKHardwareProfile.currentProfile,
				   @"the static readings are taken once");
}

- (void)testTheMachineReportsItsMemory
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	XCTAssertGreaterThan(profile.physicalMemory, 0, @"every machine reports its RAM");
}

// Metal's recommendation is the budget to size against, and it sits below the physical total. A
// reading equal to physical memory would mean the wrong property was read.
- (void)testTheRecommendedWorkingSetIsBelowThePhysicalTotal
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	if (profile.recommendedWorkingSetSize == 0) {
		return;		// no Metal device on this machine
	}
	XCTAssertLessThan(profile.recommendedWorkingSetSize, profile.physicalMemory);
	XCTAssertGreaterThan(profile.recommendedWorkingSetSize, 0);
}

// The largest single allocation is a SEPARATE ceiling from the total: a tensor cannot exceed it
// however much of the budget is unspent.
- (void)testTheMaximumBufferIsItsOwnCeiling
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	if (profile.maximumBufferLength == 0) {
		return;		// no Metal device
	}
	XCTAssertGreaterThan(profile.maximumBufferLength, 0);
	XCTAssertLessThanOrEqual(profile.maximumBufferLength, profile.physicalMemory);
}

- (void)testAvailableMemoryIsPositiveAndBoundedByTheMachine
{
	NSInteger available = [NFKHardwareProfile availableMemory];
	XCTAssertGreaterThan(available, 0, @"something is free, or the reading failed");
	XCTAssertLessThanOrEqual(available, NFKHardwareProfile.currentProfile.physicalMemory);
}

// It is live, so two readings need not agree — but both must be plausible. This asserts the contract
// rather than a value, since another process allocating between the two is normal.
- (void)testAvailableMemoryIsReadFresh
{
	NSInteger first = [NFKHardwareProfile availableMemory];
	NSInteger second = [NFKHardwareProfile availableMemory];
	XCTAssertGreaterThan(first, 0);
	XCTAssertGreaterThan(second, 0);
}

// An unknown chip reports an empty name rather than nil, so a caller formatting it cannot crash.
- (void)testTheStringsAreNeverNil
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	XCTAssertNotNil(profile.chipName);
	XCTAssertNotNil(profile.modelIdentifier);
	XCTAssertNotNil(profile.graphicsArchitecture);
	XCTAssertNotNil(profile.describedMachine);
	XCTAssertGreaterThan(profile.describedMachine.length, 0);
}

// Apple Silicon reports two performance levels; an Intel Mac reports one and both counts stay zero
// rather than inventing a split. Either way they cannot be negative or exceed the machine.
- (void)testTheCoreCountsAreConsistent
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	XCTAssertGreaterThanOrEqual(profile.performanceCoreCount, 0);
	XCTAssertGreaterThanOrEqual(profile.efficiencyCoreCount, 0);
	if (profile.performanceCoreCount > 0) {
		XCTAssertLessThanOrEqual(profile.performanceCoreCount + profile.efficiencyCoreCount,
								 (NSInteger)NSProcessInfo.processInfo.processorCount);
	}
}

@end
