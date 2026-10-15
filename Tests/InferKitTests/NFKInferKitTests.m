//
//  NFKInferKitTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferKit.h>

@interface NFKInferKitTests : XCTestCase
@end

@implementation NFKInferKitTests

- (void)testTheReleaseVersionIsANonEmptySemanticVersionString
{
	NSString *version = NFKInferKit.releaseVersion;
	XCTAssertGreaterThan(version.length, 0u, @"the version string is present");

	NSArray<NSString *> *components = [version componentsSeparatedByString:@"."];
	XCTAssertEqual(components.count, 3u, @"the version is major.minor.patch");
	for (NSString *component in components) {
		XCTAssertGreaterThan(component.length, 0u, @"each version component is present");
		NSCharacterSet *nonDigits = NSCharacterSet.decimalDigitCharacterSet.invertedSet;
		XCTAssertEqual([component rangeOfCharacterFromSet:nonDigits].location, NSNotFound,
					   @"each version component is numeric");
	}
}

- (void)testTheReleaseVersionIsStable
{
	XCTAssertEqualObjects(NFKInferKit.releaseVersion, NFKInferKit.releaseVersion,
						  @"the version is a constant, not recomputed differently");
}

- (void)testTheAPIVersionIsMajorDotMinor
{
	NSString *expected = [NSString stringWithFormat:@"%ld.%ld", (long)NFKInferKit.APIMajorVersion,
						  (long)NFKInferKit.APIMinorVersion];
	XCTAssertEqualObjects(NFKInferKit.APIVersion, expected);
}

- (void)testTheAPIVersionMatchesTheHeaders
{
	XCTAssertEqual(NFKInferKit.APIMajorVersion, NFK_API_VERSION_MAJOR);
	XCTAssertEqual(NFKInferKit.APIMinorVersion, NFK_API_VERSION_MINOR);
}

- (void)testTheReleaseVersionLeavesNSObjectClassVersioningAlone
{
	XCTAssertEqual([NSObject version], [NFKInferKit version],
				   @"+version stays NSObject's integer class version");
}

- (void)testTheAPIVersionStartsAtOneZero
{
	XCTAssertGreaterThanOrEqual(NFKInferKit.APIMajorVersion, 1);
	XCTAssertGreaterThanOrEqual(NFKInferKit.APIMinorVersion, 0);
}

@end
