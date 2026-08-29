//
//  NFKInferKitTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferKit.h>

@interface NFKInferKitTests : XCTestCase
@end

@implementation NFKInferKitTests

- (void)testTheVersionIsANonEmptySemanticVersionString
{
	NSString *version = NFKInferKit.version;
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

- (void)testTheVersionIsStable
{
	XCTAssertEqualObjects(NFKInferKit.version, NFKInferKit.version,
						  @"the version is a constant, not recomputed differently");
}

@end
