//
//  NFKCHeaderTests.m
//  InferKitTests
//
//  The C headers stay pure C99 and C++11: each compiles on its own under -pedantic -Werror in both
//  languages, a program using every declaration of NFKInferKitC.h compiles in both, and the Objective-C
//  umbrella imports neither.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"

@interface NFKCHeaderTests : XCTestCase
@end

@implementation NFKCHeaderTests

- (NSString *)pathInSourceTree:(NSString *)relativePath
{
	return [NFKPluginFixtures.sourceRootURL URLByAppendingPathComponent:relativePath].path;
}

- (NSArray<NSString *> *)strictCArguments
{
	return @[ @"clang", @"-x", @"c", @"-std=c99", @"-pedantic", @"-Wall", @"-Wextra", @"-Wstrict-prototypes", @"-Werror", @"-fsyntax-only" ];
}

- (NSArray<NSString *> *)strictCPlusPlusArguments
{
	return @[ @"clang++", @"-x", @"c++", @"-std=c++11", @"-pedantic", @"-Wall", @"-Wextra", @"-Werror", @"-fsyntax-only" ];
}

- (void)assertCompilesWithArguments:(NSArray<NSString *> *)arguments
{
	NSError *error = nil;
	NSString *output = [NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:arguments error:&error];
	XCTAssertNotNil(output, @"%@", error.localizedDescription);
	XCTAssertFalse([output containsString:@"warning:"], @"%@", output);
}

- (void)assertHeaderIsPure:(NSString *)name
{
	NSString *header = [self pathInSourceTree:[@"Sources/InferKitC/include/InferKitC/" stringByAppendingString:name]];
	[self assertCompilesWithArguments:[self.strictCArguments arrayByAddingObject:header]];
	[self assertCompilesWithArguments:[self.strictCPlusPlusArguments arrayByAddingObject:header]];
}

- (void)testThePluginABIHeaderIsPureC99AndCPlusPlus11
{
	[self assertHeaderIsPure:@"NFKPluginABI.h"];
}

- (void)testTheCAPIHeaderIsPureC99AndCPlusPlus11
{
	[self assertHeaderIsPure:@"NFKInferKitC.h"];
}

- (void)testEveryCAPIDeclarationCompilesInCAndCPlusPlus
{
	NSString *program = [self pathInSourceTree:@"Tests/PluginFixtures/headers/inferkit_c_usage.c"];
	NSArray<NSString *> *search = @[ @"-I", [self pathInSourceTree:@"Sources/InferKitC/include"], program ];
	[self assertCompilesWithArguments:[self.strictCArguments arrayByAddingObjectsFromArray:search]];
	[self assertCompilesWithArguments:[self.strictCPlusPlusArguments arrayByAddingObjectsFromArray:search]];
}

- (void)testTheObjectiveCUmbrellaImportsNeitherCHeader
{
	NSError *error = nil;
	NSString *umbrella = [NSString stringWithContentsOfFile:[self pathInSourceTree:@"Sources/InferKit/include/InferKit/InferKit.h"]
												   encoding:NSUTF8StringEncoding error:&error];
	XCTAssertNotNil(umbrella, @"%@", error);
	XCTAssertFalse([umbrella containsString:@"NFKPluginABI.h"]);
	XCTAssertFalse([umbrella containsString:@"NFKInferKitC.h"]);
}

@end

#endif
