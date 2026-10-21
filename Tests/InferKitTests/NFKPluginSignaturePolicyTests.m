//
//  NFKPluginSignaturePolicyTests.m
//  InferKitTests
//
//  Which signatures a host accepts: ad hoc under the development policy only, a Team ID pin, a
//  requirement, a manifest edited after signing, unsigned code, the cache that a later edit misses, and
//  the running process checked by its audit token. The fixtures are Tests/PluginFixtures, signed ad hoc.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"
#import <mach/mach.h>
#import <Security/Security.h>

@interface NFKPluginSignaturePolicy (NFKTeamProof)
+ (nullable NSString *)reasonCodeDoesNotProveTeam:(NSString *)team code:(SecStaticCodeRef)code;
@end

@interface NFKPluginSignaturePolicyTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginSignaturePolicyTests

- (void)setUp
{
	[super setUp];
	self.temporaryURLs = [NSMutableArray array];
}

- (void)tearDown
{
	for (NSURL *url in self.temporaryURLs) {
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}
	[super tearDown];
}

#pragma mark Helpers

- (NSURL *)objectiveCFixture
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindObjC version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	return bundle;
}

/*! A copy of the fixture in a directory of its own, so editing it touches no other test's bundle. */
- (NSURL *)copyOfFixture
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[NSString stringWithFormat:@"signature-policy-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSURL *fixture = [self objectiveCFixture];
	NSURL *copy = [directory URLByAppendingPathComponent:fixture.lastPathComponent isDirectory:YES];
	NSError *error = nil;
	XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:fixture toURL:copy error:&error], @"%@", error);
	return copy;
}

- (void)appendToManifestOfBundle:(NSURL *)bundle
{
	NSURL *manifest = [bundle URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"];
	NSFileHandle *handle = [NSFileHandle fileHandleForWritingToURL:manifest error:NULL];
	XCTAssertNotNil(handle);
	[handle seekToEndOfFile];
	[handle writeData:[@"\n" dataUsingEncoding:NSUTF8StringEncoding]];
	[handle closeFile];
}

- (void)assertError:(NSError *)error code:(NFKPluginError)code
{
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain, @"%@", error);
	XCTAssertEqual(error.code, code, @"%@", error);
}

#pragma mark Policies

- (void)testTheDefaultPolicyRequiresASignatureAndRefusesAdHoc
{
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.defaultPolicy;
	XCTAssertTrue(policy.requiresValidSignature);
	XCTAssertFalse(policy.allowsAdHocSignature);
	XCTAssertNil(policy.teamIdentifier);
	XCTAssertNil(policy.requirementString);
	XCTAssertTrue(NFKPluginSignaturePolicy.developmentPolicy.allowsAdHocSignature);
	XCTAssertTrue(NFKPluginSignaturePolicy.developmentPolicy.requiresValidSignature);
}

- (void)testACopyKeepsEveryField
{
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
	policy.teamIdentifier = @"ABCDE12345";
	policy.requirementString = @"identifier \"com.example\"";
	policy.requiresValidSignature = NO;
	NFKPluginSignaturePolicy *copy = [policy copy];
	XCTAssertEqualObjects(copy.teamIdentifier, @"ABCDE12345");
	XCTAssertEqualObjects(copy.requirementString, @"identifier \"com.example\"");
	XCTAssertTrue(copy.allowsAdHocSignature);
	XCTAssertFalse(copy.requiresValidSignature);
}

#pragma mark Code on disk

- (void)testPoliciesWithTheSameDemandsAreEqual
{
	XCTAssertEqualObjects(NFKPluginSignaturePolicy.defaultPolicy, NFKPluginSignaturePolicy.defaultPolicy);
	XCTAssertEqual(NFKPluginSignaturePolicy.defaultPolicy.hash, NFKPluginSignaturePolicy.defaultPolicy.hash);
	XCTAssertNotEqualObjects(NFKPluginSignaturePolicy.defaultPolicy, NFKPluginSignaturePolicy.developmentPolicy);

	NFKPluginSignaturePolicy *pinned = NFKPluginSignaturePolicy.developmentPolicy;
	pinned.teamIdentifier = @"ABCDE12345";
	pinned.requirementString = @"identifier \"com.example\"";
	NFKPluginSignaturePolicy *copy = [pinned copy];
	XCTAssertEqualObjects(copy, pinned);
	XCTAssertEqual(copy.hash, pinned.hash);
	copy.teamIdentifier = @"ZZZZZ99999";
	XCTAssertNotEqualObjects(copy, pinned, @"another team");
	copy.teamIdentifier = pinned.teamIdentifier;
	copy.requirementString = nil;
	XCTAssertNotEqualObjects(copy, pinned, @"no requirement");
	copy.requirementString = pinned.requirementString;
	copy.requiresValidSignature = NO;
	XCTAssertNotEqualObjects(copy, pinned, @"signatures waived");
	XCTAssertNotEqualObjects(pinned, (id)@"policy");
}

- (void)testTheDevelopmentPolicyAcceptsAnAdHocSignature
{
	NSError *error = nil;
	XCTAssertTrue([NFKPluginSignaturePolicy.developmentPolicy validateCodeAtURL:[self objectiveCFixture] error:&error], @"%@", error);
}

- (void)testTheDefaultPolicyRefusesAnAdHocSignature
{
	NSError *error = nil;
	XCTAssertFalse([NFKPluginSignaturePolicy.defaultPolicy validateCodeAtURL:[self objectiveCFixture] error:&error]);
	[self assertError:error code:NFKPluginErrorSignatureInvalid];
	XCTAssertTrue([error.localizedDescription containsString:@"ad hoc"], @"%@", error);
}

- (void)testATeamIdentifierPinRefusesAnotherSigner
{
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
	policy.teamIdentifier = @"ABCDE12345";
	NSError *error = nil;
	XCTAssertFalse([policy validateCodeAtURL:[self objectiveCFixture] error:&error]);
	[self assertError:error code:NFKPluginErrorTeamMismatch];
}

- (void)testATeamIdentifierThatIsNotOneRefusesEverySigner
{
	NSArray<NSString *> *malformed = @[ @"ABCDE1234", @"abcde12345", @"ABCDE12345X", @"ABCDE\" or \"1", @"ABCDE)(123" ];
	for (NSString *team in malformed) {
		NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
		policy.teamIdentifier = team;
		NSError *error = nil;
		XCTAssertFalse([policy validateCodeAtURL:[self objectiveCFixture] error:&error], @"%@", team);
		[self assertError:error code:NFKPluginErrorTeamMismatch];
		XCTAssertTrue([error.localizedDescription containsString:@"ten uppercase letters and digits"], @"%@: %@", team, error);
	}
}

- (void)testARequirementDecidesWhichSignerIsAllowed
{
	NSURL *fixture = [self objectiveCFixture];
	NSString *identifier = [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindObjC];
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
	policy.requirementString = [NSString stringWithFormat:@"identifier \"%@\"", identifier];
	NSError *error = nil;
	XCTAssertTrue([policy validateCodeAtURL:fixture error:&error], @"%@", error);

	policy.requirementString = @"identifier \"com.example.someone-else\"";
	XCTAssertFalse([policy validateCodeAtURL:fixture error:&error]);
	[self assertError:error code:NFKPluginErrorTeamMismatch];

	policy.requirementString = @"this is not the requirement language";
	XCTAssertFalse([policy validateCodeAtURL:fixture error:&error]);
	[self assertError:error code:NFKPluginErrorSignatureInvalid];
}

- (void)testAManifestEditedAfterSigningIsRefused
{
	NSError *error = nil;
	NSURL *tampered = [NFKPluginFixtures tamperedCopyOfBundleAtURL:[self objectiveCFixture] error:&error];
	XCTAssertNotNil(tampered, @"%@", error);
	[self.temporaryURLs addObject:tampered.URLByDeletingLastPathComponent];
	XCTAssertFalse([NFKPluginSignaturePolicy.developmentPolicy validateCodeAtURL:tampered error:&error]);
	[self assertError:error code:NFKPluginErrorSignatureInvalid];
}

- (void)testAnEditAfterAValidationMissesTheCache
{
	NSURL *copy = [self copyOfFixture];
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
	NSError *error = nil;
	XCTAssertTrue([policy validateCodeAtURL:copy error:&error], @"%@", error);
	XCTAssertTrue([policy validateCodeAtURL:copy error:&error], @"the cached success answers again: %@", error);

	[self appendToManifestOfBundle:copy];
	XCTAssertFalse([policy validateCodeAtURL:copy error:&error], @"the edit changes the bundle's fingerprint");
	[self assertError:error code:NFKPluginErrorSignatureInvalid];
}

- (void)testACachedSuccessDoesNotAnswerAStricterPolicy
{
	NSURL *copy = [self copyOfFixture];
	NSError *error = nil;
	XCTAssertTrue([NFKPluginSignaturePolicy.developmentPolicy validateCodeAtURL:copy error:&error], @"%@", error);
	XCTAssertFalse([NFKPluginSignaturePolicy.defaultPolicy validateCodeAtURL:copy error:&error]);
	[self assertError:error code:NFKPluginErrorSignatureInvalid];
}

- (void)testUnsignedCodeIsRefusedUnlessThePolicyWaivesSignatures
{
	NSURL *script = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					 [NSString stringWithFormat:@"unsigned-%@.sh", NSUUID.UUID.UUIDString]];
	[self.temporaryURLs addObject:script];
	XCTAssertTrue([@"#!/bin/sh\nexit 0\n" writeToURL:script atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NSError *error = nil;
	XCTAssertFalse([NFKPluginSignaturePolicy.developmentPolicy validateCodeAtURL:script error:&error]);
	[self assertError:error code:NFKPluginErrorSignatureInvalid];

	NFKPluginSignaturePolicy *waived = NFKPluginSignaturePolicy.developmentPolicy;
	waived.requiresValidSignature = NO;
	XCTAssertTrue([waived validateCodeAtURL:script error:&error], @"%@", error);

	waived.teamIdentifier = @"ABCDE12345";
	XCTAssertFalse([waived validateCodeAtURL:script error:&error], @"a team pin needs a signature to check");
	[self assertError:error code:NFKPluginErrorTeamMismatch];
}

#pragma mark Running code

- (void)testTheRunningProcessIsCheckedByItsAuditToken
{
	audit_token_t token;
	mach_msg_type_number_t count = TASK_AUDIT_TOKEN_COUNT;
	XCTAssertEqual(task_info(mach_task_self(), TASK_AUDIT_TOKEN, (task_info_t)&token, &count), KERN_SUCCESS);
	NSError *error = nil;
	XCTAssertTrue([NFKPluginSignaturePolicy.developmentPolicy validateProcessWithAuditToken:token error:&error], @"%@", error);

	NFKPluginSignaturePolicy *pinned = NFKPluginSignaturePolicy.developmentPolicy;
	pinned.teamIdentifier = @"ABCDE12345";
	XCTAssertFalse([pinned validateProcessWithAuditToken:token error:&error]);
	[self assertError:error code:NFKPluginErrorTeamMismatch];
}

// A team the signer claims enters a requirement's text only as ten uppercase letters and digits, so a
// claimed team that carries requirement syntax is refused before any requirement is built from it.
- (void)testAClaimedTeamWithRequirementSyntaxIsNotProven
{
	SecStaticCodeRef code = NULL;
	XCTAssertEqual(SecStaticCodeCreateWithPath((__bridge CFURLRef)NSBundle.mainBundle.executableURL, kSecCSDefaultFlags, &code),
				   errSecSuccess);
	NSString *injected = [NFKPluginSignaturePolicy reasonCodeDoesNotProveTeam:@"X\" or always or info[a] = \"b" code:code];
	XCTAssertTrue([injected containsString:@"ten uppercase letters and digits"], @"%@", injected);
	XCTAssertNotNil([NFKPluginSignaturePolicy reasonCodeDoesNotProveTeam:@"abcde12345" code:code]);
	CFRelease(code);
}

@end

#endif
