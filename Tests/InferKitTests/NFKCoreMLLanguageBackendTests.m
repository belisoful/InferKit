//
//  NFKCoreMLLanguageBackendTests.m
//  NFKTests
//
//  Contract and load-ordering paths. A live generation run is host-verified against a converted
//  model, as NFKCoreMLBackend's live run is.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKCoreMLLanguageBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKErrors.h>

API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0))
@interface NFKCoreMLLanguageBackendTests : XCTestCase
@end

@implementation NFKCoreMLLanguageBackendTests

- (void)testTheBackendReportsItsIdentifier
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		XCTAssertEqualObjects(backend.backendIdentifier, @"coreml-llm");
	}
}

- (void)testANewBackendIsNotReady
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		XCTAssertFalse(backend.isReady);
	}
}

- (void)testPreparingWithoutADirectoryFails
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		NSError *error = nil;
		XCTAssertFalse([backend prepareWithError:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
	}
}

- (void)testLoadingADirectoryWithoutAManifestFails
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self makeTemporaryDirectory];
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		NSError *error = nil;
		XCTAssertFalse([backend loadModelFromDirectory:directory error:&error]);
		XCTAssertNotNil(error);
		XCTAssertFalse(backend.isReady);
	}
}

- (void)testAnUnsupportedTokenizerTypeFailsLoading
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self makeTemporaryDirectory];
		[self writeManifest:@{ @"tokenizer": @{ @"type": @"char-level", @"vocab": @"vocab.txt" } }
				toDirectory:directory];
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		NSError *error = nil;
		XCTAssertFalse([backend loadModelFromDirectory:directory error:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	}
}

- (void)testAValidTokenizerButMissingModelFailsAfterTheTokenizer
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self makeTemporaryDirectory];
		[self writeTokenizerFilesToDirectory:directory];
		[self writeManifest:@{
			@"tokenizer": @{ @"type": @"bpe-bytelevel", @"vocab": @"vocab.json", @"merges": @"merges.txt" },
		} toDirectory:directory];
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:nil];
		NSError *error = nil;
		XCTAssertFalse([backend loadModelFromDirectory:directory error:&error]);
		XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
	}
}

#pragma mark Helpers

- (NSURL *)makeTemporaryDirectory
{
	NSURL *directory = [[NSFileManager defaultManager].temporaryDirectory
		URLByAppendingPathComponent:[NSUUID UUID].UUIDString isDirectory:YES];
	[[NSFileManager defaultManager] createDirectoryAtURL:directory
							 withIntermediateDirectories:YES
											  attributes:nil
												   error:NULL];
	return directory;
}

- (void)writeManifest:(NSDictionary *)manifest toDirectory:(NSURL *)directory
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL];
	[data writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"] atomically:YES];
}

- (void)writeTokenizerFilesToDirectory:(NSURL *)directory
{
	NSData *vocab = [NSJSONSerialization dataWithJSONObject:@{ @"h": @0, @"i": @1 } options:0 error:NULL];
	[vocab writeToURL:[directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];
	[@"#version: 0.2\n" writeToURL:[directory URLByAppendingPathComponent:@"merges.txt"]
						atomically:YES
						  encoding:NSUTF8StringEncoding
							 error:NULL];
}

@end
