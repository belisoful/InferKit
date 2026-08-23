//
//  NFKHFHubTests.m
//  NFKTests
//
//  Covers URL construction, caching, and checksum through a stub transport. A live download
//  is integration-verified.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKHFHub.h>
#import <InferKit/NFKErrors.h>

/*! A hub whose transport writes staged bytes and counts fetches. */
@interface FxHFHubStub : NFKHFHub
@property (nonatomic, strong) NSData *stagedData;
@property (nonatomic, assign) NSUInteger fetchCount;
@end

@implementation FxHFHubStub

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError **)outError
{
	self.fetchCount += 1;
	return [self.stagedData writeToURL:destinationURL atomically:YES];
}

@end

@interface NFKHFHubTests : XCTestCase
@property (nonatomic, strong) FxHFHubStub *hub;
@property (nonatomic, strong) NSURL *cacheDir;
@end

@implementation NFKHFHubTests

- (void)setUp
{
	[super setUp];
	self.cacheDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:@"NFKHFHubTests"]];
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[NSFileManager.defaultManager createDirectoryAtURL:self.cacheDir withIntermediateDirectories:YES attributes:nil error:NULL];

	self.hub = [[FxHFHubStub alloc] init];
	self.hub.cacheDirectoryURL = self.cacheDir;
	self.hub.stagedData = [@"hello" dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[super tearDown];
}

- (void)testTheRemoteURLFollowsTheResolveLayout
{
	NSURL *url = [self.hub remoteURLForRepo:@"apple/coreml-sd" revision:nil path:@"unet/model.mlpackage"];
	XCTAssertEqualObjects(url.absoluteString,
						@"https://huggingface.co/apple/coreml-sd/resolve/main/unet/model.mlpackage");
}

- (void)testTheLocalURLMirrorsTheRepoLayout
{
	NSURL *url = [self.hub localURLForRepo:@"apple/coreml-sd" revision:@"v2" path:@"unet/model.bin"];
	NSURL *expected = [[[[self.cacheDir URLByAppendingPathComponent:@"apple"]
						 URLByAppendingPathComponent:@"coreml-sd"]
						URLByAppendingPathComponent:@"v2"]
					   URLByAppendingPathComponent:@"unet/model.bin"];
	XCTAssertEqualObjects(url.path, expected.path);
}

- (void)testDownloadingWithoutACacheDirectoryFails
{
	self.hub.cacheDirectoryURL = nil;
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testDownloadingCachesTheFileAndReturnsItsURL
{
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:&error];
	XCTAssertNil(error);
	XCTAssertNotNil(result);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:result.path]);
	XCTAssertTrue([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
	XCTAssertEqual(self.hub.fetchCount, (NSUInteger)1);
}

- (void)testASecondDownloadSkipsTheFetch
{
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertEqual(self.hub.fetchCount, (NSUInteger)1, @"the cached file is reused");
}

- (void)testAChecksumMatchSucceeds
{
	// SHA-256 of "hello".
	NSString *sha = @"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:sha error:&error];
	XCTAssertNotNil(result);
	XCTAssertNil(error);
}

- (void)testAChecksumMismatchFailsAndDoesNotCache
{
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:@"deadbeef" error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceBackendFailure);
	XCTAssertFalse([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
}

- (void)testTheDefaultCacheDirectoryIsUnderApplicationSupport
{
	NSURL *directory = [NFKHFHub defaultCacheDirectoryURL];
	XCTAssertTrue([directory.path containsString:@"InferKit"]);
	XCTAssertTrue([directory.path hasSuffix:@"models"]);
	BOOL isDirectory = NO;
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:directory.path isDirectory:&isDirectory]);
	XCTAssertTrue(isDirectory, @"the default cache directory is created on demand");
}

- (void)testTheAsynchronousDownloadDeliversTheLocalURL
{
	XCTestExpectation *done = [self expectationWithDescription:@"async download"];
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil
		 completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNotNil(localURL);
		XCTAssertNil(error);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:5];
	XCTAssertTrue([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
}

@end
