//
//  NFKVideoAssetTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKVideoAsset.h>

@interface NFKVideoAssetTests : XCTestCase
@end

@implementation NFKVideoAssetTests

- (void)testFileURLFactoryLeavesMetadataUnknown
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.mp4"];
	NFKVideoAsset *asset = [NFKVideoAsset videoAssetWithFileURL:url];
	XCTAssertEqualObjects(asset.fileURL, url);
	XCTAssertEqual(asset.durationSeconds, 0.0);
	XCTAssertEqual(asset.framesPerSecond, 0.0);
	XCTAssertEqual(asset.dimensions.width, 0.0);
	XCTAssertEqual(asset.dimensions.height, 0.0);
}

- (void)testFullFactoryCarriesMetadata
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.mp4"];
	NFKVideoAsset *asset = [NFKVideoAsset videoAssetWithFileURL:url
											   durationSeconds:4.0
												framesPerSecond:24.0
													 dimensions:CGSizeMake(1920, 1080)];
	XCTAssertEqual(asset.durationSeconds, 4.0);
	XCTAssertEqual(asset.framesPerSecond, 24.0);
	XCTAssertEqual(asset.dimensions.width, 1920.0);
	XCTAssertEqual(asset.dimensions.height, 1080.0);
}

- (void)testEqualityAndImmutableCopy
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.mp4"];
	NFKVideoAsset *a = [NFKVideoAsset videoAssetWithFileURL:url durationSeconds:4 framesPerSecond:24 dimensions:CGSizeMake(64, 64)];
	NFKVideoAsset *b = [NFKVideoAsset videoAssetWithFileURL:url durationSeconds:4 framesPerSecond:24 dimensions:CGSizeMake(64, 64)];
	NFKVideoAsset *c = [NFKVideoAsset videoAssetWithFileURL:url durationSeconds:5 framesPerSecond:24 dimensions:CGSizeMake(64, 64)];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c);
	XCTAssertEqual([a copy], a, @"an immutable value copies to itself");
}

@end
