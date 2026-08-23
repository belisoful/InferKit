//
//  NFKAudioAssetTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKAudioAsset.h>

@interface NFKAudioAssetTests : XCTestCase
@end

@implementation NFKAudioAssetTests

- (void)testFileURLFactoryLeavesMetadataUnknown
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.wav"];
	NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:url];
	XCTAssertEqualObjects(asset.fileURL, url);
	XCTAssertEqual(asset.durationSeconds, 0.0);
	XCTAssertEqual(asset.sampleRate, 0.0);
	XCTAssertEqual(asset.channelCount, 0);
}

- (void)testFullFactoryCarriesMetadata
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.wav"];
	NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:url
											   durationSeconds:3.5
													sampleRate:44100.0
												  channelCount:2];
	XCTAssertEqual(asset.durationSeconds, 3.5);
	XCTAssertEqual(asset.sampleRate, 44100.0);
	XCTAssertEqual(asset.channelCount, 2);
}

- (void)testEqualityAndImmutableCopy
{
	NSURL *url = [NSURL fileURLWithPath:@"/clips/out.wav"];
	NFKAudioAsset *a = [NFKAudioAsset audioAssetWithFileURL:url durationSeconds:3.5 sampleRate:44100 channelCount:2];
	NFKAudioAsset *b = [NFKAudioAsset audioAssetWithFileURL:url durationSeconds:3.5 sampleRate:44100 channelCount:2];
	NFKAudioAsset *c = [NFKAudioAsset audioAssetWithFileURL:url durationSeconds:3.5 sampleRate:48000 channelCount:2];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c);
	XCTAssertEqual([a copy], a, @"an immutable value copies to itself");
}

@end
