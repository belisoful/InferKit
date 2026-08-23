//
//  NFKUnigramTokenizerTests.m
//  NFKTests
//
//  Viterbi unigram segmentation, ▁-space normalization, and byte fallback against a small vocab.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTokenizer.h>

@interface NFKUnigramTokenizerTests : XCTestCase
@end

@implementation NFKUnigramTokenizerTests
{
	NSURL *_directory;
	NFKTokenizer *_tokenizer;
}

- (void)setUp
{
	[super setUp];

	_directory = [[NSFileManager defaultManager].temporaryDirectory
		URLByAppendingPathComponent:[NSUUID UUID].UUIDString isDirectory:YES];
	[[NSFileManager defaultManager] createDirectoryAtURL:_directory
							 withIntermediateDirectories:YES
											  attributes:nil
												   error:NULL];

	// id order defines the token ids. The two <0xHH> pieces are the UTF-8 bytes of "±" (U+00B1).
	NSDictionary *vocabFile = @{
		@"vocab": @[
			@[@"<unk>", @0.0],
			@[@"▁", @(-5.0)],
			@[@"▁hello", @(-1.0)],
			@[@"▁world", @(-1.0)],
			@[@"h", @(-6.0)], @[@"e", @(-6.0)], @[@"l", @(-6.0)], @[@"o", @(-6.0)],
			@[@"<0xC2>", @(-10.0)], @[@"<0xB1>", @(-10.0)],
		],
		@"unkId": @0,
		@"byteFallback": @YES,
		@"addDummyPrefix": @YES,
	};
	NSData *data = [NSJSONSerialization dataWithJSONObject:vocabFile options:0 error:NULL];
	[data writeToURL:[_directory URLByAppendingPathComponent:@"unigram.json"] atomically:YES];

	NSDictionary *manifest = @{
		@"eosTokenId": @2,
		@"tokenizer": @{ @"type": @"unigram", @"vocab": @"unigram.json" },
	};
	NSError *error = nil;
	_tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:_directory error:&error];
	XCTAssertNotNil(_tokenizer, @"%@", error);
}

- (void)tearDown
{
	[[NSFileManager defaultManager] removeItemAtURL:_directory error:NULL];
	[super tearDown];
}

- (void)testViterbiPrefersTheHighestScoringSegmentation
{
	XCTAssertEqualObjects([_tokenizer encode:@"hello world"], (@[@2, @3]));
}

- (void)testDecodeRestoresSpacingAndDropsTheDummyPrefix
{
	XCTAssertEqualObjects(([_tokenizer decode:@[@2, @3]]), @"hello world");
	XCTAssertEqualObjects([_tokenizer decode:[_tokenizer encode:@"hello world"]], @"hello world");
}

- (void)testByteFallbackCoversAnUnknownCharacter
{
	// "▁" (id 1), then "±" has no piece, so it falls back to its UTF-8 bytes <0xC2><0xB1> (ids 8, 9).
	XCTAssertEqualObjects([_tokenizer encode:@"±"], (@[@1, @8, @9]));
	XCTAssertEqualObjects([_tokenizer decode:[_tokenizer encode:@"±"]], @"±");
}

- (void)testTheFactoryReadsTheEndOfSequenceId
{
	XCTAssertEqual(_tokenizer.eosTokenId, 2);
}

@end
