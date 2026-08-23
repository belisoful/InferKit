//
//  NFKWordPieceTokenizerTests.m
//  NFKTests
//
//  Basic tokenization, greedy longest-match subwording, and the unknown fallback.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTokenizer.h>

@interface NFKWordPieceTokenizerTests : XCTestCase
@end

@implementation NFKWordPieceTokenizerTests
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

	NSDictionary *vocabFile = @{
		@"vocab": @{
			@"[UNK]": @0, @"[CLS]": @1, @"[SEP]": @2,
			@"play": @3, @"##ing": @4, @"##ed": @5,
			@"hello": @6, @".": @7,
		},
		@"unkToken": @"[UNK]",
		@"continuingSubwordPrefix": @"##",
		@"lowercase": @YES,
	};
	NSData *data = [NSJSONSerialization dataWithJSONObject:vocabFile options:0 error:NULL];
	[data writeToURL:[_directory URLByAppendingPathComponent:@"wordpiece.json"] atomically:YES];

	NSDictionary *manifest = @{
		@"tokenizer": @{ @"type": @"wordpiece", @"vocab": @"wordpiece.json",
						 @"specialTokens": @{ @"[CLS]": @1, @"[SEP]": @2 } },
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

- (void)testGreedyLongestMatchSplitsIntoSubwords
{
	XCTAssertEqualObjects([_tokenizer encode:@"playing"], (@[@3, @4]));
}

- (void)testLowercasingAndPunctuationSplitting
{
	XCTAssertEqualObjects([_tokenizer encode:@"Hello."], (@[@6, @7]));
}

- (void)testAnUnsplittableWordBecomesUnknown
{
	XCTAssertEqualObjects([_tokenizer encode:@"xyzzy"], (@[@0]));
}

- (void)testSpecialTokensPassThrough
{
	XCTAssertEqualObjects([_tokenizer encode:@"[CLS]playing[SEP]"], (@[@1, @3, @4, @2]));
}

- (void)testDecodeMergesContinuationPieces
{
	XCTAssertEqualObjects(([_tokenizer decode:@[@3, @4]]), @"playing");
}

@end
