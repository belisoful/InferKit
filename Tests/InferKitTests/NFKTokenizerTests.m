//
//  NFKTokenizerTests.m
//  NFKTests
//
//  Byte-level BPE encode/decode against a small hand-authored vocabulary and merge list.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTokenizer.h>
#import <InferKit/NFKErrors.h>

@interface NFKTokenizerTests : XCTestCase
@end

@implementation NFKTokenizerTests
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

	// Every letter here is printable ASCII, so the byte-level mapping is the identity and the
	// vocabulary keys are plain text.
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"h": @0, @"e": @1, @"l": @2, @"o": @3,
		@"he": @4, @"ll": @5, @"hell": @6, @"hello": @7,
		@"<eos>": @8,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];

	NSString *merges = @"#version: 0.2\nh e\nl l\nhe ll\nhell o\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];

	NSDictionary *manifest = @{
		@"eosTokenId": @8,
		@"tokenizer": @{
			@"type": @"bpe-bytelevel",
			@"vocab": @"vocab.json",
			@"merges": @"merges.txt",
			@"specialTokens": @{ @"<eos>": @8 },
		},
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

- (void)testTheFactoryReadsTheEndOfSequenceId
{
	XCTAssertEqual(_tokenizer.eosTokenId, 8);
	XCTAssertEqual(_tokenizer.bosTokenId, -1);
}

- (void)testMergesResolveToTheLongestToken
{
	XCTAssertEqualObjects([_tokenizer encode:@"hello"], (@[@7]));
	XCTAssertEqualObjects([_tokenizer encode:@"hell"], (@[@6]));
	XCTAssertEqualObjects([_tokenizer encode:@"he"], (@[@4]));
}

- (void)testDecodeReversesEncode
{
	XCTAssertEqualObjects([_tokenizer decode:@[@7]], @"hello");
	XCTAssertEqualObjects([_tokenizer decode:[_tokenizer encode:@"hello"]], @"hello");
}

- (void)testSpecialTokensBecomeTheirOwnId
{
	XCTAssertEqualObjects([_tokenizer encode:@"hello<eos>"], (@[@7, @8]));
}

- (void)testEmptyTextEncodesToNothing
{
	XCTAssertEqualObjects([_tokenizer encode:@""], (@[]));
}

- (void)testAnUnsupportedTokenizerTypeIsRejected
{
	NSDictionary *manifest = @{ @"tokenizer": @{ @"type": @"char-level", @"vocab": @"vocab.txt" } };
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:_directory error:&error];
	XCTAssertNil(tokenizer);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

@end
