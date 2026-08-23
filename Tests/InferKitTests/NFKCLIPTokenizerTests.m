//
//  NFKCLIPTokenizerTests.m
//  NFKTests
//
//  The CLIP variant of byte-level BPE, against a small hand-authored vocabulary and merge list.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTokenizer.h>

@interface NFKCLIPTokenizerTests : XCTestCase
@end

@implementation NFKCLIPTokenizerTests
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

	// A CLIP vocabulary distinguishes a word-final piece from an interior one, so the entries a whole
	// word resolves to carry "</w>".
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"h": @0, @"e": @1, @"l": @2, @"o": @3, @"o</w>": @4,
		@"he": @5, @"ll": @6, @"hell": @7, @"hello</w>": @8,
		@"c": @9, @"a": @10, @"t": @11, @"t</w>": @12, @"ca": @13, @"cat</w>": @14,
		@"1</w>": @15, @"2</w>": @16, @"!": @17, @"!!</w>": @18, @"!</w>": @19,
		@"<|startoftext|>": @49406, @"<|endoftext|>": @49407,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];

	NSString *merges = @"#version: 0.2\nh e\nl l\nhe ll\nhell o</w>\nc a\nca t</w>\n! !</w>\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];

	NSDictionary *manifest = @{
		@"bosTokenId": @49406,
		@"eosTokenId": @49407,
		@"tokenizer": @{
			@"type": @"clip",
			@"vocab": @"vocab.json",
			@"merges": @"merges.txt",
			@"specialTokens": @{ @"<|startoftext|>": @49406, @"<|endoftext|>": @49407 },
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

- (void)testMergesResolveToTheWordFinalToken
{
	XCTAssertEqualObjects([_tokenizer encode:@"hello"], (@[@8]));
}

- (void)testTextIsLowercasedAndItsWhitespaceCollapsed
{
	XCTAssertEqualObjects([_tokenizer encode:@"  HELLO   CAT  "], (@[@8, @14]));
}

// The reference takes one digit at a time, so a number is as many tokens as it has digits.
- (void)testDigitsTokenizeIndividually
{
	XCTAssertEqualObjects([_tokenizer encode:@"12"], (@[@15, @16]));
}

- (void)testARunOfPunctuationIsOnePiece
{
	XCTAssertEqualObjects([_tokenizer encode:@"hello!!"], (@[@8, @18]));
}

- (void)testSpecialTokensBecomeTheirOwnId
{
	XCTAssertEqualObjects([_tokenizer encode:@"<|startoftext|>hello<|endoftext|>"],
						  (@[@49406, @8, @49407]));
}

// The end-of-word marker is what a word boundary is stored as, so decoding turns it back into a space.
- (void)testDecodeRestoresWordBoundaries
{
	XCTAssertEqualObjects([_tokenizer decode:(@[@8, @14])], @"hello cat");
}

- (void)testEmptyTextEncodesToNothing
{
	XCTAssertEqualObjects([_tokenizer encode:@""], @[]);
}

// The start and end ids are the model's, not the tokenizer's own output.
- (void)testTheFactoryReadsTheMarkerIds
{
	XCTAssertEqual(_tokenizer.bosTokenId, 49406);
	XCTAssertEqual(_tokenizer.eosTokenId, 49407);
}

@end
