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

// Decoding concatenates each token's bytes, so the bytes of a sequence's tokens joined equal the
// decoded text, and a special token contributes its literal.
- (void)testTokenBytesConcatenateToTheDecodedText
{
	NSArray<NSNumber *> *ids = [_tokenizer encode:@"hello <eos>"];
	NSMutableData *joined = [NSMutableData data];
	for (NSNumber *tokenId in ids) {
		NSData *bytes = [_tokenizer bytesForTokenId:tokenId.integerValue];
		XCTAssertNotNil(bytes, @"token %@ has bytes", tokenId);
		[joined appendData:bytes];
	}
	XCTAssertEqualObjects([[NSString alloc] initWithData:joined encoding:NSUTF8StringEncoding],
						  [_tokenizer decode:ids]);
	XCTAssertEqualObjects([_tokenizer bytesForTokenId:8], [@"<eos>" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertNil([_tokenizer bytesForTokenId:100000], @"an id outside the vocabulary has no bytes");
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

// A merge cannot cross a pretoken boundary, so the pattern shows through the ids: Qwen2 lets a
// letter run absorb one leading punctuation character ("-b" is one pretoken, so the "- b" merge
// applies) and splits digits singly (so the "1 2" merge cannot apply). GPT-2 does the opposite on
// both. A Qwen vocabulary encoded under the wrong pattern produces different, valid-looking ids.
- (void)testTheQwen2PretokenizationChangesTheMerges
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"a": @0, @"-": @1, @"b": @2, @"-b": @3,
		@"1": @4, @"2": @5, @"12": @6,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"qwen-vocab.json"] atomically:YES];
	NSString *merges = @"#version: 0.2\n- b\n1 2\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"qwen-merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];

	NSError *error = nil;
	NFKTokenizer *qwen = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"qwen-vocab.json", @"merges": @"qwen-merges.txt",
		@"pretokenizer": @"qwen2",
	} } directory:_directory error:&error];
	XCTAssertNotNil(qwen, @"%@", error);
	NFKTokenizer *gpt2 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"qwen-vocab.json", @"merges": @"qwen-merges.txt",
	} } directory:_directory error:&error];
	XCTAssertNotNil(gpt2, @"%@", error);

	NSArray *expectedQwen = @[@0, @3, @4, @5];
	NSArray *expectedGPT2 = @[@0, @1, @2, @6];
	XCTAssertEqualObjects([qwen encode:@"a-b12"], expectedQwen);
	XCTAssertEqualObjects([gpt2 encode:@"a-b12"], expectedGPT2);
}

- (void)testAnUnknownPretokenizationIsRejected
{
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"vocab.json", @"merges": @"merges.txt",
		@"pretokenizer": @"llama3",
	} } directory:_directory error:&error];
	XCTAssertNil(tokenizer);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

@end
