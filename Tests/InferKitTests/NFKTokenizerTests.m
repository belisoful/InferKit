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

// o200k (OpenAI's o200k_base / o200k_harmony, which gpt-oss ships) splits digits in runs of at most
// THREE and splits a word wherever its case pattern turns over — "aB" is two pretokens, a lower-led
// word and then a capital-led one — where GPT-2 takes the whole digit run and the whole letter run.
- (void)testTheO200kPretokenizationChangesTheMerges
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"1": @0, @"2": @1, @"3": @2, @"4": @3, @"12": @4, @"123": @5, @"1234": @6,
		@"a": @7, @"B": @8, @"aB": @9,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"o200k-vocab.json"] atomically:YES];
	NSString *merges = @"#version: 0.2\n1 2\n12 3\n123 4\na B\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"o200k-merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];

	NSError *error = nil;
	NFKTokenizer *o200k = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"o200k-vocab.json", @"merges": @"o200k-merges.txt",
		@"pretokenizer": @"o200k",
	} } directory:_directory error:&error];
	XCTAssertNotNil(o200k, @"%@", error);
	NFKTokenizer *gpt2 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"o200k-vocab.json", @"merges": @"o200k-merges.txt",
	} } directory:_directory error:&error];
	XCTAssertNotNil(gpt2, @"%@", error);

	NSArray *expectedO200k = @[@5, @3, @7, @8];
	NSArray *expectedGPT2 = @[@6, @9];
	XCTAssertEqualObjects([o200k encode:@"1234aB"], expectedO200k);
	XCTAssertEqualObjects([gpt2 encode:@"1234aB"], expectedGPT2);
}

// Qwen3.5 normalizes each segment to NFC before splitting, so a decomposed "e" + combining acute
// encodes as the composed "é" token; Qwen2 reads the letter and the mark as two pretokens.
- (void)testTheQwen35PretokenizationComposesBeforeSplitting
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"Ã": @0, @"©": @1, @"Ã©": @2, @"e": @3,
		@"Ì": @4, @"ģ": @5, @"Ìģ": @6,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"qwen35-vocab.json"] atomically:YES];
	NSString *merges = @"#version: 0.2\nÃ ©\nÌ ģ\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"qwen35-merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];

	NSError *error = nil;
	NFKTokenizer *qwen35 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"qwen35-vocab.json", @"merges": @"qwen35-merges.txt",
		@"pretokenizer": @"qwen35",
	} } directory:_directory error:&error];
	XCTAssertNotNil(qwen35, @"%@", error);
	NFKTokenizer *qwen2 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"qwen35-vocab.json", @"merges": @"qwen35-merges.txt",
		@"pretokenizer": @"qwen2",
	} } directory:_directory error:&error];
	XCTAssertNotNil(qwen2, @"%@", error);

	NSArray *expectedQwen35 = @[@2];
	NSArray *expectedQwen2 = @[@3, @6];
	XCTAssertEqualObjects([qwen35 encode:@"é"], expectedQwen35);
	XCTAssertEqualObjects([qwen2 encode:@"é"], expectedQwen2);

	NFKTokenizer *composedQwen2 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"qwen35-vocab.json", @"merges": @"qwen35-merges.txt",
		@"pretokenizer": @"qwen2", @"normalizer": @"nfc",
	} } directory:_directory error:&error];
	XCTAssertNotNil(composedQwen2, @"%@", error);
	XCTAssertEqualObjects([composedQwen2 encode:@"é"], expectedQwen35, @"the NFC normalizer composes under any pattern");
}

- (void)testAnUnknownPretokenizationIsRejected
{
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"vocab.json", @"merges": @"merges.txt",
		@"pretokenizer": @"llama2",
	} } directory:_directory error:&error];
	XCTAssertNil(tokenizer);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

- (void)testAnUnknownNormalizationIsRejected
{
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"vocab.json", @"merges": @"merges.txt",
		@"normalizer": @"nfkc",
	} } directory:_directory error:&error];
	XCTAssertNil(tokenizer);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

- (NFKTokenizer *)digitTokenizerWithSpec:(NSDictionary *)extra
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"1": @0, @"2": @1, @"3": @2, @"4": @3, @"12": @4, @"123": @5, @"1234": @6,
		@"a": @7, @"B": @8, @"aB": @9, @"'": @10, @"s": @11, @"'s": @12, @"aB's": @13, @"34": @14,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"digit-vocab.json"] atomically:YES];
	NSString *merges = @"#version: 0.2\n1 2\n12 3\n123 4\na B\n' s\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"digit-merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];
	NSMutableDictionary *spec = [@{ @"type": @"bpe-bytelevel", @"vocab": @"digit-vocab.json",
									@"merges": @"digit-merges.txt" } mutableCopy];
	[spec addEntriesFromDictionary:extra];
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": spec } directory:_directory error:&error];
	XCTAssertNotNil(tokenizer, @"%@", error);
	return tokenizer;
}

// Llama 3 (and Mistral's Tekken, and Granite 4.0) is Qwen2's pattern with digits in runs of at most
// three, so "1234" is "123" + "4" where Qwen2 takes four single digits.
- (void)testTheLlama3PretokenizationSplitsDigitsInThrees
{
	NFKTokenizer *llama3 = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"llama3" }];
	NFKTokenizer *qwen2 = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"qwen2" }];
	NSArray *expectedLlama3 = @[@5, @3, @9];
	NSArray *expectedQwen2 = @[@0, @1, @2, @3, @9];
	XCTAssertEqualObjects([llama3 encode:@"1234aB"], expectedLlama3);
	XCTAssertEqualObjects([qwen2 encode:@"1234aB"], expectedQwen2);
}

// Nemotron-H splits words by case pattern as o200k does, but takes single digits where o200k keeps
// runs of three.
- (void)testTheNemotronPretokenizationSplitsDigitsSingly
{
	NFKTokenizer *nemotron = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"nemotron" }];
	NFKTokenizer *o200k = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"o200k" }];
	NSArray *expectedNemotron = @[@0, @1, @2, @3, @7, @8, @12];
	NSArray *expectedO200k = @[@5, @3, @7, @8, @12];
	XCTAssertEqualObjects([nemotron encode:@"1234aB's"], expectedNemotron);
	XCTAssertEqualObjects([o200k encode:@"1234aB's"], expectedO200k);
}

// SmolLM isolates each digit before the GPT-2 pattern runs, so a space before a number stays its own
// pretoken where GPT-2 joins it to the digits.
- (void)testTheSmolLMPretokenizationIsolatesDigits
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{ @"a": @0, @"Ġ": @1, @"1": @2, @"2": @3, @"Ġ1": @4, @"12": @5, @"Ġ12": @6 };
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"smollm-vocab.json"] atomically:YES];
	[@"#version: 0.2\nĠ 1\nĠ1 2\n1 2\n" writeToURL:[_directory URLByAppendingPathComponent:@"smollm-merges.txt"]
									  atomically:YES
										encoding:NSUTF8StringEncoding
										   error:NULL];
	NSError *error = nil;
	NFKTokenizer *smollm = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"smollm-vocab.json", @"merges": @"smollm-merges.txt", @"pretokenizer": @"smollm",
	} } directory:_directory error:&error];
	XCTAssertNotNil(smollm, @"%@", error);
	NFKTokenizer *gpt2 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"smollm-vocab.json", @"merges": @"smollm-merges.txt",
	} } directory:_directory error:&error];
	NSArray *expectedSmolLM = @[@0, @1, @2, @3];
	NSArray *expectedGPT2 = @[@0, @6];
	XCTAssertEqualObjects([smollm encode:@"a 12"], expectedSmolLM);
	XCTAssertEqualObjects([gpt2 encode:@"a 12"], expectedGPT2);
}

// Hunyuan's Splits run in turn: digits in threes, then CJK runs, then the word pattern, so "中" leaves
// the Latin word it follows ("ä¸Ń" is its byte-level spelling), where Llama 3's single pattern keeps one
// letter run.
- (void)testTheHunyuanPretokenizationIsolatesIdeographRuns
{
	NSDictionary<NSString *, NSNumber *> *vocab = @{
		@"1": @0, @"2": @1, @"3": @2, @"4": @3, @"12": @4, @"123": @5, @"a": @6, @"B": @7, @"aB": @8,
		@"\u00e4": @9, @"\u00b8": @10, @"\u0143": @11, @"\u00e4\u00b8": @12, @"\u00e4\u00b8\u0143": @13,
		@"aB\u00e4\u00b8\u0143": @14,
	};
	NSData *vocabData = [NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL];
	[vocabData writeToURL:[_directory URLByAppendingPathComponent:@"hunyuan-vocab.json"] atomically:YES];
	NSString *merges = @"#version: 0.2\n1 2\n12 3\na B\n\u00e4 \u00b8\n\u00e4\u00b8 \u0143\naB \u00e4\u00b8\u0143\n";
	[merges writeToURL:[_directory URLByAppendingPathComponent:@"hunyuan-merges.txt"]
			atomically:YES
			  encoding:NSUTF8StringEncoding
				 error:NULL];
	NSError *error = nil;
	NFKTokenizer *hunyuan = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"hunyuan-vocab.json", @"merges": @"hunyuan-merges.txt",
		@"pretokenizer": @"hunyuan",
	} } directory:_directory error:&error];
	XCTAssertNotNil(hunyuan, @"%@", error);
	NFKTokenizer *llama3 = [NFKTokenizer tokenizerForManifest:@{ @"tokenizer": @{
		@"type": @"bpe-bytelevel", @"vocab": @"hunyuan-vocab.json", @"merges": @"hunyuan-merges.txt",
		@"pretokenizer": @"llama3",
	} } directory:_directory error:&error];
	NSArray *expectedHunyuan = @[@5, @3, @8, @13];
	NSArray *expectedLlama3 = @[@5, @3, @14];
	XCTAssertEqualObjects([hunyuan encode:@"1234aB\u4e2d"], expectedHunyuan);
	XCTAssertEqualObjects([llama3 encode:@"1234aB\u4e2d"], expectedLlama3);
}

// With ignore_merges a pretoken the vocabulary holds whole is that token, though no merge reaches it.
- (void)testIgnoringMergesTakesAWholePretokenFromTheVocabulary
{
	NFKTokenizer *ignoring = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"gpt2", @"ignoreMerges": @YES }];
	NFKTokenizer *merging = [self digitTokenizerWithSpec:@{ @"pretokenizer": @"gpt2" }];
	XCTAssertEqualObjects([ignoring encode:@"34"], @[@14]);
	NSArray *expectedMerged = @[@2, @3];
	XCTAssertEqualObjects([merging encode:@"34"], expectedMerged);
}

@end
