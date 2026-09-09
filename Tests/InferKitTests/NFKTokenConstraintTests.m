//
//  NFKTokenConstraintTests.m
//  NFKTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTokenConstraint.h>

@interface NFKTokenConstraintTests : XCTestCase
@end

@implementation NFKTokenConstraintTests
{
	NFKTokenVocabulary *_vocabulary;
	NSArray<NSString *> *_pieces;
}

/// Single bytes at their own ids, JSON-shaped pieces above them, and the last id as the end token.
- (void)setUp
{
	[super setUp];
	NSMutableArray<NSData *> *tokens = [NSMutableArray array];
	for (NSUInteger byte = 0; byte < 256; byte++) {
		uint8_t value = (uint8_t)byte;
		[tokens addObject:[NSData dataWithBytes:&value length:1]];
	}
	_pieces = @[ @"{\"", @"\":", @", \"", @"\"}", @"true", @"false", @"null", @"123", @"\"name\"", @"\"id\"",
				 @"[", @"]", @"{", @"}", @" ", @"\"", @"42", @"0.5", @"-7", @"1e3", @"yes", @"no", @"maybe", @"may" ];
	for (NSString *piece in _pieces) {
		[tokens addObject:[piece dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[tokens addObject:[NSData data]];		// the end token has no bytes
	_vocabulary = [[NFKTokenVocabulary alloc] initWithTokens:tokens endToken:(NSInteger)tokens.count - 1];
}

- (NSSet<NSString *> *)allowedText:(NFKTokenConstraint *)constraint after:(NSString *)text
{
	NSMutableSet<NSString *> *allowed = [NSMutableSet set];
	for (NSNumber *token in [constraint allowedTokensAfterText:text]) {
		NSData *bytes = [_vocabulary bytesForToken:token.unsignedIntegerValue];
		[allowed addObject:[[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding] ?: @"?"];
	}
	return allowed;
}

- (void)testTheJSONGrammarAcceptsValidPrefixesAndRejectsInvalidOnes
{
	NFKJSONConstraint *json = [[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootContainer];
	for (NSString *prefix in @[ @"{", @"{\"a\": 1", @"{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\\\"y\"", @"[",
								@"  {\n\"k\"\t:\ttrue", @"{\"n\": -0.5e+3", @"{\"s\": \"\\u00e9", @"[\"multi byte é\"" ]) {
		XCTAssertTrue([json acceptsText:prefix], @"%@ is a valid prefix", prefix);
	}
	for (NSString *invalid in @[ @"}", @"{a", @"{\"a\" 1", @"{\"a\": 01", @"{\"a\": 1.}", @"{\"a\": tru3", @"{\"a\": \"\\x\"",
								 @"[1,]", @"{,}", @"\"scalar root\"", @"{\"a\":1}}", @"{\"a\":1} 2" ]) {
		XCTAssertFalse([json acceptsText:invalid], @"%@ is rejected", invalid);
	}
	for (NSString *complete in @[ @"{}", @"[]", @"{\"a\": 1}", @"[1, 2.5, \"x\", true, null, {\"b\": []}]", @" {} \n" ]) {
		XCTAssertTrue([json isCompleteText:complete], @"%@ is complete", complete);
	}
	XCTAssertFalse([json isCompleteText:@"{\"a\": 1"]);
	XCTAssertFalse([json isCompleteText:@"[1, 2"]);

	// Whitespace is capped between tokens and free inside a string.
	NSString *eight = [@"" stringByPaddingToLength:8 withString:@" " startingAtIndex:0];
	NSString *nine = [@"" stringByPaddingToLength:9 withString:@" " startingAtIndex:0];
	NSString *eightBetween = [NSString stringWithFormat:@"{%@\"a\"", eight];
	NSString *nineBetween = [NSString stringWithFormat:@"{%@\"a\"", nine];
	NSString *insideString = [NSString stringWithFormat:@"{\"%@%@", nine, nine];
	XCTAssertTrue([json acceptsText:eightBetween]);
	XCTAssertFalse([json acceptsText:nineBetween]);
	XCTAssertTrue([json acceptsText:insideString]);

	NFKJSONConstraint *scalar = [[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootAny];
	XCTAssertTrue([scalar isCompleteText:@"\"text\""]);
	XCTAssertTrue([scalar isCompleteText:@"-12.5"]);
	XCTAssertTrue([scalar acceptsText:@"12."]);
	XCTAssertFalse([scalar isCompleteText:@"12."]);

	NFKJSONConstraint *object = [[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootObject];
	XCTAssertTrue([object acceptsText:@"{\"a\": [1]}"]);
	XCTAssertFalse([object acceptsText:@"["]);
	XCTAssertFalse([[[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootArray] acceptsText:@"{"]);
}

- (void)testTheAdmissibleTokensFollowTheGrammar
{
	NFKJSONConstraint *json = [[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootContainer];
	NSSet<NSString *> *opening = [self allowedText:json after:@""];
	XCTAssertTrue(([[NSSet setWithArray:@[ @"{", @"[", @" ", @"{\"" ]] isSubsetOfSet:opening]));
	XCTAssertFalse(([opening intersectsSet:[NSSet setWithArray:@[ @"}", @"\"", @"true", @"123", @"" ]]]), @"no end token before a document");

	NSSet<NSString *> *afterBrace = [self allowedText:json after:@"{"];
	XCTAssertTrue(([[NSSet setWithArray:@[ @"\"", @"}", @"\"name\"", @" " ]] isSubsetOfSet:afterBrace]));
	XCTAssertFalse([afterBrace containsObject:@"123"], @"an object wants a key");

	NSSet<NSString *> *afterColon = [self allowedText:json after:@"{\"a\":"];
	XCTAssertTrue(([[NSSet setWithArray:@[ @"123", @"true", @"null", @"[", @"{", @"\"", @"-7" ]] isSubsetOfSet:afterColon]));
	XCTAssertFalse([afterColon containsObject:@"}"]);

	NSSet<NSString *> *closed = [self allowedText:json after:@"{\"a\": 1}"];
	XCTAssertTrue([closed containsObject:@""], @"the end token is admitted once the document closes");
	XCTAssertTrue(([closed isSubsetOfSet:[NSSet setWithArray:@[ @"", @" ", @"\n", @"\t", @"\r" ]]]));
}

- (void)testTheCursorMasksScoresInPlace
{
	NFKJSONConstraint *json = [[NFKJSONConstraint alloc] initWithVocabulary:_vocabulary root:NFKJSONRootObject];
	NFKTokenConstraintCursor *cursor = [json makeCursor];
	NSUInteger count = _vocabulary.count;
	float *scores = calloc(count, sizeof(float));
	[cursor maskScores:scores count:count];
	NSUInteger brace = 256 + [_pieces indexOfObject:@"{"];
	NSUInteger bracket = 256 + [_pieces indexOfObject:@"["];
	XCTAssertEqual(scores[brace], 0.0f, @"an object may open");
	XCTAssertEqual(scores[bracket], -INFINITY, @"an array may not, with an object root");
	XCTAssertEqual(scores[(NSUInteger)cursor.endToken], -INFINITY, @"no end before a document");
	XCTAssertFalse(cursor.isComplete);

	[cursor acceptToken:(NSInteger)brace];
	[cursor acceptToken:(NSInteger)(256 + [_pieces indexOfObject:@"}"])];
	XCTAssertTrue(cursor.isComplete);
	for (NSUInteger i = 0; i < count; i++) { scores[i] = 0; }
	[cursor maskScores:scores count:count];
	XCTAssertEqual(scores[(NSUInteger)cursor.endToken], 0.0f, @"the end token is admitted once complete");
	XCTAssertEqual(scores[brace], -INFINITY);
	XCTAssertEqual(scores[256 + [_pieces indexOfObject:@" "]], 0.0f, @"trailing whitespace stays admissible");
	free(scores);
}

- (void)testAChoiceConstraintAdmitsExactlyTheChoices
{
	NFKChoiceConstraint *choice = [[NFKChoiceConstraint alloc] initWithChoices:@[ @"yes", @"no", @"maybe" ]
																	 vocabulary:_vocabulary];
	NSSet<NSString *> *opening = [self allowedText:choice after:@""];
	XCTAssertTrue(([[NSSet setWithArray:@[ @"yes", @"no", @"maybe", @"may", @"y", @"n", @"m" ]] isSubsetOfSet:opening]));
	XCTAssertFalse([opening containsObject:@"123"]);
	XCTAssertFalse([opening containsObject:@""]);
	XCTAssertTrue([choice isCompleteText:@"yes"]);
	XCTAssertTrue([choice acceptsText:@"mayb"]);
	XCTAssertFalse([choice isCompleteText:@"mayb"]);
	XCTAssertFalse([choice acceptsText:@"mayx"]);
	XCTAssertFalse([choice acceptsText:@"yesx"]);
	// "may" is a prefix of "maybe": the run may end at either, never between.
	NFKChoiceConstraint *prefixed = [[NFKChoiceConstraint alloc] initWithChoices:@[ @"may", @"maybe" ] vocabulary:_vocabulary];
	XCTAssertTrue([prefixed isCompleteText:@"may"]);
	XCTAssertTrue([[self allowedText:prefixed after:@"may"] containsObject:@""]);
	XCTAssertTrue([[self allowedText:prefixed after:@"may"] containsObject:@"b"]);
	// Two choices sharing a prefix diverge at their first different byte and never cross over.
	NFKChoiceConstraint *fork = [[NFKChoiceConstraint alloc] initWithChoices:@[ @"cat", @"car" ] vocabulary:_vocabulary];
	XCTAssertTrue([fork acceptsText:@"ca"]);
	XCTAssertTrue([fork isCompleteText:@"cat"]);
	XCTAssertTrue([fork isCompleteText:@"car"]);
	XCTAssertFalse([fork acceptsText:@"cab"]);
}

- (void)testAVocabularyReadsAByteLevelTokenizer
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString];
	[NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];
	NSDictionary *vocab = @{ @"h": @0, @"e": @1, @"he": @2, @"Ġ": @3, @"<eos>": @4 };
	[[NSJSONSerialization dataWithJSONObject:vocab options:0 error:NULL]
		writeToURL:[directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];
	[@"#version: 0.2\nh e\n" writeToURL:[directory URLByAppendingPathComponent:@"merges.txt"]
							 atomically:YES encoding:NSUTF8StringEncoding error:NULL];
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:@{
		@"tokenizer": @{ @"type": @"bpe-bytelevel", @"specialTokens": @{ @"<eos>": @4 } }, @"eosTokenId": @4,
	} directory:directory error:&error];
	XCTAssertNotNil(tokenizer, @"%@", error);
	NFKTokenVocabulary *vocabulary = [[NFKTokenVocabulary alloc] initWithTokenizer:tokenizer size:6];
	XCTAssertEqualObjects([vocabulary bytesForToken:2], [@"he" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqualObjects([vocabulary bytesForToken:3], [@" " dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqual([vocabulary bytesForToken:5].length, 0u, @"an id past the tokenizer has no bytes");
	XCTAssertEqual(vocabulary.endToken, 4);
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];
}

@end
