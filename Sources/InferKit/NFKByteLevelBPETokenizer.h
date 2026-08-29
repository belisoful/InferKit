//
//  NFKByteLevelBPETokenizer.h
//  InferKit
//
//  Internal concrete subclass; callers build it through NFKTokenizer's factory.
//

#ifndef NFKByteLevelBPETokenizer_h
#define NFKByteLevelBPETokenizer_h

#import "NFKTokenizer.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKByteLevelBPETokenizer
	@abstract   A GPT-2 / Qwen-style byte-level BPE tokenizer backed by vocab.json and merges.txt.
	@discussion Byte-level BPE maps each UTF-8 byte to a printable
				character, splits text with a pre-tokenization pattern, and applies the
				ranked merges from merges.txt. Every byte has a base token, so any text round-trips.
				Two pre-tokenizations ship: "gpt2" (the default) and "qwen2", whose pattern differs
				where the token stream shows it — a letter run may absorb one leading punctuation
				character ("-pop" is one pretoken), digits split singly, a punctuation run absorbs
				trailing newlines, and whitespace runs ending in a newline hold together. A Qwen
				vocabulary was trained on its own splits, so encoding it under the GPT-2 pattern
				produces different (valid-looking) ids for the same text.
*/
@interface NFKByteLevelBPETokenizer : NFKTokenizer

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

/*! @param pretokenization "gpt2", "qwen2", or nil for the default "gpt2". */
- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
						  pretokenization:(nullable NSString *)pretokenization
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

#pragma mark Subclass hooks

/*! The pre-tokenization pattern, compiled once during initialization. The base returns the GPT-2
	pattern. */
- (NSString *)pretokenizationPattern;

/*! Applied to a segment before pre-tokenization. The base returns text unchanged. */
- (NSString *)normalizedText:(NSString *)text;

/*! The symbols a word starts from, before any merge is applied. The base returns its characters. */
- (NSArray<NSString *> *)symbolsForWord:(NSString *)word;

/*! Applied to the decoded text of a token sequence. The base returns text unchanged. */
- (NSString *)finalizedText:(NSString *)text;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKByteLevelBPETokenizer_h */
