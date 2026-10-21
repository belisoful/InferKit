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
				The pre-tokenization is named: "gpt2" (the default), "qwen2", "qwen35", "o200k",
				"llama3", or "nemotron", each the Split pattern its family's tokenizer.json states, or
				"smollm", which isolates each digit and splits the runs between by the GPT-2 pattern. Qwen2's
				differs from GPT-2's where the token stream shows it: a letter run may absorb one
				leading punctuation character ("-pop" is one pretoken), digits split singly, a
				punctuation run absorbs trailing newlines, and whitespace runs ending in a newline hold
				together. A vocabulary was trained on its own splits, so encoding it under another
				pattern produces different (valid-looking) ids for the same text.
*/
@interface NFKByteLevelBPETokenizer : NFKTokenizer

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

/*! @param pretokenization A name the designated initializer takes, or nil for "gpt2". */
- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
						  pretokenization:(nullable NSString *)pretokenization
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

/*!
	@param pretokenization "gpt2", "qwen2", "qwen35", "o200k", "llama3", "nemotron", "smollm", or nil for
						   "gpt2".
	@param normalization   "nfc" to compose each segment before splitting, or nil for none. "qwen35"
						   composes whatever this says.
	@param ignoresMerges   YES to emit a pretoken the vocabulary holds whole as that one token, before
						   any merge (tokenizer.json's `ignore_merges`).
*/
- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
								mergesURL:(NSURL *)mergesURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
						  pretokenization:(nullable NSString *)pretokenization
							normalization:(nullable NSString *)normalization
							ignoresMerges:(BOOL)ignoresMerges
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
