//
//  NFKUnigramTokenizer.h
//  InferKit
//
//  Internal concrete subclass; callers build it through NFKTokenizer's factory.
//

#ifndef NFKUnigramTokenizer_h
#define NFKUnigramTokenizer_h

#import "NFKTokenizer.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKUnigramTokenizer
	@abstract   A SentencePiece-style unigram tokenizer (Llama, Mistral, Gemma family).
	@discussion A unigram model scores every vocabulary piece; encoding
				is the segmentation of the text that maximizes the summed scores, found with a Viterbi
				pass. Text is normalized the SentencePiece way: a leading space is added, then every
				space becomes U+2581 (LOWER ONE EIGHTH BLOCK). Characters no piece covers fall back to
				their UTF-8 bytes as <0xHH> tokens when the vocabulary defines byte fallback.

				The vocabulary is a JSON file the converter emits: an array of [piece, score] pairs
				(the id is the index), plus unkId, byteFallback, and addDummyPrefix flags.
*/
@interface NFKUnigramTokenizer : NFKTokenizer

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKUnigramTokenizer_h */
