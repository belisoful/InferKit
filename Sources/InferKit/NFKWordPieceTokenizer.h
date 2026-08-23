//
//  NFKWordPieceTokenizer.h
//  InferKit
//
//  Internal concrete subclass; callers build it through NFKTokenizer's factory.
//

#ifndef NFKWordPieceTokenizer_h
#define NFKWordPieceTokenizer_h

#import "NFKTokenizer.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKWordPieceTokenizer
	@abstract   A WordPiece tokenizer (BERT family).
	@discussion WordPiece runs a basic pass first (optional lowercasing,
				whitespace splitting, and splitting punctuation into its own tokens), then a greedy
				longest-match pass per word: it takes the longest vocabulary piece from the front,
				prefixes each following piece with the continuation marker (## by default), and emits
				the unknown token when a word has no valid split.

				The vocabulary is a JSON file the converter emits: a token-to-id map plus the
				unkToken, continuingSubwordPrefix, lowercase, and maxCharsPerWord settings.
*/
@interface NFKWordPieceTokenizer : NFKTokenizer

- (nullable instancetype)initWithVocabURL:(NSURL *)vocabURL
							specialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens
									  eos:(NSInteger)eosTokenId
									  bos:(NSInteger)bosTokenId
									error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKWordPieceTokenizer_h */
