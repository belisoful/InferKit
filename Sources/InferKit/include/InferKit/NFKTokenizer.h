//
//  NFKTokenizer.h
//  InferKit
//

#ifndef NFKTokenizer_h
#define NFKTokenizer_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKTokenizer
	@abstract   Converts text to and from the integer token ids a language model consumes.
	@discussion A Core ML language model takes token ids, not text, and
				Core ML does not tokenize. This class cluster fills that gap.

				tokenizerForManifest:directory:error: reads the manifest a converted model ships and
				returns the concrete tokenizer named by its "tokenizer.type". The shipped subclasses are
				byte-level BPE ("bpe-bytelevel", the GPT-2 and Qwen family), the CLIP variant of it
				("clip", which the Stable Diffusion text encoders take), unigram/SentencePiece
				("unigram"), and WordPiece ("wordpiece"). The factory returns an unsupported error for
				a type it has no subclass for, so the boundary is explicit.

				encode: and decode: are the round trip; subclasses implement them. The base is
				abstract: build an instance through the factory.
*/
@interface NFKTokenizer : NSObject

/*!
	@method     tokenizerForManifest:directory:error:
	@abstract   Builds the tokenizer a model's manifest describes, reading its files from directory.
	@discussion Reads manifest["tokenizer"]["type"] and the vocab file names beside it, resolving
				them against directory. Returns nil and sets error when the manifest has no tokenizer
				section, the type is unsupported, or a vocab file cannot be read.
*/
+ (nullable instancetype)tokenizerForManifest:(NSDictionary<NSString *, id> *)manifest
									directory:(NSURL *)directory
										error:(NSError * _Nullable *)outError;

/*! The token ids for text. */
- (NSArray<NSNumber *> *)encode:(NSString *)text;

/*! The text for a sequence of token ids. */
- (NSString *)decode:(NSArray<NSNumber *> *)tokenIds;

/*! The end-of-sequence token id, or -1 when the model defines none. */
@property (nonatomic, readonly) NSInteger eosTokenId;

/*! The beginning-of-sequence token id, or -1 when the model defines none. */
@property (nonatomic, readonly) NSInteger bosTokenId;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKTokenizer_h */
