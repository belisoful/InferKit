//
//  NFKCLIPTokenizer.h
//  InferKit
//
//  Internal concrete subclass; callers build it through NFKTokenizer's factory.
//

#ifndef NFKCLIPTokenizer_h
#define NFKCLIPTokenizer_h

#import "NFKByteLevelBPETokenizer.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKCLIPTokenizer
	@abstract   The byte-level BPE tokenizer CLIP and the Stable Diffusion text encoders take.
	@discussion CLIP shares byte-level BPE with GPT-2 and differs in
				four places, which this subclass supplies:

				- text is lowercased and its whitespace collapsed before pre-tokenization;
				- the pre-tokenization pattern takes a run of letters, ONE digit, or a run of other
				  non-space characters, with no leading space;
				- a word's last character carries the end-of-word marker "</w>", so the vocabulary
				  distinguishes a word-final piece from an interior one;
				- decoding turns "</w>" back into a space.

				The start and end markers are ordinary vocabulary entries; a caller supplies them as
				special tokens so they survive pre-tokenization. encode: returns the ids for the text
				alone. A text encoder's input adds the start and end ids and pads to its context
				length, which is the model's geometry rather than the tokenizer's.
*/
@interface NFKCLIPTokenizer : NFKByteLevelBPETokenizer
@end

NS_ASSUME_NONNULL_END

#endif /* NFKCLIPTokenizer_h */
