//
//  NFKModality.h
//  InferKit
//

#ifndef NFKModality_h
#define NFKModality_h

#import <Foundation/Foundation.h>

/*!
	@enum       NFKModality
	@abstract   A media modality a request consumes or produces.
	@discussion InferKit supports the generative matrix of (text, image, video, audio) inputs to
				(text, image, video, audio) outputs. Text is an NSString, an image is a
				CVPixelBuffer or a texture, a video is an NFKVideoAsset, and audio is an
				NFKAudioAsset. A request carries any mix of these as named inputs and declares the
				modality it wants back. A language model produces a text output, returned under
				NFKOutputText.
*/
typedef NS_ENUM(NSInteger, NFKModality) {
	NFKModalityText		= 0,
	NFKModalityImage	= 1,
	NFKModalityVideo	= 2,
	NFKModalityAudio	= 3,
};

#endif /* NFKModality_h */
