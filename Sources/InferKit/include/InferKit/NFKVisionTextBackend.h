//
//  NFKVisionTextBackend.h
//  InferKit
//

#ifndef NFKVisionTextBackend_h
#define NFKVisionTextBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKVisionTextBackend
	@abstract   Reads the text in an image through Apple's Vision framework.
	@discussion Text recognition is the one language task the toolkit does not otherwise ship, and
				Vision performs it on device with no weights to download and no model to convert.
				The backend takes NFKInputImage and returns the recognized text under NFKOutputText,
				lines joined by newlines in reading order, with one NFKDetection per line under
				NFKOutputDetections carrying that line's string, confidence, and box.

				Recognition is accurate by default and slower for it; set usesAccurateRecognition to
				NO for the fast path, which suits large printed text and video frames. languages
				names the languages to try, in order of preference. customWords supplements the
				vocabulary with names and terms the language model would otherwise correct away.

				Available wherever the core runs. Introduced in InferKit 0.4.0.
*/
@interface NFKVisionTextBackend : NSObject <NFKInferenceBackend>

/*! YES for the accurate recognition path, NO for the fast one. YES by default. */
@property (nonatomic) BOOL usesAccurateRecognition;

/*! YES to apply language correction to the reading. YES by default. */
@property (nonatomic) BOOL correctsLanguage;

/*! The languages to recognize, in order of preference (for example @[@"en-US"]). nil uses Vision's
	own order, which follows the user's preferred languages. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *languages;

/*! Words to add to the recognition vocabulary, for names and terms a general model corrects away. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *customWords;

/*! The smallest text to read, as a fraction of the image height. 0 uses Vision's default. */
@property (nonatomic) double minimumTextHeight;

+ (instancetype)backend;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKVisionTextBackend_h */
