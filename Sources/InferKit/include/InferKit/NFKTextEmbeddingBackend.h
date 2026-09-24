//
//  NFKTextEmbeddingBackend.h
//  InferKit
//

#ifndef NFKTextEmbeddingBackend_h
#define NFKTextEmbeddingBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKTextEmbeddingBackend
	@abstract   Embeds a sentence with Apple's Natural Language framework.
	@discussion NFKInputPrompt in, the vector under NFKOutputEmbedding, the shape every other
				embedding backend returns. The vectors come from a model the system already has, so
				there is nothing to download and nothing to convert.

				Apple's sentence embedding is smaller and older than Qwen3-Embedding or
				EmbeddingGemma, and it covers the languages Apple ships a model for. It suits
				grouping, deduplication, and a first-pass ranking a stronger model then reorders.
				`language` picks the model; without one the backend reads the text to decide.

				`isReady` is NO where Apple has no sentence model for the language, which is most
				languages. `availableLanguages` answers before a run. Introduced in InferKit 0.4.0.
*/
@interface NFKTextEmbeddingBackend : NSObject <NFKInferenceBackend>

/*! The language to embed for, BCP-47. nil detects it from the text of each request. */
@property (nonatomic, copy, nullable) NSString *language;

/*! The vector's length for the current language, or 0 when there is no model. */
@property (nonatomic, readonly) NSInteger dimension;

/*! The languages Apple ships a sentence model for on this machine. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *availableLanguages;

+ (instancetype)backend;

/*! A backend for one language. */
+ (instancetype)backendWithLanguage:(NSString *)language;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKTextEmbeddingBackend_h */
