//
//  NFKModelSanityCheck.h
//  InferKit
//

#ifndef NFKModelSanityCheck_h
#define NFKModelSanityCheck_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKModelSanityFinding
	@abstract   What a sanity check found wrong with a generation. Introduced in InferKit 0.4.0.
	@constant   NFKModelSanityFindingNone Nothing.
	@constant   NFKModelSanityFindingEmpty The reply is empty or only whitespace.
	@constant   NFKModelSanityFindingGarbled The reply holds the Unicode replacement character or a control
				character other than a tab or a line break: bytes the tokenizer did not decode.
	@constant   NFKModelSanityFindingRepetitive The reply ends in one stretch of up to 32 characters
				repeated (ten times for one or two characters, six for three or four, three for a longer
				one), or a reply of 32 characters or more uses three distinct characters or fewer: a
				decoder caught in a loop.
*/
typedef NS_OPTIONS(NSUInteger, NFKModelSanityFinding) {
	NFKModelSanityFindingNone		= 0,
	NFKModelSanityFindingEmpty		= 1 << 0,
	NFKModelSanityFindingGarbled	= 1 << 1,
	NFKModelSanityFindingRepetitive	= 1 << 2,
};

/*!
	@class      NFKModelSanityResult
	@abstract   A sanity check's verdict on one generation.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKModelSanityResult : NSObject

/*! The generated text. */
@property (nonatomic, readonly, copy) NSString *output;
/*! What the check found. */
@property (nonatomic, readonly) NFKModelSanityFinding findings;
/*! YES when the check found nothing. */
@property (nonatomic, readonly, getter=isPassed) BOOL passed;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelSanityCheck
	@abstract   A short greedy generation, tested for output a misloaded model writes.
	@discussion A model resolved through a route no reference measured (NFKModelResolver's family, Core ML
				package, and local-server routes) can load and run wrong: weights read at the wrong
				offsets, a tokenizer that splits text differently. The check runs one greedy generation
				from a fixed prompt and reports empty, garbled, and looping output. A model that writes
				fluent but wrong text passes it.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelSanityCheck : NSObject

/*! The prompt checkBackend:error: generates from. */
@property (class, nonatomic, readonly, copy) NSString *defaultPrompt;
/*! The tokens checkBackend:error: generates: 48. */
@property (class, nonatomic, readonly) NSInteger defaultTokenCount;

/*! Runs one generation from defaultPrompt, defaultTokenCount tokens at temperature 0, and judges it.
	Blocks for the generation; run it off the render thread. Returns nil and sets error when the backend
	fails or its result holds no NFKOutputText. */
+ (nullable NFKModelSanityResult *)checkBackend:(id<NFKInferenceBackend>)backend error:(NSError **)error;

/*! Runs one generation from prompt, tokenCount tokens at temperature 0, and judges it. Blocks for the
	generation. Returns nil and sets error when tokenCount is below 1, the backend fails, or its result
	holds no NFKOutputText. */
+ (nullable NFKModelSanityResult *)checkBackend:(id<NFKInferenceBackend>)backend
										 prompt:(NSString *)prompt
									 tokenCount:(NSInteger)tokenCount
										  error:(NSError **)error;

/*! The verdict on a generated text, with no backend: the judgment checkBackend: applies. */
+ (NFKModelSanityResult *)resultForOutput:(NSString *)output;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelSanityCheck_h */
