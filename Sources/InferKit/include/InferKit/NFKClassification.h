//
//  NFKClassification.h
//  InferKit
//

#ifndef NFKClassification_h
#define NFKClassification_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKClassification
	@abstract   One predicted class: a label and a confidence.
	@discussion A classification or tagging backend (audio tagging, image
				classification) returns an NSArray<NFKClassification *> under NFKOutputClassifications,
				ordered most-confident first. classIndex is the model's raw class id; label is the
				human-readable name when the backend has a class list, nil when it returns indices only.
*/
@interface NFKClassification : NSObject <NSCopying>

/*! The class name, or nil when the backend returns a class index only. */
@property (nonatomic, readonly, nullable, copy) NSString *label;

/*! The model's class index. */
@property (nonatomic, readonly) NSInteger classIndex;

/*! The confidence in 0...1. */
@property (nonatomic, readonly) double confidence;

+ (instancetype)classificationWithLabel:(nullable NSString *)label
							 classIndex:(NSInteger)classIndex
							 confidence:(double)confidence;

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKClassification_h */
