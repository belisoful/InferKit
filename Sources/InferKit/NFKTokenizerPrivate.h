//
//  NFKTokenizerPrivate.h
//  InferKit
//
//  Shared between NFKTokenizer and its concrete subclasses; not part of the public API.
//

#ifndef NFKTokenizerPrivate_h
#define NFKTokenizerPrivate_h

#import "NFKTokenizer.h"

NS_ASSUME_NONNULL_BEGIN

@interface NFKTokenizer ()

/*! The designated initializer a subclass calls to record the special token ids. */
- (instancetype)initWithEOS:(NSInteger)eosTokenId bos:(NSInteger)bosTokenId;

/*! Sets *outError to an NFKInferenceErrorDomain error, when outError is non-NULL. */
+ (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason;

/*! Splits text into segments at any special-token literal, keeping the literals as their own
	segments (longest literal wins on overlap). Returns @[text] when specialTokens is empty. A
	subclass encodes an ordinary segment and emits a matched literal's id directly. */
- (NSArray<NSString *> *)segmentsOfText:(NSString *)text
						bySpecialTokens:(nullable NSDictionary<NSString *, NSNumber *> *)specialTokens;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKTokenizerPrivate_h */
