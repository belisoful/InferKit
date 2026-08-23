//
//  NFKErrors.h
//  InferKit
//

#ifndef NFKErrors_h
#define NFKErrors_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error domain for InferKit inference failures. */
extern NSString * const NFKInferenceErrorDomain;

/*!
	@enum       NFKInferenceError
	@abstract   Error codes in NFKInferenceErrorDomain.
*/
typedef NS_ENUM(NSInteger, NFKInferenceError) {
	kNFKError_InferenceNotReady			= 1,
	kNFKError_InferenceMissingInput		= 2,
	kNFKError_InferenceBackendFailure	= 3,
	kNFKError_InferenceUnsupported		= 4,
};

NS_ASSUME_NONNULL_END

#endif /* NFKErrors_h */
