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
	/*! A remote endpoint produced no response at all: the host is down or refused the connection. The
		URL-loading error is under NSUnderlyingErrorKey. Distinct from a server that answered with an
		error, which is kNFKError_InferenceBackendFailure. Introduced in InferKit 0.3.0. */
	kNFKError_RemoteUnreachable			= 5,
	/*! The engine declined to answer this request: a guardrail, a safety policy, or a refusal by the
		model itself. The request is the problem, so sending it again produces the same answer. A
		caller changes the request or tells the user, and does not retry. Introduced in InferKit 0.4.0. */
	kNFKError_InferenceRefused			= 6,
	/*! The engine is rate limited or out of quota. The request is fine and a later one may succeed,
		so a caller backs off rather than changing it. When the engine names a reset time it rides in
		userInfo under the engine's own key. Introduced in InferKit 0.4.0. */
	kNFKError_InferenceRateLimited		= 7,
	/*! The prompt does not fit the context the model runs with: its trained context, or the window a load
		option or the request sets. userInfo carries the prompt's tokens under NFKErrorPromptTokenCountKey and
		the limit under NFKErrorContextLimitKey. The request is the problem, so sending it again produces the
		same answer; a caller shortens the prompt, as a client compacts its conversation. Introduced in
		InferKit 0.4.0. */
	kNFKError_InferenceContextExceeded	= 8,
};

/*! The prompt's tokens (NSNumber) in a kNFKError_InferenceContextExceeded error. Introduced in InferKit 0.4.0. */
extern NSString * const NFKErrorPromptTokenCountKey;

/*! The tokens the context or window holds (NSNumber) in a kNFKError_InferenceContextExceeded error.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKErrorContextLimitKey;

NS_ASSUME_NONNULL_END

#endif /* NFKErrors_h */
