//
//  NFKInferKitCDelivery.h
//  InferKit
//
//  Hands a submitted run's result and error to the C completion that owns them from then on.
//

#import <Foundation/Foundation.h>

#if __has_include(<InferKitC/NFKInferKitC.h>)
#import <InferKitC/NFKInferKitC.h>
#elif __has_include(<InferKit/NFKInferKitC.h>)
#import <InferKit/NFKInferKitC.h>
#else
#import "../InferKitC/include/InferKitC/NFKInferKitC.h"
#endif

NS_ASSUME_NONNULL_BEGIN

/*! Calls completion with a result handle and an error handle, either of them NULL. The function consumes
	one reference to each: the receiver owns them and frees them with nfk_result_free and nfk_error_free.
	It lives in a file of its own, so a caller analyzed with this declaration sees the reference pass to
	the receiver. */
void NFKCDeliverCompletion(nfk_completion_callback completion, void * _Nullable context,
						   CF_CONSUMED CFTypeRef _Nullable result, CF_CONSUMED CFTypeRef _Nullable error);

NS_ASSUME_NONNULL_END
