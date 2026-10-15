//
//  NFKInferKitCDelivery.m
//  InferKit
//

#import "NFKInferKitCDelivery.h"

void NFKCDeliverCompletion(nfk_completion_callback completion, void * _Nullable context,
						   CF_CONSUMED CFTypeRef _Nullable result, CF_CONSUMED CFTypeRef _Nullable error)
{
	completion(context, (nfk_result *)result, (nfk_error *)error);
}
