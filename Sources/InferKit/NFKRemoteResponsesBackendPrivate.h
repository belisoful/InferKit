//
//  NFKRemoteResponsesBackendPrivate.h
//  InferKit
//
//  The Responses body NFKRemoteTokenCounter sends to a service's input_tokens route. Private to the core.
//

#import <InferKit/NFKRemoteResponsesBackend.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKRemoteResponsesBackend ()

/*! The input_tokens body for request: the body a blocking run sends (the system turn as instructions,
	the input items with their media, the tools, and the parameters), which the route takes whole.
	Nil with the error a run would report. */
- (nullable NSDictionary<NSString *, id> *)inputTokensBodyForRequest:(NFKInferenceRequest *)request
															   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END
