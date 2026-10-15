//
//  NFKAnthropicBackendPrivate.h
//  InferKit
//
//  The count_tokens body NFKAnthropicBackend and NFKRemoteTokenCounter both send. Private to the core.
//

#import <InferKit/NFKAnthropicBackend.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKAnthropicBackend ()

/*! The count_tokens body for request: the body a run sends (the system turn lifted, the tool
	conversation converted to tool_use and tool_result blocks, the media attached, the tools, and the
	thinking), reduced to the fields count_tokens takes. Nil with the error a run would report. */
- (nullable NSDictionary<NSString *, id> *)countTokensBodyForRequest:(NFKInferenceRequest *)request
															   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END
