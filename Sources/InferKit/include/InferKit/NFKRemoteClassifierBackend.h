//
//  NFKRemoteClassifierBackend.h
//  InferKit
//

#ifndef NFKRemoteClassifierBackend_h
#define NFKRemoteClassifierBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteClassifierAPIStyle
	@abstract   The wire shape of a hosted text-classification endpoint.
	@constant   NFKRemoteClassifierAPIStyleMistral Mistral's /classifications and, for a
				conversation, /chat/classifications, answered with results[] of {target: {scores}}.
	@constant   NFKRemoteClassifierAPIStyleVLLM vLLM's /classify, answered with data[] of {label,
				probs}.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteClassifierAPIStyle) {
	NFKRemoteClassifierAPIStyleMistral = 0,
	NFKRemoteClassifierAPIStyleVLLM,
};

/*!
	@class      NFKRemoteClassifierBackend
	@abstract   Classifies text, or a conversation, through a hosted classifier model.
	@discussion The request reads NFKInputPrompt, or NFKInputMessages, which Mistral classifies as a
				conversation. Each class's score comes back as an NFKClassification under
				NFKOutputClassifications, most confident first. A Mistral classifier trained on
				several targets names each class target/label. The parsed reply rides under
				NFKRemoteBackendRawKey; every parameter goes out under its own name. The hosted
				counterpart of the on-device classifiers. runInferenceForRequest: blocks; run it off
				the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteClassifierBackend : NSObject <NFKInferenceBackend>

/*! The classification endpoint, for example https://api.mistral.ai/v1/classifications. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the backend speaks. */
@property (nonatomic, assign) NFKRemoteClassifierAPIStyle apiStyle;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The classifier model, for example a fine-tuned Mistral classifier's id. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A backend for the provider's classification endpoint: mistral and vllm serve one; every other
	preset returns nil. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteClassifierBackend_h */
