//
//  NFKRemoteModerationBackend.h
//  InferKit
//

#ifndef NFKRemoteModerationBackend_h
#define NFKRemoteModerationBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteModerationBackend
	@abstract   An inference backend that classifies text, and an image where the service reads one,
				against a provider's moderation categories.
	@discussion POST /moderations. The request reads NFKInputPrompt (or the joined content of
				NFKInputMessages) and, on a service that moderates images, NFKInputImage beside it.
				The reply's per-category scores come back as NFKClassifications under
				NFKOutputClassifications, most confident first, and the service's whole verdict for
				the input under NFKOutputStructured, where "flagged" says whether any category
				tripped. Verified to be served by openai and mistral at release. Blocks; run it off
				the render thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteModerationBackend : NSObject <NFKInferenceBackend>

/*! The moderation endpoint, for example https://api.openai.com/v1/moderations. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body, where the service takes one. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A backend pointed at the provider's moderation endpoint, or nil for Anthropic. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteModerationBackend_h */
