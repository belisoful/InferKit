//
//  NFKRemoteTokenCounter.h
//  InferKit
//

#ifndef NFKRemoteTokenCounter_h
#define NFKRemoteTokenCounter_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;
@class NFKInferenceRequest;

/*!
	@enum       NFKRemoteTokenCounterAPIStyle
	@abstract   The wire shape of a hosted token-counting endpoint.
	@constant   NFKRemoteTokenCounterAPIStyleAnthropic POST /messages/count_tokens with the Messages
				body, answered with input_tokens.
	@constant   NFKRemoteTokenCounterAPIStyleGemini POST models/{model}:countTokens with contents,
				answered with totalTokens.
	@constant   NFKRemoteTokenCounterAPIStyleXAI POST /tokenize-text with model and text, answered with
				token_ids.
	@constant   NFKRemoteTokenCounterAPIStyleLlamaCpp llama.cpp's POST /tokenize with content,
				answered with tokens.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteTokenCounterAPIStyle) {
	NFKRemoteTokenCounterAPIStyleAnthropic = 0,
	NFKRemoteTokenCounterAPIStyleGemini,
	NFKRemoteTokenCounterAPIStyleXAI,
	NFKRemoteTokenCounterAPIStyleLlamaCpp,
};

/*!
	@class      NFKRemoteTokenCounter
	@abstract   Counts the tokens a request costs, as the hosted model's own tokenizer counts them.
	@discussion A request is sized before it is sent: to keep it under a context window, or to
				price it. The count comes from the service, so it matches what the model is billed
				for. The text is NFKInputPrompt, or NFKInputMessages; Anthropic's count takes the
				messages with their roles (a system turn lifted to the top-level field, as the
				Messages backend does), and the other styles count the joined text. A counting
				object rather than a backend, as NFKRemoteReranker is a scoring one. Blocks; run it
				off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteTokenCounter : NSObject

/*! The counting endpoint. For the Gemini style it is the models collection (…/v1beta/models); for
	llama.cpp the server root. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the counter speaks. */
@property (nonatomic, assign) NFKRemoteTokenCounterAPIStyle apiStyle;

/*! The key, sent the way the style's service reads it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model whose tokenizer counts. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A counter for the provider's endpoint: anthropic, gemini, xai, and llamacpp serve one; every other
	preset returns nil. */
+ (nullable instancetype)counterForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The number of tokens the request's text costs, or nil with an error. */
- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError;

/*! The token ids the text becomes, from a service that returns them (xAI, llama.cpp), or nil with
	kNFKError_InferenceUnsupported from one that returns only a count. */
- (nullable NSArray<NSNumber *> *)tokenIdentifiersForText:(NSString *)text error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteTokenCounter_h */
