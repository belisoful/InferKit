//
//  NFKRemoteCompletionBackend.h
//  InferKit
//

#ifndef NFKRemoteCompletionBackend_h
#define NFKRemoteCompletionBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteCompletionAPIStyle
	@abstract   The wire shape of a hosted raw-text completion or fill-in-the-middle service.
	@constant   NFKRemoteCompletionAPIStyleOpenAI POST /completions with prompt and suffix, answered
				with choices[].text; the shape Together, vLLM, Ollama, LM Studio, DeepSeek's beta
				path, and OpenAI's legacy models serve.
	@constant   NFKRemoteCompletionAPIStyleMistral Mistral's POST /fim/completions with prompt and
				suffix, answered in the chat shape (choices[].message.content).
	@constant   NFKRemoteCompletionAPIStyleLlamaCpp llama.cpp's native POST /completion, and
				/infill (input_prefix, input_suffix) when the request carries a suffix, answered
				with content.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteCompletionAPIStyle) {
	NFKRemoteCompletionAPIStyleOpenAI = 0,
	NFKRemoteCompletionAPIStyleMistral,
	NFKRemoteCompletionAPIStyleLlamaCpp,
};

/*!
	@class      NFKRemoteCompletionBackend
	@abstract   Continues raw text, or fills the gap between a prefix and a suffix, through a hosted
				completion endpoint.
	@discussion The text model's other mode: no chat template, no roles. NFKInputPrompt is the text
				to continue, or the code before the gap; NFKInputSuffix, when present, is the code
				after it, and the model writes what goes between (fill-in-the-middle, which code
				editors use). The continuation comes back under NFKOutputText and the parsed reply
				under NFKRemoteBackendRawKey. NFKParameterMaxTokens, NFKParameterTemperature,
				NFKParameterTopP, NFKParameterTopK, NFKParameterStopSequences, and NFKParameterSeed
				map onto each style's own fields; every other parameter goes out under its own name.
				submitInferenceJobForRequest: streams the continuation into the job's partialResult.
				runInferenceForRequest: blocks; run it off the render thread. Introduced in InferKit
				0.4.0.
*/
@interface NFKRemoteCompletionBackend : NSObject <NFKInferenceBackend>

/*! The completion endpoint, for example https://api.together.xyz/v1/completions. For the llama.cpp
	style it is the server root, and the backend appends completion or infill. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the backend speaks. Defaults to NFKRemoteCompletionAPIStyleOpenAI. */
@property (nonatomic, assign) NFKRemoteCompletionAPIStyle apiStyle;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 120. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's completion endpoint in its style, or nil for a
				provider that serves none.
	@discussion openai (legacy models), together, vllm, ollama, and lmstudio serve /completions;
				deepseek serves /beta/completions; mistral /fim/completions; llamacpp its native
				/completion and /infill. anthropic, gemini, xai, groq, openrouter, and typesafe
				return nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Streams the continuation into the job's partialResult; cancelling the job closes the connection. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and
	returns the block that cancels the request. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteCompletionBackend_h */
