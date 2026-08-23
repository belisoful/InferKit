//
//  NFKRemoteProvider.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKRemoteAPIStyle
	@abstract   The wire protocol a provider speaks.
	@discussion Most hosted services and every local server expose the OpenAI chat-completions shape,
				so one backend serves them all and a provider is only a URL, a key, and a model name.
				Anthropic's Messages API differs in its authentication header, its required max-tokens
				field, its separate system prompt, and its response envelope, so it has its own backend.
*/
typedef NS_ENUM(NSInteger, NFKRemoteAPIStyle) {
	/*! POST /chat/completions with a Bearer token; the reply carries choices[0].message.content. */
	NFKRemoteAPIStyleOpenAIChat = 0,
	/*! POST /messages with an x-api-key header; the reply carries content[0].text. */
	NFKRemoteAPIStyleAnthropicMessages = 1,
};

/*!
	@class      NFKRemoteProvider
	@abstract   A named endpoint a remote backend can be pointed at.
	@discussion The presets carry the endpoint and the protocol, which are stable, and deliberately
				carry NO default model name: model identifiers change faster than a release does, and a
				stale default fails at the first call with a message about the model rather than about
				the default. Ask the provider for its list — every OpenAI-compatible one answers
				GET /v1/models — or read its documentation, and set modelName explicitly.

				Local servers (Ollama, LM Studio, llama.cpp) need no key, which is what requiresAPIKey
				reports. Introduced in InferKit 0.1.0.
*/
@interface NFKRemoteProvider : NSObject <NSCopying>

/*! A stable short name, for example "openai" or "ollama". */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! A human-readable name. */
@property (nonatomic, copy, readonly) NSString *displayName;

/*! The chat endpoint the backend posts to. */
@property (nonatomic, copy, readonly) NSURL *endpointURL;

/*! The wire protocol this provider speaks. */
@property (nonatomic, assign, readonly) NFKRemoteAPIStyle apiStyle;

/*! Whether a key is required. A local server does not need one. */
@property (nonatomic, assign, readonly) BOOL requiresAPIKey;

/*! Where the provider lists the models it serves, for a caller choosing one. */
@property (nonatomic, copy, readonly, nullable) NSURL *modelsURL;

- (instancetype)init NS_UNAVAILABLE;

/*! Every preset this release ships. */
@property (class, nonatomic, copy, readonly) NSArray<NFKRemoteProvider *> *allProviders;

/*! The preset with this identifier, or nil. */
+ (nullable NFKRemoteProvider *)providerWithIdentifier:(NSString *)identifier;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   Builds the backend a provider needs, already pointed at its endpoint.
	@discussion Returns an NFKRemoteBackend for an OpenAI-compatible provider and an
				NFKAnthropicBackend for Anthropic. The model name is required: see the class discussion.
*/
+ (id<NFKInferenceBackend>)backendForProvider:(NFKRemoteProvider *)provider
									   apiKey:(nullable NSString *)apiKey
									modelName:(nullable NSString *)modelName;

// Hosted providers, each endpoint verified to exist at the time of release.
@property (class, nonatomic, readonly) NFKRemoteProvider *openAI;
@property (class, nonatomic, readonly) NFKRemoteProvider *anthropic;
@property (class, nonatomic, readonly) NFKRemoteProvider *xAI;
@property (class, nonatomic, readonly) NFKRemoteProvider *googleGemini;
@property (class, nonatomic, readonly) NFKRemoteProvider *groq;
@property (class, nonatomic, readonly) NFKRemoteProvider *mistral;
@property (class, nonatomic, readonly) NFKRemoteProvider *deepSeek;
@property (class, nonatomic, readonly) NFKRemoteProvider *together;
@property (class, nonatomic, readonly) NFKRemoteProvider *openRouter;

// Local servers. Each is the project's own default address; nothing is assumed to be running.
@property (class, nonatomic, readonly) NFKRemoteProvider *ollama;
@property (class, nonatomic, readonly) NFKRemoteProvider *lmStudio;
@property (class, nonatomic, readonly) NFKRemoteProvider *llamaCpp;
@property (class, nonatomic, readonly) NFKRemoteProvider *vLLM;

@end

NS_ASSUME_NONNULL_END
