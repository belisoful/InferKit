//
//  NFKRemoteProvider.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteModel;

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
	@discussion A preset carries the provider's API base and its protocol, which are stable, and
				derives each operation's URL from the base: endpointURL for chat and modelsURL for the
				model list. It deliberately carries NO default model name: model identifiers change
				faster than a release does, and a stale default fails at the first call with a message
				about the model rather than about the default. modelsWithAPIKey:error: asks the
				provider for its current list; a caller sets modelName from one of those.

				Local servers (Ollama, LM Studio, llama.cpp, vLLM) need no key, which is what
				requiresAPIKey reports, and each preset names the project's own default address.
				providerWithBaseURL: re-points a preset at another port or another machine, keeping
				its identity and protocol. Introduced in InferKit 0.1.0.
*/
@interface NFKRemoteProvider : NSObject <NSCopying>

/*! A stable short name, for example "openai" or "ollama". */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! A human-readable name. */
@property (nonatomic, copy, readonly) NSString *displayName;

/*! The API base every operation's URL is built on, for example https://api.openai.com/v1. Introduced in InferKit 0.3.0. */
@property (nonatomic, copy, readonly) NSURL *baseURL;

/*! The chat endpoint the backend posts to: the base plus /chat/completions, or /messages for Anthropic. */
@property (nonatomic, copy, readonly) NSURL *endpointURL;

/*! Where the provider lists the models it serves: the base plus /models. */
@property (nonatomic, copy, readonly) NSURL *modelsURL;

/*! The wire protocol this provider speaks. */
@property (nonatomic, assign, readonly) NFKRemoteAPIStyle apiStyle;

/*! Whether a key is required. A local server does not need one. */
@property (nonatomic, assign, readonly) BOOL requiresAPIKey;

- (instancetype)init NS_UNAVAILABLE;

/*!
	@method     URLForPath:
	@abstract   The base with a path appended, for an operation this class does not name.
	@discussion Whether the provider serves that path is the provider's documentation to say; this
				only builds the URL, so a caller pointing NFKRemoteTranscriptionBackend at a provider
				writes URLForPath:@"audio/transcriptions" rather than a hand-typed address.
				Introduced in InferKit 0.3.0.
*/
- (NSURL *)URLForPath:(NSString *)path;

/*!
	@method     providerWithBaseURL:
	@abstract   This provider at another address.
	@discussion The identifier, display name, protocol, and key requirement are kept; every derived
				URL follows the new base. A runner on another port, or on another machine on the
				network, is the same preset with this one field changed. Introduced in InferKit 0.3.0.
*/
- (instancetype)providerWithBaseURL:(NSURL *)baseURL;

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

/*!
	@method     modelsWithAPIKey:error:
	@abstract   The models the provider currently serves, or nil with an error.
	@discussion Reads modelsURL through NFKRemoteModelCatalog, which is the object to use for a
				timeout, a session, or a stubbed transport. Blocks; run it off the render thread. A
				local runner that is not running fails with kNFKError_RemoteUnreachable.
				Introduced in InferKit 0.3.0.
*/
- (nullable NSArray<NFKRemoteModel *> *)modelsWithAPIKey:(nullable NSString *)apiKey
												   error:(NSError * _Nullable *)outError;

/*! The asynchronous form of modelsWithAPIKey:error:. The handler runs on a background queue. Introduced in InferKit 0.3.0. */
- (void)modelsWithAPIKey:(nullable NSString *)apiKey
	   completionHandler:(void (^)(NSArray<NFKRemoteModel *> * _Nullable models,
								   NSError * _Nullable error))completionHandler;

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
