//
//  NFKRemoteProvider.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteModel;

/*! The timeout the discovery calls give each probe when a caller names none: two seconds, which a
	server on this machine answers well inside. Introduced in InferKit 0.4.0. */
extern const NSTimeInterval NFKRemoteProviderProbeTimeout;

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

/*! Two providers are equal when their identifier, base, protocol, and key requirement match. Each
	preset getter builds a new instance and a discovered provider is a fresh one, so equality is by
	value rather than by identity. Introduced in InferKit 0.4.0. */
- (BOOL)isEqual:(nullable id)object;

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

/*! Every local-server preset, in the order the discovery calls probe them: ollama, lmstudio,
	llamacpp, vllm. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, copy, readonly) NSArray<NFKRemoteProvider *> *localProviders;

/*!
	@method     isReachableWithAPIKey:timeout:error:
	@abstract   Whether a server answers at this provider's address now.
	@discussion Reads modelsURL through NFKRemoteModelCatalog and counts any HTTP reply, a rejected
				key included: the question is whether a server is there, which for a local runner is
				whether the app is running. Returns NO with kNFKError_RemoteUnreachable when nothing
				answered. Blocks for at most the timeout; run it off the render thread.

				This is the seam every discovery call goes through, so a subclass that overrides it
				decides what "available" means. Introduced in InferKit 0.4.0.
*/
- (BOOL)isReachableWithAPIKey:(nullable NSString *)apiKey
					  timeout:(NSTimeInterval)timeout
						error:(NSError * _Nullable *)outError;

/*!
	@method     availableProvidersAmong:timeout:
	@abstract   The providers in the list that answer, in the list's own order.
	@discussion The probes run concurrently, so the call costs one timeout rather than one per
				provider, and no key is sent. A hosted provider answers 401 without one, which still
				counts as reachable, so this is aimed at the local servers, where nothing listening
				on the port is the answer that matters. Blocks. Introduced in InferKit 0.4.0.
*/
+ (NSArray<NFKRemoteProvider *> *)availableProvidersAmong:(NSArray<NFKRemoteProvider *> *)providers
												  timeout:(NSTimeInterval)timeout
	NS_SWIFT_NAME(availableProviders(among:timeout:));

/*!
	@method     firstAvailableProviderAmong:timeout:
	@abstract   The first provider in the list that answers, or nil when none does.
	@discussion Probes in the list's order and stops at the first reply, so a running server on the
				first address costs one probe. Blocks. Introduced in InferKit 0.4.0.
*/
+ (nullable NFKRemoteProvider *)firstAvailableProviderAmong:(NSArray<NFKRemoteProvider *> *)providers
													timeout:(NSTimeInterval)timeout
	NS_SWIFT_NAME(firstAvailableProvider(among:timeout:));

/*!
	@method     availableLocalProviders
	@abstract   The local servers running on this machine right now, in localProviders order.
	@discussion localProviders probed at NFKRemoteProviderProbeTimeout, which is what fills a picker
				of the runners a user can pick from. Blocks; run it off the render thread, or use the
				completion-handler form. Introduced in InferKit 0.4.0.
*/
+ (NSArray<NFKRemoteProvider *> *)availableLocalProviders;

/*!
	@method     firstAvailableLocalProvider
	@abstract   The first local server that answers, or nil when none is running.
	@discussion The call that removes the choice from the calling code: rather than naming Ollama or
				LM Studio, ask which one is up and use it. The order is localProviders order, so the
				answer is the same for the same machine. Blocks. Introduced in InferKit 0.4.0.
*/
+ (nullable NFKRemoteProvider *)firstAvailableLocalProvider NS_SWIFT_NAME(firstAvailableLocalProvider());

/*! The asynchronous form of availableLocalProviders. The handler runs on a background queue at
	user-initiated quality of service. Swift's async import of a completion handler drops the
	handler from the name, which would take the blocking call's name and leave it unreachable, so
	the awaited form is named probeAvailableLocalProviders(). Introduced in InferKit 0.4.0. */
+ (void)availableLocalProvidersWithCompletionHandler:(void (^)(NSArray<NFKRemoteProvider *> *providers))completionHandler
	NS_SWIFT_ASYNC_NAME(probeAvailableLocalProviders());

/*! The asynchronous form of firstAvailableLocalProvider, named probeFirstAvailableLocalProvider()
	in Swift for the reason above. The handler runs on a background queue at user-initiated quality
	of service. Introduced in InferKit 0.4.0. */
+ (void)firstAvailableLocalProviderWithCompletionHandler:(void (^)(NFKRemoteProvider * _Nullable provider))completionHandler
	NS_SWIFT_ASYNC_NAME(probeFirstAvailableLocalProvider());

/*!
	@method     backendForFirstAvailableLocalProviderWithModelName:
	@abstract   A backend on the first local server that answers, or nil when none is running.
	@discussion firstAvailableLocalProvider followed by backendForProvider:apiKey:modelName:, for
				code that wants a local backend without naming the app behind it. The model name is
				the caller's, since the runners name their models differently; llama.cpp serves
				whatever it has loaded, so nil is a working argument there. A caller that needs to
				know which runner it got asks firstAvailableLocalProvider instead and builds the
				backend from it. Blocks. Introduced in InferKit 0.4.0.
*/
+ (nullable id<NFKInferenceBackend>)backendForFirstAvailableLocalProviderWithModelName:(nullable NSString *)modelName;

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
