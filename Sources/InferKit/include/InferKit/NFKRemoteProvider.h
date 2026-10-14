//
//  NFKRemoteProvider.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKPinnedServers.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteModel;

/*! The timeout the discovery calls give each probe when a caller names none: two seconds, which a
	server on this machine answers well inside. Introduced in InferKit 0.3.1. */
extern const NSTimeInterval NFKRemoteProviderProbeTimeout;

/*!
	@enum       NFKRemoteAPIStyle
	@abstract   The wire protocol a provider speaks.
	@discussion Most hosted services and every local server expose the OpenAI chat-completions shape,
				so one backend serves them all and a provider is only a URL, a key, and a model name.
				Anthropic's Messages API differs in its authentication header, its required max-tokens
				field, its separate system prompt, and its response envelope, so it has its own backend.
				TypeSafe's System One API answers typed decisions rather than text, so it has its own
				backend too. An InferKit server speaks the OpenAI shape and a native route that carries
				a whole request, so its backend is the native one. A gateway such as OpenCode serves
				each model on its maker's protocol, so its preset names a style per model
				(apiStyleForModelName:), and the Responses and generateContent styles appear only there.
*/
typedef NS_ENUM(NSInteger, NFKRemoteAPIStyle) {
	/*! POST /chat/completions with a Bearer token; the reply carries choices[0].message.content. */
	NFKRemoteAPIStyleOpenAIChat = 0,
	/*! POST /messages with an x-api-key header; the reply carries content[0].text. */
	NFKRemoteAPIStyleAnthropicMessages = 1,
	/*! POST /systemone with a Bearer token; the reply carries typed answers rather than text.
		Introduced in InferKit 0.4.0. */
	NFKRemoteAPIStyleSystemOne = 2,
	/*! An NFKInferenceServer: the OpenAI-compatible routes plus POST /inferkit/run, which carries a
		whole NFKInferenceRequest and is what NFKRemoteInferKitBackend posts to. Introduced in
		InferKit 0.4.0. */
	NFKRemoteAPIStyleInferKit = 3,
	/*! POST /responses with a Bearer token, served by NFKRemoteResponsesBackend. Introduced in
		InferKit 0.4.0. */
	NFKRemoteAPIStyleOpenAIResponses = 4,
	/*! Google's native POST /models/{model}:generateContent with an x-goog-api-key header, served by
		NFKGeminiGenerateContentBackend. Introduced in InferKit 0.4.0. */
	NFKRemoteAPIStyleGeminiGenerateContent = 5,
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

/*! The endpoint the backend posts to: the base plus /chat/completions, /messages for Anthropic,
	/systemone for TypeSafe, or /inferkit/run for an InferKit server. */
@property (nonatomic, copy, readonly) NSURL *endpointURL;

/*! Where the provider lists the models it serves: the base plus /models. */
@property (nonatomic, copy, readonly) NSURL *modelsURL;

/*! The wire protocol this provider speaks, and the one a model without its own route uses. */
@property (nonatomic, assign, readonly) NFKRemoteAPIStyle apiStyle;

/*! The models served on a protocol other than apiStyle, keyed by model name, each value an
	NFKRemoteAPIStyle. Empty for a provider that speaks one protocol. The OpenCode presets carry the
	routes OpenCode documents; providerWithAPIStyle:forModelName: adds or replaces one. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSNumber *> *modelAPIStyles;

/*! Whether a key is required. A local server does not need one. */
@property (nonatomic, assign, readonly) BOOL requiresAPIKey;

/*! What a discovered InferKit server advertised in its Bonjour record: path, tls, auth, and version,
	and, when the server reports host details, chip and memory (physical bytes). Empty for a preset
	and for a provider re-pointed with providerWithBaseURL:. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSString *> *advertisedProperties;

/*! The SHA-256 fingerprint of the server certificate trusted in place of the system's trust store by
	every client a class builds from this provider: backends, the realtime session, stores, reporters,
	runners, and the model catalog. Nil trusts the system's store. Set it with
	providerWithPinnedCertificateSHA256:, or let discovery set it (see discoveryPinnedServers).
	Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, readonly, nullable) NSString *pinnedCertificateSHA256;

/*! What automatic pinning did when discovery made this provider. NFKDiscoveredPinningChanged means
	the server's certificate fingerprint differs from the one remembered for its service name, so
	pinnedCertificateSHA256 is nil and advertisedProperties' cert_sha256 is the new fingerprint.
	NFKDiscoveredPinningNone for a preset and for a provider derived with providerWithBaseURL:,
	providerWithPinnedCertificateSHA256:, or providerWithAPIStyle:forModelName:. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign, readonly) NFKDiscoveredPinning discoveredPinning;

- (instancetype)init NS_UNAVAILABLE;

/*! Two providers are equal when their identifier, base, protocol, model routes, key requirement, and
	certificate pin match. Pins compare as fingerprints, so the colon-separated uppercase and the plain
	lowercase spellings of one fingerprint are equal. Each preset getter builds a new instance and a
	discovered provider is a fresh one, so equality is by value. Introduced in InferKit 0.3.1. */
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
	@discussion The identifier, display name, protocol, model routes, and key requirement are kept;
				every derived URL follows the new base, and a certificate pin is dropped. A runner on another port, or on another machine
				on the network, is the same preset with this one field changed. Introduced in InferKit
				0.3.0.
*/
- (instancetype)providerWithBaseURL:(NSURL *)baseURL;

/*!
	@method     providerWithPinnedCertificateSHA256:
	@abstract   This provider trusting only the server certificate with this SHA-256 fingerprint.
	@discussion Every other field is kept; nil removes the pin. Every factory that takes a provider
				(backendForProvider:apiKey:modelName: here and on each remote backend class,
				NFKRealtimeSession's sessionForProvider:apiStyle:apiKey:modelName:, the store, reporter,
				counter, reranker, and voice-library factories, the local runners, and
				NFKRemoteModelCatalog's catalogForProvider:apiKey:) then uses NFKRemoteTransport's
				sessionWithPinnedCertificateSHA256: for the fingerprint, which sends only to an https
				base. A factory whose service has a fixed host of its own (a usage reporter, Gemini's
				file and retrieval stores) sends there through the same pin. providerWithBaseURL: drops the pin, since a pin names one server's certificate.
				For an NFKInferenceServer serving an NFKServerIdentity, the fingerprint is the
				identity's SHA256Fingerprint. Introduced in InferKit 0.4.0.
*/
- (instancetype)providerWithPinnedCertificateSHA256:(nullable NSString *)fingerprint;

/*!
	@method     apiStyleForModelName:
	@abstract   The protocol this provider serves a model on.
	@discussion The model's entry in modelAPIStyles, or apiStyle for a model without one and for nil.
				backendForProvider:apiKey:modelName: builds the backend this names. Introduced in
				InferKit 0.4.0.
*/
- (NFKRemoteAPIStyle)apiStyleForModelName:(nullable NSString *)modelName;

/*!
	@method     endpointURLForModelName:
	@abstract   The endpoint a request for this model posts to.
	@discussion The base plus the path of apiStyleForModelName:. The generateContent style names the
				model in the path (models/{model}:generateContent). Introduced in InferKit 0.4.0.
*/
- (NSURL *)endpointURLForModelName:(nullable NSString *)modelName;

/*!
	@method     providerWithAPIStyle:forModelName:
	@abstract   This provider with one model routed to another protocol.
	@discussion Every other field is kept. A model a gateway added after this release, or one whose
				route changed, is reached this way until the shipped routes catch up; routing a model to
				apiStyle removes its entry. Introduced in InferKit 0.4.0.
*/
- (instancetype)providerWithAPIStyle:(NFKRemoteAPIStyle)apiStyle forModelName:(NSString *)modelName;

/*! Every preset this release ships. */
@property (class, nonatomic, copy, readonly) NSArray<NFKRemoteProvider *> *allProviders;

/*! The preset with this identifier, or nil. */
+ (nullable NFKRemoteProvider *)providerWithIdentifier:(NSString *)identifier;

/*! Every local-server preset, in the order the discovery calls probe them: ollama, lmstudio,
	llamacpp, vllm, inferkit. Introduced in InferKit 0.3.1. */
@property (class, nonatomic, copy, readonly) NSArray<NFKRemoteProvider *> *localProviders;

/*!
	@method     isReachableWithAPIKey:timeout:error:
	@abstract   Whether a server answers at this provider's address now.
	@discussion Reads modelsURL through NFKRemoteModelCatalog and counts any HTTP reply, a rejected
				key included: the question is whether a server is there, which for a local runner is
				whether the app is running. Returns NO with kNFKError_RemoteUnreachable when nothing
				answered. Blocks for at most the timeout; run it off the render thread.

				This is the seam every discovery call goes through, so a subclass that overrides it
				decides what "available" means. Introduced in InferKit 0.3.1.
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
				on the port is the answer that matters. Blocks. Introduced in InferKit 0.3.1.
*/
+ (NSArray<NFKRemoteProvider *> *)availableProvidersAmong:(NSArray<NFKRemoteProvider *> *)providers
												  timeout:(NSTimeInterval)timeout
	NS_SWIFT_NAME(availableProviders(among:timeout:));

/*!
	@method     firstAvailableProviderAmong:timeout:
	@abstract   The first provider in the list that answers, or nil when none does.
	@discussion Probes in the list's order and stops at the first reply, so a running server on the
				first address costs one probe. Blocks. Introduced in InferKit 0.3.1.
*/
+ (nullable NFKRemoteProvider *)firstAvailableProviderAmong:(NSArray<NFKRemoteProvider *> *)providers
													timeout:(NSTimeInterval)timeout
	NS_SWIFT_NAME(firstAvailableProvider(among:timeout:));

/*!
	@method     availableLocalProviders
	@abstract   The local servers running on this machine right now, in localProviders order.
	@discussion localProviders probed at NFKRemoteProviderProbeTimeout, which is what fills a picker
				of the runners a user can pick from. Blocks; run it off the render thread, or use the
				completion-handler form. Introduced in InferKit 0.3.1.
*/
+ (NSArray<NFKRemoteProvider *> *)availableLocalProviders;

/*!
	@method     firstAvailableLocalProvider
	@abstract   The first local server that answers, or nil when none is running.
	@discussion The call that removes the choice from the calling code: rather than naming Ollama or
				LM Studio, ask which one is up and use it. The order is localProviders order, so the
				answer is the same for the same machine. Blocks. Introduced in InferKit 0.3.1.
*/
+ (nullable NFKRemoteProvider *)firstAvailableLocalProvider NS_SWIFT_NAME(firstAvailableLocalProvider());

/*! The asynchronous form of availableLocalProviders. The handler runs on a background queue at
	user-initiated quality of service. Swift's async import of a completion handler drops the
	handler from the name, which would take the blocking call's name and leave it unreachable, so
	the awaited form is named probeAvailableLocalProviders(). Introduced in InferKit 0.3.1. */
+ (void)availableLocalProvidersWithCompletionHandler:(void (^ NS_SWIFT_SENDABLE)(NSArray<NFKRemoteProvider *> *providers))completionHandler
	NS_SWIFT_ASYNC_NAME(probeAvailableLocalProviders());

/*! The asynchronous form of firstAvailableLocalProvider, named probeFirstAvailableLocalProvider()
	in Swift for the reason above. The handler runs on a background queue at user-initiated quality
	of service. Introduced in InferKit 0.3.1. */
+ (void)firstAvailableLocalProviderWithCompletionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKRemoteProvider * _Nullable provider))completionHandler
	NS_SWIFT_ASYNC_NAME(probeFirstAvailableLocalProvider());

/*!
	@method     backendForFirstAvailableLocalProviderWithModelName:
	@abstract   A backend on the first local server that answers, or nil when none is running.
	@discussion firstAvailableLocalProvider followed by backendForProvider:apiKey:modelName:, for
				code that wants a local backend without naming the app behind it. The model name is
				the caller's, since the runners name their models differently; llama.cpp serves
				whatever it has loaded, so nil is a working argument there. A caller that needs to
				know which runner it got asks firstAvailableLocalProvider instead and builds the
				backend from it. Blocks. Introduced in InferKit 0.3.1.
*/
+ (nullable id<NFKInferenceBackend>)backendForFirstAvailableLocalProviderWithModelName:(nullable NSString *)modelName;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   Builds the backend a provider needs, already pointed at its endpoint.
	@discussion Returns an NFKRemoteBackend for an OpenAI-compatible provider, an
				NFKAnthropicBackend for Anthropic, an NFKTypeSafeBackend for TypeSafe, and an
				NFKRemoteInferKitBackend for an InferKit server. The model name is required except
				on llama.cpp and on an InferKit server that hosts one model: see the class discussion.

				The backend follows apiStyleForModelName:, so on a gateway the model picks it: an
				NFKRemoteResponsesBackend for the Responses style and an
				NFKGeminiGenerateContentBackend for the generateContent style.
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
	   completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSArray<NFKRemoteModel *> * _Nullable models,
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
/*! TypeSafe AI's System One API, which serves Jev: typed decisions about a state rather than text.
	Served by NFKTypeSafeBackend. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NFKRemoteProvider *typeSafe;
/*! OpenCode Zen, the pay-as-you-go gateway of the OpenCode console, at
	https://opencode.ai/inference/v1. One key (OPENCODE_API_KEY in OpenCode's own tools) reaches models
	from many makers, each on its maker's protocol: modelAPIStyles routes the GPT, Grok, and Muse
	Spark models to Responses, Claude and most Qwen models to Messages, Jev to System One, and Gemini
	to generateContent. A model without a route goes to chat completions. The routes are OpenCode's
	published table of 2026-10-13. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NFKRemoteProvider *openCodeZen;
/*! OpenCode Go, the subscription gateway, at https://opencode.ai/inference/go/v1, with the same key
	as Zen. Go routes some model names differently from Zen (Qwen3.8 Max and the MiniMax models are on
	Messages here), so each preset carries its own table. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NFKRemoteProvider *openCodeGo;

// Local servers. Each is the project's own default address; nothing is assumed to be running.
@property (class, nonatomic, readonly) NFKRemoteProvider *ollama;
@property (class, nonatomic, readonly) NFKRemoteProvider *lmStudio;
@property (class, nonatomic, readonly) NFKRemoteProvider *llamaCpp;
@property (class, nonatomic, readonly) NFKRemoteProvider *vLLM;
/*! An NFKInferenceServer on this machine at NFKInferenceServerDefaultPort. A server elsewhere is this
	preset re-pointed with providerWithBaseURL:, or one discoverInferKitServersWithTimeout: found.
	Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NFKRemoteProvider *inferKit;

/*!
	@property   discoveryPinnedServers
	@abstract   The fingerprints automatic pinning remembers, or nil to leave discovered servers
				unpinned. Defaults to nil.
	@discussion Setting an NFKPinnedServers turns automatic pinning on for
				discoverInferKitServersWithTimeout:, its completion-handler form, and the servers
				NFKBalancedBackend's startDiscoveryWithInterval: finds. A discovered TLS server is then
				pinned to the cert_sha256 fingerprint its Bonjour record advertises, on the
				trust-on-first-use terms NFKPinnedServers describes, and discoveredPinning reports the
				decision. Plain HTTP servers that were never pinned are unaffected. The app keeps the
				set across launches by saving its dictionaryRepresentation and restoring it here at
				launch. Introduced in InferKit 0.4.0.
*/
@property (class, atomic, strong, nullable) NFKPinnedServers *discoveryPinnedServers;

/*!
	@method     providerForInferKitServiceNamed:host:port:TXTRecord:pinnedServers:
	@abstract   The provider discovery makes for one resolved Bonjour advertisement.
	@discussion For an app that browses NFKInferenceServerServiceType itself (with NWBrowser, for
				example) and wants the provider discoverInferKitServersWithTimeout: would make: the
				inferkit preset at host and port, https when TXTRecord's tls is 1, at TXTRecord's path
				or /v1. displayName is name, requiresAPIKey is TXTRecord's auth, and
				advertisedProperties is TXTRecord. With pinnedServers, the provider is pinned
				automatically as discoveryPinnedServers describes; nil leaves it unpinned. nil when
				host is empty or port is not a TCP port. Introduced in InferKit 0.4.0.
*/
+ (nullable instancetype)providerForInferKitServiceNamed:(NSString *)name
													host:(NSString *)host
													port:(NSInteger)port
											   TXTRecord:(NSDictionary<NSString *, NSString *> *)TXTRecord
										   pinnedServers:(nullable NFKPinnedServers *)pinnedServers;

/*!
	@method     discoverInferKitServersWithTimeout:
	@abstract   The InferKit servers advertising on the local network, found over Bonjour.
	@discussion Browses NFKInferenceServerServiceType for the timeout and resolves each advertisement
				to a provider: the inferkit preset at the advertising machine's host name and port,
				with https when the server serves TLS. displayName is the advertised name, and
				requiresAPIKey is what the server advertised. The answer is sorted by name; a
				server on this machine is included when it advertises. With discoveryPinnedServers set,
				each TLS provider is pinned automatically and reports the decision in
				discoveredPinning; a server whose fingerprint changed is still listed, unpinned, so the
				app can ask its user. Blocks for the timeout; run it off the render thread, or use the
				completion-handler form. On iOS, the app lists NFKInferenceServerServiceType under
				NSBonjourServices and carries NSLocalNetworkUsageDescription. Introduced in InferKit
				0.4.0.
*/
+ (NSArray<NFKRemoteProvider *> *)discoverInferKitServersWithTimeout:(NSTimeInterval)timeout
	NS_SWIFT_NAME(discoverInferKitServers(timeout:));

/*! The asynchronous form of discoverInferKitServersWithTimeout:, named probeInferKitServers(timeout:)
	in Swift so the awaited form does not take the blocking call's name. The handler runs on a
	background queue at user-initiated quality of service. Introduced in InferKit 0.4.0. */
+ (void)discoverInferKitServersWithTimeout:(NSTimeInterval)timeout
						 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSArray<NFKRemoteProvider *> *providers))completionHandler
	NS_SWIFT_ASYNC_NAME(probeInferKitServers(timeout:));

@end

NS_ASSUME_NONNULL_END
