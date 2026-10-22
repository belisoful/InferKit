//
//  NFKRemoteProvider.m
//  InferKit
//

#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKTypeSafeBackend.h>
#import <InferKit/NFKRemoteModelCatalog.h>
#import <InferKit/NFKRemoteInferKitBackend.h>
#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKGeminiGenerateContentBackend.h>
#import <InferKit/NFKInferenceServer.h>
#import <InferKit/NFKRemoteTransport.h>
#import "NFKInferKitDiscovery.h"
#import "NFKPinnedServersPrivate.h"
#import "NFKRemoteTransportPrivate.h"

const NSTimeInterval NFKRemoteProviderProbeTimeout = 2.0;

static NSDictionary<NSString *, NSNumber *> *NFKModelAPIStyles(NSDictionary<NSNumber *, NSArray<NSString *> *> *modelsByStyle)
{
	NSMutableDictionary<NSString *, NSNumber *> *styles = [NSMutableDictionary dictionary];
	[modelsByStyle enumerateKeysAndObjectsUsingBlock:^(NSNumber *style, NSArray<NSString *> *models, BOOL *stop) {
		for (NSString *model in models) {
			styles[model] = style;
		}
	}];
	return styles;
}

@interface NFKRemoteProvider ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite) NSString *displayName;
@property (nonatomic, copy, readwrite) NSURL *baseURL;
@property (nonatomic, assign, readwrite) NFKRemoteAPIStyle apiStyle;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSNumber *> *modelAPIStyles;
@property (nonatomic, assign, readwrite) BOOL requiresAPIKey;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSString *> *advertisedProperties;
@property (nonatomic, copy, readwrite, nullable) NSString *pinnedCertificateSHA256;
@property (nonatomic, assign, readwrite) NFKDiscoveredPinning discoveredPinning;
/*! The pin as 64 lowercase hexadecimal digits, so each spelling of one fingerprint compares equal; a
	malformed pin as given. */
@property (nonatomic, readonly, nullable) NSString *comparablePin;
@end

static NFKPinnedServers *NFKRemoteDiscoveryPinnedServers = nil;

@implementation NFKRemoteProvider

+ (instancetype)providerWithIdentifier:(NSString *)identifier
						   displayName:(NSString *)displayName
							   base:(NSString *)base
							  style:(NFKRemoteAPIStyle)style
						requiresKey:(BOOL)requiresKey
{
	NFKRemoteProvider *provider = [[self alloc] initPrivate];
	provider.identifier = identifier;
	provider.displayName = displayName;
	provider.baseURL = [NSURL URLWithString:base];
	provider.apiStyle = style;
	provider.requiresAPIKey = requiresKey;
	return provider;
}

- (instancetype)initPrivate
{
	self = [super init];
	if (self != nil) {
		_advertisedProperties = @{};
		_modelAPIStyles = @{};
	}
	return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

- (instancetype)providerWithBaseURL:(NSURL *)baseURL
{
	NFKRemoteProvider *provider = [[self.class alloc] initPrivate];
	provider.identifier = self.identifier;
	provider.displayName = self.displayName;
	provider.baseURL = baseURL;
	provider.apiStyle = self.apiStyle;
	provider.modelAPIStyles = self.modelAPIStyles;
	provider.requiresAPIKey = self.requiresAPIKey;
	return provider;
}

- (instancetype)providerWithAPIStyle:(NFKRemoteAPIStyle)apiStyle forModelName:(NSString *)modelName
{
	NFKRemoteProvider *provider = [self providerWithBaseURL:self.baseURL];
	NSMutableDictionary<NSString *, NSNumber *> *styles = [self.modelAPIStyles mutableCopy];
	styles[modelName] = apiStyle == self.apiStyle ? nil : @(apiStyle);
	provider.modelAPIStyles = styles;
	provider.pinnedCertificateSHA256 = self.pinnedCertificateSHA256;
	return provider;
}

- (instancetype)providerWithPinnedCertificateSHA256:(nullable NSString *)fingerprint
{
	NFKRemoteProvider *provider = [self providerWithBaseURL:self.baseURL];
	provider.advertisedProperties = self.advertisedProperties;
	provider.pinnedCertificateSHA256 = fingerprint;
	return provider;
}

- (NFKRemoteAPIStyle)apiStyleForModelName:(nullable NSString *)modelName
{
	NSNumber *style = modelName != nil ? self.modelAPIStyles[modelName] : nil;
	return style != nil ? (NFKRemoteAPIStyle)style.integerValue : self.apiStyle;
}

// Each preset getter builds a new instance, and a discovered provider is a fresh one, so a value
// comparison is what a caller means by "is this the Ollama preset".
- (BOOL)isEqual:(nullable id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKRemoteProvider.class]) {
		return NO;
	}
	NFKRemoteProvider *other = object;
	return [self.identifier isEqualToString:other.identifier] &&
		   [self.baseURL isEqual:other.baseURL] &&
		   self.apiStyle == other.apiStyle &&
		   [self.modelAPIStyles isEqualToDictionary:other.modelAPIStyles] &&
		   self.requiresAPIKey == other.requiresAPIKey &&
		   (self.comparablePin == other.comparablePin || [self.comparablePin isEqualToString:other.comparablePin]);
}

- (nullable NSString *)comparablePin
{
	NSString *pin = self.pinnedCertificateSHA256;
	return pin != nil ? (NFKRemoteNormalizedFingerprint(pin) ?: pin) : nil;
}

- (NSUInteger)hash
{
	return self.identifier.hash ^ self.baseURL.hash;
}

#pragma mark Derived URLs

- (NSURL *)URLForPath:(NSString *)path
{
	NSCharacterSet *slash = [NSCharacterSet characterSetWithCharactersInString:@"/"];
	NSString *base = [self.baseURL.absoluteString stringByTrimmingCharactersInSet:slash];
	NSString *relative = [path stringByTrimmingCharactersInSet:slash];
	return [NSURL URLWithString:[NSString stringWithFormat:@"%@/%@", base, relative]];
}

- (NSURL *)endpointURL
{
	return [self endpointURLForAPIStyle:self.apiStyle modelName:nil];
}

- (NSURL *)endpointURLForModelName:(nullable NSString *)modelName
{
	return [self endpointURLForAPIStyle:[self apiStyleForModelName:modelName] modelName:modelName];
}

- (NSURL *)endpointURLForAPIStyle:(NFKRemoteAPIStyle)apiStyle modelName:(nullable NSString *)modelName
{
	switch (apiStyle) {
		case NFKRemoteAPIStyleAnthropicMessages: return [self URLForPath:@"messages"];
		case NFKRemoteAPIStyleSystemOne: return [self URLForPath:@"systemone"];
		case NFKRemoteAPIStyleInferKit: return [self URLForPath:@"inferkit/run"];
		case NFKRemoteAPIStyleOpenAIResponses: return [self URLForPath:@"responses"];
		case NFKRemoteAPIStyleGeminiGenerateContent:
			return [self URLForPath:[NSString stringWithFormat:@"models/%@:generateContent", modelName ?: @""]];
		case NFKRemoteAPIStyleOpenAIChat: break;
	}
	return [self URLForPath:@"chat/completions"];
}

- (NSURL *)modelsURL
{
	return [self URLForPath:@"models"];
}

#pragma mark Hosted

+ (NFKRemoteProvider *)openAI
{
	return [self providerWithIdentifier:@"openai" displayName:@"OpenAI"
								   base:@"https://api.openai.com/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)anthropic
{
	return [self providerWithIdentifier:@"anthropic" displayName:@"Anthropic"
								   base:@"https://api.anthropic.com/v1"
								  style:NFKRemoteAPIStyleAnthropicMessages requiresKey:YES];
}

+ (NFKRemoteProvider *)xAI
{
	return [self providerWithIdentifier:@"xai" displayName:@"xAI Grok"
								   base:@"https://api.x.ai/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)googleGemini
{
	// Gemini's OpenAI-compatible layer, which is what lets one backend serve it.
	return [self providerWithIdentifier:@"gemini" displayName:@"Google Gemini"
								   base:@"https://generativelanguage.googleapis.com/v1beta/openai"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)groq
{
	return [self providerWithIdentifier:@"groq" displayName:@"Groq"
								   base:@"https://api.groq.com/openai/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)mistral
{
	return [self providerWithIdentifier:@"mistral" displayName:@"Mistral"
								   base:@"https://api.mistral.ai/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)deepSeek
{
	return [self providerWithIdentifier:@"deepseek" displayName:@"DeepSeek"
								   base:@"https://api.deepseek.com/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)together
{
	return [self providerWithIdentifier:@"together" displayName:@"Together AI"
								   base:@"https://api.together.xyz/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)openRouter
{
	return [self providerWithIdentifier:@"openrouter" displayName:@"OpenRouter"
								   base:@"https://openrouter.ai/api/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)typeSafe
{
	return [self providerWithIdentifier:@"typesafe" displayName:@"TypeSafe AI"
								   base:@"https://api.typesafe.ai/v1"
								  style:NFKRemoteAPIStyleSystemOne requiresKey:YES];
}

+ (NFKRemoteProvider *)modelStudio
{
	return [self providerWithIdentifier:@"modelstudio" displayName:@"Alibaba Model Studio"
								   base:@"https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

// OpenCode's published endpoint tables (opencode.ai/docs/zen and opencode.ai/docs/go, read
// 2026-10-13), with muse-spark-1.2-contributor-free from models.dev, which OpenCode's own client reads.
// Every model the tables send to chat completions is left out, since that is the presets' apiStyle.
+ (NFKRemoteProvider *)openCodeZen
{
	NFKRemoteProvider *provider = [self providerWithIdentifier:@"opencode" displayName:@"OpenCode Zen"
														  base:@"https://opencode.ai/inference/v1"
														 style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
	provider.modelAPIStyles = NFKModelAPIStyles(@{
		@(NFKRemoteAPIStyleOpenAIResponses): @[
			@"gpt-5", @"gpt-5-codex", @"gpt-5-nano", @"gpt-5.1", @"gpt-5.1-codex",
			@"gpt-5.1-codex-max", @"gpt-5.1-codex-mini", @"gpt-5.2", @"gpt-5.2-codex",
			@"gpt-5.3-codex", @"gpt-5.3-codex-spark", @"gpt-5.4", @"gpt-5.4-mini", @"gpt-5.4-nano",
			@"gpt-5.4-pro", @"gpt-5.5", @"gpt-5.5-pro", @"gpt-5.6-luna", @"gpt-5.6-sol",
			@"gpt-5.6-terra", @"gpt-6-astra", @"gpt-6-luna", @"gpt-6-sol", @"gpt-6.1-sol",
			@"grok-4.5", @"grok-4.6", @"grok-4.7", @"grok-build-0.1", @"muse-spark-1.2",
			@"muse-spark-1.2-contributor-free", @"muse-spark-1.3",
			@"muse-spark-1.3-contributor-free" ],
		@(NFKRemoteAPIStyleAnthropicMessages): @[
			@"claude-fable-5", @"claude-fable-5-1", @"claude-haiku-4-5", @"claude-haiku-5-5",
			@"claude-opus-4-5", @"claude-opus-4-6", @"claude-opus-4-7", @"claude-opus-4-8",
			@"claude-opus-5", @"claude-opus-5-5", @"claude-sonnet-4-5", @"claude-sonnet-4-6",
			@"claude-sonnet-5", @"claude-sonnet-5-5", @"qwen3.5-plus", @"qwen3.6-plus",
			@"qwen3.7-max", @"qwen3.7-plus", @"qwen3.8-flash" ],
		@(NFKRemoteAPIStyleSystemOne): @[
			@"jev-1.13", @"jev-1.13-free" ],
		@(NFKRemoteAPIStyleGeminiGenerateContent): @[
			@"gemini-3-flash", @"gemini-3.1-pro", @"gemini-3.5-flash", @"gemini-3.5-flash-lite",
			@"gemini-3.6-flash", @"gemini-3.7-flash", @"gemini-3.8-flash" ],
	});
	return provider;
}

+ (NFKRemoteProvider *)openCodeGo
{
	NFKRemoteProvider *provider = [self providerWithIdentifier:@"opencode-go" displayName:@"OpenCode Go"
														  base:@"https://opencode.ai/inference/go/v1"
														 style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
	provider.modelAPIStyles = NFKModelAPIStyles(@{
		@(NFKRemoteAPIStyleOpenAIResponses): @[
			@"gpt-5.6-luna", @"gpt-6-luna", @"grok-4.6", @"grok-4.7",
			@"muse-spark-1.2-contributor", @"muse-spark-1.3-contributor" ],
		@(NFKRemoteAPIStyleAnthropicMessages): @[
			@"claude-haiku-5-5", @"minimax-m2.7", @"minimax-m3", @"qwen3.7-plus", @"qwen3.8-flash",
			@"qwen3.8-max" ],
	});
	return provider;
}

#pragma mark Local

+ (NFKRemoteProvider *)ollama
{
	return [self providerWithIdentifier:@"ollama" displayName:@"Ollama"
								   base:@"http://localhost:11434/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)lmStudio
{
	return [self providerWithIdentifier:@"lmstudio" displayName:@"LM Studio"
								   base:@"http://localhost:1234/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)llamaCpp
{
	return [self providerWithIdentifier:@"llamacpp" displayName:@"llama.cpp server"
								   base:@"http://localhost:8080/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)vLLM
{
	return [self providerWithIdentifier:@"vllm" displayName:@"vLLM"
								   base:@"http://localhost:8000/v1"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)inferKit
{
	return [self providerWithIdentifier:@"inferkit" displayName:@"InferKit server"
								   base:[NSString stringWithFormat:@"http://localhost:%u/v1", (unsigned)NFKInferenceServerDefaultPort]
								  style:NFKRemoteAPIStyleInferKit requiresKey:NO];
}

#pragma mark Lookup

+ (NSArray<NFKRemoteProvider *> *)allProviders
{
	return @[ self.openAI, self.anthropic, self.xAI, self.googleGemini, self.groq, self.mistral,
			  self.deepSeek, self.together, self.openRouter, self.typeSafe, self.openCodeZen, self.openCodeGo,
			  self.modelStudio, self.ollama, self.lmStudio, self.llamaCpp, self.vLLM, self.inferKit ];
}

+ (nullable NFKRemoteProvider *)providerWithIdentifier:(NSString *)identifier
{
	for (NFKRemoteProvider *provider in self.allProviders) {
		if ([provider.identifier isEqualToString:identifier]) {
			return provider;
		}
	}
	return nil;
}

#pragma mark Discovery

+ (NSArray<NFKRemoteProvider *> *)localProviders
{
	return @[ self.ollama, self.lmStudio, self.llamaCpp, self.vLLM, self.inferKit ];
}

+ (nullable NFKPinnedServers *)discoveryPinnedServers
{
	@synchronized (NFKRemoteProvider.class) {
		return NFKRemoteDiscoveryPinnedServers;
	}
}

+ (void)setDiscoveryPinnedServers:(nullable NFKPinnedServers *)discoveryPinnedServers
{
	@synchronized (NFKRemoteProvider.class) {
		NFKRemoteDiscoveryPinnedServers = discoveryPinnedServers;
	}
}

+ (nullable instancetype)providerForInferKitServiceNamed:(NSString *)name
													host:(NSString *)host
													port:(NSInteger)port
											   TXTRecord:(NSDictionary<NSString *, NSString *> *)TXTRecord
										   pinnedServers:(nullable NFKPinnedServers *)pinnedServers
{
	if (host.length == 0 || port <= 0 || port > UINT16_MAX) {
		return nil;
	}
	NFKInferKitService *service = [[NFKInferKitService alloc] init];
	service.name = name;
	service.host = host;
	service.port = (uint16_t)port;
	service.TXTRecord = TXTRecord;
	return [self providerForService:service pinnedServers:pinnedServers];
}

+ (nullable instancetype)providerForService:(NFKInferKitService *)service pinnedServers:(nullable NFKPinnedServers *)pinnedServers
{
	NSURL *baseURL = service.baseURL;
	if (baseURL == nil) {
		return nil;
	}
	NFKRemoteProvider *provider = [self.inferKit providerWithBaseURL:baseURL];
	provider.displayName = service.name;
	provider.requiresAPIKey = [service.TXTRecord[@"auth"] isEqualToString:@"1"];
	provider.advertisedProperties = service.TXTRecord ?: @{};
	if (pinnedServers == nil) {
		return provider;
	}
	NSString *pin = nil;
	provider.discoveredPinning = [pinnedServers pinningForServiceName:service.name
															servesTLS:[baseURL.scheme isEqualToString:@"https"]
												advertisedFingerprint:service.TXTRecord[@"cert_sha256"]
													pinnedFingerprint:&pin];
	provider.pinnedCertificateSHA256 = pin;
	return provider;
}

+ (NSArray<NFKRemoteProvider *> *)discoverInferKitServersWithTimeout:(NSTimeInterval)timeout
{
	NFKPinnedServers *pinnedServers = self.discoveryPinnedServers;
	NSMutableArray<NFKRemoteProvider *> *providers = [NSMutableArray array];
	for (NFKInferKitService *service in NFKDiscoverInferKitServices(NFKInferenceServerServiceType, timeout)) {
		NFKRemoteProvider *provider = [self providerForService:service pinnedServers:pinnedServers];
		if (provider != nil) {
			[providers addObject:provider];
		}
	}
	return providers;
}

+ (void)discoverInferKitServersWithTimeout:(NSTimeInterval)timeout
						 completionHandler:(void (^)(NSArray<NFKRemoteProvider *> *))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		completionHandler([self discoverInferKitServersWithTimeout:timeout]);
	});
}

- (BOOL)isReachableWithAPIKey:(nullable NSString *)apiKey
					  timeout:(NSTimeInterval)timeout
						error:(NSError * _Nullable *)outError
{
	NFKRemoteModelCatalog *catalog = [NFKRemoteModelCatalog catalogForProvider:self apiKey:apiKey];
	catalog.timeout = timeout;
	return [catalog isReachableWithError:outError];
}

+ (NSArray<NFKRemoteProvider *> *)availableProvidersAmong:(NSArray<NFKRemoteProvider *> *)providers
												  timeout:(NSTimeInterval)timeout
{
	NSUInteger count = providers.count;
	if (count == 0) {
		return @[];
	}
	NSMutableArray<NSNumber *> *answered = [NSMutableArray arrayWithCapacity:count];
	for (NSUInteger index = 0; index < count; index++) {
		[answered addObject:@NO];
	}
	dispatch_group_t group = dispatch_group_create();
	dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
	for (NSUInteger index = 0; index < count; index++) {
		NFKRemoteProvider *provider = providers[index];
		dispatch_group_async(group, queue, ^{
			BOOL reachable = [provider isReachableWithAPIKey:nil timeout:timeout error:NULL];
			@synchronized (answered) {
				answered[index] = @(reachable);
			}
		});
	}
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);

	NSMutableArray<NFKRemoteProvider *> *available = [NSMutableArray array];
	for (NSUInteger index = 0; index < count; index++) {
		if (answered[index].boolValue) {
			[available addObject:providers[index]];
		}
	}
	return available;
}

+ (nullable NFKRemoteProvider *)firstAvailableProviderAmong:(NSArray<NFKRemoteProvider *> *)providers
													timeout:(NSTimeInterval)timeout
{
	for (NFKRemoteProvider *provider in providers) {
		if ([provider isReachableWithAPIKey:nil timeout:timeout error:NULL]) {
			return provider;
		}
	}
	return nil;
}

+ (NSArray<NFKRemoteProvider *> *)availableLocalProviders
{
	return [self availableProvidersAmong:self.localProviders timeout:NFKRemoteProviderProbeTimeout];
}

+ (nullable NFKRemoteProvider *)firstAvailableLocalProvider
{
	return [self firstAvailableProviderAmong:self.localProviders timeout:NFKRemoteProviderProbeTimeout];
}

+ (void)availableLocalProvidersWithCompletionHandler:(void (^)(NSArray<NFKRemoteProvider *> *))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		completionHandler(self.availableLocalProviders);
	});
}

+ (void)firstAvailableLocalProviderWithCompletionHandler:(void (^)(NFKRemoteProvider * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		completionHandler(self.firstAvailableLocalProvider);
	});
}

+ (nullable id<NFKInferenceBackend>)backendForFirstAvailableLocalProviderWithModelName:(nullable NSString *)modelName
{
	NFKRemoteProvider *provider = self.firstAvailableLocalProvider;
	if (provider == nil) {
		return nil;
	}
	return [self backendForProvider:provider apiKey:nil modelName:modelName];
}

#pragma mark Backends

+ (id<NFKInferenceBackend>)backendForProvider:(NFKRemoteProvider *)provider
									   apiKey:(nullable NSString *)apiKey
									modelName:(nullable NSString *)modelName
{
	id<NFKInferenceBackend> backend = [self unpinnedBackendForProvider:provider apiKey:apiKey modelName:modelName];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		[(id)backend setSession:pinned];
	}
	return backend;
}

+ (id<NFKInferenceBackend>)unpinnedBackendForProvider:(NFKRemoteProvider *)provider
											   apiKey:(nullable NSString *)apiKey
											modelName:(nullable NSString *)modelName
{
	NFKRemoteAPIStyle apiStyle = [provider apiStyleForModelName:modelName];
	NSURL *endpointURL = [provider endpointURLForModelName:modelName];
	if (apiStyle == NFKRemoteAPIStyleAnthropicMessages) {
		NFKAnthropicBackend *backend = [NFKAnthropicBackend backendWithEndpointURL:endpointURL];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	if (apiStyle == NFKRemoteAPIStyleInferKit) {
		NFKRemoteInferKitBackend *backend = [NFKRemoteInferKitBackend backendWithEndpointURL:endpointURL];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	if (apiStyle == NFKRemoteAPIStyleSystemOne) {
		NFKTypeSafeBackend *backend = [NFKTypeSafeBackend backendWithEndpointURL:endpointURL];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	if (apiStyle == NFKRemoteAPIStyleOpenAIResponses) {
		NFKRemoteResponsesBackend *backend = [NFKRemoteResponsesBackend backendWithEndpointURL:endpointURL];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	if (apiStyle == NFKRemoteAPIStyleGeminiGenerateContent) {
		NFKGeminiGenerateContentBackend *backend = [NFKGeminiGenerateContentBackend backendWithEndpointURL:[provider URLForPath:@"models"]];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:endpointURL];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	NSDictionary<NSString *, NSNumber *> *dialects = @{ @"mistral": @(NFKRemoteChatDialectMistral),
														@"openrouter": @(NFKRemoteChatDialectOpenRouter),
														@"vllm": @(NFKRemoteChatDialectVLLM),
														@"llamacpp": @(NFKRemoteChatDialectLlamaCpp),
														@"deepseek": @(NFKRemoteChatDialectDeepSeek) };
	backend.chatDialect = dialects[provider.identifier].integerValue;
	return backend;
}

#pragma mark Models

- (nullable NSArray<NFKRemoteModel *> *)modelsWithAPIKey:(nullable NSString *)apiKey
												   error:(NSError * _Nullable *)outError
{
	return [[NFKRemoteModelCatalog catalogForProvider:self apiKey:apiKey] modelsWithError:outError];
}

- (void)modelsWithAPIKey:(nullable NSString *)apiKey
	   completionHandler:(void (^)(NSArray<NFKRemoteModel *> * _Nullable, NSError * _Nullable))completionHandler
{
	[[NFKRemoteModelCatalog catalogForProvider:self apiKey:apiKey] modelsWithCompletionHandler:completionHandler];
}

@end
