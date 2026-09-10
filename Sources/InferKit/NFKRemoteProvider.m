//
//  NFKRemoteProvider.m
//  InferKit
//

#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteModelCatalog.h>

const NSTimeInterval NFKRemoteProviderProbeTimeout = 2.0;

@interface NFKRemoteProvider ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite) NSString *displayName;
@property (nonatomic, copy, readwrite) NSURL *baseURL;
@property (nonatomic, assign, readwrite) NFKRemoteAPIStyle apiStyle;
@property (nonatomic, assign, readwrite) BOOL requiresAPIKey;
@end

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
	return [super init];
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
	provider.requiresAPIKey = self.requiresAPIKey;
	return provider;
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
		   self.requiresAPIKey == other.requiresAPIKey;
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
	return [self URLForPath:self.apiStyle == NFKRemoteAPIStyleAnthropicMessages ? @"messages" : @"chat/completions"];
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

#pragma mark Lookup

+ (NSArray<NFKRemoteProvider *> *)allProviders
{
	return @[ self.openAI, self.anthropic, self.xAI, self.googleGemini, self.groq, self.mistral,
			  self.deepSeek, self.together, self.openRouter,
			  self.ollama, self.lmStudio, self.llamaCpp, self.vLLM ];
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
	return @[ self.ollama, self.lmStudio, self.llamaCpp, self.vLLM ];
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
	if (provider.apiStyle == NFKRemoteAPIStyleAnthropicMessages) {
		NFKAnthropicBackend *backend = [NFKAnthropicBackend backendWithEndpointURL:provider.endpointURL];
		backend.apiKey = apiKey;
		backend.modelName = modelName;
		return backend;
	}
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:provider.endpointURL];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
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
