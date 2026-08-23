//
//  NFKRemoteProvider.m
//  InferKit
//

#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import "NFK_ARC.h"

@interface NFKRemoteProvider ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite) NSString *displayName;
@property (nonatomic, copy, readwrite) NSURL *endpointURL;
@property (nonatomic, assign, readwrite) NFKRemoteAPIStyle apiStyle;
@property (nonatomic, assign, readwrite) BOOL requiresAPIKey;
@property (nonatomic, copy, readwrite, nullable) NSURL *modelsURL;
@end

@implementation NFKRemoteProvider

+ (instancetype)providerWithIdentifier:(NSString *)identifier
						   displayName:(NSString *)displayName
						   endpoint:(NSString *)endpoint
							  models:(nullable NSString *)models
							   style:(NFKRemoteAPIStyle)style
						  requiresKey:(BOOL)requiresKey
{
	NFKRemoteProvider *provider = [[self alloc] initPrivate];
	provider.identifier = identifier;
	provider.displayName = displayName;
	provider.endpointURL = [NSURL URLWithString:endpoint];
	provider.modelsURL = models != nil ? [NSURL URLWithString:models] : nil;
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

#pragma mark Hosted

+ (NFKRemoteProvider *)openAI
{
	return [self providerWithIdentifier:@"openai" displayName:@"OpenAI"
							   endpoint:@"https://api.openai.com/v1/chat/completions"
								 models:@"https://api.openai.com/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)anthropic
{
	return [self providerWithIdentifier:@"anthropic" displayName:@"Anthropic"
							   endpoint:@"https://api.anthropic.com/v1/messages"
								 models:@"https://api.anthropic.com/v1/models"
								  style:NFKRemoteAPIStyleAnthropicMessages requiresKey:YES];
}

+ (NFKRemoteProvider *)xAI
{
	return [self providerWithIdentifier:@"xai" displayName:@"xAI Grok"
							   endpoint:@"https://api.x.ai/v1/chat/completions"
								 models:@"https://api.x.ai/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)googleGemini
{
	// Gemini's OpenAI-compatible layer, which is what lets one backend serve it.
	return [self providerWithIdentifier:@"gemini" displayName:@"Google Gemini"
							   endpoint:@"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
								 models:@"https://generativelanguage.googleapis.com/v1beta/openai/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)groq
{
	return [self providerWithIdentifier:@"groq" displayName:@"Groq"
							   endpoint:@"https://api.groq.com/openai/v1/chat/completions"
								 models:@"https://api.groq.com/openai/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)mistral
{
	return [self providerWithIdentifier:@"mistral" displayName:@"Mistral"
							   endpoint:@"https://api.mistral.ai/v1/chat/completions"
								 models:@"https://api.mistral.ai/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)deepSeek
{
	return [self providerWithIdentifier:@"deepseek" displayName:@"DeepSeek"
							   endpoint:@"https://api.deepseek.com/v1/chat/completions"
								 models:@"https://api.deepseek.com/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)together
{
	return [self providerWithIdentifier:@"together" displayName:@"Together AI"
							   endpoint:@"https://api.together.xyz/v1/chat/completions"
								 models:@"https://api.together.xyz/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

+ (NFKRemoteProvider *)openRouter
{
	return [self providerWithIdentifier:@"openrouter" displayName:@"OpenRouter"
							   endpoint:@"https://openrouter.ai/api/v1/chat/completions"
								 models:@"https://openrouter.ai/api/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:YES];
}

#pragma mark Local

+ (NFKRemoteProvider *)ollama
{
	return [self providerWithIdentifier:@"ollama" displayName:@"Ollama"
							   endpoint:@"http://localhost:11434/v1/chat/completions"
								 models:@"http://localhost:11434/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)lmStudio
{
	return [self providerWithIdentifier:@"lmstudio" displayName:@"LM Studio"
							   endpoint:@"http://localhost:1234/v1/chat/completions"
								 models:@"http://localhost:1234/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)llamaCpp
{
	return [self providerWithIdentifier:@"llamacpp" displayName:@"llama.cpp server"
							   endpoint:@"http://localhost:8080/v1/chat/completions"
								 models:@"http://localhost:8080/v1/models"
								  style:NFKRemoteAPIStyleOpenAIChat requiresKey:NO];
}

+ (NFKRemoteProvider *)vLLM
{
	return [self providerWithIdentifier:@"vllm" displayName:@"vLLM"
							   endpoint:@"http://localhost:8000/v1/chat/completions"
								 models:@"http://localhost:8000/v1/models"
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

@end
