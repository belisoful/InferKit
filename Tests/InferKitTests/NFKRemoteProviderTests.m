//
//  NFKRemoteProviderTests.m
//  InferKitTests
//
//  The provider presets and Anthropic's wire format. Anthropic is stubbed through the same transport
//  seam NFKRemoteBackendTests uses, so the request shape is asserted without a network. A live test
//  against a local server is gated on it being present.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! An Anthropic backend whose transport is stubbed: it records the request and returns staged data. */
@interface NFKAnthropicStubBackend : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKAnthropicStubBackend

- (instancetype)init
{
	self = [super init];
	if (self) {
		_stagedStatusCode = 200;
	}
	return self;
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL
												   statusCode:self.stagedStatusCode
												  HTTPVersion:@"HTTP/1.1"
												 headerFields:nil];
	}
	return self.stagedData;
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}

@end

@interface NFKRemoteProviderTests : XCTestCase
@property (nonatomic, strong) NFKAnthropicStubBackend *anthropic;
@end

@implementation NFKRemoteProviderTests

- (void)setUp
{
	[super setUp];
	self.anthropic = [[NFKAnthropicStubBackend alloc] init];
	self.anthropic.modelName = @"a-model";
	self.anthropic.apiKey = @"a-key";
}

#pragma mark The presets

- (void)testEveryProviderCarriesAnEndpointAndAStyle
{
	NSArray<NFKRemoteProvider *> *providers = NFKRemoteProvider.allProviders;
	XCTAssertGreaterThan(providers.count, 0);
	NSMutableSet<NSString *> *seen = [NSMutableSet set];
	for (NFKRemoteProvider *provider in providers) {
		XCTAssertGreaterThan(provider.identifier.length, 0);
		XCTAssertGreaterThan(provider.displayName.length, 0);
		XCTAssertNotNil(provider.endpointURL, @"%@ has no endpoint", provider.identifier);
		XCTAssertNotNil(provider.endpointURL.scheme);
		XCTAssertFalse([seen containsObject:provider.identifier], @"%@ is duplicated", provider.identifier);
		[seen addObject:provider.identifier];
	}
}

// A model name is deliberately absent from every preset: identifiers change faster than a release,
// and a stale default fails with a message about the wrong thing. Readiness is NOT the assertion —
// an OpenAI-compatible backend is ready with an endpoint alone, which llama.cpp depends on: its
// server answers for whatever model it has loaded, with no model field in the request.
- (void)testNoProviderShipsADefaultModelName
{
	for (NFKRemoteProvider *provider in NFKRemoteProvider.allProviders) {
		id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider
																		apiKey:@"k"
																	 modelName:nil];
		NSString *model = [backend isKindOfClass:NFKAnthropicBackend.class]
			? [(NFKAnthropicBackend *)backend modelName]
			: [(NFKRemoteBackend *)backend modelName];
		XCTAssertNil(model, @"%@ must not ship a default model name", provider.identifier);
	}
}

// Anthropic requires a model in every request, so its backend reports that rather than failing at the
// first call; the OpenAI-compatible one does not, for the llama.cpp reason above.
- (void)testAnthropicIsNotReadyWithoutAModelWhileAnOpenAIEndpointIs
{
	id<NFKInferenceBackend> claude = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.anthropic
																	apiKey:@"k" modelName:nil];
	XCTAssertFalse(claude.isReady);

	id<NFKInferenceBackend> local = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.llamaCpp
																   apiKey:nil modelName:nil];
	XCTAssertTrue(local.isReady, @"a local server answers for the model it already has loaded");
}

- (void)testLocalProvidersNeedNoKey
{
	for (NSString *identifier in @[ @"ollama", @"lmstudio", @"llamacpp", @"vllm" ]) {
		NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
		XCTAssertNotNil(provider, @"%@ is missing", identifier);
		XCTAssertFalse(provider.requiresAPIKey, @"%@ is local", identifier);
		XCTAssertEqualObjects(provider.endpointURL.host, @"localhost");
	}
}

- (void)testHostedProvidersRequireAKey
{
	for (NSString *identifier in @[ @"openai", @"anthropic", @"xai", @"gemini", @"groq",
									@"mistral", @"deepseek", @"together", @"openrouter" ]) {
		NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
		XCTAssertNotNil(provider, @"%@ is missing", identifier);
		XCTAssertTrue(provider.requiresAPIKey, @"%@ is hosted", identifier);
		XCTAssertEqualObjects(provider.endpointURL.scheme, @"https", @"%@ must be https", identifier);
	}
}

- (void)testAnUnknownIdentifierHasNoProvider
{
	XCTAssertNil([NFKRemoteProvider providerWithIdentifier:@"midjourney"]);
	XCTAssertNil([NFKRemoteProvider providerWithIdentifier:@"nonesuch"]);
}

#pragma mark URLs derived from the base

// Every operation's URL is the base plus a path, so a preset carries one address rather than one
// per operation. The chat and model-list URLs are what the presets carried as literals before the
// base existed, asserted here so the derivation reproduces them exactly.
- (void)testEveryOperationURLIsDerivedFromTheBase
{
	NSDictionary<NSString *, NSString *> *bases = @{
		@"openai": @"https://api.openai.com/v1",
		@"anthropic": @"https://api.anthropic.com/v1",
		@"gemini": @"https://generativelanguage.googleapis.com/v1beta/openai",
		@"groq": @"https://api.groq.com/openai/v1",
		@"openrouter": @"https://openrouter.ai/api/v1",
		@"ollama": @"http://localhost:11434/v1",
	};
	for (NSString *identifier in bases) {
		NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
		XCTAssertEqualObjects(provider.baseURL.absoluteString, bases[identifier], @"%@", identifier);
		NSString *chat = provider.apiStyle == NFKRemoteAPIStyleAnthropicMessages ? @"/messages" : @"/chat/completions";
		XCTAssertEqualObjects(provider.endpointURL.absoluteString,
							  [bases[identifier] stringByAppendingString:chat], @"%@", identifier);
		XCTAssertEqualObjects(provider.modelsURL.absoluteString,
							  [bases[identifier] stringByAppendingString:@"/models"], @"%@", identifier);
	}
	for (NFKRemoteProvider *provider in NFKRemoteProvider.allProviders) {
		XCTAssertNotNil(provider.baseURL, @"%@ has no base", provider.identifier);
		XCTAssertTrue([provider.modelsURL.path hasSuffix:@"/models"], @"%@", provider.identifier);
	}
}

- (void)testURLForPathJoinsWithExactlyOneSlash
{
	NFKRemoteProvider *provider = NFKRemoteProvider.openAI;
	XCTAssertEqualObjects([provider URLForPath:@"audio/transcriptions"].absoluteString,
						  @"https://api.openai.com/v1/audio/transcriptions");
	XCTAssertEqualObjects([provider URLForPath:@"/embeddings"].absoluteString,
						  @"https://api.openai.com/v1/embeddings", @"a leading slash is not doubled");

	NFKRemoteProvider *trailing = [provider providerWithBaseURL:[NSURL URLWithString:@"https://example.test/v1/"]];
	XCTAssertEqualObjects([trailing URLForPath:@"models"].absoluteString,
						  @"https://example.test/v1/models", @"a trailing slash on the base is not doubled");
}

// A runner on another port or another machine is the same preset with its base changed: the
// identity, protocol, and key requirement travel with it, and every derived URL follows.
- (void)testAPresetRepointedAtAnotherBaseKeepsItsIdentity
{
	NFKRemoteProvider *remote = [NFKRemoteProvider.ollama providerWithBaseURL:
								 [NSURL URLWithString:@"http://192.168.1.20:11434/v1"]];
	XCTAssertEqualObjects(remote.identifier, @"ollama");
	XCTAssertEqualObjects(remote.displayName, NFKRemoteProvider.ollama.displayName);
	XCTAssertEqual(remote.apiStyle, NFKRemoteAPIStyleOpenAIChat);
	XCTAssertFalse(remote.requiresAPIKey);
	XCTAssertEqualObjects(remote.endpointURL.absoluteString, @"http://192.168.1.20:11434/v1/chat/completions");
	XCTAssertEqualObjects(remote.modelsURL.absoluteString, @"http://192.168.1.20:11434/v1/models");
	XCTAssertEqualObjects(NFKRemoteProvider.ollama.endpointURL.host, @"localhost", @"the preset itself is unchanged");

	NFKRemoteProvider *proxied = [NFKRemoteProvider.anthropic providerWithBaseURL:[NSURL URLWithString:@"https://proxy.test/anthropic/v1"]];
	XCTAssertEqual(proxied.apiStyle, NFKRemoteAPIStyleAnthropicMessages);
	XCTAssertEqualObjects(proxied.endpointURL.absoluteString, @"https://proxy.test/anthropic/v1/messages");
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:proxied apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects([(NFKAnthropicBackend *)backend endpointURL], proxied.endpointURL,
						  @"the factory points the backend at the re-based endpoint");
}

// Anthropic is the one provider that is not OpenAI-compatible, so the factory has to hand back its
// own backend rather than a repointed remote one.
- (void)testTheFactoryChoosesTheBackendTheProviderNeeds
{
	id<NFKInferenceBackend> claude = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.anthropic
																	apiKey:@"k" modelName:@"m"];
	XCTAssertTrue([claude isKindOfClass:NFKAnthropicBackend.class]);
	XCTAssertEqualObjects(claude.backendIdentifier, @"anthropic-messages");

	id<NFKInferenceBackend> openAI = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.openAI
																	apiKey:@"k" modelName:@"m"];
	XCTAssertTrue([openAI isKindOfClass:NFKRemoteBackend.class]);
}

#pragma mark Anthropic's wire format

- (void)testTheKeyTravelsInTheAnthropicHeaderNotAsABearerToken
{
	self.anthropic.stagedData = [@"{\"content\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	[self.anthropic runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]
									 error:NULL];

	NSDictionary *headers = self.anthropic.lastRequest.allHTTPHeaderFields;
	XCTAssertEqualObjects(headers[@"x-api-key"], @"a-key");
	XCTAssertNil(headers[@"Authorization"], @"Anthropic does not take a Bearer token");
	XCTAssertGreaterThan([headers[@"anthropic-version"] length], 0, @"the version header is required");
}

- (void)testMaxTokensIsAlwaysSentBecauseTheAPIRequiresIt
{
	self.anthropic.stagedData = [@"{\"content\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	[self.anthropic runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]
									 error:NULL];
	XCTAssertNotNil([self.anthropic decodedRequestBody][@"max_tokens"]);

	NFKInferenceRequest *bounded = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }
															  parameters:@{ NFKParameterMaxTokens: @64 }
														  outputModality:NFKModalityText];
	[self.anthropic runInferenceForRequest:bounded error:NULL];
	XCTAssertEqualObjects([self.anthropic decodedRequestBody][@"max_tokens"], @64);
}

// A system turn is a top-level field here, not a message with a role, so a caller can write the same
// request for Anthropic as for an OpenAI-compatible provider.
- (void)testASystemMessageIsLiftedOutOfTheConversation
{
	self.anthropic.stagedData = [@"{\"content\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[
		@{ @"role": @"system", @"content": @"be brief" },
		@{ @"role": @"user", @"content": @"hello" },
	] }];
	[self.anthropic runInferenceForRequest:request error:NULL];

	NSDictionary *body = [self.anthropic decodedRequestBody];
	XCTAssertEqualObjects(body[@"system"], @"be brief");
	XCTAssertEqual([body[@"messages"] count], 1, @"the system turn leaves the conversation");
	XCTAssertEqualObjects(body[@"messages"][0][@"role"], @"user");
}

- (void)testTextBlocksAreJoinedIntoTheTextOutput
{
	self.anthropic.stagedData = [@"{\"content\":[{\"type\":\"text\",\"text\":\"Hello\"},"
								  "{\"type\":\"thinking\",\"thinking\":\"ignored\"},"
								  "{\"type\":\"text\",\"text\":\", world\"}]}"
								 dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceResult *result = [self.anthropic runInferenceForRequest:
								  [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]
																  error:NULL];
	XCTAssertEqualObjects([result outputForKey:NFKRemoteBackendTextKey], @"Hello, world",
						  @"only text blocks are joined");
}

- (void)testAFailingStatusCodeBecomesAnError
{
	self.anthropic.stagedStatusCode = 401;
	self.anthropic.stagedData = [@"{\"error\":\"bad key\"}" dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	XCTAssertNil([self.anthropic runInferenceForRequest:
				  [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertNotNil(error);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testAnEmptyRequestIsRejected
{
	NSError *error = nil;
	XCTAssertNil([self.anthropic runInferenceForRequest:
				  [NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

- (void)testTheBackendIsNotReadyWithoutAModel
{
	NFKAnthropicBackend *backend = [NFKAnthropicBackend backendWithEndpointURL:nil];
	XCTAssertFalse(backend.isReady, @"the API requires a model name");
	backend.modelName = @"m";
	XCTAssertTrue(backend.isReady, @"the endpoint defaults to Anthropic's own");
}

#pragma mark A live local server

// Runs only when a local OpenAI-compatible server is actually listening. This is the one test that
// exercises the whole path — request shaping, transport, and response parsing — against a real
// implementation rather than a stub. Set INFERKIT_LIVE_LOCAL_MODEL to the model to ask for.
- (void)testALocalServerAnswersWhenOneIsRunning
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_LOCAL_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_LOCAL_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
	XCTAssertNotNil(provider, @"unknown provider %@", identifier);

	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider
																	 apiKey:nil
																  modelName:model];
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Reply with exactly: OK" }
									parameters:@{ NFKParameterMaxTokens: @16 }
								outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error.localizedDescription);
	XCTAssertGreaterThan([[result outputForKey:NFKRemoteBackendTextKey] length], 0);
}

@end
