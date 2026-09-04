//
//  NFKLocalModelRunnerTests.m
//  InferKitTests
//
//  The runner adapters through stubbed transports: the native URL each derives, the shapes Ollama
//  0.33 and LM Studio's /api/v0 answer with, and the streamed pull's three endings. The staged
//  bodies are what the real servers returned, trimmed. A live pass against a running Ollama is
//  gated on the same variable the other live tests use and only reads, apart from pulling and
//  deleting a name that does not exist, which exercises both failure paths without installing
//  or removing anything.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKLocalModelRunner.h>
#import <InferKit/NFKOllamaRunner.h>
#import <InferKit/NFKLMStudioRunner.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteModel.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

static NSData *NFKBody(NSString *json)
{
	return [json dataUsingEncoding:NSUTF8StringEncoding];
}

#pragma mark - Stubs

@interface NFKStubOllamaRunner : NFKOllamaRunner
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, copy) NSDictionary<NSString *, NSData *> *bodiesByPath;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, assign) BOOL unreachable;
@property (nonatomic, copy) NSArray<NSString *> *stagedStreamLines;
@property (nonatomic, strong, nullable) NSError *stagedStreamError;
@property (nonatomic, assign) BOOL holdStreamOpen;
@property (nonatomic, assign) BOOL streamWasCancelled;
@end

@implementation NFKStubOllamaRunner

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (self.unreachable) {
		if (outError != NULL) {
			*outError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable
										userInfo:@{ NSLocalizedDescriptionKey: @"localhost did not answer" }];
		}
		return nil;
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.bodiesByPath[request.URL.path] ?: NFKBody(@"{}");
}

- (void)streamRequest:(NSURLRequest *)request
		  lineHandler:(void (^)(NSDictionary<NSString *, id> *))lineHandler
	completionHandler:(void (^)(NSError * _Nullable))completionHandler
		 cancellation:(void (^)(void (^)(void)))cancellation
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	cancellation(^{ self.streamWasCancelled = YES; });
	for (NSString *line in self.stagedStreamLines) {
		lineHandler([NSJSONSerialization JSONObjectWithData:NFKBody(line) options:0 error:NULL]);
	}
	if (!self.holdStreamOpen) {
		completionHandler(self.stagedStreamError);
	}
}

@end

@interface NFKStubLMStudioRunner : NFKLMStudioRunner
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSDictionary<NSString *, NSData *> *bodiesByPath;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubLMStudioRunner

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.bodiesByPath[request.URL.path] ?: NFKBody(@"{}");
}

@end

#pragma mark - Tests

@interface NFKLocalModelRunnerTests : XCTestCase
@property (nonatomic, strong) NFKStubOllamaRunner *ollama;
@end

@implementation NFKLocalModelRunnerTests

- (void)setUp
{
	[super setUp];
	self.ollama = [NFKStubOllamaRunner runnerWithProvider:NFKRemoteProvider.ollama];
	self.ollama.bodiesByPath = @{
		@"/api/version": NFKBody(@"{\"version\":\"0.33.2\"}"),
		@"/api/tags": NFKBody(@"{\"models\":["
			"{\"name\":\"gpt-oss:20b\",\"model\":\"gpt-oss:20b\",\"modified_at\":\"2026-04-18T02:54:10.361731935-07:00\","
			 "\"size\":13793441244,\"digest\":\"17052f91\",\"details\":{\"format\":\"gguf\",\"family\":\"gptoss\","
			 "\"parameter_size\":\"20.9B\",\"quantization_level\":\"MXFP4\",\"context_length\":131072,\"embedding_length\":2880},"
			 "\"capabilities\":[\"completion\",\"tools\",\"thinking\"]},"
			"{\"name\":\"qwen3.5:27b\",\"model\":\"qwen3.5:27b\",\"size\":17420432728,"
			 "\"details\":{\"quantization_level\":\"Q4_K_M\",\"context_length\":262144},"
			 "\"capabilities\":[\"vision\",\"completion\",\"tools\",\"thinking\"]}]}"),
		@"/api/ps": NFKBody(@"{\"models\":[{\"name\":\"qwen3.5:27b\",\"model\":\"qwen3.5:27b\",\"size\":19000000000,"
			"\"size_vram\":19000000000,\"expires_at\":\"2026-09-03T03:00:00Z\",\"details\":{\"quantization_level\":\"Q4_K_M\"}}]}"),
		@"/api/show": NFKBody(@"{\"license\":\"...\",\"modelfile\":\"...\",\"details\":{\"family\":\"gptoss\","
			"\"quantization_level\":\"MXFP4\"},\"model_info\":{\"general.architecture\":\"gptoss\","
			"\"general.parameter_count\":20914757184,\"gptoss.context_length\":131072,"
			"\"gptoss.rope.scaling.original_context_length\":4096},\"capabilities\":[\"completion\",\"tools\",\"thinking\"]}"),
	};
}

#pragma mark The provider hands back its runner

- (void)testThePresetsWithANativeAPIHaveARunnerAndTheOthersDoNot
{
	XCTAssertTrue([NFKRemoteProvider.ollama.localRunner isKindOfClass:NFKOllamaRunner.class]);
	XCTAssertTrue([NFKRemoteProvider.lmStudio.localRunner isKindOfClass:NFKLMStudioRunner.class]);
	XCTAssertNil(NFKRemoteProvider.llamaCpp.localRunner, @"nothing beyond the OpenAI surface to adapt");
	XCTAssertNil(NFKRemoteProvider.vLLM.localRunner);
	XCTAssertNil(NFKRemoteProvider.openAI.localRunner, @"hosted");

	id<NFKLocalModelRunner> lan = [NFKRemoteProvider.ollama providerWithBaseURL:[NSURL URLWithString:@"http://192.168.1.20:11434/v1"]].localRunner;
	XCTAssertNotNil(lan, @"a re-based preset keeps its runner");
	XCTAssertEqualObjects(lan.nativeBaseURL.absoluteString, @"http://192.168.1.20:11434");
}

// The OpenAI surface is under /v1 and the native one at the host root, so the runner's base is
// the provider's with that segment removed.
- (void)testTheNativeBaseIsTheProvidersWithoutV1
{
	XCTAssertEqualObjects(self.ollama.nativeBaseURL.absoluteString, @"http://localhost:11434");
	XCTAssertEqualObjects(NFKRemoteProvider.lmStudio.localRunner.nativeBaseURL.absoluteString, @"http://localhost:1234");

	[self.ollama versionWithError:NULL];
	XCTAssertEqualObjects(self.ollama.requests.lastObject.URL.absoluteString, @"http://localhost:11434/api/version");
	[self.ollama isRunning];
	XCTAssertEqualObjects(self.ollama.requests.lastObject.URL.absoluteString, @"http://localhost:11434/",
						  @"the health probe is the root, which answers \"Ollama is running\"");
}

#pragma mark Ollama

- (void)testTheInstalledListCarriesSizeQuantizationContextAndCapabilities
{
	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [self.ollama installedModelsWithError:&error];
	XCTAssertNotNil(models, @"%@", error);
	XCTAssertEqual(models.count, 2);

	NFKRemoteModel *first = models[0];
	XCTAssertEqualObjects(first.identifier, @"gpt-oss:20b", @"the list carries no id, so name is the identifier");
	XCTAssertEqualObjects(first.displayName, @"gpt-oss:20b");
	XCTAssertEqualObjects(first.sizeBytes, @13793441244);
	XCTAssertEqualObjects(first.quantization, @"MXFP4");
	XCTAssertEqualObjects(first.contextLength, @131072, @"read from details, without a show per model");
	XCTAssertEqualObjects(first.capabilities, (@[ @"completion", @"tools", @"thinking" ]));
	XCTAssertTrue([models[1].capabilities containsObject:@"vision"]);
	XCTAssertEqualObjects(first.raw[@"digest"], @"17052f91", @"the entry is kept as received");
}

- (void)testTheLoadedListIsPS
{
	NSArray<NFKRemoteModel *> *loaded = [self.ollama loadedModelsWithError:NULL];
	XCTAssertEqual(loaded.count, 1);
	XCTAssertEqualObjects(loaded.firstObject.identifier, @"qwen3.5:27b");
	XCTAssertEqualObjects(loaded.firstObject.raw[@"size_vram"], @19000000000);
	XCTAssertTrue([self.ollama.requests.lastObject.URL.path isEqualToString:@"/api/ps"]);
}

// /api/show carries no identifier and keys the context length by architecture, so both are lifted
// to where the model type reads them.
- (void)testShowLiftsTheIdentifierAndTheArchitectureKeyedContextLength
{
	NSError *error = nil;
	NFKRemoteModel *model = [self.ollama detailsForModel:@"gpt-oss:20b" error:&error];
	XCTAssertNotNil(model, @"%@", error);
	XCTAssertEqualObjects(model.identifier, @"gpt-oss:20b");
	XCTAssertEqualObjects(model.contextLength, @131072);
	XCTAssertEqualObjects(model.quantization, @"MXFP4");
	XCTAssertEqualObjects(model.capabilities, (@[ @"completion", @"tools", @"thinking" ]));
	XCTAssertEqualObjects(model.raw[@"model_info"][@"general.parameter_count"], @20914757184);

	NSURLRequest *request = self.ollama.requests.lastObject;
	XCTAssertEqualObjects(request.HTTPMethod, @"POST");
	NSDictionary *body = [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
	XCTAssertEqualObjects(body[@"model"], @"gpt-oss:20b");
}

- (void)testAMissingModelIsAnErrorNamingIt
{
	self.ollama.stagedStatusCode = 404;
	self.ollama.bodiesByPath = @{ @"/api/show": NFKBody(@"{\"error\":\"model 'nonesuch:latest' not found\"}"),
								  @"/api/delete": NFKBody(@"{\"error\":\"model 'nonesuch:latest' not found\"}") };
	NSError *error = nil;
	XCTAssertNil([self.ollama detailsForModel:@"nonesuch:latest" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"nonesuch:latest"]);

	XCTAssertFalse([self.ollama deleteModel:@"nonesuch:latest" error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"not found"]);
	XCTAssertEqualObjects(self.ollama.requests.lastObject.HTTPMethod, @"DELETE");
}

- (void)testARunnerThatIsNotRunningIsUnreachable
{
	self.ollama.unreachable = YES;
	XCTAssertFalse(self.ollama.isRunning);
	NSError *error = nil;
	XCTAssertNil([self.ollama installedModelsWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
}

#pragma mark The streamed pull

- (void)testAPullReportsProgressThenFinishesWithTheIdentifier
{
	self.ollama.stagedStreamLines = @[
		@"{\"status\":\"pulling manifest\"}",
		@"{\"status\":\"pulling 8934d96d3f08\",\"digest\":\"sha256:8934\",\"total\":1000,\"completed\":250}",
		@"{\"status\":\"pulling 8934d96d3f08\",\"digest\":\"sha256:8934\",\"total\":1000,\"completed\":1000}",
		@"{\"status\":\"verifying sha256 digest\"}",
		@"{\"status\":\"success\"}",
	];
	NSMutableArray<NSNumber *> *progress = [NSMutableArray array];
	NSMutableArray<NSString *> *statuses = [NSMutableArray array];

	// The stub delivers synchronously, so the handlers are set through a runner that holds the
	// stream open until they are attached, then released.
	self.ollama.holdStreamOpen = YES;
	NFKInferenceJob *job = [self.ollama pullModel:@"llama3.2"];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"success arrived inside the staged lines");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputText], @"llama3.2");
	(void)progress; (void)statuses;

	NSURLRequest *request = self.ollama.requests.lastObject;
	XCTAssertEqualObjects(request.URL.absoluteString, @"http://localhost:11434/api/pull");
	NSDictionary *body = [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
	XCTAssertEqualObjects(body[@"model"], @"llama3.2");
	XCTAssertEqualObjects(body[@"stream"], @YES);
}

- (void)testAPullsProgressIsTheLayerFractionAndItsStatusTheLatestLine
{
	self.ollama.stagedStreamLines = @[
		@"{\"status\":\"pulling manifest\"}",
		@"{\"status\":\"pulling 8934\",\"total\":1000,\"completed\":250}",
	];
	self.ollama.holdStreamOpen = YES;
	NFKInferenceJob *job = [self.ollama pullModel:@"llama3.2"];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning);
	XCTAssertEqualWithAccuracy(job.progress, 0.25, 1e-9);
	XCTAssertEqualObjects([job.partialResult outputForKey:NFKOutputText], @"pulling 8934");
}

// Measured against Ollama 0.33: a pull of a name it cannot find answers HTTP 200 and puts the
// failure in an error line, so the status code is not what decides the outcome.
- (void)testAPullThatFailsInsideTheStreamFailsTheJobWithTheRunnersMessage
{
	self.ollama.stagedStreamLines = @[
		@"{\"status\":\"pulling manifest\"}",
		@"{\"error\":\"pull model manifest: file does not exist\"}",
	];
	NFKInferenceJob *job = [self.ollama pullModel:@"nonesuch:latest"];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqualObjects(job.error.localizedDescription, @"pull model manifest: file does not exist");
}

- (void)testAStreamThatEndsWithoutSuccessFailsTheJob
{
	self.ollama.stagedStreamLines = @[ @"{\"status\":\"pulling manifest\"}" ];
	NFKInferenceJob *ended = [self.ollama pullModel:@"llama3.2"];
	XCTAssertEqual(ended.status, NFKInferenceJobStatusFailed);
	XCTAssertTrue([ended.error.localizedDescription containsString:@"without reporting success"]);

	self.ollama.stagedStreamError = [NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorNetworkConnectionLost userInfo:nil];
	NFKInferenceJob *dropped = [self.ollama pullModel:@"llama3.2"];
	XCTAssertEqual(dropped.status, NFKInferenceJobStatusFailed);
	XCTAssertEqualObjects(dropped.error.domain, NSURLErrorDomain, @"the transport's own error is kept");
}

- (void)testCancellingThePullStopsTheStream
{
	self.ollama.stagedStreamLines = @[ @"{\"status\":\"pulling manifest\"}" ];
	self.ollama.holdStreamOpen = YES;
	NFKInferenceJob *job = [self.ollama pullModel:@"llama3.2"];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning);
	[job cancel];
	XCTAssertTrue(self.ollama.streamWasCancelled);
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
}

#pragma mark LM Studio

- (void)testLMStudioListsInstalledModelsAndFiltersTheLoadedOnes
{
	NFKStubLMStudioRunner *studio = [NFKStubLMStudioRunner runnerWithProvider:NFKRemoteProvider.lmStudio];
	studio.bodiesByPath = @{
		@"/api/v0/models": NFKBody(@"{\"object\":\"list\",\"data\":["
			"{\"id\":\"qwen2-vl-2b-instruct\",\"object\":\"model\",\"type\":\"vlm\",\"publisher\":\"mlx-community\","
			 "\"arch\":\"qwen2_vl\",\"compatibility_type\":\"mlx\",\"quantization\":\"4bit\",\"state\":\"not-loaded\","
			 "\"max_context_length\":32768},"
			"{\"id\":\"text-embedding-nomic-embed-text-v1.5\",\"object\":\"model\",\"type\":\"embeddings\","
			 "\"quantization\":\"Q4_K_M\",\"state\":\"loaded\",\"max_context_length\":2048}]}"),
		@"/api/v0/models/qwen2-vl-2b-instruct": NFKBody(@"{\"id\":\"qwen2-vl-2b-instruct\",\"type\":\"vlm\","
			"\"quantization\":\"4bit\",\"state\":\"not-loaded\",\"max_context_length\":32768}"),
	};

	NSError *error = nil;
	NSArray<NFKRemoteModel *> *installed = [studio installedModelsWithError:&error];
	XCTAssertNotNil(installed, @"%@", error);
	XCTAssertEqual(installed.count, 2);
	XCTAssertEqualObjects(installed[0].quantization, @"4bit");
	XCTAssertEqualObjects(installed[0].contextLength, @32768);
	XCTAssertEqualObjects(installed[0].raw[@"type"], @"vlm");
	XCTAssertEqualObjects(studio.lastRequest.URL.absoluteString, @"http://localhost:1234/api/v0/models");

	NSArray<NFKRemoteModel *> *loaded = [studio loadedModelsWithError:NULL];
	XCTAssertEqual(loaded.count, 1);
	XCTAssertEqualObjects(loaded.firstObject.identifier, @"text-embedding-nomic-embed-text-v1.5");

	NFKRemoteModel *detail = [studio detailsForModel:@"qwen2-vl-2b-instruct" error:&error];
	XCTAssertEqualObjects(detail.identifier, @"qwen2-vl-2b-instruct");
	XCTAssertEqualObjects(studio.lastRequest.URL.absoluteString, @"http://localhost:1234/api/v0/models/qwen2-vl-2b-instruct");

	XCTAssertFalse([studio respondsToSelector:@selector(pullModel:)], @"LM Studio's REST surface offers no download");
	XCTAssertFalse([studio respondsToSelector:@selector(deleteModel:error:)]);
	XCTAssertTrue([self.ollama respondsToSelector:@selector(pullModel:)]);
}

#pragma mark A live Ollama

// Reads only, apart from a pull and a delete of a name that does not exist — which is what
// exercises both failure paths against the real server without installing or removing anything.
- (void)testALiveOllamaAnswersItsNativeAPI
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_LOCAL_MODEL"];
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	if (model.length == 0 || ![identifier isEqualToString:@"ollama"]) {
		XCTSkip("set INFERKIT_LIVE_LOCAL_MODEL (and run Ollama) to exercise the live path");
	}
	NFKOllamaRunner *runner = (NFKOllamaRunner *)NFKRemoteProvider.ollama.localRunner;
	XCTAssertTrue(runner.isRunning);

	NSError *error = nil;
	NSString *version = [runner versionWithError:&error];
	XCTAssertGreaterThan(version.length, 0, @"%@", error);

	NSArray<NFKRemoteModel *> *installed = [runner installedModelsWithError:&error];
	XCTAssertNotNil(installed, @"%@", error);
	NSArray<NSString *> *names = [installed valueForKey:@"identifier"];
	XCTAssertTrue([names containsObject:model], @"%@ is not among %@", model, names);
	NFKRemoteModel *listed = installed[[names indexOfObject:model]];
	XCTAssertGreaterThan(listed.sizeBytes.longLongValue, 0);
	XCTAssertGreaterThan(listed.quantization.length, 0);

	XCTAssertNotNil([runner loadedModelsWithError:&error], @"%@", error);

	NFKRemoteModel *detail = [runner detailsForModel:model error:&error];
	XCTAssertNotNil(detail, @"%@", error);
	XCTAssertGreaterThan(detail.contextLength.integerValue, 0, @"the architecture-keyed context length was lifted");
	XCTAssertGreaterThan(detail.capabilities.count, 0);

	NSString *nonesuch = @"inferkit-nonesuch-probe:latest";
	NFKInferenceJob *pull = [runner pullModel:nonesuch];
	XCTestExpectation *ended = [self expectationWithDescription:@"pull ended"];
	pull.completionHandler = ^(NFKInferenceJob *job) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:60];
	XCTAssertEqual(pull.status, NFKInferenceJobStatusFailed, @"a name the registry does not have");
	XCTAssertTrue([pull.error.localizedDescription containsString:@"manifest"], @"%@", pull.error);

	XCTAssertFalse([runner deleteModel:nonesuch error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"not found"], @"%@", error);
}

@end
