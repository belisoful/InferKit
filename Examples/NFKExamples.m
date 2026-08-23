//
//  NFKExamples.m
//  InferKitExamples
//
//  These examples are compiled and run by CI so the code in Docs/examples.md cannot silently drift.
//  Each method mirrors a section of Docs/examples.md — change an example here and update the matching
//  snippet there, and vice versa. The weight-free paths run and assert; the model/network backends
//  are exercised at the API-shape and contract level (constructing them and reading identity), so
//  their example code keeps compiling without needing weights or a server.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <CoreML/CoreML.h>

// Docs/examples.md: Async generation service (submit → poll → fetch). A subclass maps a specific
// service's JSON to the base's template methods; the base owns the submit/poll/fetch loop and the job
// handle. Here the mapping is exercised without a network call.
@interface ExampleImageGenerationBackend : NFKAsyncGenerationBackend
@end

@implementation ExampleImageGenerationBackend
- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request
{
	return @{ @"prompt": request.prompt ?: @"", @"model": self.modelName ?: @"" };
}
- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response { return response[@"id"]; }
- (nullable NSURL *)statusURLForJobIdentifier:(NSString *)jobIdentifier
{
	return [self.submitURL URLByAppendingPathComponent:jobIdentifier];
}
- (BOOL)isSucceededStatusResponse:(NSDictionary *)response { return [response[@"status"] isEqual:@"succeeded"]; }
- (BOOL)isFailedStatusResponse:(NSDictionary *)response { return [response[@"status"] isEqual:@"failed"]; }
- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: response[@"output"] ?: @"" }];
}
@end

@interface NFKExamples : XCTestCase
@end

@implementation NFKExamples

#pragma mark The shared contract (Docs/examples.md: The shared contract)

- (void)testExampleTheSharedContractWithPassthrough
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	backend.outputMap = @{ NFKOutputText: NFKInputPrompt };  // each output key maps to an input key

	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Explain diffraction in one sentence." }
									parameters:@{ NFKParameterMaxTokens: @64 }
								outputModality:NFKModalityText];

	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Explain diffraction in one sentence.");   // convenience for NFKOutputText
}

#pragma mark Jobs, submit, completion (Docs/examples.md: Subsystems — Jobs)

- (void)testExampleSubmittingAJob
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	backend.outputMap = @{ NFKOutputImage: NFKInputImage };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: @"plate" }];

	XCTestExpectation *finished = [self expectationWithDescription:@"job finished"];
	NFKInferenceJob *job = NFKInferenceSubmit(backend, request, NULL);
	job.completionHandler = ^(NFKInferenceJob *completed) {
		XCTAssertEqual(completed.status, NFKInferenceJobStatusSucceeded);
		XCTAssertEqualObjects([completed.result outputForKey:NFKOutputImage], @"plate");
		[finished fulfill];
	};
	[self waitForExpectations:@[finished] timeout:5];
}

#pragma mark Tensor conversion (Docs/examples.md: Subsystems — Tensor conversion)

- (void)testExampleTensorConversionRoundTrip
{
	float interleavedRGBA[16] = {
		0.1f, 0.2f, 0.3f, 1.0f,  0.4f, 0.5f, 0.6f, 1.0f,
		0.7f, 0.8f, 0.9f, 1.0f,  0.15f, 0.25f, 0.35f, 1.0f,
	};
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);        // defaults: CHW, RGBA order, mean 0 / scale 1

	float tensor[12];
	NFKInterleavedToTensor(interleavedRGBA, tensor, spec);
	XCTAssertEqual(NFKTensorElementCount(spec), (NSUInteger)12);

	float restored[16];
	NFKTensorToInterleaved(tensor, restored, spec);
	XCTAssertEqualWithAccuracy(restored[0], 0.1f, 1e-5);
}

#pragma mark MLMultiArray bridge (Docs/examples.md: Subsystems — Tensor conversion)

- (void)testExampleMLMultiArrayBridge
{
	float interleavedRGBA[16] = {
		0.1f, 0.2f, 0.3f, 1.0f,  0.4f, 0.5f, 0.6f, 1.0f,
		0.7f, 0.8f, 0.9f, 1.0f,  0.15f, 0.25f, 0.35f, 1.0f,
	};
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);

	NSError *error = nil;
	MLMultiArray *array = NFKMultiArrayFromInterleaved(interleavedRGBA, spec, &error);
	XCTAssertNotNil(array, @"%@", error);
	XCTAssertEqualObjects(array.shape, (@[@1, @3, @2, @2]));    // [1, C, H, W]

	float restored[16];
	XCTAssertTrue(NFKInterleavedFromMultiArray(array, restored, spec));
	XCTAssertEqualWithAccuracy(restored[0], 0.1f, 1e-5);
}

#pragma mark Tokenizer (Docs/examples.md: Subsystems — Tokenizers)

- (void)testExampleTokenizerEncodeDecode
{
	NSURL *directory = [[NSFileManager defaultManager].temporaryDirectory
		URLByAppendingPathComponent:[NSUUID UUID].UUIDString isDirectory:YES];
	[[NSFileManager defaultManager] createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];

	NSData *vocab = [NSJSONSerialization dataWithJSONObject:@{ @"h": @0, @"e": @1, @"l": @2, @"o": @3, @"he": @4, @"ll": @5, @"hell": @6, @"hello": @7 }
												   options:0 error:NULL];
	[vocab writeToURL:[directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];
	[@"#version: 0.2\nh e\nl l\nhe ll\nhell o\n" writeToURL:[directory URLByAppendingPathComponent:@"merges.txt"]
												 atomically:YES encoding:NSUTF8StringEncoding error:NULL];

	NSDictionary *manifest = @{ @"tokenizer": @{ @"type": @"bpe-bytelevel", @"vocab": @"vocab.json", @"merges": @"merges.txt" } };
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:directory error:&error];
	XCTAssertNotNil(tokenizer, @"%@", error);
	XCTAssertEqualObjects([tokenizer encode:@"hello"], (@[@7]));
	XCTAssertEqualObjects([tokenizer decode:@[@7]], @"hello");

	[[NSFileManager defaultManager] removeItemAtURL:directory error:NULL];
}

// Docs/examples.md: Subsystems — Tokenizers. The CLIP variant, which the Stable Diffusion text
// encoders take: text lowercases, a word's last piece carries "</w>", and the markers are the model
// input's rather than the tokenizer's output.
- (void)testExampleCLIPTokenizerEncodeDecode
{
	NSURL *directory = [[NSFileManager defaultManager].temporaryDirectory
		URLByAppendingPathComponent:[NSUUID UUID].UUIDString isDirectory:YES];
	[[NSFileManager defaultManager] createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];

	NSData *vocab = [NSJSONSerialization dataWithJSONObject:@{ @"h": @0, @"e": @1, @"l": @2, @"o</w>": @3,
															   @"he": @4, @"ll": @5, @"hell": @6, @"hello</w>": @7 }
												   options:0 error:NULL];
	[vocab writeToURL:[directory URLByAppendingPathComponent:@"vocab.json"] atomically:YES];
	[@"#version: 0.2\nh e\nl l\nhe ll\nhell o</w>\n" writeToURL:[directory URLByAppendingPathComponent:@"merges.txt"]
														 atomically:YES encoding:NSUTF8StringEncoding error:NULL];

	NSDictionary *manifest = @{ @"tokenizer": @{ @"type": @"clip", @"vocab": @"vocab.json", @"merges": @"merges.txt" } };
	NSError *error = nil;
	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:directory error:&error];
	XCTAssertNotNil(tokenizer, @"%@", error);
	XCTAssertEqualObjects([tokenizer encode:@"HELLO"], (@[@7]));      // lowercased first
	XCTAssertEqualObjects([tokenizer decode:@[@7]], @"hello");

	[[NSFileManager defaultManager] removeItemAtURL:directory error:NULL];
}

#pragma mark Hugging Face hub (Docs/examples.md: Subsystems — Hugging Face hub)

- (void)testExampleHuggingFaceHubURLResolution
{
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:nil];
	NSURL *remote = [hub remoteURLForRepo:@"Qwen/Qwen2.5-0.5B-Instruct" revision:nil path:@"tokenizer.json"];
	XCTAssertTrue([remote.absoluteString containsString:@"Qwen/Qwen2.5-0.5B-Instruct"]);   // no network: builds the URL
}

#pragma mark Backend contracts (Docs/examples.md: Text → text, Image → image)

- (void)testExampleLocalLanguageBackendContract
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [NSURL fileURLWithPath:@"/models/qwen"];
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		backend.computeUnits = MLComputeUnitsAll;
		XCTAssertEqualObjects(backend.backendIdentifier, @"coreml-llm");
		XCTAssertFalse(backend.isReady);                    // not ready until prepared
	}
}

- (void)testExampleRemoteBackendContract
{
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:[NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"]];
	backend.modelName = @"llama3.2";
	XCTAssertEqualObjects(backend.backendIdentifier, @"remote");
	XCTAssertTrue(backend.isReady);                         // an endpoint is set
}

- (void)testExampleTranscriptionBackendContract
{
	// Audio → text: point at an OpenAI-compatible transcriptions endpoint, set the model, and pass
	// audio under NFKInputAudio (an NFKAudioAsset or NSData). The call itself needs the network.
	NFKRemoteTranscriptionBackend *backend =
		[NFKRemoteTranscriptionBackend backendWithEndpointURL:[NSURL URLWithString:@"https://api.example.com/v1/audio/transcriptions"]];
	backend.modelName = @"whisper-1";
	XCTAssertEqualObjects(backend.backendIdentifier, @"remote-transcription");
	XCTAssertTrue(backend.isReady);                         // an endpoint is set
}

- (void)testExampleCoreMLImageBackendContract
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:[NSURL fileURLWithPath:@"/models/style.mlpackage"]];
	XCTAssertEqualObjects(backend.backendIdentifier, @"coreml");
	XCTAssertFalse(backend.isReady);
}

- (void)testExampleRemoteProviderPresets
{
	// Point at a service by name rather than a hand-typed URL. The preset carries the endpoint and the
	// protocol; the caller supplies the key and the model.
	NFKRemoteProvider *ollama = [NFKRemoteProvider providerWithIdentifier:@"ollama"];
	XCTAssertFalse(ollama.requiresAPIKey, @"a local server needs no key");

	id<NFKInferenceBackend> local = [NFKRemoteProvider backendForProvider:ollama
																   apiKey:nil
																modelName:@"llama3.2"];
	XCTAssertTrue(local.isReady);

	// Anthropic speaks a different protocol, so the factory hands back its own backend; the calling
	// code around it is unchanged.
	id<NFKInferenceBackend> claude = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.anthropic
																	apiKey:@"sk-ant-…"
																 modelName:@"claude-sonnet-4-5"];
	XCTAssertEqualObjects(claude.backendIdentifier, @"anthropic-messages");

	// A system turn is written the same way for both; the Anthropic backend lifts it into the
	// top-level field the Messages API expects.
	NFKInferenceRequest *chat = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[
		@{ @"role": @"system", @"content": @"Answer in one sentence." },
		@{ @"role": @"user", @"content": @"What is InferKit?" },
	] }];
	XCTAssertEqual(chat.messages.count, 2);
}

- (void)testExampleAsyncGenerationBackendContract
{
	ExampleImageGenerationBackend *backend = [[ExampleImageGenerationBackend alloc] init];
	backend.submitURL = [NSURL URLWithString:@"https://api.example.com/v1/generations"];
	backend.apiKey = @"secret-key";
	backend.modelName = @"my-generator";
	backend.pollInterval = 1.0;
	XCTAssertEqualObjects(backend.backendIdentifier, @"async-generation");
	XCTAssertTrue(backend.isReady);                         // a submit URL is set

	// The template methods map the service's JSON; the base drives submit → poll → fetch (needs network).
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse" }];
	XCTAssertEqualObjects([backend submitBodyForRequest:request][@"prompt"], @"a lighthouse");
	XCTAssertEqualObjects([backend jobIdentifierFromResponse:@{ @"id": @"job-42" }], @"job-42");
	XCTAssertTrue([backend isSucceededStatusResponse:@{ @"status": @"succeeded" }]);
	NFKInferenceResult *result = [backend resultFromStatusResponse:@{ @"output": @"done" } error:NULL];
	XCTAssertEqualObjects(result.text, @"done");
}

@end
