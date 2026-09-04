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

- (void)testExampleRemoteModelDiscovery
{
	// No preset carries a default model name; the provider's own list is where a picker is filled
	// from. Every preset derives its URLs from one base, and a preset re-points at another address
	// with that one field changed — a runner on another port, or on another machine.
	NFKRemoteProvider *ollama = NFKRemoteProvider.ollama;
	XCTAssertEqualObjects(ollama.modelsURL.absoluteString, @"http://localhost:11434/v1/models");

	NFKRemoteProvider *lanOllama = [ollama providerWithBaseURL:[NSURL URLWithString:@"http://192.168.1.20:11434/v1"]];
	XCTAssertEqualObjects(lanOllama.identifier, @"ollama");
	XCTAssertEqualObjects(lanOllama.endpointURL.absoluteString, @"http://192.168.1.20:11434/v1/chat/completions");
	XCTAssertEqualObjects([NFKRemoteProvider.openAI URLForPath:@"audio/transcriptions"].absoluteString,
						  @"https://api.openai.com/v1/audio/transcriptions");

	// A runner that is not running is a different answer from an empty list. Nothing listens on the
	// discard port, so the call comes back with kNFKError_RemoteUnreachable rather than a model list.
	NFKRemoteProvider *stopped = [ollama providerWithBaseURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1"]];
	NFKRemoteModelCatalog *catalog = [NFKRemoteModelCatalog catalogForProvider:stopped apiKey:nil];
	catalog.timeout = 5;
	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [catalog modelsWithError:&error];   // blocks; off the render thread
	XCTAssertNil(models);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);

	// One entry of a provider's list, as the picker would show it.
	NFKRemoteModel *model = [NFKRemoteModel modelWithEntry:@{ @"id": @"llama3.2:latest", @"owned_by": @"library" }];
	XCTAssertEqualObjects(model.identifier, @"llama3.2:latest");
	XCTAssertEqualObjects(model.displayName, @"llama3.2:latest");
	XCTAssertEqualObjects(model.ownedBy, @"library");
}

- (void)testExampleRemoteEmbeddingsAndLocalRunners
{
	// Embeddings are the same shape everywhere, and the vector comes back under the core key the
	// on-device embedders use. Anthropic serves no embeddings endpoint, so its factory answers nil.
	NFKRemoteEmbeddingBackend *embedder = [NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.ollama
																				 apiKey:nil
																			  modelName:@"nomic-embed-text"];
	XCTAssertEqualObjects(embedder.endpointURL.absoluteString, @"http://localhost:11434/v1/embeddings");
	XCTAssertTrue(embedder.isReady);
	XCTAssertNil([NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);

	// A local runner has a second surface, its native API, which is where "what is installed",
	// "what is loaded", and "get me this model" are answered. Only the presets with one hand it back.
	id<NFKLocalModelRunner> runner = NFKRemoteProvider.ollama.localRunner;
	XCTAssertNotNil(runner);
	XCTAssertEqualObjects(runner.nativeBaseURL.absoluteString, @"http://localhost:11434");
	XCTAssertTrue([runner respondsToSelector:@selector(pullModel:)], @"Ollama downloads");
	XCTAssertFalse([NFKRemoteProvider.lmStudio.localRunner respondsToSelector:@selector(pullModel:)],
				   @"LM Studio's REST surface does not");
	XCTAssertNil(NFKRemoteProvider.llamaCpp.localRunner, @"nothing beyond the OpenAI surface to adapt");

	// One entry of Ollama's installed list, as the picker would show it: the list carries no id,
	// and the size, quantization, context length, and capabilities come with it.
	NFKRemoteModel *installed = [NFKRemoteModel modelWithEntry:@{
		@"name": @"gpt-oss:20b", @"model": @"gpt-oss:20b", @"size": @13793441244,
		@"details": @{ @"quantization_level": @"MXFP4", @"context_length": @131072 },
		@"capabilities": @[ @"completion", @"tools", @"thinking" ],
	}];
	XCTAssertEqualObjects(installed.identifier, @"gpt-oss:20b");
	XCTAssertEqualObjects(installed.quantization, @"MXFP4");
	XCTAssertEqualObjects(installed.contextLength, @131072);
	XCTAssertTrue([installed.capabilities containsObject:@"tools"]);
}

- (void)testExampleRemoteSpeechImageAndVision
{
	// Text to speech: the same NFKOutputAudio the on-device speech backend answers with, a WAV by
	// default. A voice is required and has no default, for the reason a model name has none.
	NFKRemoteSpeechBackend *speaker = [NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.openAI
																		  apiKey:@"sk-…" modelName:@"gpt-4o-mini-tts" voice:@"alloy"];
	XCTAssertEqualObjects(speaker.endpointURL.absoluteString, @"https://api.openai.com/v1/audio/speech");
	XCTAssertEqualObjects(speaker.responseFormat, @"wav");
	XCTAssertNil([NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m" voice:@"v"]);

	// Image generation chooses its operation from the request the way the Stable Diffusion backend
	// does: a prompt alone generates, an image under NFKInputImage edits it, a mask inpaints.
	NFKRemoteImageBackend *painter = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.openAI
																		apiKey:@"sk-…" modelName:@"gpt-image-1"];
	XCTAssertEqualObjects(painter.generationsURL.absoluteString, @"https://api.openai.com/v1/images/generations");
	XCTAssertEqualObjects(painter.editsURL.absoluteString, @"https://api.openai.com/v1/images/edits");

	// The codec under both, public: any of the three image representations the contract carries
	// becomes PNG bytes or a data URL, and any ImageIO-readable bytes become a 32BGRA pixel buffer.
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace,
												 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGContextSetRGBFillColor(context, 0, 0, 1, 1);
	CGContextFillRect(context, CGRectMake(0, 0, 2, 2));
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)square];
	XCTAssertNotNil(png);
	XCTAssertTrue([[NFKImageCoding dataURLForImage:(__bridge id)square] hasPrefix:@"data:image/png;base64,"]);
	CVPixelBufferRef decoded = [NFKImageCoding pixelBufferWithImageData:png];
	XCTAssertEqual(CVPixelBufferGetPixelFormatType(decoded), kCVPixelFormatType_32BGRA);
	CVPixelBufferRelease(decoded);

	// A vision question is the ordinary chat request with an image beside the prompt; the chat
	// backend attaches it to the user turn in the shape the endpoint reads. No new class.
	NFKInferenceRequest *look = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What is in this frame?",
																		   NFKInputImage: (__bridge id)square }];
	XCTAssertNotNil([look inputForKey:NFKInputImage]);
	CGImageRelease(square);
}

- (void)testExampleRemoteStreamingToolsAndRetries
{
	// Tools and a schema are contract keys, translated into each provider's shape; what the model
	// called comes back parsed under result.toolCalls, and a schema's reply under result.structured.
	NSDictionary *weather = @{ @"name": @"get_weather",
							   @"description": @"Current weather in a city.",
							   @"parameters": @{ @"type": @"object",
												 @"properties": @{ @"city": @{ @"type": @"string" } },
												 @"required": @[ @"city" ] } };
	NFKInferenceRequest *ask = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Weather in Paris?" }
														  parameters:@{ NFKParameterTools: @[ weather ] }
													  outputModality:NFKModalityText];
	XCTAssertEqual([[ask parameterForKey:NFKParameterTools] count], 1);

	NFKInferenceResult *turn = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputToolCalls: @[ @{ @"id": @"call_1", @"name": @"get_weather",
								   @"arguments": @{ @"city": @"Paris" }, @"argumentsJSON": @"{\"city\":\"Paris\"}" } ] }];
	XCTAssertEqualObjects(turn.toolCalls.firstObject[@"arguments"][@"city"], @"Paris");
	XCTAssertNil(turn.text, @"a tool-calling turn has no text");

	// The chat backends stream through the ordinary job. Nothing listens on the discard port, so the
	// job fails with the unreachable code — asynchronously, through the completion handler, which
	// is the path a chat interface reads token by token when the server is there.
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1/chat/completions"]];
	backend.timeout = 5;
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:ask];
	XCTAssertNotNil(job.cancellationHandler, @"cancelling the job cancels the request");
	XCTestExpectation *ended = [self expectationWithDescription:@"stream ended"];
	job.completionHandler = ^(NFKInferenceJob *j) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_RemoteUnreachable);

	// Every blocking remote call retries a rate limit or gateway error; the two knobs are global.
	XCTAssertEqual(NFKRemoteTransport.retryAttempts, 2);
	XCTAssertEqualWithAccuracy(NFKRemoteTransport.maximumRetryDelay, 8, 1e-9);
}

- (void)testExampleRemoteMediaModes
{
	// Beside images, the chat backends take audio, documents, and a clip beside the prompt, and can
	// answer in speech. The request is the same shape whichever engine reads it.
	NSURL *pdf = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"brief.pdf"];
	[[@"%PDF-1.4 example" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:pdf atomically:YES];
	NFKInferenceRequest *summarize = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Summarize the brief.",
																			   NFKInputDocument: pdf }];
	XCTAssertEqualObjects([summarize inputForKey:NFKInputDocument], pdf);

	NFKInferenceRequest *spoken = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Read me the summary." }
															 parameters:@{ NFKParameterAudioOutput: @{ @"voice": @"alloy" } }
														 outputModality:NFKModalityAudio];
	XCTAssertEqualObjects([spoken parameterForKey:NFKParameterAudioOutput][@"voice"], @"alloy");
	[NSFileManager.defaultManager removeItemAtURL:pdf error:NULL];

	// The transcription backend gains what the on-device Whisper backend has: timed segments under
	// NFKOutputSegments, and translation to English through the sibling endpoint.
	NFKRemoteTranscriptionBackend *ears = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.groq
																					 apiKey:@"gsk_…" modelName:@"whisper-large-v3"];
	ears.emitsTimestamps = YES;
	XCTAssertEqualObjects(ears.endpointURL.absoluteString, @"https://api.groq.com/openai/v1/audio/transcriptions");

	// Three more services: video generation as a job, rerank, and moderation.
	NFKRemoteVideoBackend *director = [NFKRemoteVideoBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"sk-…" modelName:@"sora-2"];
	XCTAssertEqualObjects(director.submitURL.absoluteString, @"https://api.openai.com/v1/videos");
	XCTAssertTrue([director isKindOfClass:NFKAsyncGenerationBackend.class], @"submit, poll, download");

	NFKRemoteReranker *ranker = [NFKRemoteReranker rerankerForProvider:NFKRemoteProvider.together apiKey:@"k" modelName:@"Salesforce/Llama-Rank-V1"];
	XCTAssertEqualObjects(ranker.endpointURL.absoluteString, @"https://api.together.xyz/v1/rerank");

	NFKRemoteModerationBackend *gate = [NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"sk-…" modelName:@"omni-moderation-latest"];
	XCTAssertEqualObjects(gate.endpointURL.absoluteString, @"https://api.openai.com/v1/moderations");
	XCTAssertNil([NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
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

// Docs/examples.md: Where Core ML actually runs. MLComputeUnits is a request; the plan reports where
// Core ML would place each operation, without running the model.
- (void)testExampleWhereCoreMLRuns
{
	NFKCoreMLBackend *backend = [NFKCoreMLBackend backendWithModelURL:nil];
	// MLComputeUnitsCPUOnly is zero, so an unset property would move every model to the CPU.
	XCTAssertEqual(backend.computeUnits, MLComputeUnitsAll);
	backend.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
	XCTAssertEqual(backend.computeUnits, MLComputeUnitsCPUAndNeuralEngine);

	// Whether this OS can answer at all. An older one fails rather than reporting an empty plan,
	// because "nothing is on the Neural Engine" and "cannot tell" are different answers.
	if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, *)) {
		XCTAssertTrue(NFKComputePlan.isAvailable);
	} else {
		XCTAssertFalse(NFKComputePlan.isAvailable);
	}

	NSURL *absent = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
					 URLByAppendingPathComponent:@"inferkit-example-absent.mlmodelc"];
	NSError *error = nil;
	NFKComputePlan *plan = [NFKComputePlan planForCompiledModelAtURL:absent
														computeUnits:backend.computeUnits
															   error:&error];
	XCTAssertNil(plan);
	XCTAssertNotNil(error);

	// With a real .mlmodelc the plan reads:
	//   plan.describedPlacement            "142 operations: 138 Neural Engine, 4 GPU, 0 CPU, …"
	//   plan.neuralEngineFraction          the share to watch while tuning a conversion
	//   plan.operatorNamesOffNeuralEngine  the operators to work on, most frequent first
}

// Docs/examples.md: Will this model fit? Three ceilings that are not interchangeable, plus a live
// reading of what is free.
- (void)testExampleWillThisModelFit
{
	NFKHardwareProfile *machine = NFKHardwareProfile.currentProfile;
	XCTAssertGreaterThan(machine.physicalMemory, 0);
	XCTAssertGreaterThan(machine.describedMachine.length, 0);

	// Size against Metal's recommendation, not the physical total: it is what the system expects to
	// stay resident, and it sits below what is installed.
	if (machine.recommendedWorkingSetSize > 0) {
		XCTAssertLessThan(machine.recommendedWorkingSetSize, machine.physicalMemory);
	}
	// A separate ceiling: one tensor cannot exceed this however much of the budget is unspent.
	XCTAssertGreaterThanOrEqual(machine.maximumBufferLength, 0);

	// Live — this is the number that decides whether a load succeeds right now.
	XCTAssertGreaterThan([NFKHardwareProfile availableMemory], 0);

	// Readings degrade rather than throwing, so a machine this was never run on still reports.
	XCTAssertNotNil(machine.chipName);
	XCTAssertNotNil(machine.graphicsArchitecture);
}

@end
