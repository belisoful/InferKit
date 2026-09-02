//
//  MLXObjCExample.m
//  InferKitMLXObjCExamples
//
//  Proves that an Objective-C consumer (an FCPX plugin, an app) builds and drives a local MLX model
//  through InferKit without writing Swift: a Swift model registers a factory by name, and this
//  Objective-C code constructs the backend by that name and runs it through the NFKInferenceBackend
//  protocol. Mirrors the "Running MLX models from Objective-C" section of Docs/examples.md — keep the
//  two in sync. Running the forward needs the MLX Metal library (host-verified via xcodebuild); this
//  verifies construction and the contract.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
@import InferKitMLX;

@interface MLXObjCExample : XCTestCase
@end

@implementation MLXObjCExample

- (void)testObjectiveCBuildsAndDrivesAnMLXModelByName
{
	[NFKMLXReferenceModels registerGreenScreenKeyer];       // done once from Swift or Objective-C
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"green-screen-keyer"]);

	NSError *error = nil;
	id<NFKInferenceBackend> keyer = [NFKMLXModelRegistry backendNamed:@"green-screen-keyer" weightsURL:nil error:&error];
	XCTAssertNotNil(keyer, @"%@", error);
	XCTAssertEqualObjects(keyer.backendIdentifier, @"green-screen-keyer");

	// From here it is a normal InferKit backend: build a request and run it (off the render thread).
	// NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: plate }];
	// NFKInferenceResult *result = [keyer runInferenceForRequest:request error:&error];  // needs MLX runtime
}

- (void)testObjectiveCBuildsTheRealESRGANUpscalerByName
{
	// Real-ESRGAN is a real single-forward model (RRDBNet), not a stand-in. MetalForge registers it
	// once and builds it by name; with a downloaded safetensors checkpoint the output is a 4× upscale.
	[NFKMLXRealESRGAN register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"real-esrgan-x4"]);

	// Building the generator constructs MLXNN layers, which initializes MLX; that needs the bundled
	// metallib the Xcode build system provides but a plain `swift test` does not.
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}

	NSError *error = nil;
	id<NFKInferenceBackend> upscaler = [NFKMLXModelRegistry backendNamed:@"real-esrgan-x4" weightsURL:nil error:&error];
	XCTAssertNotNil(upscaler, @"%@", error);
	XCTAssertEqualObjects(upscaler.backendIdentifier, @"real-esrgan-x4");

	// With real weights, MetalForge would download and build in one call, then run off the render thread:
	// id<NFKInferenceBackend> u = [NFKMLXHub backendNamed:@"real-esrgan-x4" repo:@"org/real-esrgan"
	//     weightsPath:@"RealESRGAN_x4plus.safetensors" revision:nil cacheDirectoryURL:nil error:&error];
}

- (void)testObjectiveCBuildsDepthAnythingByName
{
	[NFKMLXDepthAnything register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"depth-anything-v2-small"]);
	// Building the DINOv2/DPT net initializes MLX (needs the bundled metallib), so build only under xcodebuild.
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> depth = [NFKMLXModelRegistry backendNamed:@"depth-anything-v2-small" weightsURL:nil error:&error];
	XCTAssertNotNil(depth, @"%@", error);
	XCTAssertEqualObjects(depth.backendIdentifier, @"depth-anything-v2-small");
}

- (void)testObjectiveCBuildsSAMSegmenterByName
{
	[NFKMLXSAM register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"sam"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> sam = [NFKMLXModelRegistry backendNamed:@"sam" weightsURL:nil error:&error];
	XCTAssertNotNil(sam, @"%@", error);
	XCTAssertEqualObjects(sam.backendIdentifier, @"sam");
}

- (void)testObjectiveCBuildsU2NetBackgroundRemoverByName
{
	[NFKMLXU2Net register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"u2net"]);
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"u2netp"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> cutout = [NFKMLXModelRegistry backendNamed:@"u2netp" weightsURL:nil error:&error];
	XCTAssertNotNil(cutout, @"%@", error);
	XCTAssertEqualObjects(cutout.backendIdentifier, @"u2netp");
}

- (void)testObjectiveCBuildsNAFNetRestorerByName
{
	[NFKMLXNAFNet register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"nafnet"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> restore = [NFKMLXModelRegistry backendNamed:@"nafnet" weightsURL:nil error:&error];
	XCTAssertNotNil(restore, @"%@", error);
	XCTAssertEqualObjects(restore.backendIdentifier, @"nafnet");
}

- (void)testObjectiveCBuildsRIFEFrameInterpolatorByName
{
	[NFKMLXRIFE register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"rife"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> rife = [NFKMLXModelRegistry backendNamed:@"rife" weightsURL:nil error:&error];
	XCTAssertNotNil(rife, @"%@", error);
	XCTAssertEqualObjects(rife.backendIdentifier, @"rife");
}

- (void)testObjectiveCBuildsRAFTOpticalFlowByName
{
	[NFKMLXRAFT register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"raft"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> raft = [NFKMLXModelRegistry backendNamed:@"raft" weightsURL:nil error:&error];
	XCTAssertNotNil(raft, @"%@", error);
	XCTAssertEqualObjects(raft.backendIdentifier, @"raft");
}

- (void)testObjectiveCBuildsLaMaInpainterByName
{
	[NFKMLXLaMa register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"lama-inpaint"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> inpaint = [NFKMLXModelRegistry backendNamed:@"lama-inpaint" weightsURL:nil error:&error];
	XCTAssertNotNil(inpaint, @"%@", error);
	XCTAssertEqualObjects(inpaint.backendIdentifier, @"lama-inpaint");
}

- (void)testObjectiveCBuildsMarigoldAndUpscalerByName
{
	[NFKMLXMarigold register];
	[NFKMLXSDUpscaler register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"marigold-depth"]);
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"sd-x4-upscaler"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> depth = [NFKMLXModelRegistry backendNamed:@"marigold-depth" weightsURL:nil error:&error];
	XCTAssertEqualObjects(depth.backendIdentifier, @"marigold-depth", @"%@", error);
}

- (void)testObjectiveCBuildsTextToImageByReleaseName
{
	// The release's own directory: unet/, vae/, text_encoder/, tokenizer/. Absent here, so this
	// pins the ObjC entry point and its error, not a generated picture.
	NSURL *absent = [NSURL fileURLWithPath:@"/nonexistent/stable-diffusion-v1-5"];
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKMLXTextToImage backendWithModel:NFKMLXStableDiffusionModelStableDiffusion15
															directoryURL:absent
																   error:&error];
	XCTAssertNil(backend);
	XCTAssertNotNil(error);
}

- (void)testObjectiveCBuildsStableDiffusionInpaintByName
{
	[NFKMLXStableDiffusionInpaint register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"sd-inpaint"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> inpaint = [NFKMLXModelRegistry backendNamed:@"sd-inpaint" weightsURL:nil error:&error];
	XCTAssertNotNil(inpaint, @"%@", error);
	XCTAssertEqualObjects(inpaint.backendIdentifier, @"sd-inpaint");

	// The released weights come as two diffusers checkpoints, one per network, plus the text
	// embedding the trained UNet cross-attends to (Docs/examples.md "Latent diffusion with real
	// weights"). Missing files fail the build with an error rather than a half-loaded model.
	NSURL *absent = [NSURL fileURLWithPath:@"/nonexistent/unet.safetensors"];
	NSError *twoFileError = nil;
	id<NFKInferenceBackend> released = [NFKMLXStableDiffusionInpaint backendWithUNetWeightsURL:absent
	                                                                             vaeWeightsURL:absent
	                                                                            textContextURL:nil
	                                                                                     error:&twoFileError];
	XCTAssertNil(released);
	XCTAssertNotNil(twoFileError, @"a missing checkpoint reports, not crashes");
}

- (void)testObjectiveCBuildsWhisperByName
{
	[NFKMLXWhisper register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"whisper-tiny"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> whisper = [NFKMLXModelRegistry backendNamed:@"whisper-tiny" weightsURL:nil error:&error];
	XCTAssertNotNil(whisper, @"%@", error);
	XCTAssertEqualObjects(whisper.backendIdentifier, @"whisper-tiny");
}

- (void)testObjectiveCAsksWhisperForSegmentTimes
{
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	// Timestamps are a different decode, so they are a parameter of the factory rather than a
	// property of every result. The spans come back as NFKAudioSegments under NFKOutputSegments.
	id<NFKInferenceBackend> whisper = [NFKMLXWhisper backendWithWeightsURL:nil
																tokenizer:nil
															   timestamps:YES
																	error:&error];
	XCTAssertNotNil(whisper, @"%@", error);
	XCTAssertEqualObjects(whisper.backendIdentifier, @"whisper-tiny");
	// A result from this backend carries NSArray<NFKAudioSegment *> under NFKOutputSegments beside
	// the transcript under NFKOutputText.
}

- (void)testObjectiveCBuildsDemucsByName
{
	[NFKMLXDemucs register];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"demucs"]);
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> demucs = [NFKMLXModelRegistry backendNamed:@"demucs" weightsURL:nil error:&error];
	XCTAssertNotNil(demucs, @"%@", error);
	XCTAssertEqualObjects(demucs.backendIdentifier, @"demucs");
}

- (void)testObjectiveCBuildsTextToSpeechByName
{
	// The speech backend stores a closure and constructs no MLX at build time, so this runs anywhere.
	[NFKMLXReferenceModels registerToneSpeech];
	XCTAssertTrue([NFKMLXModelRegistry isModelRegistered:@"tone-speech"]);
	NSError *error = nil;
	id<NFKInferenceBackend> speech = [NFKMLXModelRegistry backendNamed:@"tone-speech" weightsURL:nil error:&error];
	XCTAssertNotNil(speech, @"%@", error);
	XCTAssertEqualObjects(speech.backendIdentifier, @"tone-speech");
}

- (void)testObjectiveCBuildsTheMusicBackendFromAReleaseDirectory
{
	// MiniMax Music 3: NFKInputPrompt (description) + NFKInputLyrics -> a stereo NFKAudioAsset. The
	// factory takes the downloaded release DIRECTORY — the stack is 27 GB of separately licensed
	// weights, so there is no random-weights form; isReady reports whether they are present.
	NSURL *absent = [NSURL fileURLWithPath:@"/nonexistent/minimax-music3"];
	NSError *error = nil;
	id<NFKInferenceBackend> music = [NFKMLXMusic3 backendWithDirectoryURL:absent error:&error];
	XCTAssertNotNil(music, @"%@", error);
	XCTAssertEqualObjects(music.backendIdentifier, @"minimax-music3");
	XCTAssertFalse(music.isReady);
}

- (void)testObjectiveCBuildsShippedModelsViaDirectFactories
{
	// The primary path for shipped models: build directly, no register-then-lookup. Building a net
	// initializes MLX, so this runs under xcodebuild (the bundled metallib) and returns early otherwise.
	if ([[NSBundle bundleForClass:self.class].bundlePath containsString:@"/.build/"]) {
		return;
	}
	NSError *error = nil;
	id<NFKInferenceBackend> upscaler = [NFKMLXRealESRGAN backendWithVariant:NFKMLXRealESRGANVariantX4 weightsURL:nil error:&error];
	XCTAssertEqualObjects(upscaler.backendIdentifier, @"real-esrgan-x4", @"%@", error);

	id<NFKInferenceBackend> depth = [NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantBase weightsURL:nil error:&error];
	XCTAssertEqualObjects(depth.backendIdentifier, @"depth-anything-v2-base", @"%@", error);

	id<NFKInferenceBackend> cutout = [NFKMLXU2Net backendWithVariant:NFKMLXU2NetVariantLight weightsURL:nil error:&error];
	XCTAssertEqualObjects(cutout.backendIdentifier, @"u2netp", @"%@", error);

	id<NFKInferenceBackend> restore = [NFKMLXNAFNet backendWithWeightsURL:nil error:&error];
	XCTAssertEqualObjects(restore.backendIdentifier, @"nafnet", @"%@", error);

	id<NFKInferenceBackend> inpaint = [NFKMLXLaMa backendWithWeightsURL:nil error:&error];
	XCTAssertEqualObjects(inpaint.backendIdentifier, @"lama-inpaint", @"%@", error);
}

- (void)testObjectiveCDownloadFactoryFailsCleanlyForAnInvalidRepo
{
	// The download-and-build factory (no registry): with real inputs it downloads then builds, off the
	// render thread. An empty repo fails before any network or net construction, so this needs neither.
	NSError *error = nil;
	id<NFKInferenceBackend> backend =
		[NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantSmall
										   repo:@""
									weightsPath:@"model.safetensors"
									   revision:nil
							  cacheDirectoryURL:NSFileManager.defaultManager.temporaryDirectory
										  error:&error];
	XCTAssertNil(backend);
	XCTAssertNotNil(error);

	// With a real repo (off the render thread):
	// id<NFKInferenceBackend> real = [NFKMLXDepthAnything backendWithVariant:NFKMLXDepthVariantBase
	//     repo:@"org/dav2" weightsPath:@"model.safetensors" revision:nil cacheDirectoryURL:nil error:&error];
}

- (void)testAnUnknownModelNameReturnsAnError
{
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKMLXModelRegistry backendNamed:@"not-registered" weightsURL:nil error:&error];
	XCTAssertNil(backend);
	XCTAssertNotNil(error);
}

- (void)testDownloadAndBuildFailsFastForAnUnregisteredModel
{
	// The full call downloads from Hugging Face then builds; an unregistered name fails before the
	// download, so this needs no network.
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKMLXHub backendNamed:@"not-registered"
														repo:@"org/model"
												 weightsPath:@"model.safetensors"
													revision:nil
										   cacheDirectoryURL:nil
													   error:&error];
	XCTAssertNil(backend);
	XCTAssertNotNil(error);
}

- (void)testObjectiveCReachesTheMLXRuntimeKnobs
{
	// The Swift-only MLX globals (free-function seed, the GPU enum) are reachable from Objective-C
	// through the NFKMLXRandom / NFKMLXGPU wrappers.
	[NFKMLXRandom seed:42];

	[NFKMLXGPU setCacheLimit:48 * 1024 * 1024];
	XCTAssertEqual(NFKMLXGPU.cacheLimit, 48 * 1024 * 1024, @"the cache limit round-trips");
	[NFKMLXGPU clearCache];
	[NFKMLXGPU resetPeakMemory];
	XCTAssertGreaterThanOrEqual(NFKMLXGPU.activeMemory, 0);
	XCTAssertGreaterThanOrEqual(NFKMLXGPU.cacheMemory, 0);
}

- (void)testObjectiveCSelectsTheComputeDevice
{
	// MLX models the device as a Swift struct and a scoped function, so NFKMLXDevice is what an
	// Objective-C caller has. The selection covers the work the block does on this thread, which is
	// where a synchronous inference runs.
	NFKMLXDeviceType outer = NFKMLXDevice.currentType;

	__block NFKMLXDeviceType inner = outer;
	[NFKMLXDevice performOnDeviceType:NFKMLXDeviceTypeCPU block:^{
		inner = NFKMLXDevice.currentType;
	}];

	XCTAssertEqual(inner, NFKMLXDeviceTypeCPU);
	XCTAssertEqual(NFKMLXDevice.currentType, outer, @"the previous device is restored");

	// The shape a caller uses. The block covers this thread only, so it wraps the synchronous call and
	// not submitInferenceJobForRequest:, whose queue takes the global device.
	// [NFKMLXDevice performOnDeviceType:NFKMLXDeviceTypeCPU block:^{
	//     result = [backend runInferenceForRequest:request error:&error];   // needs MLX runtime
	// }];
}

// Docs/examples.md: MLX runtime knobs from Objective-C. What the machine has, and the standing caps
// an app sets once at startup instead of remembering clearCache at every model boundary.
- (void)testExampleGPUMemoryReportingAndStandingLimits
{
	// The recommended working set is Metal's own budget and is well below the physical total, so it
	// is what a model should be sized against.
	XCTAssertGreaterThan(NFKMLXGPU.physicalMemory, 0);
	XCTAssertGreaterThan(NFKMLXGPU.recommendedWorkingSetSize, 0);
	XCTAssertLessThanOrEqual(NFKMLXGPU.recommendedWorkingSetSize, NFKMLXGPU.physicalMemory);
	XCTAssertGreaterThan(NFKMLXGPU.deviceArchitecture.length, 0);

	// Cache is reclaimable and active memory is not, which is the distinction a caller deciding
	// whether to unload a model depends on.
	XCTAssertEqual(NFKMLXGPU.reclaimableMemory, NFKMLXGPU.cacheMemory);
	XCTAssertGreaterThanOrEqual(NFKMLXGPU.memoryPressure, 0.0);

	NSInteger previousCache = NFKMLXGPU.cacheLimit;
	NSInteger previousMemory = NFKMLXGPU.memoryLimit;

	[NFKMLXGPU applyStandingLimits];
	XCTAssertEqual(NFKMLXGPU.cacheLimit, NFKMLXGPU.defaultCacheCap);
	XCTAssertGreaterThan(NFKMLXGPU.memoryLimit, 0);

	[NFKMLXGPU setCacheLimit:previousCache];
	[NFKMLXGPU setMemoryLimit:previousMemory];
}

// Docs/examples.md: Loading a PyTorch checkpoint directly
- (void)testExampleObjectiveCReadsAPyTorchCheckpointWithNoPython
{
	// A real .pth comes from a release; this embedded one keeps the example runnable offline.
	NSString *base64 = @"gAKKCmz8nEb5IGqoUBkugAJN6QMugAJ9cQAoWBAAAABwcm90b2NvbF92ZXJzaW9ucQFN6QNYDQAAAGxpdHRsZV9lbmRpYW5xAohYCgAAAHR5cGVfc2l6ZXNxA31xBChYBQAAAHNob3J0cQVLAlgDAAAAaW50cQZLBFgEAAAAbG9uZ3EHSwR1dS6AAmNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxAClScQEoWAsAAABjb252LndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMoKFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgLAAAAMzkzODAxNzMyODBxBlgDAAAAY3B1cQdLBE50cQhRSwBLAksChnEJSwJLAYZxColoAClScQt0cQxScQ1YCQAAAGNvbnYuYmlhc3EOaAMoKGgEaAVYCwAAADM5MzgwMTczMDg4cQ9oB0sCTnRxEFFLAEsChXERSwGFcRKJaAApUnETdHEUUnEVdS6AAl1xAChYCwAAADM5MzgwMTczMDg4cQFYCwAAADM5MzgwMTczMjgwcQJlLgIAAAAAAAAAAAAAPwAAAL8EAAAAAAAAAAAAgD8AAABAAABAQAAAgEA=";
	NSData *fixture = [[NSData alloc] initWithBase64EncodedString:base64 options:0];
	NSURL *url = [[NSURL fileURLWithPath:NSTemporaryDirectory()]
	    URLByAppendingPathComponent:[NSString stringWithFormat:@"example-%@.pth", NSUUID.UUID.UUIDString]];
	XCTAssertTrue([fixture writeToURL:url atomically:YES]);

	// Inspect the state dict before deciding to load it.
	NSError *error = nil;
	NFKMLXTorchCheckpoint *checkpoint = [NFKMLXTorchCheckpoint checkpointWithContentsOfURL:url error:&error];
	XCTAssertNotNil(checkpoint, @"%@", error);
	XCTAssertEqualObjects(checkpoint.tensorNames, (@[ @"conv.bias", @"conv.weight" ]));
	NFKMLXTorchTensorInfo *info = [checkpoint infoForTensor:@"conv.weight"];
	XCTAssertEqualObjects(info.shape, (@[ @2, @2 ]));
	XCTAssertEqual(info.scalarType, NFKMLXTorchScalarTypeFloat32);

	// Convert on device: the output is what the model's Tools converter produces, and every
	// weightsURL: factory reads it. Or skip this step — the model loaders sniff a .pth's bytes and
	// read it directly through the same reader.
	NSURL *converted = [[url URLByDeletingPathExtension] URLByAppendingPathExtension:@"safetensors"];
	XCTAssertTrue([checkpoint writeSafetensorsToURL:converted error:&error], @"%@", error);
	[NSFileManager.defaultManager removeItemAtURL:url error:nil];
	[NSFileManager.defaultManager removeItemAtURL:converted error:nil];
}

// Docs/examples.md: An Objective-C caller configures every MLX generation option through request
// parameters — parity with the Swift NFKMLXGenerationOptions struct.
- (void)testObjectiveCConfiguresGenerationThroughRequestParameters
{
	NFKInferenceRequest *request = [[NFKInferenceRequest alloc]
		initWithInputs:@{ NFKInputPrompt: @"Explain diffraction in one sentence." }
		parameters:@{
			NFKParameterTemperature: @0.7,
			NFKMLXGenerationParameterKey.contextWindow: @4096,
			NFKMLXGenerationParameterKey.cacheQuantizationBits: @8,
			NFKMLXGenerationParameterKey.cacheQuantizationGroupSize: @64,
			NFKMLXGenerationParameterKey.prefillChunkSize: @512,
			NFKMLXGenerationParameterKey.chatTemplate: @"chatml",
		}];
	// The keys are ordinary NSString constants the request carries; with a downloaded release,
	// runInferenceForRequest: reads them the same way it reads NFKParameterTemperature.
	XCTAssertEqualObjects(request.parameters[NFKMLXGenerationParameterKey.cacheQuantizationBits], @8);
	XCTAssertEqualObjects(request.parameters[NFKMLXGenerationParameterKey.chatTemplate], @"chatml");
	XCTAssertGreaterThan(NFKMLXGenerationParameterKey.contextWindow.length, 0);

	// The backend itself is built from a downloaded release directory through the @objc factory:
	//   NFKInferenceBackend *llm = [NFKMLXLanguage backendWithDirectoryURL:dir error:&error];
	// which reads config.json, the tokenizer, and the shards. Verify the selector is reachable.
	XCTAssertTrue([NFKMLXLanguage respondsToSelector:@selector(backendWithDirectoryURL:error:)]);
}

// Docs/examples.md: The @objc gaps closed by the parity audit — variant factories, machine probes,
// and face landmarks all reach Objective-C.
- (void)testObjectiveCReachesTheParityAuditFactories
{
	NSError *error = nil;

	// Whisper builds every released size from ObjC, not only tiny.
	id<NFKInferenceBackend> whisperSmall =
		[NFKMLXWhisper backendWithVariant:NFKMLXWhisperVariantSmall weightsURL:nil error:&error];
	XCTAssertNotNil(whisperSmall, @"%@", error);
	XCTAssertTrue([NFKMLXWhisper respondsToSelector:
		@selector(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)]);

	// The variant download factories reach every size, not just the default geometry.
	XCTAssertTrue([NFKMLXNAFNet respondsToSelector:
		@selector(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)]);
	XCTAssertTrue([NFKMLXYOLO respondsToSelector:
		@selector(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:error:)]);
	XCTAssertTrue([NFKMLXSwinIR respondsToSelector:
		@selector(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)]);

	// RetinaFace now has the async download peer every other model ships.
	XCTAssertTrue([NFKMLXRetinaFace respondsToSelector:
		@selector(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)]);

	// The measured-bandwidth machine probe is on NFKMLXGPU beside the other machine properties.
	XCTAssertGreaterThanOrEqual([NFKMLXGPU measuredMemoryBandwidthWithMegabytes:8 repetitions:2], 0.0);
	[NFKMLXGPU resetMeasuredBandwidth];

	// A face detector is @objc-constructible and its faces carry the five landmarks, not only a box.
	NFKMLXRetinaFaceDetector *detector =
		[NFKMLXRetinaFace detectorWithWeightsURL:nil confidenceThreshold:0.8 suppressionThreshold:0.4 error:&error];
	XCTAssertNotNil(detector, @"%@", error);
	XCTAssertTrue([detector respondsToSelector:@selector(facesInImage:error:)]);
	// A returned face exposes its box, confidence, and the five named landmarks to ObjC.
	XCTAssertTrue([NFKFaceObservation instancesRespondToSelector:@selector(boundingBox)]);
	XCTAssertTrue([NFKFaceObservation instancesRespondToSelector:@selector(leftEye)]);
	XCTAssertTrue([NFKFaceObservation instancesRespondToSelector:@selector(rightMouthCorner)]);
}

@end
