//
//  NFKModelAdapterTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "NFKStubHubProtocol.h"

@interface NFKModelAdapterTests : XCTestCase
@property (nonatomic, copy) NSURL *directory;
@property (nonatomic, strong) NFKModelCatalog *catalog;
@end

@implementation NFKModelAdapterTests

- (void)setUp
{
	[super setUp];
	[NFKStubHubProtocol reset];
	NSString *name = [@"NFKModelAdapterTests-" stringByAppendingString:NSUUID.UUID.UUIDString];
	self.directory = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];
	[NSFileManager.defaultManager createDirectoryAtURL:self.directory withIntermediateDirectories:YES attributes:nil error:NULL];
	self.catalog = [[NFKModelCatalog alloc] init];
	[self.catalog addDescriptor:[self descriptor:@"sd15" bases:@[ NFKModelAdapterBaseStableDiffusion1 ]]];
	[self.catalog addDescriptor:[self descriptor:@"dreamshaper" bases:@[ NFKModelAdapterBaseStableDiffusion1 ]]];
	[self.catalog addDescriptor:[self descriptor:@"sd21" bases:@[ NFKModelAdapterBaseStableDiffusion2 ]]];
	[self.catalog addDescriptor:[self descriptor:@"chat" bases:@[]]];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
	[NFKStubHubProtocol reset];
	[super tearDown];
}

#pragma mark Fixtures

- (NFKModelDescriptor *)descriptor:(NSString *)identifier bases:(NSArray<NSString *> *)bases
{
	NSDictionary *dictionary = @{ @"id": identifier, @"family": @"test", @"engine": NFKModelEngineMLX, @"adapterBases": bases };
	return [NFKModelDescriptor descriptorWithDictionary:dictionary error:NULL];
}

/*! The bytes of a safetensors file of float16 zeros with the tensors' shapes and the metadata. */
+ (NSData *)safetensorsWithTensors:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)tensors
						  metadata:(nullable NSDictionary<NSString *, NSString *> *)metadata
{
	NSMutableDictionary *header = [NSMutableDictionary dictionary];
	NSUInteger offset = 0;
	for (NSString *name in [tensors.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSUInteger elements = 1;
		for (NSNumber *dimension in tensors[name]) {
			elements *= dimension.unsignedIntegerValue;
		}
		header[name] = @{ @"dtype": @"F16", @"shape": tensors[name], @"data_offsets": @[ @(offset), @(offset + elements * 2) ] };
		offset += elements * 2;
	}
	header[@"__metadata__"] = metadata;
	NSData *json = [NSJSONSerialization dataWithJSONObject:header options:0 error:NULL];
	uint64_t length = CFSwapInt64HostToLittle(json.length);
	NSMutableData *data = [NSMutableData dataWithBytes:&length length:8];
	[data appendData:json];
	[data increaseLengthBy:offset];
	return data;
}

- (NSURL *)fileNamed:(NSString *)name data:(NSData *)data
{
	NSURL *url = [self.directory URLByAppendingPathComponent:name];
	XCTAssertTrue([data writeToURL:url atomically:YES]);
	return url;
}

- (NFKModelAdapterReport *)inspect:(NSString *)name
						   tensors:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)tensors
						  metadata:(nullable NSDictionary<NSString *, NSString *> *)metadata
						identifier:(nullable NSString *)identifier
{
	NSURL *url = [self fileNamed:name data:[self.class safetensorsWithTensors:tensors metadata:metadata]];
	NSError *error = nil;
	NFKModelAdapterReport *report = [self.catalog inspectAdapterAtURL:url forModelIdentifier:identifier error:&error];
	XCTAssertNotNil(report, @"%@", error);
	return report;
}

// A kohya SDXL adapter in the SGM block names, with both text encoders.
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)kohyaXLTensors
{
	return @{
		@"lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight": @[ @8, @2048 ],
		@"lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_up.weight": @[ @640, @8 ],
		@"lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.alpha": @[],
		@"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": @[ @8, @768 ],
		@"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": @[ @768, @8 ],
		@"lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": @[ @8, @1280 ],
		@"lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": @[ @1280, @8 ],
	};
}

// A kohya Stable Diffusion 1 adapter in the diffusers block names.
+ (NSDictionary<NSString *, NSArray<NSNumber *> *> *)kohyaSD1Tensors
{
	return @{
		@"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": @[ @4, @768 ],
		@"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight": @[ @320, @4 ],
		@"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": @[ @4, @768 ],
		@"lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight": @[ @768, @4 ],
	};
}

#pragma mark Layout, base, rank, and targets

- (void)testAKohyaSDXLAdapterIsReadFromItsShapes
{
	NFKModelAdapterReport *report = [self inspect:@"xl.safetensors" tensors:[self.class kohyaXLTensors] metadata:nil
									   identifier:nil];
	XCTAssertEqualObjects(report.layout, NFKModelAdapterLayoutKohya);
	XCTAssertEqualObjects(report.base, NFKModelAdapterBaseStableDiffusionXL);
	XCTAssertEqualObjects(report.targets, (@[ NFKModelAdapterTargetUNet, NFKModelAdapterTargetTextEncoder,
											  NFKModelAdapterTargetTextEncoder2 ]));
	XCTAssertEqual(report.rank, 8);
	XCTAssertEqual(report.tensorByteCount, (8 * 2048 + 640 * 8 + 1 + 8 * 768 + 768 * 8 + 8 * 1280 + 1280 * 8) * 2);
	XCTAssertEqual(report.fit, NFKModelAdapterFitUnsupported);
	XCTAssertEqualObjects(report.error.domain, NFKModelAdapterErrorDomain);
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorNoModel);
	XCTAssertEqualObjects(report.error.localizedDescription,
						  @"xl.safetensors adapts an SDXL model; no model in the catalog takes SDXL adapters.");
}

- (void)testAKohyaStableDiffusion1AdapterIsReadFromItsCrossAttentionWidth
{
	NFKModelAdapterReport *report = [self inspect:@"sd1.safetensors" tensors:[self.class kohyaSD1Tensors] metadata:nil
									   identifier:@"sd15"];
	XCTAssertEqualObjects(report.layout, NFKModelAdapterLayoutKohya);
	XCTAssertEqualObjects(report.base, NFKModelAdapterBaseStableDiffusion1);
	XCTAssertEqualObjects(report.targets, (@[ NFKModelAdapterTargetUNet, NFKModelAdapterTargetTextEncoder ]));
	XCTAssertEqual(report.rank, 4);
	XCTAssertEqual(report.fit, NFKModelAdapterFitFits);
	XCTAssertEqualObjects(report.fittingModelIdentifiers, (@[ @"dreamshaper", @"sd15" ]));
	XCTAssertNil(report.error);
}

- (void)testAPEFTStableDiffusion2AdapterIsReadFromItsCrossAttentionWidth
{
	NSDictionary *tensors = @{
		@"unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight": @[ @4, @1024 ],
		@"unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.weight": @[ @320, @4 ],
	};
	NFKModelAdapterReport *report = [self inspect:@"sd2.safetensors" tensors:tensors metadata:nil identifier:@"sd21"];
	XCTAssertEqualObjects(report.layout, NFKModelAdapterLayoutPEFT);
	XCTAssertEqualObjects(report.base, NFKModelAdapterBaseStableDiffusion2);
	XCTAssertEqualObjects(report.targets, @[ NFKModelAdapterTargetUNet ]);
	XCTAssertEqual(report.rank, 4);
	XCTAssertEqual(report.fit, NFKModelAdapterFitFits);
}

- (void)testTheMetadataTellsTheBaseWhereTheShapesDoNot
{
	NSDictionary *tensors = @{
		@"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": @[ @16, @320 ],
		@"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": @[ @320, @16 ],
	};
	NFKModelAdapterReport *kohya = [self inspect:@"a.safetensors" tensors:tensors
										metadata:@{ @"ss_base_model_version": @"sd_v1" } identifier:nil];
	XCTAssertEqualObjects(kohya.base, NFKModelAdapterBaseStableDiffusion1);
	XCTAssertEqual(kohya.rank, 16);
	NFKModelAdapterReport *spec = [self inspect:@"b.safetensors" tensors:tensors
									   metadata:@{ @"modelspec.architecture": @"stable-diffusion-v2-512/lora" } identifier:nil];
	XCTAssertEqualObjects(spec.base, NFKModelAdapterBaseStableDiffusion2);
	NFKModelAdapterReport *unknown = [self inspect:@"c.safetensors" tensors:tensors metadata:nil identifier:nil];
	XCTAssertNil(unknown.base);
	XCTAssertEqual(unknown.fit, NFKModelAdapterFitUnsupported);
	XCTAssertEqual(unknown.error.code, NFKModelAdapterErrorUnknownBase);
}

- (void)testTheShapesOutrankTheMetadata
{
	NFKModelAdapterReport *report = [self inspect:@"sd1.safetensors" tensors:[self.class kohyaSD1Tensors]
										 metadata:@{ @"ss_base_model_version": @"sdxl_base_v1-0" } identifier:nil];
	XCTAssertEqualObjects(report.base, NFKModelAdapterBaseStableDiffusion1);
}

- (void)testAFluxAdapterIsReadFromItsBlockNames
{
	NSDictionary *tensors = @{
		@"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": @[ @16, @3072 ],
		@"lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": @[ @9216, @16 ],
	};
	NFKModelAdapterReport *report = [self inspect:@"flux.safetensors" tensors:tensors metadata:nil identifier:nil];
	XCTAssertEqualObjects(report.base, NFKModelAdapterBaseFlux1);
	XCTAssertEqualObjects(report.targets, @[ NFKModelAdapterTargetTransformer ]);
	XCTAssertEqualObjects(report.error.localizedDescription,
						  @"flux.safetensors adapts a FLUX.1 model; no model in the catalog takes FLUX.1 adapters.");
}

- (void)testAnMLXLMAdapterTakesItsRankFromTheSecondDimension
{
	NSDictionary *tensors = @{ @"model.layers.0.self_attn.q_proj.lora_a": @[ @896, @16 ],
							   @"model.layers.0.self_attn.q_proj.lora_b": @[ @16, @896 ] };
	NFKModelAdapterReport *report = [self inspect:@"adapters.safetensors" tensors:tensors metadata:nil identifier:nil];
	XCTAssertEqualObjects(report.layout, NFKModelAdapterLayoutMLXLM);
	XCTAssertEqualObjects(report.targets, @[ NFKModelAdapterTargetDecoder ]);
	XCTAssertEqual(report.rank, 16);
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorUnknownBase);
}

- (void)testAPEFTLanguageAdapterReadsItsConfiguration
{
	NSDictionary *configuration = @{ @"r": @32, @"base_model_name_or_path": @"Qwen/Qwen3-0.6B" };
	[self fileNamed:@"adapter_config.json" data:[NSJSONSerialization dataWithJSONObject:configuration options:0 error:NULL]];
	NSDictionary *tensors = @{ @"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": @[ @1024, @32 ] };
	NFKModelAdapterReport *report = [self inspect:@"adapter_model.safetensors" tensors:tensors metadata:nil identifier:nil];
	XCTAssertEqualObjects(report.layout, NFKModelAdapterLayoutPEFT);
	XCTAssertEqualObjects(report.targets, @[ NFKModelAdapterTargetDecoder ]);
	XCTAssertEqual(report.rank, 32);
	XCTAssertEqualObjects(report.baseModelName, @"Qwen/Qwen3-0.6B");
	XCTAssertEqualObjects(report.base, @"Qwen/Qwen3-0.6B");
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorNoModel);
}

#pragma mark Refusals

- (void)testALoHaAdapterIsAnUnknownLayout
{
	NSDictionary *tensors = @{ @"lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_a": @[ @320, @4 ],
							   @"lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_b": @[ @4, @320 ] };
	NFKModelAdapterReport *report = [self inspect:@"loha.safetensors" tensors:tensors metadata:nil identifier:nil];
	XCTAssertNil(report.layout);
	XCTAssertEqual(report.fit, NFKModelAdapterFitUnsupported);
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorUnknownLayout);
	XCTAssertTrue([report.error.localizedDescription containsString:@"LoHa"], @"%@", report.error);
}

- (void)testTensorsInNoLayoutAreAnUnknownLayout
{
	NFKModelAdapterReport *report = [self inspect:@"weights.safetensors" tensors:@{ @"model.embed.weight": @[ @4, @4 ] }
										 metadata:nil identifier:nil];
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorUnknownLayout);
}

- (void)testAPickleFileIsRefusedWithoutBeingRead
{
	NSURL *url = [self.directory URLByAppendingPathComponent:@"style.ckpt"];
	NSError *error = nil;
	NFKModelAdapterReport *report = [self.catalog inspectAdapterAtURL:url forModelIdentifier:@"sd15" error:&error];
	XCTAssertNotNil(report, @"%@", error);
	XCTAssertEqual(report.fit, NFKModelAdapterFitUnsupported);
	XCTAssertEqual(report.error.code, NFKModelAdapterErrorPickle);
	XCTAssertEqual(report.fileByteCount, 0);
	XCTAssertNil(report.sha256);
	[self fileNamed:@"style.pt" data:[NSMutableData dataWithLength:100]];
	NFKModelAdapterReport *present = [self.catalog inspectAdapterAtURL:[self.directory URLByAppendingPathComponent:@"style.pt"]
													forModelIdentifier:nil error:&error];
	XCTAssertEqual(present.error.code, NFKModelAdapterErrorPickle);
	XCTAssertEqual(present.fileByteCount, 100);
}

- (void)testAnUnreadableFileIsAnError
{
	NSError *error = nil;
	NSURL *missing = [self.directory URLByAppendingPathComponent:@"missing.safetensors"];
	XCTAssertNil([self.catalog inspectAdapterAtURL:missing forModelIdentifier:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelAdapterErrorDomain);
	XCTAssertEqual(error.code, NFKModelAdapterErrorUnreadable);
	NSURL *text = [self fileNamed:@"text.safetensors" data:[@"not a safetensors file" dataUsingEncoding:NSUTF8StringEncoding]];
	XCTAssertNil([self.catalog inspectAdapterAtURL:text forModelIdentifier:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelAdapterErrorUnreadable);
	XCTAssertEqualObjects(error.localizedDescription, @"text.safetensors is not a safetensors file.");
}

#pragma mark Trigger words

- (void)testTriggerWordsComeFromTheTriggerPhrase
{
	NFKModelAdapterReport *report = [self inspect:@"a.safetensors" tensors:[self.class kohyaSD1Tensors]
										 metadata:@{ @"modelspec.trigger_phrase": @"wtrclr, watercolor style , " }
									   identifier:nil];
	XCTAssertEqualObjects(report.triggerWords, (@[ @"wtrclr", @"watercolor style" ]));
}

- (void)testTriggerWordsAreTheMostFrequentTrainingTags
{
	NSDictionary *frequency = @{ @"10_a": @{ @"cat": @5, @" dog ": @3, @"bird": @1 },
								 @"5_b": @{ @"dog": @4, @"eel": @3, @"fox": @3, @"ant": @1 } };
	NSData *json = [NSJSONSerialization dataWithJSONObject:frequency options:0 error:NULL];
	NSString *text = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
	NFKModelAdapterReport *report = [self inspect:@"a.safetensors" tensors:[self.class kohyaSD1Tensors]
										 metadata:@{ @"ss_tag_frequency": text } identifier:nil];
	XCTAssertEqualObjects(report.triggerWords, (@[ @"dog", @"cat", @"eel", @"fox", @"ant" ]));
	XCTAssertEqualObjects(report.metadata[@"ss_tag_frequency"], text);
}

#pragma mark Fit

- (void)testAnAdapterForAnotherModelNamesTheModelsThatTakeIt
{
	NFKModelAdapterReport *named = [self inspect:@"a.safetensors" tensors:[self.class kohyaSD1Tensors] metadata:nil
									  identifier:@"sd21"];
	XCTAssertEqual(named.fit, NFKModelAdapterFitFitsOtherModels);
	XCTAssertEqualObjects(named.fittingModelIdentifiers, (@[ @"dreamshaper", @"sd15" ]));
	XCTAssertNil(named.error);
	NFKModelAdapterReport *unnamed = [self inspect:@"b.safetensors" tensors:[self.class kohyaSD1Tensors] metadata:nil
										identifier:nil];
	XCTAssertEqual(unnamed.fit, NFKModelAdapterFitFitsOtherModels);
	NFKModelAdapterReport *unknown = [self inspect:@"c.safetensors" tensors:[self.class kohyaSD1Tensors] metadata:nil
										identifier:@"no-such-model"];
	XCTAssertEqual(unknown.fit, NFKModelAdapterFitFitsOtherModels);
}

- (void)testALocalFileReportsItsSizeAndChecksum
{
	NSData *data = [self.class safetensorsWithTensors:[self.class kohyaSD1Tensors] metadata:@{ @"format": @"pt" }];
	NSURL *url = [self fileNamed:@"sd1.safetensors" data:data];
	NFKModelAdapterReport *report = [self.catalog inspectAdapterAtURL:url forModelIdentifier:@"sd15" error:NULL];
	XCTAssertEqual(report.fileByteCount, (long long)data.length);
	XCTAssertEqualObjects(report.sha256, [NFKStubHubProtocol sha256OfData:data]);
	NSDictionary *dictionary = report.dictionaryRepresentation;
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:dictionary]);
	XCTAssertEqualObjects(dictionary[@"fit"], @"fits");
	XCTAssertEqualObjects(dictionary[@"base"], NFKModelAdapterBaseStableDiffusion1);
	XCTAssertEqualObjects(dictionary[@"sha256"], report.sha256);
	XCTAssertEqualObjects(dictionary[@"fileByteCount"], @(data.length));
	XCTAssertNil(dictionary[@"error"]);
}

#pragma mark Repositories

- (NFKHFHub *)stubHub
{
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:[self.directory URLByAppendingPathComponent:@"cache"]];
	hub.endpointURL = NFKStubHubProtocol.endpointURL;
	hub.session = [NFKStubHubProtocol session];
	hub.accessToken = @"";
	hub.excludesCacheFromBackup = NO;
	return hub;
}

- (void)inspectRepository:(NSString *)repository
					 path:(nullable NSString *)path
			   completion:(void (^)(NFKModelAdapterReport * _Nullable report, NSError * _Nullable error))completion
{
	XCTestExpectation *done = [self expectationWithDescription:repository];
	[self.catalog inspectAdapterInRepository:repository revision:@"main" path:path forModelIdentifier:@"sd15"
										 hub:[self stubHub]
						   completionHandler:^(NFKModelAdapterReport *report, NSError *error) {
		completion(report, error);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:30];
}

- (void)testARepositoryAdapterIsReadFromItsHeaderAndListing
{
	NSData *file = [self.class safetensorsWithTensors:[self.class kohyaSD1Tensors] metadata:nil];
	NSData *configuration = [NSJSONSerialization dataWithJSONObject:@{ @"base_model_name_or_path": @"sd-legacy/sd-v1-5" }
															options:0 error:NULL];
	[NFKStubHubProtocol serveRepo:@"org/watercolor" revision:@"main"
							files:@{ @"watercolor.safetensors": file, @"adapter_config.json": configuration,
									 @"README.md": [NSData data] }
						 lfsPaths:[NSSet setWithObject:@"watercolor.safetensors"]];
	[self inspectRepository:@"org/watercolor" path:nil completion:^(NFKModelAdapterReport *report, NSError *error) {
		XCTAssertNotNil(report, @"%@", error);
		XCTAssertEqualObjects(report.fileName, @"watercolor.safetensors");
		XCTAssertEqualObjects(report.base, NFKModelAdapterBaseStableDiffusion1);
		XCTAssertEqual(report.fit, NFKModelAdapterFitFits);
		XCTAssertEqualObjects(report.baseModelName, @"sd-legacy/sd-v1-5");
		XCTAssertEqual(report.fileByteCount, (long long)file.length);
		XCTAssertEqualObjects(report.sha256, [NFKStubHubProtocol sha256OfData:file]);
	}];
	NSString *resolve = @"/org/watercolor/resolve/main/watercolor.safetensors";
	XCTAssertEqualObjects([[NFKStubHubProtocol requestsForPath:resolve].firstObject valueForHTTPHeaderField:@"Range"], @"bytes=0-7");
}

- (void)testARepositoryOfSeveralSafetensorsFilesNeedsAPath
{
	NSData *file = [self.class safetensorsWithTensors:[self.class kohyaSD1Tensors] metadata:nil];
	[NFKStubHubProtocol serveRepo:@"org/pair" revision:@"main" files:@{ @"a.safetensors": file, @"b.safetensors": file }
						 lfsPaths:[NSSet set]];
	[self inspectRepository:@"org/pair" path:nil completion:^(NFKModelAdapterReport *report, NSError *error) {
		XCTAssertNil(report);
		XCTAssertEqual(error.code, NFKModelAdapterErrorUnreadable);
		XCTAssertTrue([error.localizedDescription containsString:@"a.safetensors, b.safetensors"], @"%@", error);
	}];
	[self inspectRepository:@"org/pair" path:@"b.safetensors" completion:^(NFKModelAdapterReport *report, NSError *error) {
		XCTAssertEqualObjects(report.fileName, @"b.safetensors", @"%@", error);
		XCTAssertEqual(report.fit, NFKModelAdapterFitFits);
	}];
}

- (void)testARepositoryOfPickleFilesIsAPickleReport
{
	[NFKStubHubProtocol serveRepo:@"org/old" revision:@"main" files:@{ @"style.bin": [NSMutableData dataWithLength:64] }
						 lfsPaths:[NSSet set]];
	[self inspectRepository:@"org/old" path:nil completion:^(NFKModelAdapterReport *report, NSError *error) {
		XCTAssertNotNil(report, @"%@", error);
		XCTAssertEqual(report.error.code, NFKModelAdapterErrorPickle);
		XCTAssertEqual(report.fileByteCount, 64);
		XCTAssertNil(report.sha256);
	}];
}

- (void)testAGatedRepositoryPassesTheHubsAuthenticationErrorThrough
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:401 body:nil]
							forPath:@"/api/models/org/gated/tree/main"];
	[self inspectRepository:@"org/gated" path:nil completion:^(NFKModelAdapterReport *report, NSError *error) {
		XCTAssertNil(report);
		XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
		XCTAssertEqual(error.code, NFKDownloadErrorAuthenticationRequired);
	}];
}

@end
