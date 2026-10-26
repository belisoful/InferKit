//
//  NFKServedModelToolTests.m
//  InferKitTests
//
//  A catalog model served as an MCP tool takes its schema, description, and argument mapping from its
//  descriptor. A test engine builds a translator that answers with what it received.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKServedToolTestEngine = @"tool-test-engine";

/*! A backend that answers with the prompt and the target language, the parameters it received, and what
	each file input was. */
@interface NFKServedToolTestBackend : NSObject <NFKInferenceBackend>
@property (atomic, copy, nullable) NSURL *lastAudioURL;
@end

@implementation NFKServedToolTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"tool-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	NSString *target = [request parameterForKey:NFKParameterTargetLanguage];
	if (target == nil) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput
									 userInfo:@{ NSLocalizedDescriptionKey: @"no target language" }];
		}
		return nil;
	}
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	outputs[NFKOutputText] = [NSString stringWithFormat:@"%@ -> %@", request.prompt ?: @"", target];
	outputs[@"parameters"] = request.parameters;
	id image = [request inputForKey:NFKInputImage];
	if (image != nil) {
		outputs[@"image"] = CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID() ? @"pixelBuffer" : NSStringFromClass([image class]);
	}
	id audio = [request inputForKey:NFKInputAudio];
	if ([audio isKindOfClass:NFKAudioAsset.class]) {
		self.lastAudioURL = [(NFKAudioAsset *)audio fileURL];
		outputs[@"audio"] = [NSFileManager.defaultManager fileExistsAtPath:self.lastAudioURL.path] ? @"file" : @"missing";
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

@end

@interface NFKServedToolTestBuilder : NSObject <NFKModelBackendBuilding>
@property (atomic, strong) NFKServedToolTestBackend *backend;
@end

@implementation NFKServedToolTestBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	return self.backend;
}

@end

@interface NFKServedModelToolTests : XCTestCase
@property (nonatomic) NFKServedToolTestBuilder *builder;
@property (nonatomic) NFKModelCatalog *catalog;
@property (nonatomic) NFKInferenceServer *server;
@end

@implementation NFKServedModelToolTests

- (void)setUp
{
	[super setUp];
	self.builder = [[NFKServedToolTestBuilder alloc] init];
	self.builder.backend = [[NFKServedToolTestBackend alloc] init];
	[NFKModelCatalog registerBackendBuilder:self.builder forEngine:NFKServedToolTestEngine];
	self.catalog = [[NFKModelCatalog alloc] init];
	self.catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
	NSError *error = nil;
	XCTAssertTrue([self.catalog addDescriptorsFromManifestData:[self manifest] error:&error], @"%@", error);
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.advertisesService = NO;
	self.server.modelCatalog = self.catalog;
	XCTAssertTrue([self.server addModelWithIdentifier:@"tool.translator" options:nil catalog:nil forModelName:@"translator" error:&error],
				  @"%@", error);
}

- (void)tearDown
{
	[self.server stop];
	[NFKModelCatalog registerBackendBuilder:nil forEngine:NFKServedToolTestEngine];
	[super tearDown];
}

#pragma mark Helpers

// A translator that reads a prompt and optionally an image and audio, with a required target language, a
// choice with a default, a bounded integer, and a load option the tool leaves out.
- (NSData *)manifest
{
	NSString *json = @"{ \"schemaVersion\": 2, \"kind\": \"inferkit.model-catalog\", \"models\": [ {"
		"\"id\": \"tool.translator\", \"displayName\": \"Test Translator\", \"family\": \"test\", \"engine\": \"tool-test-engine\","
		"\"summary\": \"Translates text between languages.\", \"tasks\": [\"translation\"],"
		"\"inputs\": [ { \"key\": \"prompt\", \"required\": true, \"help\": \"The text to translate.\" },"
		"  { \"key\": \"image\" }, { \"key\": \"audio\" } ],"
		"\"outputs\": [ { \"key\": \"text\", \"required\": true, \"title\": \"the translation\" } ],"
		"\"options\": ["
		"{ \"key\": \"targetLanguage\", \"kind\": \"text\", \"required\": true, \"help\": \"The language to translate into.\" },"
		"{ \"key\": \"sourceLanguage\", \"kind\": \"text\" },"
		"{ \"key\": \"formality\", \"kind\": \"choice\", \"default\": \"neutral\","
		"  \"choices\": [ { \"value\": \"formal\" }, { \"value\": \"neutral\" }, { \"value\": \"informal\" } ] },"
		"{ \"key\": \"alternatives\", \"kind\": \"integer\", \"minimum\": 1, \"maximum\": 4 },"
		"{ \"key\": \"mode\", \"kind\": \"text\", \"stage\": \"load\", \"default\": \"plain\" } ],"
		"\"presets\": [ { \"id\": \"french\", \"suggested\": true, \"requestDefaults\": { \"targetLanguage\": \"fr\" } } ],"
		"\"variants\": [ { \"id\": \"system\", \"isDefault\": true, \"source\": { \"kind\": \"system\" } } ] } ] }";
	return [json dataUsingEncoding:NSUTF8StringEncoding];
}

- (NFKBackendTool *)toolForModelName:(NSString *)modelName
{
	NSError *error = nil;
	NFKBackendTool *tool = [self.server addToolForModelName:modelName outputModality:NFKModalityText purpose:nil error:&error];
	XCTAssertNotNil(tool, @"%@", error);
	return tool;
}

/*! The result of one tools/call over the server's MCP route. */
- (NSDictionary *)callTool:(NSString *)name arguments:(NSDictionary *)arguments
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	request.HTTPMethod = @"POST";
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"tools/call",
																  @"params": @{ @"name": name, @"arguments": arguments } }
													   options:0 error:NULL];
	__block NSDictionary *reply = nil;
	XCTestExpectation *answered = [self expectationWithDescription:@"answered"];
	[[NSURLSession.sharedSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		id object = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
		reply = [object isKindOfClass:NSDictionary.class] ? object[@"result"] : nil;
		[answered fulfill];
	}] resume];
	[self waitForExpectations:@[ answered ] timeout:30];
	return reply;
}

- (NSString *)textOf:(NSDictionary *)result
{
	return [result[@"content"] firstObject][@"text"];
}

- (NSString *)PNGDataURL
{
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, space, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGImageRef image = CGBitmapContextCreateImage(context);
	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)image];
	CGImageRelease(image);
	CGContextRelease(context);
	CGColorSpaceRelease(space);
	return [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
}

/*! A data URL of a WAV of 0.1 s of 16 kHz mono silence. */
- (NSString *)WAVDataURL
{
	uint32_t samples = 1600, rate = 16000, dataBytes = samples * 2;
	NSMutableData *wav = [NSMutableData data];
	uint32_t riffSize = 36 + dataBytes, formatSize = 16, byteRate = rate * 2;
	uint16_t pcm = 1, channels = 1, blockAlign = 2, bits = 16;
	[wav appendBytes:"RIFF" length:4];
	[wav appendBytes:&riffSize length:4];
	[wav appendBytes:"WAVEfmt " length:8];
	[wav appendBytes:&formatSize length:4];
	[wav appendBytes:&pcm length:2];
	[wav appendBytes:&channels length:2];
	[wav appendBytes:&rate length:4];
	[wav appendBytes:&byteRate length:4];
	[wav appendBytes:&blockAlign length:2];
	[wav appendBytes:&bits length:2];
	[wav appendBytes:"data" length:4];
	[wav appendBytes:&dataBytes length:4];
	[wav increaseLengthBy:dataBytes];
	return [@"data:audio/wav;base64," stringByAppendingString:[wav base64EncodedStringWithOptions:0]];
}

#pragma mark Schema

- (void)testTheToolsSchemaComesFromTheDescriptor
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NSDictionary *properties = tool.parameters[@"properties"];
	XCTAssertEqualObjects([NSSet setWithArray:[properties allKeys]],
						  ([NSSet setWithArray:@[ @"prompt", @"image", @"audio", @"targetLanguage", @"sourceLanguage", @"formality", @"alternatives" ]]),
						  @"the load option is left out, and a model without messages takes no system argument");
	XCTAssertEqualObjects(tool.parameters[@"required"], (@[ @"prompt", @"targetLanguage" ]));
	XCTAssertEqualObjects(properties[@"prompt"][@"type"], @"string");
	XCTAssertEqualObjects(properties[@"prompt"][@"description"], @"The text to translate.");
	XCTAssertEqualObjects(properties[@"targetLanguage"][@"description"], @"The language to translate into.");
	XCTAssertEqualObjects(properties[@"formality"][@"enum"], (@[ @"formal", @"neutral", @"informal" ]));
	XCTAssertEqualObjects(properties[@"formality"][@"default"], @"neutral");
	XCTAssertEqualObjects(properties[@"alternatives"][@"type"], @"integer");
	XCTAssertEqualObjects(properties[@"alternatives"][@"minimum"], @1);
	XCTAssertEqualObjects(properties[@"alternatives"][@"maximum"], @4);
	XCTAssertEqualObjects(properties[@"image"][@"type"], (@[ @"string", @"object" ]));
	XCTAssertTrue([properties[@"image"][@"description"] containsString:@"data:image/png;base64"]);
	XCTAssertTrue([tool.toolDescription hasPrefix:@"Runs \"translator\" (Test Translator), a translation model. Translates text between languages."],
				  @"%@", tool.toolDescription);
	XCTAssertTrue([tool.toolDescription containsString:@"a path on the server's machine is refused"]);
	XCTAssertTrue([tool.toolDescription hasSuffix:@"Returns the translation."], @"%@", tool.toolDescription);
	XCTAssertFalse([tool.toolDescription containsString:@"language model"]);
}

- (void)testAPresetsDefaultCoversARequiredOption
{
	NFKBackendTool *tool = [self toolForModelName:@"translator@french"];
	XCTAssertEqualObjects(tool.parameters[@"required"], (@[ @"prompt" ]));
	XCTAssertEqualObjects(tool.parameters[@"properties"][@"targetLanguage"][@"default"], @"fr");
	XCTAssertEqualObjects([tool invokeWithArguments:@{ @"prompt": @"Hello" }].text, @"Hello -> fr");
}

- (void)testAModelWithoutADescriptorKeepsTheDefaultSchema
{
	[self.server addBackend:[[NFKServedToolTestBackend alloc] init] forModelName:@"plain"];
	NFKBackendTool *tool = [self toolForModelName:@"plain"];
	XCTAssertEqualObjects(tool.parameters, [NFKBackendTool defaultParametersForOutputModality:NFKModalityText]);
	XCTAssertTrue([tool.toolDescription containsString:@"a separate language model"]);
}

- (void)testToolsListServesTheDescriptorsSchema
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.server.localMCPURL];
	request.HTTPMethod = @"POST";
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:@{ @"jsonrpc": @"2.0", @"id": @1, @"method": @"tools/list" } options:0 error:NULL];
	__block NSDictionary *listed = nil;
	XCTestExpectation *answered = [self expectationWithDescription:@"listed"];
	[[NSURLSession.sharedSession dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *taskError) {
		listed = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
		[answered fulfill];
	}] resume];
	[self waitForExpectations:@[ answered ] timeout:30];
	NSDictionary *served = [listed[@"result"][@"tools"] firstObject];
	XCTAssertEqualObjects(served[@"name"], tool.name);
	XCTAssertEqualObjects(served[@"inputSchema"][@"required"], (@[ @"prompt", @"targetLanguage" ]));
}

#pragma mark Calls

- (void)testACallMapsArgumentsToInputsAndResolvedOptions
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
	NSDictionary *result = [self callTool:tool.name arguments:@{ @"prompt": @"The weather is lovely.", @"targetLanguage": @"fr",
																 @"formality": @"formal", @"alternatives": @2 }];
	XCTAssertEqualObjects(result[@"isError"], @NO, @"%@", result);
	XCTAssertEqualObjects([self textOf:result], @"The weather is lovely. -> fr");

	NFKToolResult *direct = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"de" }];
	XCTAssertEqualObjects(direct.text, @"Hi -> de");
	NSDictionary *parameters = direct.outputs[@"parameters"];
	XCTAssertNil(parameters[@"formality"], @"an option the call leaves out stays the model's default");
	XCTAssertNil(parameters[@"mode"]);
}

- (void)testACallMissingARequiredOptionOrOutOfBoundsIsRefused
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NFKToolResult *missing = [tool invokeWithArguments:@{ @"prompt": @"Hi" }];
	XCTAssertTrue(missing.isError);
	XCTAssertTrue([missing.text containsString:@"targetLanguage"], @"%@", missing.text);
	NFKToolResult *outOfBounds = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr", @"alternatives": @9 }];
	XCTAssertTrue(outOfBounds.isError);
	XCTAssertTrue([outOfBounds.text containsString:@"alternatives"], @"%@", outOfBounds.text);
	NFKToolResult *unknownChoice = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr", @"formality": @"rude" }];
	XCTAssertTrue(unknownChoice.isError);
}

- (void)testFileArgumentsTakeDataURLsAndTheirFilesAreRemovedAfterTheRun
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
	NSDictionary *result = [self callTool:tool.name arguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr",
																 @"image": [self PNGDataURL], @"audio": [self WAVDataURL] }];
	XCTAssertEqualObjects(result[@"isError"], @NO, @"%@", result);
	NSURL *audioURL = self.builder.backend.lastAudioURL;
	XCTAssertNotNil(audioURL);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:audioURL.path], @"the run removes the decoded file");

	NFKToolResult *direct = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr", @"image": [self PNGDataURL],
														  @"audio": [self WAVDataURL] }];
	XCTAssertEqualObjects(direct.outputs[@"image"], @"pixelBuffer");
	XCTAssertEqualObjects(direct.outputs[@"audio"], @"file");
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:self.builder.backend.lastAudioURL.path]);
}

- (void)testAFileArgumentNamingAPathIsRefused
{
	NFKBackendTool *tool = [self toolForModelName:@"translator"];
	NFKToolResult *path = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr", @"audio": @"/etc/hosts" }];
	XCTAssertTrue(path.isError);
	XCTAssertTrue([path.text containsString:@"data: URL"], @"%@", path.text);
	NFKToolResult *tagged = [tool invokeWithArguments:@{ @"prompt": @"Hi", @"targetLanguage": @"fr",
														  @"audio": @{ @"$nfk": @"audio", @"path": @"/etc/hosts", @"name": @"hosts" } }];
	XCTAssertTrue(tagged.isError, @"a tagged value names no file on this machine: %@", tagged.text);
	XCTAssertNil(self.builder.backend.lastAudioURL);
}

#pragma mark Options

- (void)testAnOptionStatesWhetherItIsRequired
{
	NSError *error = nil;
	NFKModelOption *required = [NFKModelOption optionWithDictionary:@{ @"key": @"targetLanguage", @"required": @YES } error:&error];
	XCTAssertTrue(required.isRequired, @"%@", error);
	XCTAssertEqualObjects(required.dictionaryRepresentation[@"required"], @YES);
	XCTAssertEqualObjects([NFKModelOption optionWithDictionary:required.dictionaryRepresentation error:NULL], required);
	NFKModelOption *optional = [NFKModelOption optionWithKey:@"sourceLanguage"];
	XCTAssertFalse(optional.isRequired);
	XCTAssertNil(optional.dictionaryRepresentation[@"required"], @"an optional option writes no field");
	XCTAssertTrue([NFKModelOption optionWithKey:@"targetLanguage" required:YES].isRequired);
	NSDictionary *malformed = @{ @"key": @"x", @"required": @"yes" };
	XCTAssertNil([NFKModelOption optionWithDictionary:malformed error:&error]);
}

@end
