//
//  NFKServerPresetTests.m
//  InferKitTests
//
//  Models hosted from a catalog under presets and aliases, over HTTP on a loopback port. A test engine
//  builds a backend that answers with the parameters it received and the mode it was built with.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKPresetTestEngine = @"preset-test-engine";

/*! A backend that answers with the steps and temperature it was asked for and the mode it was built
	with. */
@interface NFKPresetTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, copy, nullable) NSString *mode;
@end

@implementation NFKPresetTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"preset-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok",
													@"steps": [request parameterForKey:@"steps"] ?: @0,
													@"temperature": [request parameterForKey:@"temperature"] ?: @-1,
													@"mode": self.mode ?: @"" }];
}

@end

/*! The engine's builder, which counts the backends it builds. */
@interface NFKPresetTestBuilder : NSObject <NFKModelBackendBuilding>
@property (atomic, assign) NSInteger builds;
@end

@implementation NFKPresetTestBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	self.builds += 1;
	NFKPresetTestBackend *backend = [[NFKPresetTestBackend alloc] init];
	return [request applyPropertyOptionsToBackend:backend error:error] ? backend : nil;
}

@end

@interface NFKServerPresetTests : XCTestCase
@property (nonatomic) NFKPresetTestBuilder *builder;
@property (nonatomic) NFKModelCatalog *catalog;
@property (nonatomic) NFKInferenceServer *server;
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServerPresetTests

- (void)setUp
{
	[super setUp];
	self.savedRetryAttempts = NFKRemoteTransport.retryAttempts;
	NFKRemoteTransport.retryAttempts = 0;
	self.builder = [[NFKPresetTestBuilder alloc] init];
	[NFKModelCatalog registerBackendBuilder:self.builder forEngine:NFKPresetTestEngine];
	self.catalog = [[NFKModelCatalog alloc] init];
	// A 64 GiB working set, so no build or estimate depends on the memory of the machine that runs the tests.
	self.catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
	NSError *error = nil;
	XCTAssertTrue([self.catalog addDescriptorsFromManifestData:[self manifest] error:&error], @"%@", error);
	self.server = [[NFKInferenceServer alloc] init];
	self.server.port = 0;
	self.server.loopbackOnly = YES;
	self.server.advertisesService = NO;
	self.server.modelCatalog = self.catalog;
}

- (void)tearDown
{
	[self.server stop];
	[NFKModelCatalog registerBackendBuilder:nil forEngine:NFKPresetTestEngine];
	NFKRemoteTransport.retryAttempts = self.savedRetryAttempts;
	[super tearDown];
}

#pragma mark Helpers

// A model with a load option the build sets as a property, a context length the memory figure reads, two
// request options, and three presets: a suggested one that sets request defaults, a suggested one that
// sets a load option, and one that is not suggested.
- (NSData *)manifest
{
	NSString *json = @"{ \"schemaVersion\": 1, \"kind\": \"inferkit.model-catalog\", \"models\": [ {"
		"\"id\": \"preset.model\", \"family\": \"test\", \"engine\": \"preset-test-engine\", \"tasks\": [\"text-generation\"],"
		"\"options\": ["
		"{ \"key\": \"mode\", \"kind\": \"text\", \"stage\": \"load\", \"default\": \"plain\", \"property\": \"mode\" },"
		"{ \"key\": \"contextLength\", \"kind\": \"integer\", \"stage\": \"load\", \"default\": 4096, \"minimum\": 256, \"maximum\": 16384 },"
		"{ \"key\": \"steps\", \"kind\": \"integer\", \"default\": 20, \"minimum\": 1 },"
		"{ \"key\": \"temperature\", \"kind\": \"number\", \"default\": 0.7, \"minimum\": 0, \"maximum\": 2 } ],"
		"\"memoryModel\": { \"contextLength\": \"contextLength\" },"
		"\"presets\": ["
		"{ \"id\": \"fast\", \"suggested\": true, \"requestDefaults\": { \"steps\": 4 } },"
		"{ \"id\": \"draft\", \"suggested\": true, \"loadOptions\": { \"mode\": \"draft\" } },"
		"{ \"id\": \"slow\", \"requestDefaults\": { \"steps\": 50 } } ],"
		"\"variants\": [{ \"id\": \"small\", \"isDefault\": true, \"precision\": \"bfloat16\","
		"  \"memoryFacts\": { \"stages\": [{ \"id\": \"decoder\", \"bytesByPrecision\": { \"checkpoint\": 1000000 } }],"
		"    \"keyValueBytesPerToken\": 100, \"maximumContextLength\": 16384, \"basis\": \"measured\" } }] } ] }";
	return [json dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)start
{
	NSError *error = nil;
	XCTAssertTrue([self.server startWithError:&error], @"%@", error);
}

/*! The base model "chat", hosted from the test catalog. */
- (void)hostChat
{
	NSError *error = nil;
	XCTAssertTrue([self.server addModelWithIdentifier:@"preset.model" options:nil catalog:nil forModelName:@"chat" error:&error],
				  @"%@", error);
}

- (NFKModelPreset *)presetNamed:(NSString *)identifier
					loadOptions:(nullable NSDictionary *)loadOptions
				requestDefaults:(nullable NSDictionary *)requestDefaults
{
	return [[NFKModelPreset alloc] initWithIdentifier:identifier title:nil help:nil baseModelIdentifier:nil
										  loadOptions:loadOptions requestDefaults:requestDefaults suggested:NO extensions:nil];
}

- (NFKRemoteInferKitBackend *)clientFor:(nullable NSString *)modelName
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:self.server.localBaseURL];
	client.modelName = modelName;
	return client;
}

- (nullable NFKInferenceResult *)run:(NSString *)modelName
						  parameters:(nullable NSDictionary *)parameters
							   error:(NSError **)error
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" } parameters:parameters
														   outputModality:NFKModalityText];
	return [[self clientFor:modelName] runInferenceForRequest:request error:error];
}

/*! The JSON reply of a request to path under the base URL, with its HTTP response. */
- (nullable id)JSONAt:(NSString *)path
			   method:(NSString *)method
				 body:(nullable NSDictionary *)body
			 response:(NSHTTPURLResponse * _Nullable * _Nullable)response
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[self.server.localBaseURL URLByAppendingPathComponent:path]];
	request.HTTPMethod = method;
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:response error:NULL];
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

- (NFKServerStatus *)status
{
	NSError *error = nil;
	NFKServerStatus *status = [NFKServerStatus fetchFromBaseURL:self.server.localBaseURL apiKey:nil error:&error];
	XCTAssertNotNil(status, @"%@", error);
	return status;
}

#pragma mark Hosting through the build entry

- (void)testCatalogModelBuildsOnFirstRequestWithItsOptions
{
	NSError *error = nil;
	NSDictionary *options = @{ @"mode": @"precise", @"steps": @"9" };
	XCTAssertTrue([self.server addModelWithIdentifier:@"preset.model" options:options catalog:nil forModelName:@"chat"
												error:&error], @"%@", error);
	[self start];
	XCTAssertEqual([self.server residencyForModelName:@"chat"], NFKServerModelResidencyNotLoaded);
	XCTAssertEqual(self.builder.builds, 0);

	NFKInferenceResult *result = [self run:@"chat" parameters:nil error:&error];
	XCTAssertEqualObjects([result outputForKey:@"mode"], @"precise", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:@"steps"], @9, @"a request option given at hosting is a request default");
	XCTAssertEqual(self.builder.builds, 1);

	NFKServerModelStatus *chat = [[self status] modelNamed:@"chat"];
	XCTAssertEqualObjects(chat.catalogModelIdentifier, @"preset.model");
	XCTAssertEqualObjects(chat.loadOptions[@"mode"], @"precise");
	XCTAssertEqualObjects(chat.loadOptions[@"contextLength"], @4096);
	XCTAssertNil(chat.baseModelName);
}

- (void)testAddingAModelChecksTheCatalogAndTheOptions
{
	NSError *error = nil;
	XCTAssertFalse([self.server addModelWithIdentifier:@"ghost" options:nil catalog:nil forModelName:@"ghost" error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnknownModel);

	error = nil;
	XCTAssertFalse([self.server addModelWithIdentifier:@"preset.model" options:@{ @"contextLength": @"lots" }
											   catalog:nil forModelName:@"chat" error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
	XCTAssertEqualObjects(self.server.modelNames, @[]);

	error = nil;
	XCTAssertTrue([self.server addModelWithIdentifier:@"preset.model@draft" options:nil catalog:self.catalog
										 forModelName:@"drafting" error:&error], @"%@", error);
	[self start];
	NFKInferenceResult *result = [self run:@"drafting" parameters:nil error:&error];
	XCTAssertEqualObjects([result outputForKey:@"mode"], @"draft", @"a catalog preset's load options reach the build: %@", error);
}

#pragma mark Presets

- (void)testRequestDefaultsPresetIsAnAliasSharingTheBaseBackend
{
	[self hostChat];
	NSError *error = nil;
	XCTAssertTrue([self.server addPreset:[self presetNamed:@"careful" loadOptions:nil requestDefaults:@{ @"steps": @50 }]
						forBaseModelName:@"chat" error:&error], @"%@", error);
	XCTAssertEqualObjects(self.server.modelNames, @[ @"chat" ]);
	XCTAssertEqualObjects(self.server.aliasNames, @[ @"chat@careful" ]);
	XCTAssertEqualObjects([self.server modelNameForAlias:@"chat@careful"], @"chat");
	[self start];

	NFKInferenceResult *careful = [self run:@"chat@careful" parameters:nil error:&error];
	XCTAssertEqualObjects([careful outputForKey:@"steps"], @50, @"%@", error);
	NFKInferenceResult *plain = [self run:@"chat" parameters:nil error:&error];
	XCTAssertEqualObjects([plain outputForKey:@"steps"], @0, @"the base runs without the preset's defaults: %@", error);
	NFKInferenceResult *overridden = [self run:@"chat@careful" parameters:@{ @"steps": @3 } error:&error];
	XCTAssertEqualObjects([overridden outputForKey:@"steps"], @3, @"a request's own parameter wins: %@", error);
	XCTAssertEqual(self.builder.builds, 1, @"the preset runs on the base's backend");
}

- (void)testLoadOptionPresetIsASeparateModel
{
	[self hostChat];
	NSError *error = nil;
	XCTAssertTrue([self.server addPreset:[self presetNamed:@"turbo" loadOptions:@{ @"mode": @"turbo" }
											requestDefaults:@{ @"steps": @2 }]
						forBaseModelName:@"chat" error:&error], @"%@", error);
	XCTAssertEqualObjects(self.server.modelNames, (@[ @"chat", @"chat@turbo" ]));
	XCTAssertEqualObjects(self.server.aliasNames, @[]);
	[self start];

	NFKInferenceResult *turbo = [self run:@"chat@turbo" parameters:nil error:&error];
	XCTAssertEqualObjects([turbo outputForKey:@"mode"], @"turbo", @"%@", error);
	XCTAssertEqualObjects([turbo outputForKey:@"steps"], @2);
	XCTAssertEqual([self.server residencyForModelName:@"chat"], NFKServerModelResidencyNotLoaded,
				   @"the preset's model loads on its own");
	NFKInferenceResult *plain = [self run:@"chat" parameters:nil error:&error];
	XCTAssertEqualObjects([plain outputForKey:@"mode"], @"plain", @"%@", error);
	XCTAssertEqual(self.builder.builds, 2);

	NFKServerModelStatus *status = [[self status] modelNamed:@"chat@turbo"];
	XCTAssertEqualObjects(status.baseModelName, @"chat");
	XCTAssertEqualObjects(status.presetIdentifier, @"turbo");
	XCTAssertEqualObjects(status.loadOptions[@"mode"], @"turbo");
	XCTAssertEqualObjects(status.catalogModelIdentifier, @"preset.model");
}

- (void)testPresetsKeyedByTheirLoadOptionsShareAModel
{
	[self hostChat];
	NSError *error = nil;
	NSDictionary *defaultsWrittenOut = @{ @"mode": @"plain", @"contextLength": @"4096" };
	XCTAssertTrue([self.server addPreset:[self presetNamed:@"same" loadOptions:defaultsWrittenOut requestDefaults:nil]
						forBaseModelName:@"chat" error:&error], @"%@", error);
	XCTAssertTrue([self.server addPreset:[self presetNamed:@"turbo" loadOptions:@{ @"mode": @"turbo" } requestDefaults:@{ @"steps": @2 }]
						forBaseModelName:@"chat" error:&error], @"%@", error);
	XCTAssertTrue([self.server addPreset:[self presetNamed:@"cool" loadOptions:@{ @"mode": @"turbo" }
											requestDefaults:@{ @"temperature": @0.1 }]
						forBaseModelName:@"chat" error:&error], @"%@", error);
	XCTAssertEqualObjects(self.server.modelNames, (@[ @"chat", @"chat@turbo" ]));
	XCTAssertEqualObjects([self.server modelNameForAlias:@"chat@same"], @"chat", @"the defaults, written out, are the base's options");
	XCTAssertEqualObjects([self.server modelNameForAlias:@"chat@cool"], @"chat@turbo");
	[self start];

	NFKInferenceResult *cool = [self run:@"chat@cool" parameters:nil error:&error];
	XCTAssertEqualObjects([cool outputForKey:@"mode"], @"turbo", @"%@", error);
	XCTAssertEqualObjects([cool outputForKey:@"temperature"], @0.1);
	XCTAssertEqualObjects([cool outputForKey:@"steps"], @0, @"another preset's defaults do not apply");
	XCTAssertEqual(self.builder.builds, 1);

	NFKServerStatus *status = [self status];
	XCTAssertEqualObjects([status modelNamed:@"chat@turbo"].aliasNames, @[ @"chat@cool" ]);
	XCTAssertEqualObjects(status.aliases[@"chat@cool"], @"chat@turbo");
}

- (void)testSuggestedRequestDefaultsPresetResolvesWithoutBeingAdded
{
	[self hostChat];
	[self start];
	NSError *error = nil;
	NFKRemoteInferKitBackend *client = [self clientFor:@"chat@fast"];
	NFKInferenceResult *fast = [client runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]
														error:&error];
	XCTAssertEqualObjects([fast outputForKey:@"steps"], @4, @"%@", error);
	XCTAssertEqualObjects(client.lastReportedLoad.baseModelName, @"chat");
	XCTAssertEqualObjects(client.lastReportedLoad.presetIdentifier, @"fast");

	error = nil;
	XCTAssertNil([self run:@"chat@draft" parameters:nil error:&error], @"a preset that sets load options is added explicitly");
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
	error = nil;
	XCTAssertNil([self run:@"chat@slow" parameters:nil error:&error], @"a preset that is not suggested is added explicitly");
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
	XCTAssertTrue([self.server addPreset:[self.catalog descriptorWithIdentifier:@"preset.model"].presets.lastObject
						forBaseModelName:@"chat" error:&error], @"%@", error);
	NFKInferenceResult *slow = [self run:@"chat@slow" parameters:nil error:&error];
	XCTAssertEqualObjects([slow outputForKey:@"steps"], @50, @"%@", error);
}

- (void)testANameWithAnAtSignStaysWholeUnlessItNamesAPreset
{
	[self.server addBackend:[[NFKPresetTestBackend alloc] init] forModelName:@"claude-3"];
	NFKPresetTestBackend *dated = [[NFKPresetTestBackend alloc] init];
	dated.mode = @"dated";
	[self.server addBackend:dated forModelName:@"claude-3@20240620"];
	[self hostChat];
	[self start];
	NSError *error = nil;
	NFKInferenceResult *result = [self run:@"claude-3@20240620" parameters:nil error:&error];
	XCTAssertEqualObjects([result outputForKey:@"mode"], @"dated", @"%@", error);

	[self.server removeBackendForModelName:@"claude-3@20240620"];
	error = nil;
	XCTAssertNil([self run:@"claude-3@20240620" parameters:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
	XCTAssertTrue([error.localizedDescription containsString:@"\"claude-3@20240620\""], @"%@", error);
	error = nil;
	XCTAssertNil([self run:@"chat@20240620" parameters:nil error:&error], @"a catalog model without that preset");
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
}

- (void)testPresetsAreRefusedWhereTheyCannotApply
{
	NSError *error = nil;
	XCTAssertFalse([self.server addPreset:[self presetNamed:@"fast" loadOptions:nil requestDefaults:nil]
						 forBaseModelName:@"absent" error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);

	[self.server addBackend:[[NFKPresetTestBackend alloc] init] forModelName:@"plain"];
	error = nil;
	XCTAssertFalse([self.server addPreset:[self presetNamed:@"turbo" loadOptions:@{ @"mode": @"turbo" } requestDefaults:nil]
						 forBaseModelName:@"plain" error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorInvalidConfiguration);

	[self hostChat];
	error = nil;
	XCTAssertFalse([self.server addPreset:[self presetNamed:@"bad" loadOptions:nil requestDefaults:@{ @"steps": @0 }]
						 forBaseModelName:@"chat" error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);

	[self.server addBackend:[[NFKPresetTestBackend alloc] init] forModelName:@"chat@taken"];
	error = nil;
	XCTAssertFalse([self.server addPreset:[self presetNamed:@"taken" loadOptions:nil requestDefaults:nil]
						 forBaseModelName:@"chat" error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorNameConflict);
}

#pragma mark Aliases

- (void)testAnAliasRunsItsModelWithItsDefaults
{
	[self hostChat];
	NSError *error = nil;
	XCTAssertTrue([self.server addAlias:@"talk" forModelName:@"chat" requestDefaults:@{ @"steps": @7 } error:&error], @"%@", error);
	[self start];
	NFKInferenceResult *result = [self run:@"talk" parameters:nil error:&error];
	XCTAssertEqualObjects([result outputForKey:@"steps"], @7, @"%@", error);
	XCTAssertEqualObjects([[self status] modelNamed:@"chat"].aliasNames, @[ @"talk" ]);
	XCTAssertEqualObjects([self status].aliases, @{ @"talk": @"chat" });

	error = nil;
	XCTAssertFalse([self.server addAlias:@"chat" forModelName:@"chat" requestDefaults:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorNameConflict);
	error = nil;
	XCTAssertFalse([self.server addAlias:@"other" forModelName:@"absent" requestDefaults:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);

	[self.server removeAlias:@"talk"];
	error = nil;
	XCTAssertNil([self run:@"talk" parameters:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
}

- (void)testAnInProcessJobResolvesAnAliasAndAPresetWithTheirDefaults
{
	[self hostChat];
	NSError *error = nil;
	XCTAssertTrue([self.server addAlias:@"talk" forModelName:@"chat" requestDefaults:@{ @"steps": @7 } error:&error], @"%@", error);
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }];
	for (NSArray *expected in @[ @[ @"talk", @7 ], @[ @"chat@fast", @4 ], @[ @"chat", @0 ] ]) {
		NFKInferenceJob *job = [self.server submitInferenceJobForRequest:request modelName:expected[0]];
		XCTestExpectation *finished = [self expectationWithDescription:expected[0]];
		job.completionHandler = ^(NFKInferenceJob *ended) {
			[finished fulfill];
		};
		[self waitForExpectations:@[ finished ] timeout:10];
		XCTAssertEqualObjects([job.result outputForKey:@"steps"], expected[1], @"%@: %@", expected[0], job.error);
	}
}

#pragma mark Reporting

- (void)testRepliesNameTheModelAsAskedAndReportItsBaseAndPreset
{
	[self hostChat];
	NSError *error = nil;
	XCTAssertTrue([self.server addAlias:@"talk" forModelName:@"chat" requestDefaults:nil error:&error], @"%@", error);
	[self start];

	NSHTTPURLResponse *response = nil;
	NSDictionary *reply = [self JSONAt:@"chat/completions" method:@"POST"
								  body:@{ @"model": @"chat@fast", @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] }
							  response:&response];
	XCTAssertEqualObjects(reply[@"model"], @"chat@fast");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Base-Model"], @"chat");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Preset"], @"fast");

	reply = [self JSONAt:@"chat/completions" method:@"POST"
					body:@{ @"model": @"talk", @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] } response:&response];
	XCTAssertEqualObjects(reply[@"model"], @"talk");
	XCTAssertEqualObjects([response valueForHTTPHeaderField:@"X-InferKit-Base-Model"], @"chat");
	XCTAssertNil([response valueForHTTPHeaderField:@"X-InferKit-Preset"]);

	NSDictionary *native = [self JSONAt:@"inferkit/run" method:@"POST"
								   body:@{ @"model": @"chat@fast", @"request": @{ @"inputs": @{} } } response:&response];
	XCTAssertEqualObjects(native[@"model"], @"chat@fast", @"%@", native);
	XCTAssertEqualObjects(native[@"base"], @"chat");
	XCTAssertEqualObjects(native[@"preset"], @"fast");
	XCTAssertEqualObjects(native[@"loadOptions"][@"mode"], @"plain");

	NSArray<NSString *> *listed = [[self JSONAt:@"models" method:@"GET" body:nil response:NULL][@"data"] valueForKey:@"id"];
	XCTAssertEqualObjects(listed, (@[ @"chat", @"talk" ]));
	NSDictionary *entry = [self JSONAt:@"models/chat@fast" method:@"GET" body:nil response:NULL];
	XCTAssertEqualObjects(entry[@"id"], @"chat@fast");
	XCTAssertEqualObjects(entry[@"inferkit"][@"base"], @"chat");
	XCTAssertEqualObjects(entry[@"inferkit"][@"preset"], @"fast");
	XCTAssertEqualObjects(entry[@"inferkit"][@"requestDefaults"], @{ @"steps": @4 });
	XCTAssertEqualObjects(entry[@"inferkit"][@"catalogModel"], @"preset.model");
}

#pragma mark MCP

- (void)testModelToolNamesMapToModelNamesAndACollisionIsRefused
{
	[self.server addBackend:[[NFKPresetTestBackend alloc] init] forModelName:@"a@b"];
	NFKPresetTestBackend *other = [[NFKPresetTestBackend alloc] init];
	other.mode = @"other";
	[self.server addBackend:other forModelName:@"a__b"];
	NSError *error = nil;
	NFKBackendTool *tool = [self.server addToolForModelName:@"a@b" outputModality:NFKModalityText purpose:nil error:&error];
	XCTAssertEqualObjects(tool.name, @"a__b", @"%@", error);
	XCTAssertEqualObjects([self.server modelNameForToolName:@"a__b"], @"a@b");

	error = nil;
	XCTAssertNil([self.server addToolForModelName:@"a__b" outputModality:NFKModalityText purpose:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorNameConflict);
	XCTAssertEqualObjects([self.server modelNameForToolName:@"a__b"], @"a@b", @"the first model keeps the name");
	XCTAssertEqual(self.server.tools.count, 1u);
	XCTAssertNotNil([self.server addToolForModelName:@"a@b" outputModality:NFKModalityText purpose:@"a test needs it" error:&error],
					@"the same model replaces its own tool: %@", error);
	XCTAssertEqual(self.server.tools.count, 1u);

	error = nil;
	XCTAssertNil([self.server addToolForModelName:@"absent" outputModality:NFKModalityText purpose:nil error:&error]);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);

	NFKToolResult *answer = [self.server.tools.firstObject invokeWithArguments:@{ @"prompt": @"hi" }];
	XCTAssertEqualObjects(answer.text, @"ok", @"the tool runs in the model's queue");
	XCTAssertEqual([[self.server currentStatus] modelNamed:@"a@b"].completed, 1);
	[self.server removeToolNamed:@"a__b"];
	XCTAssertNil([self.server modelNameForToolName:@"a__b"]);
}

#pragma mark Catalog and estimate routes

- (void)testCatalogRouteServesTheManifest
{
	[self start];
	NSError *error = nil;
	NFKModelCatalog *served = [[self clientFor:nil] fetchModelCatalogWithError:&error];
	NFKModelDescriptor *model = [served descriptorWithIdentifier:@"preset.model"];
	XCTAssertNotNil(model, @"%@", error);
	XCTAssertEqual(model.presets.count, 3u);
	XCTAssertTrue([model presetWithIdentifier:@"fast"].isSuggested);
}

- (void)testEstimateRouteAnswersTheEstimateAndConstraints
{
	[self.server addModelWithIdentifier:@"preset.model" options:@{ @"contextLength": @8192 } catalog:nil forModelName:@"chat" error:NULL];
	[self start];
	NFKRemoteInferKitBackend *client = [self clientFor:nil];
	NSError *error = nil;
	NFKModelMemoryConstraints *constraints = nil;
	NFKModelMemoryEstimate *estimate = [client fetchMemoryEstimateForModelNamed:@"preset.model" options:@{ @"contextLength": @2048 }
																   constraints:&constraints error:&error];
	// 1,000,000 bytes of weights and 100 bytes a position of cache at 16 bits.
	XCTAssertEqual(estimate.peakByteCount, 1000000 + 100 * 2048, @"%@", error);
	XCTAssertNotNil(constraints);
	XCTAssertTrue(constraints.anyCombinationFits, @"a megabyte fits any machine");

	estimate = [client fetchMemoryEstimateForModelNamed:@"chat" options:nil constraints:NULL error:&error];
	XCTAssertEqual(estimate.peakByteCount, 1000000 + 100 * 8192, @"a hosted model starts from its own options: %@", error);

	error = nil;
	XCTAssertNil([client fetchMemoryEstimateForModelNamed:@"preset.model" options:@{ @"contextLength": @"lots" }
											 constraints:NULL error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
	error = nil;
	XCTAssertNil([client fetchMemoryEstimateForModelNamed:@"ghost" options:nil constraints:NULL error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
}

- (void)testEstimateRouteFollowsReportsHostDetails
{
	[self start];
	NSDictionary *body = @{ @"model": @"preset.model", @"options": @{ @"contextLength": @2048 } };
	NSDictionary *described = [self JSONAt:@"inferkit/estimate" method:@"POST" body:body response:NULL];
	XCTAssertNotNil(described[@"budget"], @"%@", described);
	XCTAssertNotNil(described[@"constraints"]);
	XCTAssertNotNil(described[@"estimate"][@"budget"]);
	NFKRemoteInferKitBackend *client = [self clientFor:nil];
	NSError *error = nil;
	NFKModelMemoryConstraints *constraints = nil;
	XCTAssertNotNil([client fetchMemoryEstimateForModelNamed:@"preset.model" options:nil constraints:&constraints error:&error],
					@"%@", error);
	XCTAssertNotNil(constraints);

	self.server.reportsHostDetails = NO;
	NSDictionary *bare = [self JSONAt:@"inferkit/estimate" method:@"POST" body:body response:NULL];
	XCTAssertNil(bare[@"budget"], @"%@", bare);
	XCTAssertNil(bare[@"constraints"], @"constraint reasons quote the budget");
	XCTAssertNil(bare[@"estimate"][@"budget"]);
	XCTAssertNil(bare[@"estimate"][@"reason"]);
	XCTAssertEqualObjects(bare[@"estimate"][@"peakBytes"], @(1000000 + 100 * 2048));
	XCTAssertNotNil(bare[@"estimate"][@"components"]);
	XCTAssertEqualObjects(bare[@"estimate"][@"options"][@"contextLength"], @2048);

	NFKModelMemoryEstimate *estimate = [client fetchMemoryEstimateForModelNamed:@"preset.model"
																	   options:@{ @"contextLength": @2048 }
																   constraints:&constraints error:&error];
	XCTAssertEqual(estimate.peakByteCount, 1000000 + 100 * 2048, @"%@", error);
	XCTAssertNil(estimate.budget);
	XCTAssertNil(constraints, @"a reply without constraints reads as nil constraints");
}

- (void)testStatusCarriesTheMachinesHardwareSnapshot
{
	[self start];
	NFKHardwareProfile *local = NFKHardwareProfile.currentProfile;
	NFKHardwareProfile *served = [self status].host.hardwareProfile;
	XCTAssertEqualObjects(served.chipName, local.chipName);
	XCTAssertEqual(served.physicalMemory, local.physicalMemory);
	XCTAssertEqual(served.recommendedWorkingSetSize, local.recommendedWorkingSetSize);

	NSError *error = nil;
	NFKHardwareProfile *fetched = [[self clientFor:nil] fetchServerHardwareProfileWithError:&error];
	XCTAssertEqual(fetched.isAppleSilicon, local.isAppleSilicon, @"%@", error);

	self.server.reportsHostDetails = NO;
	error = nil;
	XCTAssertNil([[self clientFor:nil] fetchServerHardwareProfileWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

@end
