//
//  NFKRemoteSkillStoreTests.m
//  InferKitTests
//
//  Anthropic's and Perplexity's Skills APIs through a stub transport: uploads from a zip and from a
//  folder, the records, the pages, versions, the revision guard, deletes, downloads, and the provider
//  factory. Live tests against the hosted endpoints are gated on keys.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteSkillStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubSkillStore : NFKRemoteSkillStore
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *stagedBodies;
@property (nonatomic, strong) NSMutableArray<NSNumber *> *stagedStatusCodes;
@end

@implementation NFKStubSkillStore

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	NSInteger status = self.stagedStatusCodes.firstObject.integerValue ?: 200;
	if (self.stagedStatusCodes.count > 0) {
		[self.stagedStatusCodes removeObjectAtIndex:0];
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:status HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSString *body = self.stagedBodies.firstObject ?: @"";
	if (self.stagedBodies.count > 0) {
		[self.stagedBodies removeObjectAtIndex:0];
	}
	return [body dataUsingEncoding:NSUTF8StringEncoding];
}

- (NSString *)bodyTextAt:(NSUInteger)index
{
	return [[NSString alloc] initWithData:self.requests[index].HTTPBody encoding:NSISOLatin1StringEncoding];
}

@end

@interface NFKRemoteSkillStoreTests : XCTestCase
@property (nonatomic, strong, nullable) NSURL *scratch;
@end

@implementation NFKRemoteSkillStoreTests

- (void)tearDown
{
	if (self.scratch != nil) {
		[NSFileManager.defaultManager removeItemAtURL:self.scratch error:NULL];
	}
	[super tearDown];
}

- (NFKStubSkillStore *)storeWithStyle:(NFKRemoteSkillStoreAPIStyle)style bodies:(NSArray<NSString *> *)bodies
{
	NSString *base = style == NFKRemoteSkillStoreAPIStyleAnthropic ? @"https://api.anthropic.com/v1/skills" : @"https://api.perplexity.ai/v1/skills";
	NFKStubSkillStore *store = [NFKStubSkillStore skillStoreWithEndpointURL:[NSURL URLWithString:base] apiStyle:style];
	store.apiKey = @"key";
	store.stagedBodies = [bodies mutableCopy];
	store.stagedStatusCodes = [NSMutableArray array];
	return store;
}

/*! A skill folder in a unique scratch directory: SKILL.md and a script beside it. */
- (NSURL *)skillFolderNamed:(NSString *)name
{
	self.scratch = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:[@"NFKSkillStore-" stringByAppendingString:NSUUID.UUID.UUIDString]];
	NSURL *folder = [self.scratch URLByAppendingPathComponent:name];
	[NSFileManager.defaultManager createDirectoryAtURL:[folder URLByAppendingPathComponent:@"scripts"] withIntermediateDirectories:YES attributes:nil error:NULL];
	[@"---\nname: report-writer\ndescription: Writes reports.\n---\nWrite the report.\n" writeToURL:[folder URLByAppendingPathComponent:@"SKILL.md"]
																							atomically:YES encoding:NSUTF8StringEncoding error:NULL];
	[@"print('hi')\n" writeToURL:[folder URLByAppendingPathComponent:@"scripts/run.py"] atomically:YES encoding:NSUTF8StringEncoding error:NULL];
	return folder;
}

- (NSString *)anthropicSkill
{
	return @"{\"type\":\"skill\",\"id\":\"skill_01Abc\",\"display_name\":\"Report writer\",\"latest_version_id\":\"skver_2\","
		"\"source\":{\"type\":\"custom\"},\"created_at\":\"2026-10-20T10:00:00Z\",\"updated_at\":\"2026-10-22T10:00:00.250Z\"}";
}

- (NSString *)perplexitySkill
{
	return @"{\"skill_id\":\"skill_9\",\"name\":\"report-writer\",\"description\":\"Writes reports.\",\"revision\":\"rev_1\","
		"\"latest_revision\":\"rev_2\",\"created_at\":\"2026-10-20T10:00:00Z\",\"updated_at\":\"2026-10-22T10:00:00Z\"}";
}

#pragma mark Records

- (void)testEachServicesRecordBecomesASkill
{
	NFKRemoteSkill *claude = [NFKRemoteSkill skillWithRecord:[NSJSONSerialization JSONObjectWithData:[self.anthropicSkill dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL]
													apiStyle:NFKRemoteSkillStoreAPIStyleAnthropic];
	XCTAssertEqualObjects(claude.identifier, @"skill_01Abc");
	XCTAssertEqualObjects(claude.name, @"Report writer");
	XCTAssertEqualObjects(claude.version, @"skver_2");
	XCTAssertEqualObjects(claude.latestVersion, @"skver_2");
	XCTAssertEqualObjects(claude.source, @"custom");
	XCTAssertEqualObjects(claude.createdAt, [NSDate dateWithTimeIntervalSince1970:1792490400]);
	XCTAssertNotNil(claude.updatedAt, @"a fractional timestamp reads");
	XCTAssertEqualObjects(claude.skillReference, (@{ @"id": @"skill_01Abc", @"source": @"custom" }));

	NFKRemoteSkill *perplexity = [NFKRemoteSkill skillWithRecord:[NSJSONSerialization JSONObjectWithData:[self.perplexitySkill dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL]
														apiStyle:NFKRemoteSkillStoreAPIStylePerplexity];
	XCTAssertEqualObjects(perplexity.identifier, @"skill_9");
	XCTAssertEqualObjects(perplexity.name, @"report-writer");
	XCTAssertEqualObjects(perplexity.skillDescription, @"Writes reports.");
	XCTAssertEqualObjects(perplexity.version, @"rev_1");
	XCTAssertEqualObjects(perplexity.latestVersion, @"rev_2");
	XCTAssertNil([NFKRemoteSkill skillWithRecord:@{ @"name": @"x" } apiStyle:NFKRemoteSkillStoreAPIStylePerplexity]);
}

#pragma mark Uploads

- (void)testAnthropicUploadsAFolderAsFilesNamedByTheirPaths
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[ self.anthropicSkill ]];
	NSError *error = nil;
	NFKRemoteSkill *skill = [store createSkillWithDirectoryAtURL:[self skillFolderNamed:@"report_writer"] name:@"Report writer" error:&error];
	XCTAssertEqualObjects(skill.identifier, @"skill_01Abc", @"%@", error);
	NSURLRequest *request = store.requests[0];
	XCTAssertEqualObjects(request.HTTPMethod, @"POST");
	XCTAssertEqualObjects(request.URL.absoluteString, @"https://api.anthropic.com/v1/skills");
	XCTAssertEqualObjects([request valueForHTTPHeaderField:@"x-api-key"], @"key");
	XCTAssertNotNil([request valueForHTTPHeaderField:@"anthropic-version"]);
	XCTAssertTrue([[request valueForHTTPHeaderField:@"Content-Type"] hasPrefix:@"multipart/form-data; boundary="]);
	NSString *body = [store bodyTextAt:0];
	XCTAssertTrue([body containsString:@"name=\"display_name\"\r\n\r\nReport writer\r\n"]);
	XCTAssertTrue([body containsString:@"name=\"files[]\"; filename=\"report_writer/SKILL.md\""]);
	XCTAssertTrue([body containsString:@"name=\"files[]\"; filename=\"report_writer/scripts/run.py\""]);
	XCTAssertTrue([body containsString:@"description: Writes reports."]);
}

// Perplexity takes one zip under file, and a folder is zipped with the folder at its root.
- (void)testPerplexityUploadsAFolderZippedUnderFile
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStylePerplexity bodies:@[ self.perplexitySkill ]];
	NSError *error = nil;
	NFKRemoteSkill *skill = [store createSkillWithDirectoryAtURL:[self skillFolderNamed:@"report_writer"] name:@"ignored" error:&error];
	XCTAssertEqualObjects(skill.identifier, @"skill_9", @"%@", error);
	NSURLRequest *request = store.requests[0];
	XCTAssertEqualObjects([request valueForHTTPHeaderField:@"Authorization"], @"Bearer key");
	NSString *body = [store bodyTextAt:0];
	XCTAssertTrue([body containsString:@"name=\"file\"; filename=\"skill.zip\"\r\nContent-Type: application/zip\r\n\r\nPK"]);
	XCTAssertTrue([body containsString:@"report_writer/SKILL.md"], @"the zip names the folder at its root");
	XCTAssertFalse([body containsString:@"display_name"]);
}

- (void)testAZipGoesAsIsAndAFolderWithoutASkillFileIsRefused
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[ self.anthropicSkill ]];
	NSData *zip = [@"PK\x03\x04-bundle" dataUsingEncoding:NSISOLatin1StringEncoding];
	XCTAssertNotNil([store createSkillWithBundle:zip name:nil error:NULL]);
	XCTAssertTrue([[store bodyTextAt:0] containsString:@"name=\"files[]\"; filename=\"skill.zip\"\r\nContent-Type: application/zip\r\n\r\nPK"]);

	NSURL *folder = [self skillFolderNamed:@"empty"];
	[NSFileManager.defaultManager removeItemAtURL:[folder URLByAppendingPathComponent:@"SKILL.md"] error:NULL];
	NSError *error = nil;
	XCTAssertNil([store createSkillWithDirectoryAtURL:folder name:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqual(store.requests.count, 1u, @"nothing is sent without a SKILL.md");
}

#pragma mark Lists

- (void)testAnthropicsListFollowsNextPage
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[
		[NSString stringWithFormat:@"{\"data\":[%@],\"has_more\":true,\"next_page\":\"page_2\"}", self.anthropicSkill],
		@"{\"data\":[{\"id\":\"xlsx\",\"display_name\":\"xlsx\",\"source\":{\"type\":\"anthropic\"}}],\"has_more\":false,\"next_page\":null}" ]];
	NSError *error = nil;
	NSArray<NFKRemoteSkill *> *skills = [store skillsWithError:&error];
	XCTAssertEqualObjects([skills valueForKey:@"identifier"], (@[ @"skill_01Abc", @"xlsx" ]), @"%@", error);
	XCTAssertEqualObjects(skills[1].source, @"anthropic");
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.anthropic.com/v1/skills?limit=1000");
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://api.anthropic.com/v1/skills?limit=1000&page=page_2");
}

- (void)testPerplexitysListAndRevisionsFollowNextPageToken
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStylePerplexity bodies:@[
		[NSString stringWithFormat:@"{\"skills\":[%@],\"next_page_token\":\"t2\"}", self.perplexitySkill],
		@"{\"skills\":[{\"skill_id\":\"skill_8\",\"name\":\"older\",\"revision\":\"rev_a\"}]}",
		@"{\"revisions\":[{\"revision\":\"rev_2\",\"name\":\"report-writer\",\"created_at\":\"2026-10-22T10:00:00Z\"},{\"revision\":\"rev_1\"}]}" ]];
	NSArray<NFKRemoteSkill *> *skills = [store skillsWithError:NULL];
	XCTAssertEqualObjects([skills valueForKey:@"identifier"], (@[ @"skill_9", @"skill_8" ]));
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://api.perplexity.ai/v1/skills?limit=200&page_token=t2");

	NSArray<NFKRemoteSkillVersion *> *versions = [store versionsOfSkillWithIdentifier:@"skill_9" error:NULL];
	XCTAssertEqualObjects([versions valueForKey:@"identifier"], (@[ @"rev_2", @"rev_1" ]));
	XCTAssertEqualObjects(versions[0].name, @"report-writer");
	XCTAssertNotNil(versions[0].createdAt);
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://api.perplexity.ai/v1/skills/skill_9/revisions?limit=200");
}

- (void)testOneSkillAndAPerplexityRevisionAreRead
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStylePerplexity bodies:@[ self.perplexitySkill ]];
	XCTAssertEqualObjects([store skillWithIdentifier:@"skill_9" version:@"rev_1" error:NULL].version, @"rev_1");
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/skills/skill_9?revision=rev_1");

	NFKStubSkillStore *claude = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[ self.anthropicSkill,
		@"{\"data\":[{\"type\":\"skill_version\",\"id\":\"skver_2\",\"name\":\"report-writer\",\"description\":\"Writes reports.\",\"skill_id\":\"skill_01Abc\"}],\"next_page\":null}" ]];
	XCTAssertNotNil([claude skillWithIdentifier:@"skill_01Abc" version:nil error:NULL]);
	XCTAssertEqualObjects(claude.requests[0].URL.absoluteString, @"https://api.anthropic.com/v1/skills/skill_01Abc");
	NSArray<NFKRemoteSkillVersion *> *versions = [claude versionsOfSkillWithIdentifier:@"skill_01Abc" error:NULL];
	XCTAssertEqualObjects(versions.firstObject.identifier, @"skver_2");
	XCTAssertEqualObjects(versions.firstObject.skillDescription, @"Writes reports.");
	XCTAssertEqualObjects(claude.requests[1].URL.absoluteString, @"https://api.anthropic.com/v1/skills/skill_01Abc/versions?limit=1000");
}

#pragma mark Versions and deletes

- (void)testAnthropicAddsAVersionAndReadsTheSkillBack
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[ @"{\"type\":\"skill_version\",\"id\":\"skver_3\"}", self.anthropicSkill ]];
	NSError *error = nil;
	NFKRemoteSkill *skill = [store addVersionToSkillWithIdentifier:@"skill_01Abc" directoryAtURL:[self skillFolderNamed:@"report_writer"]
												   expectedVersion:nil error:&error];
	XCTAssertEqualObjects(skill.identifier, @"skill_01Abc", @"%@", error);
	XCTAssertEqualObjects(store.requests[0].HTTPMethod, @"POST");
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.anthropic.com/v1/skills/skill_01Abc/versions");
	XCTAssertEqualObjects(store.requests[1].HTTPMethod, @"GET");
}

// A Perplexity write names the revision it expects; without one the store reads the latest first.
- (void)testPerplexityGuardsAnUpdateAndADeleteWithTheExpectedRevision
{
	NFKStubSkillStore *store = [self storeWithStyle:NFKRemoteSkillStoreAPIStylePerplexity bodies:@[ self.perplexitySkill ]];
	NSData *zip = [@"PK\x03\x04" dataUsingEncoding:NSISOLatin1StringEncoding];
	XCTAssertNotNil([store addVersionToSkillWithIdentifier:@"skill_9" bundle:zip expectedVersion:@"rev_2" error:NULL]);
	XCTAssertEqualObjects(store.requests[0].HTTPMethod, @"PUT");
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/skills/skill_9?expected_revision=rev_2");

	store.stagedBodies = [@[ self.perplexitySkill, @"" ] mutableCopy];
	store.stagedStatusCodes = [@[ @200, @204 ] mutableCopy];
	NSError *error = nil;
	XCTAssertTrue([store deleteSkillWithIdentifier:@"skill_9" expectedVersion:nil error:&error], @"%@", error);
	XCTAssertEqualObjects(store.requests[1].HTTPMethod, @"GET");
	XCTAssertEqualObjects(store.requests[2].HTTPMethod, @"DELETE");
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://api.perplexity.ai/v1/skills/skill_9?expected_revision=rev_2");

	store.stagedBodies = [@[ @"{\"error\":{\"message\":\"revision conflict\",\"type\":\"conflict\",\"code\":409}}" ] mutableCopy];
	store.stagedStatusCodes = [@[ @409 ] mutableCopy];
	XCTAssertFalse([store deleteSkillWithIdentifier:@"skill_9" expectedVersion:@"rev_1" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testEachServiceRefusesTheCallItDoesNotServe
{
	NFKStubSkillStore *claude = [self storeWithStyle:NFKRemoteSkillStoreAPIStyleAnthropic bodies:@[ @"{\"type\":\"skill_version_deleted\",\"id\":\"skver_1\"}" ]];
	NSError *error = nil;
	XCTAssertTrue([claude deleteVersion:@"skver_1" ofSkillWithIdentifier:@"skill_01Abc" error:&error], @"%@", error);
	XCTAssertEqualObjects(claude.requests[0].URL.absoluteString, @"https://api.anthropic.com/v1/skills/skill_01Abc/versions/skver_1");
	XCTAssertNil([claude downloadURLForSkillWithIdentifier:@"skill_01Abc" version:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);

	NFKStubSkillStore *perplexity = [self storeWithStyle:NFKRemoteSkillStoreAPIStylePerplexity bodies:@[
		@"{\"download_url\":\"https://downloads.example.test/skill.zip?sig=1\",\"revision\":\"rev_1\",\"name\":\"report-writer\"}" ]];
	XCTAssertEqualObjects([perplexity downloadURLForSkillWithIdentifier:@"skill_9" version:@"rev_1" error:NULL].absoluteString,
						  @"https://downloads.example.test/skill.zip?sig=1");
	XCTAssertEqualObjects(perplexity.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/skills/skill_9/download?revision=rev_1");
	XCTAssertFalse([perplexity deleteVersion:@"rev_1" ofSkillWithIdentifier:@"skill_9" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

#pragma mark The provider

- (void)testTheFactoryServesAnthropicAndPerplexityAndDeclinesTheRest
{
	NFKRemoteSkillStore *claude = [NFKRemoteSkillStore skillStoreForProvider:NFKRemoteProvider.anthropic apiKey:@"k"];
	XCTAssertEqualObjects(claude.endpointURL.absoluteString, @"https://api.anthropic.com/v1/skills");
	XCTAssertEqual(claude.apiStyle, NFKRemoteSkillStoreAPIStyleAnthropic);
	NFKRemoteSkillStore *perplexity = [NFKRemoteSkillStore skillStoreForProvider:NFKRemoteProvider.perplexity apiKey:@"k"];
	XCTAssertEqualObjects(perplexity.endpointURL.absoluteString, @"https://api.perplexity.ai/v1/skills");
	XCTAssertEqual(perplexity.apiStyle, NFKRemoteSkillStoreAPIStylePerplexity);
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.openAI, NFKRemoteProvider.googleGemini, NFKRemoteProvider.ollama ]) {
		XCTAssertNil([NFKRemoteSkillStore skillStoreForProvider:provider apiKey:@"k"], @"%@ keeps no skills", provider.identifier);
	}
}

#pragma mark Live

// Lists against the hosted stores when a key is present; listing writes nothing. Each path answers
// 401 without a key and a path beside it 404, so the paths are verified to exist.
- (void)testALiveStoreListsItsSkills
{
	NSDictionary<NSString *, NSString *> *environment = NSProcessInfo.processInfo.environment;
	NSString *anthropicKey = environment[@"INFERKIT_ANTHROPIC_API_KEY"];
	NSString *perplexityKey = environment[@"INFERKIT_PERPLEXITY_API_KEY"];
	if (anthropicKey.length == 0 && perplexityKey.length == 0) {
		XCTSkip("set INFERKIT_ANTHROPIC_API_KEY or INFERKIT_PERPLEXITY_API_KEY to exercise the live path");
	}
	if (anthropicKey.length > 0) {
		NSError *error = nil;
		XCTAssertNotNil([[NFKRemoteSkillStore skillStoreForProvider:NFKRemoteProvider.anthropic apiKey:anthropicKey] skillsWithError:&error], @"%@", error);
	}
	if (perplexityKey.length > 0) {
		NSError *error = nil;
		XCTAssertNotNil([[NFKRemoteSkillStore skillStoreForProvider:NFKRemoteProvider.perplexity apiKey:perplexityKey] skillsWithError:&error], @"%@", error);
	}
}

@end
