//
//  NFKGeminiVoiceLibraryTests.m
//  InferKitTests
//
//  The Gemini voice library through a stub transport: listing with filters and pages, lookup,
//  design, replication, deletion, and the provider factory. A live listing is gated on a key.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGeminiVoiceLibrary.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubGeminiVoiceLibrary : NFKGeminiVoiceLibrary
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *stagedBodies;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubGeminiVoiceLibrary
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requests = [NSMutableArray array];
		_stagedBodies = [NSMutableArray array];
		_stagedStatusCode = 200;
	}
	return self;
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSString *body = self.stagedBodies.count > 1 ? self.stagedBodies.firstObject : self.stagedBodies.lastObject;
	if (self.stagedBodies.count > 1) {
		[self.stagedBodies removeObjectAtIndex:0];
	}
	return [body ?: @"" dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKGeminiVoiceLibraryTests : XCTestCase
@property (nonatomic, strong) NFKStubGeminiVoiceLibrary *library;
@end

@implementation NFKGeminiVoiceLibraryTests

- (void)setUp
{
	[super setUp];
	self.library = [[NFKStubGeminiVoiceLibrary alloc] init];
	self.library.apiKey = @"g-key";
}

- (NSDictionary *)bodyOfRequest:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.library.requests[index].HTTPBody options:0 error:NULL];
}

- (NSData *)wavBytes
{
	const unsigned char header[] = { 'R', 'I', 'F', 'F', 0x24, 0x00, 0x00, 0x00, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' };
	return [NSData dataWithBytes:header length:sizeof(header)];
}

#pragma mark Listing

- (void)testAListingSendsItsFiltersAndReadsEveryPage
{
	[self.library.stagedBodies addObject:@"{\"voices\":[{\"id\":\"voice_abc\",\"type\":\"prompted\",\"display_name\":\"Astronomer\","
		"\"language_code\":\"en-GB\"}],\"next_page_token\":\"p2\"}"];
	[self.library.stagedBodies addObject:@"{\"voices\":[{\"id\":\"Kore\",\"type\":\"prebuilt\",\"display_name\":\"Kore\",\"gender\":\"female\"}]}"];
	NSError *error = nil;
	NSArray<NFKRemoteVoice *> *voices = [self.library voicesMatchingFilters:@{ @"language_code": @[ @"en-US", @"en-GB" ], @"type": @"prompted" }
																	  error:&error];
	XCTAssertNotNil(voices, @"%@", error);
	XCTAssertEqual(self.library.requests.count, 2);

	NSURLRequest *first = self.library.requests[0];
	XCTAssertEqualObjects(first.HTTPMethod, @"GET");
	XCTAssertEqualObjects(first.allHTTPHeaderFields[@"x-goog-api-key"], @"g-key");
	XCTAssertEqualObjects(first.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/voices?language_code=en-US&language_code=en-GB&type=prompted");
	XCTAssertTrue([self.library.requests[1].URL.query hasSuffix:@"&page_token=p2"]);

	XCTAssertEqualObjects([voices valueForKey:@"identifier"], (@[ @"voice_abc", @"Kore" ]));
	XCTAssertEqualObjects(voices[0].name, @"Astronomer");
	XCTAssertEqualObjects(voices[0].languages, @[ @"en-GB" ]);
	XCTAssertEqualObjects(voices[1].raw[@"gender"], @"female");
	XCTAssertNil(voices[0].sampleAudioData, @"a listing carries no preview");
}

- (void)testAPageTokenGivenTwiceEndsTheListing
{
	[self.library.stagedBodies addObject:@"{\"voices\":[{\"id\":\"Puck\"}],\"next_page_token\":\"same\"}"];
	NSArray<NFKRemoteVoice *> *voices = [self.library voicesMatchingFilters:nil error:NULL];
	XCTAssertEqual(self.library.requests.count, 2);
	XCTAssertEqual(voices.count, 2);
	XCTAssertNil(self.library.requests[0].URL.query, @"no filters, no query");
}

- (void)testALookupStripsTheCollectionPrefixAndDecodesThePreview
{
	NSString *preview = [[self wavBytes] base64EncodedStringWithOptions:0];
	[self.library.stagedBodies addObject:[NSString stringWithFormat:@"{\"id\":\"voice_abc\",\"type\":\"prompted\","
		"\"sample_audio\":{\"mime_type\":\"audio/wav\",\"data\":\"%@\"}}", preview]];
	NFKRemoteVoice *voice = [self.library voiceWithIdentifier:@"voices/voice_abc" error:NULL];
	XCTAssertEqualObjects(self.library.requests[0].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/voices/voice_abc");
	XCTAssertEqualObjects(voice.identifier, @"voice_abc");
	XCTAssertEqualObjects(voice.sampleAudioData, [self wavBytes]);
}

#pragma mark Designing

- (void)testADesignedVoiceIsStoredWithItsPromptNameModelAndAttributes
{
	self.library.modelName = @"gemini-3.8-flash-tts";
	[self.library.stagedBodies addObject:@"{\"id\":\"voice_new\",\"type\":\"prompted\",\"display_name\":\"Astronomer\","
		"\"sample_audio\":{\"mime_type\":\"audio/wav\",\"data\":\"UklGRg==\"}}"];
	NSError *error = nil;
	NFKRemoteVoice *voice = [self.library designVoiceWithDescription:@"A warm astronomer with a gentle British accent."
														 displayName:@"Astronomer"
														  attributes:@{ @"gender": @"male", @"language_code": @"en-GB", @"type": @"prebuilt" }
															   error:&error];
	XCTAssertEqualObjects(voice.identifier, @"voice_new", @"%@", error);
	XCTAssertEqualObjects(voice.sampleAudioData, [@"RIFF" dataUsingEncoding:NSASCIIStringEncoding]);

	NSURLRequest *request = self.library.requests[0];
	XCTAssertEqualObjects(request.HTTPMethod, @"POST");
	XCTAssertEqualObjects(request.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/voices");
	XCTAssertEqualObjects(request.allHTTPHeaderFields[@"Content-Type"], @"application/json");
	XCTAssertEqualObjects([self bodyOfRequest:0], (@{ @"store": @YES, @"voice": @{
		@"type": @"prompted", @"model": @"gemini-3.8-flash-tts", @"display_name": @"Astronomer",
		@"gender": @"male", @"language_code": @"en-GB",
		@"prompted": @{ @"input": @"A warm astronomer with a gentle British accent." } } }),
		@"a reserved attribute does not change the type");
}

- (void)testADesignWithoutADescriptionSendsNothing
{
	NSError *error = nil;
	XCTAssertNil([self.library designVoiceWithDescription:@"" displayName:nil attributes:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqual(self.library.requests.count, 0);
}

#pragma mark Replicating

- (void)testAStoredReplicaSendsBothClipsAndItsName
{
	[self.library.stagedBodies addObject:@"{\"id\":\"voice_clone\",\"type\":\"replicated\",\"display_name\":\"Me\"}"];
	const unsigned char tag[] = { 'I', 'D', '3', 0x04, 0x00 };
	NSData *id3 = [NSData dataWithBytes:tag length:sizeof(tag)];
	NFKRemoteVoice *voice = [self.library replicateVoiceFromAudio:[self wavBytes] consentAudio:id3 displayName:@"Me" stores:YES error:NULL];
	XCTAssertEqualObjects(voice.identifier, @"voice_clone");
	NSDictionary *body = [self bodyOfRequest:0];
	XCTAssertEqualObjects(body[@"store"], @YES);
	XCTAssertEqualObjects(body[@"voice"][@"type"], @"replicated");
	XCTAssertEqualObjects(body[@"voice"][@"display_name"], @"Me");
	XCTAssertEqualObjects(body[@"voice"][@"replicated"][@"source_audio"],
						  (@{ @"mime_type": @"audio/wav", @"data": [[self wavBytes] base64EncodedStringWithOptions:0] }));
	XCTAssertEqualObjects(body[@"voice"][@"replicated"][@"consent_audio"][@"mime_type"], @"audio/mpeg");
	XCTAssertNil(body[@"voice"][@"model"], @"no model lets the service choose");
}

- (void)testAnUnstoredReplicaIsNamedByItsKeyAndSendsNoName
{
	[self.library.stagedBodies addObject:@"{\"key\":\"voicekey_xyz\",\"type\":\"replicated\"}"];
	NSURL *file = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
		[NSString stringWithFormat:@"NFKGeminiVoiceLibraryTests-%@.flac", NSUUID.UUID.UUIDString]]];
	XCTAssertTrue([[@"fLaC" dataUsingEncoding:NSASCIIStringEncoding] writeToURL:file atomically:YES]);
	NFKRemoteVoice *voice = [self.library replicateVoiceFromAudio:file consentAudio:[self wavBytes] displayName:@"ignored" stores:NO error:NULL];
	[NSFileManager.defaultManager removeItemAtURL:file error:NULL];

	XCTAssertEqualObjects(voice.identifier, @"voicekey_xyz");
	NSDictionary *body = [self bodyOfRequest:0];
	XCTAssertEqualObjects(body[@"store"], @NO);
	XCTAssertNil(body[@"voice"][@"display_name"]);
	XCTAssertEqualObjects(body[@"voice"][@"replicated"][@"source_audio"][@"mime_type"], @"audio/flac");
}

- (void)testAudioOfAnUnknownFormatIsRefusedBeforeAnythingIsSent
{
	NSError *error = nil;
	NSData *unknown = [@"OggS...." dataUsingEncoding:NSASCIIStringEncoding];
	XCTAssertNil([self.library replicateVoiceFromAudio:unknown consentAudio:[self wavBytes] displayName:nil stores:YES error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);

	XCTAssertNil([self.library replicateVoiceFromAudio:[NSURL fileURLWithPath:@"/tmp/clip.ogg"] consentAudio:[self wavBytes]
										   displayName:nil stores:YES error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);

	XCTAssertNil([self.library replicateVoiceFromAudio:@"not audio" consentAudio:[self wavBytes] displayName:nil stores:YES error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqual(self.library.requests.count, 0);
}

#pragma mark Deleting and errors

- (void)testADeletionSucceedsOnAnEmptyReplyAndFailsOnAnError
{
	[self.library.stagedBodies addObject:@"{}"];
	NSError *error = nil;
	XCTAssertTrue([self.library deleteVoiceWithIdentifier:@"voice_abc" error:&error], @"%@", error);
	XCTAssertEqualObjects(self.library.requests[0].HTTPMethod, @"DELETE");
	XCTAssertEqualObjects(self.library.requests[0].URL.lastPathComponent, @"voice_abc");

	self.library.stagedStatusCode = 404;
	self.library.stagedBodies[0] = @"{\"error\":{\"code\":404,\"message\":\"Voice not found.\",\"status\":\"NOT_FOUND\"}}";
	XCTAssertFalse([self.library deleteVoiceWithIdentifier:@"voice_gone" error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"Voice not found"]);

	self.library.stagedStatusCode = 429;
	self.library.stagedBodies[0] = @"{\"error\":{\"code\":429,\"status\":\"RESOURCE_EXHAUSTED\"}}";
	XCTAssertNil([self.library designVoiceWithDescription:@"a voice" displayName:nil attributes:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);
}

- (void)testAReplyWithoutAnIdOrKeyIsAnError
{
	[self.library.stagedBodies addObject:@"{\"type\":\"prompted\"}"];
	NSError *error = nil;
	XCTAssertNil([self.library designVoiceWithDescription:@"a voice" displayName:nil attributes:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider

- (void)testTheFactoryDerivesTheVoicesURLFromTheGeminiPresetOnly
{
	NFKGeminiVoiceLibrary *library = [NFKGeminiVoiceLibrary voiceLibraryForProvider:NFKRemoteProvider.googleGemini apiKey:@"k"];
	XCTAssertEqualObjects(library.endpointURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/voices");
	XCTAssertEqualObjects(library.apiKey, @"k");

	NFKRemoteProvider *proxied = [NFKRemoteProvider.googleGemini providerWithBaseURL:[NSURL URLWithString:@"http://gateway.test/gemini/v1beta/openai"]];
	XCTAssertEqualObjects([NFKGeminiVoiceLibrary voiceLibraryForProvider:proxied apiKey:nil].endpointURL.absoluteString,
						  @"http://gateway.test/gemini/v1beta/voices");
	XCTAssertNil([NFKGeminiVoiceLibrary voiceLibraryForProvider:NFKRemoteProvider.openAI apiKey:@"k"]);
}

#pragma mark Live

- (void)testALiveListingNamesThePrebuiltVoices
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_GEMINI_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_GEMINI_API_KEY to exercise the live path");
	}
	NFKGeminiVoiceLibrary *library = [NFKGeminiVoiceLibrary voiceLibraryForProvider:NFKRemoteProvider.googleGemini apiKey:key];
	NSError *error = nil;
	NSArray<NFKRemoteVoice *> *voices = [library voicesMatchingFilters:@{ @"type": @"prebuilt", @"search": @"Kore" } error:&error];
	XCTAssertNotNil(voices, @"%@", error);
	XCTAssertTrue([[voices valueForKey:@"identifier"] containsObject:@"Kore"]);
}

@end
