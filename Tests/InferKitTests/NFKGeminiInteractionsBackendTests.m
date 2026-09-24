//
//  NFKGeminiInteractionsBackendTests.m
//  InferKitTests
//
//  Gemini's Interactions API through stubbed transports: the input steps and content blocks, each
//  output modality's response_format and reply, transcription, background polling, and the stream.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGeminiInteractionsBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubInteractionsBackend : NFKGeminiInteractionsBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSData *> *stagedReplies;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubInteractionsBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSData *reply = self.stagedReplies.firstObject ?: [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	if (self.stagedReplies.count > 1) {
		[self.stagedReplies removeObjectAtIndex:0];
	}
	return reply;
}
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
}
- (void)stage:(NSString *)reply
{
	if (self.stagedReplies == nil) {
		self.stagedReplies = [NSMutableArray array];
	}
	[self.stagedReplies addObject:[reply dataUsingEncoding:NSUTF8StringEncoding]];
}
- (NSDictionary *)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}
@end

@interface NFKGeminiInteractionsBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubInteractionsBackend *backend;
@property (nonatomic, strong) NSURL *directory;
@property (nonatomic, assign) CGImageRef square;
@property (nonatomic, copy) NSString *squarePNG;
@end

@implementation NFKGeminiInteractionsBackendTests

- (void)setUp
{
	[super setUp];
	self.directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	self.backend = [NFKStubInteractionsBackend backendWithAPIKey:@"AIza" modelName:@"gemini-3.8-flash"];
	self.backend.pollInterval = 0;
	self.backend.outputDirectoryURL = self.directory;
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	self.square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	self.squarePNG = [[NFKImageCoding PNGDataForImage:(__bridge id)self.square] base64EncodedStringWithOptions:0];
}

- (void)tearDown
{
	CGImageRelease(self.square);
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
	[super tearDown];
}

- (void)testTextAndAnImageBecomeAUserStepAndTheReplyTextThoughtsAndUsageComeBack
{
	[self.backend stage:@"{\"id\":\"v1_a\",\"status\":\"completed\",\"steps\":["
	 "{\"type\":\"thought\",\"content\":[{\"type\":\"text\",\"text\":\"Looking at the square.\"}]},"
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"A blue square.\"}]}],"
	 "\"usage\":{\"total_input_tokens\":12,\"total_output_tokens\":4,\"total_thought_tokens\":9}}"];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Answer briefly." }, @{ @"role": @"user", @"content": @"What is this?" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages, NFKInputImage: (__bridge id)self.square }
															   parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortModerate, @"temperature": @0.2 }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSURLRequest *sent = self.backend.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"system_instruction"], @"Answer briefly.");
	NSDictionary *step = body[@"input"][0];
	XCTAssertEqualObjects(step[@"type"], @"user_input");
	XCTAssertEqualObjects(step[@"content"][0], (@{ @"type": @"text", @"text": @"What is this?" }));
	XCTAssertEqualObjects(step[@"content"][1][@"type"], @"image");
	XCTAssertEqualObjects(body[@"generation_config"][@"thinking_level"], @"medium");
	XCTAssertEqualObjects(body[@"generation_config"][@"temperature"], @0.2);
	XCTAssertEqualObjects(result.text, @"A blue square.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Looking at the square.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageReasoningTokens], @9);
	XCTAssertEqualObjects([result outputForKey:NFKOutputResponseIdentifier], @"v1_a");
}

- (void)testAnImageRequestAsksForAnImageAndDecodesIt
{
	[self.backend stage:[NSString stringWithFormat:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
						 "{\"type\":\"image\",\"mime_type\":\"image/png\",\"data\":\"%@\"}]}]}", self.squarePNG]];
	self.backend.modelName = @"gemini-3.1-flash-image";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a square" }
															   parameters:@{ NFKParameterAspectRatio: @"16:9", NFKParameterResolution: @"2K" }
														   outputModality:NFKModalityImage];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *format = [self.backend bodyAt:0][@"response_format"];
	XCTAssertEqualObjects(format, (@{ @"type": @"image", @"aspect_ratio": @"16:9", @"image_size": @"2K" }));
	XCTAssertEqualObjects([self.backend bodyAt:0][@"input"], @"a square", @"a prompt alone goes as a string");
	XCTAssertNotNil([result outputForKey:NFKOutputImage]);
}

- (void)testSpeechNamesItsVoiceAndThePCMReplyIsWrittenAsWAV
{
	NSData *pcm = [NSData dataWithBytes:"\x01\x00\x02\x00" length:4];
	[self.backend stage:[NSString stringWithFormat:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
						 "{\"type\":\"audio\",\"mime_type\":\"audio/L16;codec=pcm;rate=24000\",\"data\":\"%@\"}]}]}",
						 [pcm base64EncodedStringWithOptions:0]]];
	self.backend.modelName = @"gemini-3.1-flash-tts-preview";
	self.backend.voice = @"Kore";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hi" } parameters:@{} outputModality:NFKModalityAudio];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"response_format"], (@{ @"type": @"audio" }));
	XCTAssertEqualObjects(body[@"generation_config"][@"speech_config"], (@[ @{ @"voice": @"Kore" } ]));
	NFKAudioAsset *audio = [result outputForKey:NFKOutputAudio];
	NSData *wav = [NSData dataWithContentsOfURL:audio.fileURL];
	XCTAssertEqualObjects(audio.fileURL.pathExtension, @"wav");
	XCTAssertEqualObjects([wav subdataWithRange:NSMakeRange(0, 4)], [@"RIFF" dataUsingEncoding:NSASCIIStringEncoding]);
	XCTAssertEqualObjects([wav subdataWithRange:NSMakeRange(44, 4)], pcm);
}

- (void)testATranscriptionAsksForVerbatimSpeakersAndReadsTheWords
{
	[self.backend stage:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Hi. Hello.\","
	 "\"annotations\":[{\"type\":\"word_info\",\"text\":\"Hi.\",\"speaker\":\"spk_1\",\"start_offset\":\"0.1s\",\"end_offset\":\"0.4s\"},"
	 "{\"type\":\"word_info\",\"text\":\"Hello.\",\"speaker\":\"spk_2\",\"start_offset\":\"1.0s\",\"end_offset\":\"1.6s\"}]}]}]}"];
	self.backend.modelName = @"gemini-3.5-transcribe";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:@{ NFKParameterSpeakerDiarization: @YES, NFKParameterWordTimestamps: @YES,
																			 NFKParameterVocabulary: @[ @"InferKit" ], NFKParameterSourceLanguage: @"en" }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *configuration = [self.backend bodyAt:0][@"generation_config"][@"transcription_config"];
	XCTAssertEqualObjects(configuration[@"language_codes"], (@[ @"en" ]));
	XCTAssertEqualObjects(configuration[@"custom_vocabulary"], (@[ @"InferKit" ]));
	XCTAssertEqualObjects(configuration[@"mode"], (@{ @"type": @"verbatim", @"diarization_mode": @"speaker", @"timestamp_granularities": @[ @"word" ] }));
	NSArray<NFKAudioSegment *> *words = [result outputForKey:NFKOutputWords];
	XCTAssertEqual(words.count, 2);
	XCTAssertEqualWithAccuracy(words[1].startSeconds, 1.0, 1e-9);
	NSArray<NFKAudioSegment *> *turns = [result outputForKey:NFKOutputSegments];
	XCTAssertEqual(turns.count, 2);
	XCTAssertEqualObjects(turns[1].speaker, @"spk_2");
}

- (void)testABackgroundVideoIsPolledAndItsURIFetchedWithTheKey
{
	self.backend.runsInBackground = YES;
	self.backend.modelName = @"gemini-omni-1.1-flash";
	[self.backend stage:@"{\"id\":\"v1_vid\",\"status\":\"in_progress\"}"];
	[self.backend stage:@"{\"id\":\"v1_vid\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
	 "{\"type\":\"video\",\"mime_type\":\"video/mp4\",\"uri\":\"https://generativelanguage.googleapis.com/v1beta/files/abc:download\"}]}]}"];
	[self.backend stage:@"ftypisom"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a wave" }
															   parameters:@{ NFKParameterAspectRatio: @"16:9", NFKParameterResolution: @"1080p" }
														   outputModality:NFKModalityVideo];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects([self.backend bodyAt:0][@"background"], @YES);
	XCTAssertEqualObjects([self.backend bodyAt:0][@"response_format"], (@{ @"type": @"video", @"aspect_ratio": @"16:9", @"resolution": @"1080p" }));
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions/v1_vid");
	XCTAssertEqualObjects([self.backend.requests[2] valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
	NFKVideoAsset *clip = [result outputForKey:NFKOutputVideo];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:clip.fileURL], [@"ftypisom" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testAFunctionCallAndAContinuationAreCarried
{
	[self.backend stage:@"{\"id\":\"v1_f\",\"status\":\"requires_action\",\"steps\":[{\"type\":\"function_call\",\"id\":\"c1\",\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"weather?" }
															   parameters:@{ NFKParameterTools: @[ @{ @"name": @"get_weather", @"parameters": @{ @"type": @"object" } },
																								 @{ @"type": @"google_search" } ],
																			 NFKParameterPreviousResponseIdentifier: @"v1_prev" }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"previous_interaction_id"], @"v1_prev");
	XCTAssertEqualObjects(body[@"tools"][1], (@{ @"type": @"google_search" }));
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Oslo" }));
}

- (void)testTheStreamGrowsTheTextAndThenReadsTheFinishedInteraction
{
	self.backend.stagedLines = @[ @"data: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"v1_s\",\"status\":\"in_progress\"}}",
								  @"data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"Hel\"}}",
								  @"data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"lo\"}}",
								  @"data: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"v1_s\",\"status\":\"completed\"}}" ];
	[self.backend stage:@"{\"id\":\"v1_s\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}]}"];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertTrue([self.backend.requests.firstObject.URL.absoluteString hasSuffix:@"interactions?alt=sse"]);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions/v1_s");
}

@end
