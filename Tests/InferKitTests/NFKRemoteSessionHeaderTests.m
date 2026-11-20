//
//  NFKRemoteSessionHeaderTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <CommonCrypto/CommonDigest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKSessionHeader = @"x-opencode-session";

// Each stub records what the backend sends and answers with a transport failure, which is enough to
// read the request's headers.
#define NFK_SESSION_RECORDING_STUB(Name, Base) \
@interface Name : Base \
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *sent; \
@end \
@implementation Name \
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError \
{ \
	if (self.sent == nil) { \
		self.sent = [NSMutableArray array]; \
	} \
	[self.sent addObject:request]; \
	if (outError != NULL) { \
		*outError = [NSError errorWithDomain:@"NFKRemoteSessionHeaderTests" code:1 userInfo:nil]; \
	} \
	return nil; \
} \
@end

NFK_SESSION_RECORDING_STUB(NFKSessionChatStub, NFKRemoteBackend)
NFK_SESSION_RECORDING_STUB(NFKSessionResponsesStub, NFKRemoteResponsesBackend)
NFK_SESSION_RECORDING_STUB(NFKSessionMessagesStub, NFKAnthropicBackend)
NFK_SESSION_RECORDING_STUB(NFKSessionGenerateContentStub, NFKGeminiGenerateContentBackend)
NFK_SESSION_RECORDING_STUB(NFKSessionSystemOneStub, NFKTypeSafeBackend)

@interface NFKRemoteSessionHeaderTests : XCTestCase
@end

@implementation NFKRemoteSessionHeaderTests

#pragma mark Helpers

- (id)stubOfClass:(Class)stubClass
{
	id backend = [stubClass backendWithEndpointURL:[NSURL URLWithString:@"https://opencode.ai/inference/go/v1/route"]];
	[backend setApiKey:@"k"];
	[backend setModelName:@"m"];
	[backend setSessionHeaderName:NFKSessionHeader];
	return backend;
}

- (NSArray *)everyStub
{
	return @[ [self stubOfClass:NFKSessionChatStub.class], [self stubOfClass:NFKSessionResponsesStub.class],
			  [self stubOfClass:NFKSessionMessagesStub.class], [self stubOfClass:NFKSessionGenerateContentStub.class],
			  [self stubOfClass:NFKSessionSystemOneStub.class] ];
}

/*! The header on the last POST the backend sent. */
- (nullable NSString *)sessionSentBy:(id)backend forRequest:(NFKInferenceRequest *)request
{
	[backend runInferenceForRequest:request error:NULL];
	for (NSURLRequest *sent in [[backend sent] reverseObjectEnumerator]) {
		if ([sent.HTTPMethod isEqualToString:@"POST"]) {
			return [sent valueForHTTPHeaderField:NFKSessionHeader];
		}
	}
	return nil;
}

- (NFKInferenceRequest *)requestWithInputs:(NSDictionary *)inputs parameters:(NSDictionary *)parameters
{
	NSMutableDictionary *all = [inputs mutableCopy];
	all[NFKInputQuestions] = @{ @"urgent": [NFKDecisionQuestion noulQuestionWithInstructions:@"Urgent?"] };
	return [NFKInferenceRequest requestWithInputs:all parameters:parameters outputModality:NFKModalityText];
}

- (NSString *)expectedSessionForKey:(NSString *)key
{
	NSData *data = [[@"key:" stringByAppendingString:key] dataUsingEncoding:NSUTF8StringEncoding];
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
	NSMutableString *expected = [NSMutableString stringWithString:@"ses_"];
	for (NSUInteger index = 0; index < 16; index++) {
		[expected appendFormat:@"%02x", digest[index]];
	}
	return expected;
}

#pragma mark The provider

- (void)testTheOpenCodePresetsNameTheSessionHeader
{
	XCTAssertEqualObjects(NFKRemoteProvider.openCodeZen.sessionHeaderName, NFKSessionHeader);
	XCTAssertEqualObjects(NFKRemoteProvider.openCodeGo.sessionHeaderName, NFKSessionHeader);
	XCTAssertNil(NFKRemoteProvider.openAI.sessionHeaderName);
	XCTAssertNil(NFKRemoteProvider.anthropic.sessionHeaderName);

	NFKRemoteProvider *moved = [NFKRemoteProvider.openCodeGo providerWithBaseURL:[NSURL URLWithString:@"https://example.com/go/v1"]];
	XCTAssertEqualObjects(moved.sessionHeaderName, NFKSessionHeader);
	NFKRemoteProvider *routed = [NFKRemoteProvider.openCodeGo providerWithAPIStyle:NFKRemoteAPIStyleAnthropicMessages forModelName:@"next"];
	XCTAssertEqualObjects(routed.sessionHeaderName, NFKSessionHeader);
}

- (void)testEveryRouteTheOpenCodeFactoriesBuildCarriesTheHeader
{
	NFKRemoteProvider *go = NFKRemoteProvider.openCodeGo;
	NSArray<NSArray *> *rows = @[
		@[ go, @"kimi-k3", NFKRemoteBackend.class ],
		@[ go, @"gpt-6-luna", NFKRemoteResponsesBackend.class ],
		@[ go, @"claude-haiku-5-5", NFKAnthropicBackend.class ],
		@[ go, @"gemini-4-argon", NFKGeminiGenerateContentBackend.class ],
		@[ NFKRemoteProvider.openCodeZen, @"jev-1.13", NFKTypeSafeBackend.class ],
	];
	for (NSArray *row in rows) {
		id backend = [NFKRemoteProvider backendForProvider:row[0] apiKey:@"k" modelName:row[1]];
		XCTAssertTrue([backend isKindOfClass:row[2]], @"%@ got %@", row[1], [backend class]);
		XCTAssertEqualObjects([backend sessionHeaderName], NFKSessionHeader, @"%@", row[1]);
	}
	XCTAssertEqualObjects([NFKRemoteResponsesBackend backendForProvider:go apiKey:@"k" modelName:@"gpt-6-luna"].sessionHeaderName, NFKSessionHeader);
	XCTAssertEqualObjects([NFKAnthropicBackend backendForProvider:go apiKey:@"k" modelName:@"claude-haiku-5-5"].sessionHeaderName, NFKSessionHeader);
	XCTAssertEqualObjects([NFKGeminiGenerateContentBackend backendForProvider:go apiKey:@"k" modelName:@"gemini-4-argon"].sessionHeaderName, NFKSessionHeader);

	id openAI = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"gpt-6-luna"];
	XCTAssertNil([openAI sessionHeaderName]);
}

#pragma mark The header

- (void)testEveryBackendSendsADigestOfTheConversationKey
{
	NFKInferenceRequest *request = [self requestWithInputs:@{ NFKInputPrompt: @"Hello" }
												parameters:@{ NFKParameterConversationKey: @"chat-1" }];
	NSString *expected = [self expectedSessionForKey:@"chat-1"];
	for (id backend in [self everyStub]) {
		XCTAssertEqualObjects([self sessionSentBy:backend forRequest:request], expected, @"%@", [backend class]);
	}
}

- (void)testAConversationKeepsItsSessionAsItGrows
{
	NSDictionary *opening = @{ @"role": @"user", @"content": @"Plan a trip." };
	NFKInferenceRequest *first = [self requestWithInputs:@{ NFKInputMessages: @[ opening ] } parameters:@{}];
	NFKInferenceRequest *later = [self requestWithInputs:@{ NFKInputMessages: @[ opening,
		@{ @"role": @"assistant", @"content": @"Where to?" }, @{ @"role": @"user", @"content": @"Kyiv." } ] } parameters:@{}];
	NFKInferenceRequest *other = [self requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"Hi." } ] }
											  parameters:@{}];
	for (id backend in [self everyStub]) {
		NSString *session = [self sessionSentBy:backend forRequest:first];
		XCTAssertTrue([session hasPrefix:@"ses_"], @"%@", [backend class]);
		XCTAssertEqual(session.length, 36u);
		XCTAssertEqualObjects([self sessionSentBy:backend forRequest:later], session, @"%@", [backend class]);
		XCTAssertNotEqualObjects([self sessionSentBy:backend forRequest:other], session, @"%@", [backend class]);
	}
}

- (void)testARequestThatNamesNoConversationCarriesTheBackendsOwnSession
{
	NFKInferenceRequest *request = [self requestWithInputs:@{ NFKInputPrompt: @"Hello" } parameters:@{}];
	NFKSessionChatStub *one = [self stubOfClass:NFKSessionChatStub.class];
	NFKSessionChatStub *two = [self stubOfClass:NFKSessionChatStub.class];
	NSString *session = [self sessionSentBy:one forRequest:request];
	XCTAssertTrue([session hasPrefix:@"ses_"]);
	XCTAssertEqualObjects([self sessionSentBy:one forRequest:request], session);
	XCTAssertNotEqualObjects([self sessionSentBy:two forRequest:request], session);
}

- (void)testABackendWithoutAHeaderNameSendsNoSession
{
	NFKInferenceRequest *request = [self requestWithInputs:@{ NFKInputPrompt: @"Hello" }
												parameters:@{ NFKParameterConversationKey: @"chat-1" }];
	for (id backend in [self everyStub]) {
		[backend setSessionHeaderName:nil];
		[backend runInferenceForRequest:request error:NULL];
		for (NSURLRequest *sent in [backend sent]) {
			XCTAssertNil([sent valueForHTTPHeaderField:NFKSessionHeader], @"%@", [backend class]);
		}
	}
}

@end
