//
//  NFKRemoteTransportTests.m
//  InferKitTests
//
//  The shared transport against an in-process NSURLProtocol, so the retry schedule and the line
//  splitter are exercised through a real NSURLSession without a network: a rate limit that clears,
//  one that does not, a Retry-After too long to wait for, and a body delivered in ragged chunks.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! Answers http://stub.test/ from a scripted list of (status, headers, body chunks); each request
	consumes the next script entry. */
@interface NFKStubURLProtocol : NSURLProtocol
@end

static NSMutableArray<NSDictionary *> *NFKStubScript;
static NSUInteger NFKStubRequestsServed;

@implementation NFKStubURLProtocol

+ (BOOL)canInitWithRequest:(NSURLRequest *)request
{
	return [request.URL.host isEqualToString:@"stub.test"];
}

+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request
{
	return request;
}

- (void)startLoading
{
	NSDictionary *step = NFKStubScript.firstObject ?: @{ @"status": @200, @"chunks": @[] };
	if (NFKStubScript.count > 0) {
		[NFKStubScript removeObjectAtIndex:0];
	}
	NFKStubRequestsServed++;
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:self.request.URL
															  statusCode:[step[@"status"] integerValue]
															 HTTPVersion:@"HTTP/1.1"
															headerFields:step[@"headers"]];
	[self.client URLProtocol:self didReceiveResponse:response cacheStoragePolicy:NSURLCacheStorageNotAllowed];
	for (NSString *chunk in step[@"chunks"]) {
		[self.client URLProtocol:self didLoadData:[chunk dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[self.client URLProtocolDidFinishLoading:self];
}

- (void)stopLoading
{
}

@end

@interface NFKRemoteTransportTests : XCTestCase
@property (nonatomic, strong) NSURLSession *session;
@property (nonatomic, strong) NSURLRequest *request;
@end

@implementation NFKRemoteTransportTests

- (void)setUp
{
	[super setUp];
	NFKStubScript = [NSMutableArray array];
	NFKStubRequestsServed = 0;
	NSURLSessionConfiguration *configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];
	configuration.protocolClasses = @[ NFKStubURLProtocol.class ];
	self.session = [NSURLSession sessionWithConfiguration:configuration];
	self.request = [NSURLRequest requestWithURL:[NSURL URLWithString:@"http://stub.test/v1/chat/completions"]];
	NFKRemoteTransport.retryAttempts = 2;
	NFKRemoteTransport.maximumRetryDelay = 8;
}

- (void)tearDown
{
	NFKRemoteTransport.retryAttempts = 2;
	NFKRemoteTransport.maximumRetryDelay = 8;
	[super tearDown];
}

#pragma mark Retries

- (void)testARateLimitThatClearsIsRetriedAfterTheProvidersRetryAfter
{
	[NFKStubScript addObject:@{ @"status": @429, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"slow down" ] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"ok" ] }];

	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:&error];
	XCTAssertEqualObjects(data, [@"ok" dataUsingEncoding:NSUTF8StringEncoding], @"%@", error);
	XCTAssertEqual(response.statusCode, 200);
	XCTAssertEqual(NFKStubRequestsServed, 2);
}

- (void)testARateLimitThatPersistsIsGivenUpAfterTheAttemptsAndReportedAsIs
{
	NFKRemoteTransport.retryAttempts = 1;
	[NFKStubScript addObject:@{ @"status": @503, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"down" ] }];
	[NFKStubScript addObject:@{ @"status": @503, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"still down" ] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"never reached" ] }];

	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	XCTAssertEqual(response.statusCode, 503, @"the last answer is what the caller sees");
	XCTAssertEqualObjects(data, [@"still down" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqual(NFKStubRequestsServed, 2, @"one attempt plus one retry");
	XCTAssertNotNil([NFKRemoteTransport errorForResponse:response data:data], @"and it maps to an error as before");
}

- (void)testARetryAfterLongerThanTheCeilingEndsTheRetries
{
	[NFKStubScript addObject:@{ @"status": @429, @"headers": @{ @"Retry-After": @"120" }, @"chunks": @[ @"later" ] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"never reached" ] }];

	NSHTTPURLResponse *response = nil;
	[NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	XCTAssertEqual(response.statusCode, 429, @"two minutes is longer than the caller would wait");
	XCTAssertEqual(NFKStubRequestsServed, 1);
}

#pragma mark The code a status maps to

- (NSHTTPURLResponse *)responseWithStatus:(NSInteger)status headers:(NSDictionary *)headers
{
	return [[NSHTTPURLResponse alloc] initWithURL:self.request.URL statusCode:status HTTPVersion:@"HTTP/1.1" headerFields:headers];
}

// Back off, change the request, or fix the configuration: the three answers an app acts on, read
// from the status, with the provider's own explanation kept beside the code.
- (void)testAStatusMapsToTheCodeAnAppActsOn
{
	NSData *body = [@"{\"error\":\"why\"}" dataUsingEncoding:NSUTF8StringEncoding];
	NSDictionary<NSNumber *, NSNumber *> *expected = @{
		@429: @(kNFKError_InferenceRateLimited), @529: @(kNFKError_InferenceRateLimited), @402: @(kNFKError_InferenceRateLimited),
		@400: @(kNFKError_InferenceRefused), @413: @(kNFKError_InferenceRefused), @422: @(kNFKError_InferenceRefused),
		@401: @(kNFKError_InferenceBackendFailure), @403: @(kNFKError_InferenceBackendFailure),
		@404: @(kNFKError_InferenceBackendFailure), @500: @(kNFKError_InferenceBackendFailure),
		@503: @(kNFKError_InferenceBackendFailure),
	};
	for (NSNumber *status in expected) {
		NSError *error = [NFKRemoteTransport errorForResponse:[self responseWithStatus:status.integerValue headers:nil] data:body];
		XCTAssertEqual(error.code, expected[status].integerValue, @"HTTP %@", status);
		XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], status);
		XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorBodyKey], @"{\"error\":\"why\"}");
		XCTAssertTrue([error.localizedDescription containsString:@"why"], @"the provider's explanation is kept");
	}
	XCTAssertNil([NFKRemoteTransport errorForResponse:[self responseWithStatus:204 headers:nil] data:nil]);
}

- (void)testARateLimitCarriesTheProvidersResetDate
{
	NSError *seconds = [NFKRemoteTransport errorForResponse:[self responseWithStatus:429 headers:@{ @"Retry-After": @"30" }] data:nil];
	NSDate *reset = seconds.userInfo[NFKRemoteErrorRetryAfterKey];
	XCTAssertNotNil(reset);
	XCTAssertEqualWithAccuracy(reset.timeIntervalSinceNow, 30, 2, @"a delay in seconds is counted from now");

	NSError *dated = [NFKRemoteTransport errorForResponse:[self responseWithStatus:529 headers:@{ @"Retry-After": @"Wed, 21 Oct 2015 07:28:00 GMT" }] data:nil];
	XCTAssertEqualObjects(dated.userInfo[NFKRemoteErrorRetryAfterKey], [NSDate dateWithTimeIntervalSince1970:1445412480], @"an HTTP-date is read as it stands");

	NSError *bare = [NFKRemoteTransport errorForResponse:[self responseWithStatus:429 headers:nil] data:nil];
	XCTAssertEqual(bare.code, kNFKError_InferenceRateLimited);
	XCTAssertNil(bare.userInfo[NFKRemoteErrorRetryAfterKey], @"no header, no date");
	NSError *refused = [NFKRemoteTransport errorForResponse:[self responseWithStatus:422 headers:@{ @"Retry-After": @"5" }] data:nil];
	XCTAssertNil(refused.userInfo[NFKRemoteErrorRetryAfterKey], @"a reset date rides only on a rate limit");
}

// The retried statuses reach the caller as the last answer, so an exhausted rate limit arrives
// under its own code rather than the generic failure it used to be.
- (void)testAnExhaustedRateLimitIsReportedAsOne
{
	NFKRemoteTransport.retryAttempts = 1;
	[NFKStubScript addObject:@{ @"status": @429, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"slow" ] }];
	[NFKStubScript addObject:@{ @"status": @429, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"still slow" ] }];
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	NSError *error = [NFKRemoteTransport errorForResponse:response data:data];
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);
	XCTAssertNotNil(error.userInfo[NFKRemoteErrorRetryAfterKey]);
	XCTAssertEqual(NFKStubRequestsServed, 2);
}

- (void)testAnOrdinaryFailureIsNotRetried
{
	[NFKStubScript addObject:@{ @"status": @401, @"chunks": @[ @"bad key" ] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"never reached" ] }];
	NSHTTPURLResponse *response = nil;
	[NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	XCTAssertEqual(response.statusCode, 401);
	XCTAssertEqual(NFKStubRequestsServed, 1, @"a rejected key does not clear by waiting");
}

// Anthropic and TypeSafe answer 529 when overloaded and ask for a backoff, the way a 503 does.
- (void)testAnOverloadIsRetriedLikeAGatewayError
{
	[NFKStubScript addObject:@{ @"status": @529, @"headers": @{ @"Retry-After": @"0" }, @"chunks": @[ @"overloaded" ] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"ok" ] }];
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	XCTAssertEqualObjects(data, [@"ok" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqual(NFKStubRequestsServed, 2);
}

- (void)testAGatewayErrorWithoutRetryAfterBacksOffExponentiallyFromHalfASecond
{
	NFKRemoteTransport.maximumRetryDelay = 0.6;   // admits the first retry (0.5 s) and not the second (1 s)
	[NFKStubScript addObject:@{ @"status": @502, @"chunks": @[] }];
	[NFKStubScript addObject:@{ @"status": @502, @"chunks": @[] }];
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"never reached" ] }];

	NSDate *start = [NSDate date];
	NSHTTPURLResponse *response = nil;
	[NFKRemoteTransport sendRequest:self.request session:self.session response:&response error:NULL];
	NSTimeInterval elapsed = -start.timeIntervalSinceNow;
	XCTAssertEqual(NFKStubRequestsServed, 2);
	XCTAssertEqual(response.statusCode, 502);
	XCTAssertGreaterThanOrEqual(elapsed, 0.45, @"the first retry waited about half a second");
	XCTAssertLessThan(elapsed, 2.0, @"the second, at a second, was over the ceiling and not waited for");
}

#pragma mark The line stream

- (void)testLinesAreDeliveredAsTheyCompleteAcrossRaggedChunks
{
	[NFKStubScript addObject:@{ @"status": @200, @"chunks": @[ @"data: one\r\nda", @"ta: two\n\n", @"data: three" ] }];
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	XCTestExpectation *ended = [self expectationWithDescription:@"stream ended"];
	__block NSHTTPURLResponse *ended_response = nil;
	__block NSError *ended_error = nil;
	[NFKRemoteTransport streamRequest:self.request session:self.session lineHandler:^(NSString *line) {
		@synchronized (lines) { [lines addObject:line]; }
	} completionHandler:^(NSHTTPURLResponse *response, NSData *errorBody, NSError *error) {
		ended_response = response;
		ended_error = error;
		[ended fulfill];
	}];
	[self waitForExpectations:@[ ended ] timeout:5];

	XCTAssertNil(ended_error);
	XCTAssertEqual(ended_response.statusCode, 200);
	XCTAssertEqualObjects(lines, (@[ @"data: one", @"data: two", @"", @"data: three" ]),
						  @"a line split across chunks joins, a CR is dropped, and the unterminated last line arrives");
}

- (void)testAFailingStatusIsCollectedWholeRatherThanStreamed
{
	[NFKStubScript addObject:@{ @"status": @401, @"chunks": @[ @"{\"error\":", @"\"bad key\"}" ] }];
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	XCTestExpectation *ended = [self expectationWithDescription:@"stream ended"];
	__block NSData *body = nil;
	__block NSHTTPURLResponse *ended_response = nil;
	[NFKRemoteTransport streamRequest:self.request session:self.session lineHandler:^(NSString *line) {
		@synchronized (lines) { [lines addObject:line]; }
	} completionHandler:^(NSHTTPURLResponse *response, NSData *errorBody, NSError *error) {
		body = errorBody;
		ended_response = response;
		[ended fulfill];
	}];
	[self waitForExpectations:@[ ended ] timeout:5];

	XCTAssertEqual(lines.count, 0, @"nothing was delivered as a line");
	XCTAssertEqualObjects(body, [@"{\"error\":\"bad key\"}" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertTrue([[NFKRemoteTransport errorForResponse:ended_response data:body].localizedDescription containsString:@"bad key"]);
}

- (void)testCancellingTheStreamEndsItWithCancelled
{
	NSURLRequest *slow = [NSURLRequest requestWithURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1/never"]];
	XCTestExpectation *ended = [self expectationWithDescription:@"stream ended"];
	__block NSError *ended_error = nil;
	void (^cancel)(void) = [NFKRemoteTransport streamRequest:slow session:NSURLSession.sharedSession lineHandler:^(NSString *line) {
	} completionHandler:^(NSHTTPURLResponse *response, NSData *errorBody, NSError *error) {
		ended_error = error;
		[ended fulfill];
	}];
	cancel();
	[self waitForExpectations:@[ ended ] timeout:5];
	// Cancelled before or after the refused connection: either ending is a real ending, never a hang.
	XCTAssertTrue(ended_error.code == NSURLErrorCancelled || ended_error.code == kNFKError_RemoteUnreachable, @"%@", ended_error);
}

#pragma mark SSE

- (void)testOnlyDataLinesCarryAPayload
{
	XCTAssertEqualObjects([NFKRemoteTransport SSEDataForLine:@"data: {\"a\":1}"], @"{\"a\":1}");
	XCTAssertEqualObjects([NFKRemoteTransport SSEDataForLine:@"data:{\"a\":1}"], @"{\"a\":1}", @"the space is optional");
	XCTAssertEqualObjects([NFKRemoteTransport SSEDataForLine:@"data: [DONE]"], @"[DONE]");
	XCTAssertNil([NFKRemoteTransport SSEDataForLine:@"event: message_start"]);
	XCTAssertNil([NFKRemoteTransport SSEDataForLine:@": keep-alive"]);
	XCTAssertNil([NFKRemoteTransport SSEDataForLine:@""]);
}

@end
