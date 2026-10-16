//
//  NFKJobGuardTests.m
//  InferKitTests
//
//  The core's job backends fail a job with an error when work on their own queues raises: a run
//  NFKInferenceSubmit queues, a remote stream's line and completion handlers, and a served job backend
//  that raises on its queue, answered 500 by a server that keeps serving.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! A synchronous backend that raises while a test asks, and answers ok otherwise. */
@interface NFKJobGuardRaisingBackend : NSObject <NFKInferenceBackend>
@property (atomic, assign) BOOL raises;
@end

@implementation NFKJobGuardRaisingBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"job-guard-raising";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	if (self.raises) {
		[NSException raise:NSInternalInconsistencyException format:@"the test backend raises on its queue"];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

/*! A job backend whose job runs its inner backend through NFKInferenceSubmit, on a queue of its own. */
@interface NFKJobGuardQueuedBackend : NFKJobGuardRaisingBackend
@end

@implementation NFKJobGuardQueuedBackend

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKJobGuardRaisingBackend *inner = [[NFKJobGuardRaisingBackend alloc] init];
	inner.raises = self.raises;
	return NFKInferenceSubmit(inner, request, NULL);
}

@end

@interface NFKJobGuardTests : XCTestCase
@end

@implementation NFKJobGuardTests

- (NFKInferenceRequest *)request
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" } parameters:@{} outputModality:NFKModalityText];
}

- (void)testSubmitFailsTheJobWhenTheRunRaises
{
	NFKJobGuardRaisingBackend *backend = [[NFKJobGuardRaisingBackend alloc] init];
	backend.raises = YES;
	XCTestExpectation *finished = [self expectationWithDescription:@"the job finishes"];
	NFKInferenceJob *job = NFKInferenceSubmit(backend, [self request], NULL);
	job.completionHandler = ^(NFKInferenceJob *done) {
		[finished fulfill];
	};
	[self waitForExpectations:@[ finished ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqualObjects(job.error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(job.error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([job.error.localizedDescription containsString:NSInternalInconsistencyException], @"%@", job.error);
}

/*! A server whose app route streams three lines, started for the test and stopped after it. */
- (NFKInferenceServer *)streamingServer
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.advertisesService = NO;
	NSError *error = nil;
	XCTAssertTrue([server addRouteForPathPrefix:@"/lines" handler:^(NFKServerRouteRequest *request, NFKServerRouteResponse *response) {
		[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
		for (NSString *line in @[ @"one", @"two", @"three" ]) {
			[response sendEventData:line];
		}
		[response endStream];
	} error:&error], @"%@", error);
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NSURLRequest *)linesRequestOn:(NFKInferenceServer *)server
{
	return [NSURLRequest requestWithURL:[NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u/lines", (unsigned)server.listeningPort]]];
}

- (void)testAStreamWhoseLineHandlerRaisesCompletesWithTheError
{
	NFKInferenceServer *server = [self streamingServer];
	XCTestExpectation *completed = [self expectationWithDescription:@"the stream completes"];
	__block NSUInteger lines = 0;
	__block NSError *reported = nil;
	[NFKRemoteTransport streamRequest:[self linesRequestOn:server] session:NSURLSession.sharedSession lineHandler:^(NSString *line) {
		lines += 1;
		[NSException raise:NSRangeException format:@"the line handler raises"];
	} completionHandler:^(NSHTTPURLResponse *response, NSData *errorBody, NSError *error) {
		reported = error;
		[completed fulfill];
	}];
	[self waitForExpectations:@[ completed ] timeout:10];
	XCTAssertEqual(lines, 1u, @"no line arrives after the handler raised");
	XCTAssertEqual(reported.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([reported.localizedDescription containsString:NSRangeException], @"%@", reported);
}

- (void)testAStreamWhoseCompletionRaisesIsCalledAgainWithTheError
{
	NFKInferenceServer *server = [self streamingServer];
	XCTestExpectation *retried = [self expectationWithDescription:@"the completion runs again"];
	__block NSUInteger calls = 0;
	__block NSError *second = nil;
	[NFKRemoteTransport streamRequest:[self linesRequestOn:server] session:NSURLSession.sharedSession lineHandler:^(NSString *line) {
	} completionHandler:^(NSHTTPURLResponse *response, NSData *errorBody, NSError *error) {
		calls += 1;
		if (calls == 1) {
			[NSException raise:NSGenericException format:@"the completion raises"];
		}
		second = error;
		[retried fulfill];
	}];
	[self waitForExpectations:@[ retried ] timeout:10];
	XCTAssertEqual(calls, 2u);
	XCTAssertEqual(second.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([second.localizedDescription containsString:NSGenericException], @"%@", second);
}

- (void)testAServedJobBackendThatRaisesOnItsQueueAnswers500
{
	NFKJobGuardQueuedBackend *backend = [[NFKJobGuardQueuedBackend alloc] init];
	backend.raises = YES;
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.advertisesService = NO;
	[server addBackend:backend forModelName:@"queued"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NSURL *chat = [NSURL URLWithString:[NSString stringWithFormat:@"http://localhost:%u/v1/chat/completions", (unsigned)server.listeningPort]];
	NSData *body = [NSJSONSerialization dataWithJSONObject:@{ @"model": @"queued", @"messages": @[ @{ @"role": @"user", @"content": @"hi" } ] }
												   options:0 error:NULL];
	NSInteger (^post)(void) = ^NSInteger {
		NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:chat];
		request.HTTPMethod = @"POST";
		request.HTTPBody = body;
		request.timeoutInterval = 30;
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
		NSHTTPURLResponse *response = nil;
		[NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:NULL];
		return response.statusCode;
	};
	XCTAssertEqual(post(), 500, @"the job fails with the exception's error");
	backend.raises = NO;
	XCTAssertEqual(post(), 200, @"the server keeps serving");
}

@end
