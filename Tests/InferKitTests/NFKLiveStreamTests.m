//
//  NFKLiveStreamTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

/*! Echoes each request's inputs and parameters as outputs. */
@interface NFKLiveEchoBackend : NSObject <NFKInferenceBackend>
@property (nonatomic) NSInteger runs;
@end

@implementation NFKLiveEchoBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"live-echo";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	self.runs += 1;
	NSMutableDictionary *outputs = [request.inputs mutableCopy];
	[outputs addEntriesFromDictionary:request.parameters];
	return [NFKInferenceResult resultWithOutputs:outputs];
}

@end

/*! Sums the numbers of the units before, a model whose result depends on the state it carries. */
@interface NFKLiveCountingBackend : NSObject <NFKInferenceBackend>
@property (nonatomic) NSInteger opened;
@end

@implementation NFKLiveCountingBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"live-counting";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ @"total": [request inputForKey:@"value"] ?: @0 }];
}

- (nullable id<NFKLiveStream>)openLiveStreamWithParameters:(nullable NSDictionary<NSString *, id> *)parameters
													 error:(NSError **)error
{
	self.opened += 1;
	__block NSInteger total = 0;
	return [[NFKBlockLiveStream alloc] initWithParameters:parameters unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **unitError) {
		total += [[unit inputForKey:@"value"] integerValue] * [[unit parameterForKey:@"scale"] ?: @1 integerValue];
		return [NFKInferenceResult resultWithOutputs:@{ @"total": @(total) }];
	} resetHandler:^{
		total = 0;
	} closeHandler:nil];
}

@end

@interface NFKLiveStreamTests : XCTestCase
@end

@implementation NFKLiveStreamTests

- (NFKInferenceRequest *)unitWithValue:(NSInteger)value
{
	return [NFKInferenceRequest requestWithInputs:@{ @"value": @(value) }];
}

- (void)testABackendWithoutAStreamRunsEachUnitAloneWithTheStreamsParameters
{
	NFKLiveEchoBackend *backend = [[NFKLiveEchoBackend alloc] init];
	NSError *error = nil;
	id<NFKLiveStream> stream = NFKInferenceOpenLiveStream(backend, @{ @"threshold": @0.5, @"mode": @"fast" }, &error);
	XCTAssertNotNil(stream, @"%@", error);
	NFKInferenceResult *result = [stream resultForUnit:[NFKInferenceRequest requestWithInputs:@{ @"value": @1 }
																				   parameters:@{ @"mode": @"exact" }]
												 error:&error];
	XCTAssertEqualObjects([result outputForKey:@"value"], @1, @"%@", error);
	XCTAssertEqualObjects([result outputForKey:@"threshold"], @0.5, @"the stream's parameters apply to the unit");
	XCTAssertEqualObjects([result outputForKey:@"mode"], @"exact", @"a unit's own parameter overrides the stream's");
	XCTAssertEqual(backend.runs, 1);
}

- (void)testAStatefulStreamCarriesItsStateUntilReset
{
	NFKLiveCountingBackend *backend = [[NFKLiveCountingBackend alloc] init];
	NSError *error = nil;
	id<NFKLiveStream> stream = NFKInferenceOpenLiveStream(backend, @{ @"scale": @10 }, &error);
	XCTAssertEqual(backend.opened, 1, @"the backend's own stream is used: %@", error);
	NSMutableArray<NSNumber *> *totals = [NSMutableArray array];
	for (NSInteger value = 1; value <= 3; value++) {
		[totals addObject:[[stream resultForUnit:[self unitWithValue:value] error:&error] outputForKey:@"total"]];
	}
	XCTAssertEqualObjects(totals, (@[ @10, @30, @60 ]), @"%@", error);
	[stream reset];
	XCTAssertEqualObjects([[stream resultForUnit:[self unitWithValue:4] error:&error] outputForKey:@"total"], @40,
						  @"reset starts a new stream: %@", error);
}

- (void)testTwoStreamsOfOneBackendKeepTheirOwnState
{
	NFKLiveCountingBackend *backend = [[NFKLiveCountingBackend alloc] init];
	id<NFKLiveStream> first = NFKInferenceOpenLiveStream(backend, nil, NULL);
	id<NFKLiveStream> second = NFKInferenceOpenLiveStream(backend, nil, NULL);
	[first resultForUnit:[self unitWithValue:5] error:NULL];
	XCTAssertEqualObjects([[second resultForUnit:[self unitWithValue:1] error:NULL] outputForKey:@"total"], @1);
	XCTAssertEqualObjects([[first resultForUnit:[self unitWithValue:1] error:NULL] outputForKey:@"total"], @6);
}

- (void)testAClosedStreamRefusesUnitsAndRunsItsCloseOnce
{
	__block NSInteger closes = 0;
	NFKBlockLiveStream *stream = [[NFKBlockLiveStream alloc] initWithParameters:nil unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **error) {
		return [NFKInferenceResult resultWithOutputs:unit.inputs];
	} resetHandler:nil closeHandler:^{
		closes += 1;
	}];
	XCTAssertNotNil([stream resultForUnit:[self unitWithValue:1] error:NULL]);
	[stream close];
	[stream close];
	[stream reset];
	XCTAssertTrue(stream.isClosed);
	XCTAssertEqual(closes, 1);
	NSError *error = nil;
	XCTAssertNil([stream resultForUnit:[self unitWithValue:1] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);
}

- (void)testUnitsPushedFromManyThreadsAreAnsweredOneAtATime
{
	__block NSInteger inFlight = 0, most = 0;
	NSObject *counter = [[NSObject alloc] init];
	NFKBlockLiveStream *stream = [[NFKBlockLiveStream alloc] initWithParameters:nil unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **error) {
		@synchronized (counter) {
			inFlight += 1;
			most = MAX(most, inFlight);
		}
		[NSThread sleepForTimeInterval:0.001];
		@synchronized (counter) {
			inFlight -= 1;
		}
		return [NFKInferenceResult resultWithOutputs:@{}];
	} resetHandler:nil closeHandler:nil];
	dispatch_apply(16, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t index) {
		[stream resultForUnit:[self unitWithValue:(NSInteger)index] error:NULL];
	});
	XCTAssertEqual(most, 1);
}

@end
