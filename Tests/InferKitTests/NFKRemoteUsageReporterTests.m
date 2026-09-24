//
//  NFKRemoteUsageReporterTests.m
//  InferKitTests
//
//  Usage, cost, and balance reports through a scripted transport: each style's query shape, the
//  unit normalization (cents to dollars, Unix seconds and RFC 3339 to dates), and paging.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteUsageReporter.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKScriptedUsageReporter : NFKRemoteUsageReporter
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *replies;
@end

@implementation NFKScriptedUsageReporter
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	NSString *reply = self.replies.firstObject ?: @"{}";
	if (self.replies.count > 0) {
		[self.replies removeObjectAtIndex:0];
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [reply dataUsingEncoding:NSUTF8StringEncoding];
}
- (NSDictionary<NSString *, NSArray<NSString *> *> *)queryAt:(NSUInteger)index
{
	NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *query = [NSMutableDictionary dictionary];
	for (NSURLQueryItem *item in [NSURLComponents componentsWithURL:self.requests[index].URL resolvingAgainstBaseURL:NO].queryItems) {
		NSMutableArray<NSString *> *values = query[item.name] ?: [NSMutableArray array];
		[values addObject:item.value ?: @""];
		query[item.name] = values;
	}
	return query;
}
@end

@interface NFKRemoteUsageReporterTests : XCTestCase
@end

@implementation NFKRemoteUsageReporterTests

- (NFKScriptedUsageReporter *)reporterFor:(NFKRemoteProvider *)provider
{
	NFKRemoteUsageReporter *made = [NFKRemoteUsageReporter reporterForProvider:provider apiKey:@"admin"];
	NFKScriptedUsageReporter *reporter = [[NFKScriptedUsageReporter alloc] init];
	reporter.endpointURL = made.endpointURL;
	reporter.apiStyle = made.apiStyle;
	reporter.apiKey = @"admin";
	reporter.replies = [NSMutableArray array];
	return reporter;
}

- (void)testOnlyTheServicesWithReportsGetAReporter
{
	XCTAssertEqual([NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.anthropic apiKey:@"k"].apiStyle, NFKRemoteUsageAPIStyleAnthropic);
	XCTAssertEqual([NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.deepSeek apiKey:@"k"].apiStyle, NFKRemoteUsageAPIStyleDeepSeek);
	XCTAssertNil([NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.groq apiKey:@"k"]);
	XCTAssertNil([NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.together apiKey:@"k"]);
	XCTAssertNil([NFKRemoteUsageReporter reporterForProvider:NFKRemoteProvider.googleGemini apiKey:@"k"]);
}

- (void)testAnthropicUsageReadsEveryPageWithRFC3339Bounds
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.anthropic];
	[reporter.replies addObjectsFromArray:@[
		(@"{\"data\":[{\"starting_at\":\"2026-09-01T00:00:00Z\",\"ending_at\":\"2026-09-02T00:00:00Z\","
		 "\"results\":[{\"model\":\"claude-opus-5-5\",\"uncached_input_tokens\":1200}]}],\"has_more\":true,\"next_page\":\"p2\"}"),
		@"{\"data\":[{\"starting_at\":\"2026-09-02T00:00:00Z\",\"ending_at\":\"2026-09-03T00:00:00Z\",\"results\":[]}],\"has_more\":false}" ]];
	NSDate *start = [NSDate dateWithTimeIntervalSince1970:1788220800];
	NSError *error = nil;
	NSArray<NFKUsageBucket *> *buckets = [reporter usageFromDate:start toDate:nil bucketWidth:@"1d" groupBy:@[ @"model", @"api_key_id" ] report:nil error:&error];
	XCTAssertEqual(buckets.count, 2u, @"%@", error);
	XCTAssertEqualObjects(buckets[0].results[0][@"model"], @"claude-opus-5-5");
	XCTAssertEqualObjects(buckets[0].startDate, start);
	XCTAssertEqualObjects(reporter.requests[0].URL.path, @"/v1/organizations/usage_report/messages");
	XCTAssertEqualObjects([reporter.requests[0] valueForHTTPHeaderField:@"x-api-key"], @"admin");
	NSDictionary<NSString *, NSArray<NSString *> *> *query = [reporter queryAt:0];
	XCTAssertEqualObjects(query[@"starting_at"].firstObject, @"2026-09-01T00:00:00Z");
	NSArray *groups = @[ @"model", @"api_key_id" ];
	XCTAssertEqualObjects(query[@"group_by[]"], groups);
	XCTAssertEqualObjects([reporter queryAt:1][@"page"].firstObject, @"p2");
}

- (void)testAnthropicCostsConvertCentsToDollars
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.anthropic];
	[reporter.replies addObject:@"{\"data\":[{\"starting_at\":\"2026-09-01T00:00:00Z\",\"ending_at\":\"2026-09-02T00:00:00Z\","
	 "\"results\":[{\"amount\":\"1234.5\",\"currency\":\"USD\",\"description\":\"Claude Opus 5.5 input\"}]}],\"has_more\":false}"];
	NSError *error = nil;
	NSArray<NFKCostEntry *> *costs = [reporter costsFromDate:[NSDate dateWithTimeIntervalSince1970:1788220800] toDate:nil groupBy:nil error:&error];
	XCTAssertEqual(costs.count, 1u, @"%@", error);
	XCTAssertEqualObjects(costs[0].amount, [NSDecimalNumber decimalNumberWithString:@"12.345"]);
	XCTAssertEqualObjects(costs[0].currency, @"USD");
	XCTAssertEqualObjects(costs[0].lineItem, @"Claude Opus 5.5 input");
	XCTAssertEqualObjects(reporter.requests[0].URL.path, @"/v1/organizations/cost_report");
}

- (void)testOpenAIUsesUnixSecondsAndTheNamedReport
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.openAI];
	[reporter.replies addObjectsFromArray:@[
		@"{\"data\":[{\"start_time\":1788220800,\"end_time\":1788307200,\"results\":[{\"input_tokens\":10}]}],\"has_more\":false}",
		(@"{\"data\":[{\"start_time\":1788220800,\"end_time\":1788307200,"
		 "\"results\":[{\"amount\":{\"value\":0.42,\"currency\":\"usd\"},\"line_item\":\"gpt-5.6-sol, input\"}]}],\"has_more\":false}") ]];
	NSDate *start = [NSDate dateWithTimeIntervalSince1970:1788220800];
	NSError *error = nil;
	NSArray<NFKUsageBucket *> *buckets = [reporter usageFromDate:start toDate:nil bucketWidth:nil groupBy:nil report:@"embeddings" error:&error];
	XCTAssertEqual(buckets.count, 1u, @"%@", error);
	XCTAssertEqualObjects(buckets[0].endDate, [NSDate dateWithTimeIntervalSince1970:1788307200]);
	XCTAssertEqualObjects(reporter.requests[0].URL.path, @"/v1/organization/usage/embeddings");
	XCTAssertEqualObjects([reporter queryAt:0][@"start_time"].firstObject, @"1788220800");
	XCTAssertEqualObjects([reporter.requests[0] valueForHTTPHeaderField:@"Authorization"], @"Bearer admin");

	NSArray<NFKCostEntry *> *costs = [reporter costsFromDate:start toDate:nil groupBy:@[ @"line_item" ] error:&error];
	XCTAssertEqualObjects(costs[0].amount, [NSDecimalNumber decimalNumberWithString:@"0.42"]);
	XCTAssertEqualObjects(costs[0].currency, @"USD");
	XCTAssertEqualObjects(reporter.requests[1].URL.path, @"/v1/organization/costs");
}

- (void)testXAINeedsATeamAndFoldsItsTimeSeries
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.xAI];
	NSError *error = nil;
	XCTAssertNil([reporter balancesWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);

	reporter.teamIdentifier = @"team-1";
	[reporter.replies addObjectsFromArray:@[
		@"{\"timeSeries\":[{\"groupLabels\":[\"grok-5\"],\"dataPoints\":[{\"timestamp\":\"2026-09-01T00:00:00Z\",\"values\":[3.5]}]}]}",
		@"{\"total\":{\"val\":\"-2500\"}}" ]];
	NSArray<NFKCostEntry *> *costs = [reporter costsFromDate:[NSDate dateWithTimeIntervalSince1970:1788220800]
													  toDate:[NSDate dateWithTimeIntervalSince1970:1788307200] groupBy:nil error:&error];
	XCTAssertEqual(costs.count, 1u, @"%@", error);
	XCTAssertEqualObjects(costs[0].amount, [NSDecimalNumber decimalNumberWithString:@"3.5"]);
	XCTAssertEqualObjects(costs[0].lineItem, @"grok-5");
	XCTAssertEqualObjects(reporter.requests[0].URL.absoluteString, @"https://management-api.x.ai/v1/billing/teams/team-1/usage");
	NSDictionary *body = [NSJSONSerialization JSONObjectWithData:reporter.requests[0].HTTPBody options:0 error:NULL];
	XCTAssertEqualObjects(body[@"analyticsRequest"][@"timeRange"][@"startTime"], @"2026-09-01 00:00:00");

	NSArray<NFKAccountBalance *> *balances = [reporter balancesWithError:&error];
	XCTAssertEqualObjects(balances[0].amount, [NSDecimalNumber decimalNumberWithString:@"25"]);
}

- (void)testOpenRouterAndDeepSeekBalances
{
	NFKScriptedUsageReporter *router = [self reporterFor:NFKRemoteProvider.openRouter];
	[router.replies addObject:@"{\"data\":{\"total_credits\":50,\"total_usage\":12.25}}"];
	NSError *error = nil;
	XCTAssertEqualObjects([router balancesWithError:&error][0].amount, [NSDecimalNumber decimalNumberWithString:@"37.75"], @"%@", error);
	XCTAssertEqualObjects(router.requests[0].URL.absoluteString, @"https://openrouter.ai/api/v1/credits");

	NFKScriptedUsageReporter *deepSeek = [self reporterFor:NFKRemoteProvider.deepSeek];
	[deepSeek.replies addObject:@"{\"is_available\":true,\"balance_infos\":[{\"currency\":\"CNY\",\"total_balance\":\"110.00\"},"
	 "{\"currency\":\"USD\",\"total_balance\":\"3.20\"}]}"];
	NSArray<NFKAccountBalance *> *balances = [deepSeek balancesWithError:&error];
	XCTAssertEqual(balances.count, 2u, @"%@", error);
	XCTAssertEqualObjects(balances[1].amount, [NSDecimalNumber decimalNumberWithString:@"3.2"]);
	XCTAssertEqualObjects(deepSeek.requests[0].URL.absoluteString, @"https://api.deepseek.com/user/balance");
	XCTAssertNil([deepSeek usageFromDate:[NSDate date] toDate:nil bucketWidth:nil groupBy:nil report:nil error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

- (void)testOpenRouterActivityBecomesDailyBuckets
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.openRouter];
	[reporter.replies addObject:@"{\"data\":[{\"date\":\"2026-09-01\",\"model\":\"a\",\"usage\":1.5},{\"date\":\"2026-09-01\",\"model\":\"b\",\"usage\":0.5},"
	 "{\"date\":\"2026-09-02\",\"model\":\"a\",\"usage\":2},{\"date\":\"2026-08-01\",\"model\":\"a\",\"usage\":9}]}"];
	NSError *error = nil;
	NSArray<NFKUsageBucket *> *buckets = [reporter usageFromDate:[NSDate dateWithTimeIntervalSince1970:1788220800] toDate:nil
													 bucketWidth:nil groupBy:nil report:nil error:&error];
	XCTAssertEqual(buckets.count, 2u, @"%@", error);
	XCTAssertEqual(buckets[0].results.count, 2u);
	XCTAssertEqualObjects(reporter.requests[0].URL.path, @"/api/v1/activity");
}

- (void)testMistralSendsTheAdminKeyAndMonth
{
	NFKScriptedUsageReporter *reporter = [self reporterFor:NFKRemoteProvider.mistral];
	[reporter.replies addObject:@"{\"total\":1}"];
	NSError *error = nil;
	NSArray<NFKUsageBucket *> *buckets = [reporter usageFromDate:[NSDate dateWithTimeIntervalSince1970:1788220800] toDate:nil
													 bucketWidth:nil groupBy:nil report:nil error:&error];
	XCTAssertEqual(buckets.count, 1u, @"%@", error);
	XCTAssertEqualObjects(buckets[0].results[0][@"total"], @1);
	XCTAssertEqualObjects([reporter.requests[0] valueForHTTPHeaderField:@"x-api-key"], @"admin");
	XCTAssertEqualObjects(reporter.requests[0].URL.path, @"/v1/admin/usage");
	XCTAssertEqualObjects([reporter queryAt:0][@"month"].firstObject, @"9");
	XCTAssertEqualObjects([reporter queryAt:0][@"year"].firstObject, @"2026");
}

@end
