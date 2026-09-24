//
//  NFKRemoteUsageReporter.m
//  InferKit
//

#import <InferKit/NFKRemoteUsageReporter.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

static NSISO8601DateFormatter *NFKUsageTimestampFormatter(void)
{
	static NSISO8601DateFormatter *formatter;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		formatter = [[NSISO8601DateFormatter alloc] init];
	});
	return formatter;
}

static NSISO8601DateFormatter *NFKUsageDayFormatter(void)
{
	static NSISO8601DateFormatter *formatter;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		formatter = [[NSISO8601DateFormatter alloc] init];
		formatter.formatOptions = NSISO8601DateFormatWithFullDate | NSISO8601DateFormatWithDashSeparatorInDate;
	});
	return formatter;
}

/*! A date from Unix seconds, RFC 3339, or a bare day. */
static NSDate * _Nullable NFKUsageDate(id value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return [NSDate dateWithTimeIntervalSince1970:[value doubleValue]];
	}
	if (![value isKindOfClass:NSString.class]) {
		return nil;
	}
	return [NFKUsageTimestampFormatter() dateFromString:value] ?: [NFKUsageDayFormatter() dateFromString:[value substringToIndex:MIN((NSUInteger)10, [value length])]];
}

/*! A decimal from a number or a numeric string. */
static NSDecimalNumber *NFKUsageDecimal(id value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return [NSDecimalNumber decimalNumberWithDecimal:[value decimalValue]];
	}
	if ([value isKindOfClass:NSString.class]) {
		NSDecimalNumber *number = [NSDecimalNumber decimalNumberWithString:value locale:@{ NSLocaleDecimalSeparator: @"." }];
		return [number isEqualToNumber:NSDecimalNumber.notANumber] ? NSDecimalNumber.zero : number;
	}
	return NSDecimalNumber.zero;
}

static NSDictionary *NFKUsageDictionary(id value)
{
	return [value isKindOfClass:NSDictionary.class] ? value : @{};
}

static NSArray *NFKUsageArray(id value)
{
	return [value isKindOfClass:NSArray.class] ? value : @[];
}

@interface NFKUsageBucket ()
- (instancetype)initWithStartDate:(NSDate *)startDate endDate:(nullable NSDate *)endDate results:(NSArray *)results NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKUsageBucket
- (instancetype)initWithStartDate:(NSDate *)startDate endDate:(nullable NSDate *)endDate results:(NSArray *)results
{
	self = [super init];
	if (self != nil) {
		_startDate = [startDate copy];
		_endDate = [endDate copy];
		_results = [results copy];
	}
	return self;
}
@end

@interface NFKCostEntry ()
- (instancetype)initWithStartDate:(NSDate *)startDate endDate:(nullable NSDate *)endDate amount:(NSDecimalNumber *)amount
						 currency:(NSString *)currency lineItem:(nullable NSString *)lineItem raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKCostEntry
- (instancetype)initWithStartDate:(NSDate *)startDate endDate:(nullable NSDate *)endDate amount:(NSDecimalNumber *)amount
						 currency:(NSString *)currency lineItem:(nullable NSString *)lineItem raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_startDate = [startDate copy];
		_endDate = [endDate copy];
		_amount = [amount copy];
		_currency = [currency.uppercaseString copy];
		_lineItem = [lineItem copy];
		_raw = [raw copy];
	}
	return self;
}
@end

@interface NFKAccountBalance ()
- (instancetype)initWithAmount:(NSDecimalNumber *)amount currency:(NSString *)currency raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKAccountBalance
- (instancetype)initWithAmount:(NSDecimalNumber *)amount currency:(NSString *)currency raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_amount = [amount copy];
		_currency = [currency.uppercaseString copy];
		_raw = [raw copy];
	}
	return self;
}
@end

@implementation NFKRemoteUsageReporter

@synthesize session = _session;

+ (nullable instancetype)reporterForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NSDictionary<NSString *, NSArray *> *routes = @{
		@"anthropic": @[ @(NFKRemoteUsageAPIStyleAnthropic), @"https://api.anthropic.com/v1" ],
		@"openai": @[ @(NFKRemoteUsageAPIStyleOpenAI), @"https://api.openai.com/v1" ],
		@"xai": @[ @(NFKRemoteUsageAPIStyleXAI), @"https://management-api.x.ai/v1" ],
		@"openrouter": @[ @(NFKRemoteUsageAPIStyleOpenRouter), @"https://openrouter.ai/api/v1" ],
		@"deepseek": @[ @(NFKRemoteUsageAPIStyleDeepSeek), @"https://api.deepseek.com" ],
		@"mistral": @[ @(NFKRemoteUsageAPIStyleMistral), @"https://api.mistral.ai/v1/admin" ],
	};
	NSArray *route = routes[provider.identifier];
	if (route == nil) {
		return nil;
	}
	NFKRemoteUsageReporter *reporter = [[self alloc] init];
	reporter.apiStyle = [route[0] integerValue];
	reporter.endpointURL = [NSURL URLWithString:route[1]];
	reporter.apiKey = apiKey;
	return reporter;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 60.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

#pragma mark Usage

- (nullable NSArray<NFKUsageBucket *> *)usageFromDate:(NSDate *)startDate
											   toDate:(nullable NSDate *)endDate
										  bucketWidth:(nullable NSString *)bucketWidth
											  groupBy:(nullable NSArray<NSString *> *)groupBy
											   report:(nullable NSString *)report
												error:(NSError * _Nullable *)outError
{
	switch (self.apiStyle) {
		case NFKRemoteUsageAPIStyleAnthropic:
			if ([report isEqualToString:@"claude_code"]) {
				return [self anthropicClaudeCodeOn:startDate error:outError];
			}
			return [self bucketsFrom:[self pagedPath:@"organizations/usage_report/messages"
											   query:[self anthropicQueryFrom:startDate to:endDate width:bucketWidth groupBy:groupBy]
											   error:outError]];
		case NFKRemoteUsageAPIStyleOpenAI:
			return [self bucketsFrom:[self pagedPath:[@"organization/usage/" stringByAppendingString:report ?: @"completions"]
											   query:[self openAIQueryFrom:startDate to:endDate width:bucketWidth groupBy:groupBy]
											   error:outError]];
		case NFKRemoteUsageAPIStyleXAI: {
			NSArray<NSDictionary *> *rows = [self xAIRowsFrom:startDate to:endDate groupBy:groupBy error:outError];
			return rows != nil ? [self dailyBucketsOfRows:rows dateKey:@"timestamp"] : nil;
		}
		case NFKRemoteUsageAPIStyleOpenRouter: {
			NSArray<NSDictionary *> *rows = [self openRouterActivityFrom:startDate to:endDate error:outError];
			return rows != nil ? [self dailyBucketsOfRows:rows dateKey:@"date"] : nil;
		}
		case NFKRemoteUsageAPIStyleMistral: {
			NSDateComponents *month = [[NSCalendar calendarWithIdentifier:NSCalendarIdentifierGregorian]
									   componentsInTimeZone:[NSTimeZone timeZoneForSecondsFromGMT:0] fromDate:startDate];
			NSDictionary *reply = [self JSONAt:@"usage" query:@[ [NSURLQueryItem queryItemWithName:@"month" value:@(month.month).stringValue],
																  [NSURLQueryItem queryItemWithName:@"year" value:@(month.year).stringValue] ]
										method:@"GET" body:nil error:outError];
			return reply != nil ? @[ [[NFKUsageBucket alloc] initWithStartDate:NFKUsageDate(reply[@"start_date"]) ?: startDate
																	  endDate:NFKUsageDate(reply[@"end_date"]) results:@[ reply ]] ] : nil;
		}
		case NFKRemoteUsageAPIStyleDeepSeek:
			return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"DeepSeek reports a balance but no usage"];
	}
	return nil;
}

- (nullable NSArray<NFKCostEntry *> *)costsFromDate:(NSDate *)startDate
											 toDate:(nullable NSDate *)endDate
											groupBy:(nullable NSArray<NSString *> *)groupBy
											  error:(NSError * _Nullable *)outError
{
	NSMutableArray<NFKCostEntry *> *entries = [NSMutableArray array];
	switch (self.apiStyle) {
		case NFKRemoteUsageAPIStyleAnthropic:
		case NFKRemoteUsageAPIStyleOpenAI: {
			BOOL anthropic = self.apiStyle == NFKRemoteUsageAPIStyleAnthropic;
			NSArray<NSURLQueryItem *> *query = anthropic ? [self anthropicQueryFrom:startDate to:endDate width:nil groupBy:groupBy]
														  : [self openAIQueryFrom:startDate to:endDate width:nil groupBy:groupBy];
			NSArray<NSDictionary *> *buckets = [self pagedPath:anthropic ? @"organizations/cost_report" : @"organization/costs" query:query error:outError];
			if (buckets == nil) {
				return nil;
			}
			for (NSDictionary *bucket in buckets) {
				NSDate *start = NFKUsageDate(bucket[@"starting_at"] ?: bucket[@"start_time"]) ?: startDate;
				NSDate *end = NFKUsageDate(bucket[@"ending_at"] ?: bucket[@"end_time"]);
				for (NSDictionary *result in NFKUsageArray(bucket[@"results"])) {
					NSDictionary *row = NFKUsageDictionary(result);
					// Anthropic writes cents as a string; OpenAI writes dollars as a number under amount.value.
					NSDecimalNumber *amount = anthropic
						? [NFKUsageDecimal(row[@"amount"]) decimalNumberByDividingBy:[NSDecimalNumber decimalNumberWithString:@"100"]]
						: NFKUsageDecimal(NFKUsageDictionary(row[@"amount"])[@"value"]);
					NSString *currency = anthropic ? row[@"currency"] : NFKUsageDictionary(row[@"amount"])[@"currency"];
					NSString *item = row[@"description"] ?: row[@"line_item"] ?: row[@"model"];
					[entries addObject:[[NFKCostEntry alloc] initWithStartDate:start endDate:end amount:amount
																	 currency:[currency isKindOfClass:NSString.class] ? currency : @"USD"
																	 lineItem:[item isKindOfClass:NSString.class] ? item : nil raw:row]];
				}
			}
			return entries;
		}
		case NFKRemoteUsageAPIStyleXAI:
		case NFKRemoteUsageAPIStyleOpenRouter: {
			BOOL xAI = self.apiStyle == NFKRemoteUsageAPIStyleXAI;
			NSArray<NSDictionary *> *rows = xAI ? [self xAIRowsFrom:startDate to:endDate groupBy:groupBy error:outError]
												: [self openRouterActivityFrom:startDate to:endDate error:outError];
			if (rows == nil) {
				return nil;
			}
			for (NSDictionary *row in rows) {
				NSDate *start = NFKUsageDate(row[xAI ? @"timestamp" : @"date"]) ?: startDate;
				NSString *item = xAI ? row[@"group"] : row[@"model"];
				[entries addObject:[[NFKCostEntry alloc] initWithStartDate:start endDate:nil amount:NFKUsageDecimal(row[xAI ? @"usd" : @"usage"])
																 currency:@"USD" lineItem:[item isKindOfClass:NSString.class] ? item : nil raw:row]];
			}
			return entries;
		}
		case NFKRemoteUsageAPIStyleDeepSeek:
		case NFKRemoteUsageAPIStyleMistral:
			return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"this service reports no spend over an API"];
	}
	return nil;
}

- (nullable NSArray<NFKAccountBalance *> *)balancesWithError:(NSError * _Nullable *)outError
{
	switch (self.apiStyle) {
		case NFKRemoteUsageAPIStyleOpenRouter: {
			NSDictionary *reply = [self JSONAt:@"credits" query:nil method:@"GET" body:nil error:outError];
			if (reply == nil) {
				return nil;
			}
			NSDictionary *credits = NFKUsageDictionary(reply[@"data"]);
			NSDecimalNumber *left = [NFKUsageDecimal(credits[@"total_credits"]) decimalNumberBySubtracting:NFKUsageDecimal(credits[@"total_usage"])];
			return @[ [[NFKAccountBalance alloc] initWithAmount:left currency:@"USD" raw:reply] ];
		}
		case NFKRemoteUsageAPIStyleDeepSeek: {
			NSDictionary *reply = [self JSONAt:@"user/balance" query:nil method:@"GET" body:nil error:outError];
			if (reply == nil) {
				return nil;
			}
			NSMutableArray<NFKAccountBalance *> *balances = [NSMutableArray array];
			for (NSDictionary *info in NFKUsageArray(reply[@"balance_infos"])) {
				NSDictionary *entry = NFKUsageDictionary(info);
				[balances addObject:[[NFKAccountBalance alloc] initWithAmount:NFKUsageDecimal(entry[@"total_balance"])
																	 currency:[entry[@"currency"] isKindOfClass:NSString.class] ? entry[@"currency"] : @"USD"
																		  raw:entry]];
			}
			return balances;
		}
		case NFKRemoteUsageAPIStyleXAI: {
			if (self.teamIdentifier.length == 0) {
				return [self fail:outError code:kNFKError_InferenceNotReady reason:@"xAI's billing paths name a team: set teamIdentifier"];
			}
			NSString *path = [NSString stringWithFormat:@"billing/teams/%@/prepaid/balance", self.teamIdentifier];
			NSDictionary *reply = [self JSONAt:path query:nil method:@"GET" body:nil error:outError];
			if (reply == nil) {
				return nil;
			}
			// xAI records credit as a negative number of cents.
			NSDecimalNumber *cents = NFKUsageDecimal(NFKUsageDictionary(reply[@"total"])[@"val"]);
			NSDecimalNumber *dollars = [cents decimalNumberByDividingBy:[NSDecimalNumber decimalNumberWithString:@"-100"]];
			return @[ [[NFKAccountBalance alloc] initWithAmount:dollars currency:@"USD" raw:reply] ];
		}
		default:
			return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"this service reports no balance over an API"];
	}
}

#pragma mark Service shapes

// Anthropic's reports take RFC 3339 bounds and repeated group_by[] values.
- (NSArray<NSURLQueryItem *> *)anthropicQueryFrom:(NSDate *)start to:(nullable NSDate *)end width:(nullable NSString *)width groupBy:(nullable NSArray<NSString *> *)groupBy
{
	NSMutableArray<NSURLQueryItem *> *query = [NSMutableArray arrayWithObject:[NSURLQueryItem queryItemWithName:@"starting_at" value:[NFKUsageTimestampFormatter() stringFromDate:start]]];
	if (end != nil) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"ending_at" value:[NFKUsageTimestampFormatter() stringFromDate:end]]];
	}
	if (width != nil) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"bucket_width" value:width]];
	}
	for (NSString *group in groupBy) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"group_by[]" value:group]];
	}
	return query;
}

// OpenAI's take Unix-second bounds and repeated group_by values.
- (NSArray<NSURLQueryItem *> *)openAIQueryFrom:(NSDate *)start to:(nullable NSDate *)end width:(nullable NSString *)width groupBy:(nullable NSArray<NSString *> *)groupBy
{
	NSMutableArray<NSURLQueryItem *> *query = [NSMutableArray arrayWithObject:[NSURLQueryItem queryItemWithName:@"start_time" value:@((long long)start.timeIntervalSince1970).stringValue]];
	if (end != nil) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"end_time" value:@((long long)end.timeIntervalSince1970).stringValue]];
	}
	if (width != nil) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"bucket_width" value:width]];
	}
	for (NSString *group in groupBy) {
		[query addObject:[NSURLQueryItem queryItemWithName:@"group_by" value:group]];
	}
	return query;
}

- (nullable NSArray<NFKUsageBucket *> *)anthropicClaudeCodeOn:(NSDate *)day error:(NSError * _Nullable *)outError
{
	NSArray<NSURLQueryItem *> *query = @[ [NSURLQueryItem queryItemWithName:@"starting_at" value:[NFKUsageDayFormatter() stringFromDate:day]] ];
	NSArray<NSDictionary *> *records = [self pagedPath:@"organizations/usage_report/claude_code" query:query error:outError];
	return records != nil ? @[ [[NFKUsageBucket alloc] initWithStartDate:day endDate:[day dateByAddingTimeInterval:86400] results:records] ] : nil;
}

// xAI answers a time series per group; each data point becomes a row {timestamp, group, usd}.
- (nullable NSArray<NSDictionary *> *)xAIRowsFrom:(NSDate *)start to:(nullable NSDate *)end groupBy:(nullable NSArray<NSString *> *)groupBy error:(NSError * _Nullable *)outError
{
	if (self.teamIdentifier.length == 0) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"xAI's billing paths name a team: set teamIdentifier"];
	}
	NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
	formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
	formatter.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
	formatter.dateFormat = @"yyyy-MM-dd HH:mm:ss";
	NSDictionary *body = @{ @"analyticsRequest": @{
		@"timeRange": @{ @"startTime": [formatter stringFromDate:start], @"endTime": [formatter stringFromDate:end ?: [NSDate date]], @"timezone": @"Etc/GMT" },
		@"timeUnit": @"TIME_UNIT_DAY",
		@"values": @[ @{ @"name": @"usd", @"aggregation": @"AGGREGATION_SUM" } ],
		@"groupBy": groupBy ?: @[ @"description" ],
		@"filters": @[] } };
	NSString *path = [NSString stringWithFormat:@"billing/teams/%@/usage", self.teamIdentifier];
	NSDictionary *reply = [self JSONAt:path query:nil method:@"POST" body:body error:outError];
	if (reply == nil) {
		return nil;
	}
	NSMutableArray<NSDictionary *> *rows = [NSMutableArray array];
	for (NSDictionary *series in NFKUsageArray(reply[@"timeSeries"])) {
		NSDictionary *entry = NFKUsageDictionary(series);
		NSString *group = [NFKUsageArray(entry[@"groupLabels"] ?: entry[@"group"]) componentsJoinedByString:@" / "];
		for (NSDictionary *point in NFKUsageArray(entry[@"dataPoints"])) {
			NSDictionary *values = NFKUsageDictionary(point);
			[rows addObject:@{ @"timestamp": values[@"timestamp"] ?: @"", @"group": group,
							   @"usd": NFKUsageArray(values[@"values"]).firstObject ?: @0 }];
		}
	}
	return rows;
}

// OpenRouter reports its last 30 completed days whatever the span; rows outside the span are dropped.
- (nullable NSArray<NSDictionary *> *)openRouterActivityFrom:(NSDate *)start to:(nullable NSDate *)end error:(NSError * _Nullable *)outError
{
	NSDictionary *reply = [self JSONAt:@"activity" query:nil method:@"GET" body:nil error:outError];
	if (reply == nil) {
		return nil;
	}
	NSMutableArray<NSDictionary *> *rows = [NSMutableArray array];
	for (NSDictionary *row in NFKUsageArray(reply[@"data"])) {
		NSDate *day = NFKUsageDate(NFKUsageDictionary(row)[@"date"]);
		if (day != nil && [day timeIntervalSinceDate:start] >= -86400 && (end == nil || [day compare:end] == NSOrderedAscending)) {
			[rows addObject:row];
		}
	}
	return rows;
}

- (NSArray<NFKUsageBucket *> *)dailyBucketsOfRows:(NSArray<NSDictionary *> *)rows dateKey:(NSString *)dateKey
{
	NSMutableDictionary<NSDate *, NSMutableArray *> *byDay = [NSMutableDictionary dictionary];
	for (NSDictionary *row in rows) {
		NSDate *day = NFKUsageDate(row[dateKey]);
		if (day == nil) {
			continue;
		}
		NSMutableArray *list = byDay[day] ?: [NSMutableArray array];
		[list addObject:row];
		byDay[day] = list;
	}
	NSMutableArray<NFKUsageBucket *> *buckets = [NSMutableArray array];
	for (NSDate *day in [byDay.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		[buckets addObject:[[NFKUsageBucket alloc] initWithStartDate:day endDate:[day dateByAddingTimeInterval:86400] results:byDay[day]]];
	}
	return buckets;
}

- (nullable NSArray<NFKUsageBucket *> *)bucketsFrom:(nullable NSArray<NSDictionary *> *)records
{
	if (records == nil) {
		return nil;
	}
	NSMutableArray<NFKUsageBucket *> *buckets = [NSMutableArray array];
	for (NSDictionary *record in records) {
		NSDate *start = NFKUsageDate(record[@"starting_at"] ?: record[@"start_time"]);
		if (start != nil) {
			[buckets addObject:[[NFKUsageBucket alloc] initWithStartDate:start endDate:NFKUsageDate(record[@"ending_at"] ?: record[@"end_time"])
																 results:NFKUsageArray(record[@"results"])]];
		}
	}
	return buckets;
}

#pragma mark Plumbing

// Every page of a report: has_more with next_page, handed back as page.
- (nullable NSArray<NSDictionary *> *)pagedPath:(NSString *)path query:(NSArray<NSURLQueryItem *> *)query error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *records = [NSMutableArray array];
	NSString *page = nil;
	while (YES) {
		NSMutableArray<NSURLQueryItem *> *items = [query mutableCopy];
		if (page != nil) {
			[items addObject:[NSURLQueryItem queryItemWithName:@"page" value:page]];
		}
		NSDictionary *reply = [self JSONAt:path query:items method:@"GET" body:nil error:outError];
		if (reply == nil) {
			return nil;
		}
		for (id record in NFKUsageArray(reply[@"data"])) {
			if ([record isKindOfClass:NSDictionary.class]) {
				[records addObject:record];
			}
		}
		page = [reply[@"has_more"] boolValue] && [reply[@"next_page"] isKindOfClass:NSString.class] ? reply[@"next_page"] : nil;
		if (page == nil) {
			return records;
		}
	}
}

- (nullable NSDictionary *)JSONAt:(NSString *)path query:(nullable NSArray<NSURLQueryItem *> *)query method:(NSString *)method
							 body:(nullable NSDictionary *)body error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:[self.endpointURL URLByAppendingPathComponent:path] resolvingAgainstBaseURL:NO];
	components.queryItems = query.count > 0 ? query : nil;
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:components.URL];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	if (self.apiStyle == NFKRemoteUsageAPIStyleAnthropic) {
		[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleAnthropicMessages];
	} else if (self.apiStyle == NFKRemoteUsageAPIStyleMistral) {
		if (self.apiKey.length > 0) {
			[request setValue:self.apiKey forHTTPHeaderField:@"x-api-key"];
		}
	} else {
		[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	}
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the report call failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (nullable id)fail:(NSError * _Nullable *)outError code:(NFKInferenceError)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
