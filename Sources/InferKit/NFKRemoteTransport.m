//
//  NFKRemoteTransport.m
//  InferKit
//

#import <InferKit/NFKRemoteTransport.h>

NSString * const NFKAnthropicAPIVersion = @"2023-06-01";

static NSUInteger NFKRemoteRetryAttempts = 2;
static NSTimeInterval NFKRemoteMaximumRetryDelay = 8.0;

static NSError *NFKRemoteUnreachableError(NSURL * _Nullable url, NSError * _Nullable underlying);

#pragma mark - Line stream

/*! Splits a streamed body into lines as the bytes arrive; collects a failing status's body whole. */
@interface NFKRemoteLineStream : NSObject <NSURLSessionDataDelegate>
@property (nonatomic, copy) void (^lineHandler)(NSString *line);
@property (nonatomic, copy) NFKRemoteStreamCompletion completionHandler;
@property (nonatomic, strong) NSMutableData *buffer;
@property (nonatomic, strong, nullable) NSHTTPURLResponse *response;
@property (nonatomic, assign) BOOL collectsWholeBody;
@end

@implementation NFKRemoteLineStream

- (void)URLSession:(NSURLSession *)session
		  dataTask:(NSURLSessionDataTask *)dataTask
didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler
{
	self.response = [response isKindOfClass:NSHTTPURLResponse.class] ? (NSHTTPURLResponse *)response : nil;
	NSInteger status = self.response.statusCode;
	self.collectsWholeBody = self.response == nil || status < 200 || status >= 300;
	completionHandler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveData:(NSData *)data
{
	if (self.buffer == nil) {
		self.buffer = [NSMutableData data];
	}
	[self.buffer appendData:data];
	if (!self.collectsWholeBody) {
		[self drainCompleteLines];
	}
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(nullable NSError *)error
{
	NSData *errorBody = nil;
	if (self.collectsWholeBody) {
		errorBody = self.buffer;
	} else if (self.buffer.length > 0) {
		// The last line may arrive without a trailing newline.
		[self deliverLine:self.buffer];
	}
	self.buffer = nil;
	NSError *reported = error;
	if (error != nil && self.response == nil && error.code != NSURLErrorCancelled) {
		reported = NFKRemoteUnreachableError(task.originalRequest.URL, error);
	}
	self.completionHandler(self.response, errorBody, reported);
	[session finishTasksAndInvalidate];
}

- (void)drainCompleteLines
{
	const uint8_t newline = '\n';
	NSData *separator = [NSData dataWithBytes:&newline length:1];
	NSRange found = [self.buffer rangeOfData:separator options:0 range:NSMakeRange(0, self.buffer.length)];
	while (found.location != NSNotFound) {
		[self deliverLine:[self.buffer subdataWithRange:NSMakeRange(0, found.location)]];
		[self.buffer replaceBytesInRange:NSMakeRange(0, found.location + 1) withBytes:NULL length:0];
		found = [self.buffer rangeOfData:separator options:0 range:NSMakeRange(0, self.buffer.length)];
	}
}

- (void)deliverLine:(NSData *)bytes
{
	NSString *line = [[NSString alloc] initWithData:bytes encoding:NSUTF8StringEncoding];
	if (line == nil) {
		return;
	}
	if ([line hasSuffix:@"\r"]) {
		line = [line substringToIndex:line.length - 1];
	}
	self.lineHandler(line);
}

@end

#pragma mark - Errors

static NSError *NFKRemoteUnreachableError(NSURL * _Nullable url, NSError * _Nullable underlying)
{
	NSString *host = url.host ?: url.absoluteString ?: @"the endpoint";
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
	userInfo[NSLocalizedDescriptionKey] = [NSString stringWithFormat:@"%@ did not answer", host];
	if (underlying != nil) {
		userInfo[NSUnderlyingErrorKey] = underlying;
	}
	if (url != nil) {
		userInfo[NSURLErrorFailingURLErrorKey] = url;
	}
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable userInfo:userInfo];
}

#pragma mark - Transport

@implementation NFKRemoteTransport

+ (NSUInteger)retryAttempts
{
	return NFKRemoteRetryAttempts;
}

+ (void)setRetryAttempts:(NSUInteger)retryAttempts
{
	NFKRemoteRetryAttempts = retryAttempts;
}

+ (NSTimeInterval)maximumRetryDelay
{
	return NFKRemoteMaximumRetryDelay;
}

+ (void)setMaximumRetryDelay:(NSTimeInterval)maximumRetryDelay
{
	NFKRemoteMaximumRetryDelay = maximumRetryDelay;
}

+ (nullable NSData *)sendRequest:(NSURLRequest *)request
						 session:(NSURLSession *)session
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	NSUInteger attempts = self.retryAttempts;
	for (NSUInteger attempt = 0; ; attempt++) {
		NSHTTPURLResponse *response = nil;
		NSError *error = nil;
		NSData *data = [self sendOnce:request session:session response:&response error:&error];
		NSTimeInterval delay = attempt < attempts ? [self retryDelayForResponse:response attempt:attempt] : -1;
		if (delay < 0 || delay > self.maximumRetryDelay) {
			if (outResponse != NULL) {
				*outResponse = response;
			}
			if (data == nil && outError != NULL) {
				*outError = error;
			}
			return data;
		}
		[NSThread sleepForTimeInterval:delay];
	}
}

+ (nullable NSData *)sendOnce:(NSURLRequest *)request
					  session:(NSURLSession *)session
					 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						error:(NSError * _Nullable *)outError
{
	__block NSData *resultData = nil;
	__block NSURLResponse *resultResponse = nil;
	__block NSError *resultError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSURLSessionDataTask *task = [session dataTaskWithRequest:request
											completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		resultData = data;
		resultResponse = response;
		resultError = error;
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	NSHTTPURLResponse *httpResponse = [resultResponse isKindOfClass:NSHTTPURLResponse.class]
		? (NSHTTPURLResponse *)resultResponse : nil;
	if (outResponse != NULL) {
		*outResponse = httpResponse;
	}
	if (resultData != nil || outError == NULL) {
		return resultData;
	}
	if (httpResponse == nil) {
		*outError = NFKRemoteUnreachableError(request.URL, resultError);
	} else {
		*outError = resultError != nil ? resultError
									  : [self errorWithCode:kNFKError_InferenceBackendFailure
													 reason:@"the request returned no data"];
	}
	return nil;
}

// A negative delay means the response is not one to retry.
+ (NSTimeInterval)retryDelayForResponse:(nullable NSHTTPURLResponse *)response attempt:(NSUInteger)attempt
{
	NSInteger status = response.statusCode;
	if (response == nil || (status != 429 && status != 502 && status != 503 && status != 504)) {
		return -1;
	}
	NSString *retryAfter = [response valueForHTTPHeaderField:@"Retry-After"];
	if (retryAfter.length > 0) {
		NSScanner *scanner = [NSScanner scannerWithString:retryAfter];
		double seconds = 0;
		if ([scanner scanDouble:&seconds] && scanner.atEnd) {
			return MAX(seconds, 0);
		}
		// An HTTP-date is the other form; a provider that sends one is asked again on the exponential schedule.
	}
	return 0.5 * pow(2, (double)attempt);
}

+ (void (^)(void))streamRequest:(NSURLRequest *)request
						session:(NSURLSession *)session
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(NFKRemoteStreamCompletion)completionHandler
{
	NFKRemoteLineStream *stream = [[NFKRemoteLineStream alloc] init];
	stream.lineHandler = lineHandler;
	stream.completionHandler = completionHandler;
	NSURLSession *streamingSession = [NSURLSession sessionWithConfiguration:session.configuration
																  delegate:stream
															 delegateQueue:nil];
	NSURLSessionDataTask *task = [streamingSession dataTaskWithRequest:request];
	[task resume];
	return ^{ [task cancel]; };
}

+ (nullable NSString *)SSEDataForLine:(NSString *)line
{
	if (![line hasPrefix:@"data:"]) {
		return nil;
	}
	NSString *payload = [line substringFromIndex:5];
	if ([payload hasPrefix:@" "]) {
		payload = [payload substringFromIndex:1];
	}
	return payload;
}

+ (void)authorizeRequest:(NSMutableURLRequest *)request
				  apiKey:(nullable NSString *)apiKey
				   style:(NFKRemoteAPIStyle)style
{
	if (style == NFKRemoteAPIStyleAnthropicMessages) {
		if (apiKey.length > 0) {
			[request setValue:apiKey forHTTPHeaderField:@"x-api-key"];
		}
		[request setValue:NFKAnthropicAPIVersion forHTTPHeaderField:@"anthropic-version"];
		return;
	}
	if (apiKey.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:apiKey] forHTTPHeaderField:@"Authorization"];
	}
}

+ (nullable NSError *)errorForResponse:(nullable NSHTTPURLResponse *)response
								  data:(nullable NSData *)data
{
	if (response == nil || (response.statusCode >= 200 && response.statusCode < 300)) {
		return nil;
	}
	NSString *detail = data.length > 0 ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
	NSString *reason = detail.length > 0
		? [NSString stringWithFormat:@"the endpoint returned HTTP %ld: %@", (long)response.statusCode, detail]
		: [NSString stringWithFormat:@"the endpoint returned HTTP %ld", (long)response.statusCode];
	return [self errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
}

+ (NSError *)errorWithCode:(NFKInferenceError)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain
							   code:code
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

@end
