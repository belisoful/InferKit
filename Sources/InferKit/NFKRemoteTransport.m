//
//  NFKRemoteTransport.m
//  InferKit
//

#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKServerIdentity.h>
#import <InferKit/NFKRemoteProvider.h>
#import "NFKRemoteTransportPrivate.h"
#import <objc/runtime.h>
#import "NFKJobGuard.h"

NSString * const NFKAnthropicAPIVersion = @"2023-06-01";
NSString * const NFKRemoteErrorStatusCodeKey = @"NFKRemoteErrorStatusCode";
NSString * const NFKRemoteErrorBodyKey = @"NFKRemoteErrorBody";
NSString * const NFKRemoteErrorRetryAfterKey = @"NFKRemoteErrorRetryAfter";

static NSUInteger NFKRemoteRetryAttempts = 2;
static NSTimeInterval NFKRemoteMaximumRetryDelay = 8.0;

static NSError *NFKRemoteUnreachableError(NSURL * _Nullable url, NSError * _Nullable underlying);

#pragma mark - Certificate pinning

typedef void (^NFKRemoteChallengeCompletion)(NSURLSessionAuthChallengeDisposition disposition, NSURLCredential * _Nullable credential);

static const void * const NFKRemotePinRefusedKey = &NFKRemotePinRefusedKey;

NSString * _Nullable NFKRemoteNormalizedFingerprint(NSString *value)
{
	NSCharacterSet *separators = [NSCharacterSet characterSetWithCharactersInString:@": "];
	NSString *hex = [[value componentsSeparatedByCharactersInSet:separators] componentsJoinedByString:@""].lowercaseString;
	NSCharacterSet *nonHex = [NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdef"].invertedSet;
	if (hex.length != 64 || [hex rangeOfCharacterFromSet:nonHex].location != NSNotFound) {
		return nil;
	}
	return hex;
}

static NSString * _Nullable NFKRemoteLeafFingerprint(SecTrustRef trust)
{
	SecCertificateRef leaf = NULL;
	NSArray *chain = nil;
	if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
		chain = CFBridgingRelease(SecTrustCopyCertificateChain(trust));
		leaf = chain.count > 0 ? (__bridge SecCertificateRef)chain.firstObject : NULL;
	} else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		leaf = SecTrustGetCertificateCount(trust) > 0 ? SecTrustGetCertificateAtIndex(trust, 0) : NULL;
#pragma clang diagnostic pop
	}
	NSData *certificateData = leaf != NULL ? CFBridgingRelease(SecCertificateCopyData(leaf)) : nil;
	return certificateData != nil ? [NFKServerIdentity SHA256FingerprintOfCertificateData:certificateData] : nil;
}

/*! Whether a task's server presented a certificate other than the pinned one. */
static BOOL NFKRemotePinRefused(NSURLSessionTask * _Nullable task)
{
	return task != nil && [objc_getAssociatedObject(task, NFKRemotePinRefusedKey) boolValue];
}

static NSError *NFKRemotePinMismatchError(NSURL * _Nullable url)
{
	NSString *host = url.host ?: @"the endpoint";
	NSString *reason = [NSString stringWithFormat:@"%@ presented a certificate other than the pinned one", host];
	NSMutableDictionary *underlyingInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
	if (url != nil) {
		underlyingInfo[NSURLErrorFailingURLErrorKey] = url;
		userInfo[NSURLErrorFailingURLErrorKey] = url;
	}
	userInfo[NSUnderlyingErrorKey] = [NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorServerCertificateUntrusted userInfo:underlyingInfo];
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable userInfo:userInfo];
}

/*! A session delegate that trusts only the server certificate with the pinned SHA-256 fingerprint and
	follows only HTTPS redirects. */
@interface NFKRemotePinnedTrust : NSObject <NSURLSessionTaskDelegate>
/*! The fingerprint as the caller gave it. */
@property (nonatomic, copy) NSString *pinnedValue;
/*! The fingerprint as 64 lowercase hexadecimal digits, or nil when pinnedValue is malformed. */
@property (nonatomic, copy, nullable) NSString *fingerprint;
@end

@implementation NFKRemotePinnedTrust

- (void)evaluateChallenge:(NSURLAuthenticationChallenge *)challenge
				  forTask:(NSURLSessionTask *)task
		completionHandler:(NFKRemoteChallengeCompletion)completionHandler
{
	if (![challenge.protectionSpace.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust]) {
		completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
		return;
	}
	SecTrustRef trust = challenge.protectionSpace.serverTrust;
	NSString *presented = trust != NULL ? NFKRemoteLeafFingerprint(trust) : nil;
	if (self.fingerprint != nil && [presented isEqualToString:self.fingerprint]) {
		completionHandler(NSURLSessionAuthChallengeUseCredential, [NSURLCredential credentialForTrust:trust]);
		return;
	}
	objc_setAssociatedObject(task, NFKRemotePinRefusedKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
	completionHandler(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
}

- (void)URLSession:(NSURLSession *)session
			  task:(NSURLSessionTask *)task
didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge
 completionHandler:(NFKRemoteChallengeCompletion)completionHandler
{
	[self evaluateChallenge:challenge forTask:task completionHandler:completionHandler];
}

- (void)URLSession:(NSURLSession *)session
					  task:(NSURLSessionTask *)task
willPerformHTTPRedirection:(NSHTTPURLResponse *)response
				newRequest:(NSURLRequest *)request
		 completionHandler:(void (^)(NSURLRequest * _Nullable))completionHandler
{
	completionHandler([request.URL.scheme.lowercaseString isEqualToString:@"https"] ? request : nil);
}

@end

static NFKRemotePinnedTrust * _Nullable NFKRemotePinOfSession(NSURLSession *session)
{
	id delegate = session.delegate;
	return [delegate isKindOfClass:NFKRemotePinnedTrust.class] ? delegate : nil;
}

NSError * _Nullable NFKRemotePinPreflightError(NSURLSession *session, NSURL * _Nullable url, NSString *secureScheme)
{
	NFKRemotePinnedTrust *pin = NFKRemotePinOfSession(session);
	if (pin == nil) {
		return nil;
	}
	if (pin.fingerprint == nil) {
		NSString *reason = [NSString stringWithFormat:@"the pinned certificate fingerprint \"%@\" is not 64 hexadecimal digits", pin.pinnedValue];
		return [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
	}
	if (![url.scheme.lowercaseString isEqualToString:secureScheme]) {
		NSString *channel = [secureScheme isEqualToString:@"wss"] ? @"secure WebSockets (wss://)" : @"HTTPS (https://)";
		NSString *reason = [NSString stringWithFormat:@"a pinned session sends only over %@, and %@ is not", channel, url.absoluteString ?: @"the URL"];
		return [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
	}
	return nil;
}

#pragma mark - Line stream

/*! Splits a streamed body into lines as the bytes arrive; collects a failing status's body whole. A line
	handler that raises ends the stream, and the completion handler receives the exception as its
	error; a completion handler that raises is called once more with that error, so the job it
	finishes fails instead of waiting. */
@interface NFKRemoteLineStream : NSObject <NSURLSessionDataDelegate>
@property (nonatomic, copy) void (^lineHandler)(NSString *line);
@property (nonatomic, copy) NFKRemoteStreamCompletion completionHandler;
@property (nonatomic, strong) NSMutableData *buffer;
@property (nonatomic, strong, nullable) NSHTTPURLResponse *response;
@property (nonatomic, assign) BOOL collectsWholeBody;
/*! The pin of the session the stream was started on, which the stream's own session applies. */
@property (nonatomic, strong, nullable) NFKRemotePinnedTrust *pin;
/*! The task the stream reads, which a raising line handler cancels. */
@property (nonatomic, weak, nullable) NSURLSessionTask *task;
/*! The error of an exception a line handler raised, which the completion handler receives. */
@property (nonatomic, strong, nullable) NSError *handlerFailure;
@end

@implementation NFKRemoteLineStream

- (void)URLSession:(NSURLSession *)session
			  task:(NSURLSessionTask *)task
didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge
 completionHandler:(NFKRemoteChallengeCompletion)completionHandler
{
	if (self.pin == nil) {
		completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);
		return;
	}
	[self.pin evaluateChallenge:challenge forTask:task completionHandler:completionHandler];
}

- (void)URLSession:(NSURLSession *)session
					  task:(NSURLSessionTask *)task
willPerformHTTPRedirection:(NSHTTPURLResponse *)response
				newRequest:(NSURLRequest *)request
		 completionHandler:(void (^)(NSURLRequest * _Nullable))completionHandler
{
	if (self.pin == nil) {
		completionHandler(request);
		return;
	}
	[self.pin URLSession:session task:task willPerformHTTPRedirection:response newRequest:request completionHandler:completionHandler];
}

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
	if (NFKRemotePinRefused(task)) {
		reported = NFKRemotePinMismatchError(task.originalRequest.URL);
	} else if (error != nil && self.response == nil && error.code != NSURLErrorCancelled) {
		reported = NFKRemoteUnreachableError(task.originalRequest.URL, error);
	}
	if (self.handlerFailure != nil) {
		reported = self.handlerFailure;
		errorBody = nil;
	}
	@try {
		self.completionHandler(self.response, errorBody, reported);
	} @catch (NSException *exception) {
		@try {
			self.completionHandler(nil, nil, NFKJobExceptionError(exception));
		} @catch (NSException *repeated) {
			// The handler raised on its error path too; the process keeps running.
		}
	}
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
	if (self.handlerFailure != nil) {
		return;
	}
	@try {
		self.lineHandler(line);
	} @catch (NSException *exception) {
		self.handlerFailure = NFKJobExceptionError(exception);
		[self.task cancel];
	}
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
	NSError *pinError = NFKRemotePinPreflightError(session, request.URL, @"https");
	if (pinError != nil) {
		if (outResponse != NULL) {
			*outResponse = nil;
		}
		if (outError != NULL) {
			*outError = pinError;
		}
		return nil;
	}
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
		*outError = NFKRemotePinRefused(task) ? NFKRemotePinMismatchError(request.URL) : NFKRemoteUnreachableError(request.URL, resultError);
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
	if (response == nil || (status != 429 && status != 502 && status != 503 && status != 504 && status != 529)) {
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
	NSError *pinError = NFKRemotePinPreflightError(session, request.URL, @"https");
	if (pinError != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			completionHandler(nil, nil, pinError);
		});
		return ^{};
	}
	NFKRemoteLineStream *stream = [[NFKRemoteLineStream alloc] init];
	stream.lineHandler = lineHandler;
	stream.completionHandler = completionHandler;
	stream.pin = NFKRemotePinOfSession(session);
	NSURLSession *streamingSession = [NSURLSession sessionWithConfiguration:session.configuration
																  delegate:stream
															 delegateQueue:nil];
	NSURLSessionDataTask *task = [streamingSession dataTaskWithRequest:request];
	stream.task = task;
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
	NSInteger status = response.statusCode;
	NSString *detail = data.length > 0 ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
	NSString *reason = detail.length > 0
		? [NSString stringWithFormat:@"the endpoint returned HTTP %ld: %@", (long)status, detail]
		: [NSString stringWithFormat:@"the endpoint returned HTTP %ld", (long)status];
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
	userInfo[NSLocalizedDescriptionKey] = reason;
	userInfo[NFKRemoteErrorStatusCodeKey] = @(status);
	if (detail != nil) {
		userInfo[NFKRemoteErrorBodyKey] = detail;
	}
	NFKInferenceError code = [self servedCodeInData:data] ?: [self codeForStatus:status];
	NSDate *retryAfter = code == kNFKError_InferenceRateLimited ? [self retryAfterDateForResponse:response] : nil;
	if (retryAfter != nil) {
		userInfo[NFKRemoteErrorRetryAfterKey] = retryAfter;
	}
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:userInfo];
}

// Back off (429, 402, 529), change the request (400, 413, 422), or fix the configuration (the rest).
+ (NFKInferenceError)codeForStatus:(NSInteger)status
{
	switch (status) {
		case 429: case 402: case 529:
			return kNFKError_InferenceRateLimited;
		case 400: case 413: case 422:
			return kNFKError_InferenceRefused;
		default:
			return kNFKError_InferenceBackendFailure;
	}
}

/*! The code an InferKit server names in its error body, which is exact where a status is a range;
	0 for any other body. */
+ (NFKInferenceError)servedCodeInData:(nullable NSData *)data
{
	id body = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	NSDictionary *error = [body isKindOfClass:NSDictionary.class] && [body[@"error"] isKindOfClass:NSDictionary.class] ? body[@"error"] : nil;
	NSNumber *code = [error[@"inferkit_code"] isKindOfClass:NSNumber.class] ? error[@"inferkit_code"] : nil;
	if (![error[@"inferkit_domain"] isEqual:NFKInferenceErrorDomain] || code.integerValue < kNFKError_InferenceNotReady
		|| code.integerValue > kNFKError_InferenceRateLimited) {
		return 0;
	}
	return (NFKInferenceError)code.integerValue;
}

/*! The Retry-After header as a date: a delay in seconds from now, or an HTTP-date as it stands. */
+ (nullable NSDate *)retryAfterDateForResponse:(NSHTTPURLResponse *)response
{
	NSString *retryAfter = [response valueForHTTPHeaderField:@"Retry-After"];
	if (retryAfter.length == 0) {
		return nil;
	}
	NSScanner *scanner = [NSScanner scannerWithString:retryAfter];
	double seconds = 0;
	if ([scanner scanDouble:&seconds] && scanner.atEnd) {
		return [NSDate dateWithTimeIntervalSinceNow:MAX(seconds, 0)];
	}
	static NSDateFormatter *httpDate;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		httpDate = [[NSDateFormatter alloc] init];
		httpDate.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
		httpDate.timeZone = [NSTimeZone timeZoneWithAbbreviation:@"GMT"];
		httpDate.dateFormat = @"EEE, dd MMM yyyy HH:mm:ss zzz";
	});
	return [httpDate dateFromString:retryAfter];
}

+ (NSError *)errorWithCode:(NFKInferenceError)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain
							   code:code
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

#pragma mark Pinned sessions

+ (NSURLSession *)sessionWithPinnedCertificateSHA256:(NSString *)fingerprint
{
	static NSMutableDictionary<NSString *, NSURLSession *> *sessions;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		sessions = [NSMutableDictionary dictionary];
	});
	NSString *key = NFKRemoteNormalizedFingerprint(fingerprint) ?: fingerprint;
	@synchronized (sessions) {
		NSURLSession *session = sessions[key];
		if (session == nil) {
			session = [self sessionWithPinnedCertificateSHA256:fingerprint configuration:NSURLSessionConfiguration.defaultSessionConfiguration];
			sessions[key] = session;
		}
		return session;
	}
}

+ (NSURLSession *)sessionWithPinnedCertificateSHA256:(NSString *)fingerprint
									   configuration:(NSURLSessionConfiguration *)configuration
{
	NFKRemotePinnedTrust *pin = [[NFKRemotePinnedTrust alloc] init];
	pin.pinnedValue = fingerprint;
	pin.fingerprint = NFKRemoteNormalizedFingerprint(fingerprint);
	return [NSURLSession sessionWithConfiguration:configuration delegate:pin delegateQueue:nil];
}

+ (nullable NSString *)pinnedCertificateSHA256ForSession:(NSURLSession *)session
{
	NFKRemotePinnedTrust *pin = NFKRemotePinOfSession(session);
	return pin.fingerprint ?: pin.pinnedValue;
}

@end

NSURLSession * _Nullable NFKRemotePinnedSessionForProvider(NFKRemoteProvider *provider)
{
	NSString *fingerprint = provider.pinnedCertificateSHA256;
	return fingerprint != nil ? [NFKRemoteTransport sessionWithPinnedCertificateSHA256:fingerprint] : nil;
}
