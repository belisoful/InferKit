//
//  NFKRemoteBackend.m
//  InferKit
//

#import "NFKRemoteBackend.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"

NSString * const NFKRemoteBackendPromptKey	= @"prompt";
NSString * const NFKRemoteBackendMessagesKey	= @"messages";
NSString * const NFKRemoteBackendTextKey		= @"text";
NSString * const NFKRemoteBackendRawKey		= @"raw";

@implementation NFKRemoteBackend

@synthesize endpointURL = _endpointURL;
@synthesize apiKey = _apiKey;
@synthesize modelName = _modelName;
@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return NARC_AUTORELEASE(backend);
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 60.0;
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_endpointURL);
	NARC_RELEASE(_apiKey);
	NARC_RELEASE(_modelName);
	NARC_RELEASE(_session);
	SUPER_DEALLOC();
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = NARC_RETAIN([NSURLSession sharedSession]);
	}
	return _session;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
		return nil;
	}

	NSError *error = nil;
	NSDictionary<NSString *, id> *body = [self requestBodyForRequest:request];
	NSData *bodyData = [NSJSONSerialization dataWithJSONObject:body options:0 error:&error];
	if (bodyData == nil) {
		[self propagateError:error to:outError];
		return nil;
	}

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = bodyData;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (self.apiKey.length > 0) {
		[urlRequest setValue:[@"Bearer " stringByAppendingString:self.apiKey] forHTTPHeaderField:@"Authorization"];
	}

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *responseData = [self sendRequest:urlRequest response:&response error:&sendError];
	if (responseData == nil) {
		[self propagateError:sendError to:outError];
		return nil;
	}
	if (response != nil && (response.statusCode < 200 || response.statusCode >= 300)) {
		NSString *reason = [NSString stringWithFormat:@"the endpoint returned HTTP %ld", (long)response.statusCode];
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}
	return [self resultFromResponseData:responseData error:outError];
}

#pragma mark Body and response

- (NSDictionary<NSString *, id> *)requestBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}

	id messages = [request inputForKey:NFKRemoteBackendMessagesKey];
	if ([messages isKindOfClass:NSArray.class]) {
		body[@"messages"] = messages;
	} else {
		id prompt = [request inputForKey:NFKRemoteBackendPromptKey];
		NSString *text = [prompt isKindOfClass:NSString.class] ? prompt : @"";
		body[@"messages"] = @[ @{ @"role": @"user", @"content": text } ];
	}

	// Parameters fold into the body so a caller sets temperature, max_tokens, and similar.
	for (NSString *key in request.parameters) {
		body[key] = request.parameters[key];
	}
	return body;
}

- (nullable NFKInferenceResult *)resultFromResponseData:(NSData *)data error:(NSError * _Nullable *)outError
{
	NSError *error = nil;
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
		return nil;
	}
	NSDictionary *responseBody = object;

	NSString *content = nil;
	NSArray *choices = responseBody[@"choices"];
	if ([choices isKindOfClass:NSArray.class] && choices.count > 0) {
		NSDictionary *choice = choices.firstObject;
		if ([choice isKindOfClass:NSDictionary.class]) {
			NSDictionary *message = choice[@"message"];
			if ([message isKindOfClass:NSDictionary.class] && [message[@"content"] isKindOfClass:NSString.class]) {
				content = message[@"content"];
			} else if ([choice[@"text"] isKindOfClass:NSString.class]) {
				content = choice[@"text"];
			}
		}
	}

	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	if (content != nil) {
		outputs[NFKRemoteBackendTextKey] = content;
	}
	outputs[NFKRemoteBackendRawKey] = responseBody;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError
{
	__block NSData *resultData = nil;
	__block NSURLResponse *resultResponse = nil;
	__block NSError *resultError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request
												completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		resultData = NARC_RETAIN(data);
		resultResponse = NARC_RETAIN(response);
		resultError = NARC_RETAIN(error);
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (outResponse != NULL && [resultResponse isKindOfClass:NSHTTPURLResponse.class]) {
		*outResponse = (NSHTTPURLResponse *)resultResponse;
	}
	if (resultData == nil && outError != NULL) {
		*outError = resultError != nil ? resultError
									  : [NSError errorWithDomain:NFKInferenceErrorDomain
															code:kNFKError_InferenceBackendFailure
														userInfo:@{ NSLocalizedDescriptionKey: @"the request returned no data" }];
	}
	return resultData;
}

#pragma mark Errors

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	*outError = error != nil ? error
							: [NSError errorWithDomain:NFKInferenceErrorDomain
												  code:kNFKError_InferenceBackendFailure
											  userInfo:@{ NSLocalizedDescriptionKey: @"the remote call failed" }];
	return NO;
}

@end
