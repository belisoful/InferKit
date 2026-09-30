//
//  NFKRemoteInferKitBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteInferKitBackend.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import "NFKInferenceWireCoding.h"

@interface NFKRemoteInferKitBackend ()
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedInputKeys;
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedParameterKeys;
@end

@implementation NFKRemoteInferKitBackend

@synthesize session = _session;

+ (instancetype)backendWithBaseURL:(NSURL *)baseURL
{
	return [self backendWithEndpointURL:[[baseURL URLByAppendingPathComponent:@"inferkit"] URLByAppendingPathComponent:@"run"]];
}

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteInferKitBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 600.0;
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

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"inferkit-remote";
}

// The key sets are the hosted backend's, so they are declared only once prepareWithError: has read
// them; before that the backend answers as one that declares nothing.
- (BOOL)respondsToSelector:(SEL)selector
{
	if (selector == @selector(supportedInputKeys)) {
		return self.preparedInputKeys != nil;
	}
	if (selector == @selector(supportedParameterKeys)) {
		return self.preparedParameterKeys != nil;
	}
	return [super respondsToSelector:selector];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return self.preparedInputKeys ?: [NSSet set];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return self.preparedParameterKeys ?: [NSSet set];
}

- (BOOL)prepareWithError:(NSError * _Nullable *)outError
{
	NSURL *base = self.endpointURL.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent;
	if (base == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSURL *url = [base URLByAppendingPathComponent:@"models"];
	if (self.modelName.length > 0) {
		url = [url URLByAppendingPathComponent:self.modelName];
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = MIN(self.timeout, 30.0);
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	NSDictionary *entry = [self JSONObjectForRequest:request error:outError];
	if (entry == nil) {
		return NO;
	}
	if (self.modelName.length == 0) {
		NSArray *models = [entry[@"data"] isKindOfClass:NSArray.class] ? entry[@"data"] : @[];
		if (models.count != 1) {
			return [self failWithCode:kNFKError_InferenceNotReady
							   reason:@"the server hosts more than one model; set modelName" error:outError];
		}
		entry = models.firstObject;
	}
	NSDictionary *description = [entry isKindOfClass:NSDictionary.class] && [entry[@"inferkit"] isKindOfClass:NSDictionary.class]
		? entry[@"inferkit"] : @{};
	if ([description[@"inputs"] isKindOfClass:NSArray.class]) {
		self.preparedInputKeys = [NSSet setWithArray:description[@"inputs"]];
	}
	if ([description[@"parameters"] isKindOfClass:NSArray.class]) {
		self.preparedParameterKeys = [NSSet setWithArray:description[@"parameters"]];
	}
	if (![description[@"ready"] boolValue]) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"the served model reports it is not ready" error:outError];
	}
	return YES;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self URLRequestForRequest:request streaming:NO error:outError];
	id reply = urlRequest != nil ? [self JSONObjectForRequest:urlRequest error:outError] : nil;
	if (reply == nil) {
		return nil;
	}
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	return [decoder resultFromJSONObject:[reply isKindOfClass:NSDictionary.class] ? reply[@"result"] : nil error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self URLRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error];
		return job;
	}
	[job reportProgress:-1.0];
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (finished || ![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
		if ([type isEqualToString:@"progress"]) {
			double progress = [event[@"progress"] isKindOfClass:NSNumber.class] ? [event[@"progress"] doubleValue] : -1.0;
			NFKInferenceResult *partial = event[@"partial"] != nil ? [decoder resultFromJSONObject:event[@"partial"] error:NULL] : nil;
			[job reportProgress:progress partialResult:partial];
		} else if ([type isEqualToString:@"result"]) {
			finished = YES;
			NSError *decodeError = nil;
			NFKInferenceResult *result = [decoder resultFromJSONObject:event[@"result"] error:&decodeError];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:decodeError];
			}
		} else if ([type isEqualToString:@"error"]) {
			finished = YES;
			[job finishWithError:[NFKInferenceWireCoding errorFromJSONObject:@{ @"error": event[@"error"] ?: @{} }]
				?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the served run failed"]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		finished = YES;
		NSError *failure = streamError ?: [self errorForResponse:response data:errorBody];
		[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
																   reason:@"the server closed the stream without a result"]];
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Requests

- (nullable NSMutableURLRequest *)URLRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
		return nil;
	}
	NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	id encoded = [encoder JSONObjectForRequest:request error:outError];
	if (encoded == nil) {
		return nil;
	}
	NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:encoded forKey:@"request"];
	body[@"model"] = self.modelName;
	body[@"stream"] = @(streaming);
	NSData *data = [NSJSONSerialization dataWithJSONObject:body options:0 error:outError];
	if (data == nil) {
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.HTTPBody = data;
	urlRequest.timeoutInterval = self.timeout;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

- (nullable id)JSONObjectForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [self errorForResponse:response data:data];
	if (failure != nil) {
		if (outError != NULL) {
			*outError = failure;
		}
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the server's reply is not a JSON object" error:outError];
		return nil;
	}
	return object;
}

/*! The server's own error when its body carries one, domain and code intact, keeping the transport's
	Retry-After date; otherwise the transport's reading of the status. */
- (nullable NSError *)errorForResponse:(nullable NSHTTPURLResponse *)response data:(nullable NSData *)data
{
	NSError *transportError = [NFKRemoteTransport errorForResponse:response data:data];
	if (transportError == nil || data.length == 0) {
		return transportError;
	}
	NSError *served = [NFKInferenceWireCoding errorFromJSONObject:[NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]];
	if (served == nil) {
		return transportError;
	}
	NSMutableDictionary *userInfo = [served.userInfo mutableCopy];
	userInfo[NFKRemoteErrorStatusCodeKey] = transportError.userInfo[NFKRemoteErrorStatusCodeKey];
	userInfo[NFKRemoteErrorRetryAfterKey] = transportError.userInfo[NFKRemoteErrorRetryAfterKey];
	return [NSError errorWithDomain:served.domain code:served.code userInfo:userInfo];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable, NSData * _Nullable, NSError * _Nullable))completionHandler
{
	return [NFKRemoteTransport streamRequest:request session:self.session lineHandler:lineHandler completionHandler:completionHandler];
}

#pragma mark Errors

- (BOOL)failWithCode:(NSInteger)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:(NFKInferenceError)code reason:reason];
	}
	return NO;
}

@end
