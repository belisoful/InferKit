//
//  NFKOllamaRunner.m
//  InferKit
//

#import <InferKit/NFKOllamaRunner.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import "NFKLocalRunnerSupport.h"

/*! A pull streams status lines with gigabytes between them; an hour idle is a dead pull. */
static const NSTimeInterval NFKOllamaPullTimeout = 3600.0;

@interface NFKOllamaRunner ()
@property (nonatomic, copy, readwrite) NFKRemoteProvider *provider;
@end

@implementation NFKOllamaRunner

@synthesize session = _session;

+ (instancetype)runnerWithProvider:(NFKRemoteProvider *)provider
{
	return [[self alloc] initWithProvider:provider];
}

- (instancetype)initWithProvider:(NFKRemoteProvider *)provider
{
	self = [super init];
	if (self != nil) {
		_provider = [provider copy];
		_timeout = 30.0;
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

- (NSURL *)nativeBaseURL
{
	return NFKLocalRunnerNativeBase(self.provider.baseURL);
}

#pragma mark Reading

- (BOOL)isRunning
{
	NSHTTPURLResponse *response = nil;
	[self sendRequest:[self requestForPath:@"" method:@"GET" body:nil] response:&response error:NULL];
	return response != nil;
}

- (nullable NSString *)versionWithError:(NSError * _Nullable *)outError
{
	NSDictionary *body = [self JSONObjectForPath:@"api/version" method:@"GET" body:nil error:outError];
	NSString *version = body[@"version"];
	return [version isKindOfClass:NSString.class] ? version : nil;
}

- (nullable NSArray<NFKRemoteModel *> *)installedModelsWithError:(NSError * _Nullable *)outError
{
	return [self modelsAtPath:@"api/tags" error:outError];
}

- (nullable NSArray<NFKRemoteModel *> *)loadedModelsWithError:(NSError * _Nullable *)outError
{
	return [self modelsAtPath:@"api/ps" error:outError];
}

- (nullable NSArray<NFKRemoteModel *> *)modelsAtPath:(NSString *)path error:(NSError * _Nullable *)outError
{
	NSDictionary *body = [self JSONObjectForPath:path method:@"GET" body:nil error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray *entries = body[@"models"];
	if (![entries isKindOfClass:NSArray.class]) {
		if (outError != NULL) {
			*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
												   reason:@"the model list carries no models array"];
		}
		return nil;
	}
	NSMutableArray<NFKRemoteModel *> *models = [NSMutableArray array];
	for (id entry in entries) {
		NFKRemoteModel *model = [NFKRemoteModel modelWithEntry:entry];
		if (model != nil) {
			[models addObject:model];
		}
	}
	return models;
}

// /api/show carries no identifier and keys the context length by architecture inside model_info
// ("llama.context_length"), so both are lifted to where NFKRemoteModel reads them; the rest of
// the body passes through under raw.
- (nullable NFKRemoteModel *)detailsForModel:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSDictionary *body = [self JSONObjectForPath:@"api/show" method:@"POST" body:@{ @"model": identifier } error:outError];
	if (body == nil) {
		return nil;
	}
	NSMutableDictionary *entry = [body mutableCopy];
	entry[@"id"] = identifier;
	NSDictionary *info = [body[@"model_info"] isKindOfClass:NSDictionary.class] ? body[@"model_info"] : @{};
	for (NSString *key in info) {
		if ([key hasSuffix:@".context_length"] && [info[key] isKindOfClass:NSNumber.class]) {
			entry[@"context_length"] = info[key];
			break;
		}
	}
	return [NFKRemoteModel modelWithEntry:entry];
}

#pragma mark Changing the machine

- (NFKInferenceJob *)pullModel:(NSString *)identifier
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job reportProgress:-1.0];
	NSMutableURLRequest *request = [[self requestForPath:@"api/pull" method:@"POST"
													body:@{ @"model": identifier, @"stream": @YES }] mutableCopy];
	request.timeoutInterval = NFKOllamaPullTimeout;
	__block BOOL finished = NO;

	[self streamRequest:request lineHandler:^(NSDictionary<NSString *, id> *line) {
		if (finished) {
			return;
		}
		NSString *failure = line[@"error"];
		if ([failure isKindOfClass:NSString.class]) {
			finished = YES;
			[job finishWithError:[NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:failure]];
			return;
		}
		NSString *status = [line[@"status"] isKindOfClass:NSString.class] ? line[@"status"] : @"";
		if ([status isEqualToString:@"success"]) {
			finished = YES;
			[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: identifier }]];
			return;
		}
		NSNumber *total = line[@"total"];
		NSNumber *completed = line[@"completed"];
		double progress = -1.0;
		if ([total isKindOfClass:NSNumber.class] && [completed isKindOfClass:NSNumber.class] && total.doubleValue > 0) {
			progress = completed.doubleValue / total.doubleValue;
		}
		[job reportProgress:progress partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: status }]];
	} completionHandler:^(NSError * _Nullable error) {
		if (finished) {
			return;
		}
		finished = YES;
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
																 reason:@"the pull ended without reporting success"]];
	} cancellation:^(void (^cancel)(void)) {
		job.cancellationHandler = cancel;
	}];
	return job;
}

- (BOOL)deleteModel:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:[self requestForPath:@"api/delete" method:@"DELETE" body:@{ @"model": identifier }]
							response:&response error:&error];
	if (data == nil) {
		if (outError != NULL) { *outError = error; }
		return NO;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return NO;
	}
	return YES;
}

#pragma mark Requests

- (NSURLRequest *)requestForPath:(NSString *)path method:(NSString *)method body:(nullable NSDictionary *)body
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:NFKLocalRunnerURL(self.nativeBaseURL, path)];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	return request;
}

- (nullable NSDictionary *)JSONObjectForPath:(NSString *)path
									  method:(NSString *)method
										body:(nullable NSDictionary *)body
									   error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:[self requestForPath:path method:method body:body] response:&response error:&error];
	if (data == nil) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		if (outError != NULL) {
			*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
												   reason:@"the reply is not a JSON object"];
		}
		return nil;
	}
	return object;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

// Each line of the pull is one JSON object; a failing status arrives whole and becomes the error.
- (void)streamRequest:(NSURLRequest *)request
		  lineHandler:(void (^)(NSDictionary<NSString *, id> *))lineHandler
	completionHandler:(void (^)(NSError * _Nullable))completionHandler
		 cancellation:(void (^)(void (^)(void)))cancellation
{
	void (^cancel)(void) = [NFKRemoteTransport streamRequest:request session:self.session lineHandler:^(NSString *line) {
		id object = [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if ([object isKindOfClass:NSDictionary.class]) {
			lineHandler(object);
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable error) {
		completionHandler(error ?: [NFKRemoteTransport errorForResponse:response data:errorBody]);
	}];
	cancellation(cancel);
}

@end

#pragma mark - Native URLs

NSURL *NFKLocalRunnerNativeBase(NSURL *baseURL)
{
	NSCharacterSet *slash = [NSCharacterSet characterSetWithCharactersInString:@"/"];
	NSString *base = [baseURL.absoluteString stringByTrimmingCharactersInSet:slash];
	if ([base hasSuffix:@"/v1"]) {
		base = [base substringToIndex:base.length - 3];
	}
	return [NSURL URLWithString:base];
}

NSURL *NFKLocalRunnerURL(NSURL *nativeBase, NSString *path)
{
	NSCharacterSet *slash = [NSCharacterSet characterSetWithCharactersInString:@"/"];
	NSString *base = [nativeBase.absoluteString stringByTrimmingCharactersInSet:slash];
	NSString *relative = [path stringByTrimmingCharactersInSet:slash];
	return [NSURL URLWithString:relative.length > 0 ? [NSString stringWithFormat:@"%@/%@", base, relative]
													: [base stringByAppendingString:@"/"]];
}
