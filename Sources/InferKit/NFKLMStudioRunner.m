//
//  NFKLMStudioRunner.m
//  InferKit
//

#import <InferKit/NFKLMStudioRunner.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import "NFKLocalRunnerSupport.h"

@interface NFKLMStudioRunner ()
@property (nonatomic, copy, readwrite) NFKRemoteProvider *provider;
@end

@implementation NFKLMStudioRunner

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
	[self sendRequest:[self requestForPath:@"api/v0/models"] response:&response error:NULL];
	return response != nil;
}

- (nullable NSArray<NFKRemoteModel *> *)installedModelsWithError:(NSError * _Nullable *)outError
{
	NSDictionary *body = [self JSONObjectForPath:@"api/v0/models" error:outError];
	if (body == nil) {
		return nil;
	}
	NSArray *entries = body[@"data"];
	if (![entries isKindOfClass:NSArray.class]) {
		if (outError != NULL) {
			*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
												   reason:@"the model list carries no data array"];
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

- (nullable NSArray<NFKRemoteModel *> *)loadedModelsWithError:(NSError * _Nullable *)outError
{
	NSArray<NFKRemoteModel *> *installed = [self installedModelsWithError:outError];
	if (installed == nil) {
		return nil;
	}
	NSMutableArray<NFKRemoteModel *> *loaded = [NSMutableArray array];
	for (NFKRemoteModel *model in installed) {
		if ([model.raw[@"state"] isEqual:@"loaded"]) {
			[loaded addObject:model];
		}
	}
	return loaded;
}

- (nullable NFKRemoteModel *)detailsForModel:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSMutableCharacterSet *allowed = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[allowed addCharactersInString:@":"];
	NSString *escaped = [identifier stringByAddingPercentEncodingWithAllowedCharacters:allowed] ?: identifier;
	NSDictionary *entry = [self JSONObjectForPath:[@"api/v0/models/" stringByAppendingString:escaped] error:outError];
	if (entry == nil) {
		return nil;
	}
	NFKRemoteModel *model = [NFKRemoteModel modelWithEntry:entry];
	if (model == nil && outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
											   reason:@"the model entry carries no id"];
	}
	return model;
}

#pragma mark Requests

- (NSURLRequest *)requestForPath:(NSString *)path
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:NFKLocalRunnerURL(self.nativeBaseURL, path)];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	return request;
}

- (nullable NSDictionary *)JSONObjectForPath:(NSString *)path error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:[self requestForPath:path] response:&response error:&error];
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

@end
