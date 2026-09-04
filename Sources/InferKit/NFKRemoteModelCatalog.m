//
//  NFKRemoteModelCatalog.m
//  InferKit
//

#import <InferKit/NFKRemoteModelCatalog.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! Anthropic's page size ceiling; its default is 20, which for its list means several round trips. */
static NSString * const NFKAnthropicPageLimit = @"1000";

@interface NFKRemoteModelCatalog ()
@property (nonatomic, copy, readwrite) NFKRemoteProvider *provider;
@end

@implementation NFKRemoteModelCatalog

@synthesize session = _session;

+ (instancetype)catalogForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NFKRemoteModelCatalog *catalog = [[self alloc] initWithProvider:provider];
	catalog.apiKey = apiKey;
	return catalog;
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

#pragma mark Listing

- (nullable NSArray<NFKRemoteModel *> *)modelsWithError:(NSError * _Nullable *)outError
{
	NSMutableArray<NFKRemoteModel *> *models = [NSMutableArray array];
	NSString *afterIdentifier = nil;
	do {
		NSDictionary *page = [self pageAfter:afterIdentifier error:outError];
		if (page == nil) {
			return nil;
		}
		NSArray *entries = page[@"data"];
		if (![entries isKindOfClass:NSArray.class]) {
			if (outError != NULL) {
				*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
													   reason:@"the model list carries no data array"];
			}
			return nil;
		}
		for (id entry in entries) {
			NFKRemoteModel *model = [NFKRemoteModel modelWithEntry:entry];
			if (model != nil) {
				[models addObject:model];
			}
		}
		afterIdentifier = [self nextPageIdentifierIn:page];
	} while (afterIdentifier != nil);
	return models;
}

- (void)modelsWithCompletionHandler:(void (^)(NSArray<NFKRemoteModel *> * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NSArray<NFKRemoteModel *> *models = [self modelsWithError:&error];
		completionHandler(models, error);
	});
}

- (nullable NFKRemoteModel *)modelWithIdentifier:(NSString *)identifier
										   error:(NSError * _Nullable *)outError
{
	// A colon is legal in a path segment and is how Ollama spells a tag (llama3.2:latest), though
	// Foundation's path set would encode it.
	NSMutableCharacterSet *allowed = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[allowed addCharactersInString:@":"];
	NSString *escaped = [identifier stringByAddingPercentEncodingWithAllowedCharacters:allowed] ?: identifier;
	NSURL *url = [NSURL URLWithString:[self.provider.modelsURL.absoluteString stringByAppendingFormat:@"/%@", escaped]];
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:self.provider.apiStyle];

	NSDictionary *entry = [self JSONObjectForRequest:request error:outError];
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

- (BOOL)isReachableWithError:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	[self sendRequest:[self requestForPageAfter:nil] response:&response error:&error];
	if (response != nil) {
		return YES;
	}
	if (outError != NULL) {
		*outError = error;
	}
	return NO;
}

#pragma mark Pages

- (nullable NSDictionary *)pageAfter:(nullable NSString *)afterIdentifier error:(NSError * _Nullable *)outError
{
	return [self JSONObjectForRequest:[self requestForPageAfter:afterIdentifier] error:outError];
}

- (nullable NSDictionary *)JSONObjectForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:request response:&response error:&error];
	if (data == nil) {
		if (outError != NULL) {
			*outError = error;
		}
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) {
			*outError = statusError;
		}
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		if (outError != NULL) {
			*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
												   reason:@"the model list is not a JSON object"];
		}
		return nil;
	}
	return object;
}

- (NSURLRequest *)requestForPageAfter:(nullable NSString *)afterIdentifier
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[self pageURLAfter:afterIdentifier]];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:self.provider.apiStyle];
	return request;
}

- (NSURL *)pageURLAfter:(nullable NSString *)afterIdentifier
{
	NSURL *base = self.provider.modelsURL;
	if (!self.paginates) {
		return base;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:base resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
	[items addObject:[NSURLQueryItem queryItemWithName:@"limit" value:NFKAnthropicPageLimit]];
	if (afterIdentifier != nil) {
		[items addObject:[NSURLQueryItem queryItemWithName:@"after_id" value:afterIdentifier]];
	}
	components.queryItems = items;
	return components.URL ?: base;
}

- (BOOL)paginates
{
	return self.provider.apiStyle == NFKRemoteAPIStyleAnthropicMessages;
}

// Anthropic's envelope: has_more, and last_id to pass back as after_id.
- (nullable NSString *)nextPageIdentifierIn:(NSDictionary *)page
{
	if (!self.paginates || ![page[@"has_more"] isKindOfClass:NSNumber.class] || ![page[@"has_more"] boolValue]) {
		return nil;
	}
	id last = page[@"last_id"];
	return [last isKindOfClass:NSString.class] && [last length] > 0 ? last : nil;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

@end
