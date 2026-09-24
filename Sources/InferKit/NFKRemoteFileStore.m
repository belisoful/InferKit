//
//  NFKRemoteFileStore.m
//  InferKit
//

#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! A date from Unix seconds or an RFC 3339 string, the two forms the services write. */
static NSDate * _Nullable NFKRemoteFileDate(id value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return [NSDate dateWithTimeIntervalSince1970:[value doubleValue]];
	}
	if (![value isKindOfClass:NSString.class]) {
		return nil;
	}
	static NSISO8601DateFormatter *plain;
	static NSISO8601DateFormatter *fractional;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		plain = [[NSISO8601DateFormatter alloc] init];
		fractional = [[NSISO8601DateFormatter alloc] init];
		fractional.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
	});
	return [plain dateFromString:value] ?: [fractional dateFromString:value];
}

static NSString * _Nullable NFKRemoteFileString(id value)
{
	return [value isKindOfClass:NSString.class] ? value : nil;
}

@interface NFKRemoteFile ()
+ (nullable instancetype)fileFromRecord:(NSDictionary *)record apiStyle:(NFKRemoteFileStoreAPIStyle)apiStyle;
@end

@implementation NFKRemoteFile

+ (instancetype)fileWithIdentifier:(NSString *)identifier mimeType:(nullable NSString *)mimeType
{
	return [[self alloc] initWithIdentifier:identifier filename:nil byteCount:0 mimeType:mimeType purpose:nil
								  createdAt:nil expiresAt:nil uri:nil state:nil downloadable:NO raw:@{}];
}

+ (instancetype)fileWithGeminiURI:(NSURL *)uri mimeType:(NSString *)mimeType
{
	NSRange files = [uri.path rangeOfString:@"files/"];
	NSString *identifier = files.location != NSNotFound ? [uri.path substringFromIndex:files.location] : uri.absoluteString;
	return [[self alloc] initWithIdentifier:identifier filename:nil byteCount:0 mimeType:mimeType purpose:nil
								  createdAt:nil expiresAt:nil uri:uri state:nil downloadable:NO raw:@{}];
}

- (instancetype)initWithIdentifier:(NSString *)identifier
						  filename:(nullable NSString *)filename
						 byteCount:(long long)byteCount
						  mimeType:(nullable NSString *)mimeType
						   purpose:(nullable NSString *)purpose
						 createdAt:(nullable NSDate *)createdAt
						 expiresAt:(nullable NSDate *)expiresAt
							   uri:(nullable NSURL *)uri
							 state:(nullable NSString *)state
					  downloadable:(BOOL)downloadable
							   raw:(NSDictionary<NSString *, id> *)raw
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_filename = [filename copy];
		_byteCount = byteCount;
		_mimeType = [mimeType copy];
		_purpose = [purpose copy];
		_createdAt = [createdAt copy];
		_expiresAt = [expiresAt copy];
		_uri = [uri copy];
		_state = [state copy];
		_downloadable = downloadable;
		_raw = [raw copy];
	}
	return self;
}

// The services spell the same facts differently: OpenAI's bytes and unix created_at, Anthropic's
// size_bytes and RFC 3339, Gemini's camelCase name, sizeBytes (a string), and uri.
+ (nullable instancetype)fileFromRecord:(NSDictionary *)record apiStyle:(NFKRemoteFileStoreAPIStyle)apiStyle
{
	NSString *identifier = NFKRemoteFileString(record[@"id"]) ?: NFKRemoteFileString(record[@"name"]);
	if (identifier == nil) {
		return nil;
	}
	id size = record[@"bytes"] ?: record[@"size_bytes"] ?: record[@"sizeBytes"];
	NSString *uri = NFKRemoteFileString(record[@"uri"]);
	BOOL downloadable = apiStyle == NFKRemoteFileStoreAPIStyleOpenAI && ![record[@"downloadable"] isEqual:@NO];
	if (apiStyle == NFKRemoteFileStoreAPIStyleAnthropic || record[@"downloadable"] != nil) {
		downloadable = [record[@"downloadable"] boolValue];
	}
	return [[self alloc] initWithIdentifier:identifier
								   filename:NFKRemoteFileString(record[@"filename"]) ?: NFKRemoteFileString(record[@"displayName"])
								  byteCount:[size respondsToSelector:@selector(longLongValue)] ? [size longLongValue] : 0
								   mimeType:NFKRemoteFileString(record[@"mime_type"]) ?: NFKRemoteFileString(record[@"mimeType"]) ?: NFKRemoteFileString(record[@"mimetype"])
									purpose:NFKRemoteFileString(record[@"purpose"])
								  createdAt:NFKRemoteFileDate(record[@"created_at"] ?: record[@"createTime"])
								  expiresAt:NFKRemoteFileDate(record[@"expires_at"] ?: record[@"expirationTime"])
										uri:uri != nil ? [NSURL URLWithString:uri] : nil
									  state:NFKRemoteFileString(record[@"state"]) ?: NFKRemoteFileString(record[@"processing_status"])
							   downloadable:downloadable
										raw:record];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ %lld bytes>", NSStringFromClass(self.class), self.identifier, self.filename ?: @"", self.byteCount];
}

@end

@implementation NFKRemoteFileStore

@synthesize session = _session;

+ (nullable instancetype)fileStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NSDictionary<NSString *, NSString *> *purposes = @{ @"openai": @"user_data", @"deepseek": @"user_data", @"xai": @"assistants",
														@"mistral": @"ocr", @"groq": @"batch", @"together": @"fine-tune" };
	NSString *identifier = provider.identifier;
	NFKRemoteFileStore *store = [[self alloc] init];
	store.apiKey = apiKey;
	if ([identifier isEqualToString:@"anthropic"]) {
		store.endpointURL = [provider URLForPath:@"files"];
		store.apiStyle = NFKRemoteFileStoreAPIStyleAnthropic;
	} else if ([identifier isEqualToString:@"gemini"]) {
		store.endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com"];
		store.apiStyle = NFKRemoteFileStoreAPIStyleGemini;
	} else if ([@[ @"openai", @"mistral", @"xai", @"deepseek", @"groq", @"together", @"openrouter" ] containsObject:identifier]) {
		// DeepSeek's files live at the API root, beside its /v1 compatibility path.
		store.endpointURL = [identifier isEqualToString:@"deepseek"]
			? [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"files"]
			: [provider URLForPath:@"files"];
		store.defaultPurpose = purposes[identifier];
		store.uploadPathComponent = [identifier isEqualToString:@"together"] ? @"upload" : nil;
	} else {
		return nil;
	}
	return store;
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

#pragma mark Upload

- (nullable NFKRemoteFile *)uploadFileAtURL:(NSURL *)fileURL purpose:(nullable NSString *)purpose error:(NSError * _Nullable *)outError
{
	NSData *data = [NSData dataWithContentsOfURL:fileURL options:NSDataReadingMappedIfSafe error:outError];
	if (data == nil) {
		return nil;
	}
	return [self uploadData:data filename:fileURL.lastPathComponent mimeType:[self mimeTypeForExtension:fileURL.pathExtension]
					purpose:purpose expiresAfter:0 error:outError];
}

- (nullable NFKRemoteFile *)uploadData:(NSData *)data
							  filename:(NSString *)filename
							  mimeType:(nullable NSString *)mimeType
							   purpose:(nullable NSString *)purpose
						  expiresAfter:(NSTimeInterval)expiresAfter
								 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
	}
	NSString *type = mimeType ?: [self mimeTypeForExtension:filename.pathExtension];
	if (self.apiStyle == NFKRemoteFileStoreAPIStyleGemini) {
		return [self geminiUploadData:data filename:filename mimeType:type error:outError];
	}
	NSMutableArray<NSArray<NSString *> *> *fields = [NSMutableArray array];
	NSString *resolvedPurpose = purpose ?: self.defaultPurpose;
	if (self.apiStyle == NFKRemoteFileStoreAPIStyleOpenAI && resolvedPurpose != nil) {
		[fields addObject:@[ @"purpose", resolvedPurpose ]];
	}
	if (self.uploadPathComponent != nil) {
		[fields addObject:@[ @"file_name", filename ]];
	}
	if (expiresAfter > 0) {
		NSString *seconds = [NSString stringWithFormat:@"%lld", (long long)expiresAfter];
		if (self.apiStyle == NFKRemoteFileStoreAPIStyleAnthropic) {
			[fields addObject:@[ @"expires_in_seconds", seconds ]];
		} else {
			[fields addObject:@[ @"expires_after[anchor]", @"created_at" ]];
			[fields addObject:@[ @"expires_after[seconds]", seconds ]];
		}
	}
	NSURL *url = self.uploadPathComponent != nil ? [self.endpointURL URLByAppendingPathComponent:self.uploadPathComponent] : self.endpointURL;
	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSData *dashBoundary = [[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding];
	// The fields go before the file: xAI refuses an expiry that follows it.
	for (NSArray<NSString *> *field in fields) {
		[body appendData:dashBoundary];
		[body appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", field[0], field[1]]
						  dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:dashBoundary];
	[body appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\nContent-Type: %@\r\n\r\n",
					   filename, type] dataUsingEncoding:NSUTF8StringEncoding]];
	[body appendData:data];
	[body appendData:[[NSString stringWithFormat:@"\r\n--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
	NSMutableURLRequest *request = [self requestTo:url method:@"POST"];
	request.HTTPBody = body;
	[request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	NSDictionary *record = [self JSONForRequest:request error:outError];
	return record != nil ? [self fileFromRecord:record error:outError] : nil;
}

// Gemini's resumable upload: a start request names the file and answers with an upload URL, then
// the bytes go to that URL with upload, finalize.
- (nullable NFKRemoteFile *)geminiUploadData:(NSData *)data filename:(NSString *)filename mimeType:(NSString *)mimeType error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *start = [self requestTo:[self.endpointURL URLByAppendingPathComponent:@"upload/v1beta/files"] method:@"POST"];
	[start setValue:@"resumable" forHTTPHeaderField:@"X-Goog-Upload-Protocol"];
	[start setValue:@"start" forHTTPHeaderField:@"X-Goog-Upload-Command"];
	[start setValue:[NSString stringWithFormat:@"%lu", (unsigned long)data.length] forHTTPHeaderField:@"X-Goog-Upload-Header-Content-Length"];
	[start setValue:mimeType forHTTPHeaderField:@"X-Goog-Upload-Header-Content-Type"];
	[start setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	start.HTTPBody = [NSJSONSerialization dataWithJSONObject:@{ @"file": @{ @"display_name": filename } } options:0 error:NULL];
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *reply = [self sendRequest:start response:&response error:&sendError];
	NSError *failure = reply == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:reply];
	NSString *uploadURL = [response valueForHTTPHeaderField:@"X-Goog-Upload-URL"];
	if (failure != nil || uploadURL == nil) {
		return [self fail:outError error:failure reason:@"Gemini answered the upload start without an upload URL"];
	}
	NSMutableURLRequest *upload = [NSMutableURLRequest requestWithURL:[NSURL URLWithString:uploadURL]];
	upload.HTTPMethod = @"POST";
	upload.timeoutInterval = self.timeout;
	upload.HTTPBody = data;
	[upload setValue:@"0" forHTTPHeaderField:@"X-Goog-Upload-Offset"];
	[upload setValue:@"upload, finalize" forHTTPHeaderField:@"X-Goog-Upload-Command"];
	[upload setValue:[NSString stringWithFormat:@"%lu", (unsigned long)data.length] forHTTPHeaderField:@"Content-Length"];
	NSDictionary *record = [self JSONForRequest:upload error:outError];
	NSDictionary *file = [record[@"file"] isKindOfClass:NSDictionary.class] ? record[@"file"] : record;
	return file != nil ? [self fileFromRecord:file error:outError] : nil;
}

#pragma mark Listing and fetching

- (nullable NSArray<NFKRemoteFile *> *)filesWithError:(NSError * _Nullable *)outError
{
	NSMutableArray<NFKRemoteFile *> *files = [NSMutableArray array];
	NSString *cursor = nil;
	do {
		NSURLComponents *components = [NSURLComponents componentsWithURL:[self collectionURL] resolvingAgainstBaseURL:NO];
		NSMutableArray<NSURLQueryItem *> *query = [NSMutableArray array];
		if (cursor != nil) {
			NSString *name = self.apiStyle == NFKRemoteFileStoreAPIStyleGemini ? @"pageToken"
						   : self.apiStyle == NFKRemoteFileStoreAPIStyleAnthropic ? @"page" : @"after";
			[query addObject:[NSURLQueryItem queryItemWithName:name value:cursor]];
		}
		components.queryItems = query.count > 0 ? query : nil;
		NSDictionary *page = [self JSONForRequest:[self requestTo:components.URL method:@"GET"] error:outError];
		if (page == nil) {
			return nil;
		}
		NSArray *records = [page[@"data"] isKindOfClass:NSArray.class] ? page[@"data"]
						 : [page[@"files"] isKindOfClass:NSArray.class] ? page[@"files"] : @[];
		for (NSDictionary *record in records) {
			NFKRemoteFile *file = [record isKindOfClass:NSDictionary.class] ? [NFKRemoteFile fileFromRecord:record apiStyle:self.apiStyle] : nil;
			if (file != nil) {
				[files addObject:file];
			}
		}
		cursor = [self nextCursorIn:page lastFile:files.lastObject];
	} while (cursor != nil);
	return files;
}

// Gemini pages by nextPageToken, Anthropic by next_page, OpenAI's shape by has_more and the last id,
// OpenRouter by cursor.
- (nullable NSString *)nextCursorIn:(NSDictionary *)page lastFile:(nullable NFKRemoteFile *)lastFile
{
	NSString *token = NFKRemoteFileString(page[@"nextPageToken"]) ?: NFKRemoteFileString(page[@"next_page"]) ?: NFKRemoteFileString(page[@"cursor"]);
	if (token.length > 0) {
		return token;
	}
	if ([page[@"has_more"] boolValue]) {
		return NFKRemoteFileString(page[@"last_id"]) ?: lastFile.identifier;
	}
	return nil;
}

- (nullable NFKRemoteFile *)fileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSDictionary *record = [self JSONForRequest:[self requestTo:[self URLForFile:identifier] method:@"GET"] error:outError];
	return record != nil ? [self fileFromRecord:record error:outError] : nil;
}

- (nullable NFKRemoteFile *)fileWhenReadyWithIdentifier:(NSString *)identifier
												timeout:(NSTimeInterval)timeout
												  error:(NSError * _Nullable *)outError
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (YES) {
		NFKRemoteFile *file = [self fileWithIdentifier:identifier error:outError];
		if (file == nil) {
			return nil;
		}
		if ([file.state isEqualToString:@"FAILED"]) {
			return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the service could not process the file"];
		}
		if (![file.state isEqualToString:@"PROCESSING"]) {
			return file;
		}
		if ([deadline timeIntervalSinceNow] <= 0) {
			return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the file was still processing when the wait ended"];
		}
		[NSThread sleepForTimeInterval:1.0];
	}
}

- (nullable NSData *)contentsOfFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteFileStoreAPIStyleGemini) {
		return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"Gemini keeps its files from download"];
	}
	NSMutableURLRequest *request = [self requestTo:[[self URLForFile:identifier] URLByAppendingPathComponent:@"content"] method:@"GET"];
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		return [self fail:outError error:failure reason:@"the file's content could not be fetched"];
	}
	return data;
}

- (BOOL)deleteFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self requestTo:[self URLForFile:identifier] method:@"DELETE"];
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		return [self fail:outError error:failure reason:@"the file could not be deleted"] != nil;
	}
	return YES;
}

- (nullable NSURL *)signedURLForFileWithIdentifier:(NSString *)identifier
									   expiryHours:(NSInteger)expiryHours
											 error:(NSError * _Nullable *)outError
{
	if (![self.endpointURL.host containsString:@"mistral"]) {
		return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"only Mistral signs file URLs"];
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:[[self URLForFile:identifier] URLByAppendingPathComponent:@"url"]
											  resolvingAgainstBaseURL:NO];
	if (expiryHours > 0) {
		components.queryItems = @[ [NSURLQueryItem queryItemWithName:@"expiry" value:[NSString stringWithFormat:@"%ld", (long)expiryHours]] ];
	}
	NSDictionary *reply = [self JSONForRequest:[self requestTo:components.URL method:@"GET"] error:outError];
	NSString *url = NFKRemoteFileString(reply[@"url"]);
	return url != nil ? [NSURL URLWithString:url] : (reply != nil ? [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply carries no url"] : nil);
}

#pragma mark Plumbing

- (NSURL *)collectionURL
{
	return self.apiStyle == NFKRemoteFileStoreAPIStyleGemini ? [self.endpointURL URLByAppendingPathComponent:@"v1beta/files"] : self.endpointURL;
}

// A Gemini file's id is already its path (files/abc) under v1beta; the others append the id.
- (NSURL *)URLForFile:(NSString *)identifier
{
	if (self.apiStyle == NFKRemoteFileStoreAPIStyleGemini) {
		NSString *path = [identifier hasPrefix:@"files/"] ? identifier : [@"files/" stringByAppendingString:identifier];
		return [[self.endpointURL URLByAppendingPathComponent:@"v1beta"] URLByAppendingPathComponent:path];
	}
	return [self.endpointURL URLByAppendingPathComponent:identifier];
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	switch (self.apiStyle) {
		case NFKRemoteFileStoreAPIStyleAnthropic:
			[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleAnthropicMessages];
			break;
		case NFKRemoteFileStoreAPIStyleGemini:
			if (self.apiKey.length > 0) {
				[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
			}
			break;
		case NFKRemoteFileStoreAPIStyleOpenAI:
			[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
			break;
	}
	return request;
}

- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		return [self fail:outError error:failure reason:@"the files call failed"];
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

- (nullable NFKRemoteFile *)fileFromRecord:(NSDictionary *)record error:(NSError * _Nullable *)outError
{
	return [NFKRemoteFile fileFromRecord:record apiStyle:self.apiStyle]
		?: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply names no file"];
}

- (NSString *)mimeTypeForExtension:(NSString *)extension
{
	NSDictionary<NSString *, NSString *> *types = @{ @"pdf": @"application/pdf", @"png": @"image/png", @"jpg": @"image/jpeg",
													 @"jpeg": @"image/jpeg", @"webp": @"image/webp", @"gif": @"image/gif",
													 @"txt": @"text/plain", @"md": @"text/markdown", @"csv": @"text/csv",
													 @"json": @"application/json", @"jsonl": @"application/jsonl",
													 @"mp3": @"audio/mpeg", @"wav": @"audio/wav", @"m4a": @"audio/mp4",
													 @"mp4": @"video/mp4", @"mov": @"video/quicktime", @"webm": @"video/webm" };
	return types[extension.lowercaseString] ?: @"application/octet-stream";
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

- (nullable id)fail:(NSError * _Nullable *)outError error:(nullable NSError *)error reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
	}
	return nil;
}

@end
