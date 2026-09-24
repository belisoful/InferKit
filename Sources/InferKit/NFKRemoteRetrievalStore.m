//
//  NFKRemoteRetrievalStore.m
//  InferKit
//

#import <InferKit/NFKRemoteRetrievalStore.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

static NSString * _Nullable NFKRetrievalString(id value)
{
	return [value isKindOfClass:NSString.class] ? value : nil;
}

static NSDictionary *NFKRetrievalDictionary(id value)
{
	return [value isKindOfClass:NSDictionary.class] ? value : @{};
}

@interface NFKRetrievalStoreRecord ()
- (instancetype)initWithIdentifier:(NSString *)identifier name:(nullable NSString *)name documentCount:(NSInteger)documentCount
							status:(nullable NSString *)status raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKRetrievalStoreRecord
- (instancetype)initWithIdentifier:(NSString *)identifier name:(nullable NSString *)name documentCount:(NSInteger)documentCount
							status:(nullable NSString *)status raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_name = [name copy];
		_documentCount = documentCount;
		_status = [status copy];
		_raw = [raw copy];
	}
	return self;
}
@end

@interface NFKRetrievalDocument ()
- (instancetype)initWithIdentifier:(NSString *)identifier filename:(nullable NSString *)filename
							status:(nullable NSString *)status raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKRetrievalDocument
- (instancetype)initWithIdentifier:(NSString *)identifier filename:(nullable NSString *)filename
							status:(nullable NSString *)status raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_filename = [filename copy];
		_status = [status copy];
		_raw = [raw copy];
	}
	return self;
}
@end

@interface NFKRetrievalMatch ()
- (instancetype)initWithFileIdentifier:(NSString *)fileIdentifier filename:(nullable NSString *)filename score:(double)score
								  text:(NSString *)text attributes:(nullable NSDictionary *)attributes raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKRetrievalMatch
- (instancetype)initWithFileIdentifier:(NSString *)fileIdentifier filename:(nullable NSString *)filename score:(double)score
								  text:(NSString *)text attributes:(nullable NSDictionary *)attributes raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_fileIdentifier = [fileIdentifier copy];
		_filename = [filename copy];
		_score = score;
		_text = [text copy];
		_attributes = [attributes copy];
		_raw = [raw copy];
	}
	return self;
}
@end

@interface NFKRemoteRetrievalStore ()
@property (nonatomic, copy, nullable) NSString *providerIdentifier;
@end

@implementation NFKRemoteRetrievalStore

@synthesize session = _session;

+ (nullable instancetype)retrievalStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NSDictionary<NSString *, NSNumber *> *styles = @{ @"openai": @(NFKRemoteRetrievalAPIStyleOpenAI), @"xai": @(NFKRemoteRetrievalAPIStyleXAI),
													  @"gemini": @(NFKRemoteRetrievalAPIStyleGemini), @"mistral": @(NFKRemoteRetrievalAPIStyleMistral) };
	NSNumber *style = styles[provider.identifier];
	if (style == nil) {
		return nil;
	}
	NFKRemoteRetrievalStore *store = [[self alloc] init];
	store.apiStyle = style.integerValue;
	store.apiKey = apiKey;
	store.providerIdentifier = provider.identifier;
	store.endpointURL = store.apiStyle == NFKRemoteRetrievalAPIStyleGemini
		? [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta"] : provider.baseURL;
	if (store.apiStyle == NFKRemoteRetrievalAPIStyleXAI) {
		store.managementURL = [NSURL URLWithString:@"https://management-api.x.ai/v1"];
	}
	return store;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 120.0;
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

#pragma mark Stores

- (nullable NFKRetrievalStoreRecord *)createStoreNamed:(NSString *)name
											   options:(nullable NSDictionary<NSString *, id> *)options
												 error:(NSError * _Nullable *)outError
{
	NSDictionary<NSNumber *, NSString *> *nameFields = @{ @(NFKRemoteRetrievalAPIStyleOpenAI): @"name", @(NFKRemoteRetrievalAPIStyleXAI): @"collection_name",
														  @(NFKRemoteRetrievalAPIStyleGemini): @"displayName", @(NFKRemoteRetrievalAPIStyleMistral): @"name" };
	NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:name forKey:nameFields[@(self.apiStyle)]];
	[body addEntriesFromDictionary:options ?: @{}];
	NSDictionary *record = [self JSONFrom:[self storesURL] method:@"POST" body:body management:YES error:outError];
	return record != nil ? [self storeFromRecord:record error:outError] : nil;
}

- (nullable NSArray<NFKRetrievalStoreRecord *> *)storesWithError:(NSError * _Nullable *)outError
{
	NSArray<NSDictionary *> *records = [self allRecordsAt:[self storesURL] listKey:[self storesListKey] management:YES error:outError];
	if (records == nil) {
		return nil;
	}
	NSMutableArray<NFKRetrievalStoreRecord *> *stores = [NSMutableArray array];
	for (NSDictionary *record in records) {
		NFKRetrievalStoreRecord *store = [self storeFromRecord:record error:NULL];
		if (store != nil) {
			[stores addObject:store];
		}
	}
	return stores;
}

- (nullable NFKRetrievalStoreRecord *)storeWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSDictionary *record = [self JSONFrom:[self URLForStore:identifier] method:@"GET" body:nil management:YES error:outError];
	return record != nil ? [self storeFromRecord:record error:outError] : nil;
}

- (BOOL)deleteStoreWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSURL *url = [self URLForStore:identifier];
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini) {
		url = [self URL:url withQuery:@{ @"force": @"true" }];
	}
	return [self JSONFrom:url method:@"DELETE" body:nil management:YES error:outError] != nil;
}

#pragma mark Documents

- (nullable NFKRetrievalDocument *)addFileWithIdentifier:(NSString *)fileIdentifier
												 toStore:(NSString *)storeIdentifier
											  attributes:(nullable NSDictionary<NSString *, id> *)attributes
												   error:(NSError * _Nullable *)outError
{
	NSDictionary *record = nil;
	switch (self.apiStyle) {
		case NFKRemoteRetrievalAPIStyleOpenAI: {
			NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:fileIdentifier forKey:@"file_id"];
			body[@"attributes"] = attributes;
			record = [self JSONFrom:[[self URLForStore:storeIdentifier] URLByAppendingPathComponent:@"files"] method:@"POST" body:body management:YES error:outError];
			break;
		}
		case NFKRemoteRetrievalAPIStyleXAI: {
			NSURL *url = [[[self URLForStore:storeIdentifier] URLByAppendingPathComponent:@"documents"] URLByAppendingPathComponent:fileIdentifier];
			NSDictionary *reply = [self JSONFrom:url method:@"POST" body:attributes != nil ? @{ @"fields": attributes } : @{} management:YES error:outError];
			record = reply != nil ? @{ @"file_id": fileIdentifier, @"reply": reply } : nil;
			break;
		}
		case NFKRemoteRetrievalAPIStyleGemini: {
			NSString *fileName = [fileIdentifier hasPrefix:@"files/"] ? fileIdentifier : [@"files/" stringByAppendingString:fileIdentifier];
			NSURL *url = [self.endpointURL URLByAppendingPathComponent:[[self geminiStoreName:storeIdentifier] stringByAppendingString:@":importFile"]];
			record = [self JSONFrom:url method:@"POST" body:@{ @"fileName": fileName } management:YES error:outError];
			break;
		}
		case NFKRemoteRetrievalAPIStyleMistral:
			return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"a Mistral library takes documents by upload; use uploadData:filename:toStore:"];
	}
	return record != nil ? [self documentFromRecord:record] : nil;
}

- (nullable NFKRetrievalDocument *)uploadData:(NSData *)data
									 filename:(NSString *)filename
									  toStore:(NSString *)storeIdentifier
										error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleMistral) {
		NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
		NSMutableData *body = [NSMutableData data];
		[body appendData:[[NSString stringWithFormat:@"--%@\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\nContent-Type: application/octet-stream\r\n\r\n",
						   boundary, filename] dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:data];
		[body appendData:[[NSString stringWithFormat:@"\r\n--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
		NSMutableURLRequest *request = [self requestTo:[[self URLForStore:storeIdentifier] URLByAppendingPathComponent:@"documents"] method:@"POST" management:YES];
		request.HTTPBody = body;
		[request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
		NSDictionary *record = [self JSONForRequest:request error:outError];
		return record != nil ? [self documentFromRecord:record] : nil;
	}
	NFKRemoteFileStore *files = [self fileStore];
	NFKRemoteFile *file = [files uploadData:data filename:filename mimeType:nil
									purpose:self.apiStyle == NFKRemoteRetrievalAPIStyleOpenAI ? @"assistants" : nil
							   expiresAfter:0 error:outError];
	return file != nil ? [self addFileWithIdentifier:file.identifier toStore:storeIdentifier attributes:nil error:outError] : nil;
}

- (nullable NSArray<NFKRetrievalDocument *> *)documentsInStore:(NSString *)storeIdentifier error:(NSError * _Nullable *)outError
{
	NSString *path = self.apiStyle == NFKRemoteRetrievalAPIStyleOpenAI ? @"files" : @"documents";
	NSString *listKey = self.apiStyle == NFKRemoteRetrievalAPIStyleOpenAI || self.apiStyle == NFKRemoteRetrievalAPIStyleMistral ? @"data" : @"documents";
	NSArray<NSDictionary *> *records = [self allRecordsAt:[[self URLForStore:storeIdentifier] URLByAppendingPathComponent:path]
												  listKey:listKey management:YES error:outError];
	if (records == nil) {
		return nil;
	}
	NSMutableArray<NFKRetrievalDocument *> *documents = [NSMutableArray array];
	for (NSDictionary *record in records) {
		NFKRetrievalDocument *document = [self documentFromRecord:record];
		if (document != nil) {
			[documents addObject:document];
		}
	}
	return documents;
}

- (BOOL)removeDocument:(NSString *)documentIdentifier fromStore:(NSString *)storeIdentifier error:(NSError * _Nullable *)outError
{
	NSURL *url = nil;
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini) {
		NSString *name = [documentIdentifier hasPrefix:@"fileSearchStores/"] ? documentIdentifier
			: [NSString stringWithFormat:@"%@/documents/%@", [self geminiStoreName:storeIdentifier], documentIdentifier];
		url = [self URL:[self.endpointURL URLByAppendingPathComponent:name] withQuery:@{ @"force": @"true" }];
	} else {
		NSString *path = self.apiStyle == NFKRemoteRetrievalAPIStyleOpenAI ? @"files" : @"documents";
		url = [[[self URLForStore:storeIdentifier] URLByAppendingPathComponent:path] URLByAppendingPathComponent:documentIdentifier];
	}
	return [self JSONFrom:url method:@"DELETE" body:nil management:YES error:outError] != nil;
}

#pragma mark Search

- (nullable NSArray<NFKRetrievalMatch *> *)searchStores:(NSArray<NSString *> *)storeIdentifiers
												  query:(NSString *)query
												  limit:(NSInteger)limit
												 filter:(nullable id)filter
												  error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini || self.apiStyle == NFKRemoteRetrievalAPIStyleMistral) {
		NSString *tool = self.apiStyle == NFKRemoteRetrievalAPIStyleGemini ? @"{type: file_search, file_search_store_names: […]}"
																		   : @"{type: document_library, library_ids: […]}";
		return [self fail:outError code:kNFKError_InferenceUnsupported
				   reason:[NSString stringWithFormat:@"this service searches its stores only through a model's tool, %@", tool]];
	}
	NSMutableArray<NFKRetrievalMatch *> *matches = [NSMutableArray array];
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleXAI) {
		NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObjectsAndKeys:query, @"query", @{ @"collection_ids": storeIdentifiers }, @"source", nil];
		body[@"limit"] = limit > 0 ? @(limit) : nil;
		body[@"filter"] = filter;
		NSDictionary *reply = [self JSONFrom:[self.endpointURL URLByAppendingPathComponent:@"documents/search"] method:@"POST" body:body management:NO error:outError];
		if (reply == nil) {
			return nil;
		}
		for (NSDictionary *match in [reply[@"matches"] isKindOfClass:NSArray.class] ? reply[@"matches"] : @[]) {
			NSDictionary *entry = NFKRetrievalDictionary(match);
			[matches addObject:[[NFKRetrievalMatch alloc] initWithFileIdentifier:NFKRetrievalString(entry[@"file_id"]) ?: @""
																		filename:nil
																		   score:[entry[@"score"] doubleValue]
																			text:NFKRetrievalString(entry[@"chunk_content"]) ?: @""
																	  attributes:[entry[@"fields"] isKindOfClass:NSDictionary.class] ? entry[@"fields"] : nil
																			 raw:entry]];
		}
		return matches;
	}
	// OpenAI searches one store per call; several are searched in turn and merged by score.
	for (NSString *store in storeIdentifiers) {
		NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:query forKey:@"query"];
		body[@"max_num_results"] = limit > 0 ? @(limit) : nil;
		body[@"filters"] = filter;
		NSDictionary *reply = [self JSONFrom:[[self URLForStore:store] URLByAppendingPathComponent:@"search"] method:@"POST" body:body management:YES error:outError];
		if (reply == nil) {
			return nil;
		}
		for (NSDictionary *result in [reply[@"data"] isKindOfClass:NSArray.class] ? reply[@"data"] : @[]) {
			NSDictionary *entry = NFKRetrievalDictionary(result);
			NSMutableArray<NSString *> *texts = [NSMutableArray array];
			for (NSDictionary *part in [entry[@"content"] isKindOfClass:NSArray.class] ? entry[@"content"] : @[]) {
				if ([NFKRetrievalDictionary(part)[@"text"] isKindOfClass:NSString.class]) {
					[texts addObject:part[@"text"]];
				}
			}
			[matches addObject:[[NFKRetrievalMatch alloc] initWithFileIdentifier:NFKRetrievalString(entry[@"file_id"]) ?: @""
																		filename:NFKRetrievalString(entry[@"filename"])
																		   score:[entry[@"score"] doubleValue]
																			text:[texts componentsJoinedByString:@"\n"]
																	  attributes:[entry[@"attributes"] isKindOfClass:NSDictionary.class] ? entry[@"attributes"] : nil
																			 raw:entry]];
		}
	}
	[matches sortUsingComparator:^NSComparisonResult(NFKRetrievalMatch *a, NFKRetrievalMatch *b) {
		return a.score > b.score ? NSOrderedAscending : a.score < b.score ? NSOrderedDescending : NSOrderedSame;
	}];
	if (limit > 0 && (NSInteger)matches.count > limit) {
		return [matches subarrayWithRange:NSMakeRange(0, (NSUInteger)limit)];
	}
	return matches;
}

#pragma mark Records

- (nullable NFKRetrievalStoreRecord *)storeFromRecord:(NSDictionary *)record error:(NSError * _Nullable *)outError
{
	NSString *identifier = NFKRetrievalString(record[@"id"]) ?: NFKRetrievalString(record[@"collection_id"]) ?: NFKRetrievalString(record[@"name"]);
	if (identifier == nil) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply names no store"];
	}
	NSString *name = NFKRetrievalString(record[@"collection_name"]) ?: NFKRetrievalString(record[@"displayName"])
		?: (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini ? nil : NFKRetrievalString(record[@"name"]));
	id count = NFKRetrievalDictionary(record[@"file_counts"])[@"total"] ?: record[@"documents_count"] ?: record[@"activeDocumentsCount"] ?: record[@"nb_documents"];
	return [[NFKRetrievalStoreRecord alloc] initWithIdentifier:identifier name:name
												 documentCount:[count respondsToSelector:@selector(integerValue)] ? [count integerValue] : 0
														status:NFKRetrievalString(record[@"status"]) raw:record];
}

// OpenAI's vector_store.file names the file by id; xAI's document by file_metadata; Gemini's document
// by name (an import answers with an operation, also named); Mistral's by id.
- (nullable NFKRetrievalDocument *)documentFromRecord:(NSDictionary *)record
{
	NSDictionary *metadata = NFKRetrievalDictionary(record[@"file_metadata"]);
	NSString *identifier = NFKRetrievalString(metadata[@"file_id"]) ?: NFKRetrievalString(record[@"file_id"])
		?: NFKRetrievalString(record[@"id"]) ?: NFKRetrievalString(record[@"name"]);
	if (identifier == nil) {
		return nil;
	}
	NSString *status = NFKRetrievalString(record[@"status"]) ?: NFKRetrievalString(record[@"state"]) ?: NFKRetrievalString(record[@"process_status"]);
	if (status == nil && record[@"done"] != nil) {
		status = [record[@"done"] boolValue] ? @"done" : @"pending";
	}
	NSString *filename = NFKRetrievalString(metadata[@"name"]) ?: NFKRetrievalString(record[@"displayName"])
		?: NFKRetrievalString(record[@"filename"]) ?: (self.apiStyle == NFKRemoteRetrievalAPIStyleMistral ? NFKRetrievalString(record[@"name"]) : nil);
	return [[NFKRetrievalDocument alloc] initWithIdentifier:identifier filename:filename status:status raw:record];
}

#pragma mark Plumbing

- (NSURL *)storesURL
{
	switch (self.apiStyle) {
		case NFKRemoteRetrievalAPIStyleOpenAI:
			return [self.endpointURL URLByAppendingPathComponent:@"vector_stores"];
		case NFKRemoteRetrievalAPIStyleXAI:
			return [self.managementURL URLByAppendingPathComponent:@"collections"];
		case NFKRemoteRetrievalAPIStyleGemini:
			return [self.endpointURL URLByAppendingPathComponent:@"fileSearchStores"];
		case NFKRemoteRetrievalAPIStyleMistral:
			return [self.endpointURL URLByAppendingPathComponent:@"libraries"];
	}
	return self.endpointURL;
}

- (NSString *)storesListKey
{
	switch (self.apiStyle) {
		case NFKRemoteRetrievalAPIStyleXAI:
			return @"collections";
		case NFKRemoteRetrievalAPIStyleGemini:
			return @"fileSearchStores";
		default:
			return @"data";
	}
}

- (NSString *)geminiStoreName:(NSString *)identifier
{
	return [identifier hasPrefix:@"fileSearchStores/"] ? identifier : [@"fileSearchStores/" stringByAppendingString:identifier];
}

- (NSURL *)URLForStore:(NSString *)identifier
{
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini) {
		return [self.endpointURL URLByAppendingPathComponent:[self geminiStoreName:identifier]];
	}
	return [[self storesURL] URLByAppendingPathComponent:identifier];
}

- (NSURL *)URL:(NSURL *)url withQuery:(NSDictionary<NSString *, NSString *> *)query
{
	NSURLComponents *components = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray arrayWithArray:components.queryItems ?: @[]];
	for (NSString *name in query) {
		[items addObject:[NSURLQueryItem queryItemWithName:name value:query[name]]];
	}
	components.queryItems = items;
	return components.URL ?: url;
}

// Every page of a list: OpenAI's has_more / last_id, Gemini's nextPageToken, xAI's pagination_token,
// Mistral's page_token.
- (nullable NSArray<NSDictionary *> *)allRecordsAt:(NSURL *)url listKey:(NSString *)listKey management:(BOOL)management error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *records = [NSMutableArray array];
	NSDictionary<NSString *, NSString *> *query = nil;
	while (YES) {
		NSDictionary *page = [self JSONFrom:query != nil ? [self URL:url withQuery:query] : url method:@"GET" body:nil management:management error:outError];
		if (page == nil) {
			return nil;
		}
		for (id record in [page[listKey] isKindOfClass:NSArray.class] ? page[listKey] : @[]) {
			if ([record isKindOfClass:NSDictionary.class]) {
				[records addObject:record];
			}
		}
		NSString *token = NFKRetrievalString(page[@"nextPageToken"]) ?: NFKRetrievalString(page[@"pagination_token"]) ?: NFKRetrievalString(page[@"next_page_token"]);
		if (token.length > 0) {
			NSString *name = self.apiStyle == NFKRemoteRetrievalAPIStyleGemini ? @"pageToken"
						   : self.apiStyle == NFKRemoteRetrievalAPIStyleXAI ? @"pagination_token" : @"page_token";
			query = @{ name: token };
			continue;
		}
		NSString *last = NFKRetrievalString(page[@"last_id"]);
		if ([page[@"has_more"] boolValue] && last != nil) {
			query = @{ @"after": last };
			continue;
		}
		return records;
	}
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method management:(BOOL)management
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleGemini) {
		if (self.apiKey.length > 0) {
			[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
		}
		return request;
	}
	NSString *key = management && self.apiStyle == NFKRemoteRetrievalAPIStyleXAI ? self.managementAPIKey : self.apiKey;
	[NFKRemoteTransport authorizeRequest:request apiKey:key style:NFKRemoteAPIStyleOpenAIChat];
	if (self.apiStyle == NFKRemoteRetrievalAPIStyleOpenAI) {
		[request setValue:@"assistants=v2" forHTTPHeaderField:@"OpenAI-Beta"];
	}
	return request;
}

- (nullable NSDictionary *)JSONFrom:(NSURL *)url method:(NSString *)method body:(nullable NSDictionary *)body
						 management:(BOOL)management error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self requestTo:url method:method management:management];
	if (body != nil) {
		request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	return [self JSONForRequest:request error:outError];
}

// A delete that answers with an empty body is a success, answered here as an empty object.
- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the retrieval call failed"];
		}
		return nil;
	}
	if (data.length == 0) {
		return @{};
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

- (nullable NFKRemoteFileStore *)fileStore
{
	NFKRemoteProvider *provider = self.providerIdentifier != nil ? [NFKRemoteProvider providerWithIdentifier:self.providerIdentifier] : nil;
	NFKRemoteFileStore *files = provider != nil ? [NFKRemoteFileStore fileStoreForProvider:provider apiKey:self.apiKey] : nil;
	files.session = self.session;
	return files;
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
