//
//  NFKRemoteSkillStore.m
//  InferKit
//

#import <InferKit/NFKRemoteSkillStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteTransportPrivate.h"

static NSString * _Nullable NFKSkillString(id value)
{
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

static NSDate * _Nullable NFKSkillDate(id value)
{
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

#pragma mark - Skill

@interface NFKRemoteSkill ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite, nullable) NSString *name;
@property (nonatomic, copy, readwrite, nullable) NSString *skillDescription;
@property (nonatomic, copy, readwrite, nullable) NSString *version;
@property (nonatomic, copy, readwrite, nullable) NSString *latestVersion;
@property (nonatomic, copy, readwrite) NSString *source;
@property (nonatomic, copy, readwrite, nullable) NSDate *createdAt;
@property (nonatomic, copy, readwrite, nullable) NSDate *updatedAt;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKRemoteSkill

+ (nullable instancetype)skillWithRecord:(NSDictionary<NSString *, id> *)record apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle
{
	if (![record isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	BOOL anthropic = apiStyle == NFKRemoteSkillStoreAPIStyleAnthropic;
	NSString *identifier = NFKSkillString(anthropic ? record[@"id"] : record[@"skill_id"]);
	if (identifier == nil) {
		return nil;
	}
	NFKRemoteSkill *skill = [[self alloc] initPrivate];
	skill.identifier = identifier;
	skill.raw = record;
	skill.createdAt = NFKSkillDate(record[@"created_at"]);
	skill.updatedAt = NFKSkillDate(record[@"updated_at"]);
	if (anthropic) {
		NSDictionary *source = [record[@"source"] isKindOfClass:NSDictionary.class] ? record[@"source"] : nil;
		skill.name = NFKSkillString(record[@"display_name"]);
		skill.version = NFKSkillString(record[@"latest_version_id"]);
		skill.latestVersion = skill.version;
		skill.source = NFKSkillString(source[@"type"]) ?: @"custom";
		return skill;
	}
	skill.name = NFKSkillString(record[@"name"]);
	skill.skillDescription = NFKSkillString(record[@"description"]);
	skill.version = NFKSkillString(record[@"revision"]);
	skill.latestVersion = NFKSkillString(record[@"latest_revision"]) ?: skill.version;
	skill.source = @"custom";
	return skill;
}

- (instancetype)initPrivate
{
	return [super init];
}

- (NSDictionary<NSString *, NSString *> *)skillReference
{
	return @{ @"id": self.identifier, @"source": self.source };
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

@end

#pragma mark - Version

@interface NFKRemoteSkillVersion ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite, nullable) NSString *name;
@property (nonatomic, copy, readwrite, nullable) NSString *skillDescription;
@property (nonatomic, copy, readwrite, nullable) NSDate *createdAt;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKRemoteSkillVersion

+ (nullable instancetype)versionWithRecord:(NSDictionary<NSString *, id> *)record apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle
{
	if (![record isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *identifier = NFKSkillString(apiStyle == NFKRemoteSkillStoreAPIStyleAnthropic ? record[@"id"] : record[@"revision"]);
	if (identifier == nil) {
		return nil;
	}
	NFKRemoteSkillVersion *version = [[self alloc] initPrivate];
	version.identifier = identifier;
	version.name = NFKSkillString(record[@"name"]);
	version.skillDescription = NFKSkillString(record[@"description"]);
	version.createdAt = NFKSkillDate(record[@"created_at"]);
	version.raw = record;
	return version;
}

- (instancetype)initPrivate
{
	return [super init];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

@end

#pragma mark - Store

/*! One part of a multipart body: a field name, the filename it carries, its type, and its bytes. */
@interface NFKSkillUploadPart : NSObject
@property (nonatomic, copy) NSString *field;
@property (nonatomic, copy) NSString *filename;
@property (nonatomic, copy) NSString *contentType;
@property (nonatomic, copy) NSData *data;
@end

@implementation NFKSkillUploadPart
@end

@implementation NFKRemoteSkillStore

@synthesize session = _session;

+ (instancetype)skillStoreWithEndpointURL:(nullable NSURL *)endpointURL apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle
{
	NFKRemoteSkillStore *store = [[self alloc] init];
	store.endpointURL = endpointURL;
	store.apiStyle = apiStyle;
	return store;
}

+ (nullable instancetype)skillStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	NSDictionary<NSString *, NSNumber *> *styles = @{ @"anthropic": @(NFKRemoteSkillStoreAPIStyleAnthropic),
													  @"perplexity": @(NFKRemoteSkillStoreAPIStylePerplexity) };
	NSNumber *style = styles[provider.identifier];
	if (style == nil) {
		return nil;
	}
	NFKRemoteSkillStore *store = [self skillStoreWithEndpointURL:[provider URLForPath:@"skills"] apiStyle:style.integerValue];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		store.session = pinned;
	}
	store.apiKey = apiKey;
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

- (BOOL)isAnthropic
{
	return self.apiStyle == NFKRemoteSkillStoreAPIStyleAnthropic;
}

#pragma mark Creating

- (nullable NFKRemoteSkill *)createSkillWithBundle:(NSData *)zip
											  name:(nullable NSString *)name
											 error:(NSError * _Nullable *)outError
{
	return [self createSkillWithParts:@[ [self partForBundle:zip] ] name:name error:outError];
}

- (nullable NFKRemoteSkill *)createSkillWithDirectoryAtURL:(NSURL *)directoryURL
													  name:(nullable NSString *)name
													 error:(NSError * _Nullable *)outError
{
	NSArray<NFKSkillUploadPart *> *parts = [self partsForDirectoryAtURL:directoryURL error:outError];
	return parts != nil ? [self createSkillWithParts:parts name:name error:outError] : nil;
}

- (nullable NFKRemoteSkill *)createSkillWithParts:(NSArray<NFKSkillUploadPart *> *)parts
											 name:(nullable NSString *)name
											error:(NSError * _Nullable *)outError
{
	NSDictionary<NSString *, NSString *> *fields = self.isAnthropic && name.length > 0 ? @{ @"display_name": name } : @{};
	NSURLRequest *request = [self uploadRequestTo:self.endpointURL method:@"POST" fields:fields parts:parts];
	NSDictionary *record = [self JSONForRequest:request error:outError];
	return record != nil ? [self skillFromRecord:record error:outError] : nil;
}

#pragma mark Reading

- (nullable NSArray<NFKRemoteSkill *> *)skillsWithError:(NSError * _Nullable *)outError
{
	NSArray<NSDictionary *> *records = [self recordsAtURL:self.endpointURL query:self.isAnthropic ? @{ @"limit": @"1000" } : @{ @"limit": @"200" }
											   listField:self.isAnthropic ? @"data" : @"skills" error:outError];
	if (records == nil) {
		return nil;
	}
	NSMutableArray<NFKRemoteSkill *> *skills = [NSMutableArray array];
	for (NSDictionary *record in records) {
		NFKRemoteSkill *skill = [NFKRemoteSkill skillWithRecord:record apiStyle:self.apiStyle];
		if (skill != nil) {
			[skills addObject:skill];
		}
	}
	return skills;
}

- (nullable NFKRemoteSkill *)skillWithIdentifier:(NSString *)identifier
										 version:(nullable NSString *)version
										   error:(NSError * _Nullable *)outError
{
	NSDictionary<NSString *, NSString *> *query = !self.isAnthropic && version.length > 0 ? @{ @"revision": version } : @{};
	NSDictionary *record = [self JSONForRequest:[self requestTo:[self URLForSkill:identifier path:nil query:query] method:@"GET"] error:outError];
	return record != nil ? [self skillFromRecord:record error:outError] : nil;
}

- (nullable NSArray<NFKRemoteSkillVersion *> *)versionsOfSkillWithIdentifier:(NSString *)identifier
																	   error:(NSError * _Nullable *)outError
{
	NSURL *url = [self URLForSkill:identifier path:self.isAnthropic ? @"versions" : @"revisions" query:@{}];
	NSArray<NSDictionary *> *records = [self recordsAtURL:url query:self.isAnthropic ? @{ @"limit": @"1000" } : @{ @"limit": @"200" }
											   listField:self.isAnthropic ? @"data" : @"revisions" error:outError];
	if (records == nil) {
		return nil;
	}
	NSMutableArray<NFKRemoteSkillVersion *> *versions = [NSMutableArray array];
	for (NSDictionary *record in records) {
		NFKRemoteSkillVersion *version = [NFKRemoteSkillVersion versionWithRecord:record apiStyle:self.apiStyle];
		if (version != nil) {
			[versions addObject:version];
		}
	}
	return versions;
}

#pragma mark Versioning

- (nullable NFKRemoteSkill *)addVersionToSkillWithIdentifier:(NSString *)identifier
													  bundle:(NSData *)zip
											 expectedVersion:(nullable NSString *)expectedVersion
													   error:(NSError * _Nullable *)outError
{
	return [self addVersionToSkillWithIdentifier:identifier parts:@[ [self partForBundle:zip] ] expectedVersion:expectedVersion error:outError];
}

- (nullable NFKRemoteSkill *)addVersionToSkillWithIdentifier:(NSString *)identifier
											directoryAtURL:(NSURL *)directoryURL
										   expectedVersion:(nullable NSString *)expectedVersion
													 error:(NSError * _Nullable *)outError
{
	NSArray<NFKSkillUploadPart *> *parts = [self partsForDirectoryAtURL:directoryURL error:outError];
	return parts != nil ? [self addVersionToSkillWithIdentifier:identifier parts:parts expectedVersion:expectedVersion error:outError] : nil;
}

// Anthropic posts a version under /versions and answers with it, so the skill is read back for its
// record. Perplexity replaces the bundle with PUT, guarded by the revision it expects.
- (nullable NFKRemoteSkill *)addVersionToSkillWithIdentifier:(NSString *)identifier
													   parts:(NSArray<NFKSkillUploadPart *> *)parts
											 expectedVersion:(nullable NSString *)expectedVersion
													   error:(NSError * _Nullable *)outError
{
	if (self.isAnthropic) {
		NSURLRequest *request = [self uploadRequestTo:[self URLForSkill:identifier path:@"versions" query:@{}] method:@"POST" fields:@{} parts:parts];
		if ([self JSONForRequest:request error:outError] == nil) {
			return nil;
		}
		return [self skillWithIdentifier:identifier version:nil error:outError];
	}
	NSString *expected = [self expectedVersion:expectedVersion ofSkill:identifier error:outError];
	if (expected == nil) {
		return nil;
	}
	NSURLRequest *request = [self uploadRequestTo:[self URLForSkill:identifier path:nil query:@{ @"expected_revision": expected }]
										   method:@"PUT" fields:@{} parts:parts];
	NSDictionary *record = [self JSONForRequest:request error:outError];
	return record != nil ? [self skillFromRecord:record error:outError] : nil;
}

#pragma mark Deleting and downloading

- (BOOL)deleteSkillWithIdentifier:(NSString *)identifier
				  expectedVersion:(nullable NSString *)expectedVersion
							error:(NSError * _Nullable *)outError
{
	NSDictionary<NSString *, NSString *> *query = @{};
	if (!self.isAnthropic) {
		NSString *expected = [self expectedVersion:expectedVersion ofSkill:identifier error:outError];
		if (expected == nil) {
			return NO;
		}
		query = @{ @"expected_revision": expected };
	}
	return [self dataForRequest:[self requestTo:[self URLForSkill:identifier path:nil query:query] method:@"DELETE"] error:outError] != nil;
}

- (BOOL)deleteVersion:(NSString *)version
ofSkillWithIdentifier:(NSString *)identifier
				error:(NSError * _Nullable *)outError
{
	if (!self.isAnthropic) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"Perplexity deletes a whole skill, not one of its revisions" error:outError];
		return NO;
	}
	NSURL *url = [self URLForSkill:identifier path:[@"versions/" stringByAppendingString:[self escapedSegment:version]] query:@{}];
	return [self dataForRequest:[self requestTo:url method:@"DELETE"] error:outError] != nil;
}

- (nullable NSURL *)downloadURLForSkillWithIdentifier:(NSString *)identifier
											  version:(nullable NSString *)version
												error:(NSError * _Nullable *)outError
{
	if (self.isAnthropic) {
		return [self failWithCode:kNFKError_InferenceUnsupported reason:@"Anthropic's Skills API serves no bundle download" error:outError];
	}
	NSDictionary<NSString *, NSString *> *query = version.length > 0 ? @{ @"revision": version } : @{};
	NSDictionary *record = [self JSONForRequest:[self requestTo:[self URLForSkill:identifier path:@"download" query:query] method:@"GET"] error:outError];
	NSString *link = NFKSkillString(record[@"download_url"]);
	NSURL *url = link != nil ? [NSURL URLWithString:link] : nil;
	if (record != nil && url == nil) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the download answer carries no download_url" error:outError];
	}
	return url;
}

/*! The revision a Perplexity write is guarded by: the caller's, or the skill's latest read now. */
- (nullable NSString *)expectedVersion:(nullable NSString *)expectedVersion ofSkill:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	if (expectedVersion.length > 0) {
		return expectedVersion;
	}
	NFKRemoteSkill *skill = [self skillWithIdentifier:identifier version:nil error:outError];
	if (skill != nil && skill.latestVersion == nil) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the skill's record names no latest revision" error:outError];
	}
	return skill.latestVersion;
}

#pragma mark Bundles

- (NFKSkillUploadPart *)partForBundle:(NSData *)zip
{
	NFKSkillUploadPart *part = [[NFKSkillUploadPart alloc] init];
	part.field = self.isAnthropic ? @"files[]" : @"file";
	part.filename = @"skill.zip";
	part.contentType = @"application/zip";
	part.data = zip;
	return part;
}

// Anthropic takes the folder's files, each named by its path under the folder's own name;
// Perplexity takes the folder zipped, which NSFileCoordinator writes with the folder at its root.
- (nullable NSArray<NFKSkillUploadPart *> *)partsForDirectoryAtURL:(NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	NSNumber *isDirectory = nil;
	[directoryURL getResourceValue:&isDirectory forKey:NSURLIsDirectoryKey error:NULL];
	if (!isDirectory.boolValue || ![NSFileManager.defaultManager fileExistsAtPath:[directoryURL URLByAppendingPathComponent:@"SKILL.md"].path]) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"a skill folder holds a SKILL.md at its top" error:outError];
	}
	if (!self.isAnthropic) {
		NSData *zip = [self zipOfDirectoryAtURL:directoryURL error:outError];
		return zip != nil ? @[ [self partForBundle:zip] ] : nil;
	}
	NSMutableArray<NFKSkillUploadPart *> *parts = [NSMutableArray array];
	NSString *root = directoryURL.URLByStandardizingPath.URLByResolvingSymlinksInPath.path;
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:directoryURL
																   includingPropertiesForKeys:@[ NSURLIsRegularFileKey ]
																					  options:NSDirectoryEnumerationSkipsHiddenFiles
																				 errorHandler:nil];
	for (NSURL *fileURL in enumerator) {
		NSNumber *regular = nil;
		[fileURL getResourceValue:&regular forKey:NSURLIsRegularFileKey error:NULL];
		NSString *path = fileURL.URLByStandardizingPath.URLByResolvingSymlinksInPath.path;
		if (!regular.boolValue || ![path hasPrefix:[root stringByAppendingString:@"/"]]) {
			continue;
		}
		NSData *data = [NSData dataWithContentsOfURL:fileURL options:0 error:outError];
		if (data == nil) {
			return nil;
		}
		NFKSkillUploadPart *part = [[NFKSkillUploadPart alloc] init];
		part.field = @"files[]";
		part.filename = [directoryURL.lastPathComponent stringByAppendingPathComponent:[path substringFromIndex:root.length + 1]];
		part.contentType = @"application/octet-stream";
		part.data = data;
		[parts addObject:part];
	}
	[parts sortUsingComparator:^NSComparisonResult(NFKSkillUploadPart *a, NFKSkillUploadPart *b) {
		return [a.filename compare:b.filename];
	}];
	return parts;
}

- (nullable NSData *)zipOfDirectoryAtURL:(NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	NSFileCoordinator *coordinator = [[NSFileCoordinator alloc] initWithFilePresenter:nil];
	__block NSData *zip = nil;
	__block NSError *readError = nil;
	NSError *coordinationError = nil;
	[coordinator coordinateReadingItemAtURL:directoryURL options:NSFileCoordinatorReadingForUploading error:&coordinationError byAccessor:^(NSURL *zipURL) {
		// The archive lives only while this block runs.
		zip = [NSData dataWithContentsOfURL:zipURL options:0 error:&readError];
	}];
	if (zip == nil && outError != NULL) {
		*outError = coordinationError ?: readError ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the skill folder could not be zipped"];
	}
	return zip;
}

#pragma mark Requests

- (NSString *)escapedSegment:(NSString *)segment
{
	NSMutableCharacterSet *allowed = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[allowed removeCharactersInString:@"/"];
	return [segment stringByAddingPercentEncodingWithAllowedCharacters:allowed] ?: segment;
}

- (NSURL *)URLForSkill:(NSString *)identifier path:(nullable NSString *)path query:(NSDictionary<NSString *, NSString *> *)query
{
	NSString *string = [self.endpointURL.absoluteString stringByAppendingFormat:@"/%@", [self escapedSegment:identifier]];
	if (path.length > 0) {
		string = [string stringByAppendingFormat:@"/%@", path];
	}
	return [self URL:[NSURL URLWithString:string] withQuery:query];
}

- (NSURL *)URL:(NSURL *)url withQuery:(NSDictionary<NSString *, NSString *> *)query
{
	if (query.count == 0) {
		return url;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
	for (NSString *name in [query.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		[items addObject:[NSURLQueryItem queryItemWithName:name value:query[name]]];
	}
	components.queryItems = items;
	return components.URL ?: url;
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey
								   style:self.isAnthropic ? NFKRemoteAPIStyleAnthropicMessages : NFKRemoteAPIStyleOpenAIChat];
	return request;
}

- (NSURLRequest *)uploadRequestTo:(NSURL *)url
						   method:(NSString *)method
						   fields:(NSDictionary<NSString *, NSString *> *)fields
							parts:(NSArray<NFKSkillUploadPart *> *)parts
{
	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSData *dashBoundary = [[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding];
	for (NSString *name in fields) {
		[body appendData:dashBoundary];
		[body appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, fields[name]]
						  dataUsingEncoding:NSUTF8StringEncoding]];
	}
	for (NFKSkillUploadPart *part in parts) {
		[body appendData:dashBoundary];
		[body appendData:[[NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"; filename=\"%@\"\r\nContent-Type: %@\r\n\r\n",
						   part.field, part.filename, part.contentType] dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:part.data];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
	NSMutableURLRequest *request = [self requestTo:url method:method];
	request.HTTPBody = body;
	[request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	return request;
}

/*! Every record across a list's pages: Anthropic's next_page passed back as page, Perplexity's
	next_page_token as page_token. A token seen before ends the reading. */
- (nullable NSArray<NSDictionary *> *)recordsAtURL:(NSURL *)url
											 query:(NSDictionary<NSString *, NSString *> *)query
										 listField:(NSString *)listField
											 error:(NSError * _Nullable *)outError
{
	NSMutableArray<NSDictionary *> *records = [NSMutableArray array];
	NSMutableSet<NSString *> *seen = [NSMutableSet set];
	NSString *token = nil;
	do {
		NSMutableDictionary<NSString *, NSString *> *pageQuery = [query mutableCopy];
		if (token != nil) {
			pageQuery[self.isAnthropic ? @"page" : @"page_token"] = token;
		}
		NSDictionary *page = [self JSONForRequest:[self requestTo:[self URL:url withQuery:pageQuery] method:@"GET"] error:outError];
		if (page == nil) {
			return nil;
		}
		NSArray *entries = page[listField];
		if (![entries isKindOfClass:NSArray.class]) {
			return [self failWithCode:kNFKError_InferenceBackendFailure
							   reason:[NSString stringWithFormat:@"the list carries no %@ array", listField] error:outError];
		}
		[records addObjectsFromArray:entries];
		token = NFKSkillString(page[self.isAnthropic ? @"next_page" : @"next_page_token"]);
		if (token != nil && [seen containsObject:token]) {
			break;
		}
		if (token != nil) {
			[seen addObject:token];
		}
	} while (token != nil);
	return records;
}

- (nullable NFKRemoteSkill *)skillFromRecord:(NSDictionary *)record error:(NSError * _Nullable *)outError
{
	NFKRemoteSkill *skill = [NFKRemoteSkill skillWithRecord:record apiStyle:self.apiStyle];
	return skill ?: [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the skill record carries no identifier" error:outError];
}

#pragma mark Transport

- (nullable NSData *)dataForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the skills call failed"];
		}
		return nil;
	}
	return data;
}

- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSData *data = [self dataForRequest:request error:outError];
	if (data == nil) {
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [object isKindOfClass:NSDictionary.class] ? object
		: [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the answer is not a JSON object" error:outError];
}

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (nullable id)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
