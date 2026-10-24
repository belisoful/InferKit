//
//  NFKHFHubDownloads.m
//  InferKit
//
//  Download plans, observable downloads, cache status, and removal.
//

#import "NFKHFHub.h"
#import "NFKHFHubInternal.h"
#import "NFKHFHubTransfer.h"
#import "NFKDownloadInternal.h"
#import "NFKModelCatalog.h"
#import "NFKModelDescriptor.h"
#import "NFKModelVocabulary.h"
#import "NFKErrors.h"
#import <CommonCrypto/CommonCrypto.h>
#include <fnmatch.h>

static NSString * const NFKHFHubURLCacheFolder = @"@url";
static NSString * const NFKHFHubTrashPrefix = @".inferkit-removing-";

/*! What a removal targets: a snapshot folder and the paths in it. */
@interface NFKHFHubRemovalTarget : NSObject
@property (nonatomic, copy, nullable) NSString *repository;
@property (nonatomic, copy, nullable) NSString *revision;
@property (nonatomic, copy, nullable) NSURL *sourceURL;
@property (nonatomic, copy, nullable) NSURL *snapshotURL;
/*! nil for every file in the snapshot. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *paths;
/*! Why the target cannot be removed, such as an unsafe repository name; nil when it can. */
@property (nonatomic, strong, nullable) NSError *refusal;
@end

@implementation NFKHFHubRemovalTarget
@end

/*! The files of a catalog source that come from one place: a repository at a revision, or a URL. */
@interface NFKHFHubSourcePart : NSObject
@property (nonatomic, copy, nullable) NSString *repository;
@property (nonatomic, copy, nullable) NSString *revision;
@property (nonatomic, copy, nullable) NSURL *URL;
/*! nil for a source that lists no files, which names the whole repository. */
@property (nonatomic, strong, nullable) NSMutableArray<NFKModelFile *> *files;
@end

@implementation NFKHFHubSourcePart
@end

static BOOL NFKHFHubIsGlob(NSString *path)
{
	return [path rangeOfCharacterFromSet:[NSCharacterSet characterSetWithCharactersInString:@"*?["]].location != NSNotFound;
}

static BOOL NFKHFHubGlobMatches(NSString *pattern, NSString *path)
{
	return fnmatch(pattern.fileSystemRepresentation, path.fileSystemRepresentation, 0) == 0;
}

static NSString *NFKHFHubHexDigest(NSData *data)
{
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
	NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
	for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
		[hex appendFormat:@"%02x", digest[i]];
	}
	return hex;
}

@implementation NFKHFHub (NFKDownloads)

#pragma mark Listing

- (NSURL *)treeURLForRepo:(NSString *)repo revision:(NSString *)revision
{
	NSMutableCharacterSet *segment = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[segment removeCharactersInString:@"/"];
	NSString *encodedRevision = [revision stringByAddingPercentEncodingWithAllowedCharacters:segment];
	NSString *base = self.endpointURL.absoluteString;
	while ([base hasSuffix:@"/"]) {
		base = [base substringToIndex:base.length - 1];
	}
	NSString *string = [NSString stringWithFormat:@"%@/api/models/%@/tree/%@?recursive=1", base, repo, encodedRevision];
	return [NSURL URLWithString:string];
}

/*! The next page of a paginated listing, from the response's Link header. */
- (nullable NSURL *)nextPageURLFromResponse:(NSHTTPURLResponse *)response
{
	NSString *link = [response valueForHTTPHeaderField:@"Link"];
	for (NSString *part in [link componentsSeparatedByString:@","]) {
		if ([part rangeOfString:@"rel=\"next\""].location == NSNotFound) {
			continue;
		}
		NSRange open = [part rangeOfString:@"<"];
		NSRange close = [part rangeOfString:@">"];
		if (open.location != NSNotFound && close.location != NSNotFound && close.location > open.location) {
			return [NSURL URLWithString:[part substringWithRange:NSMakeRange(open.location + 1, close.location - open.location - 1)]];
		}
	}
	return nil;
}

/*! The file entries of a repository at a revision, in listing order. */
- (nullable NSArray<NSDictionary *> *)fileEntriesForRepo:(NSString *)repo
												revision:(NSString *)revision
												   error:(NSError **)outError
{
	NSMutableArray<NSDictionary *> *entries = [NSMutableArray array];
	NSURL *pageURL = [self treeURLForRepo:repo revision:revision];
	BOOL hadToken = [[self requestForURL:pageURL] valueForHTTPHeaderField:@"Authorization"] != nil;
	while (pageURL != nil) {
		NSHTTPURLResponse *response = nil;
		NSData *body = [self fetchMetadataFromURL:pageURL HTTPMethod:@"GET" response:&response error:outError];
		if (body == nil) {
			return nil;
		}
		NSInteger status = response != nil ? response.statusCode : 200;
		if (status < 200 || status >= 300) {
			NSString *subject = [NSString stringWithFormat:@"%@ at %@", repo, revision];
			[self propagateError:[NFKHFHubTransfer errorForHTTPStatus:status subject:subject hadToken:hadToken] to:outError];
			return nil;
		}
		id listing = [NSJSONSerialization JSONObjectWithData:body options:0 error:NULL];
		if (![listing isKindOfClass:NSArray.class]) {
			NSString *reason = [NSString stringWithFormat:@"the listing of %@ is not a JSON array", repo];
			[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorHTTPStatus reason:reason] to:outError];
			return nil;
		}
		for (id entry in (NSArray *)listing) {
			if (![entry isKindOfClass:NSDictionary.class] || ![entry[@"type"] isEqual:@"file"]
				|| ![entry[@"path"] isKindOfClass:NSString.class]) {
				continue;
			}
			NSString *kind = [NSString stringWithFormat:@"listing of %@ at %@ names the file", repo, revision];
			NSError *refusal = NFKHFHubUnsafePathError(entry[@"path"], kind, YES);
			if (refusal != nil) {
				[self propagateError:refusal to:outError];
				return nil;
			}
			[entries addObject:entry];
		}
		pageURL = response != nil ? [self nextPageURLFromResponse:response] : nil;
	}
	return entries;
}

#pragma mark Plans

- (BOOL)requireCacheDirectory:(NSError **)outError
{
	if (self.cacheDirectoryURL == nil) {
		return [self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
	}
	return YES;
}

- (BOOL)fileAtURL:(NSURL *)url isWholeWithByteCount:(long long)byteCount
{
	__block BOOL whole = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
		if (attributes == nil) {
			return;
		}
		whole = byteCount < 0 || [attributes[NSFileSize] longLongValue] == byteCount;
	}];
	return whole;
}

- (NFKDownloadPlanFile *)planFileForRepo:(NSString *)repo
								revision:(NSString *)revision
								   entry:(NSDictionary *)entry
								 listing:(nullable NFKModelFile *)listing
								optional:(BOOL)optional
{
	NSString *path = entry[@"path"];
	NSDictionary *lfs = [entry[@"lfs"] isKindOfClass:NSDictionary.class] ? entry[@"lfs"] : nil;
	long long byteCount = NFKDownloadUnknownByteCount;
	if ([lfs[@"size"] isKindOfClass:NSNumber.class]) {
		byteCount = [lfs[@"size"] longLongValue];
	} else if ([entry[@"size"] isKindOfClass:NSNumber.class]) {
		byteCount = [entry[@"size"] longLongValue];
	} else if (listing.byteCount != nil) {
		byteCount = listing.byteCount.longLongValue;
	}
	NSString *sha256 = [lfs[@"oid"] isKindOfClass:NSString.class] ? lfs[@"oid"] : listing.sha256;
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	NFKDownloadPlanFile *file = [[NFKDownloadPlanFile alloc] initWithPath:path
																remoteURL:[self remoteURLForRepo:repo revision:revision path:path]
																 localURL:localURL
																byteCount:byteCount
																   sha256:sha256
																   cached:[self fileAtURL:localURL isWholeWithByteCount:byteCount]
																 optional:optional];
	file.snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	return file;
}

/*! Plans a repository's files. requiredPaths nil plans every file; optional paths absent from the
	repository are left out; listings fill in a size or digest the repository lacks. */
- (nullable NFKDownloadPlan *)planRepo:(NSString *)repo
							  revision:(nullable NSString *)revision
						 requiredPaths:(nullable NSArray<NSString *> *)requiredPaths
						 optionalPaths:(NSArray<NSString *> *)optionalPaths
							  listings:(NSDictionary<NSString *, NFKModelFile *> *)listings
						   singleFile:(BOOL)singleFile
								 error:(NSError **)outError
{
	if (![self requireCacheDirectory:outError]) {
		return nil;
	}
	if (repo.length == 0) {
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"no repository is named"]
						  to:outError];
		return nil;
	}
	NSError *refusal = [self refusalForRepo:repo revision:revision path:nil];
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	NSString *normalized = [self normalizedRevision:revision];
	NSArray<NSDictionary *> *entries = [self fileEntriesForRepo:repo revision:normalized error:outError];
	if (entries == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, NSDictionary *> *entriesByPath = [NSMutableDictionary dictionaryWithCapacity:entries.count];
	for (NSDictionary *entry in entries) {
		entriesByPath[entry[@"path"]] = entry;
	}

	NSMutableOrderedSet<NSString *> *chosen = [NSMutableOrderedSet orderedSet];
	NSMutableSet<NSString *> *optionalChosen = [NSMutableSet set];
	NSArray<NSString *> *wanted = requiredPaths ?: [entries valueForKey:@"path"];
	for (NSString *path in wanted) {
		if (NFKHFHubIsGlob(path)) {
			for (NSDictionary *entry in entries) {
				if (NFKHFHubGlobMatches(path, entry[@"path"])) {
					[chosen addObject:entry[@"path"]];
				}
			}
			continue;
		}
		if (entriesByPath[path] == nil) {
			NSString *reason = [NSString stringWithFormat:@"%@ at %@ has no file %@", repo, normalized, path];
			[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorNotFound reason:reason] to:outError];
			return nil;
		}
		[chosen addObject:path];
	}
	NSMutableArray<NSString *> *absentOptional = [NSMutableArray array];
	for (NSString *path in optionalPaths) {
		if (!NFKHFHubIsGlob(path) && entriesByPath[path] == nil) {
			[absentOptional addObject:path];
		}
		for (NSDictionary *entry in entries) {
			NSString *entryPath = entry[@"path"];
			BOOL matches = NFKHFHubIsGlob(path) ? NFKHFHubGlobMatches(path, entryPath) : [path isEqualToString:entryPath];
			if (matches && ![chosen containsObject:entryPath]) {
				[chosen addObject:entryPath];
				[optionalChosen addObject:entryPath];
			}
		}
	}

	NSMutableArray<NFKDownloadPlanFile *> *files = [NSMutableArray arrayWithCapacity:chosen.count];
	for (NSDictionary *entry in entries) {
		NSString *path = entry[@"path"];
		if ([chosen containsObject:path]) {
			[files addObject:[self planFileForRepo:repo revision:normalized entry:entry listing:listings[path]
										  optional:[optionalChosen containsObject:path]]];
		}
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:normalized];
	NSURL *localURL = singleFile && files.count == 1 ? files.firstObject.localURL : snapshotURL;
	NFKDownloadPlan *plan = [[NFKDownloadPlan alloc] initWithRepository:repo
															   revision:normalized
															  sourceURL:nil
																  files:files
															   localURL:files.count > 0 ? localURL : nil];
	plan.snapshotURL = snapshotURL;
	plan.cacheRepository = repo;
	plan.recordsCompletion = requiredPaths == nil;
	if (absentOptional.count > 0) {
		plan.absentOptionalFiles = @[ @{ @"repository": repo, @"revision": normalized, @"paths": absentOptional } ];
	}
	return plan;
}

- (nullable NFKDownloadPlan *)planDownloadOfRepo:(NSString *)repo
										revision:(nullable NSString *)revision
										   paths:(nullable NSArray<NSString *> *)paths
										   error:(NSError * _Nullable *)outError
{
	return [self planRepo:repo revision:revision requiredPaths:paths optionalPaths:@[] listings:@{} singleFile:NO error:outError];
}

- (void)planDownloadOfRepo:(NSString *)repo
				  revision:(nullable NSString *)revision
					 paths:(nullable NSArray<NSString *> *)paths
		 completionHandler:(void (^)(NFKDownloadPlan * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKDownloadPlan *plan = [self planDownloadOfRepo:repo revision:revision paths:paths error:&error];
		completionHandler(plan, error);
	});
}

/*! The repository-relative path of a URL's cache folder, @url/<host>/<digest>; nil with an error
	for an empty URL or a host that is not one safe folder name. */
- (nullable NSString *)URLCacheRepositoryForURL:(NSURL *)URL error:(NSError **)outError
{
	NSString *absolute = URL.absoluteString;
	if (absolute.length == 0) {
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the URL is empty"] to:outError];
		return nil;
	}
	NSString *digest = [NFKHFHubHexDigest([absolute dataUsingEncoding:NSUTF8StringEncoding]) substringToIndex:16];
	NSString *host = URL.host.length > 0 ? URL.host.lowercaseString : @"local";
	NSError *refusal = NFKHFHubUnsafePathError(host, @"URL host", NO);
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	return [NSString stringWithFormat:@"%@/%@/%@", NFKHFHubURLCacheFolder, host, digest];
}

- (nullable NSURL *)URLSnapshotURLForURL:(NSURL *)URL error:(NSError **)outError
{
	if (![self requireCacheDirectory:outError]) {
		return nil;
	}
	NSString *cacheRepository = [self URLCacheRepositoryForURL:URL error:outError];
	if (cacheRepository == nil) {
		return nil;
	}
	return [self URL:self.cacheDirectoryURL byAppendingPathString:cacheRepository];
}

/*! The cache location of a URL download; nil with an error without a cache folder, or when the URL's
	host, the file name, or the URL's last path component is unsafe. */
- (nullable NSURL *)localURLForURL:(NSURL *)URL fileName:(nullable NSString *)fileName error:(NSError **)outError
{
	NSURL *snapshotURL = [self URLSnapshotURLForURL:URL error:outError];
	if (snapshotURL == nil) {
		return nil;
	}
	NSError *refusal = fileName.length > 0 ? NFKHFHubUnsafePathError(fileName, @"file name", YES) : nil;
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	NSString *name = fileName.length > 0 ? fileName.lastPathComponent : URL.lastPathComponent;
	if (name.length == 0 || [name isEqualToString:@"/"]) {
		name = @"download";
	}
	refusal = NFKHFHubUnsafePathError(name, @"file name", NO);
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	return [snapshotURL URLByAppendingPathComponent:name];
}

- (nullable NSURL *)localURLForURL:(NSURL *)URL fileName:(nullable NSString *)fileName
{
	return [self localURLForURL:URL fileName:fileName error:NULL];
}

- (nullable NFKDownloadPlan *)planDownloadOfURL:(NSURL *)URL
										 sha256:(nullable NSString *)expectedSHA256
									   fileName:(nullable NSString *)fileName
									  byteCount:(long long)listedByteCount
										  error:(NSError **)outError
{
	return [self planDownloadOfURL:URL sha256:expectedSHA256 fileName:fileName byteCount:listedByteCount optional:NO error:outError];
}

- (nullable NFKDownloadPlan *)planDownloadOfURL:(NSURL *)URL
										 sha256:(nullable NSString *)expectedSHA256
									   fileName:(nullable NSString *)fileName
									  byteCount:(long long)listedByteCount
									   optional:(BOOL)optional
										  error:(NSError **)outError
{
	NSURL *localURL = [self localURLForURL:URL fileName:fileName error:outError];
	if (localURL == nil) {
		return nil;
	}
	NSHTTPURLResponse *response = nil;
	if ([self fetchMetadataFromURL:URL HTTPMethod:@"HEAD" response:&response error:outError] == nil) {
		return nil;
	}
	NSInteger status = response != nil ? response.statusCode : 200;
	long long byteCount = listedByteCount;
	if (status >= 200 && status < 300) {
		NSString *encoding = [response valueForHTTPHeaderField:@"Content-Encoding"];
		if (byteCount < 0 && encoding.length == 0 && response.expectedContentLength >= 0) {
			byteCount = response.expectedContentLength;
		}
	} else if (status == 401 || status == 403 || status == 404) {
		BOOL hadToken = [[self requestForURL:URL] valueForHTTPHeaderField:@"Authorization"] != nil;
		[self propagateError:[NFKHFHubTransfer errorForHTTPStatus:status subject:URL.absoluteString hadToken:hadToken] to:outError];
		return nil;
	}
	NFKDownloadPlanFile *file = [[NFKDownloadPlanFile alloc] initWithPath:localURL.lastPathComponent
																remoteURL:URL
																 localURL:localURL
																byteCount:byteCount
																   sha256:expectedSHA256
																   cached:[self fileAtURL:localURL isWholeWithByteCount:byteCount]
																 optional:optional];
	file.snapshotURL = localURL.URLByDeletingLastPathComponent;
	NFKDownloadPlan *plan = [[NFKDownloadPlan alloc] initWithRepository:nil revision:nil sourceURL:URL files:@[ file ] localURL:localURL];
	plan.snapshotURL = localURL.URLByDeletingLastPathComponent;
	plan.cacheRepository = [self URLCacheRepositoryForURL:URL error:NULL];
	return plan;
}

- (nullable NFKDownloadPlan *)planDownloadOfURL:(NSURL *)URL
										 sha256:(nullable NSString *)expectedSHA256
										  error:(NSError * _Nullable *)outError
{
	return [self planDownloadOfURL:URL sha256:expectedSHA256 fileName:nil byteCount:NFKDownloadUnknownByteCount error:outError];
}

- (void)planDownloadOfURL:(NSURL *)URL
				   sha256:(nullable NSString *)expectedSHA256
		completionHandler:(void (^)(NFKDownloadPlan * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKDownloadPlan *plan = [self planDownloadOfURL:URL sha256:expectedSHA256 error:&error];
		completionHandler(plan, error);
	});
}

/*! A source the hub refuses to plan or download, with the reason; nil for one it takes. */
- (nullable NSError *)refusalForModelSource:(NFKModelSource *)source
{
	if (source.requiresConversion) {
		NSString *name = source.repository ?: source.URL.absoluteString ?: @"the source";
		NSString *reason = [NSString stringWithFormat:@"%@ needs a converter before InferKit loads it", name];
		return [NFKDownload errorWithCode:NFKDownloadErrorRequiresConversion reason:reason];
	}
	if ([source.kind isEqualToString:NFKModelSourceKindHuggingFace] && source.repository.length == 0) {
		return [NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the Hugging Face source names no repository"];
	}
	if ([source.kind isEqualToString:NFKModelSourceKindURL] && source.URL == nil) {
		return [NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the URL source names no URL"];
	}
	NSArray<NSString *> *kinds = @[ NFKModelSourceKindHuggingFace, NFKModelSourceKindURL, NFKModelSourceKindSystem, NFKModelSourceKindNone ];
	if (![kinds containsObject:source.kind]) {
		NSString *reason = [NSString stringWithFormat:@"the hub does not download a source of kind %@", source.kind];
		return [NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:reason];
	}
	return nil;
}

- (BOOL)modelSourceHoldsNoFiles:(NFKModelSource *)source
{
	return [source.kind isEqualToString:NFKModelSourceKindSystem] || [source.kind isEqualToString:NFKModelSourceKindNone];
}

- (nullable NFKDownloadPlan *)planDownloadOfModelSource:(NFKModelSource *)source error:(NSError * _Nullable *)outError
{
	NSError *refusal = [self refusalForModelSource:source];
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	if ([self modelSourceHoldsNoFiles:source]) {
		return [[NFKDownloadPlan alloc] initWithRepository:nil revision:nil sourceURL:nil files:@[] localURL:nil];
	}
	BOOL singleFile = [source.layout isEqualToString:NFKModelSourceLayoutFile];
	if ([source.kind isEqualToString:NFKModelSourceKindURL]) {
		NFKModelFile *file = source.files.firstObject;
		return [self planDownloadOfURL:source.URL
								sha256:file.sha256
							  fileName:file.path
							 byteCount:file.byteCount != nil ? file.byteCount.longLongValue : NFKDownloadUnknownByteCount
								 error:outError];
	}
	if (source.files.count == 0) {
		return [self planRepo:source.repository revision:source.revision requiredPaths:nil optionalPaths:@[]
					 listings:@{} singleFile:singleFile error:outError];
	}
	NSArray<NFKHFHubSourcePart *> *parts = [self partsOfModelSource:source];
	if (parts.count == 1 && [self part:parts.firstObject isTheRepositoryOfSource:source]) {
		return [self planPart:parts.firstObject singleFile:singleFile error:outError];
	}
	return [self planParts:parts ofModelSource:source singleFile:singleFile error:outError];
}

#pragma mark Source parts

/*! The places a source's files come from, in the order of the first file from each. A file names its
	own place through its repository, revision, or URL; the rest come from the source. A URL file is a
	place of its own. */
- (NSArray<NFKHFHubSourcePart *> *)partsOfModelSource:(NFKModelSource *)source
{
	if ([self modelSourceHoldsNoFiles:source]) {
		return @[];
	}
	if ([source.kind isEqualToString:NFKModelSourceKindURL]) {
		if (source.URL == nil) {
			return @[];
		}
		NFKHFHubSourcePart *part = [[NFKHFHubSourcePart alloc] init];
		part.URL = source.URL;
		part.files = source.files.count > 0 ? [NSMutableArray arrayWithObject:source.files.firstObject] : [NSMutableArray array];
		return @[ part ];
	}
	if (source.files.count == 0) {
		NFKHFHubSourcePart *part = [[NFKHFHubSourcePart alloc] init];
		part.repository = source.repository;
		part.revision = source.revision;
		return @[ part ];
	}
	NSMutableArray<NFKHFHubSourcePart *> *parts = [NSMutableArray array];
	NSMutableDictionary<NSString *, NFKHFHubSourcePart *> *partsByKey = [NSMutableDictionary dictionary];
	for (NFKModelFile *file in source.files) {
		NFKHFHubSourcePart *candidate = [[NFKHFHubSourcePart alloc] init];
		NSString *key = nil;
		if (file.URL != nil) {
			candidate.URL = file.URL;
			key = [NSString stringWithFormat:@"url\n%@\n%@", file.URL.absoluteString, file.path];
		} else {
			candidate.repository = file.repository ?: source.repository;
			candidate.revision = file.repository != nil ? file.revision : (file.revision ?: source.revision);
			key = [NSString stringWithFormat:@"repository\n%@\n%@", candidate.repository ?: @"",
				   [self normalizedRevision:candidate.revision]];
		}
		NFKHFHubSourcePart *part = partsByKey[key];
		if (part == nil) {
			part = candidate;
			part.files = [NSMutableArray array];
			partsByKey[key] = part;
			[parts addObject:part];
		}
		[part.files addObject:file];
	}
	return parts;
}

- (BOOL)part:(NFKHFHubSourcePart *)part isTheRepositoryOfSource:(NFKModelSource *)source
{
	return part.URL == nil && source.repository != nil && [part.repository isEqualToString:source.repository]
		&& [[self normalizedRevision:part.revision] isEqualToString:[self normalizedRevision:source.revision]];
}

/*! The local file name of a URL part's file. */
- (nullable NSURL *)localURLOfURLPart:(NFKHFHubSourcePart *)part error:(NSError **)outError
{
	return [self localURLForURL:part.URL fileName:part.files.firstObject.path error:outError];
}

/*! Why a part cannot be planned, read, or removed: an unsafe repository, revision, path, or URL; nil
	when it can. */
- (nullable NSError *)refusalOfPart:(NFKHFHubSourcePart *)part
{
	if (part.URL != nil) {
		NSError *refusal = nil;
		return [self localURLOfURLPart:part error:&refusal] == nil ? refusal : nil;
	}
	NSError *refusal = [self refusalForRepo:part.repository revision:part.revision path:nil];
	for (NFKModelFile *file in part.files) {
		if (refusal != nil) {
			break;
		}
		refusal = NFKHFHubUnsafePathError(file.path, @"path", YES);
	}
	return refusal;
}

/*! Plans one repository part: its listed files, required and optional. */
- (nullable NFKDownloadPlan *)planPart:(NFKHFHubSourcePart *)part singleFile:(BOOL)singleFile error:(NSError **)outError
{
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	NSMutableArray<NSString *> *optional = [NSMutableArray array];
	NSMutableDictionary<NSString *, NFKModelFile *> *listings = [NSMutableDictionary dictionary];
	for (NFKModelFile *file in part.files) {
		[(file.optional ? optional : required) addObject:file.path];
		listings[file.path] = file;
	}
	return [self planRepo:part.repository revision:part.revision requiredPaths:required optionalPaths:optional
				 listings:listings singleFile:singleFile error:outError];
}

/*! The planned files of one part. An optional URL file the server does not have is left out; the
	optional files a repository part lacks join absentOptionalFiles. */
- (nullable NSArray<NFKDownloadPlanFile *> *)planFilesOfPart:(NFKHFHubSourcePart *)part
										 absentOptionalFiles:(NSMutableArray<NSDictionary<NSString *, id> *> *)absentOptionalFiles
													   error:(NSError **)outError
{
	if (part.URL == nil) {
		NFKDownloadPlan *plan = [self planPart:part singleFile:NO error:outError];
		[absentOptionalFiles addObjectsFromArray:plan.absentOptionalFiles ?: @[]];
		return plan.files;
	}
	NFKModelFile *file = part.files.firstObject;
	NSError *error = nil;
	NFKDownloadPlan *plan = [self planDownloadOfURL:part.URL
											 sha256:file.sha256
										   fileName:file.path
										  byteCount:file.byteCount != nil ? file.byteCount.longLongValue : NFKDownloadUnknownByteCount
										   optional:file.optional
											  error:&error];
	if (plan != nil) {
		return plan.files;
	}
	if (file.optional && [error.domain isEqualToString:NFKDownloadErrorDomain] && error.code == NFKDownloadErrorNotFound) {
		return @[];
	}
	[self propagateError:error to:outError];
	return nil;
}

/*! Plans a source whose files span several repositories or URLs. Every place is checked before the
	first request. The plan's localURL is the source's own snapshot; each file lands in its own
	place's. */
- (nullable NFKDownloadPlan *)planParts:(NSArray<NFKHFHubSourcePart *> *)parts
						  ofModelSource:(NFKModelSource *)source
							 singleFile:(BOOL)singleFile
								  error:(NSError **)outError
{
	NSURL *sourceSnapshotURL = [self snapshotURLForRepo:source.repository revision:source.revision error:outError];
	if (sourceSnapshotURL == nil) {
		return nil;
	}
	for (NFKHFHubSourcePart *part in parts) {
		NSError *refusal = [self refusalOfPart:part];
		if (refusal != nil) {
			[self propagateError:refusal to:outError];
			return nil;
		}
	}
	NSMutableArray<NFKDownloadPlanFile *> *files = [NSMutableArray array];
	NSMutableArray<NSDictionary<NSString *, id> *> *absentOptionalFiles = [NSMutableArray array];
	for (NFKHFHubSourcePart *part in parts) {
		NSArray<NFKDownloadPlanFile *> *partFiles = [self planFilesOfPart:part absentOptionalFiles:absentOptionalFiles
																	 error:outError];
		if (partFiles == nil) {
			return nil;
		}
		[files addObjectsFromArray:partFiles];
	}
	NSURL *localURL = singleFile && files.count == 1 ? files.firstObject.localURL : sourceSnapshotURL;
	NFKDownloadPlan *plan = [[NFKDownloadPlan alloc] initWithRepository:source.repository
															   revision:[self normalizedRevision:source.revision]
															  sourceURL:nil
																  files:files
															   localURL:files.count > 0 ? localURL : nil];
	plan.snapshotURL = sourceSnapshotURL;
	plan.cacheRepository = source.repository;
	plan.absentOptionalFiles = absentOptionalFiles;
	return plan;
}

- (void)planDownloadOfModelSource:(NFKModelSource *)source
				completionHandler:(void (^)(NFKDownloadPlan * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKDownloadPlan *plan = [self planDownloadOfModelSource:source error:&error];
		completionHandler(plan, error);
	});
}

#pragma mark Downloads

/*! Why the plan cannot start, or nil. */
- (nullable NSError *)admissionErrorForPlan:(NFKDownloadPlan *)plan
{
	if (plan.files.count == 0) {
		return nil;
	}
	if (self.cacheDirectoryURL == nil) {
		return [NSError errorWithDomain:NFKInferenceErrorDomain
								   code:kNFKError_InferenceNotReady
							   userInfo:@{ NSLocalizedDescriptionKey: @"no cache directory is set" }];
	}
	long long limit = self.cacheSizeLimit;
	if (limit >= 0 && plan.totalByteCount > limit) {
		NSString *reason = [NSString stringWithFormat:@"the download holds %lld bytes and the cache size limit is %lld",
							plan.totalByteCount, limit];
		return [NFKDownload errorWithCode:NFKDownloadErrorExceedsCacheSizeLimit reason:reason];
	}
	long long available = [self availableCapacityOfCache];
	if (available >= 0 && plan.remainingByteCount > available) {
		NSString *reason = [NSString stringWithFormat:@"the download needs %lld bytes and the cache volume has %lld free",
							plan.remainingByteCount, available];
		return [NFKDownload errorWithCode:NFKDownloadErrorInsufficientStorage reason:reason];
	}
	return nil;
}

/*! Fetches every file of the plan on the calling thread, with the download current. */
- (nullable NSError *)runPlan:(NFKDownloadPlan *)plan download:(NFKDownload *)download
{
	NSError *admission = [self admissionErrorForPlan:plan];
	if (admission != nil) {
		return admission;
	}
	if (plan.snapshotURL != nil) {
		[download registerSnapshotURL:plan.snapshotURL];
	}
	for (NFKDownloadPlanFile *file in plan.files) {
		[download registerSnapshotURL:[self snapshotURLOfPlanFile:file inPlan:plan]];
	}
	NSError *error = nil;
	[download becomeCurrent];
	@try {
		for (NFKDownloadPlanFile *file in plan.files) {
			if (download.isStopped) {
				return download.stopError;
			}
			NSURL *fetched = [self fetchFileFromURL:file.remoteURL
										   localURL:file.localURL
										snapshotURL:[self snapshotURLOfPlanFile:file inPlan:plan]
											   path:file.path
											 sha256:file.sha256
								  expectedByteCount:file.byteCount
											  error:&error];
			if (fetched == nil) {
				return error;
			}
		}
	} @finally {
		[download resignCurrent];
	}
	if (plan.recordsCompletion && plan.repository != nil) {
		[self recordCompleteDownloadOfRepo:plan.repository revision:plan.revision
									 paths:[plan.files valueForKey:@"path"] error:NULL];
	}
	for (NSDictionary<NSString *, id> *absent in plan.absentOptionalFiles) {
		[self recordCompleteDownloadOfRepo:absent[@"repository"] revision:absent[@"revision"] paths:@[]
							   absentPaths:absent[@"paths"] error:NULL];
	}
	return nil;
}

/*! The snapshot a plan file caches in: its own place's, else the plan's, else its folder. */
- (NSURL *)snapshotURLOfPlanFile:(NFKDownloadPlanFile *)file inPlan:(NFKDownloadPlan *)plan
{
	return file.snapshotURL ?: plan.snapshotURL ?: file.localURL.URLByDeletingLastPathComponent;
}

- (NFKDownload *)downloadWithPlan:(NFKDownloadPlan *)plan
				  progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
				completionHandler:(nullable void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:progressHandler];
	download.completionHandler = completionHandler;
	[download beginActivity];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[self runDownload:download withPlan:plan];
	});
	return download;
}

- (void)runDownload:(NFKDownload *)download withPlan:(NFKDownloadPlan *)plan
{
	[download adoptPlan:plan];
	[download setResultURL:plan.localURL];
	NSError *error = [self runPlan:plan download:download];
	[download endActivityWithError:error];
}

/*! Plans on a background queue with the download current, then downloads the plan. */
- (NFKDownload *)downloadPlannedBy:(NFKDownloadPlan * _Nullable (^)(NSError **error))planner
				   progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
				 completionHandler:(nullable void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:progressHandler];
	download.completionHandler = completionHandler;
	[download markPlanning];
	[download beginActivity];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		__block NSError *error = nil;
		__block NFKDownloadPlan *plan = nil;
		[download performAsCurrent:^{
			NSError *planError = nil;
			plan = planner(&planError);
			error = planError;
		}];
		if (plan == nil) {
			[download endActivityWithError:error];
			return;
		}
		[self runDownload:download withPlan:plan];
	});
	return download;
}

- (NFKDownload *)downloadRepo:(NSString *)repo
					 revision:(nullable NSString *)revision
						paths:(nullable NSArray<NSString *> *)paths
			  progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
			completionHandler:(nullable void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	return [self downloadPlannedBy:^NFKDownloadPlan *(NSError **error) {
		return [self planDownloadOfRepo:repo revision:revision paths:paths error:error];
	} progressHandler:progressHandler completionHandler:completionHandler];
}

- (NFKDownload *)downloadModelSource:(NFKModelSource *)source
					 progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
				   completionHandler:(nullable void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	return [self downloadPlannedBy:^NFKDownloadPlan *(NSError **error) {
		return [self planDownloadOfModelSource:source error:error];
	} progressHandler:progressHandler completionHandler:completionHandler];
}

#pragma mark Snapshot contents

/*! The path of url relative to base; nil when url is not under base. */
- (nullable NSString *)relativePathOfURL:(NSURL *)url underURL:(NSURL *)base
{
	NSString *resolvedParent = url.URLByDeletingLastPathComponent.URLByResolvingSymlinksInPath.path;
	NSArray<NSString *> *candidates = @[ url.path, [resolvedParent stringByAppendingPathComponent:url.lastPathComponent] ];
	for (NSString *basePath in @[ base.path, base.URLByResolvingSymlinksInPath.path ]) {
		NSString *prefix = [basePath stringByAppendingString:@"/"];
		for (NSString *candidate in candidates) {
			if ([candidate hasPrefix:prefix]) {
				return [candidate substringFromIndex:prefix.length];
			}
		}
	}
	return nil;
}

/*! The model files in a snapshot, as paths relative to it: regular files and symbolic links, less
	the hub's bookkeeping files and in-flight .download files. */
- (NSArray<NSString *> *)contentPathsInSnapshotURL:(NSURL *)snapshotURL
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsRegularFileKey, NSURLIsSymbolicLinkKey ];
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:snapshotURL
																   includingPropertiesForKeys:keys
																					  options:0
																				 errorHandler:nil];
	NSArray<NSString *> *bookkeeping = NFKHFHubBookkeepingNames();
	NSMutableArray<NSString *> *paths = [NSMutableArray array];
	for (NSURL *url in enumerator) {
		NSDictionary<NSURLResourceKey, id> *values = [url resourceValuesForKeys:keys error:NULL];
		BOOL isFile = [values[NSURLIsRegularFileKey] boolValue] || [values[NSURLIsSymbolicLinkKey] boolValue];
		if (!isFile || [url.pathExtension isEqualToString:@"download"]) {
			continue;
		}
		NSString *relative = [self relativePathOfURL:url underURL:snapshotURL];
		if (relative == nil || [bookkeeping containsObject:relative]) {
			continue;
		}
		[paths addObject:relative];
	}
	[paths sortUsingSelector:@selector(compare:)];
	return paths;
}

/*! The partial .download files in a snapshot folder, by path, with the content path each would become. */
- (NSDictionary<NSString *, NSString *> *)partialPathsInSnapshotURL:(NSURL *)snapshotURL
{
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:snapshotURL
																   includingPropertiesForKeys:@[ NSURLIsRegularFileKey ]
																					  options:0
																				 errorHandler:nil];
	NSMutableDictionary<NSString *, NSString *> *partials = [NSMutableDictionary dictionary];
	for (NSURL *url in enumerator) {
		NSNumber *isFile = nil;
		[url getResourceValue:&isFile forKey:NSURLIsRegularFileKey error:NULL];
		NSString *relative = [self relativePathOfURL:url underURL:snapshotURL];
		if (isFile.boolValue && [url.pathExtension isEqualToString:@"download"] && relative != nil) {
			partials[relative] = relative.stringByDeletingPathExtension;
		}
	}
	return partials;
}

/*! The paths of content matching each of paths (globs or literal paths), in content order. */
- (NSArray<NSString *> *)paths:(NSArray<NSString *> *)paths matchingContent:(NSArray<NSString *> *)content
{
	NSMutableOrderedSet<NSString *> *matched = [NSMutableOrderedSet orderedSet];
	NSSet<NSString *> *contentSet = [NSSet setWithArray:content];
	for (NSString *path in paths) {
		if (!NFKHFHubIsGlob(path)) {
			if ([contentSet containsObject:path]) {
				[matched addObject:path];
			}
			continue;
		}
		for (NSString *candidate in content) {
			if (NFKHFHubGlobMatches(path, candidate)) {
				[matched addObject:candidate];
			}
		}
	}
	return matched.array;
}

/*! The allocated bytes of a regular file; 0 for anything else. */
- (long long)allocatedSizeOfFileAtURL:(NSURL *)url
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsRegularFileKey, NSURLTotalFileAllocatedSizeKey ];
	NSDictionary<NSURLResourceKey, id> *values = [url resourceValuesForKeys:keys error:NULL];
	if (![values[NSURLIsRegularFileKey] boolValue]) {
		return 0;
	}
	return [values[NSURLTotalFileAllocatedSizeKey] longLongValue];
}

- (BOOL)isSymbolicLinkAtURL:(NSURL *)url
{
	NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
	return [attributes[NSFileType] isEqualToString:NSFileTypeSymbolicLink];
}

/*! The completion record of a snapshot: the paths of its complete download under "paths" and the
	optional paths the server refused under "absent". A record written as a bare array holds paths
	only. */
- (NSDictionary<NSString *, NSArray<NSString *> *> *)completionRecordInSnapshotURL:(NSURL *)snapshotURL
{
	NSData *data = [NSData dataWithContentsOfURL:[snapshotURL URLByAppendingPathComponent:NFKHFHubCompleteMarkerName]];
	id record = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	NSArray *paths = nil;
	NSArray *absent = nil;
	if ([record isKindOfClass:NSArray.class]) {
		paths = record;
	} else if ([record isKindOfClass:NSDictionary.class]) {
		paths = [record[@"paths"] isKindOfClass:NSArray.class] ? record[@"paths"] : nil;
		absent = [record[@"absent"] isKindOfClass:NSArray.class] ? record[@"absent"] : nil;
	}
	NSPredicate *strings = [NSPredicate predicateWithBlock:^BOOL(id path, NSDictionary *bindings) {
		return [path isKindOfClass:NSString.class];
	}];
	return @{ @"paths": [paths filteredArrayUsingPredicate:strings] ?: @[],
			  @"absent": [absent filteredArrayUsingPredicate:strings] ?: @[] };
}

/*! The paths a complete-download record in the snapshot names. */
- (NSArray<NSString *> *)recordedCompletePathsInSnapshotURL:(NSURL *)snapshotURL
{
	return [self completionRecordInSnapshotURL:snapshotURL][@"paths"];
}

- (BOOL)recordCompleteDownloadOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   paths:(NSArray<NSString *> *)paths
							   error:(NSError * _Nullable *)outError
{
	return [self recordCompleteDownloadOfRepo:repo revision:revision paths:paths absentPaths:@[] error:outError];
}

- (BOOL)recordCompleteDownloadOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   paths:(NSArray<NSString *> *)paths
						 absentPaths:(NSArray<NSString *> *)absentPaths
							   error:(NSError * _Nullable *)outError
{
	for (NSString *path in [paths arrayByAddingObjectsFromArray:absentPaths]) {
		NSError *refusal = NFKHFHubUnsafePathError(path, @"recorded path", YES);
		if (refusal != nil) {
			return [self propagateError:refusal to:outError];
		}
	}
	return [self updateCompletionRecordOfRepo:repo revision:revision error:outError
								   withBlock:^(NSMutableOrderedSet<NSString *> *recorded, NSMutableOrderedSet<NSString *> *absent) {
		[recorded addObjectsFromArray:paths];
		[absent addObjectsFromArray:absentPaths];
		[absent minusOrderedSet:recorded];
	}];
}

- (NSArray<NSString *> *)recordedAbsentPathsOfRepo:(NSString *)repo revision:(nullable NSString *)revision
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return @[];
	}
	__block NSArray<NSString *> *absent = @[];
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		absent = [self completionRecordInSnapshotURL:snapshotURL][@"absent"];
	}];
	return absent;
}

- (BOOL)forgetRecordedAbsentPathsOfRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	if ([self recordedAbsentPathsOfRepo:repo revision:revision].count == 0) {
		return YES;
	}
	return [self updateCompletionRecordOfRepo:repo revision:revision error:outError
								   withBlock:^(NSMutableOrderedSet<NSString *> *recorded, NSMutableOrderedSet<NSString *> *absent) {
		[absent removeAllObjects];
	}];
}

/*! Reads, changes, and writes back a snapshot's completion record under one process-wide lock. */
- (BOOL)updateCompletionRecordOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   error:(NSError **)outError
						   withBlock:(void (NS_NOESCAPE ^)(NSMutableOrderedSet<NSString *> *recorded,
														   NSMutableOrderedSet<NSString *> *absent))change
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return NO;
	}
	__block NSError *writeError = nil;
	__block BOOL written = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:&writeError]) {
			return;
		}
		@synchronized (NFKHFHub.class) {
			NSDictionary<NSString *, NSArray<NSString *> *> *record = [self completionRecordInSnapshotURL:snapshotURL];
			NSMutableOrderedSet<NSString *> *recorded = [NSMutableOrderedSet orderedSetWithArray:record[@"paths"]];
			NSMutableOrderedSet<NSString *> *absent = [NSMutableOrderedSet orderedSetWithArray:record[@"absent"]];
			change(recorded, absent);
			NSData *data = [NSJSONSerialization dataWithJSONObject:@{ @"paths": recorded.array, @"absent": absent.array }
														   options:NSJSONWritingSortedKeys
															 error:&writeError];
			if (data == nil) {
				return;
			}
			[NSFileManager.defaultManager createDirectoryAtURL:snapshotURL withIntermediateDirectories:YES attributes:nil error:NULL];
			written = [data writeToURL:[snapshotURL URLByAppendingPathComponent:NFKHFHubCompleteMarkerName]
							   options:NSDataWritingAtomic
								 error:&writeError];
		}
	}];
	if (!written) {
		return [self propagateError:writeError to:outError];
	}
	return YES;
}

#pragma mark Status

/*! The bytes a present file takes, at its target when it is a link. */
- (long long)onDiskSizeOfURL:(NSURL *)url
{
	return [self allocatedSizeOfFileAtURL:url.URLByResolvingSymlinksInPath];
}

- (NFKDownloadStatus *)statusInSnapshotURL:(nullable NSURL *)snapshotURL
							 requiredPaths:(nullable NSArray<NSString *> *)requiredPaths
							 optionalPaths:(NSArray<NSString *> *)optionalPaths
								  localURL:(nullable NSURL *)localURL
{
	if (snapshotURL == nil) {
		return [[NFKDownloadStatus alloc] initWithPresence:NFKDownloadPresenceAbsent byteCount:0 presentPaths:@[]
											  missingPaths:requiredPaths ?: @[] localURL:nil];
	}
	__block NFKDownloadStatus *status = nil;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSArray<NSString *> *content = [self contentPathsInSnapshotURL:snapshotURL];
		NSArray<NSString *> *required = requiredPaths;
		if (required == nil) {
			required = [self recordedCompletePathsInSnapshotURL:snapshotURL];
		}
		NSSet<NSString *> *contentSet = [NSSet setWithArray:content];
		NSMutableArray<NSString *> *missing = [NSMutableArray array];
		NSMutableOrderedSet<NSString *> *present = [NSMutableOrderedSet orderedSet];
		for (NSString *path in required) {
			if (NFKHFHubIsGlob(path)) {
				NSArray<NSString *> *matches = [self paths:@[ path ] matchingContent:content];
				if (matches.count == 0) {
					[missing addObject:path];
				}
				[present addObjectsFromArray:matches];
				continue;
			}
			BOOL exists = [contentSet containsObject:path]
				&& [NSFileManager.defaultManager fileExistsAtPath:[snapshotURL URLByAppendingPathComponent:path].path];
			if (exists) {
				[present addObject:path];
			} else {
				[missing addObject:path];
			}
		}
		[present addObjectsFromArray:[self paths:optionalPaths matchingContent:content]];
		if (requiredPaths == nil) {
			[present addObjectsFromArray:content];
		}
		long long bytes = 0;
		for (NSString *path in present) {
			bytes += [self onDiskSizeOfURL:[snapshotURL URLByAppendingPathComponent:path]];
		}
		NSMutableArray<NSString *> *partial = [NSMutableArray array];
		long long partialBytes = 0;
		for (NSString *path in missing) {
			long long kept = [self keptByteCountOfPartialURL:[[snapshotURL URLByAppendingPathComponent:path]
															 URLByAppendingPathExtension:@"download"]];
			if (kept > 0) {
				[partial addObject:path];
				partialBytes += kept;
			}
		}
		NFKDownloadPresence presence = NFKDownloadPresenceAbsent;
		if (present.count > 0 || partial.count > 0) {
			presence = missing.count == 0 && required.count > 0 ? NFKDownloadPresenceComplete : NFKDownloadPresencePartial;
		}
		if (requiredPaths != nil && required.count == 0 && present.count > 0) {
			presence = NFKDownloadPresenceComplete;
		}
		status = [[NFKDownloadStatus alloc] initWithPresence:presence
												   byteCount:bytes
												presentPaths:present.array
												missingPaths:missing
												partialPaths:partial
											partialByteCount:partialBytes
													localURL:localURL ?: snapshotURL];
	}];
	return status ?: [[NFKDownloadStatus alloc] initWithPresence:NFKDownloadPresenceAbsent byteCount:0 presentPaths:@[]
													missingPaths:requiredPaths ?: @[] localURL:nil];
}

- (NFKDownloadStatus *)downloadStatusForRepo:(NSString *)repo
									revision:(nullable NSString *)revision
									   paths:(nullable NSArray<NSString *> *)paths
{
	return [self statusInSnapshotURL:[self snapshotURLForRepo:repo revision:revision]
					   requiredPaths:paths
					   optionalPaths:@[]
							localURL:nil];
}

- (NFKDownloadStatus *)downloadStatusForURL:(NSURL *)URL fileName:(nullable NSString *)fileName
{
	NSURL *localURL = [self localURLForURL:URL fileName:fileName];
	if (localURL == nil) {
		return [self statusInSnapshotURL:nil requiredPaths:@[] optionalPaths:@[] localURL:nil];
	}
	return [self statusInSnapshotURL:localURL.URLByDeletingLastPathComponent
					   requiredPaths:@[ localURL.lastPathComponent ]
					   optionalPaths:@[]
							localURL:localURL];
}

- (NFKDownloadStatus *)downloadStatusForURL:(NSURL *)URL
{
	return [self downloadStatusForURL:URL fileName:nil];
}

- (NFKDownloadStatus *)downloadStatusForModelSource:(NFKModelSource *)source
{
	if ([self modelSourceHoldsNoFiles:source]) {
		return [[NFKDownloadStatus alloc] initWithPresence:NFKDownloadPresenceComplete byteCount:0 presentPaths:@[]
											  missingPaths:@[] localURL:nil];
	}
	if ([source.kind isEqualToString:NFKModelSourceKindURL] && source.URL != nil) {
		return [self downloadStatusForURL:source.URL fileName:source.files.firstObject.path];
	}
	if (source.repository.length == 0) {
		return [self statusInSnapshotURL:nil requiredPaths:@[] optionalPaths:@[] localURL:nil];
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:source.repository revision:source.revision];
	if (source.files.count == 0) {
		return [self statusInSnapshotURL:snapshotURL requiredPaths:nil optionalPaths:@[] localURL:nil];
	}
	NSArray<NFKHFHubSourcePart *> *parts = [self partsOfModelSource:source];
	NSMutableOrderedSet<NSString *> *present = [NSMutableOrderedSet orderedSet];
	NSMutableArray<NSString *> *missing = [NSMutableArray array];
	NSMutableArray<NSString *> *partial = [NSMutableArray array];
	NSMutableArray<NSString *> *unchecked = [NSMutableArray array];
	long long bytes = 0;
	long long partialBytes = 0;
	for (NFKHFHubSourcePart *part in parts) {
		NFKDownloadStatus *status = [self statusOfPart:part];
		[present addObjectsFromArray:status.presentPaths];
		[missing addObjectsFromArray:status.missingPaths];
		[partial addObjectsFromArray:status.partialPaths];
		[unchecked addObjectsFromArray:status.uncheckedOptionalPaths];
		bytes += status.byteCount;
		partialBytes += status.partialByteCount;
	}
	NSURL *localURL = snapshotURL;
	if ([source.layout isEqualToString:NFKModelSourceLayoutFile] && source.files.count == 1 && parts.count == 1) {
		NFKHFHubSourcePart *part = parts.firstObject;
		localURL = part.URL != nil ? [self localURLOfURLPart:part error:NULL]
								   : [self localURLForRepo:part.repository revision:part.revision path:part.files.firstObject.path];
	}
	NFKDownloadPresence presence = NFKDownloadPresenceAbsent;
	if (present.count > 0 || partial.count > 0) {
		presence = missing.count == 0 && unchecked.count == 0 ? NFKDownloadPresenceComplete : NFKDownloadPresencePartial;
	}
	return [[NFKDownloadStatus alloc] initWithPresence:presence
											 byteCount:bytes
										  presentPaths:present.array
										  missingPaths:missing
										  partialPaths:partial
									  partialByteCount:partialBytes
								uncheckedOptionalPaths:unchecked
											  localURL:snapshotURL != nil ? localURL : nil];
}

/*! Whether one part's files are in the cache, read in its own repository's or URL's snapshot. */
- (NFKDownloadStatus *)statusOfPart:(NFKHFHubSourcePart *)part
{
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	NSMutableArray<NSString *> *optional = [NSMutableArray array];
	for (NFKModelFile *file in part.files) {
		[(file.optional ? optional : required) addObject:file.path];
	}
	if (part.URL == nil) {
		NSURL *snapshotURL = [self snapshotURLForRepo:part.repository revision:part.revision];
		NFKDownloadStatus *status = [self statusInSnapshotURL:snapshotURL requiredPaths:required optionalPaths:optional
													 localURL:nil];
		NSArray<NSString *> *unchecked = [self uncheckedPaths:optional ofStatus:status inSnapshotURL:snapshotURL];
		if (unchecked.count == 0) {
			return status;
		}
		NFKDownloadPresence presence = status.presence == NFKDownloadPresenceComplete ? NFKDownloadPresencePartial : status.presence;
		return [[NFKDownloadStatus alloc] initWithPresence:presence
												 byteCount:status.byteCount
											  presentPaths:status.presentPaths
											  missingPaths:status.missingPaths
											  partialPaths:status.partialPaths
										  partialByteCount:status.partialByteCount
									uncheckedOptionalPaths:unchecked
												  localURL:status.localURL];
	}
	NSURL *localURL = [self localURLOfURLPart:part error:NULL];
	if (localURL == nil) {
		return [self statusInSnapshotURL:nil requiredPaths:required optionalPaths:@[] localURL:nil];
	}
	NSArray<NSString *> *name = @[ localURL.lastPathComponent ];
	BOOL optionalFile = part.files.firstObject.optional;
	return [self statusInSnapshotURL:localURL.URLByDeletingLastPathComponent
					   requiredPaths:optionalFile ? @[] : name
					   optionalPaths:optionalFile ? name : @[]
							localURL:localURL];
}

/*! The optional paths, globs aside, that are neither in the snapshot nor in its record of the files
	the server lacks. */
- (NSArray<NSString *> *)uncheckedPaths:(NSArray<NSString *> *)optionalPaths
							   ofStatus:(NFKDownloadStatus *)status
						  inSnapshotURL:(nullable NSURL *)snapshotURL
{
	if (snapshotURL == nil || optionalPaths.count == 0) {
		return @[];
	}
	NSSet<NSString *> *present = [NSSet setWithArray:status.presentPaths];
	__block NSSet<NSString *> *absent = [NSSet set];
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		absent = [NSSet setWithArray:[self completionRecordInSnapshotURL:snapshotURL][@"absent"]];
	}];
	NSMutableArray<NSString *> *unchecked = [NSMutableArray array];
	for (NSString *path in optionalPaths) {
		if (!NFKHFHubIsGlob(path) && ![present containsObject:path] && ![absent containsObject:path]) {
			[unchecked addObject:path];
		}
	}
	return unchecked;
}

#pragma mark Removal plans

/*! The paths a part names in its snapshot: the listed paths of a repository part, nil for a part that
	names the whole repository, and the cached file name of a URL part. */
- (nullable NSSet<NSString *> *)listedPathsOfPart:(NFKHFHubSourcePart *)part
{
	if (part.URL != nil) {
		NSURL *localURL = [self localURLOfURLPart:part error:NULL];
		return localURL != nil ? [NSSet setWithObject:localURL.lastPathComponent] : [NSSet set];
	}
	if (part.files == nil) {
		return nil;
	}
	return [NSSet setWithArray:[part.files valueForKey:@"path"]];
}

/*! YES when part's files cache in the target's snapshot. */
- (BOOL)part:(NFKHFHubSourcePart *)part isInSnapshotOfTarget:(NFKHFHubRemovalTarget *)target
{
	if (target.repository != nil) {
		return part.URL == nil && [part.repository isEqualToString:target.repository]
			&& [[self normalizedRevision:part.revision] isEqualToString:[self normalizedRevision:target.revision]];
	}
	if (target.sourceURL == nil || part.URL == nil) {
		return NO;
	}
	NSURL *partSnapshotURL = [self URLSnapshotURLForURL:part.URL error:NULL];
	return partSnapshotURL != nil && [partSnapshotURL.path isEqualToString:target.snapshotURL.path];
}

/*! The paths another catalog variant whose download is complete uses in the target's snapshot, each
	with the reason it is kept. A variant's files count wherever they come from: its source's
	repository, another repository, or a URL. */
- (NSDictionary<NSString *, NSString *> *)claimedPathsForTarget:(NFKHFHubRemovalTarget *)target
{
	NSMutableDictionary<NSString *, NSString *> *claims = [NSMutableDictionary dictionary];
	if (target.repository == nil && target.sourceURL == nil) {
		return claims;
	}
	NSSet<NSString *> *targetPaths = target.paths != nil ? [NSSet setWithArray:target.paths] : nil;
	for (NFKModelDescriptor *descriptor in self.modelCatalog.descriptors) {
		for (NFKModelVariant *variant in descriptor.variants) {
			NFKModelSource *source = variant.source;
			BOOL kindDownloads = [source.kind isEqualToString:NFKModelSourceKindHuggingFace]
				|| [source.kind isEqualToString:NFKModelSourceKindURL];
			if (!kindDownloads) {
				continue;
			}
			for (NFKHFHubSourcePart *part in [self partsOfModelSource:source]) {
				if (![self part:part isInSnapshotOfTarget:target]) {
					continue;
				}
				NSSet<NSString *> *listed = [self listedPathsOfPart:part];
				BOOL sameDownload = (listed == nil && targetPaths == nil) || [listed isEqualToSet:targetPaths];
				if (sameDownload || ![self downloadStatusForModelSource:source].holdsEveryRequiredFile) {
					continue;
				}
				NSArray<NSString *> *used = listed != nil ? listed.allObjects : [self recordedCompletePathsInSnapshotURL:target.snapshotURL];
				NSString *reason = [NSString stringWithFormat:@"used by %@ (%@), whose download is complete in this cache",
									descriptor.identifier, variant.identifier];
				for (NSString *path in used) {
					if (claims[path] == nil) {
						claims[path] = reason;
					}
				}
			}
		}
	}
	return claims;
}

/*! Every symbolic link under the cache, by the standardized path of its resolved target. */
- (NSDictionary<NSString *, NSArray<NSString *> *> *)symbolicLinksByTargetInCacheURL:(NSURL *)cacheDirectoryURL
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsSymbolicLinkKey ];
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:cacheDirectoryURL
																   includingPropertiesForKeys:keys
																					  options:0
																				 errorHandler:nil];
	NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *links = [NSMutableDictionary dictionary];
	for (NSURL *url in enumerator) {
		NSNumber *isLink = nil;
		[url getResourceValue:&isLink forKey:NSURLIsSymbolicLinkKey error:NULL];
		if (!isLink.boolValue) {
			continue;
		}
		NSString *target = url.URLByResolvingSymlinksInPath.path;
		NSMutableArray<NSString *> *sources = links[target];
		if (sources == nil) {
			sources = [NSMutableArray array];
			links[target] = sources;
		}
		[sources addObject:url.path];
	}
	return links;
}

- (NSString *)cachePathOfURL:(NSURL *)url cacheURL:(NSURL *)cacheDirectoryURL
{
	return [self relativePathOfURL:url underURL:cacheDirectoryURL] ?: url.path;
}

/*! The links in the cache that resolve to targetPath and stay after the removal: every link outside
	the snapshot, and the links in it that the removal does not take. */
- (NSArray<NSString *> *)remainingLinksToTargetPath:(NSString *)targetPath
									  linksByTarget:(NSDictionary<NSString *, NSArray<NSString *> *> *)linksByTarget
										snapshotURL:(NSURL *)snapshotURL
											 chosen:(NSArray<NSString *> *)chosen
											 claims:(NSDictionary<NSString *, NSString *> *)claims
{
	NSMutableArray<NSString *> *remaining = [NSMutableArray array];
	for (NSString *linkPath in linksByTarget[targetPath]) {
		NSString *relative = [self relativePathOfURL:[NSURL fileURLWithPath:linkPath] underURL:snapshotURL];
		if (relative == nil || ![chosen containsObject:relative] || claims[relative] != nil) {
			[remaining addObject:linkPath];
		}
	}
	return remaining;
}

/*! YES when url lies in a snapshot of the cache other than snapshotURL: a folder between url and the
	cache root, below snapshotURL's own, holds one of the hub's bookkeeping files. */
- (BOOL)isURL:(NSURL *)url inSnapshotOtherThanURL:(NSURL *)snapshotURL cacheURL:(NSURL *)cacheDirectoryURL
{
	NSString *cachePrefix = [cacheDirectoryURL.URLByResolvingSymlinksInPath.path stringByAppendingString:@"/"];
	NSString *ownPath = snapshotURL.URLByResolvingSymlinksInPath.path;
	NSArray<NSString *> *bookkeeping = NFKHFHubBookkeepingNames();
	NSURL *folder = url.URLByResolvingSymlinksInPath.URLByDeletingLastPathComponent;
	while ([folder.path hasPrefix:cachePrefix] && ![folder.path isEqualToString:ownPath]) {
		for (NSString *name in bookkeeping) {
			if ([NSFileManager.defaultManager fileExistsAtPath:[folder.path stringByAppendingPathComponent:name]]) {
				return YES;
			}
		}
		folder = folder.URLByDeletingLastPathComponent;
	}
	return NO;
}

- (nullable NFKRemovalPlan *)removalPlanForTarget:(NFKHFHubRemovalTarget *)target error:(NSError **)outError
{
	if (![self requireCacheDirectory:outError]) {
		return nil;
	}
	if (target.refusal != nil) {
		[self propagateError:target.refusal to:outError];
		return nil;
	}
	if (target.snapshotURL == nil) {
		return [[NFKRemovalPlan alloc] initWithRepository:target.repository revision:target.revision sourceURL:target.sourceURL
											 removedFiles:@[] keptFiles:@[] freedByteCount:0 snapshotURL:nil];
	}
	__block NFKRemovalPlan *plan = nil;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		plan = [self removalPlanForTarget:target cacheURL:cacheDirectoryURL];
	}];
	return plan;
}

- (NFKRemovalPlan *)removalPlanForTarget:(NFKHFHubRemovalTarget *)target cacheURL:(NSURL *)cacheDirectoryURL
{
	NSURL *snapshotURL = target.snapshotURL;
	NSArray<NSString *> *content = [self contentPathsInSnapshotURL:snapshotURL];
	NSArray<NSString *> *chosen = target.paths != nil ? [self paths:target.paths matchingContent:content] : content;
	NSDictionary<NSString *, NSString *> *claims = [self claimedPathsForTarget:target];

	NSMutableArray<NFKCachedFile *> *removed = [NSMutableArray array];
	NSMutableArray<NFKCachedFile *> *kept = [NSMutableArray array];
	NSMutableSet<NSString *> *removedTargets = [NSMutableSet set];
	NSDictionary<NSString *, NSArray<NSString *> *> *linksByTarget = nil;
	NSString *cachePrefix = [cacheDirectoryURL.URLByResolvingSymlinksInPath.path stringByAppendingString:@"/"];
	NSMutableSet<NSString *> *removedPaths = [NSMutableSet set];
	long long freed = 0;

	for (NSString *path in chosen) {
		NSURL *url = [snapshotURL URLByAppendingPathComponent:path];
		NSString *cachePath = [self cachePathOfURL:url cacheURL:cacheDirectoryURL];
		NSString *claim = claims[path];
		if (claim != nil) {
			[kept addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:[self onDiskSizeOfURL:url] reason:claim]];
			continue;
		}
		if (linksByTarget == nil) {
			linksByTarget = [self symbolicLinksByTargetInCacheURL:cacheDirectoryURL];
		}
		if (![self isSymbolicLinkAtURL:url]) {
			NSArray<NSString *> *linksToFile = [self remainingLinksToTargetPath:url.URLByResolvingSymlinksInPath.path
																   linksByTarget:linksByTarget
																	 snapshotURL:snapshotURL
																		  chosen:chosen
																		  claims:claims];
			if (linksToFile.count > 0) {
				NSString *reason = [NSString stringWithFormat:@"linked from %lu other cached file(s)", (unsigned long)linksToFile.count];
				[kept addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:[self allocatedSizeOfFileAtURL:url]
															reason:reason]];
				continue;
			}
			[removedPaths addObject:url.path];
			NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
			NSUInteger links = [attributes[NSFileReferenceCount] unsignedIntegerValue];
			if (links > 1) {
				NSString *reason = [NSString stringWithFormat:@"its storage is shared with %lu other hard link(s), so removing it frees nothing",
									(unsigned long)(links - 1)];
				[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:0 reason:reason]];
				continue;
			}
			long long size = [self allocatedSizeOfFileAtURL:url];
			freed += size;
			[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:size reason:nil]];
			continue;
		}

		[removedPaths addObject:url.path];
		NSURL *targetURL = url.URLByResolvingSymlinksInPath;
		NSString *targetPath = targetURL.path;
		long long targetSize = [self allocatedSizeOfFileAtURL:targetURL];
		if (![targetPath hasPrefix:cachePrefix]) {
			[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:0
														   reason:@"a symbolic link to a file outside the cache folder"]];
			if (![removedTargets containsObject:targetPath]) {
				[removedTargets addObject:targetPath];
				[kept addObject:[[NFKCachedFile alloc] initWithURL:targetURL path:targetPath byteCount:targetSize
															reason:@"outside the cache folder"]];
			}
			continue;
		}
		NSArray<NSString *> *otherLinks = [self remainingLinksToTargetPath:targetPath
															  linksByTarget:linksByTarget
																snapshotURL:snapshotURL
																	 chosen:chosen
																	 claims:claims];
		NSString *targetCachePath = [self cachePathOfURL:targetURL cacheURL:cacheDirectoryURL];
		NSString *targetInSnapshot = [self relativePathOfURL:targetURL underURL:snapshotURL];
		BOOL targetIsChosen = targetInSnapshot != nil && [chosen containsObject:targetInSnapshot] && claims[targetInSnapshot] == nil;
		BOOL targetInOtherSnapshot = targetInSnapshot == nil
			&& [self isURL:targetURL inSnapshotOtherThanURL:snapshotURL cacheURL:cacheDirectoryURL];
		if (otherLinks.count > 0 || (targetInSnapshot != nil && !targetIsChosen) || targetInOtherSnapshot) {
			[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:0
														   reason:@"a symbolic link whose target stays"]];
			if (![removedTargets containsObject:targetPath] && !targetIsChosen) {
				[removedTargets addObject:targetPath];
				NSString *reason = @"a file of the snapshot the removal keeps";
				if (otherLinks.count > 0) {
					reason = [NSString stringWithFormat:@"linked from %lu other cached file(s)", (unsigned long)otherLinks.count];
				} else if (targetInOtherSnapshot) {
					reason = @"a file of another cached snapshot";
				}
				[kept addObject:[[NFKCachedFile alloc] initWithURL:targetURL path:targetCachePath byteCount:targetSize reason:reason]];
			}
			continue;
		}
		[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:0 reason:nil]];
		if (!targetIsChosen && ![removedTargets containsObject:targetPath]) {
			[removedTargets addObject:targetPath];
			freed += targetSize;
			[removed addObject:[[NFKCachedFile alloc] initWithURL:targetURL path:targetCachePath byteCount:targetSize reason:nil]];
		}
	}

	// A paused download's kept partial goes with the file it would become.
	NSDictionary<NSString *, NSString *> *partials = [self partialPathsInSnapshotURL:snapshotURL];
	BOOL keepsPartial = NO;
	for (NSString *partialPath in [partials.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSString *base = partials[partialPath];
		BOOL named = target.paths == nil || [self paths:target.paths matchingContent:@[ base ]].count > 0;
		if (!named || claims[base] != nil) {
			keepsPartial = YES;
			continue;
		}
		NSURL *url = [snapshotURL URLByAppendingPathComponent:partialPath];
		long long size = [self allocatedSizeOfFileAtURL:url];
		freed += size;
		[removedPaths addObject:url.path];
		[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:[self cachePathOfURL:url cacheURL:cacheDirectoryURL]
													byteCount:size reason:nil]];
	}

	NSMutableArray<NSURL *> *bookkeeping = [NSMutableArray array];
	BOOL emptiesSnapshot = (content.count > 0 || partials.count > 0) && !keepsPartial;
	for (NSString *path in content) {
		if (![removedPaths containsObject:[snapshotURL URLByAppendingPathComponent:path].path]) {
			emptiesSnapshot = NO;
			break;
		}
	}
	if (emptiesSnapshot) {
		for (NSString *name in NFKHFHubBookkeepingNames()) {
			NSURL *url = [snapshotURL URLByAppendingPathComponent:name];
			if ([name isEqualToString:NFKHFHubPinMarkerName] || ![NSFileManager.defaultManager fileExistsAtPath:url.path]) {
				continue;
			}
			[bookkeeping addObject:url];
			freed += [self allocatedSizeOfFileAtURL:url];
		}
	}

	NFKRemovalPlan *plan = [[NFKRemovalPlan alloc] initWithRepository:target.repository
															 revision:target.revision
															sourceURL:target.sourceURL
														 removedFiles:removed
															keptFiles:kept
													   freedByteCount:freed
														  snapshotURL:snapshotURL];
	plan.targetPaths = target.paths;
	plan.bookkeepingURLs = bookkeeping;
	return plan;
}

- (NFKHFHubRemovalTarget *)removalTargetForRepo:(NSString *)repo
									   revision:(nullable NSString *)revision
										  paths:(nullable NSArray<NSString *> *)paths
{
	NFKHFHubRemovalTarget *target = [[NFKHFHubRemovalTarget alloc] init];
	target.repository = repo;
	target.revision = [self normalizedRevision:revision];
	target.paths = paths;
	NSError *refusal = nil;
	target.snapshotURL = [self snapshotURLForRepo:repo revision:revision error:&refusal];
	if (target.snapshotURL == nil && self.cacheDirectoryURL != nil) {
		target.refusal = refusal;
	}
	return target;
}

- (NFKHFHubRemovalTarget *)removalTargetForURL:(NSURL *)URL paths:(nullable NSArray<NSString *> *)paths
{
	NFKHFHubRemovalTarget *target = [[NFKHFHubRemovalTarget alloc] init];
	target.sourceURL = URL;
	target.paths = paths;
	NSError *refusal = nil;
	target.snapshotURL = [self URLSnapshotURLForURL:URL error:&refusal];
	if (target.snapshotURL == nil && self.cacheDirectoryURL != nil) {
		target.refusal = refusal;
	}
	return target;
}

- (nullable NFKRemovalPlan *)removalPlanForRepo:(NSString *)repo
									   revision:(nullable NSString *)revision
										  paths:(nullable NSArray<NSString *> *)paths
										  error:(NSError * _Nullable *)outError
{
	return [self removalPlanForTarget:[self removalTargetForRepo:repo revision:revision paths:paths] error:outError];
}

- (nullable NFKRemovalPlan *)removalPlanForURL:(NSURL *)URL error:(NSError * _Nullable *)outError
{
	return [self removalPlanForTarget:[self removalTargetForURL:URL paths:nil] error:outError];
}

/*! The removal target of one part of a source: its listed paths in its repository's snapshot, or its
	cached file in its URL's snapshot. */
- (NFKHFHubRemovalTarget *)removalTargetForPart:(NFKHFHubSourcePart *)part
{
	if (part.URL == nil) {
		NSArray<NSString *> *paths = part.files != nil ? [part.files valueForKey:@"path"] : nil;
		return [self removalTargetForRepo:part.repository revision:part.revision paths:paths];
	}
	NFKHFHubRemovalTarget *target = [self removalTargetForURL:part.URL paths:nil];
	NSError *refusal = nil;
	NSURL *localURL = [self localURLOfURLPart:part error:&refusal];
	if (localURL != nil) {
		target.paths = @[ localURL.lastPathComponent ];
	} else if (target.refusal == nil && self.cacheDirectoryURL != nil) {
		target.refusal = refusal;
	}
	return target;
}

/*! One plan over the plans of several snapshots: their files, kept files, and freed bytes together. */
- (NFKRemovalPlan *)removalPlanWithParts:(NSArray<NFKRemovalPlan *> *)parts
							  repository:(nullable NSString *)repository
								revision:(nullable NSString *)revision
							 snapshotURL:(nullable NSURL *)snapshotURL
{
	NSMutableArray<NFKCachedFile *> *removed = [NSMutableArray array];
	NSMutableArray<NFKCachedFile *> *kept = [NSMutableArray array];
	NSMutableArray<NSURL *> *bookkeeping = [NSMutableArray array];
	long long freed = 0;
	for (NFKRemovalPlan *part in parts) {
		[removed addObjectsFromArray:part.removedFiles];
		[kept addObjectsFromArray:part.keptFiles];
		[bookkeeping addObjectsFromArray:part.bookkeepingURLs];
		freed += part.freedByteCount;
	}
	NFKRemovalPlan *plan = [[NFKRemovalPlan alloc] initWithRepository:repository
															 revision:revision
															sourceURL:nil
														 removedFiles:removed
															keptFiles:kept
													   freedByteCount:freed
														  snapshotURL:snapshotURL];
	plan.bookkeepingURLs = bookkeeping;
	plan.parts = parts;
	return plan;
}

- (nullable NFKRemovalPlan *)removalPlanForModelSource:(NFKModelSource *)source error:(NSError * _Nullable *)outError
{
	if ([self modelSourceHoldsNoFiles:source]) {
		return [self removalPlanForTarget:[[NFKHFHubRemovalTarget alloc] init] error:outError];
	}
	NSArray<NFKHFHubSourcePart *> *parts = [self partsOfModelSource:source];
	BOOL namesSource = [source.kind isEqualToString:NFKModelSourceKindURL] || source.repository.length > 0;
	if (parts.count == 0 || !namesSource) {
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the source names no repository"]
						  to:outError];
		return nil;
	}
	if (parts.count == 1) {
		return [self removalPlanForTarget:[self removalTargetForPart:parts.firstObject] error:outError];
	}
	NSURL *sourceSnapshotURL = [self snapshotURLForRepo:source.repository revision:source.revision error:outError];
	if (sourceSnapshotURL == nil) {
		return nil;
	}
	NSMutableArray<NFKHFHubRemovalTarget *> *targets = [NSMutableArray arrayWithCapacity:parts.count];
	for (NFKHFHubSourcePart *part in parts) {
		NFKHFHubRemovalTarget *target = [self removalTargetForPart:part];
		if (target.refusal != nil) {
			[self propagateError:target.refusal to:outError];
			return nil;
		}
		[targets addObject:target];
	}
	NSMutableArray<NFKRemovalPlan *> *plans = [NSMutableArray arrayWithCapacity:targets.count];
	for (NFKHFHubRemovalTarget *target in targets) {
		NFKRemovalPlan *plan = [self removalPlanForTarget:target error:outError];
		if (plan == nil) {
			return nil;
		}
		[plans addObject:plan];
	}
	return [self removalPlanWithParts:plans
						   repository:source.repository
							 revision:[self normalizedRevision:source.revision]
						  snapshotURL:sourceSnapshotURL];
}

#pragma mark Removal

- (NFKHFHubRemovalTarget *)targetOfPlan:(NFKRemovalPlan *)plan
{
	if (plan.sourceURL != nil) {
		return [self removalTargetForURL:plan.sourceURL paths:plan.targetPaths];
	}
	if (plan.repository != nil) {
		return [self removalTargetForRepo:plan.repository revision:plan.revision paths:plan.targetPaths];
	}
	return [[NFKHFHubRemovalTarget alloc] init];
}

/*! Deletes folders an interrupted removal left behind. */
- (void)removeLeftoverTrashInCacheURL:(NSURL *)cacheDirectoryURL
{
	NSArray<NSURL *> *children = [NSFileManager.defaultManager contentsOfDirectoryAtURL:cacheDirectoryURL
															 includingPropertiesForKeys:nil
																				options:0
																				  error:NULL];
	for (NSURL *child in children) {
		if ([child.lastPathComponent hasPrefix:NFKHFHubTrashPrefix]) {
			[NSFileManager.defaultManager removeItemAtURL:child error:NULL];
		}
	}
}

/*! Moves every URL into a fresh trash folder in the cache, or none: a failed move returns the moved
	files to their places. Returns the trash folder. */
- (nullable NSURL *)moveURLs:(NSArray<NSURL *> *)URLs toTrashInCacheURL:(NSURL *)cacheDirectoryURL error:(NSError **)outError
{
	for (NSURL *url in URLs) {
		if (![self requireURL:url insideCacheURL:cacheDirectoryURL error:outError]) {
			return nil;
		}
	}
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSURL *trashURL = [cacheDirectoryURL URLByAppendingPathComponent:[NFKHFHubTrashPrefix stringByAppendingString:NSUUID.UUID.UUIDString]];
	NSError *error = nil;
	if (![fileManager createDirectoryAtURL:trashURL withIntermediateDirectories:YES attributes:nil error:&error]) {
		[self propagateError:error to:outError];
		return nil;
	}
	NSMutableArray<NSArray<NSURL *> *> *moved = [NSMutableArray arrayWithCapacity:URLs.count];
	for (NSURL *url in URLs) {
		NSURL *destination = [trashURL URLByAppendingPathComponent:[NSString stringWithFormat:@"%lu", (unsigned long)moved.count]];
		if ([fileManager moveItemAtURL:url toURL:destination error:&error]) {
			[moved addObject:@[ url, destination ]];
			continue;
		}
		for (NSArray<NSURL *> *pair in moved.reverseObjectEnumerator) {
			[fileManager moveItemAtURL:pair[1] toURL:pair[0] error:NULL];
		}
		[fileManager removeItemAtURL:trashURL error:NULL];
		[self propagateError:error to:outError];
		return nil;
	}
	return trashURL;
}

- (nullable NFKRemovalPlan *)removeFilesInPlan:(NFKRemovalPlan *)plan error:(NSError * _Nullable *)outError
{
	if (![self requireCacheDirectory:outError]) {
		return nil;
	}
	NSMutableArray<NFKRemovalPlan *> *plans = [NSMutableArray array];
	NSMutableArray<NFKHFHubRemovalTarget *> *targets = [NSMutableArray array];
	for (NFKRemovalPlan *part in plan.parts ?: @[ plan ]) {
		NFKHFHubRemovalTarget *target = [self targetOfPlan:part];
		if (target.refusal != nil) {
			[self propagateError:target.refusal to:outError];
			return nil;
		}
		if (target.snapshotURL != nil) {
			[plans addObject:part];
			[targets addObject:target];
		}
	}
	if (targets.count == 0) {
		return plan;
	}
	__block NSArray<NFKRemovalPlan *> *executed = nil;
	__block NSError *error = nil;
	[NFKDownload performWithSnapshotRegistryLocked:^{
		for (NFKHFHubRemovalTarget *target in targets) {
			if ([NFKDownload isSnapshotURLInUse:target.snapshotURL]) {
				NSString *name = target.repository ?: target.sourceURL.absoluteString;
				NSString *reason = [NSString stringWithFormat:@"%@ is being downloaded; cancel the download before removing it", name];
				error = [NFKDownload errorWithCode:NFKDownloadErrorInUse reason:reason];
				return;
			}
		}
		[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
			executed = [self executeRemovalOfPlans:plans targets:targets cacheURL:cacheDirectoryURL error:&error];
		}];
	}];
	if (executed == nil) {
		[self propagateError:error to:outError];
		return nil;
	}
	if (plan.parts == nil) {
		return executed.firstObject;
	}
	return [self removalPlanWithParts:executed repository:plan.repository revision:plan.revision snapshotURL:plan.snapshotURL];
}

/*! Removes the files of every plan in one all-or-none move: each target is planned again and only the
	files both plans remove go. Returns each plan as carried out. */
- (nullable NSArray<NFKRemovalPlan *> *)executeRemovalOfPlans:(NSArray<NFKRemovalPlan *> *)plans
													  targets:(NSArray<NFKHFHubRemovalTarget *> *)targets
													 cacheURL:(NSURL *)cacheDirectoryURL
														error:(NSError **)outError
{
	[self removeLeftoverTrashInCacheURL:cacheDirectoryURL];
	NSMutableArray<NFKRemovalPlan *> *executed = [NSMutableArray arrayWithCapacity:plans.count];
	NSMutableArray<NSURL *> *URLs = [NSMutableArray array];
	NSMutableSet<NSString *> *movedPaths = [NSMutableSet set];
	for (NSUInteger index = 0; index < plans.count; index++) {
		NFKRemovalPlan *plan = plans[index];
		NFKRemovalPlan *fresh = [self removalPlanForTarget:targets[index] cacheURL:cacheDirectoryURL];
		NSSet<NSString *> *planned = [NSSet setWithArray:[plan.removedFiles valueForKeyPath:@"URL.path"]];
		NSMutableArray<NFKCachedFile *> *removing = [NSMutableArray array];
		long long freed = 0;
		for (NFKCachedFile *file in fresh.removedFiles) {
			if ([planned containsObject:file.URL.path] && ![movedPaths containsObject:file.URL.path]) {
				[removing addObject:file];
				freed += file.byteCount;
			}
		}
		NSArray<NSURL *> *bookkeeping = removing.count == fresh.removedFiles.count ? fresh.bookkeepingURLs : @[];
		for (NSURL *url in bookkeeping) {
			freed += [self allocatedSizeOfFileAtURL:url];
		}
		for (NSURL *url in [[removing valueForKey:@"URL"] arrayByAddingObjectsFromArray:bookkeeping]) {
			if (![movedPaths containsObject:url.path]) {
				[movedPaths addObject:url.path];
				[URLs addObject:url];
			}
		}
		NFKRemovalPlan *part = [[NFKRemovalPlan alloc] initWithRepository:fresh.repository
																  revision:fresh.revision
																 sourceURL:fresh.sourceURL
															  removedFiles:removing
																 keptFiles:fresh.keptFiles
															freedByteCount:freed
															   snapshotURL:fresh.snapshotURL];
		part.targetPaths = fresh.targetPaths;
		part.bookkeepingURLs = bookkeeping;
		[executed addObject:part];
	}

	if (URLs.count > 0) {
		NSURL *trashURL = [self moveURLs:URLs toTrashInCacheURL:cacheDirectoryURL error:outError];
		if (trashURL == nil) {
			return nil;
		}
		// The files are out of the snapshots; a trash folder this fails to delete goes with the next
		// removal.
		[NSFileManager.defaultManager removeItemAtURL:trashURL error:NULL];
		for (NSURL *url in URLs) {
			[self removeEmptyFoldersAboveURL:url inDirectoryAtURL:cacheDirectoryURL];
		}
	}
	return executed;
}

- (void)removeFilesInPlan:(NFKRemovalPlan *)plan
		completionHandler:(void (^)(NFKRemovalPlan * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKRemovalPlan *removal = [self removeFilesInPlan:plan error:&error];
		completionHandler(removal, error);
	});
}

@end
