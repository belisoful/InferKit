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
@end

@implementation NFKHFHubRemovalTarget
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
			if ([entry isKindOfClass:NSDictionary.class] && [entry[@"type"] isEqual:@"file"]
				&& [entry[@"path"] isKindOfClass:NSString.class]) {
				[entries addObject:entry];
			}
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
	return [[NFKDownloadPlanFile alloc] initWithPath:path
										   remoteURL:[self remoteURLForRepo:repo revision:revision path:path]
											localURL:localURL
										   byteCount:byteCount
											  sha256:sha256
											  cached:[self fileAtURL:localURL isWholeWithByteCount:byteCount]
											optional:optional];
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
	for (NSString *path in optionalPaths) {
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

- (nullable NSString *)URLCacheRepositoryForURL:(NSURL *)URL
{
	NSString *absolute = URL.absoluteString;
	if (absolute.length == 0) {
		return nil;
	}
	NSString *digest = [NFKHFHubHexDigest([absolute dataUsingEncoding:NSUTF8StringEncoding]) substringToIndex:16];
	NSString *host = URL.host.length > 0 ? URL.host.lowercaseString : @"local";
	return [NSString stringWithFormat:@"%@/%@/%@", NFKHFHubURLCacheFolder, host, digest];
}

- (nullable NSURL *)URLSnapshotURLForURL:(NSURL *)URL
{
	NSString *cacheRepository = [self URLCacheRepositoryForURL:URL];
	if (self.cacheDirectoryURL == nil || cacheRepository == nil) {
		return nil;
	}
	return [self URL:self.cacheDirectoryURL byAppendingPathString:cacheRepository];
}

- (nullable NSURL *)localURLForURL:(NSURL *)URL fileName:(nullable NSString *)fileName
{
	NSURL *snapshotURL = [self URLSnapshotURLForURL:URL];
	if (snapshotURL == nil) {
		return nil;
	}
	NSString *name = fileName.length > 0 ? fileName.lastPathComponent : URL.lastPathComponent;
	if (name.length == 0 || [name isEqualToString:@"/"]) {
		name = @"download";
	}
	return [snapshotURL URLByAppendingPathComponent:name];
}

- (nullable NFKDownloadPlan *)planDownloadOfURL:(NSURL *)URL
										 sha256:(nullable NSString *)expectedSHA256
									   fileName:(nullable NSString *)fileName
									  byteCount:(long long)listedByteCount
										  error:(NSError **)outError
{
	if (![self requireCacheDirectory:outError]) {
		return nil;
	}
	NSURL *localURL = [self localURLForURL:URL fileName:fileName];
	if (localURL == nil) {
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the URL is empty"] to:outError];
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
		if (encoding.length == 0 && response.expectedContentLength >= 0) {
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
																 optional:NO];
	NFKDownloadPlan *plan = [[NFKDownloadPlan alloc] initWithRepository:nil revision:nil sourceURL:URL files:@[ file ] localURL:localURL];
	plan.snapshotURL = localURL.URLByDeletingLastPathComponent;
	plan.cacheRepository = [self URLCacheRepositoryForURL:URL];
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
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	NSMutableArray<NSString *> *optional = [NSMutableArray array];
	NSMutableDictionary<NSString *, NFKModelFile *> *listings = [NSMutableDictionary dictionary];
	for (NFKModelFile *file in source.files) {
		[(file.optional ? optional : required) addObject:file.path];
		listings[file.path] = file;
	}
	return [self planRepo:source.repository revision:source.revision requiredPaths:required optionalPaths:optional
				 listings:listings singleFile:singleFile error:outError];
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

/*! The free bytes of the volume holding the cache; -1 when the volume does not say. */
- (long long)availableCapacityOfCache
{
	NSURL *cacheDirectoryURL = self.cacheDirectoryURL;
	if (cacheDirectoryURL == nil) {
		return -1;
	}
#if TARGET_OS_TV
	NSURLResourceKey key = NSURLVolumeAvailableCapacityKey;
#else
	NSURLResourceKey key = NSURLVolumeAvailableCapacityForImportantUsageKey;
#endif
	NSNumber *capacity = nil;
	[cacheDirectoryURL getResourceValue:&capacity forKey:key error:NULL];
	return capacity != nil ? capacity.longLongValue : -1;
}

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
	NSError *error = nil;
	[download becomeCurrent];
	@try {
		for (NFKDownloadPlanFile *file in plan.files) {
			if (download.isCancelled) {
				return [NFKDownload cancellationError];
			}
			NSURL *fetched = [self fetchFileFromURL:file.remoteURL
										   localURL:file.localURL
										snapshotURL:plan.snapshotURL ?: file.localURL.URLByDeletingLastPathComponent
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
	return nil;
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

/*! The paths a complete-download record in the snapshot names. */
- (NSArray<NSString *> *)recordedCompletePathsInSnapshotURL:(NSURL *)snapshotURL
{
	NSData *data = [NSData dataWithContentsOfURL:[snapshotURL URLByAppendingPathComponent:NFKHFHubCompleteMarkerName]];
	id paths = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	if (![paths isKindOfClass:NSArray.class]) {
		return @[];
	}
	return [(NSArray *)paths filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id path, NSDictionary *bindings) {
		return [path isKindOfClass:NSString.class];
	}]];
}

- (BOOL)recordCompleteDownloadOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   paths:(NSArray<NSString *> *)paths
							   error:(NSError * _Nullable *)outError
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return [self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
	}
	__block NSError *writeError = nil;
	__block BOOL written = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		@synchronized (NFKHFHub.class) {
			NSMutableOrderedSet<NSString *> *recorded = [NSMutableOrderedSet orderedSetWithArray:[self recordedCompletePathsInSnapshotURL:snapshotURL]];
			[recorded addObjectsFromArray:paths];
			NSData *data = [NSJSONSerialization dataWithJSONObject:recorded.array options:0 error:&writeError];
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
		NFKDownloadPresence presence = NFKDownloadPresenceAbsent;
		if (present.count > 0) {
			presence = missing.count == 0 && required.count > 0 ? NFKDownloadPresenceComplete : NFKDownloadPresencePartial;
		}
		if (requiredPaths != nil && required.count == 0 && present.count > 0) {
			presence = NFKDownloadPresenceComplete;
		}
		status = [[NFKDownloadStatus alloc] initWithPresence:presence
												   byteCount:bytes
												presentPaths:present.array
												missingPaths:missing
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
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	NSMutableArray<NSString *> *optional = [NSMutableArray array];
	for (NFKModelFile *file in source.files) {
		[(file.optional ? optional : required) addObject:file.path];
	}
	NSURL *localURL = nil;
	if ([source.layout isEqualToString:NFKModelSourceLayoutFile] && source.files.count == 1) {
		localURL = [self localURLForRepo:source.repository revision:source.revision path:source.files.firstObject.path];
	}
	return [self statusInSnapshotURL:snapshotURL requiredPaths:required optionalPaths:optional localURL:localURL];
}

#pragma mark Removal plans

/*! The paths of a Hugging Face source, or nil when it lists none. */
- (nullable NSSet<NSString *> *)listedPathsOfSource:(NFKModelSource *)source
{
	if (source.files.count == 0) {
		return nil;
	}
	return [NSSet setWithArray:[source.files valueForKey:@"path"]];
}

/*! The paths another catalog variant whose download is complete uses in the target's snapshot, each
	with the reason it is kept. */
- (NSDictionary<NSString *, NSString *> *)claimedPathsForTarget:(NFKHFHubRemovalTarget *)target
{
	NSMutableDictionary<NSString *, NSString *> *claims = [NSMutableDictionary dictionary];
	if (target.repository == nil) {
		return claims;
	}
	NSString *revision = [self normalizedRevision:target.revision];
	NSSet<NSString *> *targetPaths = target.paths != nil ? [NSSet setWithArray:target.paths] : nil;
	for (NFKModelDescriptor *descriptor in self.modelCatalog.descriptors) {
		for (NFKModelVariant *variant in descriptor.variants) {
			NFKModelSource *source = variant.source;
			if (![source.kind isEqualToString:NFKModelSourceKindHuggingFace]
				|| ![source.repository isEqualToString:target.repository]
				|| ![[self normalizedRevision:source.revision] isEqualToString:revision]) {
				continue;
			}
			NSSet<NSString *> *listed = [self listedPathsOfSource:source];
			BOOL sameDownload = (listed == nil && targetPaths == nil) || [listed isEqualToSet:targetPaths];
			if (sameDownload || [self downloadStatusForModelSource:source].presence != NFKDownloadPresenceComplete) {
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

- (nullable NFKRemovalPlan *)removalPlanForTarget:(NFKHFHubRemovalTarget *)target error:(NSError **)outError
{
	if (![self requireCacheDirectory:outError]) {
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
		[removedPaths addObject:url.path];
		if (![self isSymbolicLinkAtURL:url]) {
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
		if (linksByTarget == nil) {
			linksByTarget = [self symbolicLinksByTargetInCacheURL:cacheDirectoryURL];
		}
		NSMutableArray<NSString *> *otherLinks = [NSMutableArray array];
		for (NSString *linkPath in linksByTarget[targetPath]) {
			NSString *relative = [self relativePathOfURL:[NSURL fileURLWithPath:linkPath] underURL:snapshotURL];
			if (relative == nil || ![chosen containsObject:relative] || claims[relative] != nil) {
				[otherLinks addObject:linkPath];
			}
		}
		NSString *targetCachePath = [self cachePathOfURL:targetURL cacheURL:cacheDirectoryURL];
		NSString *targetInSnapshot = [self relativePathOfURL:targetURL underURL:snapshotURL];
		BOOL targetIsChosen = targetInSnapshot != nil && [chosen containsObject:targetInSnapshot] && claims[targetInSnapshot] == nil;
		if (otherLinks.count > 0 || (targetInSnapshot != nil && !targetIsChosen)) {
			[removed addObject:[[NFKCachedFile alloc] initWithURL:url path:cachePath byteCount:0
														   reason:@"a symbolic link whose target stays"]];
			if (![removedTargets containsObject:targetPath] && !targetIsChosen) {
				[removedTargets addObject:targetPath];
				NSString *reason = otherLinks.count > 0
					? [NSString stringWithFormat:@"linked from %lu other cached file(s)", (unsigned long)otherLinks.count]
					: @"a file of the snapshot the removal keeps";
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

	NSMutableArray<NSURL *> *bookkeeping = [NSMutableArray array];
	BOOL emptiesSnapshot = content.count > 0;
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
	target.snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	target.paths = paths;
	return target;
}

- (NFKHFHubRemovalTarget *)removalTargetForURL:(NSURL *)URL
{
	NFKHFHubRemovalTarget *target = [[NFKHFHubRemovalTarget alloc] init];
	target.sourceURL = URL;
	target.snapshotURL = [self URLSnapshotURLForURL:URL];
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
	return [self removalPlanForTarget:[self removalTargetForURL:URL] error:outError];
}

- (nullable NFKRemovalPlan *)removalPlanForModelSource:(NFKModelSource *)source error:(NSError * _Nullable *)outError
{
	if ([self modelSourceHoldsNoFiles:source]) {
		return [self removalPlanForTarget:[[NFKHFHubRemovalTarget alloc] init] error:outError];
	}
	if ([source.kind isEqualToString:NFKModelSourceKindURL] && source.URL != nil) {
		return [self removalPlanForURL:source.URL error:outError];
	}
	if (source.repository.length == 0) {
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:@"the source names no repository"]
						  to:outError];
		return nil;
	}
	NSArray<NSString *> *paths = source.files.count > 0 ? [source.files valueForKey:@"path"] : nil;
	return [self removalPlanForRepo:source.repository revision:source.revision paths:paths error:outError];
}

#pragma mark Removal

- (NFKHFHubRemovalTarget *)targetOfPlan:(NFKRemovalPlan *)plan
{
	if (plan.sourceURL != nil) {
		return [self removalTargetForURL:plan.sourceURL];
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
	NFKHFHubRemovalTarget *target = [self targetOfPlan:plan];
	if (target.snapshotURL == nil) {
		return plan;
	}
	__block NFKRemovalPlan *executed = nil;
	__block NSError *error = nil;
	[NFKDownload performWithSnapshotRegistryLocked:^{
		if ([NFKDownload isSnapshotURLInUse:target.snapshotURL]) {
			NSString *name = target.repository ?: target.sourceURL.absoluteString;
			NSString *reason = [NSString stringWithFormat:@"%@ is being downloaded; cancel the download before removing it", name];
			error = [NFKDownload errorWithCode:NFKDownloadErrorInUse reason:reason];
			return;
		}
		[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
			executed = [self executeRemovalOfPlan:plan target:target cacheURL:cacheDirectoryURL error:&error];
		}];
	}];
	if (executed == nil) {
		[self propagateError:error to:outError];
	}
	return executed;
}

- (nullable NFKRemovalPlan *)executeRemovalOfPlan:(NFKRemovalPlan *)plan
										   target:(NFKHFHubRemovalTarget *)target
										 cacheURL:(NSURL *)cacheDirectoryURL
											error:(NSError **)outError
{
	[self removeLeftoverTrashInCacheURL:cacheDirectoryURL];
	NFKRemovalPlan *fresh = [self removalPlanForTarget:target cacheURL:cacheDirectoryURL];
	NSSet<NSString *> *planned = [NSSet setWithArray:[plan.removedFiles valueForKeyPath:@"URL.path"]];
	NSMutableArray<NFKCachedFile *> *removing = [NSMutableArray array];
	long long freed = 0;
	for (NFKCachedFile *file in fresh.removedFiles) {
		if ([planned containsObject:file.URL.path]) {
			[removing addObject:file];
			freed += file.byteCount;
		}
	}
	NSArray<NSURL *> *bookkeeping = removing.count == fresh.removedFiles.count ? fresh.bookkeepingURLs : @[];
	for (NSURL *url in bookkeeping) {
		freed += [self allocatedSizeOfFileAtURL:url];
	}

	NSMutableArray<NSURL *> *URLs = [[removing valueForKey:@"URL"] mutableCopy];
	[URLs addObjectsFromArray:bookkeeping];
	if (URLs.count > 0) {
		NSURL *trashURL = [self moveURLs:URLs toTrashInCacheURL:cacheDirectoryURL error:outError];
		if (trashURL == nil) {
			return nil;
		}
		// The files are out of the snapshot; a trash folder this fails to delete goes with the next
		// removal.
		[NSFileManager.defaultManager removeItemAtURL:trashURL error:NULL];
		for (NSURL *url in URLs) {
			[self removeEmptyFoldersAboveURL:url inDirectoryAtURL:cacheDirectoryURL];
		}
	}

	NFKRemovalPlan *executed = [[NFKRemovalPlan alloc] initWithRepository:fresh.repository
																 revision:fresh.revision
																sourceURL:fresh.sourceURL
															 removedFiles:removing
																keptFiles:fresh.keptFiles
														   freedByteCount:freed
															  snapshotURL:fresh.snapshotURL];
	executed.targetPaths = fresh.targetPaths;
	executed.bookkeepingURLs = bookkeeping;
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
