//
//  NFKHFHub.m
//  InferKit
//

#import "NFKHFHub.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"
#import <CommonCrypto/CommonCrypto.h>

NSString * const NFKHFHubDefaultRevision = @"main";
const long long NFKHFHubUnlimitedCacheSize = -1;

// Marks a snapshot the hub owns; its modification date is the snapshot's last use. Eviction only
// considers folders that carry it.
static NSString * const NFKHFHubOwnedMarkerName = @".inferkit-owned";
// Marks a pinned snapshot, which eviction skips.
static NSString * const NFKHFHubKeepMarkerName = @".inferkit-keep";

static long long NFKHFHubDefaultCacheSizeLimitValue = -1;
static BOOL NFKHFHubDefaultExcludesCacheFromBackupValue = YES;
static NSString *NFKHFHubDefaultAccessTokenValue = nil;

/*! A cached snapshot eviction may remove. */
@interface NFKHFHubSnapshot : NSObject
@property (nonatomic, copy) NSURL *URL;
@property (nonatomic, copy) NSDate *lastUsedDate;
@end

@implementation NFKHFHubSnapshot
@end

@implementation NFKHFHub

@synthesize cacheDirectoryURL = _cacheDirectoryURL;
@synthesize endpointURL = _endpointURL;
@synthesize session = _session;
@synthesize accessToken = _accessToken;
@synthesize cacheSizeLimit = _cacheSizeLimit;
@synthesize excludesCacheFromBackup = _excludesCacheFromBackup;

+ (instancetype)hubWithCacheDirectoryURL:(nullable NSURL *)cacheDirectoryURL
{
	NFKHFHub *hub = [[self alloc] init];
	hub.cacheDirectoryURL = cacheDirectoryURL;
	return NARC_AUTORELEASE(hub);
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_cacheSizeLimit = NFKHFHub.defaultCacheSizeLimit;
		_excludesCacheFromBackup = NFKHFHub.defaultExcludesCacheFromBackup;
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_cacheDirectoryURL);
	NARC_RELEASE(_endpointURL);
	NARC_RELEASE(_session);
	NARC_RELEASE(_accessToken);
	SUPER_DEALLOC();
}

- (NSURL *)endpointURL
{
	if (_endpointURL == nil) {
		_endpointURL = NARC_RETAIN([NSURL URLWithString:@"https://huggingface.co"]);
	}
	return _endpointURL;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = NARC_RETAIN([NSURLSession sharedSession]);
	}
	return _session;
}

- (nullable NSString *)accessToken
{
	if (_accessToken != nil) {
		return _accessToken;
	}
	// Read on each request, so a default set after the hub was made still applies. HF_TOKEN is the
	// conventional place a Hugging Face credential lives, and where the tooling around the hub reads it.
	return NFKHFHub.defaultAccessToken ?: NSProcessInfo.processInfo.environment[@"HF_TOKEN"];
}

#pragma mark Process-wide defaults

+ (long long)defaultCacheSizeLimit
{
	@synchronized (NFKHFHub.class) {
		return NFKHFHubDefaultCacheSizeLimitValue;
	}
}

+ (void)setDefaultCacheSizeLimit:(long long)defaultCacheSizeLimit
{
	@synchronized (NFKHFHub.class) {
		NFKHFHubDefaultCacheSizeLimitValue = defaultCacheSizeLimit;
	}
}

+ (BOOL)defaultExcludesCacheFromBackup
{
	@synchronized (NFKHFHub.class) {
		return NFKHFHubDefaultExcludesCacheFromBackupValue;
	}
}

+ (void)setDefaultExcludesCacheFromBackup:(BOOL)defaultExcludesCacheFromBackup
{
	@synchronized (NFKHFHub.class) {
		NFKHFHubDefaultExcludesCacheFromBackupValue = defaultExcludesCacheFromBackup;
	}
}

+ (nullable NSString *)defaultAccessToken
{
	@synchronized (NFKHFHub.class) {
		return NFKHFHubDefaultAccessTokenValue;
	}
}

+ (void)setDefaultAccessToken:(nullable NSString *)defaultAccessToken
{
	@synchronized (NFKHFHub.class) {
		NFKHFHubDefaultAccessTokenValue = [defaultAccessToken copy];
	}
}

#pragma mark URL construction

+ (NSURL *)defaultCacheDirectoryURL
{
	NSURL *base = [NSFileManager.defaultManager URLForDirectory:NSApplicationSupportDirectory
													  inDomain:NSUserDomainMask
											 appropriateForURL:nil
														create:YES
														 error:NULL];
	NSURL *directory = [[base URLByAppendingPathComponent:@"InferKit"] URLByAppendingPathComponent:@"models"];
	[NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];
	return directory;
}

- (NSString *)normalizedRevision:(nullable NSString *)revision
{
	return revision.length > 0 ? revision : NFKHFHubDefaultRevision;
}

- (NSURL *)URL:(NSURL *)base byAppendingPathString:(NSString *)pathString
{
	NSURL *result = base;
	for (NSString *component in [pathString componentsSeparatedByString:@"/"]) {
		if (component.length > 0) {
			result = [result URLByAppendingPathComponent:component];
		}
	}
	return result;
}

- (nullable NSURL *)remoteURLForRepo:(NSString *)repo
						  revision:(nullable NSString *)revision
							  path:(NSString *)path
{
	if (repo.length == 0 || path.length == 0) {
		return nil;
	}
	NSURL *url = [self URL:self.endpointURL byAppendingPathString:repo];
	url = [url URLByAppendingPathComponent:@"resolve"];
	url = [url URLByAppendingPathComponent:[self normalizedRevision:revision]];
	return [self URL:url byAppendingPathString:path];
}

- (nullable NSURL *)localURLForRepo:(NSString *)repo
						 revision:(nullable NSString *)revision
							 path:(NSString *)path
{
	if (self.cacheDirectoryURL == nil || repo.length == 0 || path.length == 0) {
		return nil;
	}
	NSURL *url = [self URL:self.cacheDirectoryURL byAppendingPathString:repo];
	url = [url URLByAppendingPathComponent:[self normalizedRevision:revision]];
	return [self URL:url byAppendingPathString:path];
}

- (nullable NSURL *)snapshotURLForRepo:(NSString *)repo revision:(nullable NSString *)revision
{
	if (self.cacheDirectoryURL == nil || repo.length == 0) {
		return nil;
	}
	NSURL *url = [self URL:self.cacheDirectoryURL byAppendingPathString:repo];
	return [url URLByAppendingPathComponent:[self normalizedRevision:revision]];
}

- (BOOL)isCachedRepo:(NSString *)repo revision:(nullable NSString *)revision path:(NSString *)path
{
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	return localURL != nil && [NSFileManager.defaultManager fileExistsAtPath:localURL.path];
}

#pragma mark Download

- (nullable NSURL *)downloadRepo:(NSString *)repo
					  revision:(nullable NSString *)revision
						  path:(NSString *)path
						sha256:(nullable NSString *)expectedSHA256
						 error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
		return nil;
	}
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	NSURL *remoteURL = [self remoteURLForRepo:repo revision:revision path:path];
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (localURL == nil || remoteURL == nil || snapshotURL == nil) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo or path"];
		return nil;
	}

	NSFileManager *fileManager = NSFileManager.defaultManager;
	BOOL scoped = [self.cacheDirectoryURL startAccessingSecurityScopedResource];
	@try {
		if ([fileManager fileExistsAtPath:localURL.path]) {
			if (expectedSHA256 == nil || [self fileAtURL:localURL matchesSHA256:expectedSHA256]) {
				[self excludeCacheFromBackupIfNeeded];
				[self markSnapshotUsedAtURL:snapshotURL];
				return localURL;
			}
			[fileManager removeItemAtURL:localURL error:NULL];
		}

		NSError *dirError = nil;
		if (![fileManager createDirectoryAtURL:localURL.URLByDeletingLastPathComponent
				   withIntermediateDirectories:YES
									attributes:nil
										 error:&dirError]) {
			[self propagateError:dirError to:outError];
			return nil;
		}
		[self excludeCacheFromBackupIfNeeded];
		[self markSnapshotUsedAtURL:snapshotURL];

		NSURL *partialURL = [localURL URLByAppendingPathExtension:@"download"];
		[fileManager removeItemAtURL:partialURL error:NULL];
		if (![self fetchURL:remoteURL toFileURL:partialURL error:outError]) {
			return nil;
		}

		if (expectedSHA256 != nil && ![self fileAtURL:partialURL matchesSHA256:expectedSHA256]) {
			[fileManager removeItemAtURL:partialURL error:NULL];
			[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the download failed its checksum"];
			return nil;
		}

		[fileManager removeItemAtURL:localURL error:NULL];
		NSError *moveError = nil;
		if (![fileManager moveItemAtURL:partialURL toURL:localURL error:&moveError]) {
			[self propagateError:moveError to:outError];
			return nil;
		}
		// The file is in place; a failed eviction leaves the cache over its limit and the download
		// still succeeds.
		[self trimCacheKeepingSnapshotURL:snapshotURL error:NULL];
		return localURL;
	} @finally {
		if (scoped) {
			[self.cacheDirectoryURL stopAccessingSecurityScopedResource];
		}
	}
}

- (void)downloadRepo:(NSString *)repo
			revision:(nullable NSString *)revision
				path:(NSString *)path
			  sha256:(nullable NSString *)expectedSHA256
   completionHandler:(void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NSURL *localURL = [self downloadRepo:repo revision:revision path:path sha256:expectedSHA256 error:&error];
		completionHandler(localURL, error);
	});
}

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError * _Nullable *)outError
{
	__block NSError *resultError = nil;
	__block BOOL moved = NO;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:remoteURL];
	NSString *token = self.accessToken;
	if (token.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:token] forHTTPHeaderField:@"Authorization"];
	}

	NSURLSessionDownloadTask *task = [self.session downloadTaskWithRequest:request
													completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
		// The downloaded temp file is removed when this handler returns, so move it here.
		if (location != nil && error == nil) {
			NSHTTPURLResponse *http = [response isKindOfClass:NSHTTPURLResponse.class] ? (NSHTTPURLResponse *)response : nil;
			if (http != nil && (http.statusCode < 200 || http.statusCode >= 300)) {
				NSString *reason = [NSString stringWithFormat:@"the download returned HTTP %ld", (long)http.statusCode];
				NSError *httpError = [NSError errorWithDomain:NFKInferenceErrorDomain
														 code:kNFKError_InferenceBackendFailure
													 userInfo:@{ NSLocalizedDescriptionKey: reason }];
				resultError = NARC_RETAIN(httpError);
			} else {
				NSError *moveError = nil;
				[NSFileManager.defaultManager removeItemAtURL:destinationURL error:NULL];
				moved = [NSFileManager.defaultManager moveItemAtURL:location toURL:destinationURL error:&moveError];
				resultError = NARC_RETAIN(moveError);
			}
		} else {
			resultError = NARC_RETAIN(error);
		}
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (!moved) {
		[self propagateError:resultError to:outError];
	}
	return moved;
}

#pragma mark Cache management

- (void)performWithCacheAccess:(void (^)(NSURL *cacheDirectoryURL))body
{
	NSURL *cacheDirectoryURL = self.cacheDirectoryURL;
	if (cacheDirectoryURL == nil) {
		return;
	}
	BOOL scoped = [cacheDirectoryURL startAccessingSecurityScopedResource];
	@try {
		body(cacheDirectoryURL);
	} @finally {
		if (scoped) {
			[cacheDirectoryURL stopAccessingSecurityScopedResource];
		}
	}
}

- (void)markSnapshotUsedAtURL:(NSURL *)snapshotURL
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSString *markerPath = [snapshotURL URLByAppendingPathComponent:NFKHFHubOwnedMarkerName].path;
	if (![fileManager fileExistsAtPath:markerPath]) {
		[fileManager createFileAtPath:markerPath contents:[NSData data] attributes:nil];
		return;
	}
	[fileManager setAttributes:@{ NSFileModificationDate: [NSDate date] } ofItemAtPath:markerPath error:NULL];
}

- (void)excludeCacheFromBackupIfNeeded
{
	NSURL *cacheDirectoryURL = self.cacheDirectoryURL;
	if (!self.excludesCacheFromBackup || [NFKHFHub isExcludedFromBackup:cacheDirectoryURL]) {
		return;
	}
	// A volume that refuses the attribute leaves the cache in backups; the download goes ahead.
	[NFKHFHub setExcludedFromBackup:YES forURL:cacheDirectoryURL error:NULL];
}

+ (BOOL)setExcludedFromBackup:(BOOL)excluded forURL:(NSURL *)url error:(NSError * _Nullable *)outError
{
	NSError *error = nil;
	if ([url setResourceValue:@(excluded) forKey:NSURLIsExcludedFromBackupKey error:&error]) {
		return YES;
	}
	if (outError != NULL) {
		*outError = error;
	}
	return NO;
}

+ (BOOL)isExcludedFromBackup:(NSURL *)url
{
	// NSURL caches resource values per instance; a stale cache would hide a change made elsewhere.
	[url removeCachedResourceValueForKey:NSURLIsExcludedFromBackupKey];
	NSNumber *excluded = nil;
	[url getResourceValue:&excluded forKey:NSURLIsExcludedFromBackupKey error:NULL];
	return excluded.boolValue;
}

- (long long)allocatedSizeOfDirectoryAtURL:(NSURL *)directoryURL
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsRegularFileKey, NSURLTotalFileAllocatedSizeKey ];
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:directoryURL
																   includingPropertiesForKeys:keys
																					  options:0
																				 errorHandler:nil];
	long long total = 0;
	for (NSURL *url in enumerator) {
		NSDictionary<NSURLResourceKey, id> *values = [url resourceValuesForKeys:keys error:NULL];
		if ([values[NSURLIsRegularFileKey] boolValue]) {
			total += [values[NSURLTotalFileAllocatedSizeKey] longLongValue];
		}
	}
	return total;
}

- (long long)cacheSize
{
	__block long long size = 0;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		size = [self allocatedSizeOfDirectoryAtURL:cacheDirectoryURL];
	}];
	return size;
}

/*! The owned snapshots under the cache, least recently used first, less the pinned and any holding a .download file. */
- (NSArray<NFKHFHubSnapshot *> *)evictableSnapshotsInDirectoryAtURL:(NSURL *)cacheDirectoryURL
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLContentModificationDateKey ];
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:cacheDirectoryURL
																   includingPropertiesForKeys:keys
																					  options:0
																				 errorHandler:nil];
	NSMutableArray<NFKHFHubSnapshot *> *snapshots = [NSMutableArray array];
	NSMutableArray<NSString *> *partialPaths = [NSMutableArray array];
	NSMutableSet<NSString *> *pinnedPaths = [NSMutableSet set];
	for (NSURL *url in enumerator) {
		if ([url.pathExtension isEqualToString:@"download"]) {
			[partialPaths addObject:url.path];
			continue;
		}
		if ([url.lastPathComponent isEqualToString:NFKHFHubKeepMarkerName]) {
			[pinnedPaths addObject:url.URLByDeletingLastPathComponent.path];
			continue;
		}
		if (![url.lastPathComponent isEqualToString:NFKHFHubOwnedMarkerName]) {
			continue;
		}
		NSDate *modified = nil;
		[url getResourceValue:&modified forKey:NSURLContentModificationDateKey error:NULL];
		NFKHFHubSnapshot *snapshot = [[NFKHFHubSnapshot alloc] init];
		snapshot.URL = url.URLByDeletingLastPathComponent;
		snapshot.lastUsedDate = modified ?: NSDate.distantPast;
		[snapshots addObject:snapshot];
	}

	NSIndexSet *busy = [snapshots indexesOfObjectsPassingTest:^BOOL(NFKHFHubSnapshot *snapshot, NSUInteger index, BOOL *stop) {
		if ([pinnedPaths containsObject:snapshot.URL.path]) {
			return YES;
		}
		NSString *prefix = [snapshot.URL.path stringByAppendingString:@"/"];
		for (NSString *partialPath in partialPaths) {
			if ([partialPath hasPrefix:prefix]) {
				return YES;
			}
		}
		return NO;
	}];
	[snapshots removeObjectsAtIndexes:busy];
	[snapshots sortUsingComparator:^NSComparisonResult(NFKHFHubSnapshot *a, NFKHFHubSnapshot *b) {
		return [a.lastUsedDate compare:b.lastUsedDate];
	}];
	return snapshots;
}

- (BOOL)trimCacheToSizeLimitWithError:(NSError * _Nullable *)outError
{
	return [self trimCacheKeepingSnapshotURL:nil error:outError];
}

- (BOOL)trimCacheKeepingSnapshotURL:(nullable NSURL *)keptSnapshotURL error:(NSError * _Nullable *)outError
{
	long long limit = self.cacheSizeLimit;
	if (limit < 0) {
		return YES;
	}
	__block NSError *removeError = nil;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		long long total = [self allocatedSizeOfDirectoryAtURL:cacheDirectoryURL];
		NSString *keptPath = keptSnapshotURL.URLByStandardizingPath.path;
		for (NFKHFHubSnapshot *snapshot in [self evictableSnapshotsInDirectoryAtURL:cacheDirectoryURL]) {
			if (total <= limit) {
				break;
			}
			if ([snapshot.URL.URLByStandardizingPath.path isEqualToString:keptPath]) {
				continue;
			}
			long long size = [self allocatedSizeOfDirectoryAtURL:snapshot.URL];
			if (![self removeSnapshotAtURL:snapshot.URL inDirectoryAtURL:cacheDirectoryURL error:&removeError]) {
				break;
			}
			total -= size;
		}
	}];
	if (removeError != nil) {
		return [self propagateError:removeError to:outError];
	}
	return YES;
}

- (BOOL)removeSnapshotAtURL:(NSURL *)snapshotURL
		   inDirectoryAtURL:(NSURL *)cacheDirectoryURL
					  error:(NSError * _Nullable *)outError
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSError *error = nil;
	if (![fileManager removeItemAtURL:snapshotURL error:&error] && [fileManager fileExistsAtPath:snapshotURL.path]) {
		if (outError != NULL) {
			*outError = error;
		}
		return NO;
	}
	[self removeEmptyFoldersAboveURL:snapshotURL inDirectoryAtURL:cacheDirectoryURL];
	return YES;
}

- (void)removeEmptyFoldersAboveURL:(NSURL *)url inDirectoryAtURL:(NSURL *)cacheDirectoryURL
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSString *cachePrefix = [cacheDirectoryURL.URLByStandardizingPath.path stringByAppendingString:@"/"];
	NSURL *folder = url.URLByDeletingLastPathComponent.URLByStandardizingPath;
	while ([folder.path hasPrefix:cachePrefix]) {
		NSArray<NSString *> *contents = [fileManager contentsOfDirectoryAtPath:folder.path error:NULL];
		if (contents == nil || contents.count > 0) {
			return;
		}
		[fileManager removeItemAtURL:folder error:NULL];
		folder = folder.URLByDeletingLastPathComponent;
	}
}

- (BOOL)adoptCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		return [self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo"];
	}
	__block BOOL adopted = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		BOOL isDirectory = NO;
		if (![NSFileManager.defaultManager fileExistsAtPath:snapshotURL.path isDirectory:&isDirectory] || !isDirectory) {
			return;
		}
		[self markSnapshotUsedAtURL:snapshotURL];
		adopted = YES;
	}];
	if (!adopted) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"nothing is cached for the repo and revision"];
	}
	return YES;
}

- (BOOL)pinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		return [self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo"];
	}
	__block NSError *pinError = nil;
	__block BOOL pinned = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSFileManager *fileManager = NSFileManager.defaultManager;
		if (![fileManager createDirectoryAtURL:snapshotURL withIntermediateDirectories:YES attributes:nil error:&pinError]) {
			return;
		}
		NSURL *keepURL = [snapshotURL URLByAppendingPathComponent:NFKHFHubKeepMarkerName];
		pinned = [[NSData data] writeToURL:keepURL options:0 error:&pinError];
	}];
	if (!pinned) {
		return [self propagateError:pinError to:outError];
	}
	return YES;
}

- (BOOL)unpinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		return YES;
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo"];
	}
	__block NSError *unpinError = nil;
	__block BOOL unpinned = YES;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSFileManager *fileManager = NSFileManager.defaultManager;
		NSURL *keepURL = [snapshotURL URLByAppendingPathComponent:NFKHFHubKeepMarkerName];
		if (![fileManager removeItemAtURL:keepURL error:&unpinError] && [fileManager fileExistsAtPath:keepURL.path]) {
			unpinned = NO;
			return;
		}
		[self removeEmptyFoldersAboveURL:keepURL inDirectoryAtURL:cacheDirectoryURL];
	}];
	if (!unpinned) {
		return [self propagateError:unpinError to:outError];
	}
	return YES;
}

- (BOOL)isCachedRepoPinned:(NSString *)repo revision:(nullable NSString *)revision
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return NO;
	}
	__block BOOL pinned = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSString *keepPath = [snapshotURL URLByAppendingPathComponent:NFKHFHubKeepMarkerName].path;
		pinned = [NSFileManager.defaultManager fileExistsAtPath:keepPath];
	}];
	return pinned;
}

- (BOOL)removeCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		return YES;
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo"];
	}
	__block BOOL removed = YES;
	__block NSError *removeError = nil;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		removed = [self removeSnapshotAtURL:snapshotURL inDirectoryAtURL:cacheDirectoryURL error:&removeError];
	}];
	if (!removed) {
		return [self propagateError:removeError to:outError];
	}
	return YES;
}

#pragma mark Checksum

- (BOOL)fileAtURL:(NSURL *)url matchesSHA256:(NSString *)expected
{
	NSString *actual = [self sha256OfFileAtURL:url];
	return actual != nil && [actual caseInsensitiveCompare:expected] == NSOrderedSame;
}

- (nullable NSString *)sha256OfFileAtURL:(NSURL *)url
{
	NSFileHandle *handle = [NSFileHandle fileHandleForReadingFromURL:url error:NULL];
	if (handle == nil) {
		return nil;
	}
	CC_SHA256_CTX context;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CC_SHA256_Init(&context);
#pragma clang diagnostic pop
	@try {
		while (YES) {
			@autoreleasepool {
				NSData *chunk = [handle readDataOfLength:(1 << 20)];
				if (chunk.length == 0) {
					break;
				}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				CC_SHA256_Update(&context, chunk.bytes, (CC_LONG)chunk.length);
#pragma clang diagnostic pop
			}
		}
	} @finally {
		[handle closeFile];
	}
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CC_SHA256_Final(digest, &context);
#pragma clang diagnostic pop
	NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
	for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
		[hex appendFormat:@"%02x", digest[i]];
	}
	return hex;
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
											  userInfo:@{ NSLocalizedDescriptionKey: @"the download failed" }];
	return NO;
}

@end
