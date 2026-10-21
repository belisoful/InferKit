//
//  NFKHFHub.m
//  InferKit
//

#import "NFKHFHub.h"
#import "NFKHFHubInternal.h"
#import "NFKHFHubTransfer.h"
#import "NFKDownloadInternal.h"
#import "NFKModelCatalog.h"
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
NSString * const NFKHFHubPinMarkerName = @".inferkit-keep";
NSString * const NFKHFHubCompleteMarkerName = @".inferkit-complete";
// The file a transfer reports to, set around fetchURL:toFileURL:error: on the calling thread.
static NSString * const NFKHFHubTransferReportKey = @"NFKHFHubTransferReport";

NSArray<NSString *> *NFKHFHubBookkeepingNames(void)
{
	return @[ NFKHFHubOwnedMarkerName, NFKHFHubKeepMarkerName, NFKHFHubCompleteMarkerName ];
}

/*! Why string cannot name a place inside the folder it is appended to, or nil when it can. Without
	separators the string is a single component. */
static NSString * _Nullable NFKHFHubUnsafePathReason(NSString * _Nullable string, BOOL allowsSeparators)
{
	if (string.length == 0) {
		return @"is empty";
	}
	if ([string hasPrefix:@"/"]) {
		return @"is an absolute path";
	}
	if ([string rangeOfString:@"\\"].location != NSNotFound) {
		return @"holds a backslash";
	}
	if ([string rangeOfCharacterFromSet:[NSCharacterSet characterSetWithRange:NSMakeRange(0, 1)]].location != NSNotFound) {
		return @"holds a NUL character";
	}
	NSArray<NSString *> *components = [string componentsSeparatedByString:@"/"];
	if (!allowsSeparators && components.count > 1) {
		return @"holds a /";
	}
	for (NSString *component in components) {
		if (component.length == 0 || [component isEqualToString:@"."] || [component isEqualToString:@".."]) {
			return @"holds an empty, \".\", or \"..\" component";
		}
	}
	return nil;
}

NSError * _Nullable NFKHFHubUnsafePathError(NSString * _Nullable string, NSString *kind, BOOL allowsSeparators)
{
	NSString *problem = NFKHFHubUnsafePathReason(string, allowsSeparators);
	if (problem == nil) {
		return nil;
	}
	NSString *reason = [NSString stringWithFormat:@"the %@ \"%@\" %@, so it can name a location outside the cache folder",
						kind, string ?: @"", problem];
	return [NFKDownload errorWithCode:NFKDownloadErrorUnsafePath reason:reason];
}

/*! path with "." and ".." components applied and empty ones dropped, reading nothing from disk. */
static NSString *NFKHFHubLexicalPath(NSString *path)
{
	NSMutableArray<NSString *> *stack = [NSMutableArray array];
	for (NSString *component in [path componentsSeparatedByString:@"/"]) {
		if (component.length == 0 || [component isEqualToString:@"."]) {
			continue;
		}
		if ([component isEqualToString:@".."]) {
			if (stack.count > 0) {
				[stack removeLastObject];
			}
			continue;
		}
		[stack addObject:component];
	}
	return [@"/" stringByAppendingString:[stack componentsJoinedByString:@"/"]];
}

/*! path with the links of its deepest existing ancestor resolved by realpath and the missing
	components appended, so a path that does not exist yet still resolves through the folders that do. */
static NSString *NFKHFHubRealPath(NSString *path)
{
	NSString *existing = path;
	NSMutableArray<NSString *> *missing = [NSMutableArray array];
	while (YES) {
		char resolved[PATH_MAX];
		if (realpath(existing.fileSystemRepresentation, resolved) != NULL) {
			NSString *real = [NSFileManager.defaultManager stringWithFileSystemRepresentation:resolved length:strlen(resolved)];
			for (NSString *component in missing.reverseObjectEnumerator) {
				real = [real stringByAppendingPathComponent:component];
			}
			return real;
		}
		if (existing.length <= 1) {
			return path;
		}
		[missing addObject:existing.lastPathComponent];
		existing = existing.stringByDeletingLastPathComponent;
	}
}

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

@interface NFKHFHubSnapshotHold ()
@property (nonatomic, copy) NSString *registryKey;
@property (nonatomic, readwrite, getter=isValid) BOOL valid;
- (instancetype)initWithSnapshotURL:(NSURL *)snapshotURL;
@end

@implementation NFKHFHubSnapshotHold

@synthesize snapshotURL = _snapshotURL;
@synthesize registryKey = _registryKey;
@synthesize valid = _valid;

- (instancetype)initWithSnapshotURL:(NSURL *)snapshotURL
{
	self = [super init];
	if (self != nil) {
		_snapshotURL = [snapshotURL copy];
		_registryKey = [NFKDownload holdSnapshotURL:snapshotURL];
		_valid = YES;
	}
	return self;
}

- (void)dealloc
{
	[self invalidate];
}

- (BOOL)isValid
{
	@synchronized (self) {
		return _valid;
	}
}

- (void)invalidate
{
	NSString *key = nil;
	@synchronized (self) {
		if (!self.valid) {
			return;
		}
		self.valid = NO;
		key = self.registryKey;
	}
	[NFKDownload endHoldOfSnapshotKey:key];
}

- (void)setValid:(BOOL)valid
{
	@synchronized (self) {
		_valid = valid;
	}
}

@end

@implementation NFKHFHub

@synthesize cacheDirectoryURL = _cacheDirectoryURL;
@synthesize endpointURL = _endpointURL;
@synthesize session = _session;
@synthesize accessToken = _accessToken;
@synthesize cacheSizeLimit = _cacheSizeLimit;
@synthesize excludesCacheFromBackup = _excludesCacheFromBackup;
@synthesize modelCatalog = _modelCatalog;
@synthesize followsFolderLinksOutOfCache = _followsFolderLinksOutOfCache;
@synthesize maximumFileByteCount = _maximumFileByteCount;

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
		_maximumFileByteCount = -1;
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

- (NFKModelCatalog *)modelCatalog
{
	@synchronized (self) {
		return _modelCatalog ?: NFKModelCatalog.sharedCatalog;
	}
}

- (void)setModelCatalog:(nullable NFKModelCatalog *)modelCatalog
{
	@synchronized (self) {
		_modelCatalog = modelCatalog;
	}
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

- (nullable NSError *)refusalForRepo:(nullable NSString *)repo
							revision:(nullable NSString *)revision
								path:(nullable NSString *)path
{
	NSError *refusal = NFKHFHubUnsafePathError(repo, @"repository", YES);
	if (refusal == nil && revision.length > 0) {
		refusal = NFKHFHubUnsafePathError(revision, @"revision", YES);
	}
	if (refusal == nil && path != nil) {
		refusal = NFKHFHubUnsafePathError(path, @"path", YES);
	}
	return refusal;
}

- (nullable NSURL *)remoteURLForRepo:(NSString *)repo
						  revision:(nullable NSString *)revision
							  path:(NSString *)path
{
	if ([self refusalForRepo:repo revision:revision path:path] != nil) {
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
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (snapshotURL == nil || NFKHFHubUnsafePathReason(path, YES) != nil) {
		return nil;
	}
	return [self URL:snapshotURL byAppendingPathString:path];
}

- (nullable NSURL *)snapshotURLForRepo:(NSString *)repo revision:(nullable NSString *)revision
{
	return [self snapshotURLForRepo:repo revision:revision error:NULL];
}

- (nullable NSURL *)snapshotURLForRepo:(nullable NSString *)repo
							  revision:(nullable NSString *)revision
								 error:(NSError * _Nullable *)outError
{
	NSError *refusal = [self refusalForRepo:repo revision:revision path:nil];
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	if (self.cacheDirectoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
		return nil;
	}
	NSURL *url = [self URL:self.cacheDirectoryURL byAppendingPathString:repo];
	return [url URLByAppendingPathComponent:[self revisionFolderName:revision]];
}

/*! The folder name of a revision: the normalized revision with "%" stored as %25 and "/" as %2F, so
	every revision is one folder and refs/pr/1 never nests inside a revision named refs. */
- (NSString *)revisionFolderName:(nullable NSString *)revision
{
	NSString *escaped = [[self normalizedRevision:revision] stringByReplacingOccurrencesOfString:@"%" withString:@"%25"];
	return [escaped stringByReplacingOccurrencesOfString:@"/" withString:@"%2F"];
}

- (nullable NFKHFHubSnapshotHold *)holdSnapshotOfRepo:(NSString *)repo
											 revision:(nullable NSString *)revision
												error:(NSError * _Nullable *)outError
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return nil;
	}
	return [[NFKHFHubSnapshotHold alloc] initWithSnapshotURL:snapshotURL];
}

- (BOOL)isURL:(NSURL *)url insideCacheURL:(NSURL *)cacheDirectoryURL
{
	NSString *path = NFKHFHubLexicalPath(url.path);
	NSMutableArray<NSString *> *roots = [NSMutableArray arrayWithObjects:cacheDirectoryURL.path,
										 cacheDirectoryURL.URLByResolvingSymlinksInPath.path, nil];
	// URLByResolvingSymlinksInPath drops a leading /private that realpath keeps, and a directory
	// enumerator can hand back either form.
	char resolved[PATH_MAX];
	if (realpath(cacheDirectoryURL.fileSystemRepresentation, resolved) != NULL) {
		[roots addObject:[NSFileManager.defaultManager stringWithFileSystemRepresentation:resolved length:strlen(resolved)]];
	}
	BOOL lexicallyInside = NO;
	for (NSString *root in roots) {
		NSString *prefix = NFKHFHubLexicalPath(root);
		if (![prefix hasSuffix:@"/"]) {
			prefix = [prefix stringByAppendingString:@"/"];
		}
		if ([path hasPrefix:prefix] && path.length > prefix.length) {
			lexicallyInside = YES;
			break;
		}
	}
	if (!lexicallyInside || self.followsFolderLinksOutOfCache) {
		return lexicallyInside;
	}
	// The folder the item lands in, with its links resolved, must stay under the cache's real path.
	// The item itself is not resolved: a removal moves a link, not its target.
	NSString *realRoot = NFKHFHubRealPath(NFKHFHubLexicalPath(cacheDirectoryURL.path));
	NSString *realFolder = NFKHFHubRealPath(path.stringByDeletingLastPathComponent);
	return [realFolder isEqualToString:realRoot] || [realFolder hasPrefix:[realRoot stringByAppendingString:@"/"]];
}

- (BOOL)requireURL:(NSURL *)url insideCacheURL:(NSURL *)cacheDirectoryURL error:(NSError * _Nullable *)outError
{
	if ([self isURL:url insideCacheURL:cacheDirectoryURL]) {
		return YES;
	}
	NSString *reason = [NSString stringWithFormat:@"%@ is outside the cache folder %@, directly or through a linked "
						"folder, and the hub writes and removes only inside it", url.path, cacheDirectoryURL.path];
	return [self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsafePath reason:reason] to:outError];
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
	NSError *refusal = [self refusalForRepo:repo revision:revision path:path];
	if (refusal != nil) {
		[self propagateError:refusal to:outError];
		return nil;
	}
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	NSURL *remoteURL = [self remoteURLForRepo:repo revision:revision path:path];
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision];
	if (localURL == nil || remoteURL == nil || snapshotURL == nil) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo or path"];
		return nil;
	}
	return [self fetchFileFromURL:remoteURL
						 localURL:localURL
					  snapshotURL:snapshotURL
							 path:path
						   sha256:expectedSHA256
				expectedByteCount:NFKDownloadUnknownByteCount
							error:outError];
}

- (BOOL)cachedFileAtURL:(NSURL *)localURL matchesSHA256:(nullable NSString *)expectedSHA256 byteCount:(long long)expectedByteCount
{
	if (expectedByteCount >= 0) {
		NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:localURL.path error:NULL];
		if (attributes == nil || [attributes[NSFileSize] longLongValue] != expectedByteCount) {
			return NO;
		}
	}
	return expectedSHA256 == nil || [self fileAtURL:localURL matchesSHA256:expectedSHA256];
}

- (nullable NSURL *)fetchFileFromURL:(NSURL *)remoteURL
							localURL:(NSURL *)localURL
						 snapshotURL:(NSURL *)snapshotURL
								path:(NSString *)path
							  sha256:(nullable NSString *)expectedSHA256
				   expectedByteCount:(long long)expectedByteCount
							   error:(NSError * _Nullable *)outError
{
	NFKDownload *download = NFKDownload.currentDownload;
	if (download.isFinished) {
		download = nil;
	}
	if (download.isStopped) {
		[self propagateError:download.stopError to:outError];
		return nil;
	}
	NSURL *cacheDirectoryURL = self.cacheDirectoryURL;
	if (cacheDirectoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
		return nil;
	}
	if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:outError]
		|| ![self requireURL:localURL insideCacheURL:cacheDirectoryURL error:outError]) {
		return nil;
	}

	NSFileManager *fileManager = NSFileManager.defaultManager;
	BOOL scoped = [cacheDirectoryURL startAccessingSecurityScopedResource];
	// Held for the call, so a trim another download runs meanwhile cannot evict the snapshot, with or
	// without a current NFKDownload.
	NSString *holdKey = [NFKDownload holdSnapshotURL:snapshotURL];
	@try {
		if ([fileManager fileExistsAtPath:localURL.path]) {
			if ([self cachedFileAtURL:localURL matchesSHA256:expectedSHA256 byteCount:expectedByteCount]) {
				[self excludeCacheFromBackupIfNeeded];
				[self markSnapshotUsedAtURL:snapshotURL];
				long long size = [[fileManager attributesOfItemAtPath:localURL.path error:NULL][NSFileSize] longLongValue];
				[download fileDidFinishAtLocalURL:localURL path:path byteCount:size cached:YES];
				return localURL;
			}
			[fileManager removeItemAtURL:localURL error:NULL];
		}

		// Registered before the folder exists, so a trim running meanwhile cannot evict it.
		[download registerSnapshotURL:snapshotURL];
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
		NSString *keptTag = nil;
		long long kept = [self keptByteCountOfPartialURL:partialURL expectedByteCount:expectedByteCount entityTag:&keptTag];
		if (kept == 0) {
			[fileManager removeItemAtURL:partialURL error:NULL];
		}
		[download fileWillStartAtLocalURL:localURL path:path expectedByteCount:expectedByteCount keptByteCount:kept];
		// A partial that already holds every listed byte was stopped before it was checked and moved.
		BOOL whole = kept > 0 && kept == expectedByteCount;
		if (!whole && ![self fetchURL:remoteURL toFileURL:partialURL localURL:localURL path:path
					   expectedByteCount:expectedByteCount keptByteCount:kept keptEntityTag:keptTag
								download:download error:outError]) {
			if (!(download.isPaused && !download.isCancelled)) {
				[fileManager removeItemAtURL:partialURL error:NULL];
			}
			[download fileDidFailAtLocalURL:localURL];
			return nil;
		}

		NSError *contentError = [self contentErrorOfFileAtURL:partialURL path:path
											 expectedByteCount:expectedByteCount sha256:expectedSHA256];
		if (contentError != nil) {
			[fileManager removeItemAtURL:partialURL error:NULL];
			[download fileDidFailAtLocalURL:localURL];
			[self propagateError:contentError to:outError];
			return nil;
		}

		[NFKHFHubTransfer clearEntityTagOfFileAtURL:partialURL];
		[fileManager removeItemAtURL:localURL error:NULL];
		NSError *moveError = nil;
		if (![fileManager moveItemAtURL:partialURL toURL:localURL error:&moveError]) {
			[fileManager removeItemAtURL:partialURL error:NULL];
			[download fileDidFailAtLocalURL:localURL];
			[self propagateError:moveError to:outError];
			return nil;
		}
		long long size = [[fileManager attributesOfItemAtPath:localURL.path error:NULL][NSFileSize] longLongValue];
		[download fileDidFinishAtLocalURL:localURL path:path byteCount:size cached:NO];
		// The file is in place; a failed eviction leaves the cache over its limit and the download
		// still succeeds.
		[self trimCacheKeepingSnapshotURL:snapshotURL error:NULL];
		return localURL;
	} @finally {
		[NFKDownload endHoldOfSnapshotKey:holdKey];
		if (scoped) {
			[cacheDirectoryURL stopAccessingSecurityScopedResource];
		}
	}
}

/*! Why a fetched file is not the listed one, or nil: its size differs from a non-negative
	expectedByteCount, or its contents from expectedSHA256. */
- (nullable NSError *)contentErrorOfFileAtURL:(NSURL *)url
										 path:(NSString *)path
							expectedByteCount:(long long)expectedByteCount
									   sha256:(nullable NSString *)expectedSHA256
{
	if (expectedByteCount >= 0) {
		NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
		long long byteCount = attributes != nil ? [attributes[NSFileSize] longLongValue] : -1;
		if (byteCount != expectedByteCount) {
			NSString *reason = [NSString stringWithFormat:@"%@ arrived with %lld bytes, and its listed size is %lld",
								path, byteCount, expectedByteCount];
			return [NFKDownload errorWithCode:NFKDownloadErrorSizeMismatch reason:reason];
		}
	}
	if (expectedSHA256 != nil && ![self fileAtURL:url matchesSHA256:expectedSHA256]) {
		NSString *reason = [NSString stringWithFormat:@"%@ failed its checksum", path];
		return [NFKDownload errorWithCode:NFKDownloadErrorChecksumMismatch reason:reason];
	}
	return nil;
}

/*! The bytes a paused download kept in partialURL for a later one to continue: the partial's size when
	it records a strong ETag and holds no more than expectedByteCount, else 0. */
- (long long)keptByteCountOfPartialURL:(NSURL *)partialURL
					 expectedByteCount:(long long)expectedByteCount
							 entityTag:(NSString * _Nullable * _Nonnull)entityTag
{
	long long size = [self keptByteCountOfPartialURL:partialURL];
	if (size <= 0 || (expectedByteCount >= 0 && size > expectedByteCount)) {
		return 0;
	}
	*entityTag = [NFKHFHubTransfer entityTagOfPartialFileAtURL:partialURL];
	return size;
}

- (long long)keptByteCountOfPartialURL:(NSURL *)partialURL
{
	if ([NFKHFHubTransfer entityTagOfPartialFileAtURL:partialURL] == nil) {
		return 0;
	}
	NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:partialURL.path
																									  error:NULL];
	return attributes != nil ? [attributes[NSFileSize] longLongValue] : 0;
}

/*! Runs the transport seam with the file's report set on the calling thread for the default
	transfer to read. */
- (BOOL)fetchURL:(NSURL *)remoteURL
	   toFileURL:(NSURL *)partialURL
		localURL:(NSURL *)localURL
			path:(NSString *)path
expectedByteCount:(long long)expectedByteCount
   keptByteCount:(long long)keptByteCount
   keptEntityTag:(nullable NSString *)keptEntityTag
		download:(nullable NFKDownload *)download
		   error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *threadDictionary = NSThread.currentThread.threadDictionary;
	id previous = threadDictionary[NFKHFHubTransferReportKey];
	NFKHFHubTransferReport *report = [[NFKHFHubTransferReport alloc] init];
	report.download = download;
	report.localURL = localURL;
	report.path = path;
	report.expectedByteCount = expectedByteCount;
	report.ceilingByteCount = [self ceilingByteCountForExpectedByteCount:expectedByteCount];
	report.keptByteCount = keptByteCount;
	report.keptEntityTag = keptEntityTag;
	threadDictionary[NFKHFHubTransferReportKey] = report;
	@try {
		return [self fetchURL:remoteURL toFileURL:partialURL error:outError];
	} @finally {
		if (previous != nil) {
			threadDictionary[NFKHFHubTransferReportKey] = previous;
		} else {
			[threadDictionary removeObjectForKey:NFKHFHubTransferReportKey];
		}
	}
}

/*! The most bytes a transfer streams whatever the server announces: maximumFileByteCount when set,
	else the cache volume's free space for a file of unknown size; -1 for none. */
- (long long)ceilingByteCountForExpectedByteCount:(long long)expectedByteCount
{
	long long maximum = self.maximumFileByteCount;
	if (maximum >= 0) {
		return maximum;
	}
	return expectedByteCount < 0 ? [self availableCapacityOfCache] : -1;
}

- (void)downloadRepo:(NSString *)repo
			revision:(nullable NSString *)revision
				path:(NSString *)path
			  sha256:(nullable NSString *)expectedSHA256
   completionHandler:(void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	NFKDownload *download = NFKDownload.currentDownload;
	[download beginActivity];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		[download becomeCurrent];
		NSURL *localURL = [self downloadRepo:repo revision:revision path:path sha256:expectedSHA256 error:&error];
		[download resignCurrent];
		[download endActivityWithError:error];
		completionHandler(localURL, error);
	});
}

- (NSMutableURLRequest *)requestForURL:(NSURL *)URL
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:URL];
	NSString *token = self.accessToken;
	NSString *endpointHost = self.endpointURL.host;
	if (token.length > 0 && endpointHost != nil && [URL.host caseInsensitiveCompare:endpointHost] == NSOrderedSame) {
		[request setValue:[@"Bearer " stringByAppendingString:token] forHTTPHeaderField:@"Authorization"];
	}
	return request;
}

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError * _Nullable *)outError
{
	NFKHFHubTransferReport *report = NSThread.currentThread.threadDictionary[NFKHFHubTransferReportKey];
	return [NFKHFHubTransfer transferRequest:[self requestForURL:remoteURL]
							   configuration:self.session.configuration
								   toFileURL:destinationURL
									  report:report
									   error:outError];
}

- (nullable NSData *)fetchMetadataFromURL:(NSURL *)URL
							   HTTPMethod:(NSString *)method
								 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
									error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self requestForURL:URL];
	request.HTTPMethod = method;
	[request setValue:@"application/json" forHTTPHeaderField:@"Accept"];
	return [self bodyOfRequest:request response:outResponse error:outError];
}

/*! Sends the request on session, cancelled with the current download, and returns the body whatever the
	status. */
- (nullable NSData *)bodyOfRequest:(NSURLRequest *)request
						  response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
							 error:(NSError * _Nullable *)outError
{
	__block NSData *body = nil;
	__block NSURLResponse *answer = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request
												 completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		body = data;
		answer = response;
		failure = error;
		dispatch_semaphore_signal(done);
	}];
	NFKDownload *download = NFKDownload.currentDownload;
	id token = [download addCancellationHandler:^{
		[task cancel];
	}];
	[task resume];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	if (token != nil) {
		[download removeCancellationHandler:token];
	}
	if (download.isStopped) {
		[self propagateError:download.stopError to:outError];
		return nil;
	}
	if (failure != nil) {
		[self propagateError:failure to:outError];
		return nil;
	}
	if (outResponse != NULL) {
		*outResponse = [self HTTPResponseForResponse:answer];
	}
	return body ?: [NSData data];
}

static const unsigned long long NFKSafetensorsHeaderLimit = 100000000ULL;

- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfRepo:(NSString *)repo
														  revision:(nullable NSString *)revision
															  path:(NSString *)path
															 error:(NSError * _Nullable *)outError
{
	NSString *subject = [NSString stringWithFormat:@"%@ in %@", path, repo];
	if ([self isCachedRepo:repo revision:revision path:path]) {
		return [self safetensorsHeaderOfFileAtURL:[self localURLForRepo:repo revision:revision path:path]
										  subject:subject error:outError];
	}
	NSURL *remoteURL = [self remoteURLForRepo:repo revision:revision path:path];
	if (remoteURL == nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ at %@ is not a safe location", subject, revision ?: @"main"];
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsafePath reason:reason] to:outError];
		return nil;
	}
	NSData *lengthBytes = [self bytesOfURL:remoteURL offset:0 length:8 subject:subject error:outError];
	if (lengthBytes == nil) {
		return nil;
	}
	unsigned long long headerLength = 0;
	if (![self readSafetensorsHeaderLength:&headerLength from:lengthBytes subject:subject error:outError]) {
		return nil;
	}
	NSData *headerBytes = [self bytesOfURL:remoteURL offset:8 length:headerLength subject:subject error:outError];
	if (headerBytes == nil) {
		return nil;
	}
	return [self safetensorsHeaderFromData:headerBytes subject:subject error:outError];
}

- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfFileAtURL:(NSURL *)url
																subject:(NSString *)subject
																  error:(NSError * _Nullable *)outError
{
	NSError *readError = nil;
	NSFileHandle *handle = [NSFileHandle fileHandleForReadingFromURL:url error:&readError];
	if (handle == nil) {
		[self propagateError:readError to:outError];
		return nil;
	}
	NSData *lengthBytes = [handle readDataUpToLength:8 error:&readError];
	unsigned long long headerLength = 0;
	if (lengthBytes == nil) {
		[handle closeAndReturnError:NULL];
		[self propagateError:readError to:outError];
		return nil;
	}
	if (![self readSafetensorsHeaderLength:&headerLength from:lengthBytes subject:subject error:outError]) {
		[handle closeAndReturnError:NULL];
		return nil;
	}
	NSData *headerBytes = [handle readDataUpToLength:(NSUInteger)headerLength error:&readError];
	[handle closeAndReturnError:NULL];
	if (headerBytes == nil) {
		[self propagateError:readError to:outError];
		return nil;
	}
	return [self safetensorsHeaderFromData:headerBytes subject:subject error:outError];
}

/*! The bytes [offset, offset + length) of a remote file, by an HTTP range request. A partial answer (206)
	starts at offset, and one that runs past the range's end is cut to it. A server that ignores the range
	answers 200 with the whole file, which this reads whole and cuts; a whole file shorter than the range's
	end is refused. */
- (nullable NSData *)bytesOfURL:(NSURL *)URL
						 offset:(unsigned long long)offset
						 length:(unsigned long long)length
						subject:(NSString *)subject
						  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self requestForURL:URL];
	BOOL hadToken = [request valueForHTTPHeaderField:@"Authorization"] != nil;
	[request setValue:[NSString stringWithFormat:@"bytes=%llu-%llu", offset, offset + length - 1]
   forHTTPHeaderField:@"Range"];
	NSHTTPURLResponse *response = nil;
	NSData *body = [self bodyOfRequest:request response:&response error:outError];
	if (body == nil) {
		return nil;
	}
	NSInteger status = response != nil ? response.statusCode : 200;
	if (status < 200 || status >= 300) {
		[self propagateError:[NFKHFHubTransfer errorForHTTPStatus:status subject:subject hadToken:hadToken] to:outError];
		return nil;
	}
	if (status == 200 && body.length >= offset + length) {
		body = [body subdataWithRange:NSMakeRange((NSUInteger)offset, (NSUInteger)length)];
	} else if (status == 206 && body.length > length) {
		body = [body subdataWithRange:NSMakeRange(0, (NSUInteger)length)];
	}
	if (body.length != length) {
		NSString *reason = [NSString stringWithFormat:@"%@ ended before byte %llu", subject, offset + length];
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:reason] to:outError];
		return nil;
	}
	return body;
}

- (BOOL)readSafetensorsHeaderLength:(unsigned long long *)headerLength
							   from:(NSData *)bytes
							subject:(NSString *)subject
							  error:(NSError * _Nullable *)outError
{
	if (bytes.length < 8) {
		NSString *reason = [NSString stringWithFormat:@"%@ is shorter than a safetensors header", subject];
		return [self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:reason] to:outError];
	}
	unsigned long long length = 0;
	const uint8_t *raw = bytes.bytes;
	for (int index = 7; index >= 0; index--) {
		length = (length << 8) | raw[index];
	}
	if (length == 0 || length > NFKSafetensorsHeaderLimit) {
		NSString *reason = [NSString stringWithFormat:@"%@ states a header of %llu bytes, outside the safetensors limit",
							subject, length];
		return [self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:reason] to:outError];
	}
	*headerLength = length;
	return YES;
}

- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderFromData:(NSData *)data
															 subject:(NSString *)subject
															   error:(NSError * _Nullable *)outError
{
	id header = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![header isKindOfClass:NSDictionary.class]) {
		NSString *reason = [NSString stringWithFormat:@"%@ has no safetensors JSON header", subject];
		[self propagateError:[NFKDownload errorWithCode:NFKDownloadErrorUnsupportedSource reason:reason] to:outError];
		return nil;
	}
	return header;
}

/*! The response as HTTP. A response of another scheme, such as a file URL's, reads as status 200 with
	its length. */
- (nullable NSHTTPURLResponse *)HTTPResponseForResponse:(nullable NSURLResponse *)response
{
	if (response == nil || [response isKindOfClass:NSHTTPURLResponse.class]) {
		return (NSHTTPURLResponse *)response;
	}
	NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
	if (response.expectedContentLength >= 0) {
		headers[@"Content-Length"] = [NSString stringWithFormat:@"%lld", response.expectedContentLength];
	}
	return [[NSHTTPURLResponse alloc] initWithURL:response.URL ?: [NSURL URLWithString:@"about:blank"]
									   statusCode:200
									  HTTPVersion:@"HTTP/1.1"
									 headerFields:headers];
}

#pragma mark Cache management

- (void)performWithCacheAccess:(void (NS_NOESCAPE ^)(NSURL *cacheDirectoryURL))body
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
	// A marker that is a link would carry the writes below to the link's target.
	if ([[fileManager attributesOfItemAtPath:markerPath error:NULL][NSFileType] isEqualToString:NSFileTypeSymbolicLink]) {
		[fileManager removeItemAtPath:markerPath error:NULL];
	}
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
	// NSURL caches resource values per instance, and the free space changes with every download.
	[cacheDirectoryURL removeCachedResourceValueForKey:key];
	[cacheDirectoryURL getResourceValue:&capacity forKey:key error:NULL];
	return capacity != nil ? capacity.longLongValue : -1;
}

- (long long)cacheSize
{
	__block long long size = 0;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		size = [self allocatedSizeOfDirectoryAtURL:cacheDirectoryURL];
	}];
	return size;
}

/*! The owned snapshots under the cache, least recently used first, less the pinned, any holding a
	.download file, and any an unfinished NFKDownload in this process writes into. */
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
		if ([pinnedPaths containsObject:snapshot.URL.path] || [NFKDownload isSnapshotURLInUse:snapshot.URL]) {
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
			__block long long freed = 0;
			// Checked and removed under the registry lock, so no download registers the snapshot between.
			[NFKDownload performWithSnapshotRegistryLocked:^{
				if ([NFKDownload isSnapshotURLInUse:snapshot.URL]) {
					return;
				}
				long long size = [self allocatedSizeOfDirectoryAtURL:snapshot.URL];
				if ([self removeSnapshotAtURL:snapshot.URL inDirectoryAtURL:cacheDirectoryURL error:&removeError]) {
					freed = size;
				}
			}];
			if (removeError != nil) {
				break;
			}
			total -= freed;
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
	if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:outError]) {
		return NO;
	}
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
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return NO;
	}
	__block BOOL adopted = NO;
	__block NSError *refusal = nil;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:&refusal]) {
			return;
		}
		BOOL isDirectory = NO;
		if (![NSFileManager.defaultManager fileExistsAtPath:snapshotURL.path isDirectory:&isDirectory] || !isDirectory) {
			return;
		}
		[self markSnapshotUsedAtURL:snapshotURL];
		adopted = YES;
	}];
	if (refusal != nil) {
		return [self propagateError:refusal to:outError];
	}
	if (!adopted) {
		return [self setError:outError code:kNFKError_InferenceBackendFailure reason:@"nothing is cached for the repo and revision"];
	}
	return YES;
}

- (BOOL)pinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return NO;
	}
	__block NSError *pinError = nil;
	__block BOOL pinned = NO;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSFileManager *fileManager = NSFileManager.defaultManager;
		if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:&pinError]
			|| ![fileManager createDirectoryAtURL:snapshotURL withIntermediateDirectories:YES attributes:nil error:&pinError]) {
			return;
		}
		NSURL *keepURL = [snapshotURL URLByAppendingPathComponent:NFKHFHubKeepMarkerName];
		// Written atomically, so a pin that is a link is replaced, never written through.
		pinned = [[NSData data] writeToURL:keepURL options:NSDataWritingAtomic error:&pinError];
	}];
	if (!pinned) {
		return [self propagateError:pinError to:outError];
	}
	return YES;
}

- (BOOL)unpinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError
{
	NSError *refusal = [self refusalForRepo:repo revision:revision path:nil];
	if (refusal != nil) {
		return [self propagateError:refusal to:outError];
	}
	if (self.cacheDirectoryURL == nil) {
		return YES;
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return NO;
	}
	__block NSError *unpinError = nil;
	__block BOOL unpinned = YES;
	[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
		NSFileManager *fileManager = NSFileManager.defaultManager;
		if (![self requireURL:snapshotURL insideCacheURL:cacheDirectoryURL error:&unpinError]) {
			unpinned = NO;
			return;
		}
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
	NSError *refusal = [self refusalForRepo:repo revision:revision path:nil];
	if (refusal != nil) {
		return [self propagateError:refusal to:outError];
	}
	if (self.cacheDirectoryURL == nil) {
		return YES;
	}
	NSURL *snapshotURL = [self snapshotURLForRepo:repo revision:revision error:outError];
	if (snapshotURL == nil) {
		return NO;
	}
	__block BOOL removed = YES;
	__block NSError *removeError = nil;
	[NFKDownload performWithSnapshotRegistryLocked:^{
		if ([NFKDownload isSnapshotURLInUse:snapshotURL]) {
			NSString *reason = [NSString stringWithFormat:@"%@ is being downloaded; cancel the download before removing it", repo];
			removeError = [NFKDownload errorWithCode:NFKDownloadErrorInUse reason:reason];
			removed = NO;
			return;
		}
		[self performWithCacheAccess:^(NSURL *cacheDirectoryURL) {
			removed = [self removeSnapshotAtURL:snapshotURL inDirectoryAtURL:cacheDirectoryURL error:&removeError];
		}];
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
