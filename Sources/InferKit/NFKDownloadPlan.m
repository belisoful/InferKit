//
//  NFKDownloadPlan.m
//  InferKit
//

#import "NFKDownloadPlan.h"
#import "NFKDownload.h"
#import "NFKDownloadInternal.h"

@implementation NFKDownloadPlanFile

@synthesize snapshotURL = _snapshotURL;

- (instancetype)initWithPath:(NSString *)path
				   remoteURL:(NSURL *)remoteURL
					localURL:(NSURL *)localURL
				   byteCount:(long long)byteCount
					  sha256:(nullable NSString *)sha256
					  cached:(BOOL)cached
					optional:(BOOL)optional
{
	self = [super init];
	if (self != nil) {
		_path = [path copy];
		_remoteURL = [remoteURL copy];
		_localURL = [localURL copy];
		_byteCount = byteCount;
		_sha256 = [sha256.lowercaseString copy];
		_cached = cached;
		_optional = optional;
	}
	return self;
}

- (id)copyWithZone:(NSZone *)zone
{
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %lld bytes%@>", self.class, self.path, self.byteCount,
			self.cached ? @" cached" : @""];
}

@end

@implementation NFKDownloadPlan

@synthesize snapshotURL = _snapshotURL;
@synthesize cacheRepository = _cacheRepository;
@synthesize recordsCompletion = _recordsCompletion;

- (instancetype)initWithRepository:(nullable NSString *)repository
						  revision:(nullable NSString *)revision
						 sourceURL:(nullable NSURL *)sourceURL
							 files:(NSArray<NFKDownloadPlanFile *> *)files
						  localURL:(nullable NSURL *)localURL
{
	self = [super init];
	if (self != nil) {
		_repository = [repository copy];
		_revision = [revision copy];
		_sourceURL = [sourceURL copy];
		_files = [files copy];
		_localURL = [localURL copy];
	}
	return self;
}

- (id)copyWithZone:(NSZone *)zone
{
	return self;
}

- (long long)totalByteCount
{
	long long total = 0;
	for (NFKDownloadPlanFile *file in self.files) {
		total += MAX(file.byteCount, 0);
	}
	return total;
}

- (long long)remainingByteCount
{
	long long remaining = 0;
	for (NFKDownloadPlanFile *file in self.files) {
		if (!file.cached) {
			remaining += MAX(file.byteCount, 0);
		}
	}
	return remaining;
}

- (BOOL)sizesKnown
{
	for (NFKDownloadPlanFile *file in self.files) {
		if (file.byteCount < 0) {
			return NO;
		}
	}
	return YES;
}

- (BOOL)isCached
{
	for (NFKDownloadPlanFile *file in self.files) {
		if (!file.cached) {
			return NO;
		}
	}
	return YES;
}

- (nullable NFKDownloadPlanFile *)fileAtPath:(NSString *)path
{
	for (NFKDownloadPlanFile *file in self.files) {
		if ([file.path isEqualToString:path]) {
			return file;
		}
	}
	return nil;
}

- (NFKDownloadPlan *)planRestrictedToPaths:(NSArray<NSString *> *)paths
{
	NSSet<NSString *> *wanted = [NSSet setWithArray:paths];
	NSIndexSet *kept = [self.files indexesOfObjectsPassingTest:^BOOL(NFKDownloadPlanFile *file, NSUInteger index, BOOL *stop) {
		return [wanted containsObject:file.path];
	}];
	NFKDownloadPlan *plan = [[NFKDownloadPlan alloc] initWithRepository:self.repository
															   revision:self.revision
															  sourceURL:self.sourceURL
																  files:[self.files objectsAtIndexes:kept]
															   localURL:self.localURL];
	plan.snapshotURL = self.snapshotURL;
	plan.cacheRepository = self.cacheRepository;
	return plan;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %lu files, %lld bytes, %lld remaining>", self.class,
			self.repository ?: self.sourceURL.absoluteString, (unsigned long)self.files.count,
			self.totalByteCount, self.remainingByteCount];
}

@end

@implementation NFKDownloadStatus

- (instancetype)initWithPresence:(NFKDownloadPresence)presence
					   byteCount:(long long)byteCount
					presentPaths:(NSArray<NSString *> *)presentPaths
					missingPaths:(NSArray<NSString *> *)missingPaths
						localURL:(nullable NSURL *)localURL
{
	return [self initWithPresence:presence byteCount:byteCount presentPaths:presentPaths missingPaths:missingPaths
					 partialPaths:@[] partialByteCount:0 localURL:localURL];
}

- (instancetype)initWithPresence:(NFKDownloadPresence)presence
					   byteCount:(long long)byteCount
					presentPaths:(NSArray<NSString *> *)presentPaths
					missingPaths:(NSArray<NSString *> *)missingPaths
					partialPaths:(NSArray<NSString *> *)partialPaths
				partialByteCount:(long long)partialByteCount
						localURL:(nullable NSURL *)localURL
{
	self = [super init];
	if (self != nil) {
		_presence = presence;
		_byteCount = byteCount;
		_presentPaths = [presentPaths copy];
		_missingPaths = [missingPaths copy];
		_partialPaths = [partialPaths copy];
		_partialByteCount = partialByteCount;
		_localURL = [localURL copy];
	}
	return self;
}

- (NSString *)description
{
	NSArray<NSString *> *names = @[ @"absent", @"partial", @"complete" ];
	return [NSString stringWithFormat:@"<%@ %@, %lld bytes, %lld kept partial>", self.class, names[(NSUInteger)self.presence],
			self.byteCount, self.partialByteCount];
}

@end

@implementation NFKCachedFile

- (instancetype)initWithURL:(NSURL *)URL path:(NSString *)path byteCount:(long long)byteCount reason:(nullable NSString *)reason
{
	self = [super init];
	if (self != nil) {
		_URL = [URL copy];
		_path = [path copy];
		_byteCount = byteCount;
		_reason = [reason copy];
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %lld bytes%@%@>", self.class, self.path, self.byteCount,
			self.reason != nil ? @": " : @"", self.reason ?: @""];
}

@end

@implementation NFKRemovalPlan

@synthesize targetPaths = _targetPaths;
@synthesize bookkeepingURLs = _bookkeepingURLs;
@synthesize parts = _parts;

- (instancetype)initWithRepository:(nullable NSString *)repository
						  revision:(nullable NSString *)revision
						 sourceURL:(nullable NSURL *)sourceURL
					  removedFiles:(NSArray<NFKCachedFile *> *)removedFiles
						 keptFiles:(NSArray<NFKCachedFile *> *)keptFiles
					freedByteCount:(long long)freedByteCount
					   snapshotURL:(nullable NSURL *)snapshotURL
{
	self = [super init];
	if (self != nil) {
		_repository = [repository copy];
		_revision = [revision copy];
		_sourceURL = [sourceURL copy];
		_removedFiles = [removedFiles copy];
		_keptFiles = [keptFiles copy];
		_freedByteCount = freedByteCount;
		_snapshotURL = [snapshotURL copy];
		_bookkeepingURLs = @[];
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ removes %lu, keeps %lu, frees %lld bytes>", self.class,
			self.repository ?: self.sourceURL.absoluteString, (unsigned long)self.removedFiles.count,
			(unsigned long)self.keptFiles.count, self.freedByteCount];
}

@end
