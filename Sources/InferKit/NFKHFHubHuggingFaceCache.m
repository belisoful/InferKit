//
//  NFKHFHubHuggingFaceCache.m
//  InferKit
//
//  Catalog model sources found in the caches Hugging Face's own tools keep, which the hub reads and
//  never writes.
//

#import "NFKHFHub.h"
#import "NFKHFHubInternal.h"
#import "NFKModelDescriptor.h"

/*! The prefix of a model repository's folder in a Hugging Face cache. */
static NSString * const NFKHFHubHuggingFaceRepositoryPrefix = @"models--";

/*! path with a leading ~ expanded to homeDirectory, as Python's os.path.expanduser expands it. */
static NSString *NFKHFHubExpandingHome(NSString *path, NSString *homeDirectory)
{
	if ([path isEqualToString:@"~"]) {
		return homeDirectory;
	}
	if ([path hasPrefix:@"~/"]) {
		return [homeDirectory stringByAppendingPathComponent:[path substringFromIndex:2]];
	}
	return path;
}

NSURL *NFKHFHubHuggingFaceCacheURL(NSDictionary<NSString *, NSString *> *environment, NSString *homeDirectory)
{
	NSString * _Nullable (^variable)(NSString *) = ^NSString * _Nullable (NSString *name) {
		NSString *value = environment[name];
		return value.length > 0 ? value : nil;
	};
	NSString *path = variable(@"HF_HUB_CACHE") ?: variable(@"HUGGINGFACE_HUB_CACHE");
	if (path == nil) {
		NSString *home = variable(@"HF_HOME");
		if (home == nil) {
			NSString *cache = variable(@"XDG_CACHE_HOME") ?: [homeDirectory stringByAppendingPathComponent:@".cache"];
			home = [cache stringByAppendingPathComponent:@"huggingface"];
		}
		path = [home stringByAppendingPathComponent:@"hub"];
	}
	return [NSURL fileURLWithPath:NFKHFHubExpandingHome(path, homeDirectory) isDirectory:YES];
}

@implementation NFKHFHub (NFKHuggingFaceCache)

- (nullable NSURL *)huggingFaceCacheLocationForModelSource:(NFKModelSource *)source
{
	return [self matchForModelSource:source inModelFolders:NO inHuggingFaceCaches:YES parts:NULL].location;
}

#pragma mark Matching

- (nullable NFKHFHubModelFolderMatch *)huggingFaceCacheMatchOfFiles:(nullable NSArray<NFKModelFile *> *)files
															 source:(NFKModelSource *)source
{
	if (NFKHFHubUnsafePathError(source.repository, @"repository", YES) != nil) {
		return nil;
	}
	NSString *folderName = [NFKHFHubHuggingFaceRepositoryPrefix
							stringByAppendingString:[source.repository stringByReplacingOccurrencesOfString:@"/"
																								 withString:@"--"]];
	for (NSURL *root in self.huggingFaceCacheURLs) {
		BOOL scoped = [root startAccessingSecurityScopedResource];
		NFKHFHubModelFolderMatch *match = nil;
		@try {
			for (NSURL *repositoryURL in [self entriesNamed:folderName inDirectoryURL:root]) {
				NSURL *snapshotURL = [self huggingFaceSnapshotURLInRepositoryURL:repositoryURL revision:source.revision];
				match = snapshotURL != nil ? [self matchOfFolderURL:snapshotURL files:files source:source ignoredPath:nil]
										   : nil;
				if (match != nil) {
					break;
				}
			}
		} @finally {
			if (scoped) {
				[root stopAccessingSecurityScopedResource];
			}
		}
		if (match != nil) {
			return match;
		}
	}
	return nil;
}

/*! The snapshot folder of a revision in a repository's cache folder: the folder a commit names, or the
	folder of the commit the revision's ref holds (refs/main without a revision); nil when the ref or the
	folder is missing. */
- (nullable NSURL *)huggingFaceSnapshotURLInRepositoryURL:(NSURL *)repositoryURL revision:(nullable NSString *)revision
{
	NSString *commit = revision;
	if (!NFKHFHubIsCommitHash(revision)) {
		NSString *ref = [self normalizedRevision:revision];
		if (NFKHFHubUnsafePathError(ref, @"revision", YES) != nil) {
			return nil;
		}
		NSURL *refURL = [self URL:[repositoryURL URLByAppendingPathComponent:@"refs" isDirectory:YES]
			byAppendingPathString:ref];
		NSString *text = [NSString stringWithContentsOfURL:refURL encoding:NSUTF8StringEncoding error:NULL];
		commit = [text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
		if (!NFKHFHubIsCommitHash(commit)) {
			return nil;
		}
	}
	NSURL *snapshotsURL = [repositoryURL URLByAppendingPathComponent:@"snapshots" isDirectory:YES];
	return [self entriesNamed:commit inDirectoryURL:snapshotsURL].firstObject;
}

/*! The folders directly inside directoryURL whose name equals name without regard to case, in ascending
	name order. A link to a folder counts as one. */
- (NSArray<NSURL *> *)entriesNamed:(NSString *)name inDirectoryURL:(NSURL *)directoryURL
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSArray<NSString *> *names = [[fileManager contentsOfDirectoryAtPath:directoryURL.path error:NULL]
								  sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray<NSURL *> *entries = [NSMutableArray array];
	for (NSString *candidate in names) {
		if ([candidate caseInsensitiveCompare:name] != NSOrderedSame) {
			continue;
		}
		NSURL *url = [directoryURL URLByAppendingPathComponent:candidate isDirectory:YES];
		BOOL isDirectory = NO;
		if ([fileManager fileExistsAtPath:url.path isDirectory:&isDirectory] && isDirectory) {
			[entries addObject:url];
		}
	}
	return entries;
}

@end
