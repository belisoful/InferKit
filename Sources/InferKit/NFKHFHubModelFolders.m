//
//  NFKHFHubModelFolders.m
//  InferKit
//
//  Catalog model sources found in folders kept outside the hub cache, and the status and location the
//  folders the hub only reads answer: its model folders, then its Hugging Face caches.
//

#import "NFKHFHub.h"
#import "NFKHFHubInternal.h"
#import "NFKDownloadInternal.h"
#import "NFKModelDescriptor.h"

/*! The file that keys a model folder: the repository and the revision its files come from. */
static NSString * const NFKHFHubModelFolderKeyName = @"REVISION";
/*! The file a source that requires none of its own files is found by, as the cache locator finds it. */
static NSString * const NFKHFHubModelFolderConfigurationPath = @"config.json";

BOOL NFKHFHubIsCommitHash(NSString * _Nullable string)
{
	if (string.length != 40) {
		return NO;
	}
	NSCharacterSet *hex = [NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefABCDEF"];
	return [string rangeOfCharacterFromSet:hex.invertedSet].location == NSNotFound;
}

/*! YES for a repository name: owner/name, one slash, no whitespace, and neither part ".", "..", or empty. */
static BOOL NFKHFHubIsRepositoryName(NSString *line)
{
	if ([line rangeOfCharacterFromSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].location != NSNotFound) {
		return NO;
	}
	NSArray<NSString *> *parts = [line componentsSeparatedByString:@"/"];
	if (parts.count != 2) {
		return NO;
	}
	return NFKHFHubUnsafePathError(line, @"repository", YES) == nil;
}

/*! The revision a source pins a folder to: nil without one, or at the default branch. */
static NSString * _Nullable NFKHFHubPinnedRevision(NSString * _Nullable revision)
{
	if (revision.length == 0 || [revision isEqualToString:NFKHFHubDefaultRevision]) {
		return nil;
	}
	return revision;
}

/*! One model folder: its location and the repository and revision its REVISION file names. */
@interface NFKHFHubModelFolder : NSObject
@property (nonatomic, copy) NSURL *URL;
@property (nonatomic, copy, nullable) NSString *repository;
/*! Lowercase; nil when REVISION names none. */
@property (nonatomic, copy, nullable) NSString *revision;
@end

@implementation NFKHFHubModelFolder

/*! The folder at url, keyed by its REVISION file; nil when the file is absent or unreadable. */
+ (nullable instancetype)folderAtURL:(NSURL *)url
{
	NSURL *keyURL = [url URLByAppendingPathComponent:NFKHFHubModelFolderKeyName isDirectory:NO];
	NSString *text = [NSString stringWithContentsOfURL:keyURL encoding:NSUTF8StringEncoding error:NULL];
	if (text == nil) {
		return nil;
	}
	NFKHFHubModelFolder *folder = [[NFKHFHubModelFolder alloc] init];
	folder.URL = url;
	for (NSString *rawLine in [text componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]) {
		NSString *line = [rawLine stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
		if (folder.revision == nil && NFKHFHubIsCommitHash(line)) {
			folder.revision = line.lowercaseString;
		} else if (folder.repository == nil && NFKHFHubIsRepositoryName(line)) {
			folder.repository = line;
		}
	}
	return folder;
}

- (BOOL)matchesRepository:(NSString *)repository pinnedRevision:(nullable NSString *)pinnedRevision
{
	if (self.repository == nil || [self.repository caseInsensitiveCompare:repository] != NSOrderedSame) {
		return NO;
	}
	if (pinnedRevision == nil) {
		return YES;
	}
	return self.revision != nil && [self.revision caseInsensitiveCompare:pinnedRevision] == NSOrderedSame;
}

@end

@implementation NFKHFHubModelFolderMatch
@end

@implementation NFKHFHub (NFKModelFolders)

- (nullable NSURL *)modelFolderLocationForModelSource:(NFKModelSource *)source
{
	return [self matchForModelSource:source inModelFolders:YES inHuggingFaceCaches:NO parts:NULL].location;
}

#pragma mark Status and location

- (nullable NFKDownloadStatus *)readOnlyStatusForModelSource:(NFKModelSource *)source
{
	NSArray<NFKHFHubSourcePart *> *parts = nil;
	NFKHFHubModelFolderMatch *match = [self matchForModelSource:source inModelFolders:YES inHuggingFaceCaches:YES
														 parts:&parts];
	if (match == nil) {
		return nil;
	}
	NSMutableOrderedSet<NSString *> *present = [NSMutableOrderedSet orderedSetWithArray:match.presentPaths];
	long long bytes = match.byteCount;
	for (NFKHFHubSourcePart *part in parts) {
		if ([self part:part isTheRepositoryOfSource:source]) {
			continue;
		}
		NFKDownloadStatus *status = [self statusOfPart:part];
		if (status.missingPaths.count > 0 || status.uncheckedOptionalPaths.count > 0) {
			return nil;
		}
		[present addObjectsFromArray:status.presentPaths];
		bytes += status.byteCount;
	}
	NSURL *localURL = match.folderURL;
	if ([source.layout isEqualToString:NFKModelSourceLayoutFile] && source.files.count == 1 && parts.count == 1) {
		localURL = [match.folderURL URLByAppendingPathComponent:source.files.firstObject.path isDirectory:NO];
	}
	return [[NFKDownloadStatus alloc] initWithPresence:NFKDownloadPresenceComplete
											 byteCount:bytes
										  presentPaths:present.array
										  missingPaths:@[]
											  localURL:localURL];
}

- (nullable NSURL *)buildableReadOnlyLocationForModelSource:(NFKModelSource *)source
{
	NSArray<NFKHFHubSourcePart *> *parts = nil;
	NFKHFHubModelFolderMatch *match = [self matchForModelSource:source inModelFolders:YES inHuggingFaceCaches:YES
														 parts:&parts];
	if (match == nil) {
		return nil;
	}
	for (NFKHFHubSourcePart *part in parts) {
		if (![self part:part isTheRepositoryOfSource:source] && [self statusOfPart:part].missingPaths.count > 0) {
			return nil;
		}
	}
	return match.location;
}

#pragma mark Matching

- (nullable NFKHFHubModelFolderMatch *)matchForModelSource:(NFKModelSource *)source
											inModelFolders:(BOOL)inModelFolders
									   inHuggingFaceCaches:(BOOL)inHuggingFaceCaches
													 parts:(NSArray<NFKHFHubSourcePart *> * _Nullable * _Nullable)outParts
{
	BOOL readsModelFolders = inModelFolders && self.modelFolderURLs.count > 0;
	BOOL readsHuggingFaceCaches = inHuggingFaceCaches && self.huggingFaceCacheURLs.count > 0;
	if (!readsModelFolders && !readsHuggingFaceCaches) {
		return nil;
	}
	if (![source.kind isEqualToString:NFKModelSourceKindHuggingFace] || source.repository.length == 0) {
		return nil;
	}
	NSArray<NFKHFHubSourcePart *> *parts = [self partsOfModelSource:source];
	if (outParts != NULL) {
		*outParts = parts;
	}
	NSArray<NFKModelFile *> *ownFiles = @[];
	for (NFKHFHubSourcePart *part in parts) {
		if ([self part:part isTheRepositoryOfSource:source]) {
			ownFiles = part.files;
			break;
		}
	}
	NFKHFHubModelFolderMatch *match = readsModelFolders ? [self modelFolderMatchOfFiles:ownFiles source:source] : nil;
	if (match == nil && readsHuggingFaceCaches) {
		match = [self huggingFaceCacheMatchOfFiles:ownFiles source:source];
	}
	return match;
}

/*! The first model folder, roots in order, that holds files, the source's own. */
- (nullable NFKHFHubModelFolderMatch *)modelFolderMatchOfFiles:(nullable NSArray<NFKModelFile *> *)ownFiles
														source:(NFKModelSource *)source
{
	NSString *pinnedRevision = NFKHFHubPinnedRevision(source.revision);
	for (NSURL *root in self.modelFolderURLs) {
		BOOL scoped = [root startAccessingSecurityScopedResource];
		NFKHFHubModelFolderMatch *match = nil;
		@try {
			for (NFKHFHubModelFolder *folder in [self modelFoldersInRoot:root]) {
				if (![folder matchesRepository:source.repository pinnedRevision:pinnedRevision]) {
					continue;
				}
				match = [self matchOfFolderURL:folder.URL files:ownFiles source:source
								   ignoredPath:NFKHFHubModelFolderKeyName];
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

/*! The keyed folders directly inside root, in ascending name order. Hidden entries are skipped. */
- (NSArray<NFKHFHubModelFolder *> *)modelFoldersInRoot:(NSURL *)root
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	NSArray<NSString *> *names = [[fileManager contentsOfDirectoryAtPath:root.path error:NULL]
								  sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray<NFKHFHubModelFolder *> *folders = [NSMutableArray array];
	for (NSString *name in names) {
		if ([name hasPrefix:@"."]) {
			continue;
		}
		NSURL *url = [root URLByAppendingPathComponent:name isDirectory:YES];
		BOOL isDirectory = NO;
		if (![fileManager fileExistsAtPath:url.path isDirectory:&isDirectory] || !isDirectory) {
			continue;
		}
		NFKHFHubModelFolder *folder = [NFKHFHubModelFolder folderAtURL:url];
		if (folder != nil) {
			[folders addObject:folder];
		}
	}
	return folders;
}

- (nullable NFKHFHubModelFolderMatch *)matchOfFolderURL:(NSURL *)folderURL
												  files:(nullable NSArray<NFKModelFile *> *)files
												 source:(NFKModelSource *)source
											ignoredPath:(nullable NSString *)ignoredPath
{
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	NSMutableArray<NSString *> *optional = [NSMutableArray array];
	for (NFKModelFile *file in files) {
		if (NFKHFHubUnsafePathError(file.path, @"path", YES) != nil) {
			return nil;
		}
		[(file.optional ? optional : required) addObject:file.path];
	}
	for (NSString *shard in [self shardPathsNamedByCachedIndexes:[required arrayByAddingObjectsFromArray:optional]
												   inSnapshotURL:folderURL]) {
		[optional removeObject:shard];
		if (![required containsObject:shard]) {
			[required addObject:shard];
		}
	}
	NSArray<NSString *> *content = nil;
	NSMutableOrderedSet<NSString *> *present = [NSMutableOrderedSet orderedSet];
	for (NSString *path in required) {
		if (NFKHFHubIsGlob(path)) {
			content = content ?: [self contentPathsInModelFolderURL:folderURL ignoredPath:ignoredPath];
			NSArray<NSString *> *matches = [self paths:@[ path ] matchingContent:content];
			if (matches.count == 0) {
				return nil;
			}
			[present addObjectsFromArray:matches];
		} else if ([self modelFolderURL:folderURL holdsFileAtPath:path]) {
			[present addObject:path];
		} else {
			return nil;
		}
	}
	for (NSString *path in optional) {
		if (NFKHFHubIsGlob(path)) {
			content = content ?: [self contentPathsInModelFolderURL:folderURL ignoredPath:ignoredPath];
			[present addObjectsFromArray:[self paths:@[ path ] matchingContent:content]];
		} else if ([self modelFolderURL:folderURL holdsFileAtPath:path]) {
			[present addObject:path];
		}
	}
	if (files == nil) {
		[present addObjectsFromArray:[self contentPathsInModelFolderURL:folderURL ignoredPath:ignoredPath]];
	}
	BOOL holdsConfiguration = [self modelFolderURL:folderURL holdsFileAtPath:NFKHFHubModelFolderConfigurationPath];
	BOOL foundByConfiguration = files == nil || (required.count == 0 && present.count == 0);
	if (foundByConfiguration && !holdsConfiguration) {
		return nil;
	}
	if (foundByConfiguration) {
		[present addObject:NFKHFHubModelFolderConfigurationPath];
	}
	NSURL *location = [self locationInModelFolderURL:folderURL source:source];
	if (location == nil) {
		return nil;
	}
	long long bytes = 0;
	for (NSString *path in present) {
		bytes += [self onDiskSizeOfURL:[folderURL URLByAppendingPathComponent:path isDirectory:NO]];
	}
	NFKHFHubModelFolderMatch *match = [[NFKHFHubModelFolderMatch alloc] init];
	match.folderURL = folderURL;
	match.location = location;
	match.presentPaths = present.array;
	match.byteCount = bytes;
	return match;
}

/*! The location a build reads, as the cache locator places it: the folder for a directory layout, and for
	a file layout the first required weights file, else the first required file, else config.json; nil when
	that file is absent. */
- (nullable NSURL *)locationInModelFolderURL:(NSURL *)folderURL source:(NFKModelSource *)source
{
	if (![source.layout isEqualToString:NFKModelSourceLayoutFile]) {
		return folderURL;
	}
	NFKModelFile *file = nil;
	for (NFKModelFile *candidate in source.files) {
		if (candidate.overridesSource || candidate.isOptional) {
			continue;
		}
		if (file == nil || ([candidate.role isEqualToString:NFKModelFileRoleWeights]
							&& ![file.role isEqualToString:NFKModelFileRoleWeights])) {
			file = candidate;
		}
	}
	NSString *path = file.path ?: NFKHFHubModelFolderConfigurationPath;
	if (![self modelFolderURL:folderURL holdsFileAtPath:path]) {
		return nil;
	}
	return [folderURL URLByAppendingPathComponent:path isDirectory:NO];
}

/*! YES when the folder holds a file, or a link that resolves to one, at path. */
- (BOOL)modelFolderURL:(NSURL *)folderURL holdsFileAtPath:(NSString *)path
{
	BOOL isDirectory = NO;
	NSString *filePath = [folderURL URLByAppendingPathComponent:path isDirectory:NO].path;
	return [NSFileManager.defaultManager fileExistsAtPath:filePath isDirectory:&isDirectory] && !isDirectory;
}

/*! The files in a folder, as sorted paths relative to it: regular files and the symbolic links that resolve
	to one, less hidden entries and ignoredPath. */
- (NSArray<NSString *> *)contentPathsInModelFolderURL:(NSURL *)folderURL ignoredPath:(nullable NSString *)ignoredPath
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsRegularFileKey, NSURLIsSymbolicLinkKey ];
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:folderURL
																   includingPropertiesForKeys:keys
																					  options:NSDirectoryEnumerationSkipsHiddenFiles
																				 errorHandler:nil];
	NSMutableArray<NSString *> *paths = [NSMutableArray array];
	for (NSURL *url in enumerator) {
		NSDictionary<NSURLResourceKey, id> *values = [url resourceValuesForKeys:keys error:NULL];
		BOOL isLink = [values[NSURLIsSymbolicLinkKey] boolValue];
		if (![values[NSURLIsRegularFileKey] boolValue] && !isLink) {
			continue;
		}
		BOOL isDirectory = NO;
		if (isLink && (![NSFileManager.defaultManager fileExistsAtPath:url.path isDirectory:&isDirectory] || isDirectory)) {
			continue;
		}
		NSString *relative = [self relativePathOfURL:url underURL:folderURL];
		if (relative != nil && (ignoredPath == nil || ![relative isEqualToString:ignoredPath])) {
			[paths addObject:relative];
		}
	}
	[paths sortUsingSelector:@selector(compare:)];
	return paths;
}

@end
