//
//  NFKHFHubInternal.h
//  InferKit
//
//  The cache helpers NFKHFHub shares with its NFKDownloads category. Not public.
//

#import <InferKit/NFKHFHub.h>

NS_ASSUME_NONNULL_BEGIN

/*! The snapshot marker that records the files of a complete download, a JSON array of paths. */
extern NSString * const NFKHFHubCompleteMarkerName;
/*! The bookkeeping files a snapshot can hold besides the model's files. */
extern NSArray<NSString *> *NFKHFHubBookkeepingNames(void);
/*! The marker of a pinned snapshot. */
extern NSString * const NFKHFHubPinMarkerName;
/*! An NFKDownloadErrorUnsafePath error naming kind when string cannot name a place inside the folder
	it is appended to: it is empty or absolute, holds an empty, ".", or ".." component, a backslash,
	or a NUL character, or, without separators, a /. nil when string is safe. */
extern NSError * _Nullable NFKHFHubUnsafePathError(NSString * _Nullable string, NSString *kind, BOOL allowsSeparators);

@class NFKModelFile;

/*! YES when a listed path is a glob (it holds *, ?, or [). */
extern BOOL NFKHFHubIsGlob(NSString *path);

/*! YES for a commit hash: 40 hexadecimal digits in either case. */
extern BOOL NFKHFHubIsCommitHash(NSString * _Nullable string);

/*! The Hugging Face cache root huggingface_hub reads under environment, with ~ expanded to homeDirectory:
	- HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE → that folder.
	- HF_HOME → its hub folder.
	- XDG_CACHE_HOME → its huggingface/hub folder.
	- none of them → ~/.cache/huggingface/hub.
	An empty variable counts as unset. */
extern NSURL *NFKHFHubHuggingFaceCacheURL(NSDictionary<NSString *, NSString *> *environment, NSString *homeDirectory);

/*! The files of a catalog source that come from one place: a repository at a revision, or a URL. */
@interface NFKHFHubSourcePart : NSObject
@property (nonatomic, copy, nullable) NSString *repository;
@property (nonatomic, copy, nullable) NSString *revision;
@property (nonatomic, copy, nullable) NSURL *URL;
/*! nil for a source that lists no files, which names the whole repository. */
@property (nonatomic, strong, nullable) NSMutableArray<NFKModelFile *> *files;
@end

@interface NFKHFHub ()

- (NSString *)normalizedRevision:(nullable NSString *)revision;
- (NSURL *)URL:(NSURL *)base byAppendingPathString:(NSString *)pathString;
- (nullable NSURL *)snapshotURLForRepo:(NSString *)repo revision:(nullable NSString *)revision;
/*! The snapshot folder, or nil with the refusal of an unsafe repository or revision, or with
	kNFKError_InferenceNotReady without a cache folder. */
- (nullable NSURL *)snapshotURLForRepo:(nullable NSString *)repo
							  revision:(nullable NSString *)revision
								 error:(NSError * _Nullable *)outError;
/*! The NFKDownloadErrorUnsafePath refusal of the first unsafe name, or nil. A nil or empty revision
	is "main"; a nil path is not checked. */
- (nullable NSError *)refusalForRepo:(nullable NSString *)repo
							revision:(nullable NSString *)revision
								path:(nullable NSString *)path;
/*! YES when url, with its "." and ".." components applied, lies strictly inside the cache folder as
	given or with its links resolved. Reads nothing from disk. */
- (BOOL)isURL:(NSURL *)url insideCacheURL:(NSURL *)cacheDirectoryURL;
/*! isURL:insideCacheURL:, failing with NFKDownloadErrorUnsafePath. Guards every write and removal. */
- (BOOL)requireURL:(NSURL *)url insideCacheURL:(NSURL *)cacheDirectoryURL error:(NSError * _Nullable *)outError;
- (void)markSnapshotUsedAtURL:(NSURL *)snapshotURL;
- (void)excludeCacheFromBackupIfNeeded;
- (void)performWithCacheAccess:(void (NS_NOESCAPE ^)(NSURL *cacheDirectoryURL))body;
- (long long)allocatedSizeOfDirectoryAtURL:(NSURL *)directoryURL;
- (void)removeEmptyFoldersAboveURL:(NSURL *)url inDirectoryAtURL:(NSURL *)cacheDirectoryURL;
- (BOOL)trimCacheKeepingSnapshotURL:(nullable NSURL *)keptSnapshotURL error:(NSError * _Nullable *)outError;
- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason;
- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError;

/*! A request for URL, carrying the access token when URL is on endpointURL's host. */
- (NSMutableURLRequest *)requestForURL:(NSURL *)URL;

/*!
	Returns localURL once the file is whole there, fetching remoteURL when it is not. Reports to the
	current NFKDownload. A cached file of a size other than a non-negative expectedByteCount is
	fetched again.
*/
- (nullable NSURL *)fetchFileFromURL:(NSURL *)remoteURL
							localURL:(NSURL *)localURL
						 snapshotURL:(NSURL *)snapshotURL
								path:(NSString *)path
							  sha256:(nullable NSString *)expectedSHA256
				   expectedByteCount:(long long)expectedByteCount
							   error:(NSError * _Nullable *)outError;

/*! The free bytes of the volume holding the cache; -1 when the volume does not say. */
- (long long)availableCapacityOfCache;
/*! The bytes a paused download kept in a partial .download file, which records its server's strong ETag;
	0 for none. */
- (long long)keptByteCountOfPartialURL:(NSURL *)partialURL;

/*! The SHA-256 of a file as lowercase hexadecimal digits; nil when it cannot be read. */
- (nullable NSString *)sha256OfFileAtURL:(NSURL *)url;

/*! The JSON header of the safetensors file at url, read from disk without its tensors. Fails with
	NFKDownloadErrorUnsupportedSource when the bytes are not a safetensors header, and with the file
	system's error when the file cannot be opened. subject names the file in an error. */
- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfFileAtURL:(NSURL *)url
																subject:(NSString *)subject
																  error:(NSError * _Nullable *)outError;

@end

/*! The source and status helpers NFKHFHubDownloads.m implements for the model folders. */
@interface NFKHFHub (NFKDownloadsInternal)

/*! The places a catalog source's files come from, in the order of the first file from each. */
- (NSArray<NFKHFHubSourcePart *> *)partsOfModelSource:(NFKModelSource *)source;
/*! YES when part is the source's own repository at the source's revision. */
- (BOOL)part:(NFKHFHubSourcePart *)part isTheRepositoryOfSource:(NFKModelSource *)source;
/*! Whether one part's files are in the cache, read in its own repository's or URL's snapshot. */
- (NFKDownloadStatus *)statusOfPart:(NFKHFHubSourcePart *)part;
/*! The shards the shard indexes among paths name in the folder, which a complete download holds beside
	them. */
- (NSArray<NSString *> *)shardPathsNamedByCachedIndexes:(NSArray<NSString *> *)paths
										  inSnapshotURL:(nullable NSURL *)snapshotURL;
/*! The paths of content matching each of paths (globs or literal paths), in content order. */
- (NSArray<NSString *> *)paths:(NSArray<NSString *> *)paths matchingContent:(NSArray<NSString *> *)content;
/*! The bytes a present file takes, at its target when it is a link. */
- (long long)onDiskSizeOfURL:(NSURL *)url;
/*! The status of a catalog source read in the cache alone, without the model folders. */
- (NFKDownloadStatus *)cacheStatusForModelSource:(NFKModelSource *)source;
/*! The path of url relative to base; nil when url is not under base. */
- (nullable NSString *)relativePathOfURL:(NSURL *)url underURL:(NSURL *)base;

@end

/*! A folder the hub reads and never writes that holds a source's own files: a model folder, or a snapshot
	in a Hugging Face cache. */
@interface NFKHFHubModelFolderMatch : NSObject
@property (nonatomic, copy) NSURL *folderURL;
/*! The location a build reads: the folder, or a file in it for a file layout. */
@property (nonatomic, copy) NSURL *location;
@property (nonatomic, copy) NSArray<NSString *> *presentPaths;
@property (nonatomic) long long byteCount;
@end

/*! The answers NFKHFHubModelFolders.m implements for the status and the locators, from the folders the
	hub reads and never writes: its model folders, then its Hugging Face caches. */
@interface NFKHFHub (NFKModelFoldersInternal)

/*! The status of a catalog source a model folder or a Hugging Face cache answers: complete, with the
	matching folder's files and the cached files the source takes from elsewhere; nil when nothing matches
	or a file from elsewhere is not complete in the cache. */
- (nullable NFKDownloadStatus *)readOnlyStatusForModelSource:(NFKModelSource *)source;
/*! The location a build reads from a model folder or a Hugging Face cache, when every file the source
	takes from elsewhere that it requires is in the cache; nil otherwise. */
- (nullable NSURL *)buildableReadOnlyLocationForModelSource:(NFKModelSource *)source;
/*! The first folder that holds the source's own files, with the source's parts: a model folder, then a
	snapshot in a Hugging Face cache, as each is asked for. nil for a source other than a Hugging Face one. */
- (nullable NFKHFHubModelFolderMatch *)matchForModelSource:(NFKModelSource *)source
											inModelFolders:(BOOL)inModelFolders
									   inHuggingFaceCaches:(BOOL)inHuggingFaceCaches
													 parts:(NSArray<NFKHFHubSourcePart *> * _Nullable * _Nullable)outParts;
/*! The match of a folder holding files, the source's own, with symbolic links followed; nil when a
	required file is absent. A nil files names the whole repository, which the folder holds when it holds
	its config.json. ignoredPath, when given, is left out of the folder's listing. */
- (nullable NFKHFHubModelFolderMatch *)matchOfFolderURL:(NSURL *)folderURL
												  files:(nullable NSArray<NFKModelFile *> *)files
												 source:(NFKModelSource *)source
											ignoredPath:(nullable NSString *)ignoredPath;

@end

/*! The snapshot lookup NFKHFHubHuggingFaceCache.m implements. */
@interface NFKHFHub (NFKHuggingFaceCacheInternal)

/*! The first snapshot in the Hugging Face caches, roots in order, that holds files, the source's own. */
- (nullable NFKHFHubModelFolderMatch *)huggingFaceCacheMatchOfFiles:(nullable NSArray<NFKModelFile *> *)files
															 source:(NFKModelSource *)source;

@end

NS_ASSUME_NONNULL_END
