//
//  NFKHFHubInternal.h
//  InferKit
//
//  The cache helpers NFKHFHub shares with its NFKDownloads category. Not public.
//

#import "NFKHFHub.h"

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

/*! The SHA-256 of a file as lowercase hexadecimal digits; nil when it cannot be read. */
- (nullable NSString *)sha256OfFileAtURL:(NSURL *)url;

@end

NS_ASSUME_NONNULL_END
