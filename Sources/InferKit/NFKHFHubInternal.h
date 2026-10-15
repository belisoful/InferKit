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

@interface NFKHFHub ()

- (NSString *)normalizedRevision:(nullable NSString *)revision;
- (NSURL *)URL:(NSURL *)base byAppendingPathString:(NSString *)pathString;
- (nullable NSURL *)snapshotURLForRepo:(NSString *)repo revision:(nullable NSString *)revision;
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

/*! The SHA-256 of a file as lowercase hexadecimal digits; nil when it cannot be read. */
- (nullable NSString *)sha256OfFileAtURL:(NSURL *)url;

@end

NS_ASSUME_NONNULL_END
