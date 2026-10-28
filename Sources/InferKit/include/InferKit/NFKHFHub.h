//
//  NFKHFHub.h
//  InferKit
//

#ifndef NFKHFHub_h
#define NFKHFHub_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKDownload.h>
#import <InferKit/NFKDownloadPlan.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKModelCatalog;
@class NFKModelSource;

/*! The default revision resolved when none is given: "main". */
extern NSString * const NFKHFHubDefaultRevision;

/*! A cache size limit of -1: the cache grows without bound. Introduced in InferKit 0.4.0. */
extern const long long NFKHFHubUnlimitedCacheSize;

/*!
	@class      NFKHFHubSnapshotHold
	@abstract   Keeps one cached snapshot in use until it is invalidated.
	@discussion While a hold is valid, eviction skips the snapshot and a removal of it fails with
				NFKDownloadErrorInUse, as for a snapshot an NFKDownload writes into. A loader that
				fetches a release file by file holds the release's snapshot across its calls, so a
				trim another download runs between two files cannot evict the files fetched first.
				Holds count: a snapshot stays in use while any hold on it is valid. Deallocating a
				hold invalidates it. Introduced in InferKit 0.4.0.
*/
@interface NFKHFHubSnapshotHold : NSObject

/*! The snapshot folder the hold keeps in use. */
@property (nonatomic, readonly, copy) NSURL *snapshotURL;
/*! YES until invalidate is sent. */
@property (nonatomic, readonly, getter=isValid) BOOL valid;

/*! Ends the hold. Later calls do nothing. */
- (void)invalidate;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKHFHub
	@abstract   Resolves and downloads public Hugging Face model files into a local cache.
	@discussion The access layer, kept separate from execution: it
				knows how to turn a repo, revision, and path into a download URL, fetch the file,
				verify it, and cache it locally, and it hands the local URL to a backend. It has
				no inference knowledge.

				Weights land in cacheDirectoryURL, a folder the host supplies. In a sandboxed app
				this is a security-scoped location the user controls, which keeps multi-gigabyte
				checkpoints off the sandbox container and predictable across runs. A download is
				skipped when the file is already cached; an optional SHA-256 verifies integrity and
				forces a re-fetch on mismatch.

				A gated or private repository needs an access token. Set accessToken, or leave it nil
				and let the HF_TOKEN environment variable supply one; without either, such a
				repository returns HTTP 401.

				downloadRepo:revision:path:sha256:error: blocks until the file is ready, so a
				caller runs it off the main thread. The NFKDownloads category plans a download
				before it starts, runs it as an observable and cancellable NFKDownload, reports
				whether a download is in the cache, and removes one.

				The cache is managed in snapshots, one per repository revision
				(<cache>/<repo>/<revision>). A snapshot the hub downloads into is owned by the hub,
				which marks it with an .inferkit-owned file; only an owned snapshot is ever evicted.
				A pinned snapshot carries an .inferkit-keep file as well and is never evicted. Two
				policies apply to the cache:
				- cacheSizeLimit → after each download the least recently used snapshots are removed
				  until the cache fits. A snapshot is used when a download fetches into it or finds a
				  file already cached there.
				- excludesCacheFromBackup → the cache folder is excluded from Time Machine and iCloud
				  backup, because everything in it can be downloaded again.
				Each policy has a process-wide class default that a new hub starts from, so hubs a
				companion package creates internally follow it too.

				Every repository, revision, path, and file name names a place inside the cache. One
				that is empty or absolute, holds an empty, ".", or ".." component, or holds a
				backslash or a NUL character is refused with NFKDownloadErrorUnsafePath before the hub
				reads, writes, or removes anything; the URL methods return nil for it. Each write and
				removal also checks that its location lies inside cacheDirectoryURL, and that the
				folder it lands in does not resolve through a symbolic link to a place outside the
				cache folder's real path (followsFolderLinksOutOfCache). A revision is one folder: a
				"/" in it is stored as %2F, so refs/pr/1 caches at <cache>/<repo>/refs%2Fpr%2F1.

				Each call of downloadRepo:revision:path:sha256:error: keeps its snapshot in use while
				it runs; holdSnapshotOfRepo:revision:error: keeps one in use across calls.
*/
@interface NFKHFHub : NSObject

/*! The folder that holds the local cache. A security-scoped URL under the plugin sandbox. */
@property (nonatomic, copy, nullable) NSURL *cacheDirectoryURL;

/*! The hub base URL. Defaults to https://huggingface.co. */
@property (nonatomic, copy) NSURL *endpointURL;

/*! The session used for downloads. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*!
	@property   accessToken
	@abstract   The token a gated or private repository is fetched with, sent as a bearer credential.
	@discussion Unset, a request uses defaultAccessToken, then the HF_TOKEN environment variable,
				which is where the tooling around Hugging Face conventionally keeps it; nil when neither
				is set. A public repository needs none.
*/
@property (nonatomic, copy, nullable) NSString *accessToken;

/*!
	@property   defaultAccessToken
	@abstract   The token every hub without its own accessToken sends, process-wide. Defaults to nil.
	@discussion The way an app reaches a gated repository through a factory that makes its own hub,
				such as the InferKitMLX download-and-build factories: set it once, from the app's own
				secure storage, before the first download. An app has no HF_TOKEN environment. Read on
				each request, so setting it later applies to hubs that already exist.
				Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, copy, nullable) NSString *defaultAccessToken;

/*!
	@property   modelCatalog
	@abstract   The catalog a removal plan consults for other models that use the same files.
				Defaults to NFKModelCatalog.sharedCatalog.
	@discussion Introduced in InferKit 0.4.0.
*/
@property (nonatomic, strong, null_resettable) NFKModelCatalog *modelCatalog;

+ (instancetype)hubWithCacheDirectoryURL:(nullable NSURL *)cacheDirectoryURL;

#pragma mark Cache policy

/*!
	@property   defaultCacheSizeLimit
	@abstract   The cache size limit, in bytes, that a new hub starts with. Defaults to
				NFKHFHubUnlimitedCacheSize (-1).
	@discussion Process-wide. A hub reads it once, at initialization; changing it later leaves existing
				hubs unchanged. A negative value means no limit. Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, assign) long long defaultCacheSizeLimit;

/*!
	@property   defaultExcludesCacheFromBackup
	@abstract   Whether a new hub excludes its cache folder from backup. Defaults to YES.
	@discussion Process-wide. A hub reads it once, at initialization. Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, assign) BOOL defaultExcludesCacheFromBackup;

/*!
	@property   cacheSizeLimit
	@abstract   The most bytes the cache folder holds on disk before snapshots are evicted; negative
				for no limit.
	@discussion Starts from defaultCacheSizeLimit. Size is the space allocated on disk for every file
				under cacheDirectoryURL, including files the hub did not download. Eviction:
				- Unit → a whole <repo>/<revision> snapshot, so a model split across files is never
				  left partial.
				- Order → least recently used first.
				- Kept → the snapshot of the file just requested, even when it alone exceeds the
				  limit. The limit is a target, and the requested file is always returned.
				- Kept → a snapshot holding an in-flight .download file.
				- Kept → a pinned snapshot (pinCachedRepo:revision:error:).
				- Kept → anything the hub does not own. Only snapshots carrying the .inferkit-owned
				  marker are eligible, so a shared or user-chosen folder loses nothing else. A
				  snapshot cached before the marker existed becomes owned when next downloaded into
				  or read from the cache, or through adoptCachedRepo:revision:error:.
				A backend holding a memory-mapped file from an evicted snapshot keeps working; the
				next load downloads the file again. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) long long cacheSizeLimit;

/*!
	@property   excludesCacheFromBackup
	@abstract   Whether a download marks cacheDirectoryURL as excluded from backup. Starts from
				defaultExcludesCacheFromBackup.
	@discussion YES → the first download that finds the folder included excludes it, through
				NSURLIsExcludedFromBackupKey: the sticky exclusion `tmutil addexclusion` sets on
				macOS, and the iCloud backup exclusion on iOS and tvOS. The whole folder is
				excluded, so every snapshot under it is too. NO → the hub leaves the setting as it
				finds it; setExcludedFromBackup:forURL:error: reverses an exclusion.
				Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) BOOL excludesCacheFromBackup;

/*!
	@property   followsFolderLinksOutOfCache
	@abstract   Whether the hub writes and removes through a folder inside the cache that links to a
				place outside it. Defaults to NO.
	@discussion NO → a write or removal whose folder resolves, through a symbolic link, outside the
				real path of cacheDirectoryURL fails with NFKDownloadErrorUnsafePath. A cache folder
				that is itself a link resolves to its real path and stays usable, and so does a file
				link inside a snapshot. YES → a repository or snapshot folder inside the cache may link
				to another volume. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) BOOL followsFolderLinksOutOfCache;

/*!
	@property   maximumFileByteCount
	@abstract   The most bytes one file's transfer may stream; -1 (the default) for no maximum beyond
				the limits below.
	@discussion A transfer stops with NFKDownloadErrorSizeMismatch, and keeps nothing, once it streams
				past its limit: the file's listed size when the plan has one, else the length the
				server announces, else the free space of the cache volume. A non-negative
				maximumFileByteCount lowers every limit to itself. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) long long maximumFileByteCount;

/*!
	@method     setExcludedFromBackup:forURL:error:
	@abstract   Excludes a file or folder from backup, or includes it again; YES on success.
	@discussion Sets NSURLIsExcludedFromBackupKey. The setting travels with the item when it moves.
				Works on any location, including folders the hub does not manage. An exclusion reads
				back at once. Clearing one set moments earlier is applied asynchronously on macOS, so
				for up to a few hundred milliseconds isExcludedFromBackup: can still answer YES.
				Introduced in InferKit 0.4.0.
*/
+ (BOOL)setExcludedFromBackup:(BOOL)excluded forURL:(NSURL *)url error:(NSError * _Nullable *)outError;

/*! YES when the file or folder is excluded from backup. Introduced in InferKit 0.4.0. */
+ (BOOL)isExcludedFromBackup:(NSURL *)url;

/*!
	@method     cacheSize
	@abstract   The bytes allocated on disk for every file under cacheDirectoryURL; 0 without one.
	@discussion Walks the folder, so it costs time in proportion to the number of files.
				Introduced in InferKit 0.4.0.
*/
- (long long)cacheSize;

/*!
	@method     trimCacheToSizeLimitWithError:
	@abstract   Evicts least recently used snapshots until the cache fits cacheSizeLimit; YES on
				success.
	@discussion A download calls it after each fetch; a caller calls it after lowering the limit.
				A negative limit evicts nothing. A snapshot that an unfinished NFKDownload in this
				process writes into is not evicted, including between its files, and neither is one a
				running downloadRepo:revision:path:sha256:error: call or an NFKHFHubSnapshotHold
				keeps in use. Introduced in InferKit 0.4.0.
*/
- (BOOL)trimCacheToSizeLimitWithError:(NSError * _Nullable *)outError;

/*!
	@method     adoptCachedRepo:revision:error:
	@abstract   Marks an existing snapshot as owned by the hub, which makes it eligible for eviction;
				YES on success.
	@discussion For a snapshot cached before the hub marked ownership. A download or cache hit
				adopts a snapshot on its own; this adopts one without touching its files. The
				snapshot counts as just used. Fails when nothing is cached for the repo and revision.
				A nil revision is "main". Introduced in InferKit 0.4.0.
*/
- (BOOL)adoptCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError;

/*!
	@method     pinCachedRepo:revision:error:
	@abstract   Protects a snapshot from eviction; YES on success.
	@discussion Writes an .inferkit-keep file into the snapshot folder, creating the folder when the
				repo is not downloaded yet, so a model can be pinned before its first download.
				removeCachedRepo:revision:error: still removes a pinned snapshot. A nil revision is
				"main". Introduced in InferKit 0.4.0.
*/
- (BOOL)pinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError;

/*!
	@method     unpinCachedRepo:revision:error:
	@abstract   Makes a pinned snapshot eligible for eviction again; YES on success, including when it
				was not pinned.
	@discussion A nil revision is "main". Introduced in InferKit 0.4.0.
*/
- (BOOL)unpinCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError;

/*! YES when the snapshot is pinned. A nil revision is "main". Introduced in InferKit 0.4.0. */
- (BOOL)isCachedRepoPinned:(NSString *)repo revision:(nullable NSString *)revision;

/*!
	@method     holdSnapshotOfRepo:revision:error:
	@abstract   Keeps a snapshot in use until the returned hold is invalidated; nil with an error
				without a cache folder or for an unsafe repository or revision.
	@discussion The snapshot need not exist yet. A loader holds a release's snapshot for its whole
				load and invalidates the hold when the load ends. A nil revision is "main".
				Introduced in InferKit 0.4.0.
*/
- (nullable NFKHFHubSnapshotHold *)holdSnapshotOfRepo:(NSString *)repo
											 revision:(nullable NSString *)revision
												error:(NSError * _Nullable *)outError NS_SWIFT_NAME(holdSnapshot(repo:revision:));

/*!
	@method     removeCachedRepo:revision:error:
	@abstract   Removes one cached snapshot and any folder it leaves empty; YES on success, including
				when nothing was cached.
	@discussion Refuses with NFKDownloadErrorInUse while an NFKDownload in this process writes into
				the snapshot or a hold keeps it. A nil revision is "main". Introduced in InferKit 0.4.0.
*/
- (BOOL)removeCachedRepo:(NSString *)repo revision:(nullable NSString *)revision error:(NSError * _Nullable *)outError;

/*!
	@method     defaultCacheDirectoryURL
	@abstract   A ready-to-use cache location under Application Support (InferKit/models), created on
				demand.
	@discussion A host that does not manage its own security-scoped cache uses this so downloads land
				in a stable, per-user location instead of failing. A sandboxed host that needs a
				user-controlled folder still supplies its own cacheDirectoryURL.
*/
+ (NSURL *)defaultCacheDirectoryURL;

/*! The remote resolve URL for a file: <endpoint>/<repo>/resolve/<revision>/<path>. nil for an unsafe
	repository, revision, or path. */
- (nullable NSURL *)remoteURLForRepo:(NSString *)repo
						  revision:(nullable NSString *)revision
							  path:(NSString *)path;

/*! The local cache URL for a file: <cache>/<repo>/<revision>/<path>. nil without a cache folder, or
	for an unsafe repository, revision, or path. */
- (nullable NSURL *)localURLForRepo:(NSString *)repo
						 revision:(nullable NSString *)revision
							 path:(NSString *)path;

/*! YES when the file is already in the local cache. */
- (BOOL)isCachedRepo:(NSString *)repo revision:(nullable NSString *)revision path:(NSString *)path;

/*!
	@method     downloadRepo:revision:path:sha256:error:
	@abstract   Returns the local URL of the file, downloading it when not already cached.
	@discussion Skips the download when the file is cached and its checksum matches (or no
				checksum is given). Verifies expectedSHA256 (lowercase hex) when provided and
				fails without caching on a mismatch. Returns nil with an error when no cache
				folder is set, a name is unsafe (NFKDownloadErrorUnsafePath), or the download
				fails.
*/
- (nullable NSURL *)downloadRepo:(NSString *)repo
					  revision:(nullable NSString *)revision
						  path:(NSString *)path
						sha256:(nullable NSString *)expectedSHA256
						 error:(NSError * _Nullable *)outError;

/*!
	@method     downloadRepo:revision:path:sha256:completionHandler:
	@abstract   The asynchronous form of the download: runs on a background queue and calls the handler
				with the local URL or an error.
	@discussion Saves the caller from hand-threading the blocking download off the main thread. In Swift
				this imports as `try await hub.downloadRepo(...)`. The handler runs on the background
				queue; a caller that needs the main thread hops there itself.
*/
- (void)downloadRepo:(NSString *)repo
			revision:(nullable NSString *)revision
				path:(NSString *)path
			  sha256:(nullable NSString *)expectedSHA256
   completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSURL * _Nullable localURL, NSError * _Nullable error))completionHandler;

/*!
	@method     fetchURL:toFileURL:error:
	@abstract   Downloads remoteURL to destinationURL synchronously; YES on success.
	@discussion The transport seam. The default streams the response into destinationURL over a
				session made from session's configuration, reports the bytes to the current
				NFKDownload, stops when that download is cancelled, and resumes a dropped transfer
				with an HTTP Range request. It removes destinationURL when it fails. A test or
				alternative transport overrides this.
*/
- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError * _Nullable *)outError;

/*!
	@method     fetchMetadataFromURL:HTTPMethod:response:error:
	@abstract   Sends a metadata request (the repository listing, a HEAD request for a URL) and returns
				the body; nil with an error when the request fails.
	@discussion The transport seam of download plans. The default sends the request on session with
				the access token when the URL is on endpointURL's host, and returns the body and the
				response whatever the status. A test or alternative transport overrides this.
				Introduced in InferKit 0.4.0.
*/
- (nullable NSData *)fetchMetadataFromURL:(NSURL *)URL
							   HTTPMethod:(NSString *)method
								 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
									error:(NSError * _Nullable *)outError;

/*!
	@method     safetensorsHeaderOfRepo:revision:path:error:
	@abstract   The JSON header of a safetensors file in a repository, read without its tensors.
	@discussion Two HTTP range requests read the header: the first 8 bytes give its length, and the
				second reads it. The result maps each tensor name to {"dtype", "shape", "data_offsets"},
				and "__metadata__" to the file's string metadata when it has any. A file in the local
				cache is read from disk with no request. Blocks on the network. A nil revision is
				"main".

				Fails with NFKDownloadErrorAuthenticationRequired for a gated or private repository
				without an accepted token, NFKDownloadErrorNotFound for a missing file,
				NFKDownloadErrorHTTPStatus for another refusal, and NFKDownloadErrorUnsupportedSource
				when the bytes are not a safetensors header or the header is longer than 100,000,000
				bytes, the format's limit. A server that ignores the range request sends the whole file,
				which is read and discarded but for the header. Introduced in InferKit 0.4.0.
*/
- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfRepo:(NSString *)repo
														  revision:(nullable NSString *)revision
															  path:(NSString *)path
															 error:(NSError * _Nullable *)outError;

@end

/*!
	@category   NFKHFHub (NFKDownloads)
	@abstract   Download plans, observable downloads, cache status, and removal.
	@discussion The sources a hub plans and downloads:
				- A Hugging Face repository at a revision. Its listing comes from
				  <endpoint>/api/models/<repo>/tree/<revision>?recursive=1, which gives each file's
				  size and, for a Large File Storage file, its SHA-256. A path in a file list may be a
				  glob (fnmatch, where * also matches /).
				- A URL. Its size comes from a HEAD request. The file caches at
				  <cache>/@url/<host>/<digest>/<file name>, where digest is the first 16 hexadecimal
				  digits of the URL's SHA-256. @ cannot begin a Hugging Face name, so the folder
				  never meets a repository's.
				- A catalog NFKModelSource: kind huggingface or url as above; kind system or none
				  holds no files. A source that requires conversion is refused with
				  NFKDownloadErrorRequiresConversion.

				The access token goes only to endpointURL's host. A gated or private repository
				without an accepted token fails with NFKDownloadErrorAuthenticationRequired.

				A file whose size differs from the listed size fails with
				NFKDownloadErrorSizeMismatch, and one whose contents differ from the listed SHA-256
				with NFKDownloadErrorChecksumMismatch; neither is kept. A listing that names a file
				outside its snapshot is refused with NFKDownloadErrorUnsafePath.

				A removal deletes downloaded files only, all of them or none: it moves the files out
				of the snapshot first and deletes them once every move succeeded. It removes a
				snapshot or folder it leaves empty, keeps a pin (.inferkit-keep), and refuses with
				NFKDownloadErrorInUse while an NFKDownload in this process downloads into the
				snapshot or a download call or hold keeps it in use. It keeps a file that a link
				elsewhere in the cache resolves to, and the target of a removed link when that target
				is a file of another snapshot.
				Introduced in InferKit 0.4.0.
*/
@interface NFKHFHub (NFKDownloads)

#pragma mark Plans

/*!
	@method     planDownloadOfRepo:revision:paths:error:
	@abstract   What a download of a repository's files would fetch; nil with an error when the
				listing fails.
	@discussion paths names the files, each a path or a glob; nil plans every file. A path that is
				not a glob and that the repository does not hold fails with NFKDownloadErrorNotFound.
				Blocks on the network. A nil revision is "main".
*/
- (nullable NFKDownloadPlan *)planDownloadOfRepo:(NSString *)repo
										revision:(nullable NSString *)revision
										   paths:(nullable NSArray<NSString *> *)paths
										   error:(NSError * _Nullable *)outError;

/*! The asynchronous form of planDownloadOfRepo:revision:paths:error:. The handler runs on a
	background queue. */
- (void)planDownloadOfRepo:(NSString *)repo
				  revision:(nullable NSString *)revision
					 paths:(nullable NSArray<NSString *> *)paths
		 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKDownloadPlan * _Nullable plan, NSError * _Nullable error))completionHandler;

/*!
	@method     planDownloadOfURL:sha256:error:
	@abstract   What a download of one URL would fetch; nil with an error when the HEAD request
				fails.
	@discussion A server that refuses HEAD leaves the size unknown and the plan stands. Blocks on the
				network. A URL whose host or last path component is unsafe is refused with
				NFKDownloadErrorUnsafePath before any request.
*/
- (nullable NFKDownloadPlan *)planDownloadOfURL:(NSURL *)URL
										 sha256:(nullable NSString *)expectedSHA256
										  error:(NSError * _Nullable *)outError NS_SWIFT_NAME(planDownload(url:sha256:));

/*! The asynchronous form of planDownloadOfURL:sha256:error:. The handler runs on a background
	queue. */
- (void)planDownloadOfURL:(NSURL *)URL
				   sha256:(nullable NSString *)expectedSHA256
		completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKDownloadPlan * _Nullable plan, NSError * _Nullable error))completionHandler NS_SWIFT_NAME(planDownload(url:sha256:completionHandler:));

/*!
	@method     planDownloadOfModelSource:error:
	@abstract   What a download of a catalog model source would fetch; nil with an error when the
				source cannot be downloaded or the request fails.
	@discussion A Hugging Face source that lists files plans those files: a required one the
				repository lacks fails with NFKDownloadErrorNotFound, an optional one is left out. A
				source that lists none plans the whole repository. A listed size or SHA-256 fills
				in what the listing lacks. For a URL source the listed size takes precedence over
				the HEAD answer, and its file name must be a safe relative path. A file that names
				its own repository, revision, or URL is planned from that repository's listing or
				that URL and caches in that repository's or URL's snapshot; the plan's localURL stays
				the source's own snapshot, and its totals count every file. Every repository,
				revision, and URL is checked before the first request. Blocks on the network.
*/
- (nullable NFKDownloadPlan *)planDownloadOfModelSource:(NFKModelSource *)source
												  error:(NSError * _Nullable *)outError NS_SWIFT_NAME(planDownload(source:));

/*! The asynchronous form of planDownloadOfModelSource:error:. The handler runs on a background
	queue. */
- (void)planDownloadOfModelSource:(NFKModelSource *)source
				completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKDownloadPlan * _Nullable plan, NSError * _Nullable error))completionHandler NS_SWIFT_NAME(planDownload(source:completionHandler:));

#pragma mark Downloads

/*!
	@method     downloadWithPlan:progressHandler:completionHandler:
	@abstract   Starts downloading a plan's files into the cache and returns the running download.
	@discussion Files already whole in the cache are not fetched. Before the first transfer the
				download fails with NFKDownloadErrorExceedsCacheSizeLimit when the plan's total is
				larger than a non-negative cacheSizeLimit, and with
				NFKDownloadErrorInsufficientStorage when the remaining bytes exceed the volume's free
				space. Each file is checked against its listed size and, when it has one, its SHA-256.
				The completion handler runs on a background queue after the last progress report,
				with the download's localURL or its error.
*/
- (NFKDownload *)downloadWithPlan:(NFKDownloadPlan *)plan
				  progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
				completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSURL * _Nullable localURL, NSError * _Nullable error))completionHandler NS_SWIFT_NAME(download(plan:progressHandler:completionHandler:));

/*! Plans a download of a repository's files (planDownloadOfRepo:revision:paths:error:) and runs it
	as downloadWithPlan:progressHandler:completionHandler: does. The download's state is planning
	until the listing arrives. */
- (NFKDownload *)downloadRepo:(NSString *)repo
					 revision:(nullable NSString *)revision
						paths:(nullable NSArray<NSString *> *)paths
			  progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
			completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSURL * _Nullable localURL, NSError * _Nullable error))completionHandler;

/*!
	@method     downloadModelSource:progressHandler:completionHandler:
	@abstract   Plans and downloads a catalog model source into the cache, the one call an app makes
				for any catalog model.
	@discussion Pass a variant's source (NFKModelVariant.source). A source with no files completes at
				once with a nil localURL. A page source fails with NFKDownloadErrorManualDownload, and
				a source that requires conversion fails with NFKDownloadErrorRequiresConversion.
*/
- (NFKDownload *)downloadModelSource:(NFKModelSource *)source
					 progressHandler:(nullable NFKDownloadProgressHandler)progressHandler
				   completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSURL * _Nullable localURL, NSError * _Nullable error))completionHandler NS_SWIFT_NAME(download(source:progressHandler:completionHandler:));

/*! The cache location of a URL download, <cache>/@url/<host>/<digest>/<file name>; nil without a
	cache folder, or when the host, fileName, or the URL's last path component is unsafe. fileName
	overrides the URL's last path component. */
- (nullable NSURL *)localURLForURL:(NSURL *)URL fileName:(nullable NSString *)fileName;

#pragma mark Status

/*!
	@method     downloadStatusForRepo:revision:paths:
	@abstract   Whether a repository's files are in the cache, read with no network.
	@discussion paths names the files, each a path or a glob matched against the cache. With paths
				nil the status is complete when a whole-repository download or a recorded release
				(recordCompleteDownloadOfRepo:revision:paths:error:) left every file it named in place,
				partial when any other file is there, and absent otherwise.
*/
- (NFKDownloadStatus *)downloadStatusForRepo:(NSString *)repo
									revision:(nullable NSString *)revision
									   paths:(nullable NSArray<NSString *> *)paths NS_SWIFT_NAME(downloadStatus(repo:revision:paths:));

/*! Whether a URL's file is in the cache: complete or absent. */
- (NFKDownloadStatus *)downloadStatusForURL:(NSURL *)URL NS_SWIFT_NAME(downloadStatus(url:));

/*! Whether a catalog model source's files are in the cache. Optional files count toward byteCount
	and not toward completeness. A file that names its own repository, revision, or URL is read in
	that place's snapshot, so the source is complete only when every place holds its files. A source
	of kind system or none is complete with no bytes. */
- (NFKDownloadStatus *)downloadStatusForModelSource:(NFKModelSource *)source NS_SWIFT_NAME(downloadStatus(source:));

/*!
	@method     recordCompleteDownloadOfRepo:revision:paths:error:
	@abstract   Records that the snapshot holds a complete download made of paths; YES on success.
	@discussion For a layer that fetches a release file by file, such as the InferKitMLX release
				downloads. downloadStatusForRepo:revision:paths: with nil paths reads the record.
				Records accumulate: the snapshot is complete while every recorded path is present.
*/
- (BOOL)recordCompleteDownloadOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   paths:(NSArray<NSString *> *)paths
							   error:(NSError * _Nullable *)outError;

/*!
	@method     recordCompleteDownloadOfRepo:revision:paths:absentPaths:error:
	@abstract   Records a complete download as recordCompleteDownloadOfRepo:revision:paths:error: does,
				and the optional files the server refused; YES on success.
	@discussion A later build of the same snapshot reads absentPaths through
				recordedAbsentPathsOfRepo:revision: and does not ask for them again, so a cached
				release loads with no network. A path that a later record lists as present leaves the
				absent list. Another revision is another snapshot with its own record.
*/
- (BOOL)recordCompleteDownloadOfRepo:(NSString *)repo
							revision:(nullable NSString *)revision
							   paths:(NSArray<NSString *> *)paths
						 absentPaths:(NSArray<NSString *> *)absentPaths
							   error:(NSError * _Nullable *)outError;

/*! The optional files the snapshot's completion record lists as refused by the server; empty without
	a record. */
- (NSArray<NSString *> *)recordedAbsentPathsOfRepo:(NSString *)repo revision:(nullable NSString *)revision;

/*!
	@method     forgetRecordedAbsentPathsOfRepo:revision:error:
	@abstract   Clears the snapshot's list of refused optional files, so the next build asks the server
				for them again; YES on success.
	@discussion The refresh of a release whose repository may have gained a file. Changes nothing
				when the snapshot has no record.
*/
- (BOOL)forgetRecordedAbsentPathsOfRepo:(NSString *)repo
							   revision:(nullable NSString *)revision
								  error:(NSError * _Nullable *)outError;

#pragma mark Removal

/*!
	@method     removalPlanForRepo:revision:paths:error:
	@abstract   The cached files a removal of a repository's download deletes and keeps; nil with an
				error without a cache folder or for an unsafe repository or revision.
	@discussion paths names the files, each a path or a glob matched against the cache; nil names every
				file in the snapshot. Reads the cache only.
*/
- (nullable NFKRemovalPlan *)removalPlanForRepo:(NSString *)repo
									   revision:(nullable NSString *)revision
										  paths:(nullable NSArray<NSString *> *)paths
										  error:(NSError * _Nullable *)outError NS_SWIFT_NAME(removalPlan(repo:revision:paths:));

/*! The cached file a removal of a URL's download deletes; nil with an error without a cache folder. */
- (nullable NFKRemovalPlan *)removalPlanForURL:(NSURL *)URL error:(NSError * _Nullable *)outError NS_SWIFT_NAME(removalPlan(url:));

/*! The cached files a removal of a catalog model source's download deletes and keeps. A source that
	lists files names those; one that lists none names the whole snapshot. A file that names its own
	repository, revision, or URL is removed from that place's snapshot, and another catalog variant
	whose download is complete keeps the files it uses there. A source of kind system or none yields
	an empty plan. */
- (nullable NFKRemovalPlan *)removalPlanForModelSource:(NFKModelSource *)source
												 error:(NSError * _Nullable *)outError NS_SWIFT_NAME(removalPlan(source:));

/*!
	@method     removeFilesInPlan:error:
	@abstract   Deletes a removal plan's files and returns the removal as carried out; nil with an
				error when nothing was removed.
	@discussion Plans again first, so a file that became shared since the plan was made is kept. The
				result's freedByteCount is the bytes freed. Refuses with NFKDownloadErrorInUse while an
				NFKDownload in this process downloads into the snapshot or a download call or
				NFKHFHubSnapshotHold keeps it in use.
*/
- (nullable NFKRemovalPlan *)removeFilesInPlan:(NFKRemovalPlan *)plan error:(NSError * _Nullable *)outError;

/*! The asynchronous form of removeFilesInPlan:error:. The handler runs on a background queue. */
- (void)removeFilesInPlan:(NFKRemovalPlan *)plan
		completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKRemovalPlan * _Nullable removal, NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKHFHub_h */
