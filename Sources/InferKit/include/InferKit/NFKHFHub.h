//
//  NFKHFHub.h
//  InferKit
//

#ifndef NFKHFHub_h
#define NFKHFHub_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The default revision resolved when none is given: "main". */
extern NSString * const NFKHFHubDefaultRevision;

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
				caller runs it off the main thread.
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
	@discussion Defaults to the HF_TOKEN environment variable, which is
				where the tooling around Hugging Face conventionally keeps it, and is nil when that is
				unset. A public repository needs none.
*/
@property (nonatomic, copy, nullable) NSString *accessToken;

+ (instancetype)hubWithCacheDirectoryURL:(nullable NSURL *)cacheDirectoryURL;

/*!
	@method     defaultCacheDirectoryURL
	@abstract   A ready-to-use cache location under Application Support (InferKit/models), created on
				demand.
	@discussion A host that does not manage its own security-scoped cache uses this so downloads land
				in a stable, per-user location instead of failing. A sandboxed host that needs a
				user-controlled folder still supplies its own cacheDirectoryURL.
*/
+ (NSURL *)defaultCacheDirectoryURL;

/*! The remote resolve URL for a file: <endpoint>/<repo>/resolve/<revision>/<path>. */
- (nullable NSURL *)remoteURLForRepo:(NSString *)repo
						  revision:(nullable NSString *)revision
							  path:(NSString *)path;

/*! The local cache URL for a file: <cache>/<repo>/<revision>/<path>. nil without a cache folder. */
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
				folder is set or the download fails.
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
   completionHandler:(void (^)(NSURL * _Nullable localURL, NSError * _Nullable error))completionHandler;

/*!
	@method     fetchURL:toFileURL:error:
	@abstract   Downloads remoteURL to destinationURL synchronously; YES on success.
	@discussion The transport seam. The default downloads on the session and moves the result
				into place. A test or alternative transport overrides this.
*/
- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKHFHub_h */
