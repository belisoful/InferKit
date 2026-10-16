//
//  NFKDownloadInternal.h
//  InferKit
//
//  The reporting surface NFKHFHub drives a download through. Not public.
//

#import "NFKDownload.h"
#import "NFKDownloadPlan.h"

NS_ASSUME_NONNULL_BEGIN

typedef void (^NFKDownloadCompletionHandler)(NSURL * _Nullable localURL, NSError * _Nullable error);

@interface NFKDownload ()

/*! Runs after the final progress report, on the progress queue. */
@property (nonatomic, copy, nullable) NFKDownloadCompletionHandler completionHandler;

/*! Moves a pending download to planning. */
- (void)markPlanning;
/*! Adopts the plan the download follows and declares its files. */
- (void)adoptPlan:(NFKDownloadPlan *)plan;
/*! The location reported once the download completes. */
- (void)setResultURL:(nullable NSURL *)URL;

/*! A transfer of the file at localURL begins; expected is NFKDownloadUnknownByteCount when unknown. */
- (void)fileWillStartAtLocalURL:(NSURL *)localURL path:(NSString *)path expectedByteCount:(long long)expected;
/*! The transfer of the file has written byteCount bytes of expected. */
- (void)fileAtLocalURL:(NSURL *)localURL didWriteByteCount:(long long)byteCount expectedByteCount:(long long)expected;
/*! The file is whole in the cache, byteCount bytes long; cached when no transfer was needed. */
- (void)fileDidFinishAtLocalURL:(NSURL *)localURL path:(NSString *)path byteCount:(long long)byteCount cached:(BOOL)cached;
/*! The transfer of the file stopped without the file. */
- (void)fileDidFailAtLocalURL:(NSURL *)localURL;

/*! Registers a block cancel runs; returns a token for removeCancellationHandler:. A download already
	cancelled runs the block at once. */
- (id)addCancellationHandler:(dispatch_block_t)handler;
- (void)removeCancellationHandler:(id)token;

/*! Marks a snapshot folder as one the download writes into, until it finishes. */
- (void)registerSnapshotURL:(NSURL *)snapshotURL;

/*! YES when an unfinished download in this process writes into the snapshot folder. */
+ (BOOL)isSnapshotURLInUse:(NSURL *)snapshotURL;

/*! Runs block while no download can register a snapshot. */
+ (void)performWithSnapshotRegistryLocked:(void (NS_NOESCAPE ^)(void))block;

/*! The error a cancelled download reports. */
+ (NSError *)cancellationError;

/*! An NFKDownloadErrorDomain error. */
+ (NSError *)errorWithCode:(NFKDownloadError)code reason:(NSString *)reason;

@end

@interface NFKDownloadPlanFile ()
/*! The snapshot folder the file caches in, which is its own repository's or URL's; nil for the
	plan's snapshotURL. */
@property (nonatomic, copy, nullable) NSURL *snapshotURL;
@end

@interface NFKDownloadPlan ()
/*! The snapshot folder of the plan's files; nil without files. */
@property (nonatomic, copy, nullable) NSURL *snapshotURL;
/*! The repository-relative path of the cache root of a URL plan: @url/<host>/<digest>. */
@property (nonatomic, copy, nullable) NSString *cacheRepository;
/*! YES for a plan of a whole repository, whose completion the hub records in the snapshot. */
@property (nonatomic, assign) BOOL recordsCompletion;
@end

@interface NFKDownloadStatus ()
- (instancetype)initWithPresence:(NFKDownloadPresence)presence
					   byteCount:(long long)byteCount
					presentPaths:(NSArray<NSString *> *)presentPaths
					missingPaths:(NSArray<NSString *> *)missingPaths
						localURL:(nullable NSURL *)localURL NS_DESIGNATED_INITIALIZER;
@end

@interface NFKCachedFile ()
- (instancetype)initWithURL:(NSURL *)URL
					   path:(NSString *)path
				  byteCount:(long long)byteCount
					 reason:(nullable NSString *)reason NS_DESIGNATED_INITIALIZER;
@end

@interface NFKRemovalPlan ()
- (instancetype)initWithRepository:(nullable NSString *)repository
						  revision:(nullable NSString *)revision
						 sourceURL:(nullable NSURL *)sourceURL
					  removedFiles:(NSArray<NFKCachedFile *> *)removedFiles
						 keptFiles:(NSArray<NFKCachedFile *> *)keptFiles
					freedByteCount:(long long)freedByteCount
					   snapshotURL:(nullable NSURL *)snapshotURL NS_DESIGNATED_INITIALIZER;
/*! The paths the plan was made for; nil for the whole snapshot. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *targetPaths;
/*! The snapshot's bookkeeping files the removal deletes with the last content file. */
@property (nonatomic, copy) NSArray<NSURL *> *bookkeepingURLs;
/*! The plan of each snapshot a catalog source's files span, when they span more than one; nil for a
	plan of one snapshot. The plan's own lists are the union of the parts'. */
@property (nonatomic, copy, nullable) NSArray<NFKRemovalPlan *> *parts;
@end

NS_ASSUME_NONNULL_END
