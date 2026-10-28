//
//  NFKDownload.h
//  InferKit
//

#ifndef NFKDownload_h
#define NFKDownload_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKDownloadPlan;

/*! The error domain of downloads, download plans, and cache removal. Introduced in InferKit 0.4.0. */
extern NSErrorDomain const NFKDownloadErrorDomain;

/*! The HTTP status code of a failed request, an NSNumber in the userInfo of an
	NFKDownloadErrorDomain error. Introduced in InferKit 0.4.0. */
extern NSString * const NFKDownloadHTTPStatusCodeKey;

/*!
	@enum       NFKDownloadError
	@abstract   Error codes in NFKDownloadErrorDomain. Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKDownloadErrorDomain, NFKDownloadError) {
	/*! The download was cancelled. */
	NFKDownloadErrorCancelled					= 1,
	/*! The server refused the request for want of a credential (HTTP 401) or because the account
		behind the credential has no access (HTTP 403). A gated repository answers this way until its
		terms are accepted. */
	NFKDownloadErrorAuthenticationRequired		= 2,
	/*! The repository, revision, file, or URL does not exist (HTTP 404), or a listed file is not in
		the repository. */
	NFKDownloadErrorNotFound					= 3,
	/*! The server answered with another status outside 200–299. The status is under
		NFKDownloadHTTPStatusCodeKey. */
	NFKDownloadErrorHTTPStatus					= 4,
	/*! A file's contents do not match its SHA-256. */
	NFKDownloadErrorChecksumMismatch			= 5,
	/*! The download alone is larger than the hub's cacheSizeLimit. */
	NFKDownloadErrorExceedsCacheSizeLimit		= 6,
	/*! The volume that holds the cache has less free space than the download needs. */
	NFKDownloadErrorInsufficientStorage			= 7,
	/*! The source's release needs a converter before InferKit loads it. */
	NFKDownloadErrorRequiresConversion			= 8,
	/*! The source names no repository or URL, or its kind is not one the hub downloads. */
	NFKDownloadErrorUnsupportedSource			= 9,
	/*! A removal named files that an NFKDownload in this process is downloading, or that a download
		call or an NFKHFHubSnapshotHold keeps in use. */
	NFKDownloadErrorInUse						= 10,
	/*! The transfer ended with fewer or more bytes than the server announced. */
	NFKDownloadErrorIncomplete					= 11,
	/*! A repository, revision, path, or file name names a location outside the cache folder or its
		snapshot: it is empty or absolute, holds an empty, ".", or ".." component, or holds a
		backslash or a NUL character. The hub refuses it before it reads, writes, or removes
		anything. */
	NFKDownloadErrorUnsafePath					= 12,
	/*! A downloaded file's size differs from the size the repository listing or the model source
		gives, or its transfer streamed past its limit (NFKHFHub.maximumFileByteCount). The file is
		not kept. */
	NFKDownloadErrorSizeMismatch				= 13,
	/*! The download was paused. The file in flight keeps its bytes, which a later download of the
		same source continues. */
	NFKDownloadErrorPaused						= 14,
	/*! The source is a page that describes the release (NFKModelSourceKindPage). A person downloads its
		files from the page, which the error names under NSURLErrorKey, and passes their location to
		the build. */
	NFKDownloadErrorManualDownload				= 15,
};

/*!
	@enum       NFKDownloadState
	@abstract   Where a download is in its life. Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKDownloadState) {
	/*! Made and not started. */
	NFKDownloadStatePending		= 0,
	/*! Asking the server what the download holds. */
	NFKDownloadStatePlanning	= 1,
	/*! Transferring files. */
	NFKDownloadStateDownloading	= 2,
	/*! Every file is in the cache. */
	NFKDownloadStateCompleted	= 3,
	/*! The download stopped on an error, which error holds. */
	NFKDownloadStateFailed		= 4,
	/*! The download was cancelled. */
	NFKDownloadStateCancelled	= 5,
	/*! The download was paused: whole files stay, and the file in flight keeps its bytes. */
	NFKDownloadStatePaused		= 6,
};

/*! A byte count the server did not report: -1. Introduced in InferKit 0.4.0. */
extern const long long NFKDownloadUnknownByteCount;

/*!
	@class      NFKDownloadFileProgress
	@abstract   The progress of one file of a download, as a snapshot.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKDownloadFileProgress : NSObject

/*! The file's path inside its repository, or its file name for a URL. */
@property (nonatomic, readonly, copy) NSString *path;
/*! Where the file lands in the cache. */
@property (nonatomic, readonly, copy) NSURL *localURL;
/*! The bytes received so far. Never decreases, except for a file the plan found in the cache that is
	gone or changed when its turn comes: its count starts over from the bytes its transfer continues. */
@property (nonatomic, readonly) long long receivedByteCount;
/*! The file's size; NFKDownloadUnknownByteCount until the plan or the server reports it. */
@property (nonatomic, readonly) long long expectedByteCount;
/*! YES once the file is whole in the cache. */
@property (nonatomic, readonly, getter=isCompleted) BOOL completed;
/*! YES when the file was in the cache already, so nothing was transferred. */
@property (nonatomic, readonly, getter=isCached) BOOL cached;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKDownloadProgress
	@abstract   The progress of a whole download at one moment.
	@discussion Byte counts cover every file the download holds. A file found in the cache counts
				its bytes as received and expected, and a file whose kept partial bytes a transfer
				continues counts them as received when it starts, so a download that resumes another
				reports the share already present. bytesPerSecond and estimatedSecondsRemaining count
				only the bytes transferred. expectedByteCount can grow while the download learns of
				files, such as the shards a shard index names, and never falls below
				receivedByteCount. Introduced in InferKit 0.4.0.
*/
@interface NFKDownloadProgress : NSObject

@property (nonatomic, readonly) NFKDownloadState state;
/*! The bytes received across every file. Never decreases, except when a file the plan found in the
	cache is gone or changed when its turn comes and its count starts over. */
@property (nonatomic, readonly) long long receivedByteCount;
/*! The bytes the download transfers, as far as they are known. */
@property (nonatomic, readonly) long long expectedByteCount;
/*! receivedByteCount over expectedByteCount, from 0 to 1. 1 once the download completes. */
@property (nonatomic, readonly) double fractionCompleted;
/*! The files the download holds, including the ones already cached. */
@property (nonatomic, readonly) NSUInteger fileCount;
/*! The files that are whole in the cache. */
@property (nonatomic, readonly) NSUInteger completedFileCount;
/*! The file being transferred; nil between files. */
@property (nonatomic, readonly, strong, nullable) NFKDownloadFileProgress *currentFile;
/*! Every file, in the order the download learned of it. */
@property (nonatomic, readonly, copy) NSArray<NFKDownloadFileProgress *> *files;
/*! The transfer rate over the last few seconds, in bytes per second; 0 before the first bytes. */
@property (nonatomic, readonly) double bytesPerSecond;
/*! The seconds the rest of the transfer takes at bytesPerSecond; nil while the rate or the size is
	unknown. */
@property (nonatomic, readonly, copy, nullable) NSNumber *estimatedSecondsRemaining;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*! The block a download reports its progress to. Introduced in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKDownloadProgressHandler)(NFKDownloadProgress *progress);

/*!
	@class      NFKDownload
	@abstract   An observable, cancellable download of model files into an NFKHFHub cache.
	@discussion NFKHFHub makes one for a plan, a repository, or a catalog model source and drives it
				on a background queue. The download reports through three channels:
				- progressHandler → an NFKDownloadProgress snapshot, at most ten times a second and
				  at every change of state or file, on a private serial queue. The last call carries
				  the final state.
				- progress → an NSProgress counting bytes, with the file counts, throughput, and time
				  remaining in its userInfo. Its cancellation cancels the download. It is updated on a
				  background queue, so a key-value observer hops to the main queue itself.
				- currentProgress → the latest snapshot, read at any time from any thread.

				A download can also collect the work of code that knows nothing of it. While a
				download is current on a thread (becomeCurrent, performAsCurrent:), every
				NFKHFHub download on that thread reports its files and bytes to it and stops when it
				is cancelled. The asynchronous download factories of the companion packages capture
				the current download when called and carry it to their background queue, so one
				download observes a whole model's release. Such work brackets itself with
				beginActivity and endActivityWithError:; the download finishes when its last activity
				ends.

				Cancelling stops the transfer in flight, removes its partial file, and refuses every
				later transfer. Pausing stops the same way and keeps the partial file, which a later
				cancel removes. A file reaches
				its cache location only whole: the bytes stream into a temporary .download file
				beside it, which is moved into place at the end and removed on failure or
				cancellation. A transfer that drops is resumed with an HTTP Range request, up to three
				times, and the server's ETag guards against a file that changed in between. A partial
				file whose server named a strong ETag records it on the file, so a later download of
				the same source, in this process or another, continues it with a Range request that
				If-Range guards, and starts it over when the file changed. Introduced in InferKit
				0.4.0.
*/
@interface NFKDownload : NSObject <NSProgressReporting>

/*! A pending download that reports to progressHandler. Make it current around the work it
	observes. */
+ (instancetype)downloadWithProgressHandler:(nullable NFKDownloadProgressHandler)progressHandler;

- (instancetype)initWithProgressHandler:(nullable NFKDownloadProgressHandler)progressHandler NS_DESIGNATED_INITIALIZER;
- (instancetype)init;

/*! The block progress snapshots are delivered to. */
@property (nonatomic, readonly, copy, nullable) NFKDownloadProgressHandler progressHandler;

/*! The download's state. Key-value observable on a background queue. */
@property (nonatomic, readonly) NFKDownloadState state;

/*! YES once cancel has been called. */
@property (nonatomic, readonly, getter=isCancelled) BOOL cancelled;

/*! YES once pause has been called. */
@property (nonatomic, readonly, getter=isPaused) BOOL paused;

/*! YES once the state is completed, failed, cancelled, or paused. */
@property (nonatomic, readonly, getter=isFinished) BOOL finished;

/*! Why the download failed, was cancelled, or was paused; nil otherwise. */
@property (nonatomic, readonly, copy, nullable) NSError *error;

/*! The plan the download follows; nil until a hub-driven download has planned, and for a download
	that only collects work. */
@property (nonatomic, readonly, copy, nullable) NFKDownloadPlan *plan;

/*! The downloaded files' location once completed: the snapshot folder of a repository, or the file
	of a single-file source. nil before completion and for a source with no files. */
@property (nonatomic, readonly, copy, nullable) NSURL *localURL;

/*! The latest progress snapshot. */
@property (nonatomic, readonly) NFKDownloadProgress *currentProgress;

/*! An NSProgress over the download's bytes. */
@property (nonatomic, readonly, strong) NSProgress *progress;

/*! Cancels the download: the transfer in flight stops, its partial file is removed, and no later
	transfer starts. Files already whole stay in the cache. A paused download moves to
	NFKDownloadStateCancelled and removes the partial files its pause kept, except one that a download
	in this process continues. Does nothing once completed, failed, or cancelled. */
- (void)cancel;

/*!
	@method     pause
	@abstract   Pauses the download: the transfer in flight stops, its partial file stays, and no later
				transfer starts.
	@discussion The download finishes in NFKDownloadStatePaused with NFKDownloadErrorPaused, and whole
				files stay in the cache. A pause that arrives once every file is whole leaves the
				download to complete. A later download of the same source (downloadModelSource:,
				downloadRepo:, or a plan) continues each kept file from its last byte, so a pause
				survives the process. -[NFKHFHub downloadStatusForModelSource:] counts the kept bytes,
				and a removal plan for the source or a cancel of the download removes them. Does
				nothing once finished. The
				NSProgress's pause pauses the download. Introduced in InferKit 0.4.0.
*/
- (void)pause;

/*! Blocks until the download finishes and its final progress report and completion handler have run,
	or until the timeout passes; YES when they have. A negative timeout waits without limit. Called from
	a progress or completion handler, it does not wait and answers isFinished. */
- (BOOL)waitUntilFinishedWithTimeout:(NSTimeInterval)timeout;

#pragma mark Collecting work

/*! The download current on the calling thread; nil when none is. */
@property (class, nonatomic, readonly, nullable) NFKDownload *currentDownload NS_SWIFT_NAME(current);

/*! Makes the download current on the calling thread until resignCurrent. Calls nest. */
- (void)becomeCurrent;

/*! Ends the innermost becomeCurrent on the calling thread. */
- (void)resignCurrent;

/*! Runs block with the download current on the calling thread. */
- (void)performAsCurrent:(void (NS_NOESCAPE ^)(void))block NS_SWIFT_NAME(performAsCurrent(_:));

/*! Declares the files of plan that the work is about to fetch, so their sizes count toward
	expectedByteCount before their bytes arrive. Files the download already holds are skipped. */
- (void)addPlan:(NFKDownloadPlan *)plan NS_SWIFT_NAME(add(_:));

/*! Marks the start of work that fetches into the download. The first activity moves a pending
	download to downloading. Does nothing once finished. */
- (void)beginActivity;

/*! Marks the end of work begun with beginActivity. When the last activity ends the download
	finishes: cancelled when cancel was called, paused when pause was called and a file is not whole,
	failed with the first error an activity reported, and completed otherwise. */
- (void)endActivityWithError:(nullable NSError *)error NS_SWIFT_NAME(endActivity(error:));

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDownload_h */
