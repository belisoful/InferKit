//
//  NFKDownloadPlan.h
//  InferKit
//

#ifndef NFKDownloadPlan_h
#define NFKDownloadPlan_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKDownloadPlanFile
	@abstract   One file a download would fetch.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKDownloadPlanFile : NSObject <NSCopying>

/*! The path inside the repository the file comes from, or the file name for a URL. */
@property (nonatomic, readonly, copy) NSString *path;
/*! The URL the file is fetched from. */
@property (nonatomic, readonly, copy) NSURL *remoteURL;
/*! Where the file lands in the cache. */
@property (nonatomic, readonly, copy) NSURL *localURL;
/*! The file's size; NFKDownloadUnknownByteCount when the server does not report it. */
@property (nonatomic, readonly) long long byteCount;
/*! The SHA-256 of the contents as lowercase hexadecimal digits, when known: a Large File Storage
	object's ID, or the digest a catalog source lists. A download verifies it. */
@property (nonatomic, readonly, copy, nullable) NSString *sha256;
/*! YES when the file is whole in the cache: present, and of byteCount bytes when that is known. */
@property (nonatomic, readonly, getter=isCached) BOOL cached;
/*! YES when the model loads without the file. */
@property (nonatomic, readonly, getter=isOptional) BOOL optional;

- (instancetype)initWithPath:(NSString *)path
				   remoteURL:(NSURL *)remoteURL
					localURL:(NSURL *)localURL
				   byteCount:(long long)byteCount
					  sha256:(nullable NSString *)sha256
					  cached:(BOOL)cached
					optional:(BOOL)optional NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKDownloadPlan
	@abstract   What a download would fetch, and how large it is.
	@discussion NFKHFHub makes a plan from the Hugging Face API for a repository, from a HEAD request
				for a URL, or from a catalog model source. A plan reads the cache when it is made;
				NFKHFHub's downloadWithPlan: checks each file again before fetching it.
				Introduced in InferKit 0.4.0.
*/
@interface NFKDownloadPlan : NSObject <NSCopying>

/*! The Hugging Face repository; nil for a URL. */
@property (nonatomic, readonly, copy, nullable) NSString *repository;
/*! The revision the files are read at; nil for a URL. */
@property (nonatomic, readonly, copy, nullable) NSString *revision;
/*! The download URL of a URL plan; nil for a repository. */
@property (nonatomic, readonly, copy, nullable) NSURL *sourceURL;
/*! The files, in repository order. */
@property (nonatomic, readonly, copy) NSArray<NFKDownloadPlanFile *> *files;
/*! Where the files land: the snapshot folder of a repository, or the file of a single-file source.
	A catalog source's file that names its own repository or URL lands in that place's snapshot, and
	this stays the source's own. nil when the plan holds no files. */
@property (nonatomic, readonly, copy, nullable) NSURL *localURL;
/*! The bytes of every file whose size is known. */
@property (nonatomic, readonly) long long totalByteCount;
/*! The bytes of the files not yet whole in the cache, whose size is known. */
@property (nonatomic, readonly) long long remainingByteCount;
/*! YES when every file's size is known, so the two totals are exact. */
@property (nonatomic, readonly) BOOL sizesKnown;
/*! YES when every file is whole in the cache. */
@property (nonatomic, readonly, getter=isCached) BOOL cached;

- (instancetype)initWithRepository:(nullable NSString *)repository
						  revision:(nullable NSString *)revision
						 sourceURL:(nullable NSURL *)sourceURL
							 files:(NSArray<NFKDownloadPlanFile *> *)files
						  localURL:(nullable NSURL *)localURL NS_DESIGNATED_INITIALIZER;

/*! A plan with only the files at paths, in this plan's order. A path the plan does not hold is
	skipped. */
- (NFKDownloadPlan *)planRestrictedToPaths:(NSArray<NSString *> *)paths;

/*! The file at path; nil when the plan does not hold it. */
- (nullable NFKDownloadPlanFile *)fileAtPath:(NSString *)path;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@enum       NFKDownloadPresence
	@abstract   How much of a download the cache holds. Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKDownloadPresence) {
	/*! None of the files is in the cache. */
	NFKDownloadPresenceAbsent	= 0,
	/*! Some files are in the cache, or a paused download kept part of one, and a required one is
		missing or an optional one is unchecked (NFKDownloadStatus.uncheckedOptionalPaths). */
	NFKDownloadPresencePartial	= 1,
	/*! Every required file is in the cache, and every optional one is in the cache or recorded as one
		the server lacks. */
	NFKDownloadPresenceComplete	= 2,
};

/*!
	@class      NFKDownloadStatus
	@abstract   Whether a download is in the cache, and the bytes it takes there.
	@discussion Read from the cache alone, with no network. Introduced in InferKit 0.4.0.
*/
@interface NFKDownloadStatus : NSObject

@property (nonatomic, readonly) NFKDownloadPresence presence;
/*! The bytes the present files take on disk. A file linked from elsewhere counts at its target. */
@property (nonatomic, readonly) long long byteCount;
/*! The paths in the cache. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *presentPaths;
/*! The required paths not in the cache. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *missingPaths;
/*! The snapshot folder, or the file of a single-file source; nil for a source with no files. */
@property (nonatomic, readonly, copy, nullable) NSURL *localURL;
/*! The required paths a paused download kept part of, which a later download continues. Each is also
	in missingPaths. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *partialPaths;
/*! The bytes kept for partialPaths, beside byteCount: a source downloaded byteCount + partialByteCount
	of its bytes. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly) long long partialByteCount;
/*!
	@property   uncheckedOptionalPaths
	@abstract   The optional paths of a catalog source that are not in the cache and that no download
				has found missing from the server. Introduced in InferKit 0.4.0.
	@discussion A file the catalog lists after a download of the release is one: the download that
				planned the release never asked for it. Such a path makes the status partial, and a
				download of the source fetches it or records that the server lacks it. A status whose
				only gaps are these paths holds every required file, so the model builds from it.
*/
@property (nonatomic, readonly, copy) NSArray<NSString *> *uncheckedOptionalPaths;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKCachedFile
	@abstract   One cached file a removal plan removes or keeps.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKCachedFile : NSObject

/*! The file's location. */
@property (nonatomic, readonly, copy) NSURL *URL;
/*! The path relative to the cache folder. */
@property (nonatomic, readonly, copy) NSString *path;
/*! For a removed file, the bytes its removal frees. For a kept file, the bytes it takes. */
@property (nonatomic, readonly) long long byteCount;
/*! Why the file is kept, or why removing it frees nothing; nil for an ordinary removal. */
@property (nonatomic, readonly, copy, nullable) NSString *reason;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemovalPlan
	@abstract   The cached files a removal deletes, the files it keeps, and the bytes it frees.
	@discussion A removal deletes only downloaded files. It keeps:
				- a file that another catalog variant lists, when that variant's download is complete
				  in the same cache. Its reason names the model and the variant.
				- the target of a symbolic link that lies outside the cache folder.
				- the target of a symbolic link that another cached file also links to, or that is a
				  file of another snapshot.
				- a file that a symbolic link the removal leaves resolves to.
				A removed hard link whose storage another name shares frees nothing, and its reason
				says so. A catalog source whose files come from several repositories or URLs yields
				one plan over each of their snapshots, removed together, all or none; snapshotURL is
				the source's own. Introduced in InferKit 0.4.0.
*/
@interface NFKRemovalPlan : NSObject

/*! The repository; nil for a URL. */
@property (nonatomic, readonly, copy, nullable) NSString *repository;
/*! The revision; nil for a URL. */
@property (nonatomic, readonly, copy, nullable) NSString *revision;
/*! The download URL; nil for a repository. */
@property (nonatomic, readonly, copy, nullable) NSURL *sourceURL;
/*! The files the removal deletes. */
@property (nonatomic, readonly, copy) NSArray<NFKCachedFile *> *removedFiles;
/*! The files the removal keeps, each with its reason. */
@property (nonatomic, readonly, copy) NSArray<NFKCachedFile *> *keptFiles;
/*! The bytes the removal frees, the hub's bookkeeping files included. */
@property (nonatomic, readonly) long long freedByteCount;
/*! The snapshot folder the files are in; nil for a source with no files. */
@property (nonatomic, readonly, copy, nullable) NSURL *snapshotURL;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKDownloadPlan_h */
