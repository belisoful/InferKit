//
//  NFKGeminiEnvironmentStore.h
//  InferKit
//

#ifndef NFKGeminiEnvironmentStore_h
#define NFKGeminiEnvironmentStore_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKGeminiEnvironmentFile
	@abstract   A file or directory in a Gemini agent's sandbox.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiEnvironmentFile : NSObject

/*! The file or directory name. */
@property (nonatomic, copy, readonly) NSString *name;

/*! The path from the environment's root, for example src/main.py. */
@property (nonatomic, copy, readonly) NSString *path;

/*! Whether the entry is a directory. */
@property (nonatomic, readonly, getter=isDirectory) BOOL directory;

/*! The size in bytes; 0 for a directory. */
@property (nonatomic, readonly) unsigned long long size;

/*! The MIME type of a file, for example model/stl; nil for a directory. */
@property (nonatomic, copy, readonly, nullable) NSString *mimeType;

/*! The entry as the service sent it, with its created and modified times. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! A file from one entry of a files listing ({name, path, type, size_bytes, mime_type, …}); a missing
	name is the path's last component. */
- (instancetype)initWithEntry:(NSDictionary<NSString *, id> *)entry NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKGeminiEnvironmentStore
	@abstract   The Gemini API's agent environments: the hosted sandboxes an Antigravity agent works in,
				and the files it writes there.
	@discussion /v1beta/environments. An agent run through NFKGeminiInteractionsBackend reports its
				environment's id under NFKGeminiOutputEnvironmentIdentifier. What the agent writes
				stays in the sandbox (a script, a rendered image, an STL mesh from a CAD script) until it
				is read here:

				- filesInEnvironment:atPath:recursive:error: → the entries under a directory
				- contentsOfFileAtPath:inEnvironment:recursive:error: → a file's bytes, or a directory as
				  an uncompressed POSIX tar archive
				- writeData:toPath:inEnvironment:overwrite:extract:error: → a file placed in the sandbox
				  before or between runs; extract unpacks a tar archive into the path

				A nil or empty path is the environment's root. An environment id or a path component
				that is empty, . or .., or carries a slash fails with kNFKError_InferenceMissingInput
				before anything is sent, so a path such as a//b or / fails too.

				An environment idles after 15 minutes without use and is deleted after 7 days without
				use. Every call blocks for its round trip, so run it off the main or render thread.
				Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiEnvironmentStore : NSObject

/*! The environments collection. Defaults to https://generativelanguage.googleapis.com/v1beta/environments.
	Writes go to the same collection with upload/ before its API version, for example
	https://proxy.example/g/upload/v1beta/environments for https://proxy.example/g/v1beta/environments;
	a write to an endpoint whose parent segment is no API version (v1, v1beta) fails unsent. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as x-goog-api-key. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! Seconds before a request times out. Defaults to 120. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A store at the gemini preset's environments collection, at its own base or a base it was re-pointed
	at; every other preset returns nil. */
+ (nullable instancetype)environmentStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*! Every environment the key's project holds, each as the service describes it ({id, status, sources,
	network, …}), every page of them. */
- (nullable NSArray<NSDictionary<NSString *, id> *> *)environmentsWithError:(NSError * _Nullable *)outError;

/*! The entries under path (nil or empty for the root): its immediate children, or with recursive every
	descendant. A path that names a file lists that file alone. */
- (nullable NSArray<NFKGeminiEnvironmentFile *> *)filesInEnvironment:(NSString *)environmentIdentifier
															  atPath:(nullable NSString *)path
														   recursive:(BOOL)recursive
															   error:(NSError * _Nullable *)outError;

/*! The bytes of the file at path. A directory comes back as a tar archive of its files, and of its
	subdirectories' files too with recursive. */
- (nullable NSData *)contentsOfFileAtPath:(NSString *)path
							inEnvironment:(NSString *)environmentIdentifier
								recursive:(BOOL)recursive
									error:(NSError * _Nullable *)outError;

/*!
	@method     writeData:toPath:inEnvironment:overwrite:extract:error:
	@abstract   Writes data into the environment at path.
	@discussion Without overwrite a file that already exists fails with the service's conflict. With
				extract the data is a tar archive unpacked into the directory at path.
*/
- (nullable NSArray<NFKGeminiEnvironmentFile *> *)writeData:(NSData *)data
													 toPath:(NSString *)path
											  inEnvironment:(NSString *)environmentIdentifier
												  overwrite:(BOOL)overwrite
													extract:(BOOL)extract
													  error:(NSError * _Nullable *)outError;

/*! Ends the environment's sandbox and deletes it with its files. */
- (BOOL)deleteEnvironment:(NSString *)environmentIdentifier error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGeminiEnvironmentStore_h */
