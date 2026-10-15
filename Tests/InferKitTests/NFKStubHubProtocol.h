//
//  NFKStubHubProtocol.h
//  InferKitTests
//
//  An in-process Hugging Face hub at http://hub.test, served through NSURLProtocol so downloads run
//  over a real NSURLSession with no network: the repository listing, resolve URLs, HEAD requests,
//  Range requests, a transfer that stalls, and one that drops part-way.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! One scripted answer. */
@interface NFKStubHubResponse : NSObject
@property (nonatomic, assign) NSInteger status;
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *headers;
@property (nonatomic, copy) NSData *body;
/*! Bytes per delivered chunk; 0 delivers the body whole. */
@property (nonatomic, assign) NSUInteger chunkSize;
/*! Delivers the first chunk and then nothing until the request is cancelled. */
@property (nonatomic, assign) BOOL stallsAfterFirstChunk;
/*! The first request delivers this many bytes and, a quarter second later, fails with a lost
	connection; -1 for none. */
@property (nonatomic, assign) NSInteger dropsAfterByteCount;
+ (instancetype)responseWithStatus:(NSInteger)status body:(nullable NSData *)body;
@end

@interface NFKStubHubProtocol : NSURLProtocol

/*! The endpoint the stub answers: http://hub.test. */
@property (class, nonatomic, readonly) NSURL *endpointURL;

/*! A session whose requests reach the stub. */
+ (NSURLSession *)session;

/*! Forgets every response and request. */
+ (void)reset;

/*! Answers every request for path (the URL's path, without the query). */
+ (void)setResponse:(NFKStubHubResponse *)response forPath:(NSString *)path;

/*! Serves a repository: its listing at /api/models/<repo>/tree/<revision> and each file at
	/<repo>/resolve/<revision>/<path> with a strong ETag and Range support. A path in lfsPaths is
	listed with an lfs object whose oid is its SHA-256. Returns the file responses by path, to script
	further. */
+ (NSDictionary<NSString *, NFKStubHubResponse *> *)serveRepo:(NSString *)repo
													 revision:(NSString *)revision
														files:(NSDictionary<NSString *, NSData *> *)files
													 lfsPaths:(NSSet<NSString *> *)lfsPaths;

/*! The requests served so far, in order. */
+ (NSArray<NSURLRequest *> *)requests;

/*! The requests served so far for path. */
+ (NSArray<NSURLRequest *> *)requestsForPath:(NSString *)path;

/*! The SHA-256 of data as lowercase hexadecimal digits. */
+ (NSString *)sha256OfData:(NSData *)data;

@end

NS_ASSUME_NONNULL_END
