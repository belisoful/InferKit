//
//  NFKHFHubTransfer.h
//  InferKit
//
//  The streaming transfer behind NFKHFHub's fetchURL:toFileURL:error:. Not public.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKDownload;

/*! The file a transfer reports to a download: the cache location the download keys it by. */
@interface NFKHFHubTransferReport : NSObject
@property (nonatomic, strong) NFKDownload *download;
@property (nonatomic, copy) NSURL *localURL;
@property (nonatomic, copy) NSString *path;
@property (nonatomic, assign) long long expectedByteCount;
@end

/*!
	@class      NFKHFHubTransfer
	@abstract   Streams one HTTP response into a file, reporting bytes and honoring cancellation.
	@discussion The bytes go straight into the destination file. A transfer that drops on a
				transient network error is resumed with an HTTP Range request, at most
				maximumResumeCount times; a strong ETag from the first response rides along as
				If-Range, so a file that changed restarts from its first byte. A failed or cancelled
				transfer removes the destination.
*/
@interface NFKHFHubTransfer : NSObject

/*! The resumptions a transfer attempts after the first request: 3. */
@property (class, nonatomic, readonly) NSUInteger maximumResumeCount;

/*! Runs the request to completion; YES when the destination holds the whole response body. */
+ (BOOL)transferRequest:(NSURLRequest *)request
		  configuration:(NSURLSessionConfiguration *)configuration
			  toFileURL:(NSURL *)destinationURL
				 report:(nullable NFKHFHubTransferReport *)report
				  error:(NSError * _Nullable *)outError;

/*! The NFKDownloadErrorDomain error an HTTP failure status maps to, naming subject. */
+ (NSError *)errorForHTTPStatus:(NSInteger)status subject:(NSString *)subject hadToken:(BOOL)hadToken;

@end

NS_ASSUME_NONNULL_END
