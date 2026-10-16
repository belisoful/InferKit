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
/*! The download the bytes report to; nil when the fetch runs without one. */
@property (nonatomic, strong, nullable) NFKDownload *download;
@property (nonatomic, copy) NSURL *localURL;
@property (nonatomic, copy) NSString *path;
/*! The listed size; NFKDownloadUnknownByteCount when the plan has none. */
@property (nonatomic, assign) long long expectedByteCount;
/*! The most bytes the transfer streams whatever else limits it; -1 for none. */
@property (nonatomic, assign) long long ceilingByteCount;
@end

/*!
	@class      NFKHFHubTransfer
	@abstract   Streams one HTTP response into a file, reporting bytes and honoring cancellation.
	@discussion The bytes go straight into the destination file. A transfer that drops on a
				transient network error is resumed with an HTTP Range request, at most
				maximumResumeCount times; a strong ETag from the first response rides along as
				If-Range, so a file that changed restarts from its first byte. A transfer stops with
				NFKDownloadErrorSizeMismatch once it streams past its limit: the listed size, else the
				announced length, lowered to the report's ceiling. A failed or cancelled transfer
				removes the destination.
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
