//
//  NFKRemoteMediaSupport.h
//  InferKit
//
//  Private: the media a chat request carries, gathered once for whichever wire shape a backend
//  writes, and the file writing a media reply needs.
//

#ifndef NFKRemoteMediaSupport_h
#define NFKRemoteMediaSupport_h

#import <Foundation/Foundation.h>

@class NFKInferenceRequest;

NS_ASSUME_NONNULL_BEGIN

/*! Everything a request attaches beside its text, already encoded: images and sampled video
	frames as PNG in order, one audio clip, and documents. Built once and read by each backend. */
@interface NFKRemoteAttachments : NSObject

/*! NFKInputImage, then NFKInputImages, then the frames sampled from NFKInputVideo, each as PNG. */
@property (nonatomic, copy, readonly) NSArray<NSData *> *imagePNGs;

/*! The bytes of NFKInputAudio, or nil. */
@property (nonatomic, copy, readonly, nullable) NSData *audioData;

/*! The audio's container from its file extension (wav, mp3), wav for in-memory samples. */
@property (nonatomic, copy, readonly, nullable) NSString *audioFormat;

/*! The bytes of NFKInputVideo when it was kept whole rather than sampled, or nil. */
@property (nonatomic, copy, readonly, nullable) NSData *videoData;

/*! The video's container from its file extension (mp4, mov), when videoData is set. */
@property (nonatomic, copy, readonly, nullable) NSString *videoFormat;

/*! NFKInputDocument then NFKInputDocuments, each {data, filename, mediaType}: application/pdf for a
	PDF, text/plain for an NSString or a .txt/.md/.csv/.json file, whose data is its UTF-8 text. An
	NFKRemoteFile is {fileReference, filename, mediaType}, a file the service keeps, with no data. */
@property (nonatomic, copy, readonly) NSArray<NSDictionary<NSString *, id> *> *documents;

/*! Whether anything at all is attached. */
@property (nonatomic, readonly) BOOL isEmpty;

/*! Gathers the request's media, or nil with kNFKError_InferenceMissingInput naming what could not be read. */
+ (nullable instancetype)attachmentsForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError;

/*! As attachmentsForRequest:error:, keeping NFKInputVideo whole under videoData rather than sampling
	its frames when keepsVideo is set and the request names no NFKParameterVideoFrameCount. */
+ (nullable instancetype)attachmentsForRequest:(NFKInferenceRequest *)request
									keepsVideo:(BOOL)keepsVideo
										 error:(NSError * _Nullable *)outError;

@end

/*! Writes media bytes to a uniquely named file under the directory (the InferKit temporary
	directory when nil), creating it as needed, and returns the file URL or nil with an error. */
NSURL * _Nullable NFKRemoteWriteMediaFile(NSData *data, NSString *prefix, NSString *extension,
										  NSURL * _Nullable directory, NSError * _Nullable * _Nullable outError);

/*! The NFKOutputUsage dictionary for the token counts a provider reported, or nil when it reported
	none. A count the provider leaves out is left out here, so a caller reads a key that is present
	rather than trusting a zero. Introduced in InferKit 0.4.0. */
NSDictionary<NSString *, NSNumber *> * _Nullable NFKRemoteUsage(NSNumber * _Nullable inputTokens,
														 NSNumber * _Nullable cachedTokens,
														 NSNumber * _Nullable outputTokens,
														 NSNumber * _Nullable reasoningTokens);

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteMediaSupport_h */
