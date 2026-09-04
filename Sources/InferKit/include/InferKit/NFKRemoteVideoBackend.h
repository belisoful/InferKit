//
//  NFKRemoteVideoBackend.h
//  InferKit
//

#ifndef NFKRemoteVideoBackend_h
#define NFKRemoteVideoBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKAsyncGenerationBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteVideoBackend
	@abstract   Video generation through OpenAI's videos API, the job-style shape.
	@discussion A clip takes minutes, so the service answers a submit with a job to poll and a
				download to fetch at the end; this is NFKAsyncGenerationBackend filled in for that
				service. A prompt under NFKInputPrompt generates; an image under NFKInputImage is
				the clip's reference frame (a multipart submit); NFKParameterDurationSeconds is the
				clip's length and NFKParameterWidth and NFKParameterHeight its size; other
				parameters fold into the submit by name. The job reports the service's progress,
				and the finished clip is downloaded to outputDirectoryURL and returned as an
				NFKVideoAsset under NFKOutputVideo, which is what the on-device LTX pipeline
				answers with.

				The path was verified to exist on openai at release; no other preset serves one.
				Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteVideoBackend : NFKAsyncGenerationBackend

/*! Where the clips land. Defaults to an InferKit directory under the temporary directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds for the submit, polls, and download. Defaults to 300. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! A backend pointed at the provider's videos endpoint, or nil for Anthropic. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam for the download, which is bytes rather than JSON. A test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteVideoBackend_h */
