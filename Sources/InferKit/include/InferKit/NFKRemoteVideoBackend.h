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
	@enum       NFKRemoteVideoAPIStyle
	@abstract   The wire shape of a hosted video generation service.
	@discussion Each style names the submit path, the body, the job's status values, and where the
				finished clip is fetched from. Introduced in InferKit 0.4.0.
	@constant   NFKRemoteVideoAPIStyleOpenAI OpenAI's videos API: POST /videos, poll /videos/{id},
				download /videos/{id}/content; edits and extensions take a job identifier.
	@constant   NFKRemoteVideoAPIStyleGeminiSoraCompatible Gemini's OpenAI-compatible videos path for
				Veo: a multipart submit whose Veo options are form fields, and the clip at the job's url.
	@constant   NFKRemoteVideoAPIStyleGeminiVeo Gemini's native Veo API: models/{model}:predictLongRunning
				with instances and parameters, an operation to poll, and the clip at its video uri.
	@constant   NFKRemoteVideoAPIStyleXAI xAI's video API: /videos/generations, /videos/edits, and
				/videos/extensions, a request_id to poll, and the clip at video.url.
	@constant   NFKRemoteVideoAPIStyleTogether Together's /v2/videos: width and height, keyframes and
				references under media, and the clip at outputs.video_url.
	@constant   NFKRemoteVideoAPIStyleOpenRouter OpenRouter's /videos: a job to poll and the clips at
				unsigned_urls; edits and extensions take the earlier job's identifier.
*/
typedef NS_ENUM(NSInteger, NFKRemoteVideoAPIStyle) {
	NFKRemoteVideoAPIStyleOpenAI = 0,
	NFKRemoteVideoAPIStyleGeminiSoraCompatible,
	NFKRemoteVideoAPIStyleGeminiVeo,
	NFKRemoteVideoAPIStyleXAI,
	NFKRemoteVideoAPIStyleTogether,
	NFKRemoteVideoAPIStyleOpenRouter,
};

/*!
	@class      NFKRemoteVideoBackend
	@abstract   Video generation through a hosted job-style service.
	@discussion A clip takes minutes, so the service answers a submit with a job to poll and a
				download to fetch at the end; this is NFKAsyncGenerationBackend filled in for each
				service's wire shape, chosen by apiStyle. The contract keys map onto every style:

				- NFKInputPrompt → the prompt; NFKInputNegativePrompt → the negative prompt where the
				  service takes one
				- NFKInputImage → the first frame; NFKInputLastFrame → the last frame;
				  NFKInputImages → reference images
				- NFKParameterDurationSeconds, NFKParameterWidth and NFKParameterHeight,
				  NFKParameterAspectRatio, NFKParameterResolution, NFKParameterSeed,
				  NFKParameterSteps, NFKParameterGuidanceScale, NFKParameterFramesPerSecond,
				  NFKParameterGenerateAudio, NFKParameterOutputFormat, and NFKParameterSampleCount →
				  the service's own field for each, where it has one; a size becomes a ratio and a
				  resolution tier for a service that takes those instead
				- NFKParameterVideoOperation (edit or extend) with NFKInputVideo or
				  NFKParameterSourceVideoIdentifier → the service's edit or extension request

				A contract key the service has no field for is not sent. Every other parameter goes
				out under its own name, which reaches an option the contract does not name
				(person_generation, keyframes, storage_options, provider). A request the chosen
				style cannot express fails before the submit with kNFKError_InferenceUnsupported.

				The job reports the service's progress where it reports one. The finished clip is
				downloaded to outputDirectoryURL and returned as an NFKVideoAsset under
				NFKOutputVideo, and every clip under NFKOutputVideos when there are several. The key
				goes with a download only to the service's own host.
				Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteVideoBackend : NFKAsyncGenerationBackend

/*! The wire shape the backend speaks. Defaults to NFKRemoteVideoAPIStyleOpenAI. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NFKRemoteVideoAPIStyle apiStyle;

/*! Where the clips land. Defaults to an InferKit directory under the temporary directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds for the submit, polls, and download. Defaults to 300. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend for the provider's video service in its default style, or nil for a
				provider that serves none.
	@discussion gemini → NFKRemoteVideoAPIStyleGeminiSoraCompatible; openai →
				NFKRemoteVideoAPIStyleOpenAI; xai, together, and openrouter → their own styles.
				Every other preset returns nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*!
	@method     backendForProvider:apiStyle:apiKey:modelName:
	@abstract   A backend for the provider's video service in the named style, or nil when the
				provider does not serve that style.
	@discussion gemini serves NFKRemoteVideoAPIStyleGeminiSoraCompatible and
				NFKRemoteVideoAPIStyleGeminiVeo; each other provider serves its default style only.
				Introduced in InferKit 0.4.0.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
								   apiStyle:(NFKRemoteVideoAPIStyle)apiStyle
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam for the download, which is bytes rather than JSON. A test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteVideoBackend_h */
