//
//  NFKRemoteImageBackend.h
//  InferKit
//

#ifndef NFKRemoteImageBackend_h
#define NFKRemoteImageBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteImageAPIStyle
	@abstract   The wire shape of a hosted image generation service.
	@constant   NFKRemoteImageAPIStyleOpenAI POST /images/generations as JSON and /images/edits as
				multipart (image[] and a mask); the shape OpenAI and Gemini's OpenAI layer serve.
				Streamed as image_generation.partial_image events.
	@constant   NFKRemoteImageAPIStyleXAI xAI's /images/generations and /images/edits, both JSON:
				aspect_ratio and resolution rather than a size, and the source images as
				{url} objects that take data URIs.
	@constant   NFKRemoteImageAPIStyleTogether Together's /images/generations for both operations:
				width and height, steps, seed, negative_prompt, and the source as image_url and
				reference_images.
	@constant   NFKRemoteImageAPIStyleOpenRouter OpenRouter's /images for both operations: size or
				aspect_ratio and resolution, and the sources as input_references.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteImageAPIStyle) {
	NFKRemoteImageAPIStyleOpenAI = 0,
	NFKRemoteImageAPIStyleXAI,
	NFKRemoteImageAPIStyleTogether,
	NFKRemoteImageAPIStyleOpenRouter,
};

/*!
	@class      NFKRemoteImageBackend
	@abstract   An inference backend that calls a hosted image generation service.
	@discussion Two operations, chosen by the request the way NFKMLXBackend chooses: no image under
				NFKInputImage generates from the prompt; an image (with more under NFKInputImages)
				edits them; a mask under NFKInputMask inpaints its region where the service takes
				one. The contract keys map onto every style:

				- NFKInputPrompt → the prompt; NFKInputNegativePrompt → the negative prompt where the
				  service takes one
				- NFKParameterWidth and NFKParameterHeight → the size, or width and height, or a
				  ratio and resolution tier for a service that takes those
				- NFKParameterAspectRatio, NFKParameterResolution → the service's own fields
				- NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale → where the service
				  takes them
				- NFKParameterSampleCount → the number of images
				- NFKParameterOutputFormat → the encoded format asked for

				Every other parameter goes out under its own name (quality, background, style). A
				request the style cannot express fails before the call with
				kNFKError_InferenceUnsupported. Each image in the reply, inline base64 or a URL the
				backend fetches, is decoded to a 32BGRA CVPixelBuffer: the first under
				NFKOutputImage, all of them under NFKOutputImages when there are several, and the
				parsed body under NFKRemoteBackendRawKey. streams asks the service for partial
				images, each on the job's partialResult under NFKOutputImage.

				This is the synchronous shape of image generation. A service that answers with a job
				to poll is NFKAsyncGenerationBackend. runInferenceForRequest: blocks; run it off the
				render thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteImageBackend : NSObject <NFKInferenceBackend>

/*! The text-to-image endpoint, for example https://api.openai.com/v1/images/generations. */
@property (nonatomic, copy, nullable) NSURL *generationsURL;

/*! The edit endpoint, for example https://api.openai.com/v1/images/edits. Nil where the service has none. */
@property (nonatomic, copy, nullable) NSURL *editsURL;

/*! The wire shape the backend speaks. Defaults to NFKRemoteImageAPIStyleOpenAI. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NFKRemoteImageAPIStyle apiStyle;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Asks for partial images when the request is submitted as a job. The OpenAI and OpenRouter styles.
	Off by default. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) BOOL streams;

/*! The request timeout in seconds. Defaults to 180. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithGenerationsURL:(nullable NSURL *)generationsURL editsURL:(nullable NSURL *)editsURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's image endpoints in its style, or nil for a
				provider that serves none.
	@discussion openai takes the OpenAI style; gemini the OpenAI style without edits (its layer
				serves generations only); xai, together, and openrouter their own. anthropic,
				deepseek, groq, mistral, typesafe, and the local runners return nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Runs the request as a job: streamed with partial images when streams is set and the style
	streams, and otherwise the blocking call on a background queue. Introduced in InferKit 0.4.0. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The transport seam, also used to fetch an image the reply names by URL. A test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and
	returns the block that cancels the request. Introduced in InferKit 0.4.0. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteImageBackend_h */
