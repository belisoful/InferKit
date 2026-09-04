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
	@class      NFKRemoteImageBackend
	@abstract   An inference backend that calls OpenAI-compatible image endpoints.
	@discussion Two operations, chosen by the request the way NFKMLXBackend chooses: no image under
				NFKInputImage runs text-to-image (POST /images/generations, a JSON body), an image
				runs an edit of it (POST /images/edits, a multipart body carrying the image as PNG),
				and an image with a mask under NFKInputMask runs an inpaint of the mask's region.
				The prompt is NFKInputPrompt. NFKParameterWidth and NFKParameterHeight become the
				service's size; other parameters fold into the body by name (quality, style, n,
				seed, steps). The reply's first image, whether inline base64 or a URL the backend
				then fetches, is decoded to a 32BGRA CVPixelBuffer under NFKOutputImage, with the
				parsed body under NFKRemoteBackendRawKey.

				This is the synchronous shape of image generation. A service that answers with a
				job to poll is NFKAsyncGenerationBackend. runInferenceForRequest: blocks; run it
				off the render thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteImageBackend : NSObject <NFKInferenceBackend>

/*! The text-to-image endpoint, for example https://api.openai.com/v1/images/generations. */
@property (nonatomic, copy, nullable) NSURL *generationsURL;

/*! The edit endpoint, for example https://api.openai.com/v1/images/edits. Nil where the service has none. */
@property (nonatomic, copy, nullable) NSURL *editsURL;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 180. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithGenerationsURL:(nullable NSURL *)generationsURL editsURL:(nullable NSURL *)editsURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's image endpoints.
	@discussion Returns nil for Anthropic, which serves no image endpoint. Of the OpenAI-compatible
				presets, openai, together, xai, and openrouter were verified to serve generations at
				release, and openai and xai edits; the others, the local runners among them, fail at
				the first call with the provider's own answer, and deepseek could not be told.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam, also used to fetch an image the reply names by URL. A test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteImageBackend_h */
