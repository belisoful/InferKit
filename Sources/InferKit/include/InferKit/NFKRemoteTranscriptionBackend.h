//
//  NFKRemoteTranscriptionBackend.h
//  InferKit
//

#ifndef NFKRemoteTranscriptionBackend_h
#define NFKRemoteTranscriptionBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKRemoteTranscriptionBackend
	@abstract   An inference backend that transcribes audio to text through an OpenAI-compatible
				audio-transcriptions endpoint (a Whisper API, or a local server that speaks the same
				protocol).
	@discussion Depends only on Foundation, so InferKit ships it. The
				request supplies audio under NFKInputAudio (an NFKAudioAsset whose fileURL is read,
				or NSData holding the encoded file). The backend uploads it as multipart/form-data
				with the model name and any request parameters folded in as form fields (language,
				prompt, response_format, temperature), then returns the transcript under
				NFKOutputText and the parsed response under NFKOutputStructured.

				One path serves a hosted API and a local server; the difference is the endpoint URL
				and the key. runInferenceForRequest: blocks until the call returns, so a caller runs
				it off the render thread. isReady reports whether an endpoint is set.
*/
@interface NFKRemoteTranscriptionBackend : NSObject <NFKInferenceBackend>

/*! The audio-transcriptions endpoint, for example a hosted API or a localhost server URL. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent as the multipart `model` field, for example a Whisper model name. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs the HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default runs the request on the session and blocks until it
				completes. A test or an alternative transport overrides this.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteTranscriptionBackend_h */
