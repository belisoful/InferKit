//
//  NFKRemoteInferKitBackend.h
//  InferKit
//

#ifndef NFKRemoteInferKitBackend_h
#define NFKRemoteInferKitBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKServerStatus.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKRemoteInferKitBackend
	@abstract   A backend that runs a model another InferKit process serves.
	@discussion The client of NFKInferenceServer's native route (POST /v1/inferkit/run). The whole
				request crosses: every input and parameter under the key the caller set, including
				pixel buffers, masks, multi-arrays, audio and video files, and the core's value types.
				The result comes back the same way, so a hosted depth, detection, restoration, or
				transcription model answers here exactly as it answers in its own process. A file the
				result carries is written to the InferKit temporary directory.

				runInferenceForRequest: blocks for the whole run. submitInferenceJobForRequest:
				streams: the job reports the hosted job's progress and partial results as they
				arrive, and cancelling it cancels the run on the server. A failure arrives as the
				error the hosted backend raised, domain and code intact.

				modelName picks among the server's models; nil runs the server's only model.
				supportedInputKeys and supportedParameterKeys are the hosted backend's, known once
				prepareWithError: has read the server's description of the model. Until then the
				backend does not respond to either selector, which a caller reads as "not declared".
				Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteInferKitBackend : NSObject <NFKInferenceBackend>

/*! The native route: a server's base URL plus inferkit/run. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The server's key, sent as a Bearer token. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The hosted model to run; nil for the server's only model. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 600, since a hosted model runs for as long as it runs;
	for a stream it is the longest gap between events. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A backend posting to the native route under a server's base URL (for example
	http://studio.local:11480/v1). */
+ (instancetype)backendWithBaseURL:(NSURL *)baseURL;

/*! A backend posting to this endpoint. */
+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! Reads the server's description of the model: whether it is ready, and the keys its backend
	reads. Fails with the server's error when the model is not served. */
- (BOOL)prepareWithError:(NSError * _Nullable *)outError;

/*! The served model's load as the most recent run reply's X-InferKit-* headers carried it, refusals
	included, or nil before a reply has carried one. Introduced in InferKit 0.4.0. */
@property (atomic, readonly, strong, nullable) NFKServerLoad *lastReportedLoad;

/*! Reads the status route of the server this backend posts to: every model's load and, when the
	server reports host details, the machine's state. Blocks for the round trip. Introduced in
	InferKit 0.4.0. */
- (nullable NFKServerStatus *)fetchServerStatusWithError:(NSError * _Nullable *)outError;

/*! The streamed form: the job reports the hosted run's progress and partial results, and cancelling
	it closes the connection, which cancels the run on the server. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The transport seam for the blocking form; the default delegates to NFKRemoteTransport. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport. Returns
	the block that cancels the request. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteInferKitBackend_h */
