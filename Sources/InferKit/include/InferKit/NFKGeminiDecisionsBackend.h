//
//  NFKGeminiDecisionsBackend.h
//  InferKit
//

#ifndef NFKGeminiDecisionsBackend_h
#define NFKGeminiDecisionsBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKDecisionQuestion.h>
#import <InferKit/NFKDecisionAnswer.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKGeminiDecisionsBackend
	@abstract   An inference backend that calls Gemini's Decisions API.
	@discussion POST /v1beta/decisions answers typed questions about shared evidence and returns no
				text: an option picked from a named set, a level on an ordered scale, or the
				probability that a statement holds, each with the probabilities behind it. It reads
				the request NFKTypeSafeBackend and NFKOpenAIDecisionsBackend read and returns the same
				NFKDecisionAnswers, so a feature moves between the hosted decision services and the
				on-device decision models by swapping the object.

				The request reads NFKInputState, falling back to NFKInputPrompt and then to
				NFKInputMessages, and NFKInputQuestions keyed by the caller's identifiers:

				- a string state is the input as is; a dictionary state is sent as its JSON text
				- an array state is taken as the service's own content blocks and sent as is
				- NFKInputMessages are sent as one text, a "role: content" line per turn
				- NFKInputImage, NFKInputImages, NFKInputAudio, NFKInputVideo, NFKInputDocument, and
				  NFKInputDocuments ride as content blocks after the text: inline bytes, or the URI
				  of an NFKRemoteFile the Files API keeps. A plain-text document rides as text.

				Each question is sent under its identifier as {type, instructions, criteria}. A
				choice's criteria map each option to its description, an empty string where the
				caller gave none. A score's criteria are its level descriptions, lowest first. A
				noul's criteria carry true_description and false_description, each only when given.
				A dictionary under NFKInputQuestions is taken as already in the service's shape.

				The reply comes back as NFKDecisionAnswers under NFKOutputAnswers under the same keys,
				the service's whole reply under NFKOutputStructured, and the token counts under
				NFKOutputUsage. The service reports a confidence for every type, a noul included.
				The service takes no sampling parameter. A "resolution" request parameter (low,
				medium, high, ultra_high) rides on each image and video block the backend builds. Any
				other request parameter folds into the body under its own name without overriding
				model, input, or questions. A rate limit is retried through NFKRemoteTransport.

				The service's safety filters block a whole request, and its documents describe two
				replies:

				- HTTP 400 with an error code (safety, blocklist) → the transport's error
				- HTTP 200 with prompt_feedback.block_reason and no answers → every question comes
				  back with refused set, its raw the feedback

				The key travels in the x-goog-api-key header. The model is required:
				gemini-decision-1 is the one the service accepts at release. Inference blocks for
				the round trip, so run it off the main or render thread, or submit a job. isReady
				reports whether an endpoint and a model are set. Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiDecisionsBackend : NSObject <NFKInferenceBackend>

/*! The decisions endpoint. Defaults to https://generativelanguage.googleapis.com/v1beta/decisions. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent in the x-goog-api-key header. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body, for example gemini-decision-1. Required by the API. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Seconds before the request times out. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A backend pointed at the provider's decisions endpoint: the gemini preset serves one beside its
	OpenAI layer, at its own base or a base it was re-pointed at; every other preset returns nil. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*!
	@method     answersForState:questions:error:
	@abstract   Asks the questions about the state and returns the typed answers, or nil with an error.
	@discussion The convenience over runInferenceForRequest:error: for code that holds a state and
				questions rather than a request. The answers are keyed as the questions were.
				Blocks; run it off the render thread.
*/
- (nullable NSDictionary<NSString *, NFKDecisionAnswer *> *)answersForState:(id)state
																   questions:(NSDictionary<NSString *, NFKDecisionQuestion *> *)questions
																	   error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGeminiDecisionsBackend_h */
