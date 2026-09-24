//
//  NFKTypeSafeBackend.h
//  InferKit
//

#ifndef NFKTypeSafeBackend_h
#define NFKTypeSafeBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKDecisionQuestion.h>
#import <InferKit/NFKDecisionAnswer.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKTypeSafeBackend
	@abstract   An inference backend that calls TypeSafe AI's System One API, which serves Jev.
	@discussion Jev does not generate text. A request carries a state and a map of typed
				questions about it, and the reply carries a typed answer per question: an option
				picked from a named set, a level on an ordered scale, or the probability that a
				statement holds, each with the probabilities behind it. The wire shape shares
				nothing with the chat protocols, which is why this is a separate backend.

				The request reads NFKInputState (a string, or a JSON-serializable dictionary or
				array), falling back to NFKInputPrompt and then to NFKInputMessages, and
				NFKInputQuestions, a dictionary keyed by the caller's question identifiers whose
				values are NFKDecisionQuestions or their dictionaryRepresentation. The reply comes
				back as NFKDecisionAnswers under NFKOutputAnswers under the same keys, the service's
				whole reply under NFKOutputStructured, and the token count under NFKOutputUsage.
				answersForState:questions:error: is the same call without a request.

				The service takes no sampling parameter. Any other request parameter folds into the
				body under its own name, which is how a caller reaches a field the contract does not
				name. A rate limit or an overload is retried through NFKRemoteTransport.

				The key travels as a Bearer token. The model is required: the service names
				jev-latest as the alias of its current release, and NFKRemoteProvider.typeSafe lists
				the rest. Inference blocks for the round trip, so run it off the main or render
				thread, or submit a job. isReady reports whether an endpoint and a model are set.
				Introduced in InferKit 0.4.0.
*/
@interface NFKTypeSafeBackend : NSObject <NFKInferenceBackend>

/*! The systemone endpoint. Defaults to TypeSafe's own. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as a Bearer token. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body, for example jev-latest. Required by the API. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Seconds before the request times out. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

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

#endif /* NFKTypeSafeBackend_h */
