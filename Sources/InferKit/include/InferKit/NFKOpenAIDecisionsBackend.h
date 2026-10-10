//
//  NFKOpenAIDecisionsBackend.h
//  InferKit
//

#ifndef NFKOpenAIDecisionsBackend_h
#define NFKOpenAIDecisionsBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKDecisionQuestion.h>
#import <InferKit/NFKDecisionAnswer.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKOpenAIDecisionsBackend
	@abstract   An inference backend that calls OpenAI's Decisions API.
	@discussion POST /decisions answers typed questions about shared evidence and returns no text:
				an option picked from a named set, a level on an ordered scale, or the probability
				that a statement holds, each with the probabilities behind it. It reads the request
				NFKTypeSafeBackend reads and returns the same NFKDecisionAnswers, so a feature moves
				between Jev, this service, and the on-device decision models by swapping the object.

				The request reads NFKInputState, falling back to NFKInputPrompt and then to
				NFKInputMessages, and NFKInputQuestions keyed by the caller's identifiers:

				- a string state is the input as is; a dictionary state is sent as its JSON text,
				  because the service takes a string or messages
				- an array state is taken as the service's own user messages and sent as is
				- NFKInputMessages are sent as one text, a "role: content" line per turn, because the
				  service reads user messages only
				- NFKInputImage and NFKInputImages ride as inline PNG data URLs in a user message,
				  the only image form the service reads

				Each question is sent with its identifier as its name, in sorted identifier order.
				A noul is sent as a predicate, with any "true" and "false" meanings appended to its
				instructions; a score's levels are sent as level labels. A dictionary under
				NFKInputQuestions is taken as already in the service's shape and gains the identifier
				as its name when it has none.

				The reply comes back as NFKDecisionAnswers under NFKOutputAnswers under the same keys,
				matched by name, the service's whole reply under NFKOutputStructured, and the token
				counts under NFKOutputUsage. A question the service declines comes back with refused
				set. The service takes no sampling parameter. Any other request parameter folds into
				the body under its own name (safety_identifier, for example) without overriding model,
				input, or questions. A rate limit is retried through NFKRemoteTransport.

				The key travels as a Bearer token. The model is required: gpt-6-luna is the one the
				service accepts at release. Inference blocks for the round trip, so run it off the
				main or render thread, or submit a job. isReady reports whether an endpoint and a
				model are set. Introduced in InferKit 0.4.0.
*/
@interface NFKOpenAIDecisionsBackend : NSObject <NFKInferenceBackend>

/*! The decisions endpoint. Defaults to https://api.openai.com/v1/decisions. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as a Bearer token. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body, for example gpt-6-luna. Required by the API. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Seconds before the request times out. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A backend pointed at the provider's decisions endpoint: the openai preset serves one, at its own
	base or a base it was re-pointed at; every other preset returns nil. */
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

#endif /* NFKOpenAIDecisionsBackend_h */
