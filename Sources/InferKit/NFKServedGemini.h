//
//  NFKServedGemini.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceServer;

/*! The path Gemini's routes live under, beside the OpenAI and Anthropic routes' /v1. */
FOUNDATION_EXPORT NSString * const NFKServedGeminiBasePath;

/*!
	@class      NFKServedGemini
	@abstract   Google's generateContent API, read and written from the server's side.
	@discussion The routes live under /v1beta/models, the shape Gemini CLI and the google-genai SDKs send
				to a base URL:
				- POST {model}:generateContent → a GenerateContentResponse.
				- POST {model}:streamGenerateContent → with alt=sse, one GenerateContentResponse per
				  server-sent event, each carrying what the reply added; without it, a JSON array of one.
				- POST {model}:countTokens → {totalTokens}, from the hosted backend's
				  tokenCountForRequest:error:.
				- GET /v1beta/models and /v1beta/models/{model} → the hosted models in Gemini's shape.

				The request method inverts what NFKGeminiGenerateContentBackend sends, so a conversation
				arrives at the hosted backend in the contract's OpenAI chat shape under NFKInputMessages:
				systemInstruction becomes a system turn, a model turn's functionCall parts its tool_calls,
				and functionResponse parts tool turns. A functionResponse without an id answers the
				earliest unanswered call of its name. Thought parts are not replayed. Media must be inline
				data: the server fetches no fileData URI.

				Declarations under tools.functionDeclarations become NFKParameterTools, their schema
				type names lowercased (OBJECT → object); search, code execution, and other tools that
				run at Google are left out. generationConfig maps maxOutputTokens, temperature, topP,
				topK, seed, and stopSequences to the core keys; a JSON responseMimeType with
				responseJsonSchema or responseSchema to NFKParameterJSONSchema; and thinkingConfig's
				thinkingBudget or thinkingLevel to NFKParameterReasoningEffort. toolConfig's mode ANY asks for
				a call among every declared function, whatever allowedFunctionNames lists, and a
				candidateCount above 1 is refused, since a reply holds one candidate.

				The reply's thought parts carry NFKOutputReasoning when thinkingConfig asks for
				includeThoughts. finishReason is MAX_TOKENS for a reply cut at its limit, SAFETY for a
				filtered one, and STOP otherwise, a call included. usageMetadata's candidatesTokenCount
				leaves out the thought tokens, which thoughtsTokenCount carries.
*/
@interface NFKServedGemini : NSObject

- (instancetype)init NS_UNAVAILABLE;

#pragma mark Routes

/*! Serves a request under NFKServedGeminiBasePath; path is the request's path without a trailing slash. */
+ (void)serveRequest:(NFKHTTPRequest *)request path:(NSString *)path response:(NFKHTTPResponse *)response
			  server:(NFKInferenceServer *)server;

#pragma mark Wire

/*! The request a generateContent body asks for, and whether its reply shows the model's thoughts. */
+ (nullable NFKInferenceRequest *)requestFromBody:(NSDictionary *)body
								  includeThoughts:(nullable BOOL *)outIncludeThoughts
											error:(NSError * _Nullable *)outError;

/*! The GenerateContentResponse answering request with result. */
+ (NSDictionary<NSString *, id> *)responseForResult:(NFKInferenceResult *)result
											request:(NFKInferenceRequest *)request
											  model:(NSString *)model
										 identifier:(NSString *)identifier
									includeThoughts:(BOOL)includeThoughts;

/*! One hosted model in Gemini's shape: {name: "models/…", baseModelId, displayName, version,
	supportedGenerationMethods}. */
+ (NSDictionary<NSString *, id> *)modelEntryForName:(NSString *)name;

/*! The error body the API returns for status: {error: {code, message, status}}, with the InferKit domain
	and code beside the message so an InferKit client reads the exact code. */
+ (NSDictionary<NSString *, id> *)errorObjectForError:(NSError *)error status:(NSInteger)status;

@end

/*!
	@class      NFKServedGeminiStream
	@abstract   Writes one reply as streamGenerateContent's server-sent events.
	@discussion Each event is a GenerateContentResponse whose parts carry what the reply added since the
				last: thought parts for the reasoning, text parts for the text. The last event adds the
				function calls, the finishReason, and usageMetadata. A partial result's text and
				reasoning are cumulative. Not thread-safe: the caller serializes the calls.
*/
@interface NFKServedGeminiStream : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithRequest:(NFKInferenceRequest *)request
						  model:(NSString *)model
					 identifier:(NSString *)identifier
				includeThoughts:(BOOL)includeThoughts
					   response:(NFKHTTPResponse *)response NS_DESIGNATED_INITIALIZER;

/*! Whether an event has gone out, after which a failure is an error event. */
@property (nonatomic, readonly) BOOL started;

/*! Sends what the partial adds to the reasoning and the text sent so far. */
- (void)appendPartial:(NFKInferenceResult *)partial;

/*! Sends the rest of the reply, its function calls, its finish reason, and its usage. */
- (void)finishWithResult:(NFKInferenceResult *)result;

/*! Sends an error event; the caller then ends the stream. */
- (void)failWithError:(NSError *)error status:(NSInteger)status;

@end

NS_ASSUME_NONNULL_END
