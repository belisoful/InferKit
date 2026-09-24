//
//  NFKRemoteBackend.h
//  InferKit
//

#ifndef NFKRemoteBackend_h
#define NFKRemoteBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*! Request input holding a plain prompt string, wrapped as one user message. */
extern NSString * const NFKRemoteBackendPromptKey;
/*! Request input holding an OpenAI messages array, used as-is when present. */
extern NSString * const NFKRemoteBackendMessagesKey;
/*! Result output holding the assistant's text. */
extern NSString * const NFKRemoteBackendTextKey;
/*! Result output holding the parsed response body; for a streamed run, the assembled assistant message. */
extern NSString * const NFKRemoteBackendRawKey;

/*!
	@enum       NFKRemoteChatDialect
	@abstract   How a chat-completions service spells the media parts it reads.
	@constant   NFKRemoteChatDialectStandard OpenAI's parts: image_url, input_audio {data, format},
				file {filename, file_data} for a PDF; a video is sampled into frames.
	@constant   NFKRemoteChatDialectMistral Mistral's: document_url for a PDF, and input_audio as a
				bare base64 string.
	@constant   NFKRemoteChatDialectOpenRouter OpenRouter's: the standard parts, with a whole video as
				video_url.
	@constant   NFKRemoteChatDialectVLLM vLLM's: the standard parts, with a whole video as video_url.
	@constant   NFKRemoteChatDialectLlamaCpp llama.cpp's: the standard parts, with a whole video as
				input_video.
	@constant   NFKRemoteChatDialectDeepSeek DeepSeek's: the standard parts, with an uploaded file as a
				flat {type: file, file_id}.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteChatDialect) {
	NFKRemoteChatDialectStandard = 0,
	NFKRemoteChatDialectMistral,
	NFKRemoteChatDialectOpenRouter,
	NFKRemoteChatDialectVLLM,
	NFKRemoteChatDialectLlamaCpp,
	NFKRemoteChatDialectDeepSeek,
};

/*!
	@class      NFKRemoteBackend
	@abstract   An inference backend that calls an OpenAI-compatible chat-completions endpoint.
	@discussion Depends only on Foundation, so InferKit ships it. One
				path serves both a local server (an OpenAI-compatible localhost server such as a
				BaseRT, mlx_lm, or Ollama endpoint) and a true remote API: the difference is only
				the endpoint URL and the key.

				A request supplies its prompt through NFKRemoteBackendMessagesKey (an OpenAI
				messages array, used as-is) or NFKRemoteBackendPromptKey (a string wrapped as
				one user message). An image under NFKInputImage, and any under NFKInputImages,
				ride on the last user turn as inline image_url content parts, which is how a
				vision model reads them. The contract's text parameters carry the endpoint's own
				spelling: NFKParameterMaxTokens becomes max_tokens, NFKParameterTopP becomes top_p,
				NFKParameterTopK becomes top_k, NFKParameterStopSequences becomes stop, and
				NFKParameterRepetitionPenalty becomes both repetition_penalty and repeat_penalty,
				the two names the servers use for it. NFKParameterTools becomes the endpoint's
				function tools and what the model called comes back under NFKOutputToolCalls, and
				NFKParameterJSONSchema becomes a json_schema response format whose parsed reply
				comes back under NFKOutputStructured. Every other parameter folds into the body
				under its own name, so a caller reaches a field the contract does not name, and one
				written in the endpoint's spelling keeps the value the caller wrote. The result exposes the assistant text under
				NFKRemoteBackendTextKey and the parsed body under NFKRemoteBackendRawKey.

				chatDialect names how the service spells its media parts. A dialect that reads a
				whole video sends NFKInputVideo as one part unless the request names
				NFKParameterVideoFrameCount, which asks for sampled frames. A request whose
				outputModality is NFKModalityImage asks for image output, and the images in the
				reply come back under NFKOutputImage (and NFKOutputImages). A reply's url_citation
				annotations come back under NFKOutputCitations, and the tools a service ran itself
				(Groq's executed_tools) under NFKOutputServerToolResults. A plain-text document
				under NFKInputDocument rides as a text part, and an NFKRemoteFile as the service's file
				reference ({type: file, file: {file_id}}; on Mistral, a document_url signed for it).

				runInferenceForRequest: blocks until the whole reply is back, so a caller runs it
				off the render thread. submitInferenceJobForRequest: streams instead: the text so
				far arrives in the job's partialResult as each token does, and cancelling the job
				closes the connection. isReady reports whether an endpoint is set.
*/
@interface NFKRemoteBackend : NSObject <NFKInferenceBackend>

/*! The chat-completions endpoint, for example a localhost server or a hosted API URL. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! How the service spells its media parts. Defaults to NFKRemoteChatDialectStandard;
	NFKRemoteProvider's backendForProvider: sets the preset's dialect. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NFKRemoteChatDialect chatDialect;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 60. For a stream it is the longest gap between tokens. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     submitInferenceJobForRequest:
	@abstract   Starts a streamed run and returns the job to follow it.
	@discussion The request is sent with stream enabled and the reply read as server-sent events;
				each text delta appends to the job's partialResult under NFKRemoteBackendTextKey,
				and a tool call's arguments assemble across deltas. The job finishes with the same
				result the blocking form returns. Cancelling the job cancels the request, so a long
				completion stops costing at the moment it is abandoned. Introduced in InferKit 0.3.0.
*/
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs the HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default runs the request on the session and blocks until
				it completes. A test or an alternative transport overrides this.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError;

/*!
	@method     streamRequest:lineHandler:completionHandler:
	@abstract   Performs a request whose reply is read line by line as it arrives.
	@discussion The transport seam for the streamed form; the default delegates to
				NFKRemoteTransport. Returns the block that cancels the request. A test overrides it
				to feed staged lines. Introduced in InferKit 0.3.0.
*/
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteBackend_h */
