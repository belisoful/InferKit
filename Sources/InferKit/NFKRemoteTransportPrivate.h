//
//  NFKRemoteTransportPrivate.h
//  InferKit
//

#import <InferKit/NFKRemoteTransport.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*! A SHA-256 fingerprint as 64 lowercase hexadecimal digits, from 64 hexadecimal digits in either case
	with or without colons or spaces between the byte pairs, or nil when the value is not one. */
NSString * _Nullable NFKRemoteNormalizedFingerprint(NSString *value);

/*! Why a pinned session will not send to url, or nil when it will or the session is not pinned: the pin
	is not 64 hexadecimal digits, or url's scheme is not secureScheme ("https", or "wss" for a socket).
	Every task a pinned session runs is checked first, so a key never leaves over a plaintext scheme. */
NSError * _Nullable NFKRemotePinPreflightError(NSURLSession *session, NSURL * _Nullable url, NSString *secureScheme);

/*! NFKRemoteTransport's pinned session for the provider's pinnedCertificateSHA256, or nil for a
	provider without a pin. Every class that builds a client from an NFKRemoteProvider sends through
	this session when it is non-nil. */
NSURLSession * _Nullable NFKRemotePinnedSessionForProvider(NFKRemoteProvider *provider);

@class NFKInferenceResult;

/*! The NFKFinishReason a provider's stop value names: OpenAI's finish_reason, Anthropic's stop_reason,
	Gemini's finishReason, or a Responses status reason. nil for a value none of them uses. */
NSString * _Nullable NFKFinishReasonForProviderValue(id _Nullable value);

/*! result with NFKOutputFinishReason set from the provider's stop value, kept to what the result holds: a
	stop beside tool calls is NFKFinishReasonToolCalls (Gemini reports STOP for a call), and a tool stop
	without any is NFKFinishReasonStop (a forced structured-output tool). An unknown value leaves result as it
	is. */
NFKInferenceResult * _Nullable NFKResultWithFinishReason(NFKInferenceResult * _Nullable result, id _Nullable value);

/*! The contract keys OpenAI's prompt-cache fields carry: NFKParameterConversationKey and the four
	NFKParameterPromptCache keys. */
NSArray<NSString *> *NFKRemotePromptCacheParameterKeys(void);

/*! OpenAI's prompt-cache fields for a request's parameters: NFKParameterConversationKey as
	prompt_cache_key, NFKParameterPromptCacheRetention as prompt_cache_retention, and the TTL, mode, and
	prewarm as prompt_cache_options. Prewarm goes out only when responsesRoute is YES, since chat
	completions has no such field. A key that is unset, or set to a value of another type, sends nothing. */
NSDictionary<NSString *, id> *NFKRemotePromptCacheFields(NSDictionary<NSString *, id> *parameters, BOOL responsesRoute);

/*! The inverse, for a served route: the contract parameters a body's prompt-cache fields name. */
NSDictionary<NSString *, id> *NFKRemotePromptCacheParameters(NSDictionary *body);

@interface NFKRemoteTransport (NFKPrivate)

/*! One attempt, with no retry, for a caller that answers a busy server by going elsewhere. A failure
	with no response is kNFKError_RemoteUnreachable, as from sendRequest:session:response:error:. */
+ (nullable NSData *)sendOnce:(NSURLRequest *)request
					  session:(NSURLSession *)session
					 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END
