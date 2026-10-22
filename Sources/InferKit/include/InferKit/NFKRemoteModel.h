//
//  NFKRemoteModel.h
//  InferKit
//

#ifndef NFKRemoteModel_h
#define NFKRemoteModel_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKRemoteModel
	@abstract   One model a remote provider or a local runner serves, as its list reports it.
	@discussion The identifier is what a backend's modelName takes. The other fields are read from
				the entry where the source publishes them and are nil otherwise: the OpenAI envelope
				carries an owner and a creation time, Anthropic a display name, a creation time, and
				its input and output limits, Gemini's native list its input and output limits,
				OpenRouter a context length and its top provider's output limit, Groq a context window
				and, for one model, its output limit, LM Studio a context length, and Ollama's native
				list a size, a quantization, a context length, and the model's capabilities. The entry itself is
				kept under raw for a field this type does not normalize. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteModel : NSObject <NSCopying>

/*! The name the provider answers to, for example "llama3.2:latest" or "gpt-4o". */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! The provider's display name where it publishes one, else the identifier. */
@property (nonatomic, copy, readonly) NSString *displayName;

/*! The OpenAI envelope's owned_by, where present. */
@property (nonatomic, copy, readonly, nullable) NSString *ownedBy;

/*! When the provider says the model was created, where it says. */
@property (nonatomic, copy, readonly, nullable) NSDate *createdAt;

/*! The context window in tokens, where published (context_length, max_context_length,
	context_window, Anthropic's max_input_tokens, Gemini's inputTokenLimit, OpenRouter's top
	provider's context_length, or a local runner's details). */
@property (nonatomic, copy, readonly, nullable) NSNumber *contextLength;

/*! The most tokens one reply may run to, where published: Anthropic's max_tokens, Gemini's
	outputTokenLimit, Groq's max_completion_tokens, max_output_tokens, or OpenRouter's top provider's
	max_completion_tokens. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, readonly, nullable) NSNumber *maximumOutputTokens;

/*! The weights' size on disk in bytes, where a local runner reports it. */
@property (nonatomic, copy, readonly, nullable) NSNumber *sizeBytes;

/*! The quantization the weights are stored in ("Q4_K_M", "MXFP4", "4bit"), where reported. */
@property (nonatomic, copy, readonly, nullable) NSString *quantization;

/*! What the model can do, as a local runner names it ("completion", "tools", "vision",
	"embedding", "thinking"), where reported. The OpenAI envelope carries none. */
@property (nonatomic, copy, readonly, nullable) NSArray<NSString *> *capabilities;

/*! The provider's entry as received. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

- (instancetype)init NS_UNAVAILABLE;

/*!
	@method     modelWithEntry:
	@abstract   Reads one entry of a provider's or runner's model list.
	@discussion The identifier is the entry's id, or its model or name where the list carries no
				id (Ollama's native list). Returns nil when none of those is a non-empty string,
				which is what keeps a malformed entry from becoming a model with an empty name.
*/
+ (nullable instancetype)modelWithEntry:(NSDictionary<NSString *, id> *)entry;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteModel_h */
