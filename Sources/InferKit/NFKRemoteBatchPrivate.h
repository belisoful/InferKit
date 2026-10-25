//
//  NFKRemoteBatchPrivate.h
//  InferKit
//
//  The seam NFKRemoteBatch reaches a backend through: the body a run would send, and the result a run
//  would read from a reply. Private to the core.
//

#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKGeminiGenerateContentBackend.h>
#import <InferKit/NFKRemoteEmbeddingBackend.h>
#import <InferKit/NFKRemoteCompletionBackend.h>
#import <InferKit/NFKRemoteOCRBackend.h>
#import <InferKit/NFKRemoteModerationBackend.h>
#import <InferKit/NFKRemoteClassifierBackend.h>
#import <InferKit/NFKRemoteImageBackend.h>
#import <InferKit/NFKRemoteVideoBackend.h>
#import <InferKit/NFKRemoteTranscriptionBackend.h>
#import <InferKit/NFKMistralConversationsBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceRequest;
@class NFKInferenceResult;

@protocol NFKRemoteBatchableBackend <NSObject>

/*! The model the backend names, which a batch that takes one model names too. */
@property (nonatomic, readonly, nullable) NSString *batchModelName;

/*! The request a run would send, unstreamed: its URL names the route the line takes, its JSON body is
	the line's body, and its anthropic-beta header (on the Messages API) is one the batch sends. */
- (nullable NSURLRequest *)batchURLRequestForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError;

/*! The result a run would read from the reply body. */
- (nullable NFKInferenceResult *)batchResultForReply:(NSDictionary *)reply
											 request:(NFKInferenceRequest *)request
											   error:(NSError * _Nullable *)outError;

@end

@interface NFKRemoteBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKAnthropicBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteResponsesBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKGeminiGenerateContentBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteEmbeddingBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteCompletionBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteOCRBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteModerationBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteClassifierBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteImageBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteVideoBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKRemoteTranscriptionBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

@interface NFKMistralConversationsBackend (NFKRemoteBatch) <NFKRemoteBatchableBackend>
@end

NS_ASSUME_NONNULL_END
