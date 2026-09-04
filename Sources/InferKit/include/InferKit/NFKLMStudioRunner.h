//
//  NFKLMStudioRunner.h
//  InferKit
//

#ifndef NFKLMStudioRunner_h
#define NFKLMStudioRunner_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKLocalModelRunner.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKLMStudioRunner
	@abstract   LM Studio managed through its native /api/v0 surface.
	@discussion /api/v0/models lists every downloaded model with its type (llm, vlm, embeddings),
				architecture, quantization, context length, and whether it is loaded, which is what
				the OpenAI-compatible list leaves out; loadedModelsWithError: is that list filtered
				to state "loaded", and /api/v0/models/{id} is one entry in detail. LM Studio's REST
				surface offers no download or delete, so neither optional action is adopted and a
				caller checks respondsToSelector: before offering one. Introduced in InferKit 0.3.0.
*/
@interface NFKLMStudioRunner : NSObject <NFKLocalModelRunner>

/*! The request timeout in seconds. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

- (instancetype)init NS_UNAVAILABLE;

/*! A runner for the lmstudio preset, or a re-based copy of it. */
+ (instancetype)runnerWithProvider:(NFKRemoteProvider *)provider;

- (BOOL)isRunning;
- (nullable NSArray<NFKRemoteModel *> *)installedModelsWithError:(NSError * _Nullable *)outError;
- (nullable NSArray<NFKRemoteModel *> *)loadedModelsWithError:(NSError * _Nullable *)outError;
- (nullable NFKRemoteModel *)detailsForModel:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! The transport seam; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKLMStudioRunner_h */
