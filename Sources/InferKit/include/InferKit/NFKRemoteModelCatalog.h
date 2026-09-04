//
//  NFKRemoteModelCatalog.h
//  InferKit
//

#ifndef NFKRemoteModelCatalog_h
#define NFKRemoteModelCatalog_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteModel.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKRemoteModelCatalog
	@abstract   Lists the models a remote provider serves, for a caller choosing one.
	@discussion Every preset answers GET /models with the same envelope (an array under data, each
				entry carrying an id), so one catalog serves them all; only the credential headers
				differ, and Anthropic paginates, which the catalog follows to the end. The list is
				returned as the provider orders it and is not filtered: the envelope says nothing
				about what a model does, so a hosted list includes embedding and speech models the
				chat endpoint rejects. NFKRemoteModel.raw keeps each entry for a caller that reads
				the provider's own fields.

				A local runner that is not running fails with kNFKError_RemoteUnreachable rather than
				returning an empty list, because nothing installed and nothing running are different
				answers. A server that answers with an error fails with
				kNFKError_InferenceBackendFailure carrying the status and body.

				modelsWithError: blocks; run it off the render thread, or use the completion-handler
				form, which runs the fetch at user-initiated quality of service. Introduced in
				InferKit 0.3.0.
*/
@interface NFKRemoteModelCatalog : NSObject

/*! The provider whose list this reads. */
@property (nonatomic, copy, readonly) NFKRemoteProvider *provider;

/*! The credential, sent in the provider's header style. A local server takes none. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The request timeout in seconds. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)catalogForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*! The models the provider serves, or nil with an error. Blocks. */
- (nullable NSArray<NFKRemoteModel *> *)modelsWithError:(NSError * _Nullable *)outError;

/*! The asynchronous form of modelsWithError:. The handler runs on a background queue. */
- (void)modelsWithCompletionHandler:(void (^)(NSArray<NFKRemoteModel *> * _Nullable models,
											  NSError * _Nullable error))completionHandler;

/*!
	@method     modelWithIdentifier:error:
	@abstract   One model by name, or nil with an error.
	@discussion GET /models/{id}, which every preset serves and which answers 404 for a name the
				provider does not know, so this is also the cheap way to check a name before a
				request carries it. Blocks.
*/
- (nullable NFKRemoteModel *)modelWithIdentifier:(NSString *)identifier
										   error:(NSError * _Nullable *)outError;

/*!
	@method     isReachableWithError:
	@abstract   Whether the endpoint answers at all.
	@discussion Any HTTP response counts, a rejected key included: the question is whether a server
				is there, which for a local runner is whether it is running. Returns NO with
				kNFKError_RemoteUnreachable when nothing answered. Blocks.
*/
- (BOOL)isReachableWithError:(NSError * _Nullable *)outError;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs one HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default delegates to NFKRemoteTransport; a test overrides it
				to stage a response without a network.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteModelCatalog_h */
