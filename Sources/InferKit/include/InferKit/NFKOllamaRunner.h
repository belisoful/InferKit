//
//  NFKOllamaRunner.h
//  InferKit
//

#ifndef NFKOllamaRunner_h
#define NFKOllamaRunner_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKLocalModelRunner.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKOllamaRunner
	@abstract   Ollama managed through its native /api surface.
	@discussion Installed models come from /api/tags, loaded ones from /api/ps, detail from
				/api/show, the version from /api/version, and the whole set of optional actions is
				offered: /api/pull streams a download and /api/delete removes weights. Shapes were
				measured against Ollama 0.33.

				Two things the measurement showed. The installed list already carries each model's
				context length, quantization, and capabilities, so populating a picker takes one
				call rather than one per model. And a failing pull answers HTTP 200: the failure is
				an error line inside the stream, so the pull reads every line rather than trusting
				the status. Introduced in InferKit 0.3.0.
*/
@interface NFKOllamaRunner : NSObject <NFKLocalModelRunner>

/*! The request timeout in seconds for the reading calls. Defaults to 30. A pull has no timeout. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the reading calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

- (instancetype)init NS_UNAVAILABLE;

/*! A runner for the ollama preset, or a re-based copy of it. */
+ (instancetype)runnerWithProvider:(NFKRemoteProvider *)provider;

- (BOOL)isRunning;
- (nullable NSString *)versionWithError:(NSError * _Nullable *)outError;
- (nullable NSArray<NFKRemoteModel *> *)installedModelsWithError:(NSError * _Nullable *)outError;
- (nullable NSArray<NFKRemoteModel *> *)loadedModelsWithError:(NSError * _Nullable *)outError;
- (nullable NFKRemoteModel *)detailsForModel:(NSString *)identifier error:(NSError * _Nullable *)outError;
- (NFKInferenceJob *)pullModel:(NSString *)identifier;
- (BOOL)deleteModel:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs one reading request synchronously and returns the body data, or nil.
	@discussion The transport seam for every call but the pull; a test overrides it.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*!
	@method     streamRequest:lineHandler:completionHandler:cancellation:
	@abstract   Performs a request whose reply is newline-delimited JSON, one object per line.
	@discussion The transport seam for the pull. lineHandler runs once per parsed line as it
				arrives, completionHandler once when the stream ends, with the transport error if it
				ended abnormally; cancellation is handed a block that stops the stream. A test
				overrides it to feed staged lines.
*/
- (void)streamRequest:(NSURLRequest *)request
		  lineHandler:(void (^)(NSDictionary<NSString *, id> *line))lineHandler
	completionHandler:(void (^)(NSError * _Nullable error))completionHandler
		 cancellation:(void (^)(void (^cancel)(void)))cancellation;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKOllamaRunner_h */
