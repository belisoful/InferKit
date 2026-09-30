//
//  NFKServedOpenAI.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKServedOpenAI
	@abstract   The OpenAI-compatible wire, read and written from the server's side.
	@discussion Each request method inverts what the matching client in the core sends, so a request
				that crosses from NFKRemoteBackend (or the embedding, transcription, speech, or image
				backend) arrives at the hosted backend under the keys the caller set. Each reply method
				writes the envelope that client reads. Media that arrives inline is decoded into the
				contract's types; a file the server writes for it is appended to temporaryFiles, which
				the caller removes once the run ends. A remote URL in place of inline media is refused:
				the server does not fetch on a client's behalf.
*/
@interface NFKServedOpenAI : NSObject

- (instancetype)init NS_UNAVAILABLE;

#pragma mark Chat

+ (nullable NFKInferenceRequest *)chatRequestFromBody:(NSDictionary *)body
									   temporaryFiles:(NSMutableArray<NSURL *> *)temporaryFiles
												error:(NSError * _Nullable *)outError;

/*! The assistant message for a result, with the finish reason it implies. */
+ (nullable NSDictionary<NSString *, id> *)chatMessageForResult:(NFKInferenceResult *)result
												   finishReason:(NSString * _Nullable * _Nullable)outFinishReason
														  error:(NSError * _Nullable *)outError;

/*! The usage object for NFKOutputUsage, or nil when the result reports none. */
+ (nullable NSDictionary<NSString *, id> *)usageForResult:(NFKInferenceResult *)result;

/*! The text a result answers with: NFKOutputText, or NFKOutputStructured as JSON when there is no text. */
+ (nullable NSString *)textForResult:(nullable NFKInferenceResult *)result;

#pragma mark Embeddings

+ (nullable NSArray<NSString *> *)embeddingInputsFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError;
+ (NSDictionary<NSString *, id> *)embeddingParametersFromBody:(NSDictionary *)body;
+ (NSDictionary<NSString *, id> *)embeddingReplyForVectors:(NSArray<NSArray<NSNumber *> *> *)vectors
													 model:(NSString *)model
													base64:(BOOL)base64;

#pragma mark Transcription

+ (nullable NFKInferenceRequest *)transcriptionRequestFromParts:(NSArray<NFKHTTPFormPart *> *)parts
													translating:(BOOL)translating
												 temporaryFiles:(NSMutableArray<NSURL *> *)temporaryFiles
														  error:(NSError * _Nullable *)outError;

/*! The reply body in the requested response_format, and its content type. */
+ (NSData *)transcriptionBodyForResult:(NFKInferenceResult *)result
								format:(NSString *)format
						   translating:(BOOL)translating
						   contentType:(NSString * _Nonnull * _Nonnull)outContentType;

#pragma mark Speech

+ (nullable NFKInferenceRequest *)speechRequestFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError;

/*! The spoken reply's bytes in the requested format, converting the clip when its container differs. */
+ (nullable NSData *)audioDataForResult:(NFKInferenceResult *)result
								 format:(NSString *)format
							contentType:(NSString * _Nullable * _Nullable)outContentType
								  error:(NSError * _Nullable *)outError;

#pragma mark Images

+ (nullable NFKInferenceRequest *)imageGenerationRequestFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError;
+ (nullable NFKInferenceRequest *)imageEditRequestFromParts:(NSArray<NFKHTTPFormPart *> *)parts error:(NSError * _Nullable *)outError;
+ (nullable NSDictionary<NSString *, id> *)imageReplyForResult:(NFKInferenceResult *)result
											   responseFormat:(nullable NSString *)responseFormat
												 outputFormat:(nullable NSString *)outputFormat
														error:(NSError * _Nullable *)outError;

#pragma mark Helpers

/*! The form part named name (or name[]), or nil. */
+ (nullable NFKHTTPFormPart *)partNamed:(NSString *)name in:(NSArray<NFKHTTPFormPart *> *)parts;

+ (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason;

@end

NS_ASSUME_NONNULL_END
