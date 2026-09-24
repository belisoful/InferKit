//
//  NFKRemoteOCRBackend.h
//  InferKit
//

#ifndef NFKRemoteOCRBackend_h
#define NFKRemoteOCRBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteOCRBackend
	@abstract   Reads a document or an image into markdown through a hosted OCR endpoint.
	@discussion POST /ocr, which Mistral serves. The request reads NFKInputDocument (a PDF as an
				NSURL to a local file or a hosted one, NSData, or an NFKRemoteFile uploaded to Mistral's
				Files API) or NFKInputImage. Each page comes
				back as markdown, joined under NFKOutputText with a blank line between pages; the
				parsed reply, with its pages, tables, and block layout, rides under
				NFKRemoteBackendRawKey. NFKParameterJSONSchema asks for a document annotation: the
				service fills the schema from the document, and the result comes back parsed under
				NFKOutputStructured. Images the service cut from the pages (include_image_base64)
				decode under NFKOutputImages. Every other parameter goes out under its own name
				(pages, table_format, extract_header, bbox_annotation_format). The hosted
				counterpart of NFKVisionTextBackend. runInferenceForRequest: blocks; run it off the
				render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteOCRBackend : NSObject <NFKInferenceBackend>

/*! The OCR endpoint, for example https://api.mistral.ai/v1/ocr. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name, for example mistral-ocr-latest. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 300; a long PDF takes a while. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A backend pointed at the provider's OCR endpoint: mistral serves one; every other preset
	returns nil. */
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteOCRBackend_h */
