//
//  NFKRemoteTransport.h
//  InferKit
//

#ifndef NFKRemoteTransport_h
#define NFKRemoteTransport_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

NS_ASSUME_NONNULL_BEGIN

/*! The Anthropic API version the remote classes send when a caller sets no other. */
extern NSString * const NFKAnthropicAPIVersion;

/*! What a streamed request ends with: the response, the body collected when the status was not a
	success (the provider's explanation), and the transport error when the connection failed. */
typedef void (^NFKRemoteStreamCompletion)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error);

/*!
	@class      NFKRemoteTransport
	@abstract   The HTTP plumbing every remote class shares.
	@discussion The remote classes each perform a request, authorize it in the provider's header
				style, and turn a failing status into an error; this class holds those steps once,
				for blocking requests and for streamed ones. Each remote class keeps its own
				overridable seam that delegates here by default, so a test stubs one class without
				touching the others. Introduced in InferKit 0.3.0.

				A blocking request is retried on a rate limit or a gateway error (429, 502, 503,
				504): retryAttempts more tries, each after the Retry-After the provider names or an
				exponential delay from half a second, and never after a delay above
				maximumRetryDelay, where waiting would cost more than failing. A refused connection
				is not retried: a server that is not there is an answer in itself.
*/
@interface NFKRemoteTransport : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! Further attempts after a rate limit or gateway error. Defaults to 2, so three tries in all. */
@property (class, nonatomic, assign) NSUInteger retryAttempts;

/*! The longest wait before a retry, in seconds. Defaults to 8. A longer Retry-After ends the retries. */
@property (class, nonatomic, assign) NSTimeInterval maximumRetryDelay;

/*!
	@method     sendRequest:session:response:error:
	@abstract   Performs the request on the session and blocks until it completes, retrying as above.
	@discussion Returns the body data, or nil with an error. A failure that produced no response at all
				(the host is down, the connection was refused) is reported as
				kNFKError_RemoteUnreachable with the URL-loading error under NSUnderlyingErrorKey, so a
				caller can tell a server that is not running from one that answered with an error.
*/
+ (nullable NSData *)sendRequest:(NSURLRequest *)request
						 session:(NSURLSession *)session
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*!
	@method     streamRequest:session:lineHandler:completionHandler:
	@abstract   Performs a request whose reply arrives as lines, delivering each as it does.
	@discussion The shape of server-sent events and of newline-delimited JSON alike. lineHandler runs
				once per complete line, without its newline, on the session's delegate queue; a
				success status streams, while any other status is collected whole and handed to the
				completion as errorBody, since a provider explains a rejected request in a JSON body
				rather than a stream. A connection failure arrives as kNFKError_RemoteUnreachable. The
				returned block cancels the request; a cancelled request completes with
				NSURLErrorCancelled. The session's configuration is reused, so a caller's proxy or
				headers carry over.
*/
+ (void (^)(void))streamRequest:(NSURLRequest *)request
						session:(NSURLSession *)session
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(NFKRemoteStreamCompletion)completionHandler;

/*!
	@method     SSEDataForLine:
	@abstract   The payload of a server-sent-events data line, or nil for any other line.
	@discussion A stream's comments, event names, and blank separators answer nil; "data: [DONE]"
				answers the literal "[DONE]", which the OpenAI stream ends with.
*/
+ (nullable NSString *)SSEDataForLine:(NSString *)line;

/*!
	@method     authorizeRequest:apiKey:style:
	@abstract   Adds the credential headers a provider style expects.
	@discussion An OpenAI-compatible provider takes a Bearer token. Anthropic takes the key as
				x-api-key and requires an anthropic-version header, set here to NFKAnthropicAPIVersion.
				A nil or empty key adds no credential, which is what a local server wants.
*/
+ (void)authorizeRequest:(NSMutableURLRequest *)request
				  apiKey:(nullable NSString *)apiKey
				   style:(NFKRemoteAPIStyle)style;

/*!
	@method     errorForResponse:data:
	@abstract   Returns an error for a status outside 200–299, or nil when the response is acceptable.
	@discussion The description carries the status and the body, which is where a provider explains a
				rejected key or an unknown model name.
*/
+ (nullable NSError *)errorForResponse:(nullable NSHTTPURLResponse *)response
								  data:(nullable NSData *)data;

/*! An error in NFKInferenceErrorDomain with the given code and description. */
+ (NSError *)errorWithCode:(NFKInferenceError)code reason:(NSString *)reason;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteTransport_h */
