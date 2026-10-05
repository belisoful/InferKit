//
//  NFKRemoteTransportPrivate.h
//  InferKit
//

#import <InferKit/NFKRemoteTransport.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKRemoteTransport (NFKPrivate)

/*! One attempt, with no retry, for a caller that answers a busy server by going elsewhere. A failure
	with no response is kNFKError_RemoteUnreachable, as from sendRequest:session:response:error:. */
+ (nullable NSData *)sendOnce:(NSURLRequest *)request
					  session:(NSURLSession *)session
					 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END
