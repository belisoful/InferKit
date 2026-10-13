//
//  NFKRemoteTransportPrivate.h
//  InferKit
//

#import <InferKit/NFKRemoteTransport.h>

NS_ASSUME_NONNULL_BEGIN

/*! A SHA-256 fingerprint as 64 lowercase hexadecimal digits, from 64 hexadecimal digits in either case
	with or without colons or spaces between the byte pairs, or nil when the value is not one. */
NSString * _Nullable NFKRemoteNormalizedFingerprint(NSString *value);

@interface NFKRemoteTransport (NFKPrivate)

/*! One attempt, with no retry, for a caller that answers a busy server by going elsewhere. A failure
	with no response is kNFKError_RemoteUnreachable, as from sendRequest:session:response:error:. */
+ (nullable NSData *)sendOnce:(NSURLRequest *)request
					  session:(NSURLSession *)session
					 response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END
