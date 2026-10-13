//
//  NFKPinnedServersPrivate.h
//  InferKit
//

#import <InferKit/NFKPinnedServers.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKPinnedServers (NFKPrivate)

/*! Decides what to pin for one discovered server, remembering a first-seen fingerprint, and posts the
	notifications the decision calls for. outPinnedFingerprint receives the fingerprint to pin for
	FirstUse and Remembered, and nil otherwise. */
- (NFKDiscoveredPinning)pinningForServiceName:(NSString *)serviceName
									servesTLS:(BOOL)servesTLS
						advertisedFingerprint:(nullable NSString *)advertisedFingerprint
							pinnedFingerprint:(NSString * _Nullable * _Nonnull)outPinnedFingerprint;

@end

NS_ASSUME_NONNULL_END
