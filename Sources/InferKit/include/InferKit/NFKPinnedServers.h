//
//  NFKPinnedServers.h
//  InferKit
//

#ifndef NFKPinnedServers_h
#define NFKPinnedServers_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKDiscoveredPinning
	@abstract   What automatic pinning did with a discovered InferKit server's certificate.
	@constant   NFKDiscoveredPinningNone Nothing was pinned: automatic pinning is off, the server serves
				plain HTTP, or a TLS server seen for the first time advertised no well-formed
				fingerprint. The provider trusts what an unpinned one trusts.
	@constant   NFKDiscoveredPinningFirstUse The server's service name was seen for the first time, and
				its advertised fingerprint is now remembered and pinned.
	@constant   NFKDiscoveredPinningRemembered The server's service name has a remembered fingerprint,
				which is pinned. The server advertised the same fingerprint, or none.
	@constant   NFKDiscoveredPinningChanged The server's service name has a remembered fingerprint, and
				the server advertised a different one or serves plain HTTP. Nothing is pinned, and the
				app decides through NFKPinnedServers's acceptFingerprint:forServiceName: or
				forgetServiceName:.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKDiscoveredPinning) {
	NFKDiscoveredPinningNone = 0,
	NFKDiscoveredPinningFirstUse,
	NFKDiscoveredPinningRemembered,
	NFKDiscoveredPinningChanged,
};

/*! Posted when the remembered fingerprints change: a first use, an accepted fingerprint, or a
	forgotten server. The object is the NFKPinnedServers; the userInfo names the service under
	NFKPinnedServersServiceNameKey. An app saves dictionaryRepresentation when it receives it. Posted
	on the thread that made the change. Introduced in InferKit 0.4.0. */
extern NSNotificationName const NFKPinnedServersDidChangeNotification;

/*! Posted when a discovered server's service name has a remembered fingerprint and the server
	advertises a different one, or serves plain HTTP. The object is the NFKPinnedServers; the userInfo
	carries NFKPinnedServersServiceNameKey, NFKPinnedServersRememberedFingerprintKey, and, when the
	server advertised one, NFKPinnedServersAdvertisedFingerprintKey. Posted once for each advertised
	fingerprint until the app accepts a fingerprint for the server or forgets it, so repeated
	discovery does not repeat the question. Posted on the discovering thread. Introduced in InferKit
	0.4.0. */
extern NSNotificationName const NFKPinnedServersFingerprintChangedNotification;

/*! The service name, an NSString, in both notifications' userInfo. Introduced in InferKit 0.4.0. */
extern NSString * const NFKPinnedServersServiceNameKey;

/*! The remembered fingerprint, an NSString of 64 lowercase hexadecimal digits. Introduced in InferKit
	0.4.0. */
extern NSString * const NFKPinnedServersRememberedFingerprintKey;

/*! The advertised fingerprint, an NSString of 64 lowercase hexadecimal digits. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKPinnedServersAdvertisedFingerprintKey;

/*!
	@class      NFKPinnedServers
	@abstract   The certificate fingerprints automatic pinning remembers, one per Bonjour service name.
	@discussion Setting an instance as NFKRemoteProvider's discoveryPinnedServers turns automatic
				pinning on for discovery, on trust-on-first-use terms:
				- A TLS server whose service name is new, and whose Bonjour record carries a
				  well-formed cert_sha256, is pinned to that fingerprint, which is remembered.
				- A server whose service name is remembered is pinned to the remembered fingerprint
				  when it advertises the same one or none.
				- A server whose service name is remembered and that advertises a different
				  fingerprint, or serves plain HTTP, is not pinned. The discovered provider reports
				  NFKDiscoveredPinningChanged, NFKBalancedBackend skips it, and
				  NFKPinnedServersFingerprintChangedNotification is posted. The app asks its user, then
				  calls acceptFingerprint:forServiceName: or forgetServiceName:, and the next
				  discovery follows the answer.
				- A TLS server seen for the first time without a well-formed fingerprint, and a plain
				  HTTP server never pinned, are not pinned.

				The key is the service name alone, compared without regard to case as Bonjour
				compares names. The host name and port are left out because they change without the
				server changing: a server listening on port 0 takes a new port at each launch, and a
				machine can be renamed. Leaving them out also means another machine advertising a
				remembered name with its own certificate is reported as a change. A different service
				name is a different server, so its first fingerprint is trusted on first use; an app
				shows the name when it asks its user.

				The library writes no file and no user default. dictionaryRepresentation is the whole
				state, which an app saves when NFKPinnedServersDidChangeNotification arrives and
				restores with initWithDictionaryRepresentation:, so a restored server is pinned the
				first time it is discovered again. Every method is safe to call from any thread.
				Introduced in InferKit 0.4.0.
*/
@interface NFKPinnedServers : NSObject

/*! An empty set: every TLS server's first fingerprint is trusted on first use. */
- (instancetype)init;

/*! The remembered fingerprints restored from dictionaryRepresentation. A value that is not 64
	hexadecimal digits (colons or spaces between the byte pairs allowed) is skipped, and the rest are
	kept as 64 lowercase digits. */
- (instancetype)initWithDictionaryRepresentation:(NSDictionary<NSString *, NSString *> *)dictionaryRepresentation;

/*! The remembered fingerprints, keyed by service name, each 64 lowercase hexadecimal digits. A
	property list, so it saves to user defaults or a file as it is. */
@property (atomic, readonly, copy) NSDictionary<NSString *, NSString *> *dictionaryRepresentation;

/*! The fingerprint remembered for a service name, or nil. */
- (nullable NSString *)fingerprintForServiceName:(NSString *)serviceName;

/*! Remembers a fingerprint for a service name, replacing a remembered one, which is how an app
	accepts a changed certificate. The next discovery of the server pins it. NO, and nothing changes,
	when the fingerprint is not 64 hexadecimal digits. Posts NFKPinnedServersDidChangeNotification when
	the remembered fingerprint changes. */
- (BOOL)acceptFingerprint:(NSString *)fingerprint forServiceName:(NSString *)serviceName;

/*! Forgets a service name, so the next fingerprint it advertises is trusted on first use. Posts
	NFKPinnedServersDidChangeNotification when the name was remembered. */
- (void)forgetServiceName:(NSString *)serviceName;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPinnedServers_h */
