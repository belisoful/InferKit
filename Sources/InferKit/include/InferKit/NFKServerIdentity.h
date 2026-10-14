//
//  NFKServerIdentity.h
//  InferKit
//

#ifndef NFKServerIdentity_h
#define NFKServerIdentity_h

#import <Foundation/Foundation.h>
#import <Security/Security.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error domain for creating, storing, and loading a server identity. Introduced in InferKit
	0.4.0. */
extern NSString * const NFKServerIdentityErrorDomain;

/*!
	@enum       NFKServerIdentityError
	@abstract   Error codes in NFKServerIdentityErrorDomain.
	@constant   NFKServerIdentityErrorKeyFailure The Security framework could not generate, import,
				or sign with the private key. The underlying CFError rides under NSUnderlyingErrorKey.
	@constant   NFKServerIdentityErrorStorageFailure The identity directory or one of its files could
				not be created, read, or written, or belongs to another user.
	@constant   NFKServerIdentityErrorInvalidStoredIdentity The stored identity file is not a P-256
				private key and a certificate whose public key is that key's.
	@constant   NFKServerIdentityErrorInvalidName The common name is empty, the list of subject
				alternative names is empty, or a subject alternative name is neither an IP address nor
				an ASCII DNS name of 1 to 253 characters without whitespace.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKServerIdentityErrorDomain, NFKServerIdentityError) {
	NFKServerIdentityErrorKeyFailure = 1,
	NFKServerIdentityErrorStorageFailure = 2,
	NFKServerIdentityErrorInvalidStoredIdentity = 3,
	NFKServerIdentityErrorInvalidName = 4,
};

/*! How long a new certificate is valid: 825 days, the longest validity Apple platforms accept for a
	TLS server certificate a client trusts through its trust settings. Introduced in InferKit 0.4.0. */
extern const NSTimeInterval NFKServerIdentityDefaultValidity;

/*!
	@class      NFKServerIdentity
	@abstract   A self-signed TLS identity for NFKInferenceServer, kept in a directory across launches.
	@discussion The identity is an ECDSA P-256 key and a self-signed X.509 v3 certificate for it. The
				certificate names the common name as subject and issuer, and carries basicConstraints
				(not a CA), keyUsage digitalSignature, extendedKeyUsage serverAuth, and a
				subjectAlternativeName for each given host name and IP address.

				Pass identity to NFKInferenceServer's TLSIdentity to serve HTTPS. A client trusts the
				server by pinning SHA256Fingerprint: NFKRemoteInferKitBackend's
				pinnedCertificateSHA256, NFKBalancedBackend's server pins, NFKRemoteProvider's
				providerWithPinnedCertificateSHA256:, or NFKRemoteTransport's
				sessionWithPinnedCertificateSHA256: for any other remote class. A client that installs
				the certificate in its trust settings trusts it through the system instead.

				Storage. identityAtDirectoryURL:commonName:subjectAlternativeNames:error: keeps the
				identity in two files in the directory:
				- identity.pem holds the private key (PEM "EC PRIVATE KEY", SEC 1) and the
				  certificate, mode 0600. Loading narrows a looser mode to 0600 and refuses a file
				  another user owns. A directory this class creates has mode 0700.
				- certificate.pem holds the certificate alone, mode 0644, for a client to pin or
				  install.

				Loading builds the identity in memory with SecKeyCreateWithData and SecIdentityCreate.
				No keychain is read or written, so an ad-hoc signed or rebuilt command-line tool and a
				LaunchAgent load it without a keychain prompt or an entitlement. The private key is
				as safe as the user account: any process running as the user, and an administrator,
				can read identity.pem. A process that needs the key bound to its code signature keeps
				the key in the keychain and builds its own SecIdentityRef instead.

				Renewal issues a new certificate, which changes SHA256Fingerprint, so every client
				pinning the old fingerprint must be given the new one. A pinned client does not check
				the certificate's dates; notAfter matters to a client that trusts the certificate
				through its trust settings. Introduced in InferKit 0.4.0.
*/
@interface NFKServerIdentity : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! Application Support/InferKit/identity in the user's domain, the directory used when a caller
	passes nil. */
@property (class, nonatomic, readonly, copy) NSURL *defaultDirectoryURL;

/*!
	@method     identityAtDirectoryURL:commonName:subjectAlternativeNames:error:
	@abstract   Loads the identity stored in a directory, or creates and stores one there.
	@discussion directoryURL nil uses defaultDirectoryURL. A directory that already holds an
				identity returns it unchanged, whatever names are passed, so a server keeps its
				fingerprint across launches; renew it to change its names. commonName nil uses
				"InferKit", and subjectAlternativeNames nil uses localhost, 127.0.0.1, and ::1. A name
				that parses as an IPv4 or IPv6 address becomes an iPAddress entry, and any other a
				dNSName entry. Concurrent callers in different processes serialize on a lock file in
				the directory, so they load the same identity.
*/
+ (nullable instancetype)identityAtDirectoryURL:(nullable NSURL *)directoryURL
									 commonName:(nullable NSString *)commonName
						subjectAlternativeNames:(nullable NSArray<NSString *> *)subjectAlternativeNames
										  error:(NSError * _Nullable *)outError;

/*! Loads the identity stored in a directory, failing with NFKServerIdentityErrorStorageFailure when
	it holds none. */
+ (nullable instancetype)loadIdentityAtDirectoryURL:(nullable NSURL *)directoryURL
											  error:(NSError * _Nullable *)outError;

/*!
	@method     createIdentityWithCommonName:subjectAlternativeNames:validityInterval:error:
	@abstract   Creates an identity with a new key, in memory only.
	@discussion The names follow identityAtDirectoryURL:commonName:subjectAlternativeNames:error:. The
				certificate is valid from an hour before now, which absorbs a client's clock running
				behind, for validityInterval seconds; 0 uses NFKServerIdentityDefaultValidity. Store
				it with writeToDirectoryURL:error:.
*/
+ (nullable instancetype)createIdentityWithCommonName:(nullable NSString *)commonName
							  subjectAlternativeNames:(nullable NSArray<NSString *> *)subjectAlternativeNames
									 validityInterval:(NSTimeInterval)validityInterval
												error:(NSError * _Nullable *)outError;

/*! Builds an identity from PEM text holding an EC PRIVATE KEY block and a CERTIFICATE block, as
	identity.pem does. */
+ (nullable instancetype)identityWithPEMData:(NSData *)PEMData error:(NSError * _Nullable *)outError;

/*!
	@method     writeToDirectoryURL:error:
	@abstract   Stores the identity in a directory, replacing any identity stored there.
	@discussion directoryURL nil uses defaultDirectoryURL, and a directory that does not exist is
				created with mode 0700. Each file is written whole under a temporary name with its
				final mode and renamed into place, so a reader never sees a partial key or a key with
				a looser mode. The identity's directoryURL becomes this directory.
*/
- (BOOL)writeToDirectoryURL:(nullable NSURL *)directoryURL error:(NSError * _Nullable *)outError;

/*!
	@method     renewedIdentityWithNewKey:error:
	@abstract   An identity with a new certificate for the same names, stored where this one is.
	@discussion The new certificate is valid for NFKServerIdentityDefaultValidity from an hour before
				now. newKey NO keeps the private key; YES generates a new one, which a caller does when
				the key may have been read by someone else. Either way the fingerprint changes. When
				this identity has a directoryURL, the new identity replaces it there. This identity is
				left as it is, so a running server keeps serving it until its TLSIdentity is set to the
				new one and it restarts.
*/
- (nullable NFKServerIdentity *)renewedIdentityWithNewKey:(BOOL)newKey error:(NSError * _Nullable *)outError;

/*! The identity to set as NFKInferenceServer's TLSIdentity. It lives as long as this object. */
@property (nonatomic, readonly) SecIdentityRef identity;

/*! The certificate. It lives as long as this object. */
@property (nonatomic, readonly) SecCertificateRef certificate;

/*! The certificate's DER encoding. */
@property (nonatomic, readonly, copy) NSData *certificateData;

/*! The certificate as PEM text, a CERTIFICATE block, which is what certificate.pem holds. */
@property (nonatomic, readonly, copy) NSString *PEMEncodedCertificate;

/*! The SHA-256 digest of certificateData as 64 lowercase hexadecimal digits, the form a pinning
	client takes. */
@property (nonatomic, readonly, copy) NSString *SHA256Fingerprint;

/*! The same digest as 32 uppercase byte pairs separated by colons, the form openssl x509
	-fingerprint -sha256 and Keychain Access print. */
@property (nonatomic, readonly, copy) NSString *colonSeparatedSHA256Fingerprint;

/*! The subject's common name. */
@property (nonatomic, readonly, copy) NSString *commonName;

/*! The subjectAlternativeName entries, DNS names and IP addresses in the certificate's order. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *subjectAlternativeNames;

/*! When the certificate becomes valid. */
@property (nonatomic, readonly, copy) NSDate *notBefore;

/*! When the certificate stops being valid. */
@property (nonatomic, readonly, copy) NSDate *notAfter;

/*! The directory the identity was loaded from or last written to, or nil for one only in memory. */
@property (nonatomic, readonly, copy, nullable) NSURL *directoryURL;

/*! The SHA-256 digest of certificate data as 64 lowercase hexadecimal digits. */
+ (NSString *)SHA256FingerprintOfCertificateData:(NSData *)certificateData;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKServerIdentity_h */
