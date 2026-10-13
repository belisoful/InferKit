//
//  NFKServerIdentity.m
//  InferKit
//

#import <InferKit/NFKServerIdentity.h>
#import <CommonCrypto/CommonDigest.h>
#import <arpa/inet.h>
#import <fcntl.h>
#import <sys/file.h>
#import <sys/stat.h>
#import <time.h>
#import <unistd.h>

NSString * const NFKServerIdentityErrorDomain = @"NFKServerIdentityErrorDomain";
const NSTimeInterval NFKServerIdentityDefaultValidity = 825.0 * 24.0 * 3600.0;

static NSString * const NFKIdentityFileName = @"identity.pem";
static NSString * const NFKCertificateFileName = @"certificate.pem";
static NSString * const NFKLockFileName = @".lock";
static NSString * const NFKPrivateKeyLabel = @"EC PRIVATE KEY";
static NSString * const NFKCertificateLabel = @"CERTIFICATE";
static const NSUInteger NFKP256PublicLength = 65;
static const NSUInteger NFKP256ScalarLength = 32;

static BOOL NFKIdentityFail(NFKServerIdentityError code, NSString *reason, id _Nullable underlying, NSError **outError)
{
	if (outError != NULL) {
		NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
		if (underlying != nil) {
			userInfo[NSUnderlyingErrorKey] = underlying;
		}
		*outError = [NSError errorWithDomain:NFKServerIdentityErrorDomain code:code userInfo:userInfo];
	}
	return NO;
}

static NSError *NFKPOSIXError(void)
{
	return [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:nil];
}

#pragma mark - DER writing

static void NFKDERAppendLength(NSMutableData *data, NSUInteger length)
{
	if (length < 0x80) {
		uint8_t byte = (uint8_t)length;
		[data appendBytes:&byte length:1];
		return;
	}
	uint8_t bytes[sizeof(NSUInteger)];
	NSUInteger count = 0;
	for (NSUInteger rest = length; rest > 0; rest >>= 8) {
		bytes[sizeof(bytes) - 1 - count] = (uint8_t)(rest & 0xFF);
		count += 1;
	}
	uint8_t prefix = (uint8_t)(0x80 | count);
	[data appendBytes:&prefix length:1];
	[data appendBytes:bytes + sizeof(bytes) - count length:count];
}

static NSData *NFKDER(uint8_t tag, NSData *content)
{
	NSMutableData *data = [NSMutableData dataWithBytes:&tag length:1];
	NFKDERAppendLength(data, content.length);
	[data appendData:content];
	return data;
}

static NSData *NFKDERConstructed(uint8_t tag, NSArray<NSData *> *items)
{
	NSMutableData *content = [NSMutableData data];
	for (NSData *item in items) {
		[content appendData:item];
	}
	return NFKDER(tag, content);
}

static NSData *NFKDERSequence(NSArray<NSData *> *items)
{
	return NFKDERConstructed(0x30, items);
}

static NSData *NFKDERBytes(uint8_t tag, const void *bytes, NSUInteger length)
{
	return NFKDER(tag, [NSData dataWithBytes:bytes length:length]);
}

/*! A BIT STRING whose bytes are whole, with no unused trailing bits. */
static NSData *NFKDERBitString(NSData *bytes)
{
	NSMutableData *content = [NSMutableData dataWithLength:1];
	[content appendData:bytes];
	return NFKDER(0x03, content);
}

static NSData *NFKDERTime(NSDate *date)
{
	time_t seconds = (time_t)floor(date.timeIntervalSince1970);
	struct tm parts;
	gmtime_r(&seconds, &parts);
	int year = parts.tm_year + 1900;
	char text[24];
	// RFC 5280 4.1.2.5: UTCTime through 2049, GeneralizedTime from 2050.
	if (year >= 1950 && year < 2050) {
		snprintf(text, sizeof(text), "%02d%02d%02d%02d%02d%02dZ", year % 100, parts.tm_mon + 1, parts.tm_mday,
				 parts.tm_hour, parts.tm_min, parts.tm_sec);
		return NFKDERBytes(0x17, text, strlen(text));
	}
	snprintf(text, sizeof(text), "%04d%02d%02d%02d%02d%02dZ", year, parts.tm_mon + 1, parts.tm_mday,
			 parts.tm_hour, parts.tm_min, parts.tm_sec);
	return NFKDERBytes(0x18, text, strlen(text));
}

static const uint8_t NFKOIDECPublicKey[] = { 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01 };
static const uint8_t NFKOIDPrime256v1[] = { 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07 };
static const uint8_t NFKOIDECDSAWithSHA256[] = { 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02 };
static const uint8_t NFKOIDCommonName[] = { 0x55, 0x04, 0x03 };
static const uint8_t NFKOIDBasicConstraints[] = { 0x55, 0x1D, 0x13 };
static const uint8_t NFKOIDKeyUsage[] = { 0x55, 0x1D, 0x0F };
static const uint8_t NFKOIDExtendedKeyUsage[] = { 0x55, 0x1D, 0x25 };
static const uint8_t NFKOIDSubjectAltName[] = { 0x55, 0x1D, 0x11 };
static const uint8_t NFKOIDSubjectKeyIdentifier[] = { 0x55, 0x1D, 0x0E };
static const uint8_t NFKOIDServerAuth[] = { 0x2B, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01 };

#define NFKDEROID(oid) NFKDERBytes(0x06, (oid), sizeof(oid))

static NSData *NFKDERExtension(NSData *oid, BOOL critical, NSData *value)
{
	NSMutableArray<NSData *> *items = [NSMutableArray arrayWithObject:oid];
	if (critical) {
		const uint8_t yes = 0xFF;
		[items addObject:NFKDERBytes(0x01, &yes, 1)];
	}
	[items addObject:NFKDER(0x04, value)];
	return NFKDERSequence(items);
}

static NSData *NFKDERName(NSString *commonName)
{
	NSData *attribute = NFKDERSequence(@[ NFKDEROID(NFKOIDCommonName),
										  NFKDER(0x0C, [commonName dataUsingEncoding:NSUTF8StringEncoding]) ]);
	return NFKDERSequence(@[ NFKDERConstructed(0x31, @[ attribute ]) ]);
}

#pragma mark - DER reading

typedef struct {
	const uint8_t *bytes;
	NSUInteger length;
} NFKDERSlice;

static NFKDERSlice NFKDERSliceOfData(NSData *data)
{
	return (NFKDERSlice){ data.bytes, data.length };
}

/*! Reads the element at the front of rest, advancing rest past it. NO when rest does not start with
	a whole element. */
static BOOL NFKDERNext(NFKDERSlice *rest, uint8_t *outTag, NFKDERSlice *outContent)
{
	if (rest->length < 2) {
		return NO;
	}
	NSUInteger offset = 2;
	NSUInteger length = rest->bytes[1];
	if (length & 0x80) {
		NSUInteger count = length & 0x7F;
		if (count == 0 || count > 4 || rest->length < 2 + count) {
			return NO;
		}
		length = 0;
		for (NSUInteger index = 0; index < count; index++) {
			length = (length << 8) | rest->bytes[2 + index];
		}
		offset += count;
	}
	if (rest->length - offset < length) {
		return NO;
	}
	*outTag = rest->bytes[0];
	*outContent = (NFKDERSlice){ rest->bytes + offset, length };
	rest->bytes += offset + length;
	rest->length -= offset + length;
	return YES;
}

/*! Reads the element at the front of rest and requires its tag. */
static BOOL NFKDERExpect(NFKDERSlice *rest, uint8_t tag, NFKDERSlice *outContent)
{
	uint8_t found = 0;
	return NFKDERNext(rest, &found, outContent) && found == tag;
}

static BOOL NFKDERSliceEquals(NFKDERSlice slice, const uint8_t *bytes, NSUInteger length)
{
	return slice.length == length && memcmp(slice.bytes, bytes, length) == 0;
}

static NSData *NFKDERSliceData(NFKDERSlice slice)
{
	return [NSData dataWithBytes:slice.bytes length:slice.length];
}

static NSDate * _Nullable NFKDERReadTime(uint8_t tag, NFKDERSlice content)
{
	NSString *text = [[NSString alloc] initWithBytes:content.bytes length:content.length encoding:NSASCIIStringEncoding];
	NSUInteger yearDigits = tag == 0x17 ? 2 : (tag == 0x18 ? 4 : 0);
	if (yearDigits == 0 || text.length != yearDigits + 11 || ![text hasSuffix:@"Z"]) {
		return nil;
	}
	int year = [text substringToIndex:yearDigits].intValue;
	if (tag == 0x17) {
		year += year < 50 ? 2000 : 1900;
	}
	struct tm parts = { 0 };
	parts.tm_year = year - 1900;
	parts.tm_mon = [text substringWithRange:NSMakeRange(yearDigits, 2)].intValue - 1;
	parts.tm_mday = [text substringWithRange:NSMakeRange(yearDigits + 2, 2)].intValue;
	parts.tm_hour = [text substringWithRange:NSMakeRange(yearDigits + 4, 2)].intValue;
	parts.tm_min = [text substringWithRange:NSMakeRange(yearDigits + 6, 2)].intValue;
	parts.tm_sec = [text substringWithRange:NSMakeRange(yearDigits + 8, 2)].intValue;
	return [NSDate dateWithTimeIntervalSince1970:(NSTimeInterval)timegm(&parts)];
}

static NSString * _Nullable NFKDERReadCommonName(NFKDERSlice name)
{
	NFKDERSlice set;
	while (NFKDERExpect(&name, 0x31, &set)) {
		NFKDERSlice attribute;
		while (NFKDERExpect(&set, 0x30, &attribute)) {
			NFKDERSlice oid;
			NFKDERSlice value;
			uint8_t valueTag = 0;
			if (NFKDERExpect(&attribute, 0x06, &oid) && NFKDERNext(&attribute, &valueTag, &value)
				&& NFKDERSliceEquals(oid, NFKOIDCommonName, sizeof(NFKOIDCommonName))) {
				return [[NSString alloc] initWithBytes:value.bytes length:value.length encoding:NSUTF8StringEncoding];
			}
		}
	}
	return nil;
}

static NSArray<NSString *> *NFKDERReadAlternativeNames(NFKDERSlice value)
{
	NSMutableArray<NSString *> *names = [NSMutableArray array];
	NFKDERSlice list;
	if (!NFKDERExpect(&value, 0x30, &list)) {
		return names;
	}
	uint8_t tag = 0;
	NFKDERSlice entry;
	while (NFKDERNext(&list, &tag, &entry)) {
		if (tag == 0x82) {
			NSString *host = [[NSString alloc] initWithBytes:entry.bytes length:entry.length encoding:NSASCIIStringEncoding];
			if (host != nil) {
				[names addObject:host];
			}
		} else if (tag == 0x87 && (entry.length == 4 || entry.length == 16)) {
			char text[INET6_ADDRSTRLEN];
			if (inet_ntop(entry.length == 4 ? AF_INET : AF_INET6, entry.bytes, text, sizeof(text)) != NULL) {
				[names addObject:@(text)];
			}
		}
	}
	return names;
}

#pragma mark - PEM

static NSString *NFKPEMBlock(NSString *label, NSData *der)
{
	NSString *body = [der base64EncodedStringWithOptions:NSDataBase64Encoding64CharacterLineLength | NSDataBase64EncodingEndLineWithLineFeed];
	return [NSString stringWithFormat:@"-----BEGIN %@-----\n%@\n-----END %@-----\n", label, body, label];
}

static NSData * _Nullable NFKPEMBlockData(NSString *text, NSString *label)
{
	NSString *begin = [NSString stringWithFormat:@"-----BEGIN %@-----", label];
	NSString *end = [NSString stringWithFormat:@"-----END %@-----", label];
	NSRange start = [text rangeOfString:begin];
	if (start.location == NSNotFound) {
		return nil;
	}
	NSUInteger bodyStart = NSMaxRange(start);
	NSRange stop = [text rangeOfString:end options:0 range:NSMakeRange(bodyStart, text.length - bodyStart)];
	if (stop.location == NSNotFound) {
		return nil;
	}
	NSString *body = [text substringWithRange:NSMakeRange(bodyStart, stop.location - bodyStart)];
	return [[NSData alloc] initWithBase64EncodedString:body options:NSDataBase64DecodingIgnoreUnknownCharacters];
}

#pragma mark - Identity

@interface NFKServerIdentity ()
@property (nonatomic, strong) id identityObject;
@property (nonatomic, strong) id certificateObject;
@property (nonatomic, strong) id privateKeyObject;
@property (nonatomic, readwrite, copy) NSData *certificateData;
@property (nonatomic, readwrite, copy) NSString *SHA256Fingerprint;
@property (nonatomic, readwrite, copy) NSString *commonName;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *subjectAlternativeNames;
@property (nonatomic, readwrite, copy) NSDate *notBefore;
@property (nonatomic, readwrite, copy) NSDate *notAfter;
@property (nonatomic, readwrite, copy, nullable) NSURL *directoryURL;
@end

@implementation NFKServerIdentity

+ (NSURL *)defaultDirectoryURL
{
	NSURL *support = [NSFileManager.defaultManager URLsForDirectory:NSApplicationSupportDirectory inDomains:NSUserDomainMask].firstObject;
	if (support == nil) {
		support = [NSURL fileURLWithPath:[NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support"] isDirectory:YES];
	}
	return [[support URLByAppendingPathComponent:@"InferKit" isDirectory:YES] URLByAppendingPathComponent:@"identity" isDirectory:YES];
}

+ (NSString *)SHA256FingerprintOfCertificateData:(NSData *)certificateData
{
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(certificateData.bytes, (CC_LONG)certificateData.length, digest);
	NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
	for (NSUInteger index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) {
		[hex appendFormat:@"%02x", digest[index]];
	}
	return hex;
}

#pragma mark Creating

+ (nullable instancetype)createIdentityWithCommonName:(nullable NSString *)commonName
							  subjectAlternativeNames:(nullable NSArray<NSString *> *)subjectAlternativeNames
									 validityInterval:(NSTimeInterval)validityInterval
												error:(NSError * _Nullable *)outError
{
	id privateKey = [self generatedPrivateKeyWithError:outError];
	if (privateKey == nil) {
		return nil;
	}
	return [self identityWithPrivateKey:(__bridge SecKeyRef)privateKey
							 commonName:commonName ?: @"InferKit"
				subjectAlternativeNames:subjectAlternativeNames ?: @[ @"localhost", @"127.0.0.1", @"::1" ]
					   validityInterval:validityInterval > 0 ? validityInterval : NFKServerIdentityDefaultValidity
								  error:outError];
}

+ (nullable id)generatedPrivateKeyWithError:(NSError **)outError
{
	NSDictionary *attributes = @{ (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
								  (id)kSecAttrKeySizeInBits: @256,
								  (id)kSecAttrIsPermanent: @NO };
	CFErrorRef error = NULL;
	SecKeyRef key = SecKeyCreateRandomKey((__bridge CFDictionaryRef)attributes, &error);
	if (key == NULL) {
		NFKIdentityFail(NFKServerIdentityErrorKeyFailure, @"the P-256 key could not be generated", CFBridgingRelease(error), outError);
		return nil;
	}
	return CFBridgingRelease(key);
}

/*! Issues a self-signed certificate for the key and builds the identity around it. */
+ (nullable instancetype)identityWithPrivateKey:(SecKeyRef)privateKey
									 commonName:(NSString *)commonName
						subjectAlternativeNames:(NSArray<NSString *> *)subjectAlternativeNames
							   validityInterval:(NSTimeInterval)validityInterval
										  error:(NSError **)outError
{
	NSData *alternativeNames = [self alternativeNamesExtensionValue:subjectAlternativeNames error:outError];
	if (alternativeNames == nil) {
		return nil;
	}
	if (commonName.length == 0) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidName, @"the common name is empty", nil, outError);
		return nil;
	}
	id publicKey = CFBridgingRelease(SecKeyCopyPublicKey(privateKey));
	NSData *publicBytes = publicKey != nil ? CFBridgingRelease(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)publicKey, NULL)) : nil;
	if (publicBytes.length != NFKP256PublicLength) {
		NFKIdentityFail(NFKServerIdentityErrorKeyFailure, @"the key's public half is not a P-256 point", nil, outError);
		return nil;
	}
	NSDate *now = [NSDate dateWithTimeIntervalSince1970:floor(NSDate.date.timeIntervalSince1970)];
	NSData *tbs = [self toBeSignedWithPublicKey:publicBytes
									 commonName:commonName
							   alternativeNames:alternativeNames
									  notBefore:[now dateByAddingTimeInterval:-3600.0]
									   notAfter:[now dateByAddingTimeInterval:validityInterval]];
	CFErrorRef signError = NULL;
	NSData *signature = CFBridgingRelease(SecKeyCreateSignature(privateKey, kSecKeyAlgorithmECDSASignatureMessageX962SHA256,
																(__bridge CFDataRef)tbs, &signError));
	if (signature == nil) {
		NFKIdentityFail(NFKServerIdentityErrorKeyFailure, @"the certificate could not be signed", CFBridgingRelease(signError), outError);
		return nil;
	}
	NSData *certificate = NFKDERSequence(@[ tbs,
											NFKDERSequence(@[ NFKDEROID(NFKOIDECDSAWithSHA256) ]),
											NFKDERBitString(signature) ]);
	return [[self alloc] initWithPrivateKey:privateKey certificateData:certificate error:outError];
}

+ (NSData *)toBeSignedWithPublicKey:(NSData *)publicBytes
						 commonName:(NSString *)commonName
				   alternativeNames:(NSData *)alternativeNames
						  notBefore:(NSDate *)notBefore
						   notAfter:(NSDate *)notAfter
{
	uint8_t serial[16];
	if (SecRandomCopyBytes(kSecRandomDefault, sizeof(serial), serial) != errSecSuccess) {
		arc4random_buf(serial, sizeof(serial));
	}
	// A positive INTEGER in its shortest form: the top bit clear and the first byte nonzero.
	serial[0] = (uint8_t)((serial[0] & 0x7F) | 0x01);
	const uint8_t version = 2;
	unsigned char keyDigest[CC_SHA1_DIGEST_LENGTH];
	CC_SHA1(publicBytes.bytes, (CC_LONG)publicBytes.length, keyDigest);
	const uint8_t digitalSignature[] = { 0x07, 0x80 };
	NSData *subjectPublicKeyInfo = NFKDERSequence(@[ NFKDERSequence(@[ NFKDEROID(NFKOIDECPublicKey), NFKDEROID(NFKOIDPrime256v1) ]),
													  NFKDERBitString(publicBytes) ]);
	NSArray<NSData *> *extensions = @[
		NFKDERExtension(NFKDEROID(NFKOIDBasicConstraints), YES, NFKDERSequence(@[])),
		NFKDERExtension(NFKDEROID(NFKOIDKeyUsage), YES, NFKDERBytes(0x03, digitalSignature, sizeof(digitalSignature))),
		NFKDERExtension(NFKDEROID(NFKOIDExtendedKeyUsage), NO, NFKDERSequence(@[ NFKDEROID(NFKOIDServerAuth) ])),
		NFKDERExtension(NFKDEROID(NFKOIDSubjectAltName), NO, alternativeNames),
		NFKDERExtension(NFKDEROID(NFKOIDSubjectKeyIdentifier), NO, NFKDERBytes(0x04, keyDigest, sizeof(keyDigest))),
	];
	NSData *name = NFKDERName(commonName);
	return NFKDERSequence(@[ NFKDERConstructed(0xA0, @[ NFKDERBytes(0x02, &version, 1) ]),
							 NFKDERBytes(0x02, serial, sizeof(serial)),
							 NFKDERSequence(@[ NFKDEROID(NFKOIDECDSAWithSHA256) ]),
							 name,
							 NFKDERSequence(@[ NFKDERTime(notBefore), NFKDERTime(notAfter) ]),
							 name,
							 subjectPublicKeyInfo,
							 NFKDERConstructed(0xA3, @[ NFKDERSequence(extensions) ]) ]);
}

/*! The subjectAltName extension's value: an IP address entry for each name that parses as one, a DNS
	name entry for each other name. */
+ (nullable NSData *)alternativeNamesExtensionValue:(NSArray<NSString *> *)names error:(NSError **)outError
{
	NSMutableArray<NSData *> *entries = [NSMutableArray array];
	for (NSString *name in names) {
		struct in_addr v4;
		struct in6_addr v6;
		if (inet_pton(AF_INET, name.UTF8String, &v4) == 1) {
			[entries addObject:NFKDERBytes(0x87, &v4, sizeof(v4))];
			continue;
		}
		if (inet_pton(AF_INET6, name.UTF8String, &v6) == 1) {
			[entries addObject:NFKDERBytes(0x87, &v6, sizeof(v6))];
			continue;
		}
		NSData *host = [name dataUsingEncoding:NSASCIIStringEncoding];
		if (host.length == 0 || host.length > 253 || [name rangeOfCharacterFromSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].location != NSNotFound) {
			NSString *reason = [NSString stringWithFormat:@"\"%@\" is neither an IP address nor an ASCII DNS name", name];
			NFKIdentityFail(NFKServerIdentityErrorInvalidName, reason, nil, outError);
			return nil;
		}
		[entries addObject:NFKDER(0x82, host)];
	}
	if (entries.count == 0) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidName, @"the certificate needs at least one subject alternative name", nil, outError);
		return nil;
	}
	return NFKDERSequence(entries);
}

#pragma mark Building

- (nullable instancetype)initWithPrivateKey:(SecKeyRef)privateKey certificateData:(NSData *)certificateData error:(NSError **)outError
{
	self = [super init];
	if (self == nil) {
		return nil;
	}
	id certificate = CFBridgingRelease(SecCertificateCreateWithData(NULL, (__bridge CFDataRef)certificateData));
	if (certificate == nil) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidStoredIdentity, @"the certificate is not DER X.509", nil, outError);
		return nil;
	}
	id identity = CFBridgingRelease(SecIdentityCreate(NULL, (__bridge SecCertificateRef)certificate, privateKey));
	if (identity == nil) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidStoredIdentity, @"the certificate's public key is not the private key's", nil, outError);
		return nil;
	}
	if (![self readCertificateFields:certificateData]) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidStoredIdentity, @"the certificate's subject, validity, or names cannot be read", nil, outError);
		return nil;
	}
	_identityObject = identity;
	_certificateObject = certificate;
	_privateKeyObject = (__bridge id)privateKey;
	_certificateData = [certificateData copy];
	_SHA256Fingerprint = [NFKServerIdentity SHA256FingerprintOfCertificateData:certificateData];
	return self;
}

/*! Reads the common name, the validity, and the subjectAltName entries from the certificate's DER. */
- (BOOL)readCertificateFields:(NSData *)certificateData
{
	NFKDERSlice whole = NFKDERSliceOfData(certificateData);
	NFKDERSlice certificate;
	NFKDERSlice tbs;
	if (!NFKDERExpect(&whole, 0x30, &certificate) || !NFKDERExpect(&certificate, 0x30, &tbs)) {
		return NO;
	}
	uint8_t tag = 0;
	NFKDERSlice field;
	if (!NFKDERNext(&tbs, &tag, &field)) {
		return NO;
	}
	if (tag == 0xA0 && !NFKDERNext(&tbs, &tag, &field)) {
		return NO;
	}
	NFKDERSlice algorithm;
	NFKDERSlice issuer;
	NFKDERSlice validity;
	NFKDERSlice subject;
	NFKDERSlice publicKeyInfo;
	if (!NFKDERExpect(&tbs, 0x30, &algorithm) || !NFKDERExpect(&tbs, 0x30, &issuer) || !NFKDERExpect(&tbs, 0x30, &validity)
		|| !NFKDERExpect(&tbs, 0x30, &subject) || !NFKDERExpect(&tbs, 0x30, &publicKeyInfo)) {
		return NO;
	}
	uint8_t startTag = 0;
	uint8_t endTag = 0;
	NFKDERSlice start;
	NFKDERSlice end;
	if (!NFKDERNext(&validity, &startTag, &start) || !NFKDERNext(&validity, &endTag, &end)) {
		return NO;
	}
	NSDate *notBefore = NFKDERReadTime(startTag, start);
	NSDate *notAfter = NFKDERReadTime(endTag, end);
	if (notBefore == nil || notAfter == nil) {
		return NO;
	}
	NSArray<NSString *> *alternativeNames = @[];
	while (NFKDERNext(&tbs, &tag, &field)) {
		NFKDERSlice extensions;
		if (tag != 0xA3 || !NFKDERExpect(&field, 0x30, &extensions)) {
			continue;
		}
		NFKDERSlice extension;
		while (NFKDERExpect(&extensions, 0x30, &extension)) {
			NFKDERSlice oid;
			NFKDERSlice value;
			if (!NFKDERExpect(&extension, 0x06, &oid) || !NFKDERSliceEquals(oid, NFKOIDSubjectAltName, sizeof(NFKOIDSubjectAltName))) {
				continue;
			}
			uint8_t valueTag = 0;
			while (NFKDERNext(&extension, &valueTag, &value)) {
				if (valueTag == 0x04) {
					alternativeNames = NFKDERReadAlternativeNames(value);
				}
			}
		}
	}
	_commonName = NFKDERReadCommonName(subject) ?: @"";
	_notBefore = notBefore;
	_notAfter = notAfter;
	_subjectAlternativeNames = alternativeNames;
	return YES;
}

+ (nullable instancetype)identityWithPEMData:(NSData *)PEMData error:(NSError * _Nullable *)outError
{
	NSString *text = [[NSString alloc] initWithData:PEMData encoding:NSUTF8StringEncoding];
	NSData *keyDER = text != nil ? NFKPEMBlockData(text, NFKPrivateKeyLabel) : nil;
	NSData *certificateDER = text != nil ? NFKPEMBlockData(text, NFKCertificateLabel) : nil;
	if (keyDER == nil || certificateDER == nil) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidStoredIdentity, @"the PEM text lacks an EC PRIVATE KEY block or a CERTIFICATE block", nil, outError);
		return nil;
	}
	NSData *x963 = [self x963RepresentationOfECPrivateKey:keyDER];
	if (x963 == nil) {
		NFKIdentityFail(NFKServerIdentityErrorInvalidStoredIdentity, @"the EC PRIVATE KEY block is not a P-256 key with its public point", nil, outError);
		return nil;
	}
	NSDictionary *attributes = @{ (id)kSecAttrKeyType: (id)kSecAttrKeyTypeECSECPrimeRandom,
								  (id)kSecAttrKeyClass: (id)kSecAttrKeyClassPrivate,
								  (id)kSecAttrKeySizeInBits: @256 };
	CFErrorRef error = NULL;
	SecKeyRef key = SecKeyCreateWithData((__bridge CFDataRef)x963, (__bridge CFDictionaryRef)attributes, &error);
	if (key == NULL) {
		NFKIdentityFail(NFKServerIdentityErrorKeyFailure, @"the stored private key could not be imported", CFBridgingRelease(error), outError);
		return nil;
	}
	id privateKey = CFBridgingRelease(key);
	return [[self alloc] initWithPrivateKey:(__bridge SecKeyRef)privateKey certificateData:certificateDER error:outError];
}

/*! SEC 1 ECPrivateKey: SEQUENCE { 1, OCTET STRING d, [0] prime256v1, [1] BIT STRING point }. */
+ (NSData *)ECPrivateKeyWithX963Representation:(NSData *)x963
{
	const uint8_t version = 1;
	NSData *point = [x963 subdataWithRange:NSMakeRange(0, NFKP256PublicLength)];
	NSData *scalar = [x963 subdataWithRange:NSMakeRange(NFKP256PublicLength, NFKP256ScalarLength)];
	return NFKDERSequence(@[ NFKDERBytes(0x02, &version, 1),
							 NFKDER(0x04, scalar),
							 NFKDERConstructed(0xA0, @[ NFKDEROID(NFKOIDPrime256v1) ]),
							 NFKDERConstructed(0xA1, @[ NFKDERBitString(point) ]) ]);
}

/*! The point followed by the scalar, which is what SecKeyCreateWithData takes, or nil for any key
	other than a P-256 ECPrivateKey that carries its point. */
+ (nullable NSData *)x963RepresentationOfECPrivateKey:(NSData *)der
{
	NFKDERSlice whole = NFKDERSliceOfData(der);
	NFKDERSlice key;
	NFKDERSlice version;
	NFKDERSlice scalar;
	if (!NFKDERExpect(&whole, 0x30, &key) || !NFKDERExpect(&key, 0x02, &version) || !NFKDERExpect(&key, 0x04, &scalar)
		|| scalar.length != NFKP256ScalarLength) {
		return nil;
	}
	NSData *point = nil;
	BOOL curveIsP256 = NO;
	uint8_t tag = 0;
	NFKDERSlice field;
	while (NFKDERNext(&key, &tag, &field)) {
		NFKDERSlice inner;
		if (tag == 0xA0 && NFKDERExpect(&field, 0x06, &inner)) {
			curveIsP256 = NFKDERSliceEquals(inner, NFKOIDPrime256v1, sizeof(NFKOIDPrime256v1));
		} else if (tag == 0xA1 && NFKDERExpect(&field, 0x03, &inner) && inner.length == NFKP256PublicLength + 1 && inner.bytes[0] == 0) {
			point = [NSData dataWithBytes:inner.bytes + 1 length:NFKP256PublicLength];
		}
	}
	if (!curveIsP256 || point == nil) {
		return nil;
	}
	NSMutableData *x963 = [point mutableCopy];
	[x963 appendData:NFKDERSliceData(scalar)];
	return x963;
}

#pragma mark Accessors

- (SecIdentityRef)identity
{
	return (__bridge SecIdentityRef)self.identityObject;
}

- (SecCertificateRef)certificate
{
	return (__bridge SecCertificateRef)self.certificateObject;
}

- (NSString *)PEMEncodedCertificate
{
	return NFKPEMBlock(NFKCertificateLabel, self.certificateData);
}

- (NSString *)colonSeparatedSHA256Fingerprint
{
	NSString *hex = self.SHA256Fingerprint.uppercaseString;
	NSMutableArray<NSString *> *pairs = [NSMutableArray arrayWithCapacity:hex.length / 2];
	for (NSUInteger index = 0; index + 1 < hex.length; index += 2) {
		[pairs addObject:[hex substringWithRange:NSMakeRange(index, 2)]];
	}
	return [pairs componentsJoinedByString:@":"];
}

/*! identity.pem's contents: the private key, then the certificate. */
- (nullable NSData *)identityPEMDataWithError:(NSError **)outError
{
	CFErrorRef error = NULL;
	NSData *x963 = CFBridgingRelease(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)self.privateKeyObject, &error));
	if (x963.length != NFKP256PublicLength + NFKP256ScalarLength) {
		NFKIdentityFail(NFKServerIdentityErrorKeyFailure, @"the private key cannot be exported", CFBridgingRelease(error), outError);
		return nil;
	}
	if (error != NULL) {
		CFRelease(error);
	}
	NSString *text = [NFKPEMBlock(NFKPrivateKeyLabel, [NFKServerIdentity ECPrivateKeyWithX963Representation:x963])
					  stringByAppendingString:self.PEMEncodedCertificate];
	return [text dataUsingEncoding:NSUTF8StringEncoding];
}

#pragma mark Storage

+ (nullable instancetype)identityAtDirectoryURL:(nullable NSURL *)directoryURL
									 commonName:(nullable NSString *)commonName
						subjectAlternativeNames:(nullable NSArray<NSString *> *)subjectAlternativeNames
										  error:(NSError * _Nullable *)outError
{
	NSURL *directory = directoryURL ?: self.defaultDirectoryURL;
	int lock = [self lockDirectory:directory error:outError];
	if (lock < 0) {
		return nil;
	}
	NFKServerIdentity *identity = nil;
	NSURL *file = [directory URLByAppendingPathComponent:NFKIdentityFileName];
	if ([NSFileManager.defaultManager fileExistsAtPath:file.path]) {
		identity = [self loadIdentityFromFile:file error:outError];
	} else {
		identity = [self createIdentityWithCommonName:commonName subjectAlternativeNames:subjectAlternativeNames validityInterval:0 error:outError];
		if (identity != nil && ![identity writeFilesToDirectory:directory error:outError]) {
			identity = nil;
		}
	}
	flock(lock, LOCK_UN);
	close(lock);
	identity.directoryURL = identity != nil ? directory : nil;
	return identity;
}

+ (nullable instancetype)loadIdentityAtDirectoryURL:(nullable NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	NSURL *directory = directoryURL ?: self.defaultDirectoryURL;
	NFKServerIdentity *identity = [self loadIdentityFromFile:[directory URLByAppendingPathComponent:NFKIdentityFileName] error:outError];
	identity.directoryURL = identity != nil ? directory : nil;
	return identity;
}

/*! Reads identity.pem after confirming the user owns it, narrowing its mode to 0600 first. */
+ (nullable instancetype)loadIdentityFromFile:(NSURL *)file error:(NSError **)outError
{
	int descriptor = open(file.fileSystemRepresentation, O_RDONLY | O_NOFOLLOW);
	if (descriptor < 0) {
		NSString *reason = [NSString stringWithFormat:@"%@ cannot be opened", file.path];
		NFKIdentityFail(NFKServerIdentityErrorStorageFailure, reason, NFKPOSIXError(), outError);
		return nil;
	}
	struct stat status;
	BOOL owned = fstat(descriptor, &status) == 0 && status.st_uid == geteuid() && S_ISREG(status.st_mode);
	if (owned && (status.st_mode & 0077) != 0) {
		owned = fchmod(descriptor, 0600) == 0;
	}
	NSFileHandle *handle = [[NSFileHandle alloc] initWithFileDescriptor:descriptor closeOnDealloc:YES];
	if (!owned) {
		NSString *reason = [NSString stringWithFormat:@"%@ is not a file this user owns with mode 0600", file.path];
		NFKIdentityFail(NFKServerIdentityErrorStorageFailure, reason, nil, outError);
		return nil;
	}
	NSData *contents = [handle readDataToEndOfFile];
	return [self identityWithPEMData:contents error:outError];
}

- (BOOL)writeToDirectoryURL:(nullable NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	NSURL *directory = directoryURL ?: NFKServerIdentity.defaultDirectoryURL;
	int lock = [NFKServerIdentity lockDirectory:directory error:outError];
	if (lock < 0) {
		return NO;
	}
	BOOL written = [self writeFilesToDirectory:directory error:outError];
	flock(lock, LOCK_UN);
	close(lock);
	if (written) {
		self.directoryURL = directory;
	}
	return written;
}

/*! Writes both files; the caller holds the directory's lock. */
- (BOOL)writeFilesToDirectory:(NSURL *)directory error:(NSError **)outError
{
	NSData *identityPEM = [self identityPEMDataWithError:outError];
	if (identityPEM == nil) {
		return NO;
	}
	return [NFKServerIdentity replaceFile:[directory URLByAppendingPathComponent:NFKIdentityFileName] withData:identityPEM mode:0600 error:outError]
		&& [NFKServerIdentity replaceFile:[directory URLByAppendingPathComponent:NFKCertificateFileName]
								 withData:[self.PEMEncodedCertificate dataUsingEncoding:NSUTF8StringEncoding] mode:0644 error:outError];
}

/*! Creates the directory when missing and takes its lock file's exclusive lock, returning the
	locked descriptor, or -1. */
+ (int)lockDirectory:(NSURL *)directory error:(NSError **)outError
{
	NSError *createError = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory
								withIntermediateDirectories:YES
												 attributes:@{ NSFilePosixPermissions: @0700 }
													  error:&createError]) {
		NSString *reason = [NSString stringWithFormat:@"%@ cannot be created", directory.path];
		NFKIdentityFail(NFKServerIdentityErrorStorageFailure, reason, createError, outError);
		return -1;
	}
	NSURL *lockFile = [directory URLByAppendingPathComponent:NFKLockFileName];
	int descriptor = open(lockFile.fileSystemRepresentation, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
	if (descriptor < 0 || flock(descriptor, LOCK_EX) != 0) {
		NSError *underlying = NFKPOSIXError();
		if (descriptor >= 0) {
			close(descriptor);
		}
		NSString *reason = [NSString stringWithFormat:@"%@ cannot be locked", directory.path];
		NFKIdentityFail(NFKServerIdentityErrorStorageFailure, reason, underlying, outError);
		return -1;
	}
	return descriptor;
}

/*! Writes data under a temporary name created with the final mode, then renames it over file. */
+ (BOOL)replaceFile:(NSURL *)file withData:(NSData *)data mode:(mode_t)mode error:(NSError **)outError
{
	NSString *temporaryName = [NSString stringWithFormat:@".%@.%@", file.lastPathComponent, NSUUID.UUID.UUIDString];
	NSURL *temporary = [file.URLByDeletingLastPathComponent URLByAppendingPathComponent:temporaryName];
	int descriptor = open(temporary.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, mode);
	BOOL written = descriptor >= 0 && fchmod(descriptor, mode) == 0
		&& write(descriptor, data.bytes, data.length) == (ssize_t)data.length && fsync(descriptor) == 0;
	NSError *underlying = written ? nil : NFKPOSIXError();
	if (descriptor >= 0) {
		written = close(descriptor) == 0 && written;
	}
	if (written && rename(temporary.fileSystemRepresentation, file.fileSystemRepresentation) == 0) {
		return YES;
	}
	underlying = underlying ?: NFKPOSIXError();
	if (descriptor >= 0) {
		unlink(temporary.fileSystemRepresentation);
	}
	NSString *reason = [NSString stringWithFormat:@"%@ cannot be written", file.path];
	return NFKIdentityFail(NFKServerIdentityErrorStorageFailure, reason, underlying, outError);
}

#pragma mark Renewal

- (nullable NFKServerIdentity *)renewedIdentityWithNewKey:(BOOL)newKey error:(NSError * _Nullable *)outError
{
	id privateKey = newKey ? [NFKServerIdentity generatedPrivateKeyWithError:outError] : self.privateKeyObject;
	if (privateKey == nil) {
		return nil;
	}
	NFKServerIdentity *renewed = [NFKServerIdentity identityWithPrivateKey:(__bridge SecKeyRef)privateKey
																commonName:self.commonName
												   subjectAlternativeNames:self.subjectAlternativeNames
														  validityInterval:NFKServerIdentityDefaultValidity
																	 error:outError];
	if (renewed != nil && self.directoryURL != nil && ![renewed writeToDirectoryURL:self.directoryURL error:outError]) {
		return nil;
	}
	return renewed;
}

@end
