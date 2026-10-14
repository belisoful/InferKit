//
//  NFKServerIdentityTests.m
//  InferKitTests
//
//  The identity tests create keys in memory and store them in a temporary directory; the TLS tests
//  start a real HTTPS server on a free loopback port and reach it with the core's own clients.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <CommonCrypto/CommonDigest.h>
#import <sys/stat.h>

/*! A backend that answers every request with "ok". */
@interface NFKIdentityTestBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKIdentityTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"identity-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

@interface NFKServerIdentityTests : XCTestCase
@property (nonatomic, assign) NSUInteger savedRetryAttempts;
@end

@implementation NFKServerIdentityTests

- (void)setUp
{
	[super setUp];
	self.savedRetryAttempts = NFKRemoteTransport.retryAttempts;
	NFKRemoteTransport.retryAttempts = 0;
}

- (void)tearDown
{
	NFKRemoteTransport.retryAttempts = self.savedRetryAttempts;
	[super tearDown];
}

#pragma mark Helpers

/*! A directory path that does not exist yet, removed when the test ends. */
- (NSURL *)temporaryDirectory
{
	NSURL *parent = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
	NSURL *directory = [parent URLByAppendingPathComponent:[@"NFKServerIdentityTests-" stringByAppendingString:NSUUID.UUID.UUIDString] isDirectory:YES];
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];
	}];
	return [directory URLByAppendingPathComponent:@"identity" isDirectory:YES];
}

- (NFKServerIdentity *)createdIdentity
{
	NSError *error = nil;
	NFKServerIdentity *identity = [NFKServerIdentity createIdentityWithCommonName:nil subjectAlternativeNames:nil validityInterval:0 error:&error];
	XCTAssertNotNil(identity, @"%@", error);
	return identity;
}

- (mode_t)modeOfFileAtURL:(NSURL *)url
{
	struct stat status;
	XCTAssertEqual(stat(url.fileSystemRepresentation, &status), 0);
	return status.st_mode & 0777;
}

- (NSString *)SHA256OfData:(NSData *)data
{
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
	NSMutableString *hex = [NSMutableString string];
	for (NSUInteger index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) {
		[hex appendFormat:@"%02x", digest[index]];
	}
	return hex;
}

- (NFKInferenceServer *)startedServerWithIdentity:(NFKServerIdentity *)identity
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.TLSIdentity = identity.identity;
	[server addBackend:[[NFKIdentityTestBackend alloc] init] forModelName:@"echo"];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NFKRemoteInferKitBackend *)clientFor:(NFKInferenceServer *)server pinnedTo:(nullable NSString *)fingerprint
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	client.timeout = 10.0;
	client.pinnedCertificateSHA256 = fingerprint;
	return client;
}

- (NFKInferenceRequest *)request
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }];
}

- (nullable NFKInferenceJob *)finishedJob:(NFKInferenceJob *)job
{
	XCTestExpectation *finished = [self expectationWithDescription:@"the job finishes"];
	job.completionHandler = ^(NFKInferenceJob *done) {
		[finished fulfill];
	};
	[self waitForExpectations:@[ finished ] timeout:15.0];
	return job;
}

- (void)assertRefusedCertificate:(nullable NSError *)error
{
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqualObjects(underlying.domain, NSURLErrorDomain);
	XCTAssertEqual(underlying.code, NSURLErrorServerCertificateUntrusted);
}

#pragma mark The certificate

- (void)testACreatedCertificateParsesAndNamesItsSubjectAndHosts
{
	NSError *error = nil;
	NFKServerIdentity *identity = [NFKServerIdentity createIdentityWithCommonName:@"studio"
														 subjectAlternativeNames:@[ @"studio.local", @"192.168.1.20", @"fe80::1" ]
																validityInterval:86400.0
																		   error:&error];
	XCTAssertNotNil(identity, @"%@", error);
	id certificate = CFBridgingRelease(SecCertificateCreateWithData(NULL, (__bridge CFDataRef)identity.certificateData));
	XCTAssertNotNil(certificate);
	NSString *summary = CFBridgingRelease(SecCertificateCopySubjectSummary((__bridge SecCertificateRef)certificate));
	XCTAssertEqualObjects(summary, @"studio");
	XCTAssertEqualObjects(identity.commonName, @"studio");
	NSArray<NSString *> *names = @[ @"studio.local", @"192.168.1.20", @"fe80::1" ];
	XCTAssertEqualObjects(identity.subjectAlternativeNames, names);
	XCTAssertEqualWithAccuracy([identity.notAfter timeIntervalSinceDate:identity.notBefore], 86400.0, 1.0);
	XCTAssertTrue(identity.identity != NULL);
	XCTAssertTrue(identity.certificate != NULL);
	XCTAssertTrue([identity.PEMEncodedCertificate hasPrefix:@"-----BEGIN CERTIFICATE-----\n"]);
}

- (void)testTheDefaultCertificateCoversThisMachinesLoopbackNames
{
	NFKServerIdentity *identity = [self createdIdentity];
	XCTAssertEqualObjects(identity.commonName, @"InferKit");
	NSArray<NSString *> *names = @[ @"localhost", @"127.0.0.1", @"::1" ];
	XCTAssertEqualObjects(identity.subjectAlternativeNames, names);
	XCTAssertEqualWithAccuracy([identity.notAfter timeIntervalSinceDate:identity.notBefore], NFKServerIdentityDefaultValidity, 1.0);
	XCTAssertEqualWithAccuracy(identity.notBefore.timeIntervalSinceNow, -3600.0, 5.0, @"the certificate is valid from an hour before now");
}

- (void)testTheCertificateIsTrustedAsItsOwnAnchorForTLSService
{
	NFKServerIdentity *identity = [self createdIdentity];
	SecPolicyRef policy = SecPolicyCreateSSL(true, CFSTR("localhost"));
	SecTrustRef trust = NULL;
	XCTAssertEqual(SecTrustCreateWithCertificates(identity.certificate, policy, &trust), errSecSuccess);
	SecTrustSetAnchorCertificates(trust, (__bridge CFArrayRef)@[ (__bridge id)identity.certificate ]);
	CFErrorRef error = NULL;
	BOOL trusted = SecTrustEvaluateWithError(trust, &error);
	XCTAssertTrue(trusted, @"%@", (__bridge NSError *)error);
	if (error != NULL) {
		CFRelease(error);
	}
	CFRelease(trust);
	CFRelease(policy);
}

- (void)testTheFingerprintIsTheSHA256OfTheDER
{
	NFKServerIdentity *identity = [self createdIdentity];
	NSString *expected = [self SHA256OfData:identity.certificateData];
	XCTAssertEqualObjects(identity.SHA256Fingerprint, expected);
	XCTAssertEqualObjects([NFKServerIdentity SHA256FingerprintOfCertificateData:identity.certificateData], expected);
	NSString *colons = identity.colonSeparatedSHA256Fingerprint;
	XCTAssertEqual(colons.length, 95u);
	XCTAssertEqualObjects([colons stringByReplacingOccurrencesOfString:@":" withString:@""], expected.uppercaseString);
}

- (void)testANameThatIsNeitherAnAddressNorAHostIsRefused
{
	NSError *error = nil;
	NFKServerIdentity *identity = [NFKServerIdentity createIdentityWithCommonName:nil subjectAlternativeNames:@[ @"two words" ] validityInterval:0 error:&error];
	XCTAssertNil(identity);
	XCTAssertEqualObjects(error.domain, NFKServerIdentityErrorDomain);
	XCTAssertEqual(error.code, NFKServerIdentityErrorInvalidName);
}

#if TARGET_OS_OSX
- (void)testOpenSSLReadsTheCertificateAndTheKey
{
	NSString *openssl = @"/usr/bin/openssl";
	if (![NSFileManager.defaultManager isExecutableFileAtPath:openssl]) {
		XCTSkip(@"no openssl at /usr/bin/openssl");
	}
	NSURL *directory = [self temporaryDirectory];
	NSError *error = nil;
	NFKServerIdentity *identity = [NFKServerIdentity identityAtDirectoryURL:directory commonName:@"probe" subjectAlternativeNames:nil error:&error];
	XCTAssertNotNil(identity, @"%@", error);
	NSArray<NSArray<NSString *> *> *commands = @[
		@[ @"x509", @"-noout", @"-text", @"-in", [directory URLByAppendingPathComponent:@"certificate.pem"].path ],
		@[ @"ec", @"-noout", @"-in", [directory URLByAppendingPathComponent:@"identity.pem"].path ],
	];
	for (NSArray<NSString *> *arguments in commands) {
		NSTask *task = [[NSTask alloc] init];
		task.executableURL = [NSURL fileURLWithPath:openssl];
		task.arguments = arguments;
		NSPipe *output = [NSPipe pipe];
		task.standardOutput = output;
		task.standardError = output;
		XCTAssertTrue([task launchAndReturnError:&error], @"%@", error);
		NSData *printed = [output.fileHandleForReading readDataToEndOfFile];
		[task waitUntilExit];
		NSString *text = [[NSString alloc] initWithData:printed encoding:NSUTF8StringEncoding];
		XCTAssertEqual(task.terminationStatus, 0, @"%@", text);
		if ([arguments.firstObject isEqualToString:@"x509"]) {
			XCTAssertTrue([text containsString:@"CN=probe"] || [text containsString:@"CN = probe"], @"%@", text);
			XCTAssertTrue([text containsString:@"DNS:localhost"], @"%@", text);
			XCTAssertTrue([text containsString:@"TLS Web Server Authentication"], @"%@", text);
		}
	}
}
#endif

#pragma mark Storage

- (void)testAStoredIdentityReloadsWithTheSameFingerprint
{
	NSURL *directory = [self temporaryDirectory];
	NSError *error = nil;
	NFKServerIdentity *created = [NFKServerIdentity identityAtDirectoryURL:directory commonName:@"first" subjectAlternativeNames:nil error:&error];
	XCTAssertNotNil(created, @"%@", error);
	XCTAssertEqualObjects(created.directoryURL, directory);

	NFKServerIdentity *reloaded = [NFKServerIdentity identityAtDirectoryURL:directory commonName:@"second" subjectAlternativeNames:@[ @"other.local" ] error:&error];
	XCTAssertNotNil(reloaded, @"%@", error);
	XCTAssertEqualObjects(reloaded.SHA256Fingerprint, created.SHA256Fingerprint);
	XCTAssertEqualObjects(reloaded.commonName, @"first", @"a stored identity keeps its names");
	XCTAssertEqualObjects(reloaded.certificateData, created.certificateData);
	XCTAssertEqualObjects(reloaded.notAfter, created.notAfter);

	NFKServerIdentity *loaded = [NFKServerIdentity loadIdentityAtDirectoryURL:directory error:&error];
	XCTAssertEqualObjects(loaded.SHA256Fingerprint, created.SHA256Fingerprint);
	NSString *certificatePEM = [NSString stringWithContentsOfURL:[directory URLByAppendingPathComponent:@"certificate.pem"] encoding:NSUTF8StringEncoding error:NULL];
	XCTAssertEqualObjects(certificatePEM, created.PEMEncodedCertificate);
}

- (void)testTheKeyFileIsReadableOnlyByItsOwner
{
	NSURL *directory = [self temporaryDirectory];
	NSError *error = nil;
	XCTAssertNotNil([NFKServerIdentity identityAtDirectoryURL:directory commonName:nil subjectAlternativeNames:nil error:&error], @"%@", error);
	XCTAssertEqual([self modeOfFileAtURL:[directory URLByAppendingPathComponent:@"identity.pem"]], (mode_t)0600);
	XCTAssertEqual([self modeOfFileAtURL:[directory URLByAppendingPathComponent:@"certificate.pem"]], (mode_t)0644);
	XCTAssertEqual([self modeOfFileAtURL:directory], (mode_t)0700);
}

- (void)testLoadingNarrowsALooserKeyFileMode
{
	NSURL *directory = [self temporaryDirectory];
	NSError *error = nil;
	XCTAssertNotNil([NFKServerIdentity identityAtDirectoryURL:directory commonName:nil subjectAlternativeNames:nil error:&error], @"%@", error);
	NSURL *keyFile = [directory URLByAppendingPathComponent:@"identity.pem"];
	XCTAssertEqual(chmod(keyFile.fileSystemRepresentation, 0644), 0);
	XCTAssertNotNil([NFKServerIdentity loadIdentityAtDirectoryURL:directory error:&error], @"%@", error);
	XCTAssertEqual([self modeOfFileAtURL:keyFile], (mode_t)0600);
}

- (void)testAMalformedStoredIdentityFailsAndIsKept
{
	NSURL *directory = [self temporaryDirectory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSURL *keyFile = [directory URLByAppendingPathComponent:@"identity.pem"];
	NSData *garbage = [@"not a key" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertTrue([garbage writeToURL:keyFile atomically:YES]);
	NSError *error = nil;
	XCTAssertNil([NFKServerIdentity identityAtDirectoryURL:directory commonName:nil subjectAlternativeNames:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKServerIdentityErrorDomain);
	XCTAssertEqual(error.code, NFKServerIdentityErrorInvalidStoredIdentity);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:keyFile], garbage, @"a stored identity is never replaced by loading");
}

- (void)testLoadingAnEmptyDirectoryFails
{
	NSError *error = nil;
	XCTAssertNil([NFKServerIdentity loadIdentityAtDirectoryURL:[self temporaryDirectory] error:&error]);
	XCTAssertEqual(error.code, NFKServerIdentityErrorStorageFailure);
}

- (void)testAnIdentityRoundTripsThroughItsPEMText
{
	NSURL *directory = [self temporaryDirectory];
	NFKServerIdentity *identity = [self createdIdentity];
	NSError *error = nil;
	XCTAssertTrue([identity writeToDirectoryURL:directory error:&error], @"%@", error);
	NSData *pem = [NSData dataWithContentsOfURL:[directory URLByAppendingPathComponent:@"identity.pem"]];
	NFKServerIdentity *parsed = [NFKServerIdentity identityWithPEMData:pem error:&error];
	XCTAssertEqualObjects(parsed.SHA256Fingerprint, identity.SHA256Fingerprint, @"%@", error);
}

- (void)testRenewalReissuesTheCertificateAndReplacesTheStoredOne
{
	NSURL *directory = [self temporaryDirectory];
	NSError *error = nil;
	NFKServerIdentity *original = [NFKServerIdentity identityAtDirectoryURL:directory commonName:@"renewed" subjectAlternativeNames:@[ @"studio.local" ] error:&error];
	XCTAssertNotNil(original, @"%@", error);

	NFKServerIdentity *sameKey = [original renewedIdentityWithNewKey:NO error:&error];
	XCTAssertNotNil(sameKey, @"%@", error);
	XCTAssertNotEqualObjects(sameKey.SHA256Fingerprint, original.SHA256Fingerprint);
	XCTAssertEqualObjects(sameKey.commonName, @"renewed");
	NSArray<NSString *> *names = @[ @"studio.local" ];
	XCTAssertEqualObjects(sameKey.subjectAlternativeNames, names);
	id originalKey = CFBridgingRelease(SecCertificateCopyKey(original.certificate));
	id sameKeyKey = CFBridgingRelease(SecCertificateCopyKey(sameKey.certificate));
	NSData *originalPoint = CFBridgingRelease(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)originalKey, NULL));
	NSData *sameKeyPoint = CFBridgingRelease(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)sameKeyKey, NULL));
	XCTAssertEqualObjects(sameKeyPoint, originalPoint);
	XCTAssertEqualObjects([NFKServerIdentity loadIdentityAtDirectoryURL:directory error:NULL].SHA256Fingerprint, sameKey.SHA256Fingerprint);

	NFKServerIdentity *newKey = [sameKey renewedIdentityWithNewKey:YES error:&error];
	XCTAssertNotNil(newKey, @"%@", error);
	id newKeyKey = CFBridgingRelease(SecCertificateCopyKey(newKey.certificate));
	NSData *newKeyPoint = CFBridgingRelease(SecKeyCopyExternalRepresentation((__bridge SecKeyRef)newKeyKey, NULL));
	XCTAssertNotEqualObjects(newKeyPoint, originalPoint);
	XCTAssertEqualObjects([NFKServerIdentity loadIdentityAtDirectoryURL:directory error:NULL].SHA256Fingerprint, newKey.SHA256Fingerprint);
}

#pragma mark Pinned sessions

- (void)testPinnedSessionsAreSharedPerFingerprintInEitherForm
{
	NFKServerIdentity *identity = [self createdIdentity];
	NSURLSession *hex = [NFKRemoteTransport sessionWithPinnedCertificateSHA256:identity.SHA256Fingerprint];
	NSURLSession *colons = [NFKRemoteTransport sessionWithPinnedCertificateSHA256:identity.colonSeparatedSHA256Fingerprint];
	XCTAssertEqual(hex, colons);
	XCTAssertEqualObjects([NFKRemoteTransport pinnedCertificateSHA256ForSession:colons], identity.SHA256Fingerprint);
	XCTAssertNil([NFKRemoteTransport pinnedCertificateSHA256ForSession:NSURLSession.sharedSession]);

	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:[NSURL URLWithString:@"https://localhost:1/v1"]];
	XCTAssertNil(client.pinnedCertificateSHA256);
	client.pinnedCertificateSHA256 = identity.colonSeparatedSHA256Fingerprint;
	XCTAssertEqualObjects(client.pinnedCertificateSHA256, identity.SHA256Fingerprint);
	client.pinnedCertificateSHA256 = nil;
	XCTAssertEqual(client.session, NSURLSession.sharedSession);
}

#pragma mark TLS

- (void)testAPinnedClientReachesTheServer
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	XCTAssertEqualObjects(server.localBaseURL.scheme, @"https");
	NSError *error = nil;
	NFKInferenceResult *result = [[self clientFor:server pinnedTo:identity.SHA256Fingerprint] runInferenceForRequest:self.request error:&error];
	XCTAssertEqualObjects(result.outputs[NFKOutputText], @"ok", @"%@", error);
	NFKServerStatus *status = [[self clientFor:server pinnedTo:identity.colonSeparatedSHA256Fingerprint] fetchServerStatusWithError:&error];
	XCTAssertNotNil([status modelNamed:@"echo"], @"%@", error);
}

- (void)testAClientPinnedToAnotherCertificateIsRefused
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKServerIdentity *other = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	NSError *error = nil;
	XCTAssertNotNil([[self clientFor:server pinnedTo:identity.SHA256Fingerprint] runInferenceForRequest:self.request error:&error], @"%@", error);
	error = nil;
	XCTAssertNil([[self clientFor:server pinnedTo:other.SHA256Fingerprint] runInferenceForRequest:self.request error:&error],
				 @"a connection the matching pin made earlier does not carry over");
	[self assertRefusedCertificate:error];
}

- (void)testAnUnpinnedClientRefusesTheSelfSignedServer
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	NSError *error = nil;
	XCTAssertNil([[self clientFor:server pinnedTo:nil] runInferenceForRequest:self.request error:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqualObjects(underlying.domain, NSURLErrorDomain);
	XCTAssertLessThanOrEqual(underlying.code, NSURLErrorServerCertificateHasBadDate);
	XCTAssertGreaterThanOrEqual(underlying.code, NSURLErrorClientCertificateRequired);
}

- (void)testAPinnedStreamReachesTheServerAndAnotherPinIsRefused
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKServerIdentity *other = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	NFKInferenceJob *pinned = [self finishedJob:[[self clientFor:server pinnedTo:identity.SHA256Fingerprint] submitInferenceJobForRequest:self.request]];
	XCTAssertEqualObjects(pinned.result.outputs[NFKOutputText], @"ok", @"%@", pinned.error);
	NFKInferenceJob *refused = [self finishedJob:[[self clientFor:server pinnedTo:other.SHA256Fingerprint] submitInferenceJobForRequest:self.request]];
	XCTAssertNil(refused.result);
	[self assertRefusedCertificate:refused.error];
}

- (void)testAPinnedClientSendsNothingOverPlainHTTP
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1"]];
	client.pinnedCertificateSHA256 = identity.SHA256Fingerprint;
	NSError *error = nil;
	XCTAssertNil([client runInferenceForRequest:self.request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"HTTPS"], @"%@", error);
}

- (void)testAMalformedPinFailsEveryRequest
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	NSError *error = nil;
	XCTAssertNil([[self clientFor:server pinnedTo:@"not-a-fingerprint"] runInferenceForRequest:self.request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"not-a-fingerprint"], @"%@", error);
}

- (void)testAPinnedProviderBuildsPinnedBackendsAndCatalogs
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];
	NFKRemoteProvider *located = [NFKRemoteProvider.inferKit providerWithBaseURL:server.localBaseURL];
	NFKRemoteProvider *provider = [located providerWithPinnedCertificateSHA256:identity.SHA256Fingerprint];
	XCTAssertEqualObjects(provider.pinnedCertificateSHA256, identity.SHA256Fingerprint);
	XCTAssertNotEqualObjects(provider, located);
	XCTAssertNil([provider providerWithBaseURL:server.localBaseURL].pinnedCertificateSHA256);

	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider apiKey:nil modelName:@"echo"];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:self.request error:&error];
	XCTAssertEqualObjects(result.outputs[NFKOutputText], @"ok", @"%@", error);
	NSArray<NFKRemoteModel *> *models = [provider modelsWithAPIKey:nil error:&error];
	XCTAssertEqual(models.count, 1u, @"%@", error);
	XCTAssertNil([located modelsWithAPIKey:nil error:NULL], @"the unpinned provider still refuses the self-signed server");
}

- (void)testABalancerReachesAPinnedServerAndSkipsOneItCannotTrust
{
	NFKServerIdentity *identity = [self createdIdentity];
	NFKServerIdentity *other = [self createdIdentity];
	NFKInferenceServer *server = [self startedServerWithIdentity:identity];

	NFKBalancedBackend *pinned = [NFKBalancedBackend backendWithModelName:@"echo"];
	[pinned addServerWithBaseURL:server.localBaseURL apiKey:nil pinnedCertificateSHA256:identity.SHA256Fingerprint];
	NSError *error = nil;
	NFKInferenceResult *result = [pinned runInferenceForRequest:self.request error:&error];
	XCTAssertEqualObjects(result.outputs[NFKOutputText], @"ok", @"%@", error);

	NFKBalancedBackend *fleet = [NFKBalancedBackend backendWithModelName:@"echo"];
	fleet.pinnedCertificateSHA256 = identity.colonSeparatedSHA256Fingerprint;
	[fleet addServerWithBaseURL:server.localBaseURL apiKey:nil];
	XCTAssertNotNil([fleet runInferenceForRequest:self.request error:&error], @"%@", error);

	NFKBalancedBackend *mismatched = [NFKBalancedBackend backendWithModelName:@"echo"];
	[mismatched addServerWithBaseURL:server.localBaseURL apiKey:nil pinnedCertificateSHA256:other.SHA256Fingerprint];
	XCTAssertNil([mismatched runInferenceForRequest:self.request error:NULL]);
}

@end
