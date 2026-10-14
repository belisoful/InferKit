//
//  NFKPinnedServersTests.m
//  InferKitTests
//
//  Discovered servers are made from Bonjour advertisements given directly, through the same
//  provider factory discovery uses, so no test browses the network.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKPinTestName = @"Studio";
static NSString * const NFKPinTestHost = @"Studio.local";

@interface NFKPinnedServersTests : XCTestCase
@end

@implementation NFKPinnedServersTests

#pragma mark Helpers

- (NSString *)fingerprintRepeating:(NSString *)pair
{
	return [@"" stringByPaddingToLength:64 withString:pair startingAtIndex:0];
}

- (NSString *)firstFingerprint
{
	return [self fingerprintRepeating:@"ab"];
}

- (NSString *)secondFingerprint
{
	return [self fingerprintRepeating:@"cd"];
}

- (NSString *)colonSeparatedFirstFingerprint
{
	NSMutableArray<NSString *> *pairs = [NSMutableArray array];
	for (NSUInteger index = 0; index < 32; index++) {
		[pairs addObject:@"AB"];
	}
	return [pairs componentsJoinedByString:@":"];
}

/*! The provider discovery makes for an advertisement of name on the test host. */
- (nullable NFKRemoteProvider *)providerNamed:(NSString *)name
										 port:(NSInteger)port
										  TLS:(BOOL)TLS
								  fingerprint:(nullable NSString *)fingerprint
								pinnedServers:(nullable NFKPinnedServers *)pinnedServers
{
	NSMutableDictionary<NSString *, NSString *> *record = [NSMutableDictionary dictionaryWithDictionary:@{ @"path": @"/v1", @"auth": @"1" }];
	record[@"tls"] = TLS ? @"1" : nil;
	record[@"cert_sha256"] = fingerprint;
	return [NFKRemoteProvider providerForInferKitServiceNamed:name host:NFKPinTestHost port:port TXTRecord:record pinnedServers:pinnedServers];
}

- (nullable NFKRemoteProvider *)TLSProviderWithFingerprint:(nullable NSString *)fingerprint pinnedServers:(nullable NFKPinnedServers *)pinnedServers
{
	return [self providerNamed:NFKPinTestName port:11480 TLS:YES fingerprint:fingerprint pinnedServers:pinnedServers];
}

/*! The notifications of one name a set posts from now until the test ends. */
- (NSMutableArray<NSNotification *> *)notificationsNamed:(NSNotificationName)name postedBy:(NFKPinnedServers *)pinnedServers
{
	NSMutableArray<NSNotification *> *received = [NSMutableArray array];
	id observer = [NSNotificationCenter.defaultCenter addObserverForName:name object:pinnedServers queue:nil usingBlock:^(NSNotification *notification) {
		[received addObject:notification];
	}];
	[self addTeardownBlock:^{
		[NSNotificationCenter.defaultCenter removeObserver:observer];
	}];
	return received;
}

#pragma mark Discovery

- (void)testWithoutPinnedServersADiscoveredTLSServerIsNotPinned
{
	XCTAssertNil(NFKRemoteProvider.discoveryPinnedServers, @"automatic pinning is off by default");
	NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:nil];
	XCTAssertEqualObjects(provider.baseURL.absoluteString, @"https://Studio.local:11480/v1");
	XCTAssertEqualObjects(provider.displayName, NFKPinTestName);
	XCTAssertTrue(provider.requiresAPIKey);
	XCTAssertNil(provider.pinnedCertificateSHA256);
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningNone);
	XCTAssertEqualObjects(provider.advertisedProperties[@"cert_sha256"], self.firstFingerprint);
}

- (void)testTheFirstFingerprintSeenIsPinnedAndRemembered
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	NSMutableArray<NSNotification *> *changes = [self notificationsNamed:NFKPinnedServersDidChangeNotification postedBy:pinnedServers];

	NFKRemoteProvider *first = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(first.discoveredPinning, NFKDiscoveredPinningFirstUse);
	XCTAssertEqualObjects(first.pinnedCertificateSHA256, self.firstFingerprint);
	XCTAssertEqualObjects(pinnedServers.dictionaryRepresentation, (@{ NFKPinTestName: self.firstFingerprint }));
	XCTAssertEqual(changes.count, 1u);
	XCTAssertEqualObjects(changes.firstObject.userInfo[NFKPinnedServersServiceNameKey], NFKPinTestName);

	NFKRemoteProvider *again = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(again.discoveredPinning, NFKDiscoveredPinningRemembered);
	XCTAssertEqualObjects(again.pinnedCertificateSHA256, self.firstFingerprint);
	XCTAssertEqual(changes.count, 1u, @"seeing the remembered fingerprint again changes nothing");
}

- (void)testAColonSeparatedUppercaseFingerprintIsRememberedInLowercaseDigits
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	NSMutableArray<NSString *> *pairs = [NSMutableArray array];
	for (NSUInteger index = 0; index < 32; index++) {
		[pairs addObject:@"AB"];
	}
	NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:[pairs componentsJoinedByString:@":"] pinnedServers:pinnedServers];
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningFirstUse);
	XCTAssertEqualObjects(provider.pinnedCertificateSHA256, self.firstFingerprint);
	XCTAssertEqualObjects([pinnedServers fingerprintForServiceName:NFKPinTestName], self.firstFingerprint);
}

- (void)testAChangedFingerprintIsNotPinnedAndIsReportedOnce
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NSMutableArray<NSNotification *> *reports = [self notificationsNamed:NFKPinnedServersFingerprintChangedNotification postedBy:pinnedServers];
	NSMutableArray<NSNotification *> *changes = [self notificationsNamed:NFKPinnedServersDidChangeNotification postedBy:pinnedServers];

	NFKRemoteProvider *changed = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(changed.discoveredPinning, NFKDiscoveredPinningChanged);
	XCTAssertNil(changed.pinnedCertificateSHA256);
	XCTAssertEqualObjects(changed.advertisedProperties[@"cert_sha256"], self.secondFingerprint);
	XCTAssertEqualObjects([pinnedServers fingerprintForServiceName:NFKPinTestName], self.firstFingerprint, @"the remembered fingerprint is kept");
	XCTAssertEqual(reports.count, 1u);
	XCTAssertEqualObjects(reports.firstObject.userInfo[NFKPinnedServersServiceNameKey], NFKPinTestName);
	XCTAssertEqualObjects(reports.firstObject.userInfo[NFKPinnedServersRememberedFingerprintKey], self.firstFingerprint);
	XCTAssertEqualObjects(reports.firstObject.userInfo[NFKPinnedServersAdvertisedFingerprintKey], self.secondFingerprint);

	NFKRemoteProvider *stillChanged = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(stillChanged.discoveredPinning, NFKDiscoveredPinningChanged);
	XCTAssertEqual(reports.count, 1u, @"the same change is reported once");
	XCTAssertEqual(changes.count, 0u);
}

- (void)testAcceptingTheNewFingerprintPinsItOnTheNextDiscovery
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	[self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	NSMutableArray<NSNotification *> *changes = [self notificationsNamed:NFKPinnedServersDidChangeNotification postedBy:pinnedServers];

	XCTAssertTrue([pinnedServers acceptFingerprint:self.secondFingerprint forServiceName:NFKPinTestName]);
	XCTAssertEqual(changes.count, 1u);
	NFKRemoteProvider *accepted = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(accepted.discoveredPinning, NFKDiscoveredPinningRemembered);
	XCTAssertEqualObjects(accepted.pinnedCertificateSHA256, self.secondFingerprint);
	XCTAssertEqualObjects(pinnedServers.dictionaryRepresentation, (@{ NFKPinTestName: self.secondFingerprint }));
}

- (void)testAcceptingAMalformedFingerprintChangesNothing
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	XCTAssertFalse([pinnedServers acceptFingerprint:@"not a fingerprint" forServiceName:NFKPinTestName]);
	XCTAssertEqualObjects([pinnedServers fingerprintForServiceName:NFKPinTestName], self.firstFingerprint);
}

- (void)testAForgottenServerIsTrustedOnFirstUseAgain
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NSMutableArray<NSNotification *> *changes = [self notificationsNamed:NFKPinnedServersDidChangeNotification postedBy:pinnedServers];

	[pinnedServers forgetServiceName:NFKPinTestName];
	XCTAssertEqualObjects(pinnedServers.dictionaryRepresentation, @{});
	XCTAssertEqual(changes.count, 1u);
	[pinnedServers forgetServiceName:NFKPinTestName];
	XCTAssertEqual(changes.count, 1u, @"forgetting an unknown name changes nothing");

	NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningFirstUse);
	XCTAssertEqualObjects(provider.pinnedCertificateSHA256, self.secondFingerprint);
}

- (void)testARestoredSetPinsARememberedServerWithoutAFirstUse
{
	NFKPinnedServers *saved = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:saved];
	NSMutableDictionary<NSString *, NSString *> *stored = [saved.dictionaryRepresentation mutableCopy];
	stored[@"Broken"] = @"zz";

	NFKPinnedServers *restored = [[NFKPinnedServers alloc] initWithDictionaryRepresentation:stored];
	XCTAssertEqualObjects(restored.dictionaryRepresentation, (@{ NFKPinTestName: self.firstFingerprint }), @"a malformed entry is skipped");
	NSMutableArray<NSNotification *> *changes = [self notificationsNamed:NFKPinnedServersDidChangeNotification postedBy:restored];

	NFKRemoteProvider *remembered = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:restored];
	XCTAssertEqual(remembered.discoveredPinning, NFKDiscoveredPinningRemembered);
	XCTAssertEqualObjects(remembered.pinnedCertificateSHA256, self.firstFingerprint);
	XCTAssertEqual(changes.count, 0u);

	NFKRemoteProvider *changed = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:restored];
	XCTAssertEqual(changed.discoveredPinning, NFKDiscoveredPinningChanged);
	XCTAssertNil(changed.pinnedCertificateSHA256);
}

- (void)testATLSServerWithoutAWellFormedFingerprintIsNotPinned
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	for (NSString *fingerprint in @[ [NSNull null], @"", @"abc", [self fingerprintRepeating:@"zz"] ]) {
		NSString *advertised = [fingerprint isKindOfClass:NSString.class] ? fingerprint : nil;
		NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:advertised pinnedServers:pinnedServers];
		XCTAssertEqualObjects(provider.baseURL.scheme, @"https");
		XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningNone, @"%@", advertised);
		XCTAssertNil(provider.pinnedCertificateSHA256, @"%@", advertised);
	}
	XCTAssertEqualObjects(pinnedServers.dictionaryRepresentation, @{});
}

- (void)testARememberedServerAdvertisingNoFingerprintStaysPinned
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:nil pinnedServers:pinnedServers];
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningRemembered);
	XCTAssertEqualObjects(provider.pinnedCertificateSHA256, self.firstFingerprint);
}

- (void)testAPlainHTTPServerIsUnaffected
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	NFKRemoteProvider *provider = [self providerNamed:NFKPinTestName port:11480 TLS:NO fingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	XCTAssertEqualObjects(provider.baseURL.scheme, @"http");
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningNone);
	XCTAssertNil(provider.pinnedCertificateSHA256);
	XCTAssertEqualObjects(pinnedServers.dictionaryRepresentation, @{});
}

- (void)testARememberedServerServingPlainHTTPIsReportedAsChanged
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NSMutableArray<NSNotification *> *reports = [self notificationsNamed:NFKPinnedServersFingerprintChangedNotification postedBy:pinnedServers];

	NFKRemoteProvider *provider = [self providerNamed:NFKPinTestName port:11480 TLS:NO fingerprint:nil pinnedServers:pinnedServers];
	XCTAssertEqual(provider.discoveredPinning, NFKDiscoveredPinningChanged);
	XCTAssertNil(provider.pinnedCertificateSHA256);
	XCTAssertEqual(reports.count, 1u);
	XCTAssertNil(reports.firstObject.userInfo[NFKPinnedServersAdvertisedFingerprintKey]);
}

- (void)testTheServiceNameKeysThePinWithoutRegardToCaseHostOrPort
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	[self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];

	NFKRemoteProvider *moved = [self providerNamed:@"studio" port:54321 TLS:YES fingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(moved.discoveredPinning, NFKDiscoveredPinningRemembered, @"a new port is the same server");
	NFKRemoteProvider *impostor = [self providerNamed:@"STUDIO" port:11480 TLS:YES fingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(impostor.discoveredPinning, NFKDiscoveredPinningChanged);
	NFKRemoteProvider *other = [self providerNamed:@"Laptop" port:11480 TLS:YES fingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	XCTAssertEqual(other.discoveredPinning, NFKDiscoveredPinningFirstUse);
	XCTAssertEqual(pinnedServers.dictionaryRepresentation.count, 2u);
}

- (void)testAnAdvertisementWithoutAHostOrPortMakesNoProvider
{
	XCTAssertNil([NFKRemoteProvider providerForInferKitServiceNamed:NFKPinTestName host:@"" port:11480 TXTRecord:@{} pinnedServers:nil]);
	XCTAssertNil([NFKRemoteProvider providerForInferKitServiceNamed:NFKPinTestName host:NFKPinTestHost port:0 TXTRecord:@{} pinnedServers:nil]);
	XCTAssertNil([NFKRemoteProvider providerForInferKitServiceNamed:NFKPinTestName host:NFKPinTestHost port:70000 TXTRecord:@{} pinnedServers:nil]);
}

- (void)testADerivedProviderReportsNoDiscoveredPinning
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	NFKRemoteProvider *provider = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NFKRemoteProvider *derived = [provider providerWithPinnedCertificateSHA256:self.secondFingerprint];
	XCTAssertEqual(derived.discoveredPinning, NFKDiscoveredPinningNone);
	XCTAssertEqualObjects(derived.pinnedCertificateSHA256, self.secondFingerprint);
}

#pragma mark Balancer

- (void)testABalancerPinsADiscoveredServerAndRemovesItWhenItsFingerprintChanges
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] init];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"model"];
	NFKRemoteProvider *pinned = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:pinnedServers];
	NFKRemoteProvider *plain = [self providerNamed:@"Laptop" port:11480 TLS:NO fingerprint:nil pinnedServers:pinnedServers];

	[balancer mergeDiscoveredProviders:@[ pinned, plain ]];
	XCTAssertEqualObjects([NSSet setWithArray:balancer.serverBaseURLs], ([NSSet setWithObjects:pinned.baseURL, plain.baseURL, nil]));
	XCTAssertEqualObjects([balancer pinnedCertificateSHA256ForServerWithBaseURL:pinned.baseURL], self.firstFingerprint);
	XCTAssertNil([balancer pinnedCertificateSHA256ForServerWithBaseURL:plain.baseURL]);

	NFKRemoteProvider *changed = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	[balancer mergeDiscoveredProviders:@[ changed, plain ]];
	XCTAssertEqualObjects(balancer.serverBaseURLs, (@[ plain.baseURL ]), @"a server whose fingerprint changed is skipped");
}

- (void)testABalancerSkipsAChangedServerUntilItsNewFingerprintIsAccepted
{
	NFKPinnedServers *pinnedServers = [[NFKPinnedServers alloc] initWithDictionaryRepresentation:@{ NFKPinTestName: self.firstFingerprint }];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"model"];
	[balancer mergeDiscoveredProviders:@[ [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers] ]];
	XCTAssertEqualObjects(balancer.serverBaseURLs, @[]);

	[pinnedServers acceptFingerprint:self.secondFingerprint forServiceName:NFKPinTestName];
	NFKRemoteProvider *accepted = [self TLSProviderWithFingerprint:self.secondFingerprint pinnedServers:pinnedServers];
	[balancer mergeDiscoveredProviders:@[ accepted ]];
	XCTAssertEqualObjects(balancer.serverBaseURLs, (@[ accepted.baseURL ]));
	XCTAssertEqualObjects([balancer pinnedCertificateSHA256ForServerWithBaseURL:accepted.baseURL], self.secondFingerprint);
}

- (void)testABalancerFallsBackToItsFleetPinForAServerAutomaticPinningLeftUnpinned
{
	NSString *fleet = [self fingerprintRepeating:@"ef"];
	NFKBalancedBackend *balancer = [NFKBalancedBackend backendWithModelName:@"model"];
	balancer.pinnedCertificateSHA256 = fleet;
	NFKRemoteProvider *unpinned = [self TLSProviderWithFingerprint:self.firstFingerprint pinnedServers:nil];
	NFKRemoteProvider *bare = [self providerNamed:@"Laptop" port:11481 TLS:YES fingerprint:nil pinnedServers:[[NFKPinnedServers alloc] init]];

	[balancer mergeDiscoveredProviders:@[ unpinned, bare ]];
	XCTAssertEqualObjects([balancer pinnedCertificateSHA256ForServerWithBaseURL:unpinned.baseURL], fleet);
	XCTAssertEqualObjects([balancer pinnedCertificateSHA256ForServerWithBaseURL:bare.baseURL], fleet);
	XCTAssertNil([balancer pinnedCertificateSHA256ForServerWithBaseURL:[NSURL URLWithString:@"https://elsewhere.local:1/v1"]]);
}

#pragma mark Provider equality

- (void)testEachSpellingOfOnePinMakesEqualProviders
{
	NFKRemoteProvider *lowercase = [NFKRemoteProvider.inferKit providerWithPinnedCertificateSHA256:self.firstFingerprint];
	NFKRemoteProvider *colons = [NFKRemoteProvider.inferKit providerWithPinnedCertificateSHA256:self.colonSeparatedFirstFingerprint];
	XCTAssertEqualObjects(lowercase, colons);
	XCTAssertEqual(lowercase.hash, colons.hash);
	XCTAssertNotEqualObjects(lowercase, [NFKRemoteProvider.inferKit providerWithPinnedCertificateSHA256:self.secondFingerprint]);
	XCTAssertNotEqualObjects(lowercase, NFKRemoteProvider.inferKit);
}

#pragma mark Per-class factories

/*! Builds a client from the preset pinned with the colon-separated spelling of firstFingerprint and from
	the bare preset, and checks that only the first sends through a pinned session. */
- (void)assertClientSessionFor:(NFKRemoteProvider *)preset
					   honorsPin:(NSURLSession * _Nullable (^)(NFKRemoteProvider *provider))sessionOfClient
{
	NFKRemoteProvider *pinned = [preset providerWithPinnedCertificateSHA256:self.colonSeparatedFirstFingerprint];
	NSURLSession *session = sessionOfClient(pinned);
	XCTAssertNotNil(session, @"%@ builds no client", preset.identifier);
	if (session != nil) {
		XCTAssertEqualObjects([NFKRemoteTransport pinnedCertificateSHA256ForSession:session], self.firstFingerprint, @"%@", preset.identifier);
	}
	NSURLSession *unpinned = sessionOfClient(preset);
	if (unpinned != nil) {
		XCTAssertNil([NFKRemoteTransport pinnedCertificateSHA256ForSession:unpinned], @"%@", preset.identifier);
	}
}

- (void)testTheTranscriptionFactoryHonorsThePin
{
	for (NFKRemoteProvider *preset in @[ NFKRemoteProvider.inferKit, NFKRemoteProvider.openAI, NFKRemoteProvider.mistral, NFKRemoteProvider.xAI ]) {
		[self assertClientSessionFor:preset honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
			return [NFKRemoteTranscriptionBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
		}];
	}
}

- (void)testTheSpeechFactoryHonorsThePin
{
	for (NFKRemoteProvider *preset in @[ NFKRemoteProvider.inferKit, NFKRemoteProvider.mistral, NFKRemoteProvider.xAI ]) {
		[self assertClientSessionFor:preset honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
			return [NFKRemoteSpeechBackend backendForProvider:provider apiKey:nil modelName:@"m" voice:@"v"].session;
		}];
	}
}

- (void)testTheImageFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.inferKit honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteImageBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheEmbeddingFactoryHonorsThePin
{
	for (NFKRemoteProvider *preset in @[ NFKRemoteProvider.inferKit, NFKRemoteProvider.googleGemini ]) {
		[self assertClientSessionFor:preset honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
			return [NFKRemoteEmbeddingBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
		}];
	}
}

- (void)testTheCompletionFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.vLLM honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteCompletionBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheResponsesFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteResponsesBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheGenerateContentFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.googleGemini honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKGeminiGenerateContentBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheDecisionsFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKOpenAIDecisionsBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheModerationFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.mistral honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteModerationBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheOCRFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.mistral honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteOCRBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheClassifierFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.vLLM honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteClassifierBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheVideoFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteVideoBackend backendForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheRerankerFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.vLLM honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteReranker rerankerForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheTokenCounterFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.llamaCpp honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteTokenCounter counterForProvider:provider apiKey:nil modelName:@"m"].session;
	}];
}

- (void)testTheVoiceLibraryFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.googleGemini honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKGeminiVoiceLibrary voiceLibraryForProvider:provider apiKey:nil].session;
	}];
}

- (void)testTheUsageReporterFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteUsageReporter reporterForProvider:provider apiKey:nil].session;
	}];
}

- (void)testTheFileStoreFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteFileStore fileStoreForProvider:provider apiKey:nil].session;
	}];
}

- (void)testTheRetrievalStoreFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRemoteRetrievalStore retrievalStoreForProvider:provider apiKey:nil].session;
	}];
}

- (void)testTheLocalRunnersHonorThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.ollama honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKOllamaRunner runnerWithProvider:provider].session;
	}];
	[self assertClientSessionFor:NFKRemoteProvider.lmStudio honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKLMStudioRunner runnerWithProvider:provider].session;
	}];
}

- (void)testTheRealtimeSessionFactoryHonorsThePin
{
	[self assertClientSessionFor:NFKRemoteProvider.openAI honorsPin:^NSURLSession *(NFKRemoteProvider *provider) {
		return [NFKRealtimeSession sessionForProvider:provider apiStyle:NFKRealtimeAPIStyleOpenAIConversation apiKey:nil modelName:@"m"].URLSession;
	}];
	XCTAssertNil([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.openAI apiStyle:NFKRealtimeAPIStyleOpenAIConversation apiKey:nil modelName:@"m"].URLSession);
}

- (void)testAPinnedSocketRefusesAPlainWebSocketURLUnopened
{
	NSURLSession *pinned = [NFKRemoteTransport sessionWithPinnedCertificateSHA256:self.firstFingerprint];
	NSURLRequest *request = [NSURLRequest requestWithURL:[NSURL URLWithString:@"ws://127.0.0.1:9/v1/realtime"]];
	NFKRealtimeWebSocket *socket = [[NFKRealtimeWebSocket alloc] initWithRequest:request session:pinned];
	XCTestExpectation *closed = [self expectationWithDescription:@"closed"];
	__block NSError *closeError = nil;
	[socket openWithMessageHandler:^(NSString *text, NSData *data) {
		XCTFail(@"a refused socket receives nothing");
	} closeHandler:^(NSError *error) {
		closeError = error;
		[closed fulfill];
	}];
	[self waitForExpectations:@[ closed ] timeout:5.0];
	XCTAssertEqual(closeError.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([closeError.localizedDescription containsString:@"wss://"], @"%@", closeError);
}

@end
