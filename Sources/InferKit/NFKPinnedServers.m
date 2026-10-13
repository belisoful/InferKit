//
//  NFKPinnedServers.m
//  InferKit
//

#import "NFKPinnedServersPrivate.h"
#import "NFKRemoteTransportPrivate.h"

NSNotificationName const NFKPinnedServersDidChangeNotification = @"NFKPinnedServersDidChangeNotification";
NSNotificationName const NFKPinnedServersFingerprintChangedNotification = @"NFKPinnedServersFingerprintChangedNotification";
NSString * const NFKPinnedServersServiceNameKey = @"NFKPinnedServersServiceName";
NSString * const NFKPinnedServersRememberedFingerprintKey = @"NFKPinnedServersRememberedFingerprint";
NSString * const NFKPinnedServersAdvertisedFingerprintKey = @"NFKPinnedServersAdvertisedFingerprint";

/*! What reportedChanges holds for a change to plain HTTP, which advertises no fingerprint. */
static NSString * const NFKPinnedServersNoFingerprint = @"";

@interface NFKPinnedServers ()
/*! Read and written on the lock. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString *> *pins;
/*! The advertised fingerprint last reported as a change, by remembered service name. Read and
	written on the lock. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString *> *reportedChanges;
@end

@implementation NFKPinnedServers

- (instancetype)init
{
	return [self initWithDictionaryRepresentation:@{}];
}

- (instancetype)initWithDictionaryRepresentation:(NSDictionary<NSString *, NSString *> *)dictionaryRepresentation
{
	self = [super init];
	if (self != nil) {
		_pins = [NSMutableDictionary dictionary];
		_reportedChanges = [NSMutableDictionary dictionary];
		[dictionaryRepresentation enumerateKeysAndObjectsUsingBlock:^(NSString *serviceName, NSString *fingerprint, BOOL *stop) {
			NSString *normalized = [fingerprint isKindOfClass:NSString.class] ? NFKRemoteNormalizedFingerprint(fingerprint) : nil;
			if ([serviceName isKindOfClass:NSString.class] && normalized != nil) {
				self->_pins[serviceName] = normalized;
			}
		}];
	}
	return self;
}

- (NSDictionary<NSString *, NSString *> *)dictionaryRepresentation
{
	@synchronized (self) {
		return [self.pins copy];
	}
}

/*! The remembered key naming the same service, compared without regard to case. Runs on the lock. */
- (nullable NSString *)rememberedKeyForServiceName:(NSString *)serviceName
{
	if (self.pins[serviceName] != nil) {
		return serviceName;
	}
	for (NSString *key in self.pins) {
		if ([key caseInsensitiveCompare:serviceName] == NSOrderedSame) {
			return key;
		}
	}
	return nil;
}

- (nullable NSString *)fingerprintForServiceName:(NSString *)serviceName
{
	@synchronized (self) {
		NSString *key = [self rememberedKeyForServiceName:serviceName];
		return key != nil ? self.pins[key] : nil;
	}
}

- (BOOL)acceptFingerprint:(NSString *)fingerprint forServiceName:(NSString *)serviceName
{
	NSString *normalized = NFKRemoteNormalizedFingerprint(fingerprint);
	if (normalized == nil) {
		return NO;
	}
	BOOL changed = NO;
	@synchronized (self) {
		NSString *key = [self rememberedKeyForServiceName:serviceName] ?: serviceName;
		changed = ![self.pins[key] isEqualToString:normalized];
		self.pins[key] = normalized;
		[self.reportedChanges removeObjectForKey:key];
	}
	if (changed) {
		[self postDidChangeForServiceName:serviceName];
	}
	return YES;
}

- (void)forgetServiceName:(NSString *)serviceName
{
	BOOL changed = NO;
	@synchronized (self) {
		NSString *key = [self rememberedKeyForServiceName:serviceName];
		if (key != nil) {
			[self.pins removeObjectForKey:key];
			[self.reportedChanges removeObjectForKey:key];
			changed = YES;
		}
	}
	if (changed) {
		[self postDidChangeForServiceName:serviceName];
	}
}

- (NFKDiscoveredPinning)pinningForServiceName:(NSString *)serviceName
									servesTLS:(BOOL)servesTLS
						advertisedFingerprint:(nullable NSString *)advertisedFingerprint
							pinnedFingerprint:(NSString * _Nullable * _Nonnull)outPinnedFingerprint
{
	NSString *advertised = advertisedFingerprint != nil ? NFKRemoteNormalizedFingerprint(advertisedFingerprint) : nil;
	NSString *remembered = nil;
	NFKDiscoveredPinning pinning = NFKDiscoveredPinningNone;
	BOOL reportChange = NO;
	@synchronized (self) {
		NSString *key = [self rememberedKeyForServiceName:serviceName];
		remembered = key != nil ? self.pins[key] : nil;
		pinning = [self pinningWithRemembered:remembered servesTLS:servesTLS advertised:advertised];
		if (pinning == NFKDiscoveredPinningFirstUse && advertised != nil) {
			self.pins[serviceName] = advertised;
		}
		if (pinning == NFKDiscoveredPinningRemembered && key != nil) {
			[self.reportedChanges removeObjectForKey:key];
		}
		if (pinning == NFKDiscoveredPinningChanged && key != nil) {
			NSString *reported = advertised ?: NFKPinnedServersNoFingerprint;
			reportChange = ![self.reportedChanges[key] isEqualToString:reported];
			self.reportedChanges[key] = reported;
		}
	}
	*outPinnedFingerprint = pinning == NFKDiscoveredPinningFirstUse ? advertised
		: (pinning == NFKDiscoveredPinningRemembered ? remembered : nil);
	if (pinning == NFKDiscoveredPinningFirstUse) {
		[self postDidChangeForServiceName:serviceName];
	}
	if (reportChange && remembered != nil) {
		[self postFingerprintChangedForServiceName:serviceName remembered:remembered advertised:advertised];
	}
	return pinning;
}

/*! The trust-on-first-use decision. A remembered fingerprint holds while a TLS server advertises
	nothing, so removing the fingerprint from the record cannot unpin a server. */
- (NFKDiscoveredPinning)pinningWithRemembered:(nullable NSString *)remembered
									servesTLS:(BOOL)servesTLS
								   advertised:(nullable NSString *)advertised
{
	if (remembered == nil) {
		return servesTLS && advertised != nil ? NFKDiscoveredPinningFirstUse : NFKDiscoveredPinningNone;
	}
	if (!servesTLS) {
		return NFKDiscoveredPinningChanged;
	}
	if (advertised == nil || [advertised isEqualToString:remembered]) {
		return NFKDiscoveredPinningRemembered;
	}
	return NFKDiscoveredPinningChanged;
}

#pragma mark Notifications

- (void)postDidChangeForServiceName:(NSString *)serviceName
{
	[NSNotificationCenter.defaultCenter postNotificationName:NFKPinnedServersDidChangeNotification
													  object:self
													userInfo:@{ NFKPinnedServersServiceNameKey: serviceName }];
}

- (void)postFingerprintChangedForServiceName:(NSString *)serviceName
								  remembered:(NSString *)remembered
								  advertised:(nullable NSString *)advertised
{
	NSMutableDictionary<NSString *, NSString *> *userInfo = [NSMutableDictionary dictionary];
	userInfo[NFKPinnedServersServiceNameKey] = serviceName;
	userInfo[NFKPinnedServersRememberedFingerprintKey] = remembered;
	userInfo[NFKPinnedServersAdvertisedFingerprintKey] = advertised;
	[NSNotificationCenter.defaultCenter postNotificationName:NFKPinnedServersFingerprintChangedNotification
													  object:self
													userInfo:userInfo];
}

@end
