//
//  NFKInferKitDiscovery.m
//  InferKit
//

#import "NFKInferKitDiscovery.h"
#import <dns_sd.h>
#import <arpa/inet.h>

@implementation NFKInferKitService

- (nullable NSURL *)baseURL
{
	NSURLComponents *components = [[NSURLComponents alloc] init];
	components.scheme = [self.TXTRecord[@"tls"] isEqualToString:@"1"] ? @"https" : @"http";
	components.host = self.host;
	components.port = @(self.port);
	NSString *path = self.TXTRecord[@"path"];
	components.path = [path hasPrefix:@"/"] ? path : @"/v1";
	return components.URL;
}

@end

/*! The browse and the resolves it started, all on one serial queue, which DNS-SD requires of every
	reference that shares it. */
@interface NFKInferKitBrowse : NSObject
@property (nonatomic, strong) dispatch_queue_t queue;
@property (nonatomic, assign) DNSServiceRef browseReference;
@property (nonatomic, strong) NSMutableArray<NSValue *> *resolveReferences;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKInferKitService *> *services;
@property (nonatomic, strong) NSMutableArray *contexts;
@end

@implementation NFKInferKitBrowse
@end

/*! What one resolve callback needs: the browse it reports to and the service name it resolves. */
@interface NFKInferKitResolveContext : NSObject
@property (nonatomic, weak) NFKInferKitBrowse *browse;
@property (nonatomic, copy) NSString *name;
@end

@implementation NFKInferKitResolveContext
@end

static NSDictionary<NSString *, NSString *> *NFKInferKitTXTDictionary(uint16_t length, const unsigned char *record)
{
	NSMutableDictionary<NSString *, NSString *> *entries = [NSMutableDictionary dictionary];
	uint16_t count = TXTRecordGetCount(length, record);
	for (uint16_t index = 0; index < count; index++) {
		char key[256];
		uint8_t valueLength = 0;
		const void *value = NULL;
		if (TXTRecordGetItemAtIndex(length, record, index, sizeof(key), key, &valueLength, &value) != kDNSServiceErr_NoError) {
			continue;
		}
		NSString *text = value != NULL ? [[NSString alloc] initWithBytes:value length:valueLength encoding:NSUTF8StringEncoding] : @"";
		entries[@(key)] = text ?: @"";
	}
	return entries;
}

static void DNSSD_API NFKInferKitResolveReply(DNSServiceRef reference, DNSServiceFlags flags, uint32_t interfaceIndex,
											  DNSServiceErrorType error, const char *fullName, const char *hostTarget,
											  uint16_t port, uint16_t TXTLength, const unsigned char *TXTRecord, void *context)
{
	NFKInferKitResolveContext *resolve = (__bridge NFKInferKitResolveContext *)context;
	NFKInferKitBrowse *browse = resolve.browse;
	if (error != kDNSServiceErr_NoError || browse == nil || hostTarget == NULL) {
		return;
	}
	NFKInferKitService *service = [[NFKInferKitService alloc] init];
	service.name = resolve.name;
	NSString *host = @(hostTarget);
	service.host = [host hasSuffix:@"."] ? [host substringToIndex:host.length - 1] : host;
	service.port = ntohs(port);
	service.TXTRecord = NFKInferKitTXTDictionary(TXTLength, TXTRecord);
	browse.services[resolve.name] = service;
}

static void DNSSD_API NFKInferKitBrowseReply(DNSServiceRef reference, DNSServiceFlags flags, uint32_t interfaceIndex,
											 DNSServiceErrorType error, const char *serviceName, const char *serviceType,
											 const char *domain, void *context)
{
	NFKInferKitBrowse *browse = (__bridge NFKInferKitBrowse *)context;
	if (error != kDNSServiceErr_NoError || (flags & kDNSServiceFlagsAdd) == 0) {
		return;
	}
	NFKInferKitResolveContext *resolve = [[NFKInferKitResolveContext alloc] init];
	resolve.browse = browse;
	resolve.name = @(serviceName);
	[browse.contexts addObject:resolve];
	DNSServiceRef resolveReference = NULL;
	if (DNSServiceResolve(&resolveReference, 0, interfaceIndex, serviceName, serviceType, domain,
						  NFKInferKitResolveReply, (__bridge void *)resolve) != kDNSServiceErr_NoError) {
		return;
	}
	if (DNSServiceSetDispatchQueue(resolveReference, browse.queue) != kDNSServiceErr_NoError) {
		DNSServiceRefDeallocate(resolveReference);
		return;
	}
	[browse.resolveReferences addObject:[NSValue valueWithPointer:resolveReference]];
}

NSArray<NFKInferKitService *> *NFKDiscoverInferKitServices(NSString *serviceType, NSTimeInterval timeout)
{
	NFKInferKitBrowse *browse = [[NFKInferKitBrowse alloc] init];
	dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
	browse.queue = dispatch_queue_create("com.inferkit.discovery", attributes);
	browse.resolveReferences = [NSMutableArray array];
	browse.services = [NSMutableDictionary dictionary];
	browse.contexts = [NSMutableArray array];

	__block BOOL browsing = NO;
	dispatch_sync(browse.queue, ^{
		DNSServiceRef reference = NULL;
		if (DNSServiceBrowse(&reference, 0, kDNSServiceInterfaceIndexAny, serviceType.UTF8String, NULL,
							 NFKInferKitBrowseReply, (__bridge void *)browse) != kDNSServiceErr_NoError) {
			return;
		}
		if (DNSServiceSetDispatchQueue(reference, browse.queue) != kDNSServiceErr_NoError) {
			DNSServiceRefDeallocate(reference);
			return;
		}
		browse.browseReference = reference;
		browsing = YES;
	});
	if (!browsing) {
		return @[];
	}
	[NSThread sleepForTimeInterval:MAX(timeout, 0.1)];

	__block NSArray<NFKInferKitService *> *found = nil;
	dispatch_sync(browse.queue, ^{
		DNSServiceRefDeallocate(browse.browseReference);
		for (NSValue *reference in browse.resolveReferences) {
			DNSServiceRefDeallocate((DNSServiceRef)reference.pointerValue);
		}
		[browse.resolveReferences removeAllObjects];
		found = [browse.services.allValues sortedArrayUsingComparator:^NSComparisonResult(NFKInferKitService *left, NFKInferKitService *right) {
			return [left.name localizedStandardCompare:right.name];
		}];
	});
	return found;
}
