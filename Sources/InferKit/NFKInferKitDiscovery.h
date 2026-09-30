//
//  NFKInferKitDiscovery.h
//  InferKit
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! One resolved Bonjour advertisement of an InferKit server. */
@interface NFKInferKitService : NSObject
@property (nonatomic, copy) NSString *name;
/*! The advertising machine's host name, without the trailing dot (for example Studio.local). */
@property (nonatomic, copy) NSString *host;
@property (nonatomic, assign) uint16_t port;
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *TXTRecord;
/*! scheme://host:port plus the advertised path. */
@property (nonatomic, readonly, nullable) NSURL *baseURL;
@end

/*! Browses the service type for the timeout and returns every advertisement resolved in that time,
	one per service name, sorted by name. Blocks for the timeout. */
NSArray<NFKInferKitService *> *NFKDiscoverInferKitServices(NSString *serviceType, NSTimeInterval timeout);

NS_ASSUME_NONNULL_END
