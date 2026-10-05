//
//  NFKServedHostStatus.h
//  InferKit
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKServedHostStatus
	@abstract   Reads the serving machine's state for the status route: memory, thermal state, load,
				GPU utilization, and storage.
	@discussion CPU usage is the share of time the processors were busy since the previous reading,
				so the monitor keeps that reading; the first is measured from the monitor's creation.
				Memory pressure comes from the kernel's level where the process can read it, and from
				a memory-pressure dispatch source otherwise. GPU utilization is the IORegistry's
				undocumented "Device Utilization %" on macOS, absent where it cannot be read and on
				every other platform.
*/
@interface NFKServedHostStatus : NSObject

/*! The host's state as a JSON object. storageURL names the directory whose volume the storage figures
	describe; nil uses the app's caches directory. */
- (NSDictionary<NSString *, id> *)JSONObjectForStorageURL:(nullable NSURL *)storageURL;

/*! The thermal state's name: nominal, fair, serious, or critical. */
+ (NSString *)thermalStateName;

/*! The static facts a Bonjour TXT record carries: the chip and the physical memory in bytes. */
+ (NSDictionary<NSString *, NSString *> *)TXTRecordEntries;

@end

NS_ASSUME_NONNULL_END
