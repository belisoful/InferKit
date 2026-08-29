//
//  NFKHardwareProfile.h
//  InferKit
//

#ifndef NFKHardwareProfile_h
#define NFKHardwareProfile_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKHardwareProfile
	@abstract   What the machine is and how much of it is left, for deciding whether a model fits
				before loading it.
	@discussion Loading a model that does not fit does not fail politely: the process is killed, or the
				system pages until the run is useless. Both are avoidable by asking first, and asking
				needs two numbers — what the machine has, and what is still free.

				Every reading degrades rather than throwing. An unknown chip reports an empty name and
				zero counts, not an error, because a profile is used to decide and a decision with a
				missing field is still better than a crash on a machine this was never run on.

				The static facts are read once and cached; availableMemory is live and is read on
				every call.

				Introduced in InferKit 0.1.0.
*/
@interface NFKHardwareProfile : NSObject

/*! The shared profile. The static readings are taken once, on first use. */
@property (class, nonatomic, readonly) NFKHardwareProfile *currentProfile;

#pragma mark What the machine is

/*! The chip, as the system names it — "Apple M1 Max". Empty when it cannot be read. */
@property (nonatomic, readonly, copy) NSString *chipName;

/*! The model identifier — "MacBookPro18,2". Empty when it cannot be read. */
@property (nonatomic, readonly, copy) NSString *modelIdentifier;

/*! The GPU architecture Metal reports. Empty when there is no Metal device. */
@property (nonatomic, readonly, copy) NSString *graphicsArchitecture;

/*! Performance cores, or 0 when the system does not report a split. */
@property (nonatomic, readonly) NSInteger performanceCoreCount;

/*! Efficiency cores, or 0 when the system does not report a split. */
@property (nonatomic, readonly) NSInteger efficiencyCoreCount;

/*! Whether the CPU and GPU share one pool, which is what makes the whole question one number. */
@property (nonatomic, readonly) BOOL hasUnifiedMemory;

#pragma mark What it has

/*! Physical memory in bytes. */
@property (nonatomic, readonly) NSInteger physicalMemory;

/*!
	@property   recommendedWorkingSetSize
	@abstract   Metal's recommended resident budget in bytes, or 0 with no Metal device.
	@discussion This is what a model should be sized against, not physicalMemory. It sits well below
				the physical total, and staying inside it is what keeps the system from evicting.
*/
@property (nonatomic, readonly) NSInteger recommendedWorkingSetSize;

/*! The largest single buffer Metal will allocate, or 0. A tensor cannot exceed this however much
	memory is free, which is a separate ceiling from the total. */
@property (nonatomic, readonly) NSInteger maximumBufferLength;

/*!
	@method     availableMemory
	@abstract   Bytes that could still be allocated right now, or 0 when it cannot be determined.
	@discussion LIVE — it is read on each call and changes as other processes run, which is the point:
				a fit decision made against the physical total is a decision made against a number that
				was never available.

				On macOS this counts the free, inactive and purgeable pages the kernel reports, since
				all three are reclaimable under pressure. On iOS and tvOS it is the process's own
				remaining allowance before the system terminates it, which is the ceiling that
				actually applies there.
*/
+ (NSInteger)availableMemory;

/*! A one-line summary, for a log or a bug report. */
@property (nonatomic, readonly, copy) NSString *describedMachine;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKHardwareProfile_h */
