//
//  NFKHardwareProfile.m
//  InferKit
//

#import "NFKHardwareProfile.h"
#import "NFK_ARC.h"

#import <sys/sysctl.h>
#import <Metal/Metal.h>

#if TARGET_OS_OSX
#import <mach/mach.h>
#else
#import <os/proc.h>
#endif

@implementation NFKHardwareProfile
{
	NSString *_chipName;
	NSString *_modelIdentifier;
	NSString *_graphicsArchitecture;
	NSInteger _performanceCoreCount;
	NSInteger _efficiencyCoreCount;
	NSInteger _physicalMemory;
	NSInteger _recommendedWorkingSetSize;
	NSInteger _maximumBufferLength;
	BOOL _hasUnifiedMemory;
}

#pragma mark Reading the machine

/*! A sysctl string, or an empty string when the key is absent. */
static NSString *NFKSysctlString(const char *name)
{
	size_t length = 0;
	if (sysctlbyname(name, NULL, &length, NULL, 0) != 0 || length == 0) {
		return @"";
	}
	char *buffer = malloc(length);
	if (buffer == NULL) {
		return @"";
	}
	NSString *value = @"";
	if (sysctlbyname(name, buffer, &length, NULL, 0) == 0) {
		value = [NSString stringWithUTF8String:buffer] ?: @"";
	}
	free(buffer);
	return value;
}

/*! A sysctl integer, or 0 when the key is absent. The width varies by key, so both are tried. */
static NSInteger NFKSysctlInteger(const char *name)
{
	int64_t wide = 0;
	size_t length = sizeof(wide);
	if (sysctlbyname(name, &wide, &length, NULL, 0) == 0) {
		if (length == sizeof(int64_t)) {
			return (NSInteger)wide;
		}
		if (length == sizeof(int32_t)) {
			return (NSInteger)(*(int32_t *)&wide);
		}
	}
	return 0;
}

+ (NFKHardwareProfile *)currentProfile
{
	static NFKHardwareProfile *shared = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		shared = [[NFKHardwareProfile alloc] init];
	});
	return shared;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_chipName = [NFKSysctlString("machdep.cpu.brand_string") copy];
		_modelIdentifier = [NFKSysctlString("hw.model") copy];
		_physicalMemory = NFKSysctlInteger("hw.memsize");

		// Apple Silicon reports two performance levels, level 0 being the performance cores. An Intel
		// Mac reports one, and both counts stay zero rather than guessing a split that is not there.
		if (NFKSysctlInteger("hw.nperflevels") >= 2) {
			_performanceCoreCount = NFKSysctlInteger("hw.perflevel0.physicalcpu");
			_efficiencyCoreCount = NFKSysctlInteger("hw.perflevel1.physicalcpu");
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device != nil) {
			_hasUnifiedMemory = device.hasUnifiedMemory;
			_maximumBufferLength = (NSInteger)device.maxBufferLength;
			// Metal has always published this on macOS; iOS and tvOS only from 16. Below that it
			// stays zero, which is the same answer as "no Metal device" and is what the header
			// promises — a caller falls back to physicalMemory rather than reading a wrong budget.
			if (@available(macOS 10.12, iOS 16.0, tvOS 16.0, *)) {
				uint64_t recommended = device.recommendedMaxWorkingSetSize;
				_recommendedWorkingSetSize = recommended > (uint64_t)NSIntegerMax
					? NSIntegerMax : (NSInteger)recommended;
			}
			if (@available(macOS 14.0, iOS 17.0, tvOS 17.0, *)) {
				_graphicsArchitecture = [(device.architecture.name ?: @"") copy];
			} else {
				_graphicsArchitecture = [(device.name ?: @"") copy];
			}
		} else {
			_graphicsArchitecture = @"";
		}
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_chipName);
	NARC_RELEASE(_modelIdentifier);
	NARC_RELEASE(_graphicsArchitecture);
	SUPER_DEALLOC();
}

#pragma mark Static readings

- (NSString *)chipName { return _chipName; }
- (NSString *)modelIdentifier { return _modelIdentifier; }
- (NSString *)graphicsArchitecture { return _graphicsArchitecture; }
- (NSInteger)performanceCoreCount { return _performanceCoreCount; }
- (NSInteger)efficiencyCoreCount { return _efficiencyCoreCount; }
- (BOOL)hasUnifiedMemory { return _hasUnifiedMemory; }
- (NSInteger)physicalMemory { return _physicalMemory; }
- (NSInteger)recommendedWorkingSetSize { return _recommendedWorkingSetSize; }
- (NSInteger)maximumBufferLength { return _maximumBufferLength; }

#pragma mark Live readings

+ (NSInteger)availableMemory
{
#if TARGET_OS_OSX
	mach_port_t host = mach_host_self();
	vm_size_t pageSize = 0;
	if (host_page_size(host, &pageSize) != KERN_SUCCESS || pageSize == 0) {
		return 0;
	}

	vm_statistics64_data_t statistics;
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&statistics, &count) != KERN_SUCCESS) {
		return 0;
	}

	// Free pages are available outright. Inactive and purgeable pages are backed elsewhere or
	// discardable, so the kernel reclaims them under pressure rather than paging; counting them is
	// what makes this the number a large allocation can actually reach.
	uint64_t reclaimable = (uint64_t)statistics.free_count
		+ (uint64_t)statistics.inactive_count
		+ (uint64_t)statistics.purgeable_count;
	uint64_t bytes = reclaimable * (uint64_t)pageSize;
	return bytes > (uint64_t)NSIntegerMax ? NSIntegerMax : (NSInteger)bytes;
#else
	// On iOS and tvOS the system's free memory is not the ceiling that applies: a process is
	// terminated at its own allowance long before the device runs out.
	if (@available(iOS 13.0, tvOS 13.0, *)) {
		size_t remaining = os_proc_available_memory();
		return remaining > (size_t)NSIntegerMax ? NSIntegerMax : (NSInteger)remaining;
	}
	return 0;
#endif
}

#pragma mark Reporting

- (NSString *)describedMachine
{
	double gigabytes = (double)_physicalMemory / (1024.0 * 1024.0 * 1024.0);
	double budget = (double)_recommendedWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
	NSString *cores = (_performanceCoreCount > 0)
		? [NSString stringWithFormat:@"%ldP+%ldE", (long)_performanceCoreCount, (long)_efficiencyCoreCount]
		: @"cores unreported";
	return [NSString stringWithFormat:@"%@ (%@), %@, %.1f GB physical, %.1f GB recommended working set",
			_chipName.length > 0 ? _chipName : @"unknown chip",
			_modelIdentifier.length > 0 ? _modelIdentifier : @"unknown model",
			cores, gigabytes, budget];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@>", NSStringFromClass([self class]), [self describedMachine]];
}

@end
