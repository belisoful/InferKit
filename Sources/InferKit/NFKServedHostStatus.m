//
//  NFKServedHostStatus.m
//  InferKit
//

#import "NFKServedHostStatus.h"
#import <InferKit/NFKHardwareProfile.h>
#import <TargetConditionals.h>
#import <mach/mach.h>
#import <stdlib.h>
#import <sys/sysctl.h>
#if TARGET_OS_OSX
#import <IOKit/IOKitLib.h>
#endif

/*! Busy and total processor ticks summed over every core since boot. */
typedef struct {
	uint64_t busy;
	uint64_t total;
} NFKServedProcessorTicks;

static BOOL NFKServedReadProcessorTicks(NFKServedProcessorTicks *ticks)
{
	natural_t processorCount = 0;
	processor_info_array_t info = NULL;
	mach_msg_type_number_t infoCount = 0;
	static mach_port_t host = MACH_PORT_NULL;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		host = mach_host_self();
	});
	if (host_processor_info(host, PROCESSOR_CPU_LOAD_INFO, &processorCount, &info, &infoCount) != KERN_SUCCESS) {
		return NO;
	}
	processor_cpu_load_info_t loads = (processor_cpu_load_info_t)(void *)info;
	NFKServedProcessorTicks sum = { 0, 0 };
	for (natural_t processor = 0; processor < processorCount; processor++) {
		const unsigned int *state = loads[processor].cpu_ticks;
		uint64_t busy = (uint64_t)state[CPU_STATE_USER] + state[CPU_STATE_SYSTEM] + state[CPU_STATE_NICE];
		sum.busy += busy;
		sum.total += busy + state[CPU_STATE_IDLE];
	}
	vm_deallocate(mach_task_self(), (vm_address_t)info, (vm_size_t)infoCount * sizeof(integer_t));
	*ticks = sum;
	return YES;
}

/*! The kernel's memory-pressure level (1 normal, 2 warning, 4 critical), or 0 when it cannot be read. */
static int NFKServedKernelPressureLevel(void)
{
	int level = 0;
	size_t size = sizeof(level);
	if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, NULL, 0) != 0) {
		return 0;
	}
	return level;
}

static NSString *NFKServedPressureName(unsigned long level)
{
	if (level & DISPATCH_MEMORYPRESSURE_CRITICAL) {
		return @"critical";
	}
	if (level & DISPATCH_MEMORYPRESSURE_WARN) {
		return @"warning";
	}
	return @"normal";
}

static NSNumber * _Nullable NFKServedProcessFootprint(void)
{
	task_vm_info_data_t info;
	mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
		return nil;
	}
	return @(info.phys_footprint);
}

/*! The busiest GPU's utilization as a fraction, or nil where the IORegistry does not report one. */
static NSNumber * _Nullable NFKServedGPUUtilization(void)
{
#if TARGET_OS_OSX
	io_iterator_t iterator = IO_OBJECT_NULL;
	// MACH_PORT_NULL selects the default main port on every macOS the core supports.
	if (IOServiceGetMatchingServices(MACH_PORT_NULL, IOServiceMatching("IOAccelerator"), &iterator) != KERN_SUCCESS) {
		return nil;
	}
	NSNumber *busiest = nil;
	for (io_registry_entry_t entry = IOIteratorNext(iterator); entry != IO_OBJECT_NULL; entry = IOIteratorNext(iterator)) {
		CFTypeRef statistics = IORegistryEntryCreateCFProperty(entry, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
		IOObjectRelease(entry);
		NSDictionary *values = (__bridge_transfer NSDictionary *)statistics;
		NSNumber *percent = [values isKindOfClass:NSDictionary.class] ? values[@"Device Utilization %"] : nil;
		if ([percent isKindOfClass:NSNumber.class] && (busiest == nil || percent.doubleValue > busiest.doubleValue)) {
			busiest = percent;
		}
	}
	IOObjectRelease(iterator);
	return busiest != nil ? @(MIN(MAX(busiest.doubleValue / 100.0, 0.0), 1.0)) : nil;
#else
	return nil;
#endif
}

@interface NFKServedHostStatus ()
@property (nonatomic, assign) NFKServedProcessorTicks lastTicks;
@property (nonatomic, assign) BOOL hasTicks;
@property (nonatomic, strong, nullable) dispatch_source_t pressureSource;
@property (atomic, assign) unsigned long observedPressure;
@end

@implementation NFKServedHostStatus

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		NFKServedProcessorTicks ticks;
		_hasTicks = NFKServedReadProcessorTicks(&ticks);
		_lastTicks = _hasTicks ? ticks : (NFKServedProcessorTicks){ 0, 0 };
		_observedPressure = DISPATCH_MEMORYPRESSURE_NORMAL;
		unsigned long mask = DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL;
		dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, mask,
														  dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
		__weak NFKServedHostStatus *weakSelf = self;
		dispatch_source_set_event_handler(source, ^{
			weakSelf.observedPressure = dispatch_source_get_data(source);
		});
		dispatch_resume(source);
		_pressureSource = source;
	}
	return self;
}

- (void)dealloc
{
	if (_pressureSource != nil) {
		dispatch_source_cancel(_pressureSource);
	}
}

+ (NSString *)thermalStateName
{
	switch (NSProcessInfo.processInfo.thermalState) {
		case NSProcessInfoThermalStateNominal:
			return @"nominal";
		case NSProcessInfoThermalStateFair:
			return @"fair";
		case NSProcessInfoThermalStateSerious:
			return @"serious";
		case NSProcessInfoThermalStateCritical:
			return @"critical";
	}
	return @"nominal";
}

+ (NSDictionary<NSString *, NSString *> *)TXTRecordEntries
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	return @{ @"chip": profile.chipName, @"memory": [NSString stringWithFormat:@"%ld", (long)profile.physicalMemory] };
}

/*! The share of processor time spent busy since the previous call, or nil when it cannot be measured. */
- (nullable NSNumber *)processorUsage
{
	NFKServedProcessorTicks now;
	if (!NFKServedReadProcessorTicks(&now)) {
		return nil;
	}
	@synchronized (self) {
		NFKServedProcessorTicks before = self.lastTicks;
		BOOL comparable = self.hasTicks && now.total > before.total;
		if (!comparable) {
			self.lastTicks = now;
			self.hasTicks = YES;
			return nil;
		}
		self.lastTicks = now;
		return @((double)(now.busy - before.busy) / (double)(now.total - before.total));
	}
}

- (NSString *)memoryPressureName
{
	int level = NFKServedKernelPressureLevel();
	if (level == 0) {
		return NFKServedPressureName(self.observedPressure);
	}
	return NFKServedPressureName((unsigned long)level);
}

- (NSDictionary<NSString *, id> *)memoryJSONObject
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	NSMutableDictionary<NSString *, id> *memory = [NSMutableDictionary dictionary];
	memory[@"physical_bytes"] = @(profile.physicalMemory);
	NSInteger available = NFKHardwareProfile.availableMemory;
	if (available > 0) {
		memory[@"available_bytes"] = @(available);
	}
	if (profile.recommendedWorkingSetSize > 0) {
		memory[@"recommended_working_set_bytes"] = @(profile.recommendedWorkingSetSize);
	}
	memory[@"pressure"] = [self memoryPressureName];
	memory[@"process_footprint_bytes"] = NFKServedProcessFootprint();
	return memory;
}

- (NSDictionary<NSString *, id> *)processorJSONObject
{
	NSMutableDictionary<NSString *, id> *processor = [NSMutableDictionary dictionary];
	processor[@"usage"] = [self processorUsage];
	double averages[3];
	if (getloadavg(averages, 3) == 3) {
		processor[@"load_average"] = @[ @(averages[0]), @(averages[1]), @(averages[2]) ];
	}
	return processor;
}

- (nullable NSDictionary<NSString *, id> *)storageJSONObjectForURL:(nullable NSURL *)storageURL
{
#if TARGET_OS_TV
	// tvOS has no important-usage reading; its plain available capacity is the nearest one.
	NSURLResourceKey availableKey = NSURLVolumeAvailableCapacityKey;
#else
	NSURLResourceKey availableKey = NSURLVolumeAvailableCapacityForImportantUsageKey;
#endif
	NSURL *directory = storageURL ?: [NSFileManager.defaultManager URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask].firstObject;
	NSDictionary<NSURLResourceKey, id> *values = [directory resourceValuesForKeys:@[ availableKey, NSURLVolumeTotalCapacityKey ] error:NULL];
	if (values[availableKey] == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *storage = [NSMutableDictionary dictionary];
	storage[@"available_bytes"] = values[availableKey];
	storage[@"total_bytes"] = values[NSURLVolumeTotalCapacityKey];
	return storage;
}

- (NSDictionary<NSString *, id> *)JSONObjectForStorageURL:(nullable NSURL *)storageURL
{
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	NSMutableDictionary<NSString *, id> *host = [NSMutableDictionary dictionary];
	host[@"chip"] = profile.chipName;
	host[@"model_identifier"] = profile.modelIdentifier;
	host[@"performance_cores"] = @(profile.performanceCoreCount);
	host[@"efficiency_cores"] = @(profile.efficiencyCoreCount);
	host[@"thermal_state"] = [NFKServedHostStatus thermalStateName];
	if (@available(macOS 12.0, iOS 9.0, tvOS 9.0, *)) {
		host[@"low_power_mode"] = @(NSProcessInfo.processInfo.lowPowerModeEnabled);
	}
	host[@"memory"] = [self memoryJSONObject];
	host[@"cpu"] = [self processorJSONObject];
	NSNumber *utilization = NFKServedGPUUtilization();
	if (utilization != nil) {
		host[@"gpu"] = @{ @"utilization": utilization, @"source": @"ioregistry" };
	}
	host[@"storage"] = [self storageJSONObjectForURL:storageURL];
	return host;
}

@end
