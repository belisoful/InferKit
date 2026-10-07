//
//  NFKServerStatus.m
//  InferKit
//

#import <InferKit/NFKServerStatus.h>
#import <InferKit/NFKRemoteInferKitBackend.h>

static NSDictionary * _Nullable NFKStatusDictionary(id _Nullable value)
{
	return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

static NSNumber * _Nullable NFKStatusNumber(id _Nullable value)
{
	return [value isKindOfClass:NSNumber.class] && isfinite([value doubleValue]) ? value : nil;
}

static NSString *NFKStatusString(id _Nullable value)
{
	return [value isKindOfClass:NSString.class] ? value : @"";
}

static NSInteger NFKStatusInteger(id _Nullable value)
{
	return NFKStatusNumber(value).integerValue;
}

static NFKServerThermalState NFKStatusThermalState(id _Nullable name)
{
	NSDictionary<NSString *, NSNumber *> *states = @{ @"nominal": @(NFKServerThermalStateNominal),
													  @"fair": @(NFKServerThermalStateFair),
													  @"serious": @(NFKServerThermalStateSerious),
													  @"critical": @(NFKServerThermalStateCritical) };
	return (NFKServerThermalState)states[NFKStatusString(name)].integerValue;
}

@interface NFKServerLoad ()
@property (nonatomic, readwrite) NSInteger limit;
@property (nonatomic, readwrite) NSInteger running;
@property (nonatomic, readwrite) NSInteger queued;
@property (nonatomic, readwrite, copy, nullable) NSNumber *estimatedWaitSeconds;
@property (nonatomic, readwrite) NFKServerThermalState thermalState;
@property (nonatomic, readwrite, copy) NSDate *date;
@end

@implementation NFKServerLoad

+ (nullable instancetype)loadWithHTTPHeaders:(NSDictionary *)headers
{
	NSMutableDictionary<NSString *, NSString *> *fields = [NSMutableDictionary dictionary];
	[headers enumerateKeysAndObjectsUsingBlock:^(id name, id value, BOOL *stop) {
		if ([name isKindOfClass:NSString.class] && [value isKindOfClass:NSString.class]) {
			fields[[name lowercaseString]] = value;
		}
	}];
	NSString *limit = fields[@"x-inferkit-limit"];
	if (limit.integerValue <= 0) {
		return nil;
	}
	NFKServerLoad *load = [[self alloc] init];
	load.limit = limit.integerValue;
	load.running = fields[@"x-inferkit-running"].integerValue;
	load.queued = fields[@"x-inferkit-queued"].integerValue;
	NSString *wait = fields[@"x-inferkit-estimated-wait"];
	load.estimatedWaitSeconds = wait.length > 0 ? @(wait.doubleValue) : nil;
	load.thermalState = NFKStatusThermalState(fields[@"x-inferkit-thermal-state"]);
	load.date = [NSDate date];
	return load;
}

@end

@interface NFKServerRunStatus ()
@property (nonatomic, readwrite) NSTimeInterval elapsedSeconds;
@property (nonatomic, readwrite, copy, nullable) NSNumber *progress;
@property (nonatomic, readwrite, copy, nullable) NSNumber *estimatedRemainingSeconds;
@end

@implementation NFKServerRunStatus
@end

@interface NFKServerModelStatus ()
@property (nonatomic, readwrite, copy) NSString *name;
@property (nonatomic, readwrite, copy) NSString *backendIdentifier;
@property (nonatomic, readwrite, getter=isReady) BOOL ready;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *modelInfo;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *backendStatus;
@property (nonatomic, readwrite) NSInteger limit;
@property (nonatomic, readwrite) NSInteger running;
@property (nonatomic, readwrite) NSInteger queued;
@property (nonatomic, readwrite) NSInteger queueLimit;
@property (nonatomic, readwrite) NSInteger completed;
@property (nonatomic, readwrite) NSInteger failed;
@property (nonatomic, readwrite) NSInteger cancelled;
@property (nonatomic, readwrite) NSInteger refused;
@property (nonatomic, readwrite, copy, nullable) NSNumber *averageRunSeconds;
@property (nonatomic, readwrite, copy, nullable) NSNumber *averageWaitSeconds;
@property (nonatomic, readwrite, copy, nullable) NSNumber *outputTokensPerSecond;
@property (nonatomic, readwrite) NSInteger inputTokens;
@property (nonatomic, readwrite) NSInteger cachedInputTokens;
@property (nonatomic, readwrite, copy, nullable) NSNumber *cachedInputShare;
@property (nonatomic, readwrite, copy, nullable) NSNumber *estimatedWaitSeconds;
@property (nonatomic, readwrite, copy) NSArray<NFKServerRunStatus *> *runs;
@end

@implementation NFKServerModelStatus

+ (instancetype)statusWithJSONObject:(NSDictionary *)entry
{
	NFKServerModelStatus *status = [[self alloc] init];
	status.name = NFKStatusString(entry[@"id"]);
	status.backendIdentifier = NFKStatusString(entry[@"backend"]);
	status.ready = [NFKStatusNumber(entry[@"ready"]) boolValue];
	status.modelInfo = NFKStatusDictionary(entry[@"model"]) ?: @{};
	status.backendStatus = NFKStatusDictionary(entry[@"status"]) ?: @{};
	NSDictionary *load = NFKStatusDictionary(entry[@"load"]) ?: @{};
	status.limit = NFKStatusInteger(load[@"limit"]);
	status.running = NFKStatusInteger(load[@"running"]);
	status.queued = NFKStatusInteger(load[@"queued"]);
	status.queueLimit = NFKStatusInteger(load[@"queue_limit"]);
	status.completed = NFKStatusInteger(load[@"completed"]);
	status.failed = NFKStatusInteger(load[@"failed"]);
	status.cancelled = NFKStatusInteger(load[@"cancelled"]);
	status.refused = NFKStatusInteger(load[@"refused"]);
	status.averageRunSeconds = NFKStatusNumber(load[@"average_run_seconds"]);
	status.averageWaitSeconds = NFKStatusNumber(load[@"average_wait_seconds"]);
	status.outputTokensPerSecond = NFKStatusNumber(load[@"output_tokens_per_second"]);
	status.inputTokens = NFKStatusInteger(load[@"input_tokens"]);
	status.cachedInputTokens = NFKStatusInteger(load[@"cached_input_tokens"]);
	status.cachedInputShare = NFKStatusNumber(load[@"cached_input_share"]);
	status.estimatedWaitSeconds = NFKStatusNumber(load[@"estimated_wait_seconds"]);
	NSMutableArray<NFKServerRunStatus *> *runs = [NSMutableArray array];
	for (id value in [load[@"runs"] isKindOfClass:NSArray.class] ? load[@"runs"] : @[]) {
		NSDictionary *entryRun = NFKStatusDictionary(value);
		if (entryRun == nil) {
			continue;
		}
		NFKServerRunStatus *run = [[NFKServerRunStatus alloc] init];
		run.elapsedSeconds = NFKStatusNumber(entryRun[@"elapsed_seconds"]).doubleValue;
		run.progress = NFKStatusNumber(entryRun[@"progress"]);
		run.estimatedRemainingSeconds = NFKStatusNumber(entryRun[@"estimated_remaining_seconds"]);
		[runs addObject:run];
	}
	status.runs = runs;
	return status;
}

@end

@interface NFKServerHostStatus ()
@property (nonatomic, readwrite, copy) NSString *chipName;
@property (nonatomic, readwrite, copy) NSString *modelIdentifier;
@property (nonatomic, readwrite) NSInteger performanceCoreCount;
@property (nonatomic, readwrite) NSInteger efficiencyCoreCount;
@property (nonatomic, readwrite) NFKServerThermalState thermalState;
@property (nonatomic, readwrite, getter=isLowPowerModeEnabled) BOOL lowPowerModeEnabled;
@property (nonatomic, readwrite) NSInteger physicalMemory;
@property (nonatomic, readwrite, copy, nullable) NSNumber *availableMemory;
@property (nonatomic, readwrite, copy, nullable) NSNumber *recommendedWorkingSetSize;
@property (nonatomic, readwrite) NFKServerMemoryPressure memoryPressure;
@property (nonatomic, readwrite, copy, nullable) NSNumber *processFootprint;
@property (nonatomic, readwrite, copy, nullable) NSNumber *cpuUsage;
@property (nonatomic, readwrite, copy) NSArray<NSNumber *> *loadAverages;
@property (nonatomic, readwrite, copy, nullable) NSNumber *gpuUtilization;
@property (nonatomic, readwrite, copy, nullable) NSNumber *storageAvailableBytes;
@property (nonatomic, readwrite, copy, nullable) NSNumber *storageTotalBytes;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, NSDictionary<NSString *, id> *> *runtimes;
@end

@implementation NFKServerHostStatus

+ (instancetype)statusWithJSONObject:(NSDictionary *)host
{
	NFKServerHostStatus *status = [[self alloc] init];
	status.chipName = NFKStatusString(host[@"chip"]);
	status.modelIdentifier = NFKStatusString(host[@"model_identifier"]);
	status.performanceCoreCount = NFKStatusInteger(host[@"performance_cores"]);
	status.efficiencyCoreCount = NFKStatusInteger(host[@"efficiency_cores"]);
	status.thermalState = NFKStatusThermalState(host[@"thermal_state"]);
	status.lowPowerModeEnabled = [NFKStatusNumber(host[@"low_power_mode"]) boolValue];
	NSDictionary *memory = NFKStatusDictionary(host[@"memory"]) ?: @{};
	status.physicalMemory = NFKStatusInteger(memory[@"physical_bytes"]);
	status.availableMemory = NFKStatusNumber(memory[@"available_bytes"]);
	status.recommendedWorkingSetSize = NFKStatusNumber(memory[@"recommended_working_set_bytes"]);
	NSDictionary<NSString *, NSNumber *> *pressures = @{ @"normal": @(NFKServerMemoryPressureNormal),
														 @"warning": @(NFKServerMemoryPressureWarning),
														 @"critical": @(NFKServerMemoryPressureCritical) };
	status.memoryPressure = (NFKServerMemoryPressure)pressures[NFKStatusString(memory[@"pressure"])].integerValue;
	status.processFootprint = NFKStatusNumber(memory[@"process_footprint_bytes"]);
	NSDictionary *processor = NFKStatusDictionary(host[@"cpu"]) ?: @{};
	status.cpuUsage = NFKStatusNumber(processor[@"usage"]);
	NSMutableArray<NSNumber *> *averages = [NSMutableArray array];
	for (id value in [processor[@"load_average"] isKindOfClass:NSArray.class] ? processor[@"load_average"] : @[]) {
		if (NFKStatusNumber(value) != nil) {
			[averages addObject:value];
		}
	}
	status.loadAverages = averages;
	status.gpuUtilization = NFKStatusNumber(NFKStatusDictionary(host[@"gpu"])[@"utilization"]);
	NSDictionary *storage = NFKStatusDictionary(host[@"storage"]) ?: @{};
	status.storageAvailableBytes = NFKStatusNumber(storage[@"available_bytes"]);
	status.storageTotalBytes = NFKStatusNumber(storage[@"total_bytes"]);
	NSMutableDictionary<NSString *, NSDictionary<NSString *, id> *> *runtimes = [NSMutableDictionary dictionary];
	[NFKStatusDictionary(host[@"runtimes"]) enumerateKeysAndObjectsUsingBlock:^(id name, id report, BOOL *stop) {
		if ([name isKindOfClass:NSString.class] && NFKStatusDictionary(report) != nil) {
			runtimes[name] = report;
		}
	}];
	status.runtimes = runtimes;
	return status;
}

@end

@interface NFKServerStatus ()
@property (nonatomic, readwrite, copy) NSString *version;
@property (nonatomic, readwrite, copy) NSDate *startDate;
@property (nonatomic, readwrite) NSTimeInterval uptime;
@property (nonatomic, readwrite, copy) NSArray<NFKServerModelStatus *> *models;
@property (nonatomic, readwrite, strong, nullable) NFKServerHostStatus *host;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *JSONObject;
@end

@implementation NFKServerStatus

+ (nullable instancetype)statusWithJSONObject:(id)object
{
	NSDictionary *reply = NFKStatusDictionary(object);
	if (![reply[@"object"] isEqual:@"inferkit.status"]) {
		return nil;
	}
	NFKServerStatus *status = [[self alloc] init];
	NSDictionary *server = NFKStatusDictionary(reply[@"server"]) ?: @{};
	status.version = NFKStatusString(server[@"version"]);
	status.startDate = [NSDate dateWithTimeIntervalSince1970:NFKStatusNumber(server[@"started"]).doubleValue];
	status.uptime = NFKStatusNumber(server[@"uptime_seconds"]).doubleValue;
	NSMutableArray<NFKServerModelStatus *> *models = [NSMutableArray array];
	for (id value in [reply[@"models"] isKindOfClass:NSArray.class] ? reply[@"models"] : @[]) {
		NSDictionary *entry = NFKStatusDictionary(value);
		if (entry != nil) {
			[models addObject:[NFKServerModelStatus statusWithJSONObject:entry]];
		}
	}
	status.models = models;
	NSDictionary *host = NFKStatusDictionary(reply[@"host"]);
	status.host = host != nil ? [NFKServerHostStatus statusWithJSONObject:host] : nil;
	status.JSONObject = reply;
	return status;
}

- (nullable NFKServerModelStatus *)modelNamed:(NSString *)name
{
	for (NFKServerModelStatus *model in self.models) {
		if ([model.name isEqualToString:name]) {
			return model;
		}
	}
	return nil;
}

+ (nullable instancetype)fetchFromBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey error:(NSError **)error
{
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:baseURL];
	client.apiKey = apiKey;
	return (NFKServerStatus *)[client fetchServerStatusWithError:error];
}

+ (void)fetchFromBaseURL:(NSURL *)baseURL
				  apiKey:(nullable NSString *)apiKey
	   completionHandler:(void (^)(NFKServerStatus * _Nullable status, NSError * _Nullable error))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKServerStatus *status = [self fetchFromBaseURL:baseURL apiKey:apiKey error:&error];
		completionHandler(status, status != nil ? nil : error);
	});
}

@end
