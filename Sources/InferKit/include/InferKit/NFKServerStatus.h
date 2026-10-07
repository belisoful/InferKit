//
//  NFKServerStatus.h
//  InferKit
//

#ifndef NFKServerStatus_h
#define NFKServerStatus_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKServerThermalState
	@abstract   A served machine's thermal state, as NSProcessInfo reports it there.
	@constant   NFKServerThermalStateUnknown The server did not report one.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerThermalState) {
	NFKServerThermalStateUnknown = 0,
	NFKServerThermalStateNominal,
	NFKServerThermalStateFair,
	NFKServerThermalStateSerious,
	NFKServerThermalStateCritical,
};

/*!
	@enum       NFKServerMemoryPressure
	@abstract   A served machine's memory pressure, as its kernel reports it.
	@constant   NFKServerMemoryPressureUnknown The server did not report one.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerMemoryPressure) {
	NFKServerMemoryPressureUnknown = 0,
	NFKServerMemoryPressureNormal,
	NFKServerMemoryPressureWarning,
	NFKServerMemoryPressureCritical,
};

/*!
	@class      NFKServerLoad
	@abstract   A served model's load as one reply's X-InferKit-* headers carried it.
	@discussion NFKInferenceServer writes the headers on every run reply, taken when the reply's head is
				written, so a client learns a server's load from the replies it already receives.
				Introduced in InferKit 0.4.0.
*/
@interface NFKServerLoad : NSObject

/*! How many runs the model serves at once. */
@property (nonatomic, readonly) NSInteger limit;
@property (nonatomic, readonly) NSInteger running;
@property (nonatomic, readonly) NSInteger queued;

/*! Seconds until a request arriving then would start, or nil when the server had no estimate. */
@property (nonatomic, readonly, copy, nullable) NSNumber *estimatedWaitSeconds;

/*! The machine's thermal state, Unknown when the server leaves host details out. */
@property (nonatomic, readonly) NFKServerThermalState thermalState;

/*! When the reply arrived. */
@property (nonatomic, readonly, copy) NSDate *date;

/*! The load a reply's headers carry, or nil when they carry none. Header names match in any case. */
+ (nullable instancetype)loadWithHTTPHeaders:(NSDictionary *)headers;

@end

/*! One run a served model is executing. Introduced in InferKit 0.4.0. */
@interface NFKServerRunStatus : NSObject

/*! Seconds since the run took its slot. */
@property (nonatomic, readonly) NSTimeInterval elapsedSeconds;

/*! The run's last reported progress in 0...1, or nil when its backend reports none. */
@property (nonatomic, readonly, copy, nullable) NSNumber *progress;

/*! The server's estimate of the seconds left, or nil before it has one. */
@property (nonatomic, readonly, copy, nullable) NSNumber *estimatedRemainingSeconds;

@end

/*!
	@class      NFKServerModelStatus
	@abstract   One served model: what its backend loaded and how loaded it is.
	@discussion The counts and averages are the server's own since it started; NFKInferenceServer's
				documentation describes each. A value the server did not report is nil.
				Introduced in InferKit 0.4.0.
*/
@interface NFKServerModelStatus : NSObject

/*! The name the model is served under. */
@property (nonatomic, readonly, copy) NSString *name;

/*! The hosted backend's backendIdentifier. */
@property (nonatomic, readonly, copy) NSString *backendIdentifier;

/*! Whether the hosted backend reports it can serve a run. */
@property (nonatomic, readonly, getter=isReady) BOOL ready;

/*! What the backend reports about its loaded model, keyed by NFKModelInfo*. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *modelInfo;

/*! How many runs the model serves at once. */
@property (nonatomic, readonly) NSInteger limit;

/*! How many runs hold a slot now. */
@property (nonatomic, readonly) NSInteger running;

/*! How many runs wait for a slot. */
@property (nonatomic, readonly) NSInteger queued;

/*! The most runs the model queues, or 0 when its queue is unbounded. */
@property (nonatomic, readonly) NSInteger queueLimit;

@property (nonatomic, readonly) NSInteger completed;
@property (nonatomic, readonly) NSInteger failed;
@property (nonatomic, readonly) NSInteger cancelled;

/*! Requests refused because the queue was full. */
@property (nonatomic, readonly) NSInteger refused;

@property (nonatomic, readonly, copy, nullable) NSNumber *averageRunSeconds;
@property (nonatomic, readonly, copy, nullable) NSNumber *averageWaitSeconds;
@property (nonatomic, readonly, copy, nullable) NSNumber *outputTokensPerSecond;

/*! The input tokens the completed runs reported since the server started, or 0 when none did. */
@property (nonatomic, readonly) NSInteger inputTokens;

/*! The input tokens a backend's cache served, over the runs that reported a cached count. */
@property (nonatomic, readonly) NSInteger cachedInputTokens;

/*! The share of input tokens a backend's cache served, in 0...1, over the runs that reported a cached
	count, or nil before one has. */
@property (nonatomic, readonly, copy, nullable) NSNumber *cachedInputShare;

/*! Seconds until a request arriving now would start, or nil before the server has an estimate. */
@property (nonatomic, readonly, copy, nullable) NSNumber *estimatedWaitSeconds;

/*! The runs executing now. */
@property (nonatomic, readonly, copy) NSArray<NFKServerRunStatus *> *runs;

@end

/*!
	@class      NFKServerHostStatus
	@abstract   The served machine's state, present when its server reports host details.
	@discussion A value the server could not read is nil. gpuUtilization covers every process on the
				machine and reads near 1 while a single run executes, so a balancer routes on the
				models' queue figures. Introduced in InferKit 0.4.0.
*/
@interface NFKServerHostStatus : NSObject

@property (nonatomic, readonly, copy) NSString *chipName;
@property (nonatomic, readonly, copy) NSString *modelIdentifier;
@property (nonatomic, readonly) NSInteger performanceCoreCount;
@property (nonatomic, readonly) NSInteger efficiencyCoreCount;
@property (nonatomic, readonly) NFKServerThermalState thermalState;
@property (nonatomic, readonly, getter=isLowPowerModeEnabled) BOOL lowPowerModeEnabled;

/*! Bytes of physical memory. */
@property (nonatomic, readonly) NSInteger physicalMemory;
/*! Bytes the machine could still allocate when the status was taken. */
@property (nonatomic, readonly, copy, nullable) NSNumber *availableMemory;
/*! Metal's recommended working-set size in bytes. */
@property (nonatomic, readonly, copy, nullable) NSNumber *recommendedWorkingSetSize;
@property (nonatomic, readonly) NFKServerMemoryPressure memoryPressure;
/*! The serving process's memory footprint in bytes. */
@property (nonatomic, readonly, copy, nullable) NSNumber *processFootprint;

/*! The share of processor time spent busy since the server's previous status reading, in 0...1. */
@property (nonatomic, readonly, copy, nullable) NSNumber *cpuUsage;
/*! The 1-, 5-, and 15-minute load averages. */
@property (nonatomic, readonly, copy) NSArray<NSNumber *> *loadAverages;
/*! The busiest GPU's utilization in 0...1, macOS only. */
@property (nonatomic, readonly, copy, nullable) NSNumber *gpuUtilization;

/*! Bytes free on the volume the server's storageDirectoryURL names. */
@property (nonatomic, readonly, copy, nullable) NSNumber *storageAvailableBytes;
@property (nonatomic, readonly, copy, nullable) NSNumber *storageTotalBytes;

/*! Each linked runtime's report (NFKServingRuntimeStatus), keyed by its name, such as "mlx". */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSDictionary<NSString *, id> *> *runtimes;

@end

/*!
	@class      NFKServerStatus
	@abstract   An NFKInferenceServer's status route, read by a client.
	@discussion fetchFromBaseURL:apiKey:error: reads GET /v1/inferkit/status from a server and
				presents it typed; NFKRemoteInferKitBackend's fetchServerStatusWithError: reads the
				server it runs on. JSONObject keeps the whole reply for a field these classes do
				not name. Introduced in InferKit 0.4.0.
*/
@interface NFKServerStatus : NSObject

/*! The server's InferKit version. */
@property (nonatomic, readonly, copy) NSString *version;

/*! When the server started listening. */
@property (nonatomic, readonly, copy) NSDate *startDate;

@property (nonatomic, readonly) NSTimeInterval uptime;

/*! The served models, sorted by name. */
@property (nonatomic, readonly, copy) NSArray<NFKServerModelStatus *> *models;

/*! The machine's state, or nil when the server leaves host details out. */
@property (nonatomic, readonly, strong, nullable) NFKServerHostStatus *host;

/*! The reply as the server sent it. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *JSONObject;

/*! The status a status-route reply describes, or nil when object is not one. */
+ (nullable instancetype)statusWithJSONObject:(id)object;

/*! The served model under a name, or nil. */
- (nullable NFKServerModelStatus *)modelNamed:(NSString *)name;

/*!
	@method     fetchFromBaseURL:apiKey:error:
	@abstract   Reads a server's status route; nil with the server's error when it refuses.
	@discussion baseURL is the server's base, such as http://studio.local:11480/v1, which a
				discovered provider's baseURL is. Blocks for the round trip; run it off the main
				thread, or use the completion-handler form.
*/
+ (nullable instancetype)fetchFromBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey error:(NSError **)error
	NS_SWIFT_NAME(fetch(baseURL:apiKey:));

/*! The asynchronous form of fetchFromBaseURL:apiKey:error:. The handler runs on a background queue at
	user-initiated quality of service. */
+ (void)fetchFromBaseURL:(NSURL *)baseURL
				  apiKey:(nullable NSString *)apiKey
	   completionHandler:(void (^)(NFKServerStatus * _Nullable status, NSError * _Nullable error))completionHandler
	NS_SWIFT_ASYNC_NAME(fetchStatus(baseURL:apiKey:));

@end

NS_ASSUME_NONNULL_END

#endif /* NFKServerStatus_h */
