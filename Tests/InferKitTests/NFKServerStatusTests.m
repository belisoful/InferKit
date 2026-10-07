//
//  NFKServerStatusTests.m
//  InferKitTests
//
//  Reading a status-route reply into the typed classes. The live round trip is in
//  NFKInferenceServerTests.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKServerStatusTests : XCTestCase
@end

@implementation NFKServerStatusTests

- (NSDictionary *)reply
{
	return @{ @"object": @"inferkit.status",
			  @"server": @{ @"version": @"0.4.0", @"started": @1800000000, @"uptime_seconds": @12.5 },
			  @"models": @[ @{ @"id": @"chat", @"backend": @"mlx-language", @"ready": @YES,
							   @"model": @{ NFKModelInfoParameterCount: @600000000, NFKModelInfoPrecision: @"bfloat16" },
							   @"load": @{ @"limit": @1, @"running": @1, @"queued": @2, @"queue_limit": @4,
										   @"completed": @10, @"failed": @1, @"cancelled": @2, @"refused": @3,
										   @"average_run_seconds": @4.5, @"average_wait_seconds": @1.25,
										   @"estimated_wait_seconds": @9.0,
										   @"input_tokens": @400, @"cached_input_tokens": @150, @"cached_input_share": @0.5,
										   @"runs": @[ @{ @"elapsed_seconds": @2.0, @"progress": @0.5,
														  @"estimated_remaining_seconds": @2.0 } ] } } ],
			  @"host": @{ @"chip": @"Apple M1 Max", @"model_identifier": @"MacBookPro18,2",
						  @"performance_cores": @8, @"efficiency_cores": @2, @"thermal_state": @"serious",
						  @"low_power_mode": @YES,
						  @"memory": @{ @"physical_bytes": @34359738368, @"available_bytes": @6000000000,
										@"pressure": @"warning", @"process_footprint_bytes": @8000000 },
						  @"cpu": @{ @"usage": @0.25, @"load_average": @[ @1.5, @2.0, @2.5 ] },
						  @"gpu": @{ @"utilization": @0.82, @"source": @"ioregistry" },
						  @"storage": @{ @"available_bytes": @100, @"total_bytes": @200 },
						  @"runtimes": @{ @"mlx": @{ @"active_memory_bytes": @4096 } } } };
}

- (void)testAReplyReadsIntoTheTypedStatus
{
	NFKServerStatus *status = [NFKServerStatus statusWithJSONObject:[self reply]];
	XCTAssertEqualObjects(status.version, @"0.4.0");
	XCTAssertEqualObjects(status.startDate, [NSDate dateWithTimeIntervalSince1970:1800000000]);
	XCTAssertEqual(status.uptime, 12.5);

	NFKServerModelStatus *model = [status modelNamed:@"chat"];
	XCTAssertNotNil(model);
	XCTAssertNil([status modelNamed:@"absent"]);
	XCTAssertEqualObjects(model.backendIdentifier, @"mlx-language");
	XCTAssertTrue(model.isReady);
	XCTAssertEqualObjects(model.modelInfo[NFKModelInfoPrecision], @"bfloat16");
	XCTAssertEqual(model.limit, 1);
	XCTAssertEqual(model.running, 1);
	XCTAssertEqual(model.queued, 2);
	XCTAssertEqual(model.queueLimit, 4);
	XCTAssertEqual(model.completed, 10);
	XCTAssertEqual(model.failed, 1);
	XCTAssertEqual(model.cancelled, 2);
	XCTAssertEqual(model.refused, 3);
	XCTAssertEqualObjects(model.averageRunSeconds, @4.5);
	XCTAssertEqualObjects(model.averageWaitSeconds, @1.25);
	XCTAssertNil(model.outputTokensPerSecond, @"absent from the reply");
	XCTAssertEqual(model.inputTokens, 400);
	XCTAssertEqual(model.cachedInputTokens, 150);
	XCTAssertEqualObjects(model.cachedInputShare, @0.5, @"the server's share, read as sent");
	XCTAssertEqualObjects(model.estimatedWaitSeconds, @9.0);
	XCTAssertEqual(model.runs.count, 1u);
	XCTAssertEqual(model.runs.firstObject.elapsedSeconds, 2.0);
	XCTAssertEqualObjects(model.runs.firstObject.progress, @0.5);

	NFKServerHostStatus *host = status.host;
	XCTAssertEqualObjects(host.chipName, @"Apple M1 Max");
	XCTAssertEqual(host.performanceCoreCount, 8);
	XCTAssertEqual(host.thermalState, NFKServerThermalStateSerious);
	XCTAssertTrue(host.isLowPowerModeEnabled);
	XCTAssertEqual(host.physicalMemory, 34359738368);
	XCTAssertEqual(host.memoryPressure, NFKServerMemoryPressureWarning);
	XCTAssertEqualObjects(host.cpuUsage, @0.25);
	XCTAssertEqualObjects(host.loadAverages, (@[ @1.5, @2.0, @2.5 ]));
	XCTAssertEqualObjects(host.gpuUtilization, @0.82);
	XCTAssertEqualObjects(host.storageAvailableBytes, @100);
	XCTAssertEqualObjects(host.runtimes[@"mlx"][@"active_memory_bytes"], @4096);
	XCTAssertEqualObjects(status.JSONObject, [self reply]);
}

- (void)testMissingAndMalformedValuesReadAsUnknown
{
	NSMutableDictionary *reply = [[self reply] mutableCopy];
	reply[@"host"] = @{ @"thermal_state": @"molten", @"memory": @{ @"pressure": @7, @"available_bytes": @"lots" } };
	reply[@"models"] = @[ @{ @"id": @"bare" }, @"not a model" ];
	NFKServerStatus *status = [NFKServerStatus statusWithJSONObject:reply];
	XCTAssertEqual(status.host.thermalState, NFKServerThermalStateUnknown);
	XCTAssertEqual(status.host.memoryPressure, NFKServerMemoryPressureUnknown);
	XCTAssertNil(status.host.availableMemory);
	XCTAssertNil(status.host.gpuUtilization);
	XCTAssertEqualObjects(status.host.loadAverages, @[]);
	XCTAssertEqual(status.models.count, 1u, @"an entry that is not an object is skipped");
	XCTAssertEqual(status.models.firstObject.limit, 0);
	XCTAssertNil(status.models.firstObject.estimatedWaitSeconds);

	reply[@"host"] = nil;
	XCTAssertNil([NFKServerStatus statusWithJSONObject:reply].host, @"a server that leaves host details out");
	XCTAssertNil([NFKServerStatus statusWithJSONObject:@{ @"object": @"list" }]);
	XCTAssertNil([NFKServerStatus statusWithJSONObject:@[]]);
}

@end
