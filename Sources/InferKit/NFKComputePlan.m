//
//  NFKComputePlan.m
//  InferKit
//

#import "NFKComputePlan.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"

@implementation NFKComputePlan
{
	NSInteger _neuralEngineOperationCount;
	NSInteger _gpuOperationCount;
	NSInteger _cpuOperationCount;
	NSInteger _unknownOperationCount;
	NSCountedSet<NSString *> *_operatorsOffNeuralEngine;
}

+ (BOOL)isAvailable
{
	if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, *)) {
		return YES;
	}
	return NO;
}

+ (NSError *)unsupportedErrorWithMessage:(NSString *)message
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain
							   code:kNFKError_InferenceUnsupported
						   userInfo:@{ NSLocalizedDescriptionKey: message }];
}

/*! NO with a reason when the plan cannot be attempted at all. */
+ (BOOL)validateURL:(nullable NSURL *)url error:(NSError * _Nullable *)outError
{
	NSString *reason = nil;
	if (url == nil) {
		reason = @"a compiled model URL is required";
	} else if (![self isAvailable]) {
		reason = @"Core ML publishes compute plans from macOS 14.4 / iOS 17.4 / tvOS 17.4";
	}
	if (reason == nil) {
		return YES;
	}
	if (outError != NULL) {
		*outError = [self unsupportedErrorWithMessage:reason];
	}
	return NO;
}

+ (void)planForCompiledModelAtURL:(NSURL *)url
					 computeUnits:(MLComputeUnits)computeUnits
				completionHandler:(void (^)(NFKComputePlan * _Nullable, NSError * _Nullable))completionHandler
{
	NSError *reason = nil;
	if (![self validateURL:url error:&reason]) {
		completionHandler(nil, reason);
		return;
	}

	if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, *)) {
		MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
		configuration.computeUnits = computeUnits;
		[MLComputePlan loadContentsOfURL:url
						   configuration:configuration
					   completionHandler:^(MLComputePlan * _Nullable plan, NSError * _Nullable error) {
			if (plan == nil) {
				completionHandler(nil, error ?: [self unsupportedErrorWithMessage:@"the compute plan could not be loaded"]);
				return;
			}
			NFKComputePlan *result = [[self alloc] init];
			[result accumulateFromPlan:plan];
			completionHandler(NARC_AUTORELEASE(result), nil);
		}];
		NARC_RELEASE(configuration);
		return;
	}
	completionHandler(nil, [self unsupportedErrorWithMessage:@"Core ML publishes compute plans from macOS 14.4 / iOS 17.4 / tvOS 17.4"]);
}

+ (nullable instancetype)planForCompiledModelAtURL:(NSURL *)url
									  computeUnits:(MLComputeUnits)computeUnits
											 error:(NSError * _Nullable *)outError
{
	// Core ML answers on its own queue, so waiting here blocks only the calling thread. The header
	// says to call this off the main thread, and the completion-handler variant above is the path for
	// a caller that cannot block at all.
	__block NFKComputePlan *loaded = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	[self planForCompiledModelAtURL:url computeUnits:computeUnits
				  completionHandler:^(NFKComputePlan *plan, NSError *error) {
		loaded = NARC_RETAIN(plan);
		failure = NARC_RETAIN(error);
		dispatch_semaphore_signal(semaphore);
	}];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (loaded == nil && outError != NULL) {
		*outError = failure;
	}
	return NARC_AUTORELEASE(loaded);
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_operatorsOffNeuralEngine = [[NSCountedSet alloc] init];
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_operatorsOffNeuralEngine);
	SUPER_DEALLOC();
}

#pragma mark Traversal

- (void)accumulateFromPlan:(MLComputePlan *)plan API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4))
{
	[self accumulateFromStructure:plan.modelStructure plan:plan];
}

- (void)accumulateFromStructure:(MLModelStructure *)structure
						   plan:(MLComputePlan *)plan API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4))
{
	if (structure.program != nil) {
		for (MLModelStructureProgramFunction *function in structure.program.functions.allValues) {
			[self accumulateFromBlock:function.block plan:plan];
		}
		return;
	}
	if (structure.neuralNetwork != nil) {
		for (MLModelStructureNeuralNetworkLayer *layer in structure.neuralNetwork.layers) {
			[self recordDeviceUsage:[plan computeDeviceUsageForNeuralNetworkLayer:layer]
					   operatorName:layer.type];
		}
		return;
	}
	// A pipeline is a model of models, and each stage is placed on its own.
	for (MLModelStructure *subModel in structure.pipeline.subModels) {
		[self accumulateFromStructure:subModel plan:plan];
	}
}

- (void)accumulateFromBlock:(MLModelStructureProgramBlock *)block
					   plan:(MLComputePlan *)plan API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4))
{
	for (MLModelStructureProgramOperation *operation in block.operations) {
		// A control-flow operation carries the real work in its nested blocks; counting the wrapper
		// as one placement would hide everything inside it.
		if (operation.blocks.count > 0) {
			for (MLModelStructureProgramBlock *nested in operation.blocks) {
				[self accumulateFromBlock:nested plan:plan];
			}
			continue;
		}
		[self recordDeviceUsage:[plan computeDeviceUsageForMLProgramOperation:operation]
				   operatorName:operation.operatorName];
	}
}

- (void)recordDeviceUsage:(nullable MLComputePlanDeviceUsage *)usage
			 operatorName:(NSString *)operatorName API_AVAILABLE(macos(14.4), ios(17.4), tvos(17.4))
{
	id preferred = usage.preferredComputeDevice;
	if (preferred == nil) {
		_unknownOperationCount += 1;
		return;
	}
	if ([preferred isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
		_neuralEngineOperationCount += 1;
		return;
	}

	if ([preferred isKindOfClass:[MLGPUComputeDevice class]]) {
		_gpuOperationCount += 1;
	} else {
		_cpuOperationCount += 1;
	}
	if (operatorName.length > 0) {
		[_operatorsOffNeuralEngine addObject:operatorName];
	}
}

#pragma mark Reporting

- (NSInteger)neuralEngineOperationCount { return _neuralEngineOperationCount; }
- (NSInteger)gpuOperationCount { return _gpuOperationCount; }
- (NSInteger)cpuOperationCount { return _cpuOperationCount; }
- (NSInteger)unknownOperationCount { return _unknownOperationCount; }

- (NSInteger)operationCount
{
	return _neuralEngineOperationCount + _gpuOperationCount + _cpuOperationCount + _unknownOperationCount;
}

- (NSInteger)placedOperationCount
{
	return _neuralEngineOperationCount + _gpuOperationCount + _cpuOperationCount;
}

- (double)neuralEngineFraction
{
	NSInteger placed = [self placedOperationCount];
	if (placed <= 0) {
		return 0.0;
	}
	return (double)_neuralEngineOperationCount / (double)placed;
}

- (BOOL)runsEntirelyOnNeuralEngine
{
	return [self placedOperationCount] > 0 && _gpuOperationCount == 0 && _cpuOperationCount == 0;
}

- (NSArray<NSString *> *)operatorNamesOffNeuralEngine
{
	NSCountedSet<NSString *> *counted = _operatorsOffNeuralEngine;
	return [counted.allObjects sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
		NSUInteger left = [counted countForObject:a];
		NSUInteger right = [counted countForObject:b];
		if (left != right) {
			return left > right ? NSOrderedAscending : NSOrderedDescending;
		}
		return [a compare:b];
	}];
}

- (NSString *)describedPlacement
{
	return [NSString stringWithFormat:
			@"%ld operations: %ld Neural Engine, %ld GPU, %ld CPU, %ld unreported (%.1f%% Neural Engine)",
			(long)[self operationCount], (long)_neuralEngineOperationCount, (long)_gpuOperationCount,
			(long)_cpuOperationCount, (long)_unknownOperationCount, [self neuralEngineFraction] * 100.0];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@>", NSStringFromClass([self class]), [self describedPlacement]];
}

@end
