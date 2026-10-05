//
//  NFKModelInfoSupport.m
//  InferKit
//

#import "NFKModelInfoSupport.h"
#import <InferKit/NFKInferenceKeys.h>

NSString *NFKModelInfoComputeUnitsName(MLComputeUnits computeUnits)
{
	if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
		if (computeUnits == MLComputeUnitsCPUAndNeuralEngine) {
			return @"cpu_and_neural_engine";
		}
	}
	switch (computeUnits) {
		case MLComputeUnitsCPUOnly:
			return @"cpu_only";
		case MLComputeUnitsCPUAndGPU:
			return @"cpu_and_gpu";
		default:
			return @"all";
	}
}

NSNumber * _Nullable NFKModelInfoStorageBytesAtURL(NSURL * _Nullable url)
{
	NSArray<NSURLResourceKey> *keys = @[ NSURLIsRegularFileKey, NSURLIsDirectoryKey, NSURLTotalFileAllocatedSizeKey ];
	NSDictionary<NSURLResourceKey, id> *values = [url resourceValuesForKeys:keys error:NULL];
	if (values == nil) {
		return nil;
	}
	if (![values[NSURLIsDirectoryKey] boolValue]) {
		return @([values[NSURLTotalFileAllocatedSizeKey] longLongValue]);
	}
	NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager enumeratorAtURL:url
																   includingPropertiesForKeys:keys
																					  options:0
																				 errorHandler:nil];
	long long total = 0;
	for (NSURL *file in enumerator) {
		NSDictionary<NSURLResourceKey, id> *fileValues = [file resourceValuesForKeys:keys error:NULL];
		if ([fileValues[NSURLIsRegularFileKey] boolValue]) {
			total += [fileValues[NSURLTotalFileAllocatedSizeKey] longLongValue];
		}
	}
	return @(total);
}

NSDictionary<NSString *, id> *NFKModelInfoForCoreMLModel(MLModel *model, NSURL * _Nullable compiledURL)
{
	NSMutableDictionary<NSString *, id> *info = [NSMutableDictionary dictionary];
	info[NFKModelInfoComputeUnits] = NFKModelInfoComputeUnitsName(model.configuration.computeUnits);
	id version = model.modelDescription.metadata[MLModelVersionStringKey];
	if ([version isKindOfClass:NSString.class] && [version length] > 0) {
		info[NFKModelInfoVersion] = version;
	}
	info[NFKModelInfoStorageBytes] = NFKModelInfoStorageBytesAtURL(compiledURL);
	return info;
}
