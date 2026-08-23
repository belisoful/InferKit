//
//  NFKMLMultiArray.m
//  InferKit
//

#import "NFKMLMultiArray.h"

MLMultiArray * _Nullable NFKMultiArrayFromInterleaved(const float *interleaved,
													  NFKTensorSpec spec,
													  NSError * _Nullable * _Nullable error)
{
	NSArray<NSNumber *> *shape;
	if (spec.layout == NFKTensorLayoutHWC) {
		shape = @[ @1, @(spec.height), @(spec.width), @(spec.channelCount) ];
	} else {
		shape = @[ @1, @(spec.channelCount), @(spec.height), @(spec.width) ];
	}

	MLMultiArray *array = [[MLMultiArray alloc] initWithShape:shape
													dataType:MLMultiArrayDataTypeFloat32
													   error:error];
	if (array == nil) {
		return nil;
	}
	// A freshly created MLMultiArray is contiguous, so its memory order matches the planar buffer
	// NFKTensorConversion writes for the same layout.
	NFKInterleavedToTensor(interleaved, (float *)array.dataPointer, spec);
	return array;
}

BOOL NFKInterleavedFromMultiArray(MLMultiArray *array, float *interleaved, NFKTensorSpec spec)
{
	if (array.dataType != MLMultiArrayDataTypeFloat32) {
		return NO;
	}
	NFKTensorToInterleaved((const float *)array.dataPointer, interleaved, spec);
	return YES;
}
