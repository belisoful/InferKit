//
//  NFKPassthroughBackend.m
//  InferKit
//

#import "NFKPassthroughBackend.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"

@implementation NFKPassthroughBackend

+ (instancetype)backend
{
	return NARC_AUTORELEASE([[self alloc] init]);
}

- (void)dealloc
{
	NARC_RELEASE(_outputMap);
	SUPER_DEALLOC();
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"passthrough";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError **)outError
{
	if (self.outputMap == nil) {
		return [NFKInferenceResult resultWithOutputs:request.inputs];
	}

	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithCapacity:self.outputMap.count];
	for (NSString *outputName in self.outputMap) {
		NSString *inputName = self.outputMap[outputName];
		id value = [request inputForKey:inputName];
		if (value == nil) {
			if (outError != NULL) {
				NSString *reason = [NSString stringWithFormat:@"passthrough output '%@' maps to missing input '%@'",
									outputName, inputName];
				*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
												code:kNFKError_InferenceMissingInput
											userInfo:@{ NSLocalizedDescriptionKey: reason }];
			}
			return nil;
		}
		outputs[outputName] = value;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

@end
