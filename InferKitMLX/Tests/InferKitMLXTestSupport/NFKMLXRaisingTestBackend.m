//
//  NFKMLXRaisingTestBackend.m
//  InferKitMLXTestSupport
//

#import "NFKMLXRaisingTestBackend.h"

@implementation NFKMLXRaisingTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"lifecycle-test-raising";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ran" }];
}

- (void)raiseAnException
{
	@throw [NSException exceptionWithName:NSGenericException reason:@"the held backend raised" userInfo:nil];
}

@end
