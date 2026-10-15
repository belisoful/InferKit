//
//  DuplicateRuntimePlugin.m
//  InferKit plugin fixtures
//
//  An Objective-C plugin bundle that defines a class of its own named NFKInferenceRequest, as a bundle
//  that embeds a copy of InferKit would. The loader refuses it with NFKPluginErrorDuplicateRuntime. It
//  imports no InferKit header, since the class it defines would clash with the declaration there.
//

#import <Foundation/Foundation.h>

@interface NFKInferenceRequest : NSObject
@end

@implementation NFKInferenceRequest
@end

@interface DuplicateRuntimePlugin : NSObject
@end

@implementation DuplicateRuntimePlugin

- (instancetype)initWithContext:(id)context
{
	return [super init];
}

- (NSArray *)modelDescriptors
{
	return @[];
}

- (nullable id)backendForModelIdentifier:(NSString *)modelIdentifier options:(NSDictionary *)options error:(NSError **)error
{
	return nil;
}

@end
