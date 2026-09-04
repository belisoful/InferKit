//
//  NFKLocalModelRunner.m
//  InferKit
//

#import <InferKit/NFKLocalModelRunner.h>
#import <InferKit/NFKOllamaRunner.h>
#import <InferKit/NFKLMStudioRunner.h>

@implementation NFKRemoteProvider (LocalRunner)

- (nullable id<NFKLocalModelRunner>)localRunner
{
	if ([self.identifier isEqualToString:@"ollama"]) {
		return [NFKOllamaRunner runnerWithProvider:self];
	}
	if ([self.identifier isEqualToString:@"lmstudio"]) {
		return [NFKLMStudioRunner runnerWithProvider:self];
	}
	return nil;
}

@end
