//
//  NFKModelEngineBuilders.m
//  InferKit
//

#import "NFKModelEngineBuilders.h"
#import "NFKCoreMLLanguageBackend.h"
#import "NFKModelResolver.h"
#import "NFKModelVocabulary.h"
#import "NFKRemoteProvider.h"

static NSError *NFKModelEngineBuildError(NFKModelCatalogError code, NSString *reason)
{
	return [NSError errorWithDomain:NFKModelCatalogErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

@implementation NFKCoreMLEngineBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	if (request.location == nil) {
		if (error != NULL) {
			*error = NFKModelEngineBuildError(NFKModelCatalogErrorNotDownloaded,
				[NSString stringWithFormat:@"%@'s Core ML package is not downloaded.", request.descriptor.displayName]);
		}
		return nil;
	}
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:request.location];
		return [request applyPropertyOptionsToBackend:backend error:error] ? backend : nil;
	}
	if (error != NULL) {
		*error = NFKModelEngineBuildError(NFKModelCatalogErrorUnavailable,
			@"A Core ML language model keeps its cache in Core ML state, which needs macOS 15, iOS 18, or tvOS 18.");
	}
	return nil;
}

@end

@implementation NFKLocalServerEngineBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	NSDictionary *server = request.descriptor.extensions[@"localServer"];
	NFKModelResolutionAction *action = nil;
	if ([server isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *object = [server mutableCopy];
		object[@"kind"] = @"use";
		action = [NFKModelResolutionAction actionWithDictionary:object error:NULL];
	}
	NFKRemoteProvider *provider = action.provider;
	NSString *model = action.model;
	if (provider == nil || provider.requiresAPIKey) {
		if (error != NULL) {
			*error = NFKModelEngineBuildError(NFKModelCatalogErrorMalformedDescriptor,
				[NSString stringWithFormat:@"%@ names no keyless server preset and model in its localServer extension.",
				 request.descriptor.identifier]);
		}
		return nil;
	}
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider apiKey:nil modelName:model];
	return [request applyPropertyOptionsToBackend:backend error:error] ? backend : nil;
}

@end
