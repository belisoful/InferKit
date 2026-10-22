//
//  NFKModelEngineBuilders.m
//  InferKit
//

#import "NFKModelEngineBuilders.h"
#import "NFKCoreMLLanguageBackend.h"
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
	NSString *providerIdentifier = [server isKindOfClass:NSDictionary.class] ? server[@"provider"] : nil;
	NSString *model = [server isKindOfClass:NSDictionary.class] ? server[@"model"] : nil;
	NFKRemoteProvider *provider = nil;
	for (NFKRemoteProvider *candidate in NFKRemoteProvider.localProviders) {
		if ([providerIdentifier isKindOfClass:NSString.class] && [candidate.identifier isEqualToString:providerIdentifier]) {
			provider = candidate;
		}
	}
	if (provider == nil || ![model isKindOfClass:NSString.class]) {
		if (error != NULL) {
			*error = NFKModelEngineBuildError(NFKModelCatalogErrorMalformedDescriptor,
				[NSString stringWithFormat:@"%@ names no local server preset and model in its localServer extension.",
				 request.descriptor.identifier]);
		}
		return nil;
	}
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:provider apiKey:nil modelName:model];
	return [request applyPropertyOptionsToBackend:backend error:error] ? backend : nil;
}

@end
