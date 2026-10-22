//
//  NFKModelEngineBuilders.h
//  InferKit
//
//  The builders the core registers for its own engines: a Core ML language model and a model server on
//  this machine, the descriptors NFKModelResolver writes for those routes.
//

#ifndef NFKModelEngineBuilders_h
#define NFKModelEngineBuilders_h

#import <Foundation/Foundation.h>
#import "NFKModelBuilding.h"

NS_ASSUME_NONNULL_BEGIN

/*! Builds NFKCoreMLLanguageBackend from the variant's located directory (NFKModelEngineCoreML). On an OS
	before macOS 15, iOS 18, or tvOS 18 the build fails with NFKModelCatalogErrorUnavailable. */
@interface NFKCoreMLEngineBuilder : NSObject <NFKModelBackendBuilding>
@end

/*! Builds the chat backend of the descriptor's "localServer" extension: the NFKRemoteProvider local preset
	it names (provider) and the model the server serves (model) (NFKModelEngineLocalServer). */
@interface NFKLocalServerEngineBuilder : NSObject <NFKModelBackendBuilding>
@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelEngineBuilders_h */
