//
//  NFKLocalRunnerSupport.h
//  InferKit
//
//  Private: the URL arithmetic the runner adapters share.
//

#ifndef NFKLocalRunnerSupport_h
#define NFKLocalRunnerSupport_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The provider's base with a trailing /v1 removed: where a runner's native API lives. */
NSURL *NFKLocalRunnerNativeBase(NSURL *baseURL);

/*! The native base joined to a path with exactly one slash; an empty path is the root. */
NSURL *NFKLocalRunnerURL(NSURL *nativeBase, NSString *path);

NS_ASSUME_NONNULL_END

#endif /* NFKLocalRunnerSupport_h */
