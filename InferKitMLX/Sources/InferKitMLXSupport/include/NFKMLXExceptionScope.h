//
//  NFKMLXExceptionScope.h
//  InferKitMLXSupport
//
//  Objective-C exception scopes for InferKitMLX, which Swift cannot write. The module is internal to the
//  package: InferKitMLX imports it with `internal import`, and no product exports it.
//
//  Introduced in InferKit 0.4.0.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@function   NFKMLXPerformWithCleanup
	@abstract   Runs body, then cleanup, also when body raises an Objective-C exception.
	@discussion The exception continues to the caller once cleanup has run. Introduced in InferKit 0.4.0.
*/
void NFKMLXPerformWithCleanup(NS_NOESCAPE void (^body)(void), NS_NOESCAPE void (^cleanup)(void));

/*!
	@function   NFKMLXCatchException
	@abstract   Runs body and returns the Objective-C exception it raised, or nil when it raised none.
	@discussion For tests that drive a path raising an exception from Swift. Introduced in InferKit 0.4.0.
*/
NSException * _Nullable NFKMLXCatchException(NS_NOESCAPE void (^body)(void));

NS_ASSUME_NONNULL_END
