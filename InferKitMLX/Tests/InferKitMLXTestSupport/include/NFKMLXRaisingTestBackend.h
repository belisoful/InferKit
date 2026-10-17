//
//  NFKMLXRaisingTestBackend.h
//  InferKitMLXTestSupport
//
//  A backend implemented in Objective-C whose own method raises, for the forwarding tests. A raise from
//  a Swift `@objc` method leaks its receiver, since the method's thunk retains it and an Objective-C
//  exception runs no Swift cleanup, so only an Objective-C raise shows what the wrapper itself releases.
//

#import <Foundation/Foundation.h>
#import <InferKit/InferKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKMLXRaisingTestBackend : NSObject <NFKInferenceBackend>

/*! Raises NSGenericException with the reason "the held backend raised". */
- (void)raiseAnException;

@end

NS_ASSUME_NONNULL_END
