//
//  NFKSubprocessPrivate.h
//  InferKit
//
//  The hooks an owner inside InferKit uses to read a subprocess it speaks a protocol with. They run
//  before the public handlers, so an application's handlers observe the same lines and states.
//

#import "NFKSubprocess.h"

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

@interface NFKSubprocess ()

/*! Receives each standard output line before lineHandler. */
@property (atomic, copy, nullable) NFKSubprocessLineHandler ownerLineHandler;

/*! Receives each state before stateHandler. */
@property (atomic, copy, nullable) NFKSubprocessStateHandler ownerStateHandler;

@end

#endif

NS_ASSUME_NONNULL_END
