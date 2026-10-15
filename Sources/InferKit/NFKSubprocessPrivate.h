//
//  NFKSubprocessPrivate.h
//  InferKit
//
//  The hooks an owner inside InferKit uses to read a subprocess it speaks a protocol with. They run
//  before the public handlers, so an application's handlers observe the same lines and states.
//

#import <InferKit/NFKSubprocess.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

@interface NFKSubprocess ()

/*! Receives each standard output line before lineHandler. */
@property (atomic, copy, nullable) NFKSubprocessLineHandler ownerLineHandler;

/*! Receives each state before stateHandler. */
@property (atomic, copy, nullable) NFKSubprocessStateHandler ownerStateHandler;

/*! Descriptors the child inherits beyond its standard streams. Each key is the descriptor's number in
	the child, 3 or higher; each value is a descriptor of this process the child receives a duplicate
	of. The subprocess neither takes nor closes the values. Read at each launch; a key below 3 or a
	value that is not open fails the launch. */
@property (atomic, copy, nullable) NSDictionary<NSNumber *, NSNumber *> *ownerInheritedDescriptors;

/*! The quality-of-service class the child starts at, or QOS_CLASS_UNSPECIFIED (the default) for the
	system's choice. Read at each launch. */
@property (atomic, assign) qos_class_t ownerQualityOfService;

@end

#endif

NS_ASSUME_NONNULL_END
