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
	the child, 3 or higher; each value is a descriptor of this process. Setting the property duplicates
	every value, and the subprocess owns the duplicates: it closes them, exactly once, when the property
	is set again or the subprocess is released. The caller keeps its own descriptors and may close them
	at once. Reading the property returns the subprocess's duplicates, which the caller does not close.
	Each launch, relaunches included, gives the child copies of the duplicates. While the subprocess
	holds a duplicate, the file stays open in this process, so a reader of a pipe or socket sees end of
	file only after the property is cleared or the subprocess is released. A key below 3, or a value
	that was not open when set, fails the launch. */
@property (atomic, copy, nullable) NSDictionary<NSNumber *, NSNumber *> *ownerInheritedDescriptors;

/*! The quality-of-service class the child starts at, or QOS_CLASS_UNSPECIFIED (the default) for the
	system's choice. Read at each launch. */
@property (atomic, assign) qos_class_t ownerQualityOfService;

@end

#endif

NS_ASSUME_NONNULL_END
