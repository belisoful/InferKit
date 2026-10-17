//
//  NFKMLXExceptionScope.m
//  InferKitMLXSupport
//
//  Introduced in InferKit 0.4.0.
//

#import "NFKMLXExceptionScope.h"
#import <objc/runtime.h>

#if !__has_feature(objc_arc)
#error "InferKitMLXSupport compiles under ARC: the caught exception it returns needs ARC's retain."
#endif

NSException * _Nullable NFKMLXCatchException(NS_NOESCAPE void (^body)(void))
{
	@try {
		body();
	} @catch (NSException *exception) {
		return exception;
	}
	return nil;
}

NSException * _Nullable NFKMLXInvokeCatching(id invocation, id target)
{
	NSInvocation *message = invocation;
	id sender = message.target;
	NSException *raised = nil;
	@try {
		[message invokeWithTarget:target];
	} @catch (NSException *exception) {
		raised = exception;
	}
	// The invocation outlives a raise the forwarding machinery does not clean up, and must not keep
	// target alive then.
	message.target = sender;
	return raised;
}

IMP NFKMLXForwardInvocationImplementation(NSException * _Nullable (^handler)(id receiver, id invocation))
{
	// The parameters stay unretained: ARC skips their release when the raise unwinds this frame.
	return imp_implementationWithBlock(^(__unsafe_unretained id receiver, __unsafe_unretained NSInvocation *invocation) {
		NSException *exception = handler(receiver, invocation);
		if (exception != nil) {
			@throw exception;
		}
	});
}
