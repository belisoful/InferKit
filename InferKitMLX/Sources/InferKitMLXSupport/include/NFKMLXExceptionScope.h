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
	@function   NFKMLXCatchException
	@abstract   Runs body and returns the Objective-C exception it raised, or nil when it raised none.
	@discussion A frame of body between the raise and this function runs no cleanup, so a Swift body
				that holds strong references across the raise leaks them. Introduced in InferKit 0.4.0.
*/
NSException * _Nullable NFKMLXCatchException(NS_NOESCAPE void (^body)(void));

/*!
	@function   NFKMLXInvokeCatching
	@abstract   Sends invocation (an NSInvocation) to target and returns the Objective-C exception the
				message raised, or nil when it raised none.
	@discussion No Swift frame lies between the catch and the message. The invocation's target is
				restored afterward. Introduced in InferKit 0.4.0.
*/
NSException * _Nullable NFKMLXInvokeCatching(id invocation, id target);

/*!
	@function   NFKMLXForwardInvocationImplementation
	@abstract   An implementation of forwardInvocation: that calls handler and raises the exception it
				returns.
	@discussion The raise happens in this implementation after handler has returned, so no frame of
				handler is on the stack and every reference handler took is already released. The
				implementation holds the receiver and the invocation unretained. Introduced in
				InferKit 0.4.0.
*/
IMP NFKMLXForwardInvocationImplementation(NSException * _Nullable (^handler)(id receiver, id invocation));

NS_ASSUME_NONNULL_END
