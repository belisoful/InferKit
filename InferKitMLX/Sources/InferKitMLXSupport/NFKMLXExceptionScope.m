//
//  NFKMLXExceptionScope.m
//  InferKitMLXSupport
//
//  Introduced in InferKit 0.4.0.
//

#import "NFKMLXExceptionScope.h"

#if !__has_feature(objc_arc)
#error "InferKitMLXSupport compiles under ARC: the caught exception it returns needs ARC's retain."
#endif

void NFKMLXPerformWithCleanup(NS_NOESCAPE void (^body)(void), NS_NOESCAPE void (^cleanup)(void))
{
	@try {
		body();
	} @finally {
		cleanup();
	}
}

NSException * _Nullable NFKMLXCatchException(NS_NOESCAPE void (^body)(void))
{
	@try {
		body();
	} @catch (NSException *exception) {
		return exception;
	}
	return nil;
}
