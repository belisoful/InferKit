//
//  NFKLiveStream.m
//  InferKit
//

#import "NFKLiveStream.h"
#import "NFKErrors.h"

@interface NFKBlockLiveStream ()
@property (nonatomic, copy, nullable) NFKLiveStreamUnitHandler unitHandler;
@property (nonatomic, copy, nullable) void (^resetHandler)(void);
@property (nonatomic, copy, nullable) void (^closeHandler)(void);
@property (nonatomic, readwrite, getter=isClosed) BOOL closed;
@property (nonatomic, readonly) NSObject *unitLock;
@end

@implementation NFKBlockLiveStream

- (instancetype)initWithParameters:(nullable NSDictionary<NSString *, id> *)parameters
					   unitHandler:(NFKLiveStreamUnitHandler)unitHandler
					  resetHandler:(nullable void (^)(void))resetHandler
					  closeHandler:(nullable void (^)(void))closeHandler
{
	self = [super init];
	if (self != nil) {
		_parameters = [parameters copy] ?: @{};
		_unitHandler = [unitHandler copy];
		_resetHandler = [resetHandler copy];
		_closeHandler = [closeHandler copy];
		_unitLock = [[NSObject alloc] init];
	}
	return self;
}

- (nullable NFKInferenceResult *)resultForUnit:(NFKInferenceRequest *)unit error:(NSError **)error
{
	@synchronized (self.unitLock) {
		if (self.isClosed) {
			if (error != NULL) {
				*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceNotReady
										 userInfo:@{ NSLocalizedDescriptionKey: @"The live stream is closed." }];
			}
			return nil;
		}
		return self.unitHandler([self mergedUnit:unit], error);
	}
}

- (NFKInferenceRequest *)mergedUnit:(NFKInferenceRequest *)unit
{
	if (self.parameters.count == 0) {
		return unit;
	}
	NSMutableDictionary<NSString *, id> *parameters = [self.parameters mutableCopy];
	[parameters addEntriesFromDictionary:unit.parameters];
	return [NFKInferenceRequest requestWithInputs:unit.inputs parameters:parameters outputModality:unit.outputModality];
}

- (void)reset
{
	@synchronized (self.unitLock) {
		if (!self.isClosed && self.resetHandler != nil) {
			self.resetHandler();
		}
	}
}

- (void)close
{
	@synchronized (self.unitLock) {
		if (self.isClosed) {
			return;
		}
		self.closed = YES;
		void (^closeHandler)(void) = self.closeHandler;
		self.unitHandler = nil;
		self.resetHandler = nil;
		self.closeHandler = nil;
		if (closeHandler != nil) {
			closeHandler();
		}
	}
}

@end

id<NFKLiveStream> _Nullable NFKInferenceOpenLiveStream(id<NFKInferenceBackend> backend,
													   NSDictionary<NSString *, id> * _Nullable parameters,
													   NSError * _Nullable * _Nullable error)
{
	if ([backend respondsToSelector:@selector(openLiveStreamWithParameters:error:)]) {
		NSError *openError = nil;
		id<NFKLiveStream> stream = [backend openLiveStreamWithParameters:parameters error:&openError];
		if (stream == nil && error != NULL) {
			*error = openError;
		}
		return stream;
	}
	return [[NFKBlockLiveStream alloc] initWithParameters:parameters
											  unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **unitError) {
		return [backend runInferenceForRequest:unit error:unitError];
	} resetHandler:nil closeHandler:nil];
}
