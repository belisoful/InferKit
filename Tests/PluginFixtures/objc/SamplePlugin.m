//
//  SamplePlugin.m
//  InferKit plugin fixtures
//
//  An Objective-C plugin bundle. It links InferKit's interface without embedding a copy: the bundle is
//  built with -undefined dynamic_lookup, so every NFK symbol binds to the InferKit already loaded in the
//  process that loads it. No class here carries the NFK prefix, which the loader reserves for InferKit.
//
//  The models and the tool behave as the C fixture's do:
//  - "invert" inverts a 32BGRA pixel buffer under NFKInputImage (each color byte becomes 255 minus
//    itself, alpha kept) and an array of floats under "values" (each element becomes 1 minus itself).
//  - "slow" counts "steps" steps of "stepMilliseconds" each. Submitted as a job, it reports progress
//    with a partial result {"step": n} after each step and stops when the job is cancelled.
//    "exitAtStep" ends the process at that step.
//  - "echo" returns the "text" argument after the plugin's "echoPrefix" option.
//  Every result carries "pluginVersion".
//

#import <InferKit/InferKit.h>
#import <CoreVideo/CoreVideo.h>
#import <unistd.h>

#ifndef SAMPLE_PLUGIN_VERSION
#define SAMPLE_PLUGIN_VERSION "1.0.0"
#endif

static NSString * const SamplePluginVersion = @SAMPLE_PLUGIN_VERSION;
static NSString * const SamplePluginValuesKey = @"values";
static NSString * const SamplePluginVersionKey = @"pluginVersion";
static NSString * const SamplePluginStepsKey = @"steps";
static const NSInteger SamplePluginWeightBytes = 4096;

static NSError *SamplePluginError(NSString *domain, NSInteger code, NSString *message)
{
	return [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: message }];
}

static NSInteger SamplePluginIntegerParameter(NFKInferenceRequest *request, NSString *key, NSInteger fallback)
{
	id value = request.parameters[key];
	return [value isKindOfClass:NSNumber.class] ? [value integerValue] : fallback;
}

#pragma mark - Weights

/*! The state every sample backend shares: weights that load on first use and unload on request. */
@interface SamplePluginBackendBase : NSObject
@property (nonatomic, strong, nullable) NSMutableData *weights;
@property (nonatomic, readonly, copy) NSString *architecture;
@property (nonatomic, readonly) BOOL isReady;
@property (nonatomic, readonly) BOOL isLoaded;
- (void)loadWeights;
@end

@implementation SamplePluginBackendBase

- (NSString *)architecture
{
	return @"sample";
}

- (void)loadWeights
{
	@synchronized (self) {
		if (self.weights == nil) {
			self.weights = [NSMutableData dataWithLength:SamplePluginWeightBytes];
		}
	}
}

- (BOOL)isReady
{
	return YES;
}

- (BOOL)prepareWithError:(NSError **)error
{
	[self loadWeights];
	return YES;
}

- (void)unload
{
	@synchronized (self) {
		self.weights = nil;
	}
}

- (BOOL)isLoaded
{
	@synchronized (self) {
		return self.weights != nil;
	}
}

- (NSInteger)residentByteCount
{
	return self.isLoaded ? SamplePluginWeightBytes : 0;
}

- (NSInteger)peakByteCount
{
	return SamplePluginWeightBytes;
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoArchitecture: self.architecture };
}

@end

#pragma mark - Invert

@interface SamplePluginInvertBackend : SamplePluginBackendBase <NFKInferenceBackend>
@end

@implementation SamplePluginInvertBackend

- (NSString *)backendIdentifier
{
	return @"sample-invert";
}

- (NSString *)architecture
{
	return @"sample-invert";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObjects:NFKInputImage, SamplePluginValuesKey, nil];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet set];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObjects:NFKOutputImage, SamplePluginValuesKey, SamplePluginVersionKey, nil];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	id image = request.inputs[NFKInputImage];
	id values = request.inputs[SamplePluginValuesKey];
	if (image == nil && values == nil) {
		if (error != NULL) {
			*error = SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, @"invert needs \"image\" or \"values\"");
		}
		return nil;
	}
	[self loadWeights];
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:SamplePluginVersion forKey:SamplePluginVersionKey];
	if (image != nil) {
		id inverted = [self invertedImage:image error:error];
		if (inverted == nil) {
			return nil;
		}
		outputs[NFKOutputImage] = inverted;
	}
	if (values != nil) {
		NSArray<NSNumber *> *inverted = [self invertedValues:values error:error];
		if (inverted == nil) {
			return nil;
		}
		outputs[SamplePluginValuesKey] = inverted;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (nullable id)invertedImage:(id)image error:(NSError **)error
{
	BOOL pixelBuffer = CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID();
	CVPixelBufferRef source = pixelBuffer ? (__bridge CVPixelBufferRef)image : NULL;
	if (source == NULL || CVPixelBufferGetPixelFormatType(source) != kCVPixelFormatType_32BGRA) {
		if (error != NULL) {
			*error = SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, @"invert takes a 32BGRA pixel buffer");
		}
		return nil;
	}
	size_t width = CVPixelBufferGetWidth(source);
	size_t height = CVPixelBufferGetHeight(source);
	CVPixelBufferRef target = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes, &target) != kCVReturnSuccess) {
		if (error != NULL) {
			*error = SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the output pixel buffer could not be made");
		}
		return nil;
	}
	CVPixelBufferLockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferLockBaseAddress(target, 0);
	const uint8_t *from = CVPixelBufferGetBaseAddress(source);
	uint8_t *to = CVPixelBufferGetBaseAddress(target);
	size_t fromStride = CVPixelBufferGetBytesPerRow(source);
	size_t toStride = CVPixelBufferGetBytesPerRow(target);
	for (size_t row = 0; row < height; row++) {
		for (size_t column = 0; column < width * 4; column++) {
			uint8_t value = from[row * fromStride + column];
			to[row * toStride + column] = column % 4 == 3 ? value : (uint8_t)(255 - value);
		}
	}
	CVPixelBufferUnlockBaseAddress(target, 0);
	CVPixelBufferUnlockBaseAddress(source, kCVPixelBufferLock_ReadOnly);
	return (__bridge_transfer id)target;
}

- (nullable NSArray<NSNumber *> *)invertedValues:(id)values error:(NSError **)error
{
	NSArray *array = [values isKindOfClass:NSArray.class] ? values : nil;
	NSMutableArray<NSNumber *> *inverted = [NSMutableArray arrayWithCapacity:array.count];
	for (id element in array) {
		if (![element isKindOfClass:NSNumber.class]) {
			break;
		}
		BOOL single = strcmp([element objCType], @encode(float)) == 0;
		[inverted addObject:single ? @(1.0f - [element floatValue]) : @(1.0 - [element doubleValue])];
	}
	if (array == nil || inverted.count != array.count) {
		if (error != NULL) {
			*error = SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, @"invert takes an array of numbers under \"values\"");
		}
		return nil;
	}
	return inverted;
}

@end

#pragma mark - Slow

@interface SamplePluginSlowBackend : SamplePluginBackendBase <NFKInferenceBackend>
@end

@implementation SamplePluginSlowBackend

- (NSString *)backendIdentifier
{
	return @"sample-slow";
}

- (NSString *)architecture
{
	return @"sample-slow";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet set];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObjects:SamplePluginStepsKey, @"stepMilliseconds", @"exitAtStep", nil];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObjects:SamplePluginStepsKey, SamplePluginVersionKey, nil];
}

/*! Counts the request's steps, calling step after each, until isCancelled answers YES. */
- (nullable NFKInferenceResult *)countStepsOfRequest:(NFKInferenceRequest *)request
										 isCancelled:(BOOL (^)(void))isCancelled
												step:(nullable void (^)(NSInteger step, NSInteger steps))step
											   error:(NSError **)error
{
	NSInteger steps = SamplePluginIntegerParameter(request, SamplePluginStepsKey, 10);
	NSInteger stepMilliseconds = SamplePluginIntegerParameter(request, @"stepMilliseconds", 20);
	NSInteger exitAtStep = SamplePluginIntegerParameter(request, @"exitAtStep", 0);
	if (steps < 1 || stepMilliseconds < 0) {
		if (error != NULL) {
			*error = SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, @"steps is at least 1 and stepMilliseconds at least 0");
		}
		return nil;
	}
	[self loadWeights];
	for (NSInteger index = 1; index <= steps; index++) {
		[NSThread sleepForTimeInterval:stepMilliseconds / 1000.0];
		if (isCancelled()) {
			if (error != NULL) {
				*error = SamplePluginError(NSCocoaErrorDomain, NSUserCancelledError, @"the run was cancelled");
			}
			return nil;
		}
		if (index == exitAtStep) {
			_exit(70);
		}
		if (step != nil) {
			step(index, steps);
		}
	}
	return [NFKInferenceResult resultWithOutputs:@{ SamplePluginStepsKey: @(steps), SamplePluginVersionKey: SamplePluginVersion }];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [self countStepsOfRequest:request isCancelled:^BOOL { return NO; } step:nil error:error];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	__block BOOL cancelled = NO;
	NSObject *lock = [[NSObject alloc] init];
	job.cancellationHandler = ^{
		@synchronized (lock) {
			cancelled = YES;
		}
	};
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKInferenceResult *result = [self countStepsOfRequest:request isCancelled:^BOOL {
			@synchronized (lock) {
				return cancelled;
			}
		} step:^(NSInteger step, NSInteger steps) {
			NFKInferenceResult *partial = [NFKInferenceResult resultWithOutputs:@{ @"step": @(step) }];
			[job reportProgress:(double)step / (double)steps partialResult:partial];
		} error:&error];
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:error ?: SamplePluginError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the run failed")];
		}
	});
	return job;
}

@end

#pragma mark - Plugin

@interface SamplePlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@end

@implementation SamplePlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	self = [super init];
	if (self != nil) {
		_context = context;
		[context logMessage:[NSString stringWithFormat:@"sample plugin %@ started", SamplePluginVersion] level:NFKPluginLogLevelInfo];
	}
	return self;
}

- (NSArray<NSDictionary<NSString *, id> *> *)modelDictionaries
{
	NSDictionary *pluginVersion = @{ @"key": SamplePluginVersionKey, @"valueType": @"text", @"required": @YES };
	NSDictionary *invert = @{
		@"id": @"invert", @"displayName": @"Invert", @"family": @"sample-invert", @"engine": @"plugin",
		@"summary": @"Inverts a 32BGRA image or a float array.", @"tasks": @[ @"image-to-image" ],
		@"inputs": @[ @{ @"key": NFKInputImage, @"valueType": @"image" }, @{ @"key": SamplePluginValuesKey, @"valueType": @"embedding" } ],
		@"outputs": @[ @{ @"key": NFKOutputImage, @"valueType": @"image" }, @{ @"key": SamplePluginValuesKey, @"valueType": @"embedding" }, pluginVersion ],
	};
	NSDictionary *slow = @{
		@"id": @"slow", @"displayName": @"Slow", @"family": @"sample-slow", @"engine": @"plugin",
		@"summary": @"Counts steps, reporting progress after each, until it finishes or is cancelled.",
		@"outputs": @[ @{ @"key": SamplePluginStepsKey, @"valueType": @"integer", @"required": @YES }, pluginVersion ],
		@"parameters": @[
			@{ @"key": SamplePluginStepsKey, @"kind": @"integer", @"default": @10, @"minimum": @1 },
			@{ @"key": @"stepMilliseconds", @"kind": @"integer", @"default": @20, @"minimum": @0 },
			@{ @"key": @"exitAtStep", @"kind": @"integer", @"minimum": @1,
			   @"help": @"Ends the plugin's process at this step. Only for a plugin in a process of its own." },
		],
	};
	return @[ invert, slow ];
}

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	NSMutableArray<NFKModelDescriptor *> *descriptors = [NSMutableArray array];
	for (NSDictionary *dictionary in [self modelDictionaries]) {
		NSError *error = nil;
		NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:&error];
		if (descriptor == nil) {
			[self.context logMessage:[NSString stringWithFormat:@"a model description is malformed: %@", error] level:NFKPluginLogLevelError];
			continue;
		}
		[descriptors addObject:descriptor];
	}
	return descriptors;
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if ([modelIdentifier isEqualToString:@"invert"]) {
		return [[SamplePluginInvertBackend alloc] init];
	}
	if ([modelIdentifier isEqualToString:@"slow"]) {
		return [[SamplePluginSlowBackend alloc] init];
	}
	if (error != NULL) {
		*error = SamplePluginError(NFKPluginErrorDomain, NFKPluginErrorUnknownModel,
								   [NSString stringWithFormat:@"the plugin has no model \"%@\"", modelIdentifier]);
	}
	return nil;
}

- (NSArray<NFKInferenceTool *> *)tools
{
	id prefixValue = self.context.options[@"echoPrefix"];
	NSString *prefix = [prefixValue isKindOfClass:NSString.class] ? prefixValue : @"";
	NSDictionary *parameters = @{ @"type": @"object", @"properties": @{ @"text": @{ @"type": @"string" } }, @"required": @[ @"text" ] };
	NFKInferenceTool *echo = [NFKInferenceTool toolWithName:@"echo"
												description:@"Returns the text it is given, after the echoPrefix option."
												 parameters:parameters
													handler:^NFKToolResult *(NSDictionary<NSString *, id> *arguments, NSError **error) {
		return [NFKToolResult resultWithText:[prefix stringByAppendingString:arguments[@"text"]]];
	}];
	return @[ echo ];
}

- (NSArray<NFKModelOption *> *)pluginOptions
{
	NSDictionary *dictionary = @{ @"key": @"echoPrefix", @"kind": @"text", @"default": @"", @"title": @"Echo prefix" };
	NFKModelOption *option = [NFKModelOption optionWithDictionary:dictionary error:NULL];
	return option != nil ? @[ option ] : @[];
}

- (void)willUnload
{
	[self.context logMessage:[NSString stringWithFormat:@"sample plugin %@ unloading", SamplePluginVersion] level:NFKPluginLogLevelInfo];
}

@end
