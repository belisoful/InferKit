//
//  MedianCutPalettePlugin.m
//  InferKit plugin examples
//
//  An Objective-C plugin whose model joins the "color-palette" family. The family's contract is its
//  ports and its option: a 32BGRA pixel buffer under NFKInputImage, a palette under
//  NFKOutputStructured, and the request parameter "samples". This model meets the contract with the
//  median-cut algorithm: it splits the sampled colors at the median of their widest channel until it
//  holds eight boxes, and reports each box's mean color.
//
//  The bundle binds to the InferKit already loaded in the process that loads it, so no class here
//  carries the NFK prefix.
//

#import <InferKit/InferKit.h>
#import <CoreVideo/CoreVideo.h>

/*! The family contract's request parameter and structured-output fields. */
static NSString * const MedianCutSamplesKey = @"samples";
static NSString * const MedianCutColorsField = @"colors";
static NSString * const MedianCutSampledPixelsField = @"sampledPixels";

static NSString * const MedianCutModelIdentifier = @"median-cut-8";
enum { MedianCutColorCount = 8 };
static const NSInteger MedianCutDefaultSamples = 4096;
static const uint8_t MedianCutMinimumAlpha = 128;

static NSError *MedianCutError(NSInteger code, NSString *message)
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: message }];
}

#pragma mark - Sampling and splitting

/*! One sampled pixel, its channels in the order red, green, blue. */
typedef struct MedianCutPixel {
	uint8_t channels[3];
} MedianCutPixel;

/*! A run of the sample array that one palette color stands for. */
typedef struct MedianCutBox {
	NSUInteger start;
	NSUInteger count;
} MedianCutBox;

/*! A fixed-seed generator, so the same image and settings always give the same palette. */
static uint64_t MedianCutNextRandom(uint64_t *state)
{
	*state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
	return *state >> 11;
}

/*! Reads up to count pixels at positions drawn from the generator, every pixel when count covers the
	image, and skips pixels whose alpha is below MedianCutMinimumAlpha. Returns the number read. */
static NSUInteger MedianCutSamplePixels(CVPixelBufferRef buffer, NSUInteger count, MedianCutPixel *samples)
{
	size_t width = CVPixelBufferGetWidth(buffer);
	size_t height = CVPixelBufferGetHeight(buffer);
	size_t stride = CVPixelBufferGetBytesPerRow(buffer);
	uint64_t total = (uint64_t)width * height;
	BOOL everyPixel = count >= total;
	NSUInteger positions = everyPixel ? (NSUInteger)total : count;
	uint64_t state = 0x9E3779B97F4A7C15ULL;
	NSUInteger taken = 0;
	CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	const uint8_t *bytes = CVPixelBufferGetBaseAddress(buffer);
	for (NSUInteger index = 0; index < positions; index++) {
		uint64_t position = everyPixel ? index : MedianCutNextRandom(&state) % total;
		const uint8_t *pixel = bytes + (position / width) * stride + (position % width) * 4;
		if (pixel[3] < MedianCutMinimumAlpha) {
			continue;
		}
		samples[taken++] = (MedianCutPixel){ .channels = { pixel[2], pixel[1], pixel[0] } };
	}
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	return taken;
}

/*! The channel with the widest range in a box, written to *outChannel, and that range. */
static int MedianCutWidestRange(const MedianCutPixel *samples, MedianCutBox box, int *outChannel)
{
	int widest = -1;
	for (int channel = 0; channel < 3; channel++) {
		int low = 255;
		int high = 0;
		for (NSUInteger index = box.start; index < box.start + box.count; index++) {
			low = MIN(low, samples[index].channels[channel]);
			high = MAX(high, samples[index].channels[channel]);
		}
		if (high - low > widest) {
			widest = high - low;
			*outChannel = channel;
		}
	}
	return widest;
}

/*! Sorts a box's pixels by one channel with a counting sort, which needs no comparator context. */
static BOOL MedianCutSortByChannel(MedianCutPixel *samples, MedianCutBox box, int channel)
{
	NSUInteger counts[257] = { 0 };
	MedianCutPixel *sorted = malloc(box.count * sizeof *sorted);
	if (sorted == NULL) {
		return NO;
	}
	for (NSUInteger index = box.start; index < box.start + box.count; index++) {
		counts[samples[index].channels[channel] + 1]++;
	}
	for (int value = 1; value < 257; value++) {
		counts[value] += counts[value - 1];
	}
	for (NSUInteger index = box.start; index < box.start + box.count; index++) {
		sorted[counts[samples[index].channels[channel]]++] = samples[index];
	}
	memcpy(samples + box.start, sorted, box.count * sizeof *sorted);
	free(sorted);
	return YES;
}

/*! Where to split a box sorted by channel: the boundary between two values nearest the median. Splitting
	inside a run of one value would leave that value in both halves, so the split moves to the run's
	nearer end. The box holds at least two values, so one end lies strictly inside it. */
static NSUInteger MedianCutSplitIndex(const MedianCutPixel *samples, MedianCutBox box, int channel)
{
	NSUInteger half = box.count / 2;
	uint8_t median = samples[box.start + half].channels[channel];
	NSUInteger lower = half;
	while (lower > 0 && samples[box.start + lower - 1].channels[channel] == median) {
		lower--;
	}
	NSUInteger upper = half;
	while (upper < box.count && samples[box.start + upper].channels[channel] == median) {
		upper++;
	}
	if (lower == 0) {
		return upper;
	}
	if (upper == box.count) {
		return lower;
	}
	return half - lower <= upper - half ? lower : upper;
}

/*! Splits the box with the widest channel range at its median until there are k boxes or no box holds
	two distinct colors. Returns the number of boxes. */
static NSUInteger MedianCutSplit(MedianCutPixel *samples, NSUInteger count, NSUInteger k, MedianCutBox *boxes)
{
	boxes[0] = (MedianCutBox){ .start = 0, .count = count };
	NSUInteger boxCount = 1;
	while (boxCount < k) {
		NSUInteger chosen = NSNotFound;
		int chosenChannel = 0;
		int widest = 0;
		for (NSUInteger index = 0; index < boxCount; index++) {
			int channel = 0;
			int range = boxes[index].count > 1 ? MedianCutWidestRange(samples, boxes[index], &channel) : 0;
			if (range > widest) {
				widest = range;
				chosen = index;
				chosenChannel = channel;
			}
		}
		if (chosen == NSNotFound || !MedianCutSortByChannel(samples, boxes[chosen], chosenChannel)) {
			break;
		}
		MedianCutBox box = boxes[chosen];
		NSUInteger split = MedianCutSplitIndex(samples, box, chosenChannel);
		boxes[chosen] = (MedianCutBox){ .start = box.start, .count = split };
		boxes[boxCount++] = (MedianCutBox){ .start = box.start + split, .count = box.count - split };
	}
	return boxCount;
}

static NSDictionary<NSString *, id> *MedianCutEntry(const MedianCutPixel *samples, MedianCutBox box, NSUInteger total)
{
	double sums[3] = { 0, 0, 0 };
	for (NSUInteger index = box.start; index < box.start + box.count; index++) {
		for (int channel = 0; channel < 3; channel++) {
			sums[channel] += samples[index].channels[channel];
		}
	}
	long red = lround(sums[0] / box.count);
	long green = lround(sums[1] / box.count);
	long blue = lround(sums[2] / box.count);
	return @{
		@"hex": [NSString stringWithFormat:@"#%02lX%02lX%02lX", red, green, blue],
		@"red": @(red),
		@"green": @(green),
		@"blue": @(blue),
		@"share": @((double)box.count / (double)total),
	};
}

#pragma mark - Backend

/*! The median-cut palette model. Stateless, so runs may overlap. */
@interface MedianCutPaletteBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, readonly) NSInteger defaultSamples;
- (instancetype)initWithDefaultSamples:(NSInteger)defaultSamples;
@end

@implementation MedianCutPaletteBackend

- (instancetype)initWithDefaultSamples:(NSInteger)defaultSamples
{
	self = [super init];
	if (self != nil) {
		_defaultSamples = defaultSamples;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return MedianCutModelIdentifier;
}

- (BOOL)isReady
{
	return YES;
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObject:MedianCutSamplesKey];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObject:NFKOutputStructured];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoArchitecture: @"color-palette" };
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	id image = request.inputs[NFKInputImage];
	if (image == nil) {
		if (error != NULL) {
			*error = MedianCutError(kNFKError_InferenceMissingInput, @"a palette needs an image under \"image\"");
		}
		return nil;
	}
	BOOL isPixelBuffer = CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID();
	CVPixelBufferRef buffer = isPixelBuffer ? (__bridge CVPixelBufferRef)image : NULL;
	if (buffer == NULL || CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_32BGRA) {
		if (error != NULL) {
			*error = MedianCutError(kNFKError_InferenceUnsupported, @"a palette reads a 32BGRA pixel buffer");
		}
		return nil;
	}
	id value = request.parameters[MedianCutSamplesKey];
	NSInteger samples = [value isKindOfClass:NSNumber.class] ? [value integerValue] : self.defaultSamples;
	if (samples < 1) {
		if (error != NULL) {
			*error = MedianCutError(kNFKError_InferenceUnsupported, @"\"samples\" is at least 1");
		}
		return nil;
	}
	MedianCutPixel *pixels = calloc((size_t)samples, sizeof *pixels);
	if (pixels == NULL) {
		if (error != NULL) {
			*error = MedianCutError(kNFKError_InferenceBackendFailure, @"the samples could not be allocated");
		}
		return nil;
	}
	NSUInteger read = MedianCutSamplePixels(buffer, (NSUInteger)samples, pixels);
	MedianCutBox boxes[MedianCutColorCount];
	NSUInteger boxCount = read > 0 ? MedianCutSplit(pixels, read, MedianCutColorCount, boxes) : 0;
	NSMutableArray<NSDictionary<NSString *, id> *> *entries = [NSMutableArray arrayWithCapacity:boxCount];
	for (NSUInteger index = 0; index < boxCount; index++) {
		[entries addObject:MedianCutEntry(pixels, boxes[index], read)];
	}
	free(pixels);
	NSArray *ordered = [entries sortedArrayUsingDescriptors:@[ [NSSortDescriptor sortDescriptorWithKey:@"share" ascending:NO] ]];
	NSDictionary *palette = @{ MedianCutColorsField: ordered, MedianCutSampledPixelsField: @(read) };
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: palette }];
}

@end

#pragma mark - Plugin

@interface MedianCutPalettePlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@property (nonatomic, strong, nullable) NFKPluginManifest *manifest;
@end

@implementation MedianCutPalettePlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	self = [super init];
	if (self != nil) {
		_context = context;
		NSError *error = nil;
		_manifest = [NFKPluginManifest manifestWithBundleURL:[NSBundle bundleForClass:self.class].bundleURL error:&error];
		if (_manifest == nil) {
			[context logMessage:[NSString stringWithFormat:@"the manifest could not be read: %@", error] level:NFKPluginLogLevelError];
		}
	}
	return self;
}

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return self.manifest.models ?: @[];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if (![modelIdentifier isEqualToString:MedianCutModelIdentifier]) {
		if (error != NULL) {
			NSString *message = [NSString stringWithFormat:@"the plugin has no model \"%@\"", modelIdentifier];
			*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel userInfo:@{ NSLocalizedDescriptionKey: message }];
		}
		return nil;
	}
	id samples = options[MedianCutSamplesKey];
	return [[MedianCutPaletteBackend alloc] initWithDefaultSamples:[samples isKindOfClass:NSNumber.class] ? [samples integerValue] : MedianCutDefaultSamples];
}

@end
