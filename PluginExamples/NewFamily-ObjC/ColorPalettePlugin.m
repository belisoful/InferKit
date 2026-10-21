//
//  ColorPalettePlugin.m
//  InferKit plugin examples
//
//  An Objective-C plugin that introduces the "color-palette" model family. Each model reads a 32BGRA
//  pixel buffer under NFKInputImage and returns the image's dominant colors under NFKOutputStructured,
//  found by k-means clustering over a sample of its pixels. The bundle's manifest declares the models
//  and the options, and the plugin reads them from there, so the two never disagree.
//
//  The bundle binds to the InferKit already loaded in the process that loads it, so no class here
//  carries the NFK prefix.
//

#import <InferKit/InferKit.h>
#import <CoreVideo/CoreVideo.h>

/*! The request parameter that sets how many pixels a model reads. Part of the family contract. */
static NSString * const ColorPaletteSamplesKey = @"samples";
/*! The fields of the structured output. Part of the family contract. */
static NSString * const ColorPaletteColorsField = @"colors";
static NSString * const ColorPaletteSampledPixelsField = @"sampledPixels";
/*! The plugin option that orders each palette: "share" (most common first) or "lightness". */
static NSString * const ColorPaletteOrderOption = @"order";
static NSString * const ColorPaletteOrderLightness = @"lightness";

static const NSInteger ColorPaletteDefaultSamples = 4096;
static const NSUInteger ColorPaletteIterations = 16;
static const uint8_t ColorPaletteMinimumAlpha = 128;

static NSError *ColorPaletteError(NSInteger code, NSString *message)
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: message }];
}

#pragma mark - Sampling and clustering

typedef struct ColorPaletteColor {
	float red;
	float green;
	float blue;
} ColorPaletteColor;

/*! A fixed-seed generator, so the same image and settings always give the same palette. */
static uint64_t ColorPaletteNextRandom(uint64_t *state)
{
	*state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
	return *state >> 11;
}

static float ColorPaletteDistance(ColorPaletteColor a, ColorPaletteColor b)
{
	float red = a.red - b.red;
	float green = a.green - b.green;
	float blue = a.blue - b.blue;
	return red * red + green * green + blue * blue;
}

/*! Reads up to count pixels at positions drawn from the generator, every pixel when count covers the
	image, and skips pixels whose alpha is below ColorPaletteMinimumAlpha. Returns the number read. */
static NSUInteger ColorPaletteSamplePixels(CVPixelBufferRef buffer, NSUInteger count, ColorPaletteColor *samples)
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
		uint64_t position = everyPixel ? index : ColorPaletteNextRandom(&state) % total;
		const uint8_t *pixel = bytes + (position / width) * stride + (position % width) * 4;
		if (pixel[3] < ColorPaletteMinimumAlpha) {
			continue;
		}
		samples[taken++] = (ColorPaletteColor){ .red = pixel[2], .green = pixel[1], .blue = pixel[0] };
	}
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	return taken;
}

/*! Picks up to k starting centers by k-means++: each next center is drawn with a probability that grows
	with its squared distance from the centers already picked. Fewer than k distinct colors give fewer
	centers. Returns the number picked. */
static NSUInteger ColorPaletteSeedCenters(const ColorPaletteColor *samples, NSUInteger count, NSUInteger k,
										  ColorPaletteColor *centers, float *distances)
{
	uint64_t state = 0x2545F4914F6CDD1DULL;
	centers[0] = samples[ColorPaletteNextRandom(&state) % count];
	for (NSUInteger index = 0; index < count; index++) {
		distances[index] = ColorPaletteDistance(samples[index], centers[0]);
	}
	NSUInteger picked = 1;
	while (picked < k) {
		double sum = 0;
		for (NSUInteger index = 0; index < count; index++) {
			sum += distances[index];
		}
		if (sum <= 0) {
			break;
		}
		double target = (double)(ColorPaletteNextRandom(&state) % 1000000) / 1000000.0 * sum;
		NSUInteger chosen = count - 1;
		for (NSUInteger index = 0; index < count; index++) {
			target -= distances[index];
			if (target < 0) {
				chosen = index;
				break;
			}
		}
		centers[picked] = samples[chosen];
		for (NSUInteger index = 0; index < count; index++) {
			distances[index] = MIN(distances[index], ColorPaletteDistance(samples[index], centers[picked]));
		}
		picked++;
	}
	return picked;
}

/*! Clusters the samples around at most k centers by Lloyd's algorithm from a k-means++ start, and
	writes each center's member count. Returns the number of centers that kept members. */
static NSUInteger ColorPaletteCluster(const ColorPaletteColor *samples, NSUInteger count, NSUInteger k,
									  ColorPaletteColor *centers, NSUInteger *members)
{
	float *distances = calloc(count, sizeof *distances);
	NSUInteger *assignments = calloc(count, sizeof *assignments);
	ColorPaletteColor *sums = calloc(k, sizeof *sums);
	if (distances == NULL || assignments == NULL || sums == NULL) {
		free(distances);
		free(assignments);
		free(sums);
		return 0;
	}
	NSUInteger centerCount = ColorPaletteSeedCenters(samples, count, k, centers, distances);
	for (NSUInteger iteration = 0; iteration < ColorPaletteIterations; iteration++) {
		BOOL moved = NO;
		memset(sums, 0, k * sizeof *sums);
		memset(members, 0, k * sizeof *members);
		for (NSUInteger index = 0; index < count; index++) {
			NSUInteger nearest = 0;
			float best = ColorPaletteDistance(samples[index], centers[0]);
			for (NSUInteger center = 1; center < centerCount; center++) {
				float distance = ColorPaletteDistance(samples[index], centers[center]);
				if (distance < best) {
					best = distance;
					nearest = center;
				}
			}
			moved = moved || assignments[index] != nearest || iteration == 0;
			assignments[index] = nearest;
			sums[nearest].red += samples[index].red;
			sums[nearest].green += samples[index].green;
			sums[nearest].blue += samples[index].blue;
			members[nearest]++;
		}
		for (NSUInteger center = 0; center < centerCount; center++) {
			if (members[center] > 0) {
				float n = (float)members[center];
				centers[center] = (ColorPaletteColor){ sums[center].red / n, sums[center].green / n, sums[center].blue / n };
			}
		}
		if (!moved) {
			break;
		}
	}
	NSUInteger kept = 0;
	for (NSUInteger center = 0; center < centerCount; center++) {
		if (members[center] > 0) {
			centers[kept] = centers[center];
			members[kept] = members[center];
			kept++;
		}
	}
	free(distances);
	free(assignments);
	free(sums);
	return kept;
}

#pragma mark - Output

static NSDictionary<NSString *, id> *ColorPaletteEntry(ColorPaletteColor color, double share)
{
	long red = lroundf(color.red);
	long green = lroundf(color.green);
	long blue = lroundf(color.blue);
	return @{
		@"hex": [NSString stringWithFormat:@"#%02lX%02lX%02lX", red, green, blue],
		@"red": @(red),
		@"green": @(green),
		@"blue": @(blue),
		@"share": @(share),
	};
}

/*! Relative luminance from 8-bit channels, for the "lightness" order. */
static double ColorPaletteLightness(NSDictionary<NSString *, id> *entry)
{
	return 0.2126 * [entry[@"red"] doubleValue] + 0.7152 * [entry[@"green"] doubleValue] + 0.0722 * [entry[@"blue"] doubleValue];
}

static NSArray<NSDictionary<NSString *, id> *> *ColorPaletteOrdered(NSArray<NSDictionary<NSString *, id> *> *entries, NSString *order)
{
	BOOL byLightness = [order isEqualToString:ColorPaletteOrderLightness];
	return [entries sortedArrayUsingComparator:^NSComparisonResult(NSDictionary *first, NSDictionary *second) {
		double a = byLightness ? ColorPaletteLightness(first) : -[first[@"share"] doubleValue];
		double b = byLightness ? ColorPaletteLightness(second) : -[second[@"share"] doubleValue];
		return a < b ? NSOrderedAscending : (a > b ? NSOrderedDescending : NSOrderedSame);
	}];
}

#pragma mark - Backend

/*! One palette model: a k-means clustering into a fixed number of colors. Stateless, so runs may
	overlap, which the manifest's maximumConcurrentRuns allows. */
@interface ColorPaletteKMeansBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, readonly, copy) NSString *modelIdentifier;
@property (nonatomic, readonly) NSUInteger colorCount;
@property (nonatomic, readonly) NSInteger defaultSamples;
@property (nonatomic, readonly, copy) NSString *order;
- (instancetype)initWithModelIdentifier:(NSString *)modelIdentifier
							 colorCount:(NSUInteger)colorCount
						 defaultSamples:(NSInteger)defaultSamples
								  order:(NSString *)order;
@end

@implementation ColorPaletteKMeansBackend

- (instancetype)initWithModelIdentifier:(NSString *)modelIdentifier
							 colorCount:(NSUInteger)colorCount
						 defaultSamples:(NSInteger)defaultSamples
								  order:(NSString *)order
{
	self = [super init];
	if (self != nil) {
		_modelIdentifier = [modelIdentifier copy];
		_colorCount = colorCount;
		_defaultSamples = defaultSamples;
		_order = [order copy];
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return self.modelIdentifier;
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
	return [NSSet setWithObject:ColorPaletteSamplesKey];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObject:NFKOutputStructured];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return @{ NFKModelInfoArchitecture: @"color-palette" };
}

- (NSInteger)samplesForRequest:(NFKInferenceRequest *)request
{
	id value = request.parameters[ColorPaletteSamplesKey];
	return [value isKindOfClass:NSNumber.class] ? [value integerValue] : self.defaultSamples;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	id image = request.inputs[NFKInputImage];
	if (image == nil) {
		if (error != NULL) {
			*error = ColorPaletteError(kNFKError_InferenceMissingInput, @"a palette needs an image under \"image\"");
		}
		return nil;
	}
	BOOL isPixelBuffer = CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID();
	CVPixelBufferRef buffer = isPixelBuffer ? (__bridge CVPixelBufferRef)image : NULL;
	if (buffer == NULL || CVPixelBufferGetPixelFormatType(buffer) != kCVPixelFormatType_32BGRA) {
		if (error != NULL) {
			*error = ColorPaletteError(kNFKError_InferenceUnsupported, @"a palette reads a 32BGRA pixel buffer");
		}
		return nil;
	}
	NSInteger samples = [self samplesForRequest:request];
	if (samples < 1) {
		if (error != NULL) {
			*error = ColorPaletteError(kNFKError_InferenceUnsupported, @"\"samples\" is at least 1");
		}
		return nil;
	}
	return [self paletteOfPixelBuffer:buffer samples:(NSUInteger)samples error:error];
}

- (nullable NFKInferenceResult *)paletteOfPixelBuffer:(CVPixelBufferRef)buffer samples:(NSUInteger)samples error:(NSError **)error
{
	ColorPaletteColor *pixels = calloc(samples, sizeof *pixels);
	ColorPaletteColor *centers = calloc(self.colorCount, sizeof *centers);
	NSUInteger *members = calloc(self.colorCount, sizeof *members);
	if (pixels == NULL || centers == NULL || members == NULL) {
		free(pixels);
		free(centers);
		free(members);
		if (error != NULL) {
			*error = ColorPaletteError(kNFKError_InferenceBackendFailure, @"the samples could not be allocated");
		}
		return nil;
	}
	NSUInteger read = ColorPaletteSamplePixels(buffer, samples, pixels);
	NSUInteger kept = read > 0 ? ColorPaletteCluster(pixels, read, self.colorCount, centers, members) : 0;
	NSMutableArray<NSDictionary<NSString *, id> *> *entries = [NSMutableArray arrayWithCapacity:kept];
	for (NSUInteger index = 0; index < kept; index++) {
		[entries addObject:ColorPaletteEntry(centers[index], (double)members[index] / (double)read)];
	}
	free(pixels);
	free(centers);
	free(members);
	NSDictionary *palette = @{
		ColorPaletteColorsField: ColorPaletteOrdered(entries, self.order),
		ColorPaletteSampledPixelsField: @(read),
	};
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: palette }];
}

@end

#pragma mark - Plugin

@interface ColorPalettePlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@property (nonatomic, strong, nullable) NFKPluginManifest *manifest;
@end

@implementation ColorPalettePlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	self = [super init];
	if (self != nil) {
		_context = context;
		NSError *error = nil;
		NSURL *bundleURL = [NSBundle bundleForClass:self.class].bundleURL;
		_manifest = [NFKPluginManifest manifestWithBundleURL:bundleURL error:&error];
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

- (NSArray<NFKModelOption *> *)pluginOptions
{
	return self.manifest.options ?: @[];
}

/*! A plugin option's value: the app's value when it set one, else the manifest's default. The context
	carries only the values the app set. */
- (nullable id)optionValueForKey:(NSString *)key
{
	id value = self.context.options[key];
	if (value != nil) {
		return value;
	}
	for (NFKModelOption *option in self.manifest.options) {
		if ([option.key isEqualToString:key]) {
			return option.defaultValue;
		}
	}
	return nil;
}

/*! The number of colors each model returns, by model identifier. */
+ (NSDictionary<NSString *, NSNumber *> *)colorCounts
{
	return @{ @"palette-5": @5, @"palette-8": @8 };
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	NSNumber *colorCount = self.class.colorCounts[modelIdentifier];
	if (colorCount == nil) {
		if (error != NULL) {
			NSString *message = [NSString stringWithFormat:@"the plugin has no model \"%@\"", modelIdentifier];
			*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel userInfo:@{ NSLocalizedDescriptionKey: message }];
		}
		return nil;
	}
	id samples = options[ColorPaletteSamplesKey];
	id order = [self optionValueForKey:ColorPaletteOrderOption];
	return [[ColorPaletteKMeansBackend alloc] initWithModelIdentifier:modelIdentifier
														  colorCount:colorCount.unsignedIntegerValue
													  defaultSamples:[samples isKindOfClass:NSNumber.class] ? [samples integerValue] : ColorPaletteDefaultSamples
															   order:[order isKindOfClass:NSString.class] ? order : @"share"];
}

@end
