//
//  NFKModelLive.m
//  InferKit
//

#import "NFKModelLive.h"
#import "NFKModelDescriptorPrivate.h"

NSString * const NFKModelLiveUnitVideoFrame	= @"videoFrame";
NSString * const NFKModelLiveUnitAudioChunk	= @"audioChunk";

#pragma mark - NFKModelLiveTiming

static NSArray<NSString *> *NFKModelLiveTimingFields(void)
{
	return @[ @"variant", @"chip", @"width", @"height", @"milliseconds", @"measured", @"note" ];
}

@implementation NFKModelLiveTiming

- (instancetype)initWithVariantIdentifier:(nullable NSString *)variantIdentifier
									 chip:(NSString *)chip
									width:(NSInteger)width
								   height:(NSInteger)height
					  millisecondsPerUnit:(double)millisecondsPerUnit
							 measuredDate:(NSString *)measuredDate
									 note:(nullable NSString *)note
							   extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_variantIdentifier = [variantIdentifier copy];
		_chip = [chip copy];
		_width = width;
		_height = height;
		_millisecondsPerUnit = millisecondsPerUnit;
		_measuredDate = [measuredDate copy];
		_note = [note copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelLiveTimingFields());
	}
	return self;
}

+ (nullable instancetype)timingWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self timingFromObject:dictionary path:@"timing" error:error];
}

+ (nullable instancetype)timingFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *variant = nil, *chip = nil, *measured = nil, *note = nil;
	NSNumber *width = nil, *height = nil, *milliseconds = nil;
	if (reader == nil
		|| ![reader readString:@"variant" required:NO into:&variant error:error]
		|| ![reader readString:@"chip" required:YES into:&chip error:error]
		|| ![reader readNumber:@"width" into:&width error:error]
		|| ![reader readNumber:@"height" into:&height error:error]
		|| ![reader readNumber:@"milliseconds" into:&milliseconds error:error]
		|| ![reader readString:@"measured" required:YES into:&measured error:error]
		|| ![reader readString:@"note" required:NO into:&note error:error]) {
		return nil;
	}
	if (milliseconds == nil) {
		[reader failField:@"milliseconds" reason:@"is missing" error:error];
		return nil;
	}
	if (!(milliseconds.doubleValue > 0)) {
		[reader failField:@"milliseconds" reason:@"is not positive" error:error];
		return nil;
	}
	if (width.integerValue < 0 || height.integerValue < 0) {
		[reader failField:width.integerValue < 0 ? @"width" : @"height" reason:@"is negative" error:error];
		return nil;
	}
	return [[self alloc] initWithVariantIdentifier:variant chip:chip width:width.integerValue height:height.integerValue
							   millisecondsPerUnit:milliseconds.doubleValue measuredDate:measured note:note
										extensions:[reader extensionsWithoutFields:NFKModelLiveTimingFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"variant"] = self.variantIdentifier;
	dictionary[@"chip"] = self.chip;
	if (self.width > 0 || self.height > 0) {
		dictionary[@"width"] = @(self.width);
		dictionary[@"height"] = @(self.height);
	}
	dictionary[@"milliseconds"] = @(self.millisecondsPerUnit);
	dictionary[@"measured"] = self.measuredDate;
	dictionary[@"note"] = self.note;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelLiveTiming.class);
}

- (NSUInteger)hash
{
	return self.chip.hash ^ (NSUInteger)(self.width * 31 + self.height);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ %ldx%ld %g ms>", NSStringFromClass(self.class), self.variantIdentifier ?: @"default",
			self.chip, (long)self.width, (long)self.height, self.millisecondsPerUnit];
}

@end

#pragma mark - NFKModelLiveProfile

static NSArray<NSString *> *NFKModelLiveProfileFields(void)
{
	return @[ @"unit", @"inputs", @"outputs", @"stateful", @"chunkDuration", @"sampleRate", @"channelCount", @"lookahead",
			  @"timings" ];
}

@implementation NFKModelLiveProfile

+ (instancetype)videoFrameProfileWithInputKeys:(NSArray<NSString *> *)inputKeys
									outputKeys:(NSArray<NSString *> *)outputKeys
									  stateful:(BOOL)stateful
									   timings:(nullable NSArray<NFKModelLiveTiming *> *)timings
{
	return [[self alloc] initWithUnit:NFKModelLiveUnitVideoFrame inputKeys:inputKeys outputKeys:outputKeys stateful:stateful
						chunkDuration:0 sampleRate:0 channelCount:0 lookaheadMilliseconds:0 timings:timings extensions:nil];
}

+ (instancetype)audioChunkProfileWithInputKeys:(NSArray<NSString *> *)inputKeys
									outputKeys:(NSArray<NSString *> *)outputKeys
									  stateful:(BOOL)stateful
								 chunkDuration:(double)chunkDuration
									sampleRate:(double)sampleRate
								  channelCount:(NSInteger)channelCount
						 lookaheadMilliseconds:(double)lookaheadMilliseconds
									   timings:(nullable NSArray<NFKModelLiveTiming *> *)timings
{
	return [[self alloc] initWithUnit:NFKModelLiveUnitAudioChunk inputKeys:inputKeys outputKeys:outputKeys stateful:stateful
						chunkDuration:chunkDuration sampleRate:sampleRate channelCount:channelCount
				lookaheadMilliseconds:lookaheadMilliseconds timings:timings extensions:nil];
}

- (instancetype)initWithUnit:(NSString *)unit
				   inputKeys:(NSArray<NSString *> *)inputKeys
				  outputKeys:(NSArray<NSString *> *)outputKeys
					stateful:(BOOL)stateful
			   chunkDuration:(double)chunkDuration
				  sampleRate:(double)sampleRate
				channelCount:(NSInteger)channelCount
	   lookaheadMilliseconds:(double)lookaheadMilliseconds
					 timings:(nullable NSArray<NFKModelLiveTiming *> *)timings
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_unit = [unit copy];
		_inputKeys = [inputKeys copy];
		_outputKeys = [outputKeys copy];
		_stateful = stateful;
		_chunkDuration = chunkDuration;
		_sampleRate = sampleRate;
		_channelCount = channelCount;
		_lookaheadMilliseconds = lookaheadMilliseconds;
		_timings = [timings copy] ?: @[];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelLiveProfileFields());
	}
	return self;
}

+ (nullable instancetype)profileWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self profileFromObject:dictionary path:@"live" error:error];
}

+ (nullable instancetype)profileFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *unit = nil;
	NSArray<NSString *> *inputs = nil, *outputs = nil;
	NSNumber *chunkDuration = nil, *sampleRate = nil, *channelCount = nil, *lookahead = nil;
	NSArray *timings = nil;
	BOOL stateful = NO;
	NFKModelElementReader timingReader = ^id(id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelLiveTiming timingFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"unit" required:YES into:&unit error:error]
		|| ![reader readStringArray:@"inputs" into:&inputs error:error]
		|| ![reader readStringArray:@"outputs" into:&outputs error:error]
		|| ![reader readNumber:@"chunkDuration" into:&chunkDuration error:error]
		|| ![reader readNumber:@"sampleRate" into:&sampleRate error:error]
		|| ![reader readNumber:@"channelCount" into:&channelCount error:error]
		|| ![reader readNumber:@"lookahead" into:&lookahead error:error]
		|| ![reader readObjects:@"timings" withReader:timingReader into:&timings error:error]) {
		return nil;
	}
	if (reader.dictionary[@"stateful"] != nil && ![reader readBoolean:@"stateful" into:&stateful error:error]) {
		return nil;
	}
	if (![self reader:reader acceptsUnit:unit inputs:inputs outputs:outputs chunkDuration:chunkDuration.doubleValue
		   sampleRate:sampleRate.doubleValue channelCount:channelCount.integerValue timings:timings error:error]) {
		return nil;
	}
	if (lookahead.doubleValue < 0) {
		[reader failField:@"lookahead" reason:@"is negative" error:error];
		return nil;
	}
	return [[self alloc] initWithUnit:unit inputKeys:inputs outputKeys:outputs stateful:stateful
						chunkDuration:chunkDuration.doubleValue sampleRate:sampleRate.doubleValue
						 channelCount:channelCount.integerValue lookaheadMilliseconds:lookahead.doubleValue timings:timings
						   extensions:[reader extensionsWithoutFields:NFKModelLiveProfileFields()]];
}

+ (BOOL)reader:(NFKModelFieldReader *)reader
	acceptsUnit:(NSString *)unit
		 inputs:(nullable NSArray<NSString *> *)inputs
		outputs:(nullable NSArray<NSString *> *)outputs
  chunkDuration:(double)chunkDuration
	 sampleRate:(double)sampleRate
   channelCount:(NSInteger)channelCount
		timings:(nullable NSArray<NFKModelLiveTiming *> *)timings
		  error:(NSError **)error
{
	BOOL audio = [unit isEqualToString:NFKModelLiveUnitAudioChunk];
	if (!audio && ![unit isEqualToString:NFKModelLiveUnitVideoFrame]) {
		return [reader failField:@"unit" reason:[NSString stringWithFormat:@"names no live unit (\"%@\")", unit] error:error];
	}
	if (inputs.count == 0) {
		return [reader failField:@"inputs" reason:@"is missing or empty" error:error];
	}
	if (outputs.count == 0) {
		return [reader failField:@"outputs" reason:@"is missing or empty" error:error];
	}
	NSDictionary<NSString *, NSNumber *> *audioFields = @{ @"chunkDuration": @(chunkDuration), @"sampleRate": @(sampleRate),
														   @"channelCount": @(channelCount) };
	for (NSString *field in [audioFields.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		double value = audioFields[field].doubleValue;
		if (audio && !(value > 0)) {
			return [reader failField:field reason:@"is not positive for an audio chunk" error:error];
		}
		if (!audio && value != 0) {
			return [reader failField:field reason:@"belongs to an audio chunk" error:error];
		}
	}
	NSUInteger index = 0;
	for (NFKModelLiveTiming *timing in timings) {
		if (!audio && (timing.width <= 0 || timing.height <= 0)) {
			return [reader failField:[NSString stringWithFormat:@"timings[%lu]", (unsigned long)index]
							  reason:@"names no frame size" error:error];
		}
		index++;
	}
	return YES;
}

- (BOOL)isAudio
{
	return [self.unit isEqualToString:NFKModelLiveUnitAudioChunk];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"unit"] = self.unit;
	dictionary[@"inputs"] = self.inputKeys;
	dictionary[@"outputs"] = self.outputKeys;
	dictionary[@"stateful"] = @(self.isStateful);
	if (self.isAudio) {
		dictionary[@"chunkDuration"] = @(self.chunkDuration);
		dictionary[@"sampleRate"] = @(self.sampleRate);
		dictionary[@"channelCount"] = @(self.channelCount);
	}
	if (self.lookaheadMilliseconds > 0) {
		dictionary[@"lookahead"] = @(self.lookaheadMilliseconds);
	}
	if (self.timings.count > 0) {
		dictionary[@"timings"] = NFKModelDictionariesOf(self.timings);
	}
	return [dictionary copy];
}

- (NSArray<NFKModelLiveTiming *> *)timingsForVariant:(nullable NSString *)variantIdentifier
{
	NSMutableArray<NFKModelLiveTiming *> *timings = [NSMutableArray array];
	for (NFKModelLiveTiming *timing in self.timings) {
		if (timing.variantIdentifier == variantIdentifier || [timing.variantIdentifier isEqualToString:variantIdentifier]) {
			[timings addObject:timing];
		}
	}
	return timings;
}

- (nullable NFKModelLiveTiming *)timingForVariant:(nullable NSString *)variantIdentifier
											width:(NSInteger)width
										   height:(NSInteger)height
											 chip:(NSString *)chip
{
	NFKModelLiveTiming *best = nil;
	for (NFKModelLiveTiming *timing in [self timingsForVariant:variantIdentifier]) {
		if (![timing.chip isEqualToString:chip]) {
			continue;
		}
		if (self.isAudio) {
			return timing;
		}
		if (timing.width < width || timing.height < height) {
			continue;
		}
		if (best == nil || timing.width * timing.height < best.width * best.height) {
			best = timing;
		}
	}
	return best;
}

- (nullable NSNumber *)keepsUpWithRate:(double)rate
							   variant:(nullable NSString *)variantIdentifier
								 width:(NSInteger)width
								height:(NSInteger)height
								  chip:(NSString *)chip
{
	NFKModelLiveTiming *timing = [self timingForVariant:variantIdentifier width:width height:height chip:chip];
	if (timing == nil || (!self.isAudio && !(rate > 0))) {
		return nil;
	}
	double budget = self.isAudio ? self.chunkDuration * 1000.0 : 1000.0 / rate;
	return @(self.isAudio ? timing.millisecondsPerUnit < budget : timing.millisecondsPerUnit <= budget);
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelLiveProfile.class);
}

- (NSUInteger)hash
{
	return self.unit.hash ^ self.inputKeys.hash ^ self.outputKeys.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ → %@%@, %lu timings>", NSStringFromClass(self.class), self.unit,
			[self.inputKeys componentsJoinedByString:@","], [self.outputKeys componentsJoinedByString:@","],
			self.isStateful ? @", stateful" : @"", (unsigned long)self.timings.count];
}

@end

#pragma mark - NFKModelCatalog (NFKModelLive)

@implementation NFKModelCatalog (NFKModelLive)

- (NSArray<NFKModelDescriptor *> *)liveDescriptorsForUnit:(nullable NSString *)unit
{
	NSMutableArray<NFKModelDescriptor *> *live = [NSMutableArray array];
	for (NFKModelDescriptor *descriptor in self.visibleDescriptors) {
		NFKModelLiveProfile *profile = descriptor.liveProfile;
		if (profile != nil && (unit == nil || [profile.unit isEqualToString:unit])) {
			[live addObject:descriptor];
		}
	}
	[live sortUsingComparator:^NSComparisonResult(NFKModelDescriptor *a, NFKModelDescriptor *b) {
		return [a.identifier compare:b.identifier];
	}];
	return live;
}

@end
