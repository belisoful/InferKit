//
//  NFKModelMemory.m
//  InferKit
//

#import "NFKModelMemory.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelCatalog.h"
#import "NFKModelOptions.h"
#import "NFKHardwareProfile.h"

NSString * const NFKModelMemoryBasisMeasured		= @"measured";
NSString * const NFKModelMemoryBasisStoredDType		= @"stored-dtype";
NSString * const NFKModelMemoryBasisDeclared		= @"declared";
NSString * const NFKModelMemoryBasisUnknown			= @"unknown";

NSString * const NFKModelMemoryComponentWeights			= @"weights";
NSString * const NFKModelMemoryComponentKeyValueCache	= @"keyValueCache";
NSString * const NFKModelMemoryComponentDraft			= @"draft";
NSString * const NFKModelMemoryComponentExpertCache		= @"expertCache";
NSString * const NFKModelMemoryComponentLinearOptions	= @"linearOptions";
NSString * const NFKModelMemoryComponentLoadOverhead	= @"loadOverhead";

NSString * const NFKModelMemoryEstimateKey = @"NFKModelMemoryEstimate";
NSString * const NFKModelMemoryFittingOptionsKey = @"NFKModelMemoryFittingOptions";
NSString * const NFKModelMemoryFittingEstimateKey = @"NFKModelMemoryFittingEstimate";

static const double NFKModelWorkingSetFraction = 0.85;
static const NSInteger NFKModelMaximumReserveByteCount = (NSInteger)4 << 30;
static const NSInteger NFKModelDefaultExpertCacheByteCount = (NSInteger)4 << 30;
static const NSInteger NFKModelDefaultQuantizationGroupSize = 64;
static const NSUInteger NFKModelMaximumCombinationCount = 4096;
static const NSInteger NFKModelMaximumLayerCount = 4096;

#pragma mark - Budget rules

static NSString *NFKModelGiB(NSInteger bytes)
{
	return [NSString stringWithFormat:@"%.1f GiB", (double)bytes / 1073741824.0];
}

static NSInteger NFKModelReserveByteCount(NSInteger budget)
{
	if (budget <= 0) {
		return 0;
	}
	return MIN(NFKModelMaximumReserveByteCount, budget - budget / 2);
}

static NSString *NFKModelReserveGiB(NSInteger budget)
{
	return NFKModelGiB(NFKModelReserveByteCount(budget));
}

static BOOL NFKModelBudgetHolds(NSInteger bytes, NSInteger budget)
{
	return budget > 0 && bytes + NFKModelReserveByteCount(budget) <= budget;
}

static BOOL NFKModelBudgetAdmits(NSInteger bytes, NSInteger budget)
{
	return budget <= 0 || bytes + NFKModelReserveByteCount(budget) <= budget;
}

static NSInteger NFKModelBudgetOfWorkingSet(NSInteger workingSet)
{
	return workingSet > 0 ? (NSInteger)((double)workingSet * NFKModelWorkingSetFraction) : 0;
}

static NSError *NFKModelDoesNotFitError(NSString *reason, NFKModelMemoryEstimate * _Nullable estimate)
{
	NSMutableDictionary<NSString *, id> *userInfo = [NSMutableDictionary dictionaryWithObject:reason
																					   forKey:NSLocalizedDescriptionKey];
	userInfo[NFKModelMemoryEstimateKey] = estimate;
	return [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorDoesNotFit userInfo:userInfo];
}

#pragma mark - Precisions

// The bytes of one element of a precision, 0 for a precision the core does not size.
static double NFKModelElementByteCount(NSString * _Nullable precision)
{
	static NSDictionary<NSString *, NSNumber *> *sizes;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		sizes = @{ @"float64": @8, NFKModelPrecisionFloat32: @4, NFKModelPrecisionFloat16: @2,
				   NFKModelPrecisionBFloat16: @2, @"float8_e4m3": @1, @"float8_e5m2": @1 };
	});
	return precision != nil ? sizes[precision].doubleValue : 0;
}

// A precision with NFKModelPrecisionCheckpoint replaced by the type the checkpoint is stored in.
static NSString * _Nullable NFKModelConcretePrecision(NSString *precision, NSString * _Nullable checkpointPrecision)
{
	return [precision isEqualToString:NFKModelPrecisionCheckpoint] ? checkpointPrecision : precision;
}

// The bytes a dictionary states at a precision. A precision it does not list takes the figure of a listed
// precision of the same element size (float16 reads bfloat16's), else converts from the checkpoint's
// bytes, else from its first listed precision, by the ratio of element sizes; integer and packed tensors
// convert with the floating ones, so the converted figure is an approximation.
static NSInteger NFKModelBytesAtPrecision(NSDictionary<NSString *, NSNumber *> *bytes, NSString *precision,
										  NSString * _Nullable checkpointPrecision)
{
	if (bytes.count == 0) {
		return 0;
	}
	NSNumber *exact = bytes[precision];
	if (exact != nil) {
		return exact.integerValue;
	}
	NSString *concrete = NFKModelConcretePrecision(precision, checkpointPrecision);
	if (concrete != nil && bytes[concrete] != nil) {
		return bytes[concrete].integerValue;
	}
	if (checkpointPrecision != nil && [precision isEqualToString:checkpointPrecision]
		&& bytes[NFKModelPrecisionCheckpoint] != nil) {
		return bytes[NFKModelPrecisionCheckpoint].integerValue;
	}
	double wanted = NFKModelElementByteCount(concrete);
	NSArray<NSString *> *listed = [bytes.allKeys sortedArrayUsingSelector:@selector(compare:)];
	for (NSString *key in listed) {
		if (wanted > 0 && NFKModelElementByteCount(key) == wanted) {
			return bytes[key].integerValue;
		}
	}
	NSString *sourceKey = bytes[NFKModelPrecisionCheckpoint] != nil ? NFKModelPrecisionCheckpoint : listed.firstObject;
	NSInteger source = bytes[sourceKey].integerValue;
	double stated = NFKModelElementByteCount(NFKModelConcretePrecision(sourceKey, checkpointPrecision));
	if (wanted <= 0 || stated <= 0) {
		return source;
	}
	return (NSInteger)llround((double)source * wanted / stated);
}

// The bytes elements take quantized to bits in groups: bits/8 each, and a 16-bit scale and bias per group.
static NSInteger NFKModelQuantizedByteCount(double elements, NSInteger bits, NSInteger groupSize)
{
	NSInteger group = groupSize > 0 ? groupSize : NFKModelDefaultQuantizationGroupSize;
	return (NSInteger)ceil(elements * ((double)bits / 8.0 + 4.0 / (double)group));
}

// An option value read as an integer: a number, or a string of one. Any other value reads as 0.
static NSInteger NFKModelIntegerValue(id _Nullable value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return [value integerValue];
	}
	if ([value isKindOfClass:NSString.class]) {
		return [value integerValue];
	}
	return 0;
}

// YES for a number that is a whole number from minimum to maximum. NaN and infinities are not.
static BOOL NFKModelIsWholeNumberInRange(NSNumber *number, NSInteger minimum, NSInteger maximum)
{
	double value = number.doubleValue;
	return value >= (double)minimum && value <= (double)maximum && value == floor(value);
}

#pragma mark - NFKModelMemoryBudget

@implementation NFKModelMemoryBudget

+ (double)workingSetFraction
{
	return NFKModelWorkingSetFraction;
}

+ (NSInteger)maximumReserveByteCount
{
	return NFKModelMaximumReserveByteCount;
}

+ (NSInteger)reserveByteCountForBudgetByteCount:(NSInteger)budgetByteCount
{
	return NFKModelReserveByteCount(budgetByteCount);
}

static NSString * const NFKModelScopedBudgetKey = @"NFKModelScopedBudget";

+ (instancetype)currentBudget
{
	NFKModelMemoryBudget *scoped = NSThread.currentThread.threadDictionary[NFKModelScopedBudgetKey];
	return scoped ?: [self budgetForHardwareProfile:NFKHardwareProfile.currentProfile];
}

+ (void)withCurrentBudget:(NFKModelMemoryBudget *)budget perform:(NS_NOESCAPE void (^)(void))block
{
	NSParameterAssert(budget != nil);
	NSMutableDictionary *dictionary = NSThread.currentThread.threadDictionary;
	id outer = dictionary[NFKModelScopedBudgetKey];
	dictionary[NFKModelScopedBudgetKey] = budget;
	@try {
		block();
	} @finally {
		dictionary[NFKModelScopedBudgetKey] = outer;
	}
}

+ (instancetype)budgetForHardwareProfile:(NFKHardwareProfile *)profile
{
	return [self budgetWithWorkingSetByteCount:profile.recommendedWorkingSetSize];
}

+ (instancetype)budgetWithWorkingSetByteCount:(NSInteger)workingSetByteCount
{
	return [[self alloc] initWithWorkingSetByteCount:workingSetByteCount];
}

- (instancetype)initWithWorkingSetByteCount:(NSInteger)workingSetByteCount
{
	self = [super init];
	if (self != nil) {
		_workingSetByteCount = MAX(workingSetByteCount, 0);
		_byteCount = NFKModelBudgetOfWorkingSet(_workingSetByteCount);
	}
	return self;
}

- (BOOL)isKnown
{
	return self.byteCount > 0;
}

- (NSInteger)reserveByteCount
{
	return NFKModelReserveByteCount(self.byteCount);
}

- (BOOL)holdsByteCount:(NSInteger)bytes
{
	return NFKModelBudgetHolds(bytes, self.byteCount);
}

- (BOOL)admitsByteCount:(NSInteger)bytes
{
	return NFKModelBudgetAdmits(bytes, self.byteCount);
}

+ (NSInteger)leastBudgetByteCountHoldingByteCount:(NSInteger)bytes
{
	NSInteger held = MIN(MAX(bytes, 0), NSIntegerMax - NFKModelMaximumReserveByteCount);
	return MAX(held + MIN(held, NFKModelMaximumReserveByteCount), 1);
}

+ (NSInteger)leastWorkingSetHoldingByteCount:(NSInteger)bytes
{
	NSInteger budget = [self leastBudgetByteCountHoldingByteCount:bytes];
	if ((double)budget / NFKModelWorkingSetFraction >= (double)NSIntegerMax) {
		return NSIntegerMax;
	}
	// The division rounds in floating point, so the working set steps to the least one whose budget,
	// truncated as the initializer truncates it, reaches the bytes.
	NSInteger workingSet = (NSInteger)ceil((double)budget / NFKModelWorkingSetFraction);
	while (NFKModelBudgetOfWorkingSet(workingSet) < budget) {
		workingSet += 1;
	}
	while (workingSet > 1 && NFKModelBudgetOfWorkingSet(workingSet - 1) >= budget) {
		workingSet -= 1;
	}
	return workingSet;
}

+ (NSString *)formattedByteCount:(NSInteger)bytes
{
	return NFKModelGiB(bytes);
}

+ (nullable instancetype)budgetWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self budgetFromObject:dictionary path:@"budget" error:error];
}

+ (nullable instancetype)budgetFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSNumber *workingSet = nil;
	if (reader == nil || ![reader readNumber:@"workingSetBytes" into:&workingSet error:error]) {
		return nil;
	}
	if (workingSet == nil) {
		[reader failField:@"workingSetBytes" reason:@"is missing" error:error];
		return nil;
	}
	return [self budgetWithWorkingSetByteCount:workingSet.integerValue];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	return @{ @"workingSetBytes": @(self.workingSetByteCount), @"budgetBytes": @(self.byteCount),
			  @"reserveBytes": @(self.reserveByteCount), @"workingSetFraction": @(NFKModelWorkingSetFraction) };
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryBudget.class);
}

- (NSUInteger)hash
{
	return (NSUInteger)self.workingSetByteCount;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ of %@>", NSStringFromClass(self.class), NFKModelGiB(self.byteCount),
			NFKModelGiB(self.workingSetByteCount)];
}

@end

#pragma mark - NFKModelStageFacts

static NSArray<NSString *> *NFKModelStageFactsFields(void)
{
	return @[ @"id", @"title", @"bytesByPrecision", @"quantizableElements", @"pageableBytesByPrecision", @"layerCount",
			  @"largestLayerBytesByPrecision", @"widenedPrecision" ];
}

@implementation NFKModelStageFacts

+ (NSInteger)maximumLayerCount
{
	return NFKModelMaximumLayerCount;
}

- (instancetype)initWithIdentifier:(NSString *)identifier
							 title:(nullable NSString *)title
			  byteCountByPrecision:(NSDictionary<NSString *, NSNumber *> *)byteCountByPrecision
		   quantizableElementCount:(NSInteger)quantizableElementCount
	  pageableByteCountByPrecision:(nullable NSDictionary<NSString *, NSNumber *> *)pageableByteCountByPrecision
						layerCount:(NSInteger)layerCount
  largestLayerByteCountByPrecision:(nullable NSDictionary<NSString *, NSNumber *> *)largestLayerByteCountByPrecision
				  widenedPrecision:(nullable NSString *)widenedPrecision
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_title = [title copy] ?: _identifier;
		_byteCountByPrecision = [byteCountByPrecision copy];
		_quantizableElementCount = quantizableElementCount;
		_pageableByteCountByPrecision = [pageableByteCountByPrecision copy] ?: @{};
		_layerCount = MIN(MAX(layerCount, 0), NFKModelMaximumLayerCount);
		_largestLayerByteCountByPrecision = [largestLayerByteCountByPrecision copy] ?: @{};
		_widenedPrecision = [widenedPrecision copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelStageFactsFields());
	}
	return self;
}

+ (instancetype)stageWithIdentifier:(NSString *)identifier
			   byteCountByPrecision:(NSDictionary<NSString *, NSNumber *> *)byteCountByPrecision
{
	return [[self alloc] initWithIdentifier:identifier title:nil byteCountByPrecision:byteCountByPrecision
					quantizableElementCount:0 pageableByteCountByPrecision:nil layerCount:0
		   largestLayerByteCountByPrecision:nil widenedPrecision:nil extensions:nil];
}

+ (nullable instancetype)stageWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self stageFromObject:dictionary path:@"stage" error:error];
}

+ (nullable instancetype)stageFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *identifier = nil, *title = nil, *widenedPrecision = nil;
	NSDictionary<NSString *, NSNumber *> *bytes = nil, *pageable = nil, *largestLayer = nil;
	NSNumber *quantizable = nil, *layerCount = nil;
	if (reader == nil
		|| ![reader readString:@"id" required:YES into:&identifier error:error]
		|| ![reader readString:@"title" required:NO into:&title error:error]
		|| ![reader readNumberDictionary:@"bytesByPrecision" into:&bytes error:error]
		|| ![reader readNumber:@"quantizableElements" into:&quantizable error:error]
		|| ![reader readNumberDictionary:@"pageableBytesByPrecision" into:&pageable error:error]
		|| ![reader readNumber:@"layerCount" into:&layerCount error:error]
		|| ![reader readNumberDictionary:@"largestLayerBytesByPrecision" into:&largestLayer error:error]
		|| ![reader readString:@"widenedPrecision" required:NO into:&widenedPrecision error:error]) {
		return nil;
	}
	if (bytes == nil) {
		[reader failField:@"bytesByPrecision" reason:@"is missing" error:error];
		return nil;
	}
	if (layerCount != nil && !NFKModelIsWholeNumberInRange(layerCount, 0, NFKModelMaximumLayerCount)) {
		[reader failField:@"layerCount"
				   reason:[NSString stringWithFormat:@"is not a whole number from 0 to %ld", (long)NFKModelMaximumLayerCount]
					error:error];
		return nil;
	}
	return [[self alloc] initWithIdentifier:identifier title:title byteCountByPrecision:bytes
					quantizableElementCount:quantizable.integerValue pageableByteCountByPrecision:pageable
								 layerCount:layerCount.integerValue largestLayerByteCountByPrecision:largestLayer
						   widenedPrecision:widenedPrecision
								 extensions:[reader extensionsWithoutFields:NFKModelStageFactsFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"id"] = self.identifier;
	dictionary[@"title"] = self.title;
	dictionary[@"bytesByPrecision"] = self.byteCountByPrecision;
	if (self.quantizableElementCount > 0) {
		dictionary[@"quantizableElements"] = @(self.quantizableElementCount);
	}
	if (self.pageableByteCountByPrecision.count > 0) {
		dictionary[@"pageableBytesByPrecision"] = self.pageableByteCountByPrecision;
	}
	if (self.layerCount > 0) {
		dictionary[@"layerCount"] = @(self.layerCount);
	}
	if (self.largestLayerByteCountByPrecision.count > 0) {
		dictionary[@"largestLayerBytesByPrecision"] = self.largestLayerByteCountByPrecision;
	}
	dictionary[@"widenedPrecision"] = self.widenedPrecision;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelStageFacts.class);
}

- (NSUInteger)hash
{
	return self.identifier.hash;
}

@end

#pragma mark - NFKModelMemoryFacts

static NSArray<NSString *> *NFKModelMemoryFactsFields(void)
{
	return @[ @"stages", @"keyValueBytesPerToken", @"boundedKeyValueBytes", @"maximumContextLength", @"vocabularySize",
			  @"loadOverheadBytes", @"basis" ];
}

@implementation NFKModelMemoryFacts

- (instancetype)initWithStages:(NSArray<NFKModelStageFacts *> *)stages
	 keyValueByteCountPerToken:(nullable NSNumber *)keyValueByteCountPerToken
	  boundedKeyValueByteCount:(nullable NSNumber *)boundedKeyValueByteCount
		  maximumContextLength:(nullable NSNumber *)maximumContextLength
				vocabularySize:(nullable NSNumber *)vocabularySize
		 loadOverheadByteCount:(NSInteger)loadOverheadByteCount
						 basis:(nullable NSString *)basis
					extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_stages = [stages copy];
		_keyValueByteCountPerToken = [keyValueByteCountPerToken copy];
		_boundedKeyValueByteCount = [boundedKeyValueByteCount copy];
		_maximumContextLength = [maximumContextLength copy];
		_vocabularySize = [vocabularySize copy];
		_loadOverheadByteCount = loadOverheadByteCount;
		_basis = [basis copy] ?: NFKModelMemoryBasisMeasured;
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelMemoryFactsFields());
	}
	return self;
}

- (nullable NFKModelStageFacts *)stageWithIdentifier:(NSString *)identifier
{
	for (NFKModelStageFacts *stage in self.stages) {
		if ([stage.identifier isEqualToString:identifier]) {
			return stage;
		}
	}
	return nil;
}

+ (nullable instancetype)factsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self factsFromObject:dictionary path:@"memoryFacts" error:error];
}

+ (nullable instancetype)factsFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSArray *stages = nil;
	NSNumber *perToken = nil, *bounded = nil, *maximumContext = nil, *vocabulary = nil, *overhead = nil;
	NSString *basis = nil;
	NFKModelElementReader stageReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelStageFacts stageFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readObjects:@"stages" withReader:stageReader into:&stages error:error]
		|| ![reader readNumber:@"keyValueBytesPerToken" into:&perToken error:error]
		|| ![reader readNumber:@"boundedKeyValueBytes" into:&bounded error:error]
		|| ![reader readNumber:@"maximumContextLength" into:&maximumContext error:error]
		|| ![reader readNumber:@"vocabularySize" into:&vocabulary error:error]
		|| ![reader readNumber:@"loadOverheadBytes" into:&overhead error:error]
		|| ![reader readString:@"basis" required:NO into:&basis error:error]
		|| ![reader requireUniqueElements:stages ofField:@"stages" keyPath:@"identifier" name:@"id" error:error]) {
		return nil;
	}
	if (stages == nil) {
		[reader failField:@"stages" reason:@"is missing" error:error];
		return nil;
	}
	return [[self alloc] initWithStages:stages keyValueByteCountPerToken:perToken boundedKeyValueByteCount:bounded
				   maximumContextLength:maximumContext vocabularySize:vocabulary
				  loadOverheadByteCount:overhead.integerValue basis:basis
							 extensions:[reader extensionsWithoutFields:NFKModelMemoryFactsFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"stages"] = NFKModelDictionariesOf(self.stages);
	dictionary[@"keyValueBytesPerToken"] = self.keyValueByteCountPerToken;
	dictionary[@"boundedKeyValueBytes"] = self.boundedKeyValueByteCount;
	dictionary[@"maximumContextLength"] = self.maximumContextLength;
	dictionary[@"vocabularySize"] = self.vocabularySize;
	dictionary[@"loadOverheadBytes"] = @(self.loadOverheadByteCount);
	dictionary[@"basis"] = self.basis;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryFacts.class);
}

- (NSUInteger)hash
{
	return self.stages.count;
}

@end

#pragma mark - NFKModelMemoryFigure

@implementation NFKModelMemoryFigure

- (instancetype)initWithResidentByteCount:(NSInteger)residentByteCount peakByteCount:(NSInteger)peakByteCount
{
	self = [super init];
	if (self != nil) {
		_residentByteCount = residentByteCount;
		_peakByteCount = peakByteCount;
	}
	return self;
}

+ (nullable instancetype)figureWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self figureFromObject:dictionary path:@"memory" error:error];
}

+ (nullable instancetype)figureFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSNumber *resident = nil, *peak = nil;
	if (reader == nil
		|| ![reader readNumber:@"residentBytes" into:&resident error:error]
		|| ![reader readNumber:@"peakBytes" into:&peak error:error]) {
		return nil;
	}
	if (resident == nil) {
		[reader failField:@"residentBytes" reason:@"is missing" error:error];
		return nil;
	}
	return [[self alloc] initWithResidentByteCount:resident.integerValue
									 peakByteCount:(peak ?: resident).integerValue];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	return @{ @"residentBytes": @(self.residentByteCount), @"peakBytes": @(self.peakByteCount) };
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryFigure.class);
}

- (NSUInteger)hash
{
	return (NSUInteger)self.residentByteCount;
}

@end

#pragma mark - NFKModelMemoryModel

// The memory model's dictionary fields that name options.
static NSArray<NSString *> *NFKModelMemoryModelOptionFields(void)
{
	return @[ @"precision", @"quantizationBits", @"quantizationGroupSize", @"contextLength",
			  @"cacheBits", @"cacheGroupSize", @"draft", @"draftBits", @"residency" ];
}

static NSArray<NSString *> *NFKModelMemoryModelFields(void)
{
	return @[ @"precision", @"quantizationBits", @"quantizationGroupSize", @"contextLength", @"cacheBits",
			  @"cacheGroupSize", @"draft", @"draftBits", @"residency", @"cacheStage", @"draftStage",
			  @"stagePrecisions" ];
}

@implementation NFKModelMemoryModel

- (instancetype)initWithPrecisionOptionKey:(nullable NSString *)precisionOptionKey
				 quantizationBitsOptionKey:(nullable NSString *)quantizationBitsOptionKey
			quantizationGroupSizeOptionKey:(nullable NSString *)quantizationGroupSizeOptionKey
					contextLengthOptionKey:(nullable NSString *)contextLengthOptionKey
						cacheBitsOptionKey:(nullable NSString *)cacheBitsOptionKey
				   cacheGroupSizeOptionKey:(nullable NSString *)cacheGroupSizeOptionKey
							draftOptionKey:(nullable NSString *)draftOptionKey
						draftBitsOptionKey:(nullable NSString *)draftBitsOptionKey
						residencyOptionKey:(nullable NSString *)residencyOptionKey
								cacheStage:(nullable NSString *)cacheStage
								draftStage:(nullable NSString *)draftStage
						   stagePrecisions:(nullable NSDictionary<NSString *, NSString *> *)stagePrecisions
								extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_precisionOptionKey = [precisionOptionKey copy];
		_quantizationBitsOptionKey = [quantizationBitsOptionKey copy];
		_quantizationGroupSizeOptionKey = [quantizationGroupSizeOptionKey copy];
		_contextLengthOptionKey = [contextLengthOptionKey copy];
		_cacheBitsOptionKey = [cacheBitsOptionKey copy];
		_cacheGroupSizeOptionKey = [cacheGroupSizeOptionKey copy];
		_draftOptionKey = [draftOptionKey copy];
		_draftBitsOptionKey = [draftBitsOptionKey copy];
		_residencyOptionKey = [residencyOptionKey copy];
		_cacheStage = [cacheStage copy];
		_draftStage = [draftStage copy];
		_stagePrecisions = [stagePrecisions copy] ?: @{};
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelMemoryModelFields());
	}
	return self;
}

- (NSArray<NSString *> *)optionKeys
{
	NSMutableArray<NSString *> *keys = [NSMutableArray array];
	for (NSString *key in @[ self.precisionOptionKey ?: @"", self.quantizationBitsOptionKey ?: @"",
							 self.quantizationGroupSizeOptionKey ?: @"", self.contextLengthOptionKey ?: @"",
							 self.cacheBitsOptionKey ?: @"", self.cacheGroupSizeOptionKey ?: @"",
							 self.draftOptionKey ?: @"", self.draftBitsOptionKey ?: @"",
							 self.residencyOptionKey ?: @"" ]) {
		if (key.length > 0 && ![keys containsObject:key]) {
			[keys addObject:key];
		}
	}
	return [keys copy];
}

+ (nullable instancetype)memoryModelWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self memoryModelFromObject:dictionary path:@"memoryModel" error:error];
}

+ (nullable instancetype)memoryModelFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	if (reader == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, NSString *> *keys = [NSMutableDictionary dictionary];
	for (NSString *field in NFKModelMemoryModelOptionFields()) {
		NSString *value = nil;
		if (![reader readString:field required:NO into:&value error:error]) {
			return nil;
		}
		keys[field] = value;
	}
	NSString *cacheStage = nil, *draftStage = nil;
	NSDictionary<NSString *, NSString *> *stagePrecisions = nil;
	if (![reader readString:@"cacheStage" required:NO into:&cacheStage error:error]
		|| ![reader readString:@"draftStage" required:NO into:&draftStage error:error]
		|| ![reader readStringDictionary:@"stagePrecisions" into:&stagePrecisions error:error]) {
		return nil;
	}
	return [[self alloc] initWithPrecisionOptionKey:keys[@"precision"] quantizationBitsOptionKey:keys[@"quantizationBits"]
					 quantizationGroupSizeOptionKey:keys[@"quantizationGroupSize"]
							 contextLengthOptionKey:keys[@"contextLength"] cacheBitsOptionKey:keys[@"cacheBits"]
							cacheGroupSizeOptionKey:keys[@"cacheGroupSize"] draftOptionKey:keys[@"draft"]
								 draftBitsOptionKey:keys[@"draftBits"] residencyOptionKey:keys[@"residency"]
										 cacheStage:cacheStage draftStage:draftStage stagePrecisions:stagePrecisions
										 extensions:[reader extensionsWithoutFields:NFKModelMemoryModelFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"precision"] = self.precisionOptionKey;
	dictionary[@"quantizationBits"] = self.quantizationBitsOptionKey;
	dictionary[@"quantizationGroupSize"] = self.quantizationGroupSizeOptionKey;
	dictionary[@"contextLength"] = self.contextLengthOptionKey;
	dictionary[@"cacheBits"] = self.cacheBitsOptionKey;
	dictionary[@"cacheGroupSize"] = self.cacheGroupSizeOptionKey;
	dictionary[@"draft"] = self.draftOptionKey;
	dictionary[@"draftBits"] = self.draftBitsOptionKey;
	dictionary[@"residency"] = self.residencyOptionKey;
	dictionary[@"cacheStage"] = self.cacheStage;
	dictionary[@"draftStage"] = self.draftStage;
	if (self.stagePrecisions.count > 0) {
		dictionary[@"stagePrecisions"] = self.stagePrecisions;
	}
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryModel.class);
}

- (NSUInteger)hash
{
	return self.optionKeys.count;
}

@end

#pragma mark - NFKModelStageFootprint

@implementation NFKModelStageFootprint

- (instancetype)initWithByteCount:(NSInteger)byteCount
				pageableByteCount:(NSInteger)pageableByteCount
				 widenedByteCount:(nullable NSNumber *)widenedByteCount
				  layerByteCounts:(nullable NSArray<NSNumber *> *)layerByteCounts
{
	self = [super init];
	if (self != nil) {
		_byteCount = byteCount;
		_pageableByteCount = pageableByteCount;
		_widenedByteCount = [widenedByteCount copy];
		_layerByteCounts = [layerByteCounts copy] ?: @[];
	}
	return self;
}

+ (instancetype)footprintWithByteCount:(NSInteger)byteCount
{
	return [[self alloc] initWithByteCount:byteCount pageableByteCount:0 widenedByteCount:nil layerByteCounts:nil];
}

- (NSInteger)unpagedByteCount
{
	return self.byteCount - self.pageableByteCount;
}

- (NSInteger)unstreamableByteCount
{
	NSInteger layers = 0;
	for (NSNumber *bytes in self.layerByteCounts) {
		layers += bytes.integerValue;
	}
	return self.byteCount - layers;
}

- (NSInteger)streamedMinimumByteCount
{
	NSArray<NSNumber *> *largestFirst = [self.layerByteCounts sortedArrayUsingComparator:^NSComparisonResult(NSNumber *a, NSNumber *b) {
		return [b compare:a];
	}];
	NSInteger inFlight = 0;
	for (NSUInteger index = 0; index < MIN((NSUInteger)2, largestFirst.count); index += 1) {
		inFlight += largestFirst[index].integerValue;
	}
	return self.unstreamableByteCount + inFlight;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (![object isKindOfClass:NFKModelStageFootprint.class]) {
		return NO;
	}
	NFKModelStageFootprint *other = object;
	return other.byteCount == self.byteCount && other.pageableByteCount == self.pageableByteCount
		&& (other.widenedByteCount == self.widenedByteCount || [other.widenedByteCount isEqual:self.widenedByteCount])
		&& [other.layerByteCounts isEqualToArray:self.layerByteCounts];
}

- (NSUInteger)hash
{
	return (NSUInteger)self.byteCount;
}

@end

#pragma mark - NFKModelResidencyPlan

@interface NFKModelResidencyPlan ()
@property (nonatomic, readwrite) BOOL holdsStagesResident;
@property (nonatomic, readwrite) BOOL pagesExperts;
@property (nonatomic, readwrite) NSInteger expertCacheByteCount;
@property (nonatomic, readwrite, copy) NSIndexSet *widenedStageIndexes;
@property (nonatomic, readwrite, copy) NSDictionary<NSNumber *, NSNumber *> *streamedStageByteCounts;
@end

@implementation NFKModelResidencyPlan

- (instancetype)initHoldingResident:(BOOL)resident pagesExperts:(BOOL)pagesExperts expertCacheByteCount:(NSInteger)cache
{
	self = [super init];
	if (self != nil) {
		_holdsStagesResident = resident;
		_pagesExperts = pagesExperts;
		_expertCacheByteCount = cache;
		_widenedStageIndexes = [NSIndexSet indexSet];
		_streamedStageByteCounts = @{};
	}
	return self;
}

+ (instancetype)residentPlan
{
	return [[self alloc] initHoldingResident:YES pagesExperts:NO expertCacheByteCount:0];
}

+ (instancetype)stagedPlan
{
	return [[self alloc] initHoldingResident:NO pagesExperts:NO expertCacheByteCount:0];
}

- (NSString *)residency
{
	if (self.holdsStagesResident) {
		return NFKModelResidencyResident;
	}
	if (self.pagesExperts) {
		return NFKModelResidencyPaged;
	}
	return self.streamedStageByteCounts.count > 0 ? NFKModelResidencyStreamed : NFKModelResidencyStaged;
}

- (BOOL)widensStageAtIndex:(NSUInteger)index
{
	return [self.widenedStageIndexes containsIndex:index];
}

- (BOOL)streamsStageAtIndex:(NSUInteger)index
{
	return self.streamedStageByteCounts[@(index)] != nil;
}

- (NSRange)heldLayerRangeOfStageAtIndex:(NSUInteger)index footprint:(NFKModelStageFootprint *)footprint
{
	NSNumber *room = self.streamedStageByteCounts[@(index)];
	if (room == nil) {
		return NSMakeRange(0, footprint.layerByteCounts.count);
	}
	NSInteger total = 0;
	NSUInteger count = 0;
	for (NSNumber *bytes in footprint.layerByteCounts) {
		if (total + bytes.integerValue > room.integerValue) {
			break;
		}
		total += bytes.integerValue;
		count += 1;
	}
	return NSMakeRange(0, count);
}

+ (nullable instancetype)planForStages:(NSArray<NFKModelStageFootprint *> *)stages
							 residency:(NSString *)residency
								budget:(NSInteger)budget
								 error:(NSError **)error
{
	NSString *refusal = nil;
	NFKModelResidencyPlan *plan = [self placementForStages:stages residency:residency budget:budget refusal:&refusal];
	if (refusal != nil) {
		if (error != NULL) {
			*error = NFKModelDoesNotFitError(refusal, nil);
		}
		return nil;
	}
	return plan;
}

// The plan the residency asks for, and the planner's refusal when it is known not to fit. The plan is
// what the residency would hold even when refused, so an estimate can state its figure.
+ (NFKModelResidencyPlan *)placementForStages:(NSArray<NFKModelStageFootprint *> *)stages
									residency:(NSString *)residency
									   budget:(NSInteger)budget
									  refusal:(NSString * _Nullable * _Nonnull)refusal
{
	__block NSString *found = nil;
	NSInteger total = 0, largestBytes = 0, largestUnpaged = 0, largestPageable = 0;
	BOOL pageable = NO;
	NSMutableArray<NSNumber *> *streamable = [NSMutableArray array];
	NSMutableArray<NSNumber *> *oversized = [NSMutableArray array];
	for (NSUInteger index = 0; index < stages.count; index += 1) {
		NFKModelStageFootprint *stage = stages[index];
		total += stage.byteCount;
		largestBytes = MAX(largestBytes, stage.byteCount);
		largestUnpaged = MAX(largestUnpaged, stage.unpagedByteCount);
		largestPageable = MAX(largestPageable, stage.pageableByteCount);
		pageable = pageable || stage.pageableByteCount > 0;
		if (stage.layerByteCounts.count > 0) {
			[streamable addObject:@(index)];
			if (budget <= 0 || !NFKModelBudgetAdmits(stage.byteCount, budget)) {
				[oversized addObject:@(index)];
			}
		}
	}
	NSString *reserve = NFKModelReserveGiB(budget);
	NFKModelResidencyPlan *(^paged)(void) = ^NFKModelResidencyPlan *(void) {
		if (!NFKModelBudgetAdmits(largestUnpaged, budget)) {
			found = [NSString stringWithFormat:@"the largest stage needs %@ with its routed experts paged, plus a %@ reserve, against a %@ budget",
						NFKModelGiB(largestUnpaged), reserve, NFKModelGiB(budget)];
		}
		NSInteger headroom = budget > 0 ? (budget - NFKModelReserveByteCount(budget) - largestUnpaged) / 2
			: NFKModelDefaultExpertCacheByteCount;
		return [[NFKModelResidencyPlan alloc] initHoldingResident:NO pagesExperts:YES
											 expertCacheByteCount:MAX(0, MIN(largestPageable, headroom))];
	};
	NFKModelResidencyPlan *(^streamed)(void) = ^NFKModelResidencyPlan *(void) {
		NFKModelResidencyPlan *plan = [NFKModelResidencyPlan stagedPlan];
		NSMutableDictionary<NSNumber *, NSNumber *> *rooms = [NSMutableDictionary dictionary];
		for (NSNumber *index in oversized) {
			NSInteger minimum = stages[index.unsignedIntegerValue].streamedMinimumByteCount;
			if (!NFKModelBudgetAdmits(minimum, budget) && found == nil) {
				found = [NSString stringWithFormat:@"stage %ld needs %@ streamed, two layers in flight, plus a %@ reserve, against a %@ budget",
							(long)index.integerValue, NFKModelGiB(minimum), reserve, NFKModelGiB(budget)];
			}
			rooms[index] = @(budget > 0 ? MAX(0, budget - NFKModelReserveByteCount(budget) - minimum) : 0);
		}
		plan.streamedStageByteCounts = rooms;
		return plan;
	};
	NFKModelResidencyPlan *placement = nil;
	if ([residency isEqualToString:NFKModelResidencyResident]) {
		if (!NFKModelBudgetAdmits(total, budget)) {
			found = [NSString stringWithFormat:@"the stages need %@ together plus a %@ reserve, against a %@ budget; hold them staged%@%@",
						NFKModelGiB(total), reserve, NFKModelGiB(budget), pageable ? @" or paged" : @"",
						streamable.count == 0 ? @"" : @" or streamed"];
		}
		placement = [NFKModelResidencyPlan residentPlan];
	} else if ([residency isEqualToString:NFKModelResidencyStaged]) {
		placement = [NFKModelResidencyPlan stagedPlan];
	} else if ([residency isEqualToString:NFKModelResidencyPaged]) {
		placement = pageable ? paged() : [NFKModelResidencyPlan stagedPlan];
	} else if ([residency isEqualToString:NFKModelResidencyStreamed]) {
		placement = streamed();
	} else if (NFKModelBudgetHolds(total, budget)) {
		placement = [NFKModelResidencyPlan residentPlan];
	} else if (pageable && budget > 0 && !NFKModelBudgetHolds(largestBytes, budget)) {
		placement = paged();
	} else if (budget > 0 && oversized.count > 0) {
		placement = streamed();
	} else {
		placement = [NFKModelResidencyPlan stagedPlan];
	}
	placement.widenedStageIndexes = [self widenedStagesOf:stages resident:placement.holdsStagesResident budget:budget];
	*refusal = found;
	return placement;
}

+ (NSIndexSet *)widenedStagesOf:(NSArray<NFKModelStageFootprint *> *)stages resident:(BOOL)resident budget:(NSInteger)budget
{
	NSInteger total = 0;
	for (NFKModelStageFootprint *stage in stages) {
		total += stage.byteCount;
	}
	NSMutableIndexSet *widened = [NSMutableIndexSet indexSet];
	for (NSUInteger index = 0; index < stages.count; index += 1) {
		NFKModelStageFootprint *stage = stages[index];
		if (stage.widenedByteCount == nil) {
			continue;
		}
		NSInteger wide = stage.widenedByteCount.integerValue;
		BOOL fits = resident ? NFKModelBudgetHolds(total - stage.byteCount + wide, budget) : NFKModelBudgetHolds(wide, budget);
		if (fits) {
			[widened addIndex:index];
			total += wide - stage.byteCount;
		}
	}
	return widened;
}

+ (BOOL)verifyEachStageLoads:(NSArray<NFKModelStageFootprint *> *)stages
						plan:(NFKModelResidencyPlan *)plan
					  budget:(NSInteger)budget
					   names:(NSArray<NSString *> *)names
					   error:(NSError **)error
{
	if (plan.holdsStagesResident) {
		return YES;
	}
	for (NSUInteger index = 0; index < stages.count; index += 1) {
		if ([plan streamsStageAtIndex:index]) {
			continue;
		}
		NFKModelStageFootprint *stage = stages[index];
		NSInteger bytes = [plan widensStageAtIndex:index] ? (stage.widenedByteCount ?: @(stage.byteCount)).integerValue
			: (plan.pagesExperts ? stage.unpagedByteCount : stage.byteCount);
		if (!NFKModelBudgetAdmits(bytes, budget)) {
			NSString *name = index < names.count ? names[index] : [NSString stringWithFormat:@"stage %lu", (unsigned long)index];
			NSString *reason = [NSString stringWithFormat:@"%@ alone needs %@ plus a %@ reserve, against a %@ budget",
								name, NFKModelGiB(bytes), NFKModelReserveGiB(budget), NFKModelGiB(budget)];
			if (error != NULL) {
				*error = NFKModelDoesNotFitError(reason, nil);
			}
			return NO;
		}
	}
	return YES;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (![object isKindOfClass:NFKModelResidencyPlan.class]) {
		return NO;
	}
	NFKModelResidencyPlan *other = object;
	return other.holdsStagesResident == self.holdsStagesResident && other.pagesExperts == self.pagesExperts
		&& other.expertCacheByteCount == self.expertCacheByteCount
		&& [other.widenedStageIndexes isEqualToIndexSet:self.widenedStageIndexes]
		&& [other.streamedStageByteCounts isEqualToDictionary:self.streamedStageByteCounts];
}

- (NSUInteger)hash
{
	return (NSUInteger)self.expertCacheByteCount ^ (NSUInteger)self.holdsStagesResident;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ cache %@ widened %@ streamed %@>", NSStringFromClass(self.class),
			self.residency, NFKModelGiB(self.expertCacheByteCount), self.widenedStageIndexes,
			self.streamedStageByteCounts];
}

@end

#pragma mark - NFKModelMemoryEstimate

@interface NFKModelMemoryEstimate ()
- (instancetype)initWithResidentByteCount:(NSInteger)residentByteCount
							peakByteCount:(NSInteger)peakByteCount
					   requestedResidency:(NSString *)requestedResidency
								residency:(NSString *)residency
							   components:(NSDictionary<NSString *, NSNumber *> *)components
						  stageByteCounts:(NSDictionary<NSString *, NSNumber *> *)stageByteCounts
						  resolvedOptions:(NSDictionary<NSString *, id> *)resolvedOptions
								   budget:(nullable NFKModelMemoryBudget *)budget
									 fits:(BOOL)fits
								   reason:(nullable NSString *)reason
									basis:(NSString *)basis NS_DESIGNATED_INITIALIZER;
@end

@implementation NFKModelMemoryEstimate

- (instancetype)initWithResidentByteCount:(NSInteger)residentByteCount
							peakByteCount:(NSInteger)peakByteCount
					   requestedResidency:(NSString *)requestedResidency
								residency:(NSString *)residency
							   components:(NSDictionary<NSString *, NSNumber *> *)components
						  stageByteCounts:(NSDictionary<NSString *, NSNumber *> *)stageByteCounts
						  resolvedOptions:(NSDictionary<NSString *, id> *)resolvedOptions
								   budget:(nullable NFKModelMemoryBudget *)budget
									 fits:(BOOL)fits
								   reason:(nullable NSString *)reason
									basis:(NSString *)basis
{
	self = [super init];
	if (self != nil) {
		_residentByteCount = residentByteCount;
		_peakByteCount = peakByteCount;
		_minimumWorkingSetByteCount = [NFKModelMemoryBudget leastWorkingSetHoldingByteCount:peakByteCount];
		_requestedResidency = [requestedResidency copy];
		_residency = [residency copy];
		_components = [components copy];
		_stageByteCounts = [stageByteCounts copy];
		_resolvedOptions = [resolvedOptions copy];
		_budget = [budget copy];
		_fits = fits;
		_reason = [reason copy];
		_basis = [basis copy];
	}
	return self;
}

- (NFKModelMemoryFigure *)figure
{
	return [[NFKModelMemoryFigure alloc] initWithResidentByteCount:self.residentByteCount peakByteCount:self.peakByteCount];
}

+ (nullable instancetype)estimateWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:dictionary path:@"estimate" error:error];
	NSNumber *resident = nil, *peak = nil;
	NSString *requested = nil, *residency = nil, *reason = nil, *basis = nil;
	NSDictionary<NSString *, NSNumber *> *components = nil, *stageBytes = nil;
	NSDictionary<NSString *, id> *options = nil;
	id budget = nil;
	BOOL fits = NO;
	NFKModelElementReader budgetReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelMemoryBudget budgetFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readNumber:@"residentBytes" into:&resident error:error]
		|| ![reader readNumber:@"peakBytes" into:&peak error:error]
		|| ![reader readString:@"requestedResidency" required:NO into:&requested error:error]
		|| ![reader readString:@"residency" required:NO into:&residency error:error]
		|| ![reader readNumberDictionary:@"components" into:&components error:error]
		|| ![reader readNumberDictionary:@"stageBytes" into:&stageBytes error:error]
		|| ![reader readJSONDictionary:@"options" into:&options error:error]
		|| ![reader readObject:@"budget" withReader:budgetReader into:&budget error:error]
		|| ![reader readBoolean:@"fits" into:&fits error:error]
		|| ![reader readString:@"reason" required:NO into:&reason error:error]
		|| ![reader readString:@"basis" required:NO into:&basis error:error]) {
		return nil;
	}
	return [[self alloc] initWithResidentByteCount:resident.integerValue peakByteCount:peak.integerValue
								requestedResidency:requested ?: NFKModelResidencyAutomatic
										 residency:residency ?: NFKModelResidencyStaged components:components ?: @{}
								   stageByteCounts:stageBytes ?: @{} resolvedOptions:options ?: @{} budget:budget
											  fits:fits reason:reason basis:basis ?: NFKModelMemoryBasisUnknown];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"residentBytes"] = @(self.residentByteCount);
	dictionary[@"peakBytes"] = @(self.peakByteCount);
	dictionary[@"minimumWorkingSetBytes"] = @(self.minimumWorkingSetByteCount);
	dictionary[@"requestedResidency"] = self.requestedResidency;
	dictionary[@"residency"] = self.residency;
	dictionary[@"components"] = self.components;
	dictionary[@"stageBytes"] = self.stageByteCounts;
	if ([NSJSONSerialization isValidJSONObject:self.resolvedOptions]) {
		dictionary[@"options"] = self.resolvedOptions;
	}
	dictionary[@"budget"] = self.budget.dictionaryRepresentation;
	dictionary[@"fits"] = @(self.fits);
	dictionary[@"reason"] = self.reason;
	dictionary[@"basis"] = self.basis;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryEstimate.class);
}

- (NSUInteger)hash
{
	return (NSUInteger)self.peakByteCount;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ resident, %@ peak, %@%@>", NSStringFromClass(self.class),
			NFKModelGiB(self.residentByteCount), NFKModelGiB(self.peakByteCount), self.residency,
			self.fits ? @"" : @", does not fit"];
}

@end

#pragma mark - Constraints

static NSString *NFKModelChoiceStatusName(NFKModelChoiceStatus status)
{
	switch (status) {
		case NFKModelChoiceStatusRequired:
			return @"required";
		case NFKModelChoiceStatusRuledOut:
			return @"ruled-out";
		case NFKModelChoiceStatusAllowed:
			break;
	}
	return @"allowed";
}

@implementation NFKModelChoiceConstraint

- (instancetype)initWithValue:(id)value
					   status:(NFKModelChoiceStatus)status
					   reason:(nullable NSString *)reason
				peakByteCount:(nullable NSNumber *)peakByteCount
{
	self = [super init];
	if (self != nil) {
		_value = [value copy];
		_status = status;
		_reason = [reason copy];
		_peakByteCount = [peakByteCount copy];
	}
	return self;
}

+ (nullable instancetype)constraintWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self constraintFromObject:dictionary path:@"choice" error:error];
}

+ (nullable instancetype)constraintFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *status = nil, *reason = nil;
	NSNumber *peak = nil;
	if (reader == nil
		|| ![reader readString:@"status" required:YES into:&status error:error]
		|| ![reader readString:@"reason" required:NO into:&reason error:error]
		|| ![reader readNumber:@"peakBytes" into:&peak error:error]) {
		return nil;
	}
	id value = reader.dictionary[@"value"];
	if (value == nil) {
		[reader failField:@"value" reason:@"is missing" error:error];
		return nil;
	}
	NFKModelChoiceStatus parsed = NFKModelChoiceStatusAllowed;
	if ([status isEqualToString:@"required"]) {
		parsed = NFKModelChoiceStatusRequired;
	} else if ([status isEqualToString:@"ruled-out"]) {
		parsed = NFKModelChoiceStatusRuledOut;
	}
	return [[self alloc] initWithValue:value status:parsed reason:reason peakByteCount:peak];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"value"] = self.value;
	dictionary[@"status"] = NFKModelChoiceStatusName(self.status);
	dictionary[@"reason"] = self.reason;
	dictionary[@"peakBytes"] = self.peakByteCount;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelChoiceConstraint.class);
}

- (NSUInteger)hash
{
	return [self.value hash];
}

@end

@implementation NFKModelOptionConstraint

- (instancetype)initWithKey:(NSString *)key
				  available:(BOOL)available
					choices:(nullable NSArray<NFKModelChoiceConstraint *> *)choices
			  requiredValue:(nullable id)requiredValue
			   maximumValue:(nullable NSNumber *)maximumValue
					 reason:(nullable NSString *)reason
	 fullyResidentByteCount:(nullable NSNumber *)fullyResidentByteCount
			budgetByteCount:(nullable NSNumber *)budgetByteCount
		  techniqueByteCount:(nullable NSNumber *)techniqueByteCount
{
	self = [super init];
	if (self != nil) {
		_key = [key copy];
		_available = available;
		_choices = [choices copy] ?: @[];
		_requiredValue = [requiredValue copy];
		_maximumValue = [maximumValue copy];
		_reason = [reason copy];
		_fullyResidentByteCount = [fullyResidentByteCount copy];
		_budgetByteCount = [budgetByteCount copy];
		_techniqueByteCount = [techniqueByteCount copy];
	}
	return self;
}

- (nullable NFKModelChoiceConstraint *)constraintForChoiceValue:(id)value
{
	for (NFKModelChoiceConstraint *choice in self.choices) {
		if ([choice.value isEqual:value]) {
			return choice;
		}
	}
	return nil;
}

+ (nullable instancetype)constraintWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self constraintFromObject:dictionary path:@"option" error:error];
}

+ (nullable instancetype)constraintFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *key = nil, *reason = nil;
	NSNumber *maximum = nil, *fullyResident = nil, *budget = nil, *technique = nil;
	NSArray *choices = nil;
	BOOL available = YES;
	NFKModelElementReader choiceReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelChoiceConstraint constraintFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"key" required:YES into:&key error:error]
		|| ![reader readObjects:@"choices" withReader:choiceReader into:&choices error:error]
		|| ![reader readNumber:@"maximumValue" into:&maximum error:error]
		|| ![reader readString:@"reason" required:NO into:&reason error:error]
		|| ![reader readNumber:@"fullyResidentBytes" into:&fullyResident error:error]
		|| ![reader readNumber:@"budgetBytes" into:&budget error:error]
		|| ![reader readNumber:@"techniqueBytes" into:&technique error:error]) {
		return nil;
	}
	if (reader.dictionary[@"available"] != nil && ![reader readBoolean:@"available" into:&available error:error]) {
		return nil;
	}
	return [[self alloc] initWithKey:key available:available choices:choices
					   requiredValue:reader.dictionary[@"requiredValue"] maximumValue:maximum reason:reason
			  fullyResidentByteCount:fullyResident budgetByteCount:budget techniqueByteCount:technique];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"key"] = self.key;
	dictionary[@"available"] = @(self.isAvailable);
	if (self.choices.count > 0) {
		dictionary[@"choices"] = NFKModelDictionariesOf(self.choices);
	}
	dictionary[@"requiredValue"] = self.requiredValue;
	dictionary[@"maximumValue"] = self.maximumValue;
	dictionary[@"reason"] = self.reason;
	dictionary[@"fullyResidentBytes"] = self.fullyResidentByteCount;
	dictionary[@"budgetBytes"] = self.budgetByteCount;
	dictionary[@"techniqueBytes"] = self.techniqueByteCount;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelOptionConstraint.class);
}

- (NSUInteger)hash
{
	return self.key.hash;
}

@end

@implementation NFKModelMemoryConstraints

- (instancetype)initWithBudget:(NFKModelMemoryBudget *)budget
			anyCombinationFits:(BOOL)anyCombinationFits
						reason:(nullable NSString *)reason
			 optionConstraints:(nullable NSDictionary<NSString *, NFKModelOptionConstraint *> *)optionConstraints
				fittingOptions:(nullable NSDictionary<NSString *, id> *)fittingOptions
			   fittingEstimate:(nullable NFKModelMemoryEstimate *)fittingEstimate
					exhaustive:(BOOL)exhaustive
{
	self = [super init];
	if (self != nil) {
		_budget = [budget copy];
		_anyCombinationFits = anyCombinationFits;
		_exhaustive = exhaustive;
		_reason = [reason copy];
		_optionConstraints = [optionConstraints copy] ?: @{};
		_fittingOptions = [fittingOptions copy];
		_fittingEstimate = [fittingEstimate copy];
	}
	return self;
}

- (nullable NFKModelOptionConstraint *)constraintForOptionKey:(NSString *)key
{
	return self.optionConstraints[key];
}

+ (nullable instancetype)constraintsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:dictionary path:@"constraints" error:error];
	id budget = nil, estimate = nil;
	NSString *reason = nil;
	NSDictionary *options = nil, *fittingOptions = nil;
	BOOL anyFits = NO, exhaustive = YES;
	NFKModelElementReader budgetReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelMemoryBudget budgetFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader optionReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelOptionConstraint constraintFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader estimateReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [element isKindOfClass:NSDictionary.class]
			? [NFKModelMemoryEstimate estimateWithDictionary:element error:elementError] : nil;
	};
	if (reader == nil
		|| ![reader readObject:@"budget" withReader:budgetReader into:&budget error:error]
		|| ![reader readBoolean:@"anyCombinationFits" into:&anyFits error:error]
		|| ![reader readString:@"reason" required:NO into:&reason error:error]
		|| ![reader readObjectDictionary:@"options" withReader:optionReader into:&options error:error]
		|| ![reader readJSONDictionary:@"fittingOptions" into:&fittingOptions error:error]
		|| ![reader readObject:@"fittingEstimate" withReader:estimateReader into:&estimate error:error]) {
		return nil;
	}
	if (budget == nil) {
		[reader failField:@"budget" reason:@"is missing" error:error];
		return nil;
	}
	if (reader.dictionary[@"exhaustive"] != nil && ![reader readBoolean:@"exhaustive" into:&exhaustive error:error]) {
		return nil;
	}
	return [[self alloc] initWithBudget:budget anyCombinationFits:anyFits reason:reason optionConstraints:options
						 fittingOptions:fittingOptions fittingEstimate:estimate exhaustive:exhaustive];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"budget"] = self.budget.dictionaryRepresentation;
	dictionary[@"anyCombinationFits"] = @(self.anyCombinationFits);
	dictionary[@"exhaustive"] = @(self.isExhaustive);
	dictionary[@"reason"] = self.reason;
	NSMutableDictionary<NSString *, id> *options = [NSMutableDictionary dictionary];
	[self.optionConstraints enumerateKeysAndObjectsUsingBlock:^(NSString *key, NFKModelOptionConstraint *constraint, BOOL *stop) {
		options[key] = constraint.dictionaryRepresentation;
	}];
	dictionary[@"options"] = options;
	dictionary[@"fittingOptions"] = self.fittingOptions;
	dictionary[@"fittingEstimate"] = self.fittingEstimate.dictionaryRepresentation;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelMemoryConstraints.class);
}

- (NSUInteger)hash
{
	return self.optionConstraints.count ^ (NSUInteger)self.anyCombinationFits;
}

@end

#pragma mark - Estimating

// One combination of the memory-affecting choices and what it costs.
@interface NFKModelMemoryCombination : NSObject
@property (nonatomic, copy) NSDictionary<NSString *, id> *choices;
@property (nonatomic, copy) NSDictionary<NSString *, id> *options;
@property (nonatomic) NSInteger peak;
@property (nonatomic) BOOL fits;
// option key -> the largest value of a linear option that fits, -1 where none does.
@property (nonatomic, copy) NSDictionary<NSString *, NSNumber *> *linearMaximums;
// The option a one-at-a-time combination varies; nil in an exhaustive enumeration.
@property (nonatomic, copy, nullable) NSString *variedKey;
@end

@implementation NFKModelMemoryCombination
@end

@interface NFKModelDescriptor (NFKModelMemoryCombinations)
- (NSArray<NFKModelMemoryCombination *> *)combinationsOfOptions:(NSArray<NFKModelOption *> *)choiceOptions
													candidates:(NSArray<NSArray *> *)candidates
												 linearOptions:(NSArray<NFKModelOption *> *)linearOptions
													  defaults:(NSDictionary<NSString *, id> *)defaults
														 given:(NSDictionary<NSString *, id> *)given
														budget:(NFKModelMemoryBudget *)budget
													   catalog:(NFKModelCatalog *)catalog;
- (NSArray<NFKModelMemoryCombination *> *)oneAtATimeCombinationsOfOptions:(NSArray<NFKModelOption *> *)choiceOptions
															   candidates:(NSArray<NSArray *> *)candidates
															linearOptions:(NSArray<NFKModelOption *> *)linearOptions
																	 base:(NSDictionary<NSString *, id> *)base
																 defaults:(NSDictionary<NSString *, id> *)defaults
																	given:(NSDictionary<NSString *, id> *)given
																   budget:(NFKModelMemoryBudget *)budget
																  catalog:(NFKModelCatalog *)catalog;
- (NSArray<NFKModelMemoryCombination *> *)combinations:(NSArray<NFKModelMemoryCombination *> *)combinations
											  matching:(NSDictionary<NSString *, id> *)given
												except:(nullable NSString *)exceptKey;
- (NFKModelOptionConstraint *)constraintOfChoiceOption:(NFKModelOption *)option
											candidates:(NSArray *)candidates
										   unavailable:(NSDictionary<NSString *, NSString *> *)unavailable
										  combinations:(NSArray<NFKModelMemoryCombination *> *)combinations
												 given:(NSDictionary<NSString *, id> *)given
											  defaults:(NSDictionary<NSString *, id> *)defaults
												budget:(NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog;
- (nullable NFKModelOptionConstraint *)constraintOfLinearOption:(NFKModelOption *)option
													   matching:(NSArray<NFKModelMemoryCombination *> *)matching
														 budget:(NFKModelMemoryBudget *)budget;
- (void)addRequirementConstraintsTo:(NSMutableDictionary<NSString *, NFKModelOptionConstraint *> *)constraints
							profile:(NFKHardwareProfile *)profile;
- (NFKModelMemoryConstraints *)constraintsWithMatching:(NSArray<NFKModelMemoryCombination *> *)matching
									 optionConstraints:(NSDictionary<NSString *, NFKModelOptionConstraint *> *)constraints
										 choiceOptions:(NSArray<NFKModelOption *> *)choiceOptions
										 linearOptions:(NSArray<NFKModelOption *> *)linearOptions
											  preferred:(NSDictionary<NSString *, id> *)preferred
												 given:(NSDictionary<NSString *, id> *)given
												budget:(NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog
											exhaustive:(BOOL)exhaustive;
- (double)bytesPerUnitOfLinearOption:(NFKModelOption *)option
							 options:(NSDictionary<NSString *, id> *)options
							 catalog:(NFKModelCatalog *)catalog;
@end

// One stage of a combination: its weights at the stage's precision after any quantization, and the
// cache, draft, and linear-option bytes the cache stage holds beside them.
@interface NFKModelStageLoad : NSObject
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic) NSInteger weights;
@property (nonatomic) NSInteger pageable;
@property (nonatomic) NSInteger layerCount;
@property (nonatomic) NSInteger largestLayer;
@property (nonatomic, copy, nullable) NSNumber *widenedWeights;
@property (nonatomic) NSInteger extra;
@end

@implementation NFKModelStageLoad

- (NFKModelStageFootprint *)footprint
{
	NSInteger bytes = self.weights + self.extra;
	NSMutableArray<NSNumber *> *layers = [NSMutableArray array];
	NSInteger layer = self.largestLayer;
	if (self.layerCount > 0 && layer > 0) {
		layer = MIN(layer, self.weights / self.layerCount);
		for (NSInteger index = 0; index < self.layerCount; index += 1) {
			[layers addObject:@(layer)];
		}
	}
	NSNumber *widened = self.widenedWeights != nil ? @(self.widenedWeights.integerValue + self.extra) : nil;
	return [[NFKModelStageFootprint alloc] initWithByteCount:bytes pageableByteCount:MIN(self.pageable, self.weights)
											widenedByteCount:widened layerByteCounts:layers];
}

@end

@implementation NFKModelDescriptor (NFKModelMemoryInternal)

- (NSString *)requestedResidencyForOptions:(NSDictionary<NSString *, id> *)resolved
{
	NSString *key = self.memoryModel.residencyOptionKey;
	id value = key != nil ? resolved[key] : nil;
	if ([value isKindOfClass:NSString.class]) {
		return value;
	}
	return self.residencies.count == 1 ? self.residencies.firstObject : NFKModelResidencyAutomatic;
}

- (nullable id)memoryOptionValue:(nullable NSString *)key in:(NSDictionary<NSString *, id> *)resolved
{
	return key != nil ? resolved[key] : nil;
}

- (NFKModelMemoryEstimate *)estimateForResolvedOptions:(NSDictionary<NSString *, id> *)resolved
										  memoryBudget:(nullable NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog
{
	NFKModelVariant *variant = [self variantWithIdentifier:[resolved[NFKModelLoadOptionVariant] description] ?: @""]
		?: self.defaultVariant;
	NSString *requested = [self requestedResidencyForOptions:resolved];
	NSMutableDictionary<NSString *, NSNumber *> *components = [NSMutableDictionary dictionary];
	NSArray<NFKModelStageLoad *> *loads = nil;
	NSInteger overhead = 0;
	NSString *basis = nil;
	if (variant.memoryFacts != nil) {
		loads = [self stageLoadsForVariant:variant options:resolved catalog:catalog components:components];
		overhead = variant.memoryFacts.loadOverheadByteCount;
		basis = variant.memoryFacts.basis;
	} else if (variant.residentByteCount != nil) {
		NFKModelStageLoad *load = [[NFKModelStageLoad alloc] init];
		load.identifier = variant.identifier;
		load.weights = variant.residentByteCount.integerValue;
		loads = @[ load ];
		overhead = MAX(0, (variant.peakByteCount ?: variant.residentByteCount).integerValue - load.weights);
		basis = NFKModelMemoryBasisDeclared;
	} else {
		loads = @[];
		basis = NFKModelMemoryBasisUnknown;
	}
	return [self estimateWithLoads:loads requested:requested overhead:overhead basis:basis components:components
						  resolved:resolved budget:budget];
}

- (NFKModelMemoryEstimate *)estimateWithLoads:(NSArray<NFKModelStageLoad *> *)loads
									requested:(NSString *)requested
									 overhead:(NSInteger)overhead
										basis:(NSString *)basis
								   components:(NSMutableDictionary<NSString *, NSNumber *> *)components
									 resolved:(NSDictionary<NSString *, id> *)resolved
									   budget:(nullable NFKModelMemoryBudget *)budget
{
	NSInteger budgetBytes = budget.byteCount;
	NSMutableArray<NFKModelStageFootprint *> *footprints = [NSMutableArray arrayWithCapacity:loads.count];
	for (NFKModelStageLoad *load in loads) {
		[footprints addObject:load.footprint];
	}
	NSString *refusal = nil;
	NFKModelResidencyPlan *plan = [NFKModelResidencyPlan placementForStages:footprints residency:requested
																	 budget:budgetBytes refusal:&refusal];
	NSMutableDictionary<NSString *, NSNumber *> *stageBytes = [NSMutableDictionary dictionary];
	NSInteger resident = 0, heldWeights = 0;
	NSInteger largestIndex = -1, largestHeld = -1;
	for (NSUInteger index = 0; index < loads.count; index += 1) {
		NFKModelStageFootprint *footprint = footprints[index];
		NSInteger held = [self heldBytesOfStage:index footprint:footprint plan:plan];
		stageBytes[loads[index].identifier] = @(held);
		if (plan.holdsStagesResident) {
			resident += held;
			heldWeights += held - loads[index].extra;
		} else if (held > largestHeld) {
			largestHeld = held;
			largestIndex = (NSInteger)index;
		}
	}
	NSInteger extras = components[NFKModelMemoryComponentKeyValueCache].integerValue
		+ components[NFKModelMemoryComponentDraft].integerValue
		+ components[NFKModelMemoryComponentLinearOptions].integerValue;
	if (!plan.holdsStagesResident && largestIndex >= 0) {
		NFKModelStageLoad *largest = loads[(NSUInteger)largestIndex];
		resident = largestHeld + (plan.pagesExperts ? plan.expertCacheByteCount : 0);
		heldWeights = largestHeld - largest.extra;
		if (largest.extra == 0 && extras > 0) {
			components[NFKModelMemoryComponentKeyValueCache] = @0;
			components[NFKModelMemoryComponentDraft] = @0;
			components[NFKModelMemoryComponentLinearOptions] = @0;
		}
		if (plan.pagesExperts) {
			components[NFKModelMemoryComponentExpertCache] = @(plan.expertCacheByteCount);
		}
	}
	components[NFKModelMemoryComponentWeights] = @(heldWeights);
	components[NFKModelMemoryComponentLoadOverhead] = @(overhead);
	NSInteger peak = resident + overhead;
	BOOL fits = budget == nil || (refusal == nil && [budget admitsByteCount:peak]);
	NSString *reason = fits ? nil : [NSString stringWithFormat:@"The model needs %@ plus a %@ reserve, against a %@ budget.",
									 NFKModelGiB(peak), NFKModelReserveGiB(budgetBytes), NFKModelGiB(budgetBytes)];
	return [[NFKModelMemoryEstimate alloc] initWithResidentByteCount:resident peakByteCount:peak requestedResidency:requested
														   residency:plan.residency components:components
													 stageByteCounts:stageBytes resolvedOptions:resolved budget:budget
																fits:fits reason:reason basis:basis];
}

// What a stage holds in its turn under the plan.
- (NSInteger)heldBytesOfStage:(NSUInteger)index footprint:(NFKModelStageFootprint *)footprint plan:(NFKModelResidencyPlan *)plan
{
	if ([plan widensStageAtIndex:index]) {
		return footprint.widenedByteCount.integerValue;
	}
	if ([plan streamsStageAtIndex:index]) {
		NSRange held = [plan heldLayerRangeOfStageAtIndex:index footprint:footprint];
		NSInteger layers = 0;
		for (NSUInteger layer = held.location; layer < NSMaxRange(held); layer += 1) {
			layers += footprint.layerByteCounts[layer].integerValue;
		}
		return MIN(footprint.byteCount, footprint.streamedMinimumByteCount + layers);
	}
	if (plan.pagesExperts) {
		return footprint.unpagedByteCount;
	}
	return footprint.byteCount;
}

// The stages of a variant under the options, with the cache, draft, and linear options on the cache stage.
- (NSArray<NFKModelStageLoad *> *)stageLoadsForVariant:(NFKModelVariant *)variant
											   options:(NSDictionary<NSString *, id> *)resolved
											   catalog:(NFKModelCatalog *)catalog
											components:(NSMutableDictionary<NSString *, NSNumber *> *)components
{
	NFKModelMemoryModel *model = self.memoryModel;
	NFKModelMemoryFacts *facts = variant.memoryFacts;
	id precisionValue = [self memoryOptionValue:model.precisionOptionKey in:resolved];
	NSString *precision = [precisionValue isKindOfClass:NSString.class] ? precisionValue : nil;
	NSInteger bits = NFKModelIntegerValue([self memoryOptionValue:model.quantizationBitsOptionKey in:resolved]);
	NSInteger group = NFKModelIntegerValue([self memoryOptionValue:model.quantizationGroupSizeOptionKey in:resolved]);
	NSMutableArray<NFKModelStageLoad *> *loads = [NSMutableArray arrayWithCapacity:facts.stages.count];
	for (NFKModelStageFacts *stage in facts.stages) {
		NSString *stagePrecision = model.stagePrecisions[stage.identifier] ?: precision ?: NFKModelPrecisionCheckpoint;
		[loads addObject:[self loadOfStage:stage precision:stagePrecision bits:bits group:group
						checkpointPrecision:variant.precision]];
	}
	if (loads.count == 0) {
		return loads;
	}
	NSUInteger cacheIndex = loads.count - 1;
	for (NSUInteger index = 0; index < loads.count; index += 1) {
		if ([loads[index].identifier isEqualToString:model.cacheStage ?: @""]) {
			cacheIndex = index;
		}
	}
	NFKModelStageFacts *cacheFacts = facts.stages[cacheIndex];
	NSString *cachePrecision = model.stagePrecisions[cacheFacts.identifier] ?: precision ?: NFKModelPrecisionCheckpoint;
	NSString *concreteCachePrecision = NFKModelConcretePrecision(cachePrecision, variant.precision);
	NSInteger keyValue = [self keyValueBytesForFacts:facts options:resolved precision:concreteCachePrecision];
	NSInteger draft = [self draftBytesForOptions:resolved precision:precision group:group
								  cachePrecision:concreteCachePrecision
										 context:[self contextLengthIn:resolved facts:facts] catalog:catalog];
	NSInteger linear = [self linearOptionBytesForOptions:resolved];
	components[NFKModelMemoryComponentKeyValueCache] = @(keyValue);
	components[NFKModelMemoryComponentDraft] = @(draft);
	components[NFKModelMemoryComponentLinearOptions] = @(linear);
	loads[cacheIndex].extra = keyValue + draft + linear;
	return loads;
}

- (NFKModelStageLoad *)loadOfStage:(NFKModelStageFacts *)stage
						 precision:(NSString *)precision
							  bits:(NSInteger)bits
							 group:(NSInteger)group
			   checkpointPrecision:(nullable NSString *)checkpointPrecision
{
	NFKModelStageLoad *load = [[NFKModelStageLoad alloc] init];
	load.identifier = stage.identifier;
	NSInteger atPrecision = NFKModelBytesAtPrecision(stage.byteCountByPrecision, precision, checkpointPrecision);
	NSInteger weights = atPrecision;
	BOOL quantizes = bits > 0 && stage.quantizableElementCount > 0;
	if (quantizes) {
		double elementBytes = NFKModelElementByteCount(NFKModelConcretePrecision(precision, checkpointPrecision));
		NSInteger replaced = (NSInteger)llround((double)stage.quantizableElementCount * (elementBytes > 0 ? elementBytes : 2));
		weights = MAX(0, atPrecision - replaced + NFKModelQuantizedByteCount((double)stage.quantizableElementCount, bits, group));
	}
	double scale = atPrecision > 0 ? (double)weights / (double)atPrecision : 1;
	load.weights = weights;
	load.pageable = (NSInteger)((double)NFKModelBytesAtPrecision(stage.pageableByteCountByPrecision, precision, checkpointPrecision) * scale);
	load.layerCount = stage.layerCount;
	load.largestLayer = (NSInteger)((double)NFKModelBytesAtPrecision(stage.largestLayerByteCountByPrecision, precision,
																	 checkpointPrecision) * scale);
	if (stage.widenedPrecision != nil && !quantizes && ![stage.widenedPrecision isEqualToString:precision]) {
		load.widenedWeights = @(NFKModelBytesAtPrecision(stage.byteCountByPrecision, stage.widenedPrecision, checkpointPrecision));
	}
	return load;
}

// The positions the cache holds: the context-length option's value, else the facts' maximum.
- (NSInteger)contextLengthIn:(NSDictionary<NSString *, id> *)resolved facts:(NFKModelMemoryFacts *)facts
{
	id contextValue = [self memoryOptionValue:self.memoryModel.contextLengthOptionKey in:resolved];
	return contextValue != nil ? NFKModelIntegerValue(contextValue) : facts.maximumContextLength.integerValue;
}

// The cache that sixteenBitBytes of 16-bit cache take in the run: quantized to the run's cache bits, else
// at the element size of precision.
- (NSInteger)cacheBytesForSixteenBitBytes:(double)sixteenBitBytes
								  options:(NSDictionary<NSString *, id> *)resolved
								precision:(nullable NSString *)precision
{
	if (sixteenBitBytes <= 0) {
		return 0;
	}
	NSInteger cacheBits = NFKModelIntegerValue([self memoryOptionValue:self.memoryModel.cacheBitsOptionKey in:resolved]);
	if (cacheBits > 0) {
		NSInteger cacheGroup = NFKModelIntegerValue([self memoryOptionValue:self.memoryModel.cacheGroupSizeOptionKey
																		  in:resolved]);
		return NFKModelQuantizedByteCount(sixteenBitBytes / 2.0, cacheBits, cacheGroup);
	}
	double elementBytes = NFKModelElementByteCount(precision);
	return (NSInteger)llround(sixteenBitBytes * (elementBytes > 0 ? elementBytes : 2) / 2.0);
}

// The bytes one 16-bit cache byte takes in the run, the per-unit rate of cacheBytesForSixteenBitBytes:.
- (double)cacheBytesPerSixteenBitByteWithOptions:(NSDictionary<NSString *, id> *)resolved
									   precision:(nullable NSString *)precision
{
	NSInteger cacheBits = NFKModelIntegerValue([self memoryOptionValue:self.memoryModel.cacheBitsOptionKey in:resolved]);
	if (cacheBits > 0) {
		NSInteger cacheGroup = NFKModelIntegerValue([self memoryOptionValue:self.memoryModel.cacheGroupSizeOptionKey
																		  in:resolved]);
		NSInteger group = cacheGroup > 0 ? cacheGroup : NFKModelDefaultQuantizationGroupSize;
		return ((double)cacheBits / 8.0 + 4.0 / (double)group) / 2.0;
	}
	double elementBytes = NFKModelElementByteCount(precision);
	return (elementBytes > 0 ? elementBytes : 2) / 2.0;
}

// The key-value cache: (perToken·c + bounded) 16-bit bytes, at the cache stage's element size or quantized.
- (NSInteger)keyValueBytesForFacts:(NFKModelMemoryFacts *)facts
						   options:(NSDictionary<NSString *, id> *)resolved
						 precision:(nullable NSString *)precision
{
	double sixteenBitBytes = facts.keyValueByteCountPerToken.doubleValue * (double)[self contextLengthIn:resolved facts:facts]
		+ facts.boundedKeyValueByteCount.doubleValue;
	return [self cacheBytesForSixteenBitBytes:sixteenBitBytes options:resolved precision:precision];
}

// The draft descriptor of the options, or nil for none.
- (nullable NFKModelDescriptor *)draftDescriptorForOptions:(NSDictionary<NSString *, id> *)resolved
												   catalog:(NFKModelCatalog *)catalog
{
	id draftValue = [self memoryOptionValue:self.memoryModel.draftOptionKey in:resolved];
	if (![draftValue isKindOfClass:NSString.class] || [draftValue length] == 0) {
		return nil;
	}
	return [catalog descriptorWithIdentifier:draftValue];
}

// The draft model: its draft stage's weights, and its own cache at the run's context length, cache
// precision, and cache bits.
- (NSInteger)draftBytesForOptions:(NSDictionary<NSString *, id> *)resolved
						precision:(nullable NSString *)precision
							group:(NSInteger)group
				   cachePrecision:(nullable NSString *)cachePrecision
						  context:(NSInteger)context
						  catalog:(NFKModelCatalog *)catalog
{
	NFKModelDescriptor *draft = [self draftDescriptorForOptions:resolved catalog:catalog];
	NFKModelVariant *variant = draft.defaultVariant;
	if (variant == nil) {
		return 0;
	}
	if (variant.memoryFacts == nil) {
		return variant.residentByteCount.integerValue;
	}
	NSInteger bits = NFKModelIntegerValue([self memoryOptionValue:self.memoryModel.draftBitsOptionKey in:resolved]);
	NSString *draftStage = draft.memoryModel.draftStage;
	NSInteger total = 0;
	for (NFKModelStageFacts *stage in variant.memoryFacts.stages) {
		if (draftStage != nil && ![stage.identifier isEqualToString:draftStage]) {
			continue;
		}
		NSString *stagePrecision = draft.memoryModel.stagePrecisions[stage.identifier] ?: precision ?: NFKModelPrecisionCheckpoint;
		total += [self loadOfStage:stage precision:stagePrecision bits:bits group:group
			   checkpointPrecision:variant.precision].weights;
	}
	NFKModelMemoryFacts *facts = variant.memoryFacts;
	double sixteenBitBytes = facts.keyValueByteCountPerToken.doubleValue * (double)context
		+ facts.boundedKeyValueByteCount.doubleValue;
	return total + [self cacheBytesForSixteenBitBytes:sixteenBitBytes options:resolved precision:cachePrecision];
}

// The bytes the integer options with memoryBytesPerUnit add, the context length among them only when the
// facts state no per-token cache.
- (NSInteger)linearOptionBytesForOptions:(NSDictionary<NSString *, id> *)resolved
{
	NSInteger total = 0;
	NSString *contextKey = self.memoryModel.contextLengthOptionKey;
	BOOL factsSizeContext = NO;
	NFKModelVariant *variant = [self variantWithIdentifier:[resolved[NFKModelLoadOptionVariant] description] ?: @""]
		?: self.defaultVariant;
	factsSizeContext = variant.memoryFacts.keyValueByteCountPerToken != nil;
	for (NFKModelOption *option in self.options) {
		if (option.memoryBytesPerUnit == nil || resolved[option.key] == nil) {
			continue;
		}
		if ([option.key isEqualToString:contextKey ?: @""] && factsSizeContext) {
			continue;
		}
		total += NFKModelIntegerValue(resolved[option.key]) * option.memoryBytesPerUnit.integerValue;
	}
	return total;
}

@end

#pragma mark - Combinations

@implementation NFKModelDescriptor (NFKModelMemoryCombinations)

- (NSArray<NFKModelMemoryCombination *> *)combinationsOfOptions:(NSArray<NFKModelOption *> *)choiceOptions
													candidates:(NSArray<NSArray *> *)candidates
												 linearOptions:(NSArray<NFKModelOption *> *)linearOptions
													  defaults:(NSDictionary<NSString *, id> *)defaults
														 given:(NSDictionary<NSString *, id> *)given
														budget:(NFKModelMemoryBudget *)budget
													   catalog:(NFKModelCatalog *)catalog
{
	NSMutableArray<NFKModelMemoryCombination *> *combinations = [NSMutableArray array];
	NSMutableArray<NSNumber *> *counters = [NSMutableArray arrayWithCapacity:choiceOptions.count];
	for (NSArray *values in candidates) {
		if (values.count == 0) {
			return combinations;
		}
		[counters addObject:@0];
	}
	while (YES) {
		NSMutableDictionary<NSString *, id> *choices = [NSMutableDictionary dictionary];
		for (NSUInteger index = 0; index < choiceOptions.count; index += 1) {
			choices[choiceOptions[index].key] = candidates[index][counters[index].unsignedIntegerValue];
		}
		[combinations addObject:[self combinationWithChoices:choices linearOptions:linearOptions defaults:defaults
													   given:given budget:budget catalog:catalog]];
		NSInteger position = (NSInteger)choiceOptions.count - 1;
		while (position >= 0) {
			NSUInteger next = counters[(NSUInteger)position].unsignedIntegerValue + 1;
			if (next < candidates[(NSUInteger)position].count) {
				counters[(NSUInteger)position] = @(next);
				break;
			}
			counters[(NSUInteger)position] = @0;
			position -= 1;
		}
		if (position < 0) {
			break;
		}
	}
	return combinations;
}

// Past the cap: each option's candidates in turn, every other option at its base value (the caller's
// selection, else the default), each combination tagged with the option it varies.
- (NSArray<NFKModelMemoryCombination *> *)oneAtATimeCombinationsOfOptions:(NSArray<NFKModelOption *> *)choiceOptions
															   candidates:(NSArray<NSArray *> *)candidates
															linearOptions:(NSArray<NFKModelOption *> *)linearOptions
																	 base:(NSDictionary<NSString *, id> *)base
																 defaults:(NSDictionary<NSString *, id> *)defaults
																	given:(NSDictionary<NSString *, id> *)given
																   budget:(NFKModelMemoryBudget *)budget
																  catalog:(NFKModelCatalog *)catalog
{
	NSMutableDictionary<NSString *, id> *baseChoices = [NSMutableDictionary dictionary];
	for (NSUInteger index = 0; index < choiceOptions.count; index += 1) {
		NSString *key = choiceOptions[index].key;
		baseChoices[key] = base[key] ?: ([key isEqualToString:self.memoryModel.draftOptionKey ?: @""] ? @""
										 : candidates[index].firstObject);
	}
	NSMutableArray<NFKModelMemoryCombination *> *combinations = [NSMutableArray array];
	for (NSUInteger index = 0; index < choiceOptions.count; index += 1) {
		NSString *key = choiceOptions[index].key;
		for (id value in candidates[index]) {
			NSMutableDictionary<NSString *, id> *choices = [baseChoices mutableCopy];
			choices[key] = value;
			NFKModelMemoryCombination *combination = [self combinationWithChoices:choices linearOptions:linearOptions
																		 defaults:defaults given:given budget:budget
																		  catalog:catalog];
			combination.variedKey = key;
			[combinations addObject:combination];
		}
	}
	return combinations;
}

// The value a linear option is checked at: the given value, else its minimum, else 0.
- (NSNumber *)checkValueOfLinearOption:(NFKModelOption *)option given:(NSDictionary<NSString *, id> *)given
{
	id value = given[option.key];
	if (value != nil) {
		return @(NFKModelIntegerValue(value));
	}
	return option.minimum ?: @0;
}

- (NFKModelMemoryCombination *)combinationWithChoices:(NSDictionary<NSString *, id> *)choices
										linearOptions:(NSArray<NFKModelOption *> *)linearOptions
											 defaults:(NSDictionary<NSString *, id> *)defaults
												given:(NSDictionary<NSString *, id> *)given
											   budget:(NFKModelMemoryBudget *)budget
											  catalog:(NFKModelCatalog *)catalog
{
	NSMutableDictionary<NSString *, id> *options = [defaults mutableCopy];
	[options addEntriesFromDictionary:given];
	[options addEntriesFromDictionary:choices];
	for (NFKModelOption *option in linearOptions) {
		options[option.key] = [self checkValueOfLinearOption:option given:given];
	}
	NSString *draftKey = self.memoryModel.draftOptionKey;
	if (draftKey != nil && [options[draftKey] isEqual:@""]) {
		[options removeObjectForKey:draftKey];
	}
	NSDictionary<NSString *, id> *resolved = [options copy];
	NFKModelMemoryEstimate *estimate = [self estimateForResolvedOptions:resolved memoryBudget:budget catalog:catalog];
	NFKModelMemoryCombination *combination = [[NFKModelMemoryCombination alloc] init];
	combination.choices = choices;
	combination.options = resolved;
	combination.peak = estimate.peakByteCount;
	combination.fits = estimate.fits;
	NSMutableDictionary<NSString *, NSNumber *> *maximums = [NSMutableDictionary dictionary];
	for (NFKModelOption *option in linearOptions) {
		maximums[option.key] = @([self largestFittingValueOfOption:option options:resolved budget:budget catalog:catalog]);
	}
	combination.linearMaximums = maximums;
	return combination;
}

// (budget − reserve − peak at 0) / bytes per unit, clipped to the option's range and rounded down to its
// step; -1 where not even the minimum fits.
- (NSInteger)largestFittingValueOfOption:(NFKModelOption *)option
								 options:(NSDictionary<NSString *, id> *)options
								  budget:(NFKModelMemoryBudget *)budget
								 catalog:(NFKModelCatalog *)catalog
{
	NSInteger minimum = option.minimum.integerValue;
	NSInteger maximum = [self upperBoundOfLinearOption:option options:options];
	NSMutableDictionary<NSString *, id> *atZero = [options mutableCopy];
	atZero[option.key] = @0;
	NFKModelMemoryEstimate *zero = [self estimateForResolvedOptions:atZero memoryBudget:budget catalog:catalog];
	double perUnit = [self bytesPerUnitOfLinearOption:option options:options catalog:catalog];
	NSInteger largest = maximum;
	if (budget.isKnown && perUnit > 0) {
		double room = (double)(budget.byteCount - budget.reserveByteCount - zero.peakByteCount);
		largest = room < 0 ? -1 : (NSInteger)floor(room / perUnit);
	} else if (budget.isKnown && !zero.fits) {
		largest = -1;
	}
	if (largest > maximum) {
		largest = maximum;
	}
	NSInteger step = option.step.integerValue;
	if (step > 1 && largest >= minimum) {
		largest = minimum + (largest - minimum) / step * step;
	}
	return largest < minimum ? -1 : largest;
}

// The bytes one unit of a linear option adds: for the context length, one position of the key-value
// cache, the draft model's included, at the combination's cache precision or quantization; for another
// option, its memoryBytesPerUnit.
- (double)bytesPerUnitOfLinearOption:(NFKModelOption *)option
							 options:(NSDictionary<NSString *, id> *)options
							 catalog:(NFKModelCatalog *)catalog
{
	NFKModelMemoryModel *model = self.memoryModel;
	NFKModelVariant *variant = [self variantWithIdentifier:[options[NFKModelLoadOptionVariant] description] ?: @""]
		?: self.defaultVariant;
	NSNumber *perToken = variant.memoryFacts.keyValueByteCountPerToken;
	if (![option.key isEqualToString:model.contextLengthOptionKey ?: @""] || perToken == nil) {
		return option.memoryBytesPerUnit.doubleValue;
	}
	NFKModelStageFacts *cacheStage = [variant.memoryFacts stageWithIdentifier:model.cacheStage ?: @""]
		?: variant.memoryFacts.stages.lastObject;
	id precisionValue = [self memoryOptionValue:model.precisionOptionKey in:options];
	NSString *precision = model.stagePrecisions[cacheStage.identifier ?: @""]
		?: ([precisionValue isKindOfClass:NSString.class] ? precisionValue : NFKModelPrecisionCheckpoint);
	NSString *concrete = NFKModelConcretePrecision(precision, variant.precision);
	double draftPerToken = [self draftDescriptorForOptions:options catalog:catalog]
		.defaultVariant.memoryFacts.keyValueByteCountPerToken.doubleValue;
	return (perToken.doubleValue + draftPerToken) * [self cacheBytesPerSixteenBitByteWithOptions:options precision:concrete];
}

- (NSInteger)upperBoundOfLinearOption:(NFKModelOption *)option options:(NSDictionary<NSString *, id> *)options
{
	NSInteger bound = option.maximum != nil ? option.maximum.integerValue : NSIntegerMax / 2;
	if ([option.key isEqualToString:self.memoryModel.contextLengthOptionKey ?: @""]) {
		NFKModelVariant *variant = [self variantWithIdentifier:[options[NFKModelLoadOptionVariant] description] ?: @""]
			?: self.defaultVariant;
		NSNumber *context = variant.memoryFacts.maximumContextLength;
		if (context != nil) {
			bound = MIN(bound, context.integerValue);
		}
	}
	return bound;
}

- (NSArray<NFKModelMemoryCombination *> *)combinations:(NSArray<NFKModelMemoryCombination *> *)combinations
											  matching:(NSDictionary<NSString *, id> *)given
												except:(nullable NSString *)exceptKey
{
	NSMutableArray<NFKModelMemoryCombination *> *matching = [NSMutableArray array];
	for (NFKModelMemoryCombination *combination in combinations) {
		BOOL matches = YES;
		for (NSString *key in combination.choices) {
			id value = given[key];
			if (value == nil || [key isEqualToString:exceptKey ?: @""]) {
				continue;
			}
			if (![[combination.choices[key] description] isEqualToString:[value description]]) {
				matches = NO;
				break;
			}
		}
		if (matches) {
			[matching addObject:combination];
		}
	}
	return matching;
}

- (nullable NFKModelMemoryCombination *)smallestOf:(NSArray<NFKModelMemoryCombination *> *)combinations
										 fittingOnly:(BOOL)fittingOnly
{
	NFKModelMemoryCombination *smallest = nil;
	for (NFKModelMemoryCombination *combination in combinations) {
		if (fittingOnly && !combination.fits) {
			continue;
		}
		if (smallest == nil || combination.peak < smallest.peak) {
			smallest = combination;
		}
	}
	return smallest;
}

- (NSString *)titleOfValue:(id)value option:(NFKModelOption *)option
{
	if ([value isEqual:@""] && [option.key isEqualToString:self.memoryModel.draftOptionKey ?: @""]) {
		return @"No draft model";
	}
	NFKModelOptionChoice *choice = [option choiceWithValue:value];
	return choice.title ?: [value description];
}

- (NFKModelOptionConstraint *)constraintOfChoiceOption:(NFKModelOption *)option
											candidates:(NSArray *)candidates
										   unavailable:(NSDictionary<NSString *, NSString *> *)unavailable
										  combinations:(NSArray<NFKModelMemoryCombination *> *)combinations
												 given:(NSDictionary<NSString *, id> *)given
											  defaults:(NSDictionary<NSString *, id> *)defaults
												budget:(NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog
{
	NSArray<NFKModelMemoryCombination *> *matching = [self combinations:combinations matching:given except:option.key];
	NSMutableArray<NFKModelChoiceConstraint *> *choices = [NSMutableArray array];
	NSMutableArray *allowed = [NSMutableArray array];
	BOOL isResidency = [option.key isEqualToString:self.memoryModel.residencyOptionKey ?: @""];
	NSUInteger countedCandidates = 0;
	NSMutableDictionary<NSString *, NFKModelMemoryCombination *> *smallestByValue = [NSMutableDictionary dictionary];
	for (id value in candidates) {
		NSMutableArray<NFKModelMemoryCombination *> *withValue = [NSMutableArray array];
		for (NFKModelMemoryCombination *combination in matching) {
			if ([[combination.choices[option.key] description] isEqualToString:[value description]]) {
				[withValue addObject:combination];
			}
		}
		NFKModelMemoryCombination *fitting = [self smallestOf:withValue fittingOnly:YES];
		NFKModelMemoryCombination *smallest = fitting ?: [self smallestOf:withValue fittingOnly:NO];
		smallestByValue[[value description]] = smallest;
		BOOL counted = !(isResidency && [value isEqual:NFKModelResidencyAutomatic]);
		countedCandidates += counted ? 1 : 0;
		if (fitting != nil) {
			if (counted) {
				[allowed addObject:value];
			}
			[choices addObject:[[NFKModelChoiceConstraint alloc] initWithValue:value status:NFKModelChoiceStatusAllowed
																		reason:nil peakByteCount:@(fitting.peak)]];
			continue;
		}
		NSString *reason = [NSString stringWithFormat:@"%@ needs at least %@ plus a %@ reserve, against a %@ budget.",
							[self titleOfValue:value option:option], NFKModelGiB(smallest.peak),
							NFKModelGiB(budget.reserveByteCount), NFKModelGiB(budget.byteCount)];
		[choices addObject:[[NFKModelChoiceConstraint alloc] initWithValue:value status:NFKModelChoiceStatusRuledOut
																	reason:reason peakByteCount:smallest != nil ? @(smallest.peak) : nil]];
	}
	for (NFKModelOptionChoice *choice in option.choices) {
		NSString *problem = unavailable[[choice.value description]];
		if (problem != nil) {
			NSString *reason = [NSString stringWithFormat:@"%@ is unavailable here. %@", choice.title, problem];
			[choices addObject:[[NFKModelChoiceConstraint alloc] initWithValue:choice.value status:NFKModelChoiceStatusRuledOut
																		reason:reason peakByteCount:nil]];
		}
	}
	if (allowed.count != 1 || countedCandidates < 2) {
		return [[NFKModelOptionConstraint alloc] initWithKey:option.key available:YES choices:choices requiredValue:nil
												maximumValue:nil reason:nil fullyResidentByteCount:nil budgetByteCount:nil
										   techniqueByteCount:nil];
	}
	id required = allowed.firstObject;
	NSMutableArray<NFKModelChoiceConstraint *> *marked = [NSMutableArray arrayWithCapacity:choices.count];
	for (NFKModelChoiceConstraint *choice in choices) {
		BOOL isRequired = [[choice.value description] isEqualToString:[required description]];
		[marked addObject:isRequired ? [[NFKModelChoiceConstraint alloc] initWithValue:choice.value
																				 status:NFKModelChoiceStatusRequired
																				 reason:nil
																		  peakByteCount:choice.peakByteCount] : choice];
	}
	NFKModelMemoryCombination *technique = smallestByValue[[required description]];
	NSInteger compared = 0;
	NSString *reason = nil;
	if (isResidency) {
		compared = [self fullyResidentPeakOf:technique budget:budget catalog:catalog];
		reason = [NSString stringWithFormat:@"The model needs %@ fully resident plus a %@ reserve, against a %@ budget. %@, it needs %@.",
				  NFKModelGiB(compared), NFKModelGiB(budget.reserveByteCount), NFKModelGiB(budget.byteCount),
				  [self titleOfValue:required option:option], NFKModelGiB(technique.peak)];
	} else {
		id defaultValue = defaults[option.key] ?: @"";
		NFKModelMemoryCombination *atDefault = smallestByValue[[defaultValue description]];
		compared = atDefault != nil ? atDefault.peak : technique.peak;
		if ([[defaultValue description] isEqualToString:[required description]]) {
			reason = [NSString stringWithFormat:@"Only %@ fits: it needs %@ plus a %@ reserve, against a %@ budget.",
					  [self titleOfValue:required option:option], NFKModelGiB(technique.peak),
					  NFKModelGiB(budget.reserveByteCount), NFKModelGiB(budget.byteCount)];
		} else {
			reason = [NSString stringWithFormat:@"With %@, the model needs %@ plus a %@ reserve, against a %@ budget. With %@, it needs %@.",
					  [self titleOfValue:defaultValue option:option], NFKModelGiB(compared),
					  NFKModelGiB(budget.reserveByteCount), NFKModelGiB(budget.byteCount),
					  [self titleOfValue:required option:option], NFKModelGiB(technique.peak)];
		}
	}
	return [[NFKModelOptionConstraint alloc] initWithKey:option.key available:YES choices:marked requiredValue:required
											maximumValue:nil reason:reason fullyResidentByteCount:@(compared)
										 budgetByteCount:@(budget.byteCount) techniqueByteCount:@(technique.peak)];
}

// The peak of a combination with its residency set to resident.
- (NSInteger)fullyResidentPeakOf:(NFKModelMemoryCombination *)combination
						  budget:(NFKModelMemoryBudget *)budget
						 catalog:(NFKModelCatalog *)catalog
{
	NSMutableDictionary<NSString *, id> *options = [combination.options mutableCopy];
	NSString *key = self.memoryModel.residencyOptionKey;
	if (key != nil) {
		options[key] = NFKModelResidencyResident;
	}
	return [self estimateForResolvedOptions:options memoryBudget:budget catalog:catalog].peakByteCount;
}

- (nullable NFKModelOptionConstraint *)constraintOfLinearOption:(NFKModelOption *)option
													   matching:(NSArray<NFKModelMemoryCombination *> *)matching
														 budget:(NFKModelMemoryBudget *)budget
{
	if (!budget.isKnown) {
		return nil;
	}
	NSInteger largest = -1;
	for (NFKModelMemoryCombination *combination in matching) {
		largest = MAX(largest, combination.linearMaximums[option.key].integerValue);
	}
	if (largest < 0 || (option.maximum != nil && largest >= option.maximum.integerValue)) {
		return nil;
	}
	NSString *reason = [NSString stringWithFormat:@"%@ fits up to %ld against a %@ budget.", option.title ?: option.key,
						(long)largest, NFKModelGiB(budget.byteCount)];
	return [[NFKModelOptionConstraint alloc] initWithKey:option.key available:YES choices:nil requiredValue:nil
											maximumValue:@(largest) reason:reason fullyResidentByteCount:nil
										 budgetByteCount:@(budget.byteCount) techniqueByteCount:nil];
}

- (void)addRequirementConstraintsTo:(NSMutableDictionary<NSString *, NFKModelOptionConstraint *> *)constraints
							profile:(NFKHardwareProfile *)profile
{
	for (NFKModelOption *option in self.options) {
		NSString *problem = [option.requirements problemForHardwareProfile:profile];
		if (problem != nil) {
			NSString *reason = [NSString stringWithFormat:@"%@ is unavailable here. %@", option.title ?: option.key, problem];
			constraints[option.key] = [[NFKModelOptionConstraint alloc] initWithKey:option.key available:NO choices:nil
																	  requiredValue:nil maximumValue:nil reason:reason
															 fullyResidentByteCount:nil budgetByteCount:nil
																 techniqueByteCount:nil];
			continue;
		}
		if (constraints[option.key] != nil) {
			continue;
		}
		NSMutableArray<NFKModelChoiceConstraint *> *choices = [NSMutableArray array];
		for (NFKModelOptionChoice *choice in option.choices) {
			NSString *choiceProblem = [choice.requirements problemForHardwareProfile:profile];
			NSString *reason = choiceProblem != nil
				? [NSString stringWithFormat:@"%@ is unavailable here. %@", choice.title, choiceProblem] : nil;
			[choices addObject:[[NFKModelChoiceConstraint alloc] initWithValue:choice.value
																		status:choiceProblem != nil ? NFKModelChoiceStatusRuledOut : NFKModelChoiceStatusAllowed
																		reason:reason peakByteCount:nil]];
		}
		BOOL anyRuledOut = NO;
		for (NFKModelChoiceConstraint *choice in choices) {
			anyRuledOut = anyRuledOut || choice.status == NFKModelChoiceStatusRuledOut;
		}
		if (anyRuledOut) {
			constraints[option.key] = [[NFKModelOptionConstraint alloc] initWithKey:option.key available:YES choices:choices
																	  requiredValue:nil maximumValue:nil reason:nil
															 fullyResidentByteCount:nil budgetByteCount:nil
																 techniqueByteCount:nil];
		}
	}
}

- (NFKModelMemoryConstraints *)constraintsWithMatching:(NSArray<NFKModelMemoryCombination *> *)matching
									 optionConstraints:(NSDictionary<NSString *, NFKModelOptionConstraint *> *)constraints
										 choiceOptions:(NSArray<NFKModelOption *> *)choiceOptions
										 linearOptions:(NSArray<NFKModelOption *> *)linearOptions
											  preferred:(NSDictionary<NSString *, id> *)preferred
												 given:(NSDictionary<NSString *, id> *)given
												budget:(NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog
											exhaustive:(BOOL)exhaustive
{
	NFKModelMemoryCombination *nearest = nil;
	NSUInteger nearestChanges = NSUIntegerMax;
	for (NFKModelMemoryCombination *combination in matching) {
		if (!combination.fits) {
			continue;
		}
		NSUInteger changes = 0;
		for (NFKModelOption *option in choiceOptions) {
			id value = combination.choices[option.key];
			id defaultValue = preferred[option.key] ?: @"";
			changes += [[value description] isEqualToString:[defaultValue description]] ? 0 : 1;
		}
		if (changes < nearestChanges) {
			nearest = combination;
			nearestChanges = changes;
		}
	}
	if (nearest == nil) {
		NFKModelMemoryCombination *smallest = [self smallestOf:matching fittingOnly:NO];
		NSString *reason = [NSString stringWithFormat:@"No combination of options fits: the smallest needs %@ plus a %@ reserve, against a %@ budget.",
							NFKModelGiB(smallest.peak), NFKModelGiB(budget.reserveByteCount), NFKModelGiB(budget.byteCount)];
		return [[NFKModelMemoryConstraints alloc] initWithBudget:budget anyCombinationFits:NO reason:reason
											   optionConstraints:constraints fittingOptions:nil fittingEstimate:nil
													  exhaustive:exhaustive];
	}
	NSMutableDictionary<NSString *, id> *fitting = [nearest.options mutableCopy];
	for (NFKModelOption *option in linearOptions) {
		if (given[option.key] != nil) {
			continue;
		}
		NSInteger largest = nearest.linearMaximums[option.key].integerValue;
		id defaultValue = preferred[option.key];
		NSInteger preferred = defaultValue != nil ? NFKModelIntegerValue(defaultValue) : largest;
		fitting[option.key] = @(MIN(preferred, largest));
	}
	NFKModelMemoryEstimate *estimate = [self estimateForResolvedOptions:fitting memoryBudget:budget catalog:catalog];
	return [[NFKModelMemoryConstraints alloc] initWithBudget:budget anyCombinationFits:YES reason:nil
										   optionConstraints:constraints fittingOptions:fitting fittingEstimate:estimate
												  exhaustive:exhaustive];
}

@end

#pragma mark - Public API

@implementation NFKModelDescriptor (NFKModelMemory)

- (nullable NFKModelMemoryEstimate *)estimatedMemoryForOptions:(nullable NSDictionary<NSString *, id> *)options
														 error:(NSError **)error
{
	return [self estimatedMemoryForOptions:options memoryBudget:nil error:error];
}

- (nullable NFKModelMemoryEstimate *)estimatedMemoryForOptions:(nullable NSDictionary<NSString *, id> *)options
												  memoryBudget:(nullable NFKModelMemoryBudget *)budget
														 error:(NSError **)error
{
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	NSDictionary<NSString *, id> *resolved = [self resolvedOptions:options stage:NFKModelOptionStageLoad catalog:catalog
														   profile:nil error:error];
	if (resolved == nil) {
		return nil;
	}
	return [self estimateForResolvedOptions:resolved memoryBudget:budget catalog:catalog];
}

#pragma mark Constraints

- (NFKModelMemoryConstraints *)constraintsForMemoryBudget:(NFKModelMemoryBudget *)budget
{
	return [self constraintsForOptions:nil memoryBudget:budget profile:nil catalog:NFKModelCatalog.sharedCatalog
							 preferred:nil];
}

- (NFKModelMemoryConstraints *)constraintsForHardwareProfile:(NFKHardwareProfile *)profile
{
	return [self constraintsForOptions:nil memoryBudget:[NFKModelMemoryBudget budgetForHardwareProfile:profile]
							   profile:profile catalog:NFKModelCatalog.sharedCatalog preferred:nil];
}

- (NFKModelMemoryConstraints *)constraintsForOptions:(nullable NSDictionary<NSString *, id> *)options
										memoryBudget:(NFKModelMemoryBudget *)budget
{
	return [self constraintsForOptions:options memoryBudget:budget profile:nil catalog:NFKModelCatalog.sharedCatalog
							 preferred:nil];
}

// The options a combination varies: each memory-affecting option with choices, in a fixed order.
- (NSArray<NFKModelOption *> *)memoryChoiceOptions
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	NSMutableArray<NSString *> *keys = [NSMutableArray arrayWithObject:NFKModelLoadOptionVariant];
	[keys addObjectsFromArray:self.memoryModel.optionKeys];
	for (NSString *key in keys) {
		NFKModelOption *option = [self optionWithKey:key];
		if (option == nil || [key isEqualToString:self.memoryModel.contextLengthOptionKey ?: @""]) {
			continue;
		}
		BOOL isDraft = [key isEqualToString:self.memoryModel.draftOptionKey ?: @""];
		if (option.choices.count > 0 || isDraft) {
			[options addObject:option];
		}
	}
	return options;
}

// The integer options whose memory is linear: the context length and any with memoryBytesPerUnit.
- (NSArray<NFKModelOption *> *)linearMemoryOptions
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	for (NFKModelOption *option in self.options) {
		BOOL isContext = [option.key isEqualToString:self.memoryModel.contextLengthOptionKey ?: @""];
		if ((isContext || option.memoryBytesPerUnit != nil) && option.choices.count == 0) {
			[options addObject:option];
		}
	}
	return options;
}

- (NSArray *)candidateValuesOfOption:(NFKModelOption *)option
							 catalog:(NFKModelCatalog *)catalog
							 profile:(nullable NFKHardwareProfile *)profile
						 unavailable:(NSMutableDictionary<NSString *, NSString *> *)unavailable
{
	NSMutableArray *values = [NSMutableArray array];
	BOOL isDraft = [option.key isEqualToString:self.memoryModel.draftOptionKey ?: @""];
	if (isDraft) {
		[values addObject:@""];
		if (option.choices.count == 0) {
			for (NFKModelDescriptor *draft in [self draftDescriptorsForVariant:nil inCatalog:catalog]) {
				[values addObject:draft.identifier];
			}
			return values;
		}
	}
	for (NFKModelOptionChoice *choice in option.choices) {
		NSString *problem = profile != nil ? [choice.requirements problemForHardwareProfile:profile] : nil;
		if (problem != nil) {
			unavailable[[choice.value description]] = problem;
			continue;
		}
		[values addObject:choice.value];
	}
	return values;
}

- (NFKModelMemoryConstraints *)constraintsForOptions:(nullable NSDictionary<NSString *, id> *)options
										memoryBudget:(NFKModelMemoryBudget *)budget
											 profile:(nullable NFKHardwareProfile *)profile
											 catalog:(NFKModelCatalog *)catalog
										   preferred:(nullable NSDictionary<NSString *, id> *)preferredOptions
{
	NSDictionary<NSString *, id> *defaults = [self resolvedOptions:nil stage:NFKModelOptionStageLoad catalog:catalog
														   profile:nil error:NULL] ?: @{};
	NSDictionary<NSString *, id> *given = [self coercedValidOptions:options catalog:catalog];
	NSMutableDictionary<NSString *, id> *preferred = [defaults mutableCopy];
	[preferred addEntriesFromDictionary:[self coercedValidOptions:preferredOptions catalog:catalog]];
	[preferred addEntriesFromDictionary:given];
	NSArray<NFKModelOption *> *choiceOptions = [self memoryChoiceOptions];
	NSArray<NFKModelOption *> *linearOptions = [self linearMemoryOptions];
	NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, NSString *> *> *unavailable = [NSMutableDictionary dictionary];
	NSMutableArray<NSArray *> *candidates = [NSMutableArray array];
	for (NFKModelOption *option in choiceOptions) {
		unavailable[option.key] = [NSMutableDictionary dictionary];
		[candidates addObject:[self candidateValuesOfOption:option catalog:catalog profile:profile
												unavailable:unavailable[option.key]]];
	}
	NSUInteger count = 1;
	for (NSArray *values in candidates) {
		count = values.count == 0 ? 0 : MIN(count * values.count, NFKModelMaximumCombinationCount + 1);
	}
	BOOL exhaustive = count <= NFKModelMaximumCombinationCount;
	NSArray<NFKModelMemoryCombination *> *combinations = exhaustive
		? [self combinationsOfOptions:choiceOptions candidates:candidates linearOptions:linearOptions defaults:defaults
								given:given budget:budget catalog:catalog]
		: [self oneAtATimeCombinationsOfOptions:choiceOptions candidates:candidates linearOptions:linearOptions
										   base:preferred defaults:defaults given:given budget:budget catalog:catalog];
	NSMutableDictionary<NSString *, NFKModelOptionConstraint *> *constraints = [NSMutableDictionary dictionary];
	for (NSUInteger index = 0; index < choiceOptions.count; index += 1) {
		NFKModelOption *option = choiceOptions[index];
		NSArray<NFKModelMemoryCombination *> *varying = combinations;
		if (!exhaustive) {
			varying = [combinations filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(NFKModelMemoryCombination *combination, NSDictionary *bindings) {
				return [combination.variedKey isEqualToString:option.key];
			}]];
		}
		constraints[option.key] = [self constraintOfChoiceOption:option candidates:candidates[index]
													 unavailable:unavailable[option.key] combinations:varying
														   given:given defaults:defaults budget:budget catalog:catalog];
	}
	NSArray<NFKModelMemoryCombination *> *matching = [self combinations:combinations matching:given except:nil];
	for (NFKModelOption *option in linearOptions) {
		NFKModelOptionConstraint *constraint = [self constraintOfLinearOption:option matching:matching budget:budget];
		if (constraint != nil) {
			constraints[option.key] = constraint;
		}
	}
	if (profile != nil) {
		[self addRequirementConstraintsTo:constraints profile:profile];
	}
	return [self constraintsWithMatching:matching optionConstraints:constraints choiceOptions:choiceOptions
						   linearOptions:linearOptions preferred:preferred given:given budget:budget catalog:catalog
							  exhaustive:exhaustive];
}

// The given options that resolve, coerced; a value that does not resolve is left out.
- (NSDictionary<NSString *, id> *)coercedValidOptions:(nullable NSDictionary<NSString *, id> *)options
											  catalog:(NFKModelCatalog *)catalog
{
	NSMutableDictionary<NSString *, id> *coerced = [NSMutableDictionary dictionary];
	[options enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
		NSDictionary<NSString *, id> *one = [self resolvedOptions:@{ key: value } stage:NFKModelOptionStageLoad
														  catalog:catalog profile:nil error:NULL];
		if (one[key] != nil) {
			coerced[key] = one[key];
		}
	}];
	return coerced;
}

#pragma mark Deriving figures

- (NFKModelDescriptor *)descriptorByDerivingMemoryFigures
{
	return [self descriptorByDerivingMemoryFiguresInCatalog:NFKModelCatalog.sharedCatalog];
}

- (NFKModelDescriptor *)descriptorByDerivingMemoryFiguresInCatalog:(NFKModelCatalog *)catalog
{
	NSDictionary<NSString *, id> *defaults = [self resolvedOptions:nil stage:NFKModelOptionStageLoad catalog:catalog
														   profile:nil error:NULL] ?: @{};
	NSSet<NSString *> *memoryKeys = [NSSet setWithArray:self.memoryModel.optionKeys ?: @[]];
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray arrayWithCapacity:self.options.count];
	for (NFKModelOption *option in self.options) {
		BOOL affectsMemory = [memoryKeys containsObject:option.key] || [option.key isEqualToString:NFKModelLoadOptionVariant];
		BOOL isContext = [option.key isEqualToString:self.memoryModel.contextLengthOptionKey ?: @""];
		if (isContext && option.memoryBytesPerUnit == nil) {
			NSInteger perUnit = (NSInteger)llround([self bytesPerUnitOfLinearOption:option options:defaults catalog:catalog]);
			[options addObject:[self option:option withMemoryBytesPerUnit:@(perUnit)]];
		} else if (affectsMemory && option.choices.count > 0) {
			[options addObject:[self option:option withFiguresAt:defaults catalog:catalog]];
		} else {
			[options addObject:option];
		}
	}
	NSMutableArray<NFKModelVariant *> *variants = [NSMutableArray arrayWithCapacity:self.variants.count];
	for (NFKModelVariant *variant in self.variants) {
		[variants addObject:[self variant:variant withFiguresAt:defaults catalog:catalog]];
	}
	return [[NFKModelDescriptor alloc] initWithIdentifier:self.identifier displayName:self.displayName family:self.family
												  summary:self.summary engine:self.engine registryName:self.registryName
										providerClassName:self.providerClassName tasks:self.tasks
										  inputModalities:self.inputModalities outputModalities:self.outputModalities
												   inputs:self.inputs outputs:self.outputs options:options variants:variants
												  license:self.license requirements:self.requirements
											  residencies:self.residencies memoryModel:self.memoryModel presets:self.presets
											   modelTypes:self.modelTypes standIn:self.isStandIn introducedVersion:self.introducedVersion
											   extensions:self.extensions];
}

- (NFKModelOption *)option:(NFKModelOption *)option withMemoryBytesPerUnit:(NSNumber *)perUnit
{
	return [[NFKModelOption alloc] initWithKey:option.key kind:option.kind title:option.title help:option.help
								  defaultValue:option.defaultValue minimum:option.minimum maximum:option.maximum
										  step:option.step choices:option.choices stage:option.stage group:option.group
								  requirements:option.requirements propertyKeyPath:option.propertyKeyPath
							memoryBytesPerUnit:perUnit.integerValue > 0 ? perUnit : nil extensions:option.extensions];
}

- (NFKModelOption *)option:(NFKModelOption *)option
			 withFiguresAt:(NSDictionary<NSString *, id> *)defaults
				   catalog:(NFKModelCatalog *)catalog
{
	NSMutableArray<NFKModelOptionChoice *> *choices = [NSMutableArray arrayWithCapacity:option.choices.count];
	BOOL isResidency = [option.key isEqualToString:self.memoryModel.residencyOptionKey ?: @""];
	for (NFKModelOptionChoice *choice in option.choices) {
		if (isResidency && [choice.value isEqual:NFKModelResidencyAutomatic]) {
			[choices addObject:choice];
			continue;
		}
		NSMutableDictionary<NSString *, id> *options = [defaults mutableCopy];
		options[option.key] = choice.value;
		NFKModelMemoryFigure *memory = choice.memory ?: [self estimateForResolvedOptions:options memoryBudget:nil
																					  catalog:catalog].figure;
		NSMutableDictionary<NSString *, NFKModelMemoryFigure *> *byVariant = [NSMutableDictionary dictionary];
		if (self.variants.count > 1) {
			for (NFKModelVariant *variant in self.variants) {
				options[NFKModelLoadOptionVariant] = variant.identifier;
				byVariant[variant.identifier] = [self estimateForResolvedOptions:options memoryBudget:nil catalog:catalog].figure;
			}
		}
		[byVariant addEntriesFromDictionary:choice.memoryByVariant];
		[choices addObject:[[NFKModelOptionChoice alloc] initWithValue:choice.value title:choice.title
														  requirements:choice.requirements memory:memory
													   memoryByVariant:byVariant extensions:choice.extensions]];
	}
	return [[NFKModelOption alloc] initWithKey:option.key kind:option.kind title:option.title help:option.help
								  defaultValue:option.defaultValue minimum:option.minimum maximum:option.maximum
										  step:option.step choices:choices stage:option.stage group:option.group
								  requirements:option.requirements propertyKeyPath:option.propertyKeyPath
							memoryBytesPerUnit:option.memoryBytesPerUnit extensions:option.extensions];
}

- (NFKModelVariant *)variant:(NFKModelVariant *)variant
			   withFiguresAt:(NSDictionary<NSString *, id> *)defaults
					 catalog:(NFKModelCatalog *)catalog
{
	if (variant.memoryFacts == nil) {
		return variant;
	}
	NSMutableDictionary<NSString *, id> *options = [defaults mutableCopy];
	options[NFKModelLoadOptionVariant] = variant.identifier;
	NFKModelMemoryEstimate *estimate = [self estimateForResolvedOptions:options memoryBudget:nil catalog:catalog];
	return [[NFKModelVariant alloc] initWithIdentifier:variant.identifier displayName:variant.displayName
											 isDefault:variant.isDefault registryName:variant.registryName source:variant.source
									 downloadByteCount:variant.downloadByteCount
									 residentByteCount:variant.residentByteCount ?: @(estimate.residentByteCount)
										 peakByteCount:variant.peakByteCount ?: @(estimate.peakByteCount)
										parameterCount:variant.parameterCount precision:variant.precision
								minimumMemoryByteCount:variant.minimumMemoryByteCount ?: @(estimate.minimumWorkingSetByteCount)
											   license:variant.license memoryFacts:variant.memoryFacts
											extensions:variant.extensions];
}

@end
