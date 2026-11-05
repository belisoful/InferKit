//
//  NFKModelAdapter.m
//  InferKit
//

#import "NFKModelAdapter.h"
#import "NFKModelAdapterPrivate.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelVocabulary.h"
#import "NFKHFHubInternal.h"
#import "NFKDownload.h"
#import "NFKDownloadPlan.h"

NSErrorDomain const NFKModelAdapterErrorDomain = @"NFKModelAdapterErrorDomain";

static NSString * const NFKModelAdapterConfigurationName = @"adapter_config.json";
static NSString * const NFKModelAdapterMetadataKey = @"__metadata__";
static const NSUInteger NFKModelAdapterTriggerTagCount = 5;

#pragma mark - Reading

static NSError *NFKModelAdapterMakeError(NFKModelAdapterError code, NSString *reason, NSError * _Nullable underlying)
{
	NSMutableDictionary<NSString *, id> *userInfo = [NSMutableDictionary dictionaryWithObject:reason
																					   forKey:NSLocalizedDescriptionKey];
	userInfo[NSUnderlyingErrorKey] = underlying;
	return [NSError errorWithDomain:NFKModelAdapterErrorDomain code:code userInfo:userInfo];
}

BOOL NFKModelAdapterIsPicklePath(NSString *path)
{
	return [@[ @"bin", @"pt", @"pth", @"ckpt", @"pkl" ] containsObject:path.pathExtension.lowercaseString];
}

static BOOL NFKModelAdapterHasPrefix(NSString *key, NSArray<NSString *> *prefixes)
{
	for (NSString *prefix in prefixes) {
		if ([key hasPrefix:prefix]) {
			return YES;
		}
	}
	return NO;
}

static BOOL NFKModelAdapterHasSuffix(NSString *key, NSArray<NSString *> *suffixes)
{
	for (NSString *suffix in suffixes) {
		if ([key hasSuffix:suffix]) {
			return YES;
		}
	}
	return NO;
}

// The header's tensor entries, without the metadata.
static NSDictionary<NSString *, NSDictionary *> *NFKModelAdapterTensors(NSDictionary<NSString *, id> *header)
{
	NSMutableDictionary<NSString *, NSDictionary *> *tensors = [NSMutableDictionary dictionary];
	[header enumerateKeysAndObjectsUsingBlock:^(NSString *name, id entry, BOOL *stop) {
		if (![name isEqualToString:NFKModelAdapterMetadataKey] && [entry isKindOfClass:NSDictionary.class]) {
			tensors[name] = entry;
		}
	}];
	return tensors;
}

static NSArray<NSNumber *> *NFKModelAdapterShape(NSDictionary *tensor)
{
	id shape = tensor[@"shape"];
	if (![shape isKindOfClass:NSArray.class]) {
		return @[];
	}
	for (id dimension in shape) {
		if (![dimension isKindOfClass:NSNumber.class]) {
			return @[];
		}
	}
	return shape;
}

static NSInteger NFKModelAdapterElementCount(NSDictionary *tensor)
{
	NSInteger count = 1;
	for (NSNumber *dimension in NFKModelAdapterShape(tensor)) {
		count *= MAX(0, dimension.integerValue);
	}
	return count;
}

// The bytes a tensor stores: its data offsets' span, else its elements at its dtype's width.
static NSInteger NFKModelAdapterStoredByteCount(NSDictionary *tensor)
{
	id offsets = tensor[@"data_offsets"];
	if ([offsets isKindOfClass:NSArray.class] && [offsets count] == 2 && [[offsets firstObject] isKindOfClass:NSNumber.class]
		&& [[offsets lastObject] isKindOfClass:NSNumber.class]) {
		return MAX(0, [[offsets lastObject] integerValue] - [[offsets firstObject] integerValue]);
	}
	NSDictionary<NSString *, NSNumber *> *widths = @{ @"F64": @8, @"I64": @8, @"U64": @8, @"F32": @4, @"I32": @4,
													  @"U32": @4, @"F16": @2, @"BF16": @2, @"I16": @2, @"U16": @2 };
	id dtype = tensor[@"dtype"];
	NSNumber *width = [dtype isKindOfClass:NSString.class] ? widths[dtype] : nil;
	return NFKModelAdapterElementCount(tensor) * (width != nil ? width.integerValue : 1);
}

static NSDictionary<NSString *, NSString *> *NFKModelAdapterMetadata(NSDictionary<NSString *, id> *header)
{
	id metadata = header[NFKModelAdapterMetadataKey];
	if (![metadata isKindOfClass:NSDictionary.class]) {
		return @{};
	}
	NSMutableDictionary<NSString *, NSString *> *strings = [NSMutableDictionary dictionary];
	[(NSDictionary *)metadata enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL *stop) {
		if ([key isKindOfClass:NSString.class] && [value isKindOfClass:NSString.class]) {
			strings[key] = value;
		}
	}];
	return strings;
}

static NSDictionary<NSString *, id> * _Nullable NFKModelAdapterConfiguration(NSData * _Nullable data)
{
	id configuration = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	return [configuration isKindOfClass:NSDictionary.class] ? configuration : nil;
}

// The header of a local safetensors file, or nil with an NFKModelAdapterErrorUnreadable error.
static NSDictionary<NSString *, id> * _Nullable NFKModelAdapterReadHeader(NSURL *url, NSError **error)
{
	NSString *name = url.lastPathComponent;
	NSError *readError = nil;
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:nil];
	NSDictionary<NSString *, id> *header = [hub safetensorsHeaderOfFileAtURL:url subject:name error:&readError];
	if (header != nil) {
		return header;
	}
	if (error != NULL) {
		NSString *reason = nil;
		if (![NSFileManager.defaultManager fileExistsAtPath:url.path]) {
			reason = [NSString stringWithFormat:@"%@ does not exist.", name];
		} else if ([readError.domain isEqualToString:NFKDownloadErrorDomain]) {
			reason = [NSString stringWithFormat:@"%@ is not a safetensors file.", name];
		} else {
			reason = [NSString stringWithFormat:@"%@ cannot be read: %@", name, readError.localizedDescription];
		}
		*error = NFKModelAdapterMakeError(NFKModelAdapterErrorUnreadable, reason, readError);
	}
	return nil;
}

BOOL NFKModelAdapterTensorTotals(NSString *path, NSInteger *elementCount, NSInteger *byteCount, NSError **error)
{
	static NSCache<NSString *, NSArray<NSNumber *> *> *totalsByFile;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		totalsByFile = [[NSCache alloc] init];
	});
	NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:path error:NULL];
	NSString *cacheKey = attributes != nil
		? [NSString stringWithFormat:@"%@|%llu|%f", path, attributes.fileSize,
		   attributes.fileModificationDate.timeIntervalSinceReferenceDate]
		: nil;
	NSArray<NSNumber *> *totals = cacheKey != nil ? [totalsByFile objectForKey:cacheKey] : nil;
	if (totals == nil) {
		NSDictionary<NSString *, id> *header = NFKModelAdapterReadHeader([NSURL fileURLWithPath:path], error);
		if (header == nil) {
			return NO;
		}
		NSInteger elements = 0, bytes = 0;
		for (NSDictionary *tensor in NFKModelAdapterTensors(header).allValues) {
			elements += NFKModelAdapterElementCount(tensor);
			bytes += NFKModelAdapterStoredByteCount(tensor);
		}
		totals = @[ @(elements), @(bytes) ];
		if (cacheKey != nil) {
			[totalsByFile setObject:totals forKey:cacheKey];
		}
	}
	*elementCount = totals.firstObject.integerValue;
	*byteCount = totals.lastObject.integerValue;
	return YES;
}

#pragma mark - Vocabulary

static NSArray<NSString *> *NFKModelAdapterKohyaPrefixes(void)
{
	return @[ @"lora_unet_", @"lora_te_", @"lora_te1_", @"lora_te2_" ];
}

static NSArray<NSString *> *NFKModelAdapterPEFTPrefixes(void)
{
	return @[ @"unet.", @"text_encoder.", @"text_encoder_2.", @"transformer.", @"diffusion_model.", @"base_model.model." ];
}

// The down (A) halves of the low-rank pairs, whose first dimension is the rank.
static NSArray<NSString *> *NFKModelAdapterDownSuffixes(void)
{
	return @[ @".lora_down.weight", @".lora_A.weight", @".lora.down.weight" ];
}

static NSString *NFKModelAdapterBaseName(NSString *base)
{
	NSDictionary<NSString *, NSString *> *names = @{
		NFKModelAdapterBaseStableDiffusion1: @"Stable Diffusion 1", NFKModelAdapterBaseStableDiffusion2: @"Stable Diffusion 2",
		NFKModelAdapterBaseStableDiffusionXL: @"SDXL", NFKModelAdapterBaseFlux1: @"FLUX.1",
		NFKModelAdapterBaseStableDiffusion3: @"Stable Diffusion 3",
	};
	return names[base] ?: base;
}

static NSString *NFKModelAdapterLayoutName(NSString *layout)
{
	NSDictionary<NSString *, NSString *> *names = @{ NFKModelAdapterLayoutKohya: @"kohya", NFKModelAdapterLayoutPEFT: @"PEFT",
													 NFKModelAdapterLayoutMLXLM: @"mlx-lm" };
	return names[layout] ?: layout;
}

static NSString * _Nullable NFKModelAdapterLayoutOfKey(NSString *key)
{
	if (NFKModelAdapterHasPrefix(key, NFKModelAdapterKohyaPrefixes())
		&& NFKModelAdapterHasSuffix(key, @[ @".lora_down.weight", @".lora_up.weight", @".alpha" ])) {
		return NFKModelAdapterLayoutKohya;
	}
	if (NFKModelAdapterHasPrefix(key, NFKModelAdapterPEFTPrefixes())
		&& NFKModelAdapterHasSuffix(key, @[ @".lora_A.weight", @".lora_B.weight", @".lora.down.weight", @".lora.up.weight" ])) {
		return NFKModelAdapterLayoutPEFT;
	}
	if (NFKModelAdapterHasSuffix(key, @[ @".lora_a", @".lora_b" ])) {
		return NFKModelAdapterLayoutMLXLM;
	}
	return nil;
}

// kohya spells a FLUX.1 or Stable Diffusion 3 transformer under lora_unet_.
static NSString * _Nullable NFKModelAdapterTargetOfKey(NSString *key, NSString *layout)
{
	if ([layout isEqualToString:NFKModelAdapterLayoutMLXLM]) {
		return NFKModelAdapterTargetDecoder;
	}
	if (NFKModelAdapterHasPrefix(key, @[ @"lora_unet_double_blocks_", @"lora_unet_single_blocks_", @"lora_unet_joint_blocks_",
										 @"transformer.", @"diffusion_model." ])) {
		return NFKModelAdapterTargetTransformer;
	}
	if (NFKModelAdapterHasPrefix(key, @[ @"lora_unet_", @"unet." ])) {
		return NFKModelAdapterTargetUNet;
	}
	if (NFKModelAdapterHasPrefix(key, @[ @"lora_te2_", @"text_encoder_2." ])) {
		return NFKModelAdapterTargetTextEncoder2;
	}
	if (NFKModelAdapterHasPrefix(key, @[ @"lora_te_", @"lora_te1_", @"text_encoder." ])) {
		return NFKModelAdapterTargetTextEncoder;
	}
	if ([key hasPrefix:@"base_model.model."]) {
		return NFKModelAdapterTargetDecoder;
	}
	return nil;
}

// The Stable Diffusion base whose conditioning width is width: CLIP-L 768, OpenCLIP-H 1024, SDXL's
// concatenated encoders 2048.
static NSString * _Nullable NFKModelAdapterBaseOfCrossAttentionWidth(NSInteger width)
{
	NSDictionary<NSNumber *, NSString *> *bases = @{ @768: NFKModelAdapterBaseStableDiffusion1,
													 @1024: NFKModelAdapterBaseStableDiffusion2,
													 @2048: NFKModelAdapterBaseStableDiffusionXL };
	return bases[@(width)];
}

#pragma mark - NFKModelAdapterReport

@interface NFKModelAdapterReport ()
@property (nonatomic, readwrite, copy) NSString *fileName;
@property (nonatomic, readwrite, copy, nullable) NSString *layout;
@property (nonatomic, readwrite, copy, nullable) NSString *base;
@property (nonatomic, readwrite, copy, nullable) NSString *baseModelName;
@property (nonatomic, readwrite) NSInteger rank;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *targets;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *triggerWords;
@property (nonatomic, readwrite) NSInteger tensorByteCount;
@property (nonatomic, readwrite) long long fileByteCount;
@property (nonatomic, readwrite, copy, nullable) NSString *sha256;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, NSString *> *metadata;
@property (nonatomic, readwrite) NFKModelAdapterFit fit;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *fittingModelIdentifiers;
@property (nonatomic, readwrite, copy, nullable) NSError *error;
- (instancetype)initWithFileName:(NSString *)fileName NS_DESIGNATED_INITIALIZER;
+ (instancetype)pickleReportWithFileName:(NSString *)fileName reason:(NSString *)reason;
@end

@implementation NFKModelAdapterReport

- (instancetype)initWithFileName:(NSString *)fileName
{
	self = [super init];
	if (self != nil) {
		_fileName = [fileName copy];
		_targets = @[];
		_triggerWords = @[];
		_metadata = @{};
		_fittingModelIdentifiers = @[];
		_fit = NFKModelAdapterFitUnsupported;
	}
	return self;
}

+ (instancetype)pickleReportWithFileName:(NSString *)fileName reason:(NSString *)reason
{
	NFKModelAdapterReport *report = [[self alloc] initWithFileName:fileName];
	report.error = NFKModelAdapterMakeError(NFKModelAdapterErrorPickle, reason, nil);
	return report;
}

- (NSString *)fitName
{
	switch (self.fit) {
		case NFKModelAdapterFitFits:
			return @"fits";
		case NFKModelAdapterFitFitsOtherModels:
			return @"fits-other-models";
		case NFKModelAdapterFitUnsupported:
			return @"unsupported";
	}
	return @"unsupported";
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"fileName"] = self.fileName;
	dictionary[@"layout"] = self.layout;
	dictionary[@"base"] = self.base;
	dictionary[@"baseModelName"] = self.baseModelName;
	dictionary[@"rank"] = @(self.rank);
	dictionary[@"targets"] = self.targets;
	dictionary[@"triggerWords"] = self.triggerWords;
	dictionary[@"tensorByteCount"] = @(self.tensorByteCount);
	dictionary[@"fileByteCount"] = @(self.fileByteCount);
	dictionary[@"sha256"] = self.sha256;
	dictionary[@"metadata"] = self.metadata;
	dictionary[@"fit"] = self.fitName;
	dictionary[@"fittingModelIdentifiers"] = self.fittingModelIdentifiers;
	if (self.error != nil) {
		dictionary[@"error"] = @{ @"domain": self.error.domain, @"code": @(self.error.code),
								  @"message": self.error.localizedDescription };
	}
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelAdapterReport.class);
}

- (NSUInteger)hash
{
	return self.fileName.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ %@ %@>", NSStringFromClass(self.class), self.fileName,
			self.layout ?: @"-", self.base ?: @"-", self.fitName];
}

@end

#pragma mark - Classification

// Reads what a safetensors header tells about its adapter. The fit is left for the catalog.
@interface NFKModelAdapterClassifier : NSObject
@property (nonatomic, copy) NSString *fileName;
@property (nonatomic, copy) NSDictionary<NSString *, NSDictionary *> *tensors;
@property (nonatomic, copy) NSArray<NSString *> *names;
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *metadata;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, id> *configuration;
@end

@implementation NFKModelAdapterClassifier

- (instancetype)initWithFileName:(NSString *)fileName
						  header:(NSDictionary<NSString *, id> *)header
				   configuration:(nullable NSDictionary<NSString *, id> *)configuration
{
	self = [super init];
	if (self != nil) {
		_fileName = [fileName copy];
		_tensors = NFKModelAdapterTensors(header);
		_names = [_tensors.allKeys sortedArrayUsingSelector:@selector(compare:)];
		_metadata = NFKModelAdapterMetadata(header);
		_configuration = [configuration copy];
	}
	return self;
}

- (NFKModelAdapterReport *)report
{
	NFKModelAdapterReport *report = [[NFKModelAdapterReport alloc] initWithFileName:self.fileName];
	report.metadata = self.metadata;
	report.tensorByteCount = [self tensorByteCount];
	report.triggerWords = [self triggerWords];
	id baseModelName = self.configuration[@"base_model_name_or_path"];
	report.baseModelName = [baseModelName isKindOfClass:NSString.class] && [baseModelName length] > 0 ? baseModelName : nil;
	NSString *unread = [self unreadLayoutDescription];
	if (unread != nil) {
		report.error = NFKModelAdapterMakeError(NFKModelAdapterErrorUnknownLayout,
			[NSString stringWithFormat:@"%@ holds %@, a layout InferKit does not read.", self.fileName, unread], nil);
		return report;
	}
	NSString *layout = [self layout];
	if (layout == nil) {
		NSString *reason = self.names.count == 0
			? [NSString stringWithFormat:@"%@ holds no tensors.", self.fileName]
			: [NSString stringWithFormat:@"%@ holds no LoRA tensors in a layout InferKit reads (kohya, PEFT, or mlx-lm); "
			   "its first tensor is %@.", self.fileName, self.names.firstObject];
		report.error = NFKModelAdapterMakeError(NFKModelAdapterErrorUnknownLayout, reason, nil);
		return report;
	}
	NSArray<NSString *> *keys = [self keysOfLayout:layout];
	report.layout = layout;
	report.targets = [self targetsOfKeys:keys layout:layout];
	NSInteger rank = [self rankOfKeys:keys];
	report.rank = rank > 0 ? rank : [self configuredRank];
	report.base = [self baseOfKeys:keys targets:report.targets baseModelName:report.baseModelName];
	if (report.base == nil) {
		report.error = NFKModelAdapterMakeError(NFKModelAdapterErrorUnknownBase,
			[NSString stringWithFormat:@"%@ holds a %@ adapter, and neither its tensors nor its metadata tell the model it "
			 "adapts.", self.fileName, NFKModelAdapterLayoutName(layout)], nil);
	}
	return report;
}

- (NSInteger)tensorByteCount
{
	NSInteger total = 0;
	for (NSDictionary *tensor in self.tensors.allValues) {
		total += NFKModelAdapterStoredByteCount(tensor);
	}
	return total;
}

// The adapter kind InferKit does not read, named with the tensors that mark it, or nil.
- (nullable NSString *)unreadLayoutDescription
{
	NSArray<NSArray<NSString *> *> *kinds = @[ @[ @"hada_", @"a LoHa adapter (hada_ tensors)" ],
												@[ @"lokr_", @"a LoKr adapter (lokr_ tensors)" ],
												@[ @"dora_scale", @"a DoRA adapter (dora_scale tensors)" ],
												@[ @"lora_magnitude_vector", @"a DoRA adapter (lora_magnitude_vector tensors)" ] ];
	for (NSArray<NSString *> *kind in kinds) {
		for (NSString *name in self.names) {
			if ([name containsString:kind.firstObject]) {
				return kind.lastObject;
			}
		}
	}
	return nil;
}

// The layout most of the tensor names follow; nil when none follows a layout InferKit reads.
- (nullable NSString *)layout
{
	NSCountedSet<NSString *> *layouts = [NSCountedSet set];
	for (NSString *name in self.names) {
		NSString *layout = NFKModelAdapterLayoutOfKey(name);
		if (layout != nil) {
			[layouts addObject:layout];
		}
	}
	NSString *chosen = nil;
	for (NSString *layout in @[ NFKModelAdapterLayoutKohya, NFKModelAdapterLayoutPEFT, NFKModelAdapterLayoutMLXLM ]) {
		if ([layouts countForObject:layout] > (chosen != nil ? [layouts countForObject:chosen] : 0)) {
			chosen = layout;
		}
	}
	return chosen;
}

- (NSArray<NSString *> *)keysOfLayout:(NSString *)layout
{
	NSMutableArray<NSString *> *keys = [NSMutableArray array];
	for (NSString *name in self.names) {
		if ([NFKModelAdapterLayoutOfKey(name) isEqualToString:layout]) {
			[keys addObject:name];
		}
	}
	return keys;
}

- (NSArray<NSString *> *)targetsOfKeys:(NSArray<NSString *> *)keys layout:(NSString *)layout
{
	NSMutableSet<NSString *> *found = [NSMutableSet set];
	for (NSString *key in keys) {
		NSString *target = NFKModelAdapterTargetOfKey(key, layout);
		if (target != nil) {
			[found addObject:target];
		}
	}
	NSMutableArray<NSString *> *targets = [NSMutableArray array];
	for (NSString *target in @[ NFKModelAdapterTargetUNet, NFKModelAdapterTargetTextEncoder, NFKModelAdapterTargetTextEncoder2,
								NFKModelAdapterTargetTransformer, NFKModelAdapterTargetDecoder ]) {
		if ([found containsObject:target]) {
			[targets addObject:target];
		}
	}
	return targets;
}

// The most frequent rank, the larger on a tie: a down half's first dimension, an mlx-lm lora_a's second.
- (NSInteger)rankOfKeys:(NSArray<NSString *> *)keys
{
	NSCountedSet<NSNumber *> *ranks = [NSCountedSet set];
	for (NSString *key in keys) {
		NSArray<NSNumber *> *shape = NFKModelAdapterShape(self.tensors[key]);
		if (NFKModelAdapterHasSuffix(key, NFKModelAdapterDownSuffixes()) && shape.count >= 1) {
			[ranks addObject:shape.firstObject];
		} else if ([key hasSuffix:@".lora_a"] && shape.count >= 2) {
			[ranks addObject:shape[1]];
		}
	}
	NSNumber *chosen = nil;
	for (NSNumber *rank in ranks) {
		NSUInteger count = [ranks countForObject:rank];
		NSUInteger chosenCount = chosen != nil ? [ranks countForObject:chosen] : 0;
		if (count > chosenCount || (count == chosenCount && [rank compare:chosen] == NSOrderedDescending)) {
			chosen = rank;
		}
	}
	return chosen.integerValue;
}

- (NSInteger)configuredRank
{
	id rank = self.configuration[@"r"];
	return [rank isKindOfClass:NSNumber.class] ? [rank integerValue] : 0;
}

- (nullable NSString *)baseOfKeys:(NSArray<NSString *> *)keys
						  targets:(NSArray<NSString *> *)targets
					baseModelName:(nullable NSString *)baseModelName
{
	NSString *metadataBase = [self baseFromMetadata];
	BOOL decoderOnly = targets.count > 0 && [[NSSet setWithArray:targets] isEqualToSet:[NSSet setWithObject:NFKModelAdapterTargetDecoder]];
	if (decoderOnly) {
		return metadataBase ?: baseModelName;
	}
	return [self baseFromKeys:keys targets:targets] ?: metadataBase;
}

- (BOOL)anyKeyOf:(NSArray<NSString *> *)keys contains:(NSArray<NSString *> *)markers
{
	for (NSString *key in keys) {
		for (NSString *marker in markers) {
			if ([key containsString:marker]) {
				return YES;
			}
		}
	}
	return NO;
}

// The base the tensor names and shapes tell. Shape evidence outranks the metadata, which a training
// script can fill wrongly.
- (nullable NSString *)baseFromKeys:(NSArray<NSString *> *)keys targets:(NSArray<NSString *> *)targets
{
	if ([self anyKeyOf:keys contains:@[ @"lora_unet_double_blocks_", @"lora_unet_single_blocks_",
										@"transformer.single_transformer_blocks.", @"diffusion_model.double_blocks.",
										@"diffusion_model.single_blocks." ]]) {
		return NFKModelAdapterBaseFlux1;
	}
	if ([self anyKeyOf:keys contains:@[ @"joint_blocks" ]]) {
		return NFKModelAdapterBaseStableDiffusion3;
	}
	BOOL stableDiffusion = [targets containsObject:NFKModelAdapterTargetUNet]
		|| [targets containsObject:NFKModelAdapterTargetTextEncoder] || [targets containsObject:NFKModelAdapterTargetTextEncoder2];
	if (!stableDiffusion) {
		return nil;
	}
	for (NSString *key in keys) {
		BOOL crossAttentionKey = [key containsString:@"attn2_to_k.lora_down.weight"]
			|| [key containsString:@"attn2.to_k.lora_A.weight"] || [key containsString:@"attn2.to_k.lora.down.weight"];
		NSArray<NSNumber *> *shape = NFKModelAdapterShape(self.tensors[key]);
		NSString *base = crossAttentionKey && shape.count >= 2 ? NFKModelAdapterBaseOfCrossAttentionWidth(shape[1].integerValue) : nil;
		if (base != nil) {
			return base;
		}
	}
	if ([targets containsObject:NFKModelAdapterTargetTextEncoder2]) {
		return NFKModelAdapterBaseStableDiffusionXL;
	}
	return [self baseFromTextEncoderKeys:keys];
}

// The base the first text encoder's width tells: CLIP-L's 768 is Stable Diffusion 1, or SDXL when the
// file names its first encoder lora_te1_; OpenCLIP-H's 1024 is Stable Diffusion 2.
- (nullable NSString *)baseFromTextEncoderKeys:(NSArray<NSString *> *)keys
{
	BOOL namesFirstEncoder = [self anyKeyOf:keys contains:@[ @"lora_te1_" ]];
	for (NSString *key in keys) {
		BOOL textEncoderKey = NFKModelAdapterHasPrefix(key, @[ @"lora_te_", @"lora_te1_", @"text_encoder." ]);
		if (!textEncoderKey || ![key containsString:@"q_proj"] || !NFKModelAdapterHasSuffix(key, NFKModelAdapterDownSuffixes())) {
			continue;
		}
		NSArray<NSNumber *> *shape = NFKModelAdapterShape(self.tensors[key]);
		NSInteger width = shape.count >= 2 ? shape[1].integerValue : 0;
		if (width == 768) {
			return namesFirstEncoder ? NFKModelAdapterBaseStableDiffusionXL : NFKModelAdapterBaseStableDiffusion1;
		}
		if (width == 1024) {
			return NFKModelAdapterBaseStableDiffusion2;
		}
	}
	return nil;
}

// The base the kohya (ss_base_model_version) or ModelSpec (modelspec.architecture) metadata states.
- (nullable NSString *)baseFromMetadata
{
	NSString *version = self.metadata[@"ss_base_model_version"].lowercaseString;
	NSArray<NSArray<NSString *> *> *versions = @[ @[ @"sdxl", NFKModelAdapterBaseStableDiffusionXL ],
												   @[ @"sd_v2", NFKModelAdapterBaseStableDiffusion2 ],
												   @[ @"sd_2", NFKModelAdapterBaseStableDiffusion2 ],
												   @[ @"sd_v1", NFKModelAdapterBaseStableDiffusion1 ],
												   @[ @"sd_1", NFKModelAdapterBaseStableDiffusion1 ],
												   @[ @"sd3", NFKModelAdapterBaseStableDiffusion3 ],
												   @[ @"flux", NFKModelAdapterBaseFlux1 ] ];
	for (NSArray<NSString *> *entry in versions) {
		if ([version hasPrefix:entry.firstObject]) {
			return entry.lastObject;
		}
	}
	NSString *architecture = self.metadata[@"modelspec.architecture"].lowercaseString;
	NSArray<NSArray<NSString *> *> *architectures = @[ @[ @"stable-diffusion-xl", NFKModelAdapterBaseStableDiffusionXL ],
														@[ @"stable-diffusion-v2", NFKModelAdapterBaseStableDiffusion2 ],
														@[ @"stable-diffusion-v1", NFKModelAdapterBaseStableDiffusion1 ],
														@[ @"stable-diffusion-v3", NFKModelAdapterBaseStableDiffusion3 ],
														@[ @"flux-1", NFKModelAdapterBaseFlux1 ] ];
	for (NSArray<NSString *> *entry in architectures) {
		if ([architecture hasPrefix:entry.firstObject]) {
			return entry.lastObject;
		}
	}
	return nil;
}

- (NSArray<NSString *> *)triggerWords
{
	NSCharacterSet *blanks = NSCharacterSet.whitespaceAndNewlineCharacterSet;
	NSMutableArray<NSString *> *phrases = [NSMutableArray array];
	for (NSString *part in [self.metadata[@"modelspec.trigger_phrase"] componentsSeparatedByString:@","]) {
		NSString *phrase = [part stringByTrimmingCharactersInSet:blanks];
		if (phrase.length > 0) {
			[phrases addObject:phrase];
		}
	}
	if (phrases.count > 0) {
		return phrases;
	}
	return [self mostFrequentTags];
}

// The most frequent tags of ss_tag_frequency, {dataset: {tag: count}}, summed over the datasets; a tie
// goes to the tag that sorts first.
- (NSArray<NSString *> *)mostFrequentTags
{
	NSData *data = [self.metadata[@"ss_tag_frequency"] dataUsingEncoding:NSUTF8StringEncoding];
	id datasets = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	if (![datasets isKindOfClass:NSDictionary.class]) {
		return @[];
	}
	NSMutableDictionary<NSString *, NSNumber *> *counts = [NSMutableDictionary dictionary];
	NSCharacterSet *blanks = NSCharacterSet.whitespaceAndNewlineCharacterSet;
	for (id dataset in [datasets allValues]) {
		if (![dataset isKindOfClass:NSDictionary.class]) {
			continue;
		}
		[(NSDictionary *)dataset enumerateKeysAndObjectsUsingBlock:^(id tag, id count, BOOL *stop) {
			NSString *trimmed = [tag isKindOfClass:NSString.class] ? [tag stringByTrimmingCharactersInSet:blanks] : nil;
			if (trimmed.length > 0 && [count isKindOfClass:NSNumber.class]) {
				counts[trimmed] = @(counts[trimmed].doubleValue + [count doubleValue]);
			}
		}];
	}
	NSArray<NSString *> *sorted = [counts.allKeys sortedArrayUsingComparator:^NSComparisonResult(NSString *first, NSString *second) {
		NSComparisonResult byCount = [counts[second] compare:counts[first]];
		return byCount != NSOrderedSame ? byCount : [first compare:second];
	}];
	return [sorted subarrayWithRange:NSMakeRange(0, MIN(sorted.count, NFKModelAdapterTriggerTagCount))];
}

@end

#pragma mark - NFKModelCatalog (NFKModelAdapter)

@implementation NFKModelCatalog (NFKModelAdapter)

- (nullable NFKModelAdapterReport *)inspectAdapterAtURL:(NSURL *)url
									 forModelIdentifier:(nullable NSString *)identifier
												  error:(NSError **)error
{
	NSString *name = url.lastPathComponent ?: @"";
	if (!url.isFileURL) {
		if (error != NULL) {
			*error = NFKModelAdapterMakeError(NFKModelAdapterErrorUnreadable,
				[NSString stringWithFormat:@"%@ is not a file URL.", url.absoluteString], nil);
		}
		return nil;
	}
	NSDictionary<NSFileAttributeKey, id> *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:url.path error:NULL];
	if (NFKModelAdapterIsPicklePath(url.path)) {
		NFKModelAdapterReport *report = [self pickleReportForFileName:name];
		report.fileByteCount = (long long)attributes.fileSize;
		return report;
	}
	NSDictionary<NSString *, id> *header = NFKModelAdapterReadHeader(url, error);
	if (header == nil) {
		return nil;
	}
	NSURL *configurationURL = [url.URLByDeletingLastPathComponent URLByAppendingPathComponent:NFKModelAdapterConfigurationName];
	NSDictionary<NSString *, id> *configuration = NFKModelAdapterConfiguration([NSData dataWithContentsOfURL:configurationURL]);
	NFKModelAdapterClassifier *classifier = [[NFKModelAdapterClassifier alloc] initWithFileName:name header:header
																			   configuration:configuration];
	NFKModelAdapterReport *report = classifier.report;
	report.fileByteCount = (long long)attributes.fileSize;
	report.sha256 = [[NFKHFHub hubWithCacheDirectoryURL:nil] sha256OfFileAtURL:url];
	return [self report:report fittedForModelIdentifier:identifier];
}

- (void)inspectAdapterInRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							  path:(nullable NSString *)path
				forModelIdentifier:(nullable NSString *)identifier
				 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelAdapterReport * _Nullable report,
															   NSError * _Nullable error))completionHandler
{
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:NFKHFHub.defaultCacheDirectoryURL];
	[self inspectAdapterInRepository:repository revision:revision path:path forModelIdentifier:identifier hub:hub
				   completionHandler:completionHandler];
}

- (void)inspectAdapterInRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							  path:(nullable NSString *)path
				forModelIdentifier:(nullable NSString *)identifier
							   hub:(NFKHFHub *)hub
				 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelAdapterReport * _Nullable report,
															   NSError * _Nullable error))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKModelAdapterReport *report = [self inspectAdapterInRepository:repository revision:revision path:path
													  forModelIdentifier:identifier hub:hub error:&error];
		completionHandler(report, report != nil ? nil : error);
	});
}

- (nullable NFKModelAdapterReport *)inspectAdapterInRepository:(NSString *)repository
													  revision:(nullable NSString *)revision
														  path:(nullable NSString *)path
											forModelIdentifier:(nullable NSString *)identifier
														   hub:(NFKHFHub *)hub
														 error:(NSError **)error
{
	NSError *listingError = nil;
	NFKDownloadPlan *plan = [hub planDownloadOfRepo:repository revision:revision paths:path != nil ? @[ path ] : nil
											  error:&listingError];
	if (plan == nil) {
		NSString *reason = path != nil ? [NSString stringWithFormat:@"%@ in %@ cannot be listed.", path, repository]
			: [NSString stringWithFormat:@"The listing of %@ cannot be read.", repository];
		return [self failUnreadable:reason underlying:listingError error:error];
	}
	NFKDownloadPlanFile *file = nil;
	if (path != nil) {
		file = [self fileAtPath:path inPlan:plan];
	} else {
		NFKModelAdapterReport *pickled = [self pickleReportOfListing:plan.files repository:repository];
		if (pickled != nil) {
			return pickled;
		}
		file = [self adapterFileInListing:plan.files repository:repository error:error];
		if (file == nil) {
			return nil;
		}
	}
	NSString *filePath = file.path ?: path;
	if (filePath == nil) {
		return [self failUnreadable:[NSString stringWithFormat:@"The listing of %@ names no adapter file.", repository]
						 underlying:nil error:error];
	}
	if (NFKModelAdapterIsPicklePath(filePath)) {
		NFKModelAdapterReport *report = [self pickleReportForFileName:filePath.lastPathComponent];
		report.fileByteCount = MAX(0, file.byteCount);
		return report;
	}
	NSError *readError = nil;
	NSDictionary<NSString *, id> *header = [hub safetensorsHeaderOfRepo:repository revision:revision path:filePath
																  error:&readError];
	if (header == nil) {
		return [self failUnreadable:[NSString stringWithFormat:@"%@ in %@ cannot be read as a safetensors file.", filePath,
									 repository]
						 underlying:readError error:error];
	}
	NSString *folder = filePath.stringByDeletingLastPathComponent;
	NSString *configurationPath = folder.length > 0
		? [folder stringByAppendingPathComponent:NFKModelAdapterConfigurationName] : NFKModelAdapterConfigurationName;
	NSDictionary<NSString *, id> *configuration = nil;
	if (path != nil || [self fileAtPath:configurationPath inPlan:plan] != nil) {
		configuration = [self configurationInRepository:repository revision:revision path:configurationPath hub:hub];
	}
	NFKModelAdapterClassifier *classifier = [[NFKModelAdapterClassifier alloc] initWithFileName:filePath.lastPathComponent
																						 header:header
																				  configuration:configuration];
	NFKModelAdapterReport *report = classifier.report;
	report.fileByteCount = MAX(0, file.byteCount);
	report.sha256 = file.sha256.lowercaseString;
	return [self report:report fittedForModelIdentifier:identifier];
}

- (nullable NFKDownloadPlanFile *)fileAtPath:(NSString *)path inPlan:(NFKDownloadPlan *)plan
{
	for (NFKDownloadPlanFile *file in plan.files) {
		if ([file.path isEqualToString:path]) {
			return file;
		}
	}
	return nil;
}

static NSArray<NFKDownloadPlanFile *> *NFKModelAdapterFilesWhere(NSArray<NFKDownloadPlanFile *> *files,
																  BOOL (^predicate)(NSString *path))
{
	NSMutableArray<NFKDownloadPlanFile *> *matching = [NSMutableArray array];
	for (NFKDownloadPlanFile *file in files) {
		if (predicate(file.path)) {
			[matching addObject:file];
		}
	}
	[matching sortUsingComparator:^NSComparisonResult(NFKDownloadPlanFile *first, NFKDownloadPlanFile *second) {
		return [first.path compare:second.path];
	}];
	return matching;
}

static NSArray<NFKDownloadPlanFile *> *NFKModelAdapterSafetensorsFiles(NSArray<NFKDownloadPlanFile *> *files)
{
	return NFKModelAdapterFilesWhere(files, ^BOOL(NSString *path) {
		return [path.pathExtension.lowercaseString isEqualToString:@"safetensors"];
	});
}

// The listing's one safetensors file, or nil with an error for none or several.
- (nullable NFKDownloadPlanFile *)adapterFileInListing:(NSArray<NFKDownloadPlanFile *> *)files
											repository:(NSString *)repository
												 error:(NSError **)error
{
	NSArray<NFKDownloadPlanFile *> *safetensors = NFKModelAdapterSafetensorsFiles(files);
	if (safetensors.count == 1) {
		return safetensors.firstObject;
	}
	NSString *reason = safetensors.count == 0
		? [NSString stringWithFormat:@"%@ holds no safetensors file.", repository]
		: [NSString stringWithFormat:@"%@ holds several safetensors files (%@); name the adapter's path.", repository,
		   [[safetensors valueForKey:@"path"] componentsJoinedByString:@", "]];
	[self failUnreadable:reason underlying:nil error:error];
	return nil;
}

// The pickle report of a listing whose adapter files are all pickle files; nil for any other listing.
- (nullable NFKModelAdapterReport *)pickleReportOfListing:(NSArray<NFKDownloadPlanFile *> *)files
											   repository:(NSString *)repository
{
	if (NFKModelAdapterSafetensorsFiles(files).count > 0) {
		return nil;
	}
	NSArray<NFKDownloadPlanFile *> *pickles = NFKModelAdapterFilesWhere(files, ^BOOL(NSString *path) {
		return NFKModelAdapterIsPicklePath(path);
	});
	if (pickles.count == 0) {
		return nil;
	}
	NSString *reason = [NSString stringWithFormat:@"%@ holds only pickle files (%@), which InferKit does not read because "
						"loading one runs code it carries.", repository,
						[[pickles valueForKey:@"path"] componentsJoinedByString:@", "]];
	NFKModelAdapterReport *report = [NFKModelAdapterReport pickleReportWithFileName:pickles.firstObject.path.lastPathComponent
																			  reason:reason];
	report.fileByteCount = MAX(0, pickles.firstObject.byteCount);
	return report;
}

- (NFKModelAdapterReport *)pickleReportForFileName:(NSString *)fileName
{
	NSString *reason = [NSString stringWithFormat:@"%@ is a pickle file, which InferKit does not read because loading it "
						"runs code it carries. Use the adapter's safetensors file.", fileName];
	return [NFKModelAdapterReport pickleReportWithFileName:fileName reason:reason];
}

// An NFKModelAdapterErrorUnreadable error; the hub's authentication refusal passes through as the hub gave
// it, so a caller handles a gated repository as it does for a download.
- (nullable id)failUnreadable:(NSString *)reason underlying:(nullable NSError *)underlying error:(NSError **)error
{
	if (error == NULL) {
		return nil;
	}
	BOOL refused = [underlying.domain isEqualToString:NFKDownloadErrorDomain]
		&& underlying.code == NFKDownloadErrorAuthenticationRequired;
	*error = refused ? underlying : NFKModelAdapterMakeError(NFKModelAdapterErrorUnreadable, reason, underlying);
	return nil;
}

// The adapter_config.json at path: from the hub cache when it holds the file, else by one request.
- (nullable NSDictionary<NSString *, id> *)configurationInRepository:(NSString *)repository
															revision:(nullable NSString *)revision
																path:(NSString *)path
																 hub:(NFKHFHub *)hub
{
	if ([hub isCachedRepo:repository revision:revision path:path]) {
		NSURL *localURL = [hub localURLForRepo:repository revision:revision path:path];
		return NFKModelAdapterConfiguration(localURL != nil ? [NSData dataWithContentsOfURL:localURL] : nil);
	}
	NSURL *remoteURL = [hub remoteURLForRepo:repository revision:revision path:path];
	if (remoteURL == nil) {
		return nil;
	}
	NSHTTPURLResponse *response = nil;
	NSData *body = [hub fetchMetadataFromURL:remoteURL HTTPMethod:@"GET" response:&response error:NULL];
	return response.statusCode == 200 ? NFKModelAdapterConfiguration(body) : nil;
}

// The report with its fit against the catalog's descriptors. A report that already carries an error is
// unsupported.
- (NFKModelAdapterReport *)report:(NFKModelAdapterReport *)report fittedForModelIdentifier:(nullable NSString *)identifier
{
	if (report.error != nil) {
		report.fit = NFKModelAdapterFitUnsupported;
		return report;
	}
	NSMutableArray<NSString *> *fitting = [NSMutableArray array];
	for (NFKModelDescriptor *descriptor in self.descriptors) {
		if ([descriptor.adapterBases containsObject:report.base]) {
			[fitting addObject:descriptor.identifier];
		}
	}
	[fitting sortUsingSelector:@selector(compare:)];
	report.fittingModelIdentifiers = fitting;
	if (identifier != nil && [fitting containsObject:identifier]) {
		report.fit = NFKModelAdapterFitFits;
	} else if (fitting.count > 0) {
		report.fit = NFKModelAdapterFitFitsOtherModels;
	} else {
		NSString *baseName = NFKModelAdapterBaseName(report.base);
		NSString *article = [baseName hasPrefix:@"SD"] ? @"an" : @"a";
		report.fit = NFKModelAdapterFitUnsupported;
		report.error = NFKModelAdapterMakeError(NFKModelAdapterErrorNoModel,
			[NSString stringWithFormat:@"%@ adapts %@ %@ model; no model in the catalog takes %@ adapters.", report.fileName,
			 article, baseName, baseName], nil);
	}
	return report;
}

@end
