//
//  NFKModelQuality.m
//  InferKit
//

#import "NFKModelQuality.h"
#import "NFKModelDescriptorPrivate.h"

NSString * const NFKModelBenchmarkMMLUPro				= @"mmlu-pro";
NSString * const NFKModelBenchmarkLiveCodeBench			= @"livecodebench";
NSString * const NFKModelBenchmarkOpenASRLeaderboard	= @"open-asr-leaderboard";
NSString * const NFKModelBenchmarkFLORES200				= @"flores-200";
NSString * const NFKModelBenchmarkGenEval				= @"geneval";
NSString * const NFKModelBenchmarkTTSArena				= @"tts-arena";

#pragma mark - NFKModelQualityScore

static NSArray<NSString *> *NFKModelQualityScoreFields(void)
{
	return @[ @"task", @"benchmark", @"snapshot", @"metric", @"value", @"higherIsBetter", @"variant", @"source",
			  @"retrieved", @"note" ];
}

@implementation NFKModelQualityScore

- (instancetype)initWithTask:(NSString *)task
				   benchmark:(NSString *)benchmark
					snapshot:(NSString *)snapshot
					  metric:(NSString *)metric
					   value:(double)value
			  higherIsBetter:(BOOL)higherIsBetter
		   variantIdentifier:(nullable NSString *)variantIdentifier
				   sourceURL:(NSURL *)sourceURL
			   retrievedDate:(nullable NSString *)retrievedDate
						note:(nullable NSString *)note
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_task = [task copy];
		_benchmark = [benchmark copy];
		_snapshot = [snapshot copy];
		_metric = [metric copy];
		_value = value;
		_higherIsBetter = higherIsBetter;
		_variantIdentifier = [variantIdentifier copy];
		_sourceURL = [sourceURL copy];
		_retrievedDate = [retrievedDate copy];
		_note = [note copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelQualityScoreFields());
	}
	return self;
}

- (NSString *)benchmarkIdentifier
{
	return [NSString stringWithFormat:@"%@@%@", self.benchmark, self.snapshot];
}

- (BOOL)isBetterThanScore:(NFKModelQualityScore *)other
{
	if (![self.benchmarkIdentifier isEqualToString:other.benchmarkIdentifier]) {
		return NO;
	}
	return self.higherIsBetter ? self.value > other.value : self.value < other.value;
}

+ (nullable instancetype)scoreWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self scoreFromObject:dictionary path:@"qualityScore" error:error];
}

+ (nullable instancetype)scoreFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *task = nil, *benchmark = nil, *snapshot = nil, *metric = nil, *variant = nil, *retrieved = nil;
	NSString *note = nil;
	NSNumber *value = nil;
	NSURL *source = nil;
	BOOL higherIsBetter = YES;
	if (reader == nil
		|| ![reader readString:@"task" required:YES into:&task error:error]
		|| ![reader readString:@"benchmark" required:YES into:&benchmark error:error]
		|| ![reader readString:@"snapshot" required:YES into:&snapshot error:error]
		|| ![reader readString:@"metric" required:YES into:&metric error:error]
		|| ![reader readNumber:@"value" into:&value error:error]
		|| ![reader readString:@"variant" required:NO into:&variant error:error]
		|| ![reader readURL:@"source" into:&source error:error]
		|| ![reader readString:@"retrieved" required:NO into:&retrieved error:error]
		|| ![reader readString:@"note" required:NO into:&note error:error]) {
		return nil;
	}
	if (reader.dictionary[@"higherIsBetter"] != nil && ![reader readBoolean:@"higherIsBetter" into:&higherIsBetter error:error]) {
		return nil;
	}
	if (value == nil) {
		[reader failField:@"value" reason:@"is missing" error:error];
		return nil;
	}
	if (source == nil) {
		[reader failField:@"source" reason:@"is missing" error:error];
		return nil;
	}
	return [[self alloc] initWithTask:task benchmark:benchmark snapshot:snapshot metric:metric value:value.doubleValue
					   higherIsBetter:higherIsBetter variantIdentifier:variant sourceURL:source retrievedDate:retrieved
								 note:note extensions:[reader extensionsWithoutFields:NFKModelQualityScoreFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"task"] = self.task;
	dictionary[@"benchmark"] = self.benchmark;
	dictionary[@"snapshot"] = self.snapshot;
	dictionary[@"metric"] = self.metric;
	dictionary[@"value"] = @(self.value);
	dictionary[@"higherIsBetter"] = @(self.higherIsBetter);
	dictionary[@"variant"] = self.variantIdentifier;
	dictionary[@"source"] = self.sourceURL.absoluteString;
	dictionary[@"retrieved"] = self.retrievedDate;
	dictionary[@"note"] = self.note;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelQualityScore.class);
}

- (NSUInteger)hash
{
	return self.benchmarkIdentifier.hash ^ self.task.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ %g %@>", NSStringFromClass(self.class), self.task,
			self.benchmarkIdentifier, self.value, self.metric];
}

@end

#pragma mark - NFKModelQualityRanking

@interface NFKModelQualityRanking ()
- (instancetype)initWithModelIdentifier:(NSString *)modelIdentifier
								   task:(NSString *)task
								   rank:(NSInteger)rank
							rankedCount:(NSInteger)rankedCount
								  score:(NFKModelQualityScore *)score;
@end

@implementation NFKModelQualityRanking

- (instancetype)initWithModelIdentifier:(NSString *)modelIdentifier
								   task:(NSString *)task
								   rank:(NSInteger)rank
							rankedCount:(NSInteger)rankedCount
								  score:(NFKModelQualityScore *)score
{
	self = [super init];
	if (self != nil) {
		_modelIdentifier = [modelIdentifier copy];
		_task = [task copy];
		_rank = rank;
		_rankedCount = rankedCount;
		_score = [score copy];
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %ld/%ld %@>", NSStringFromClass(self.class), self.modelIdentifier,
			(long)self.rank, (long)self.rankedCount, self.score];
}

@end

#pragma mark - NFKModelCatalog (NFKModelQuality)

@implementation NFKModelCatalog (NFKModelQuality)

+ (nullable NSString *)referenceBenchmarkForTask:(NSString *)task
{
	static NSDictionary<NSString *, NSString *> *references;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		references = @{ NFKModelTaskTextGeneration: NFKModelBenchmarkMMLUPro,
						NFKModelTaskCodeGeneration: NFKModelBenchmarkLiveCodeBench,
						NFKModelTaskTranscription: NFKModelBenchmarkOpenASRLeaderboard,
						NFKModelTaskTranslation: NFKModelBenchmarkFLORES200,
						NFKModelTaskTextToImage: NFKModelBenchmarkGenEval,
						NFKModelTaskSpeechSynthesis: NFKModelBenchmarkTTSArena };
	});
	return references[task];
}

- (NSArray<NFKModelQualityRanking *> *)qualityRankingsForTask:(NSString *)task
{
	NSString *benchmark = [NFKModelCatalog referenceBenchmarkForTask:task];
	if (benchmark == nil) {
		return @[];
	}
	NSCountedSet<NSString *> *snapshots = [[NSCountedSet alloc] init];
	NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, NFKModelQualityScore *> *> *bySnapshot =
		[NSMutableDictionary dictionary];
	for (NFKModelDescriptor *descriptor in self.visibleDescriptors) {
		if (![descriptor.tasks containsObject:task]) {
			continue;
		}
		for (NFKModelQualityScore *score in [descriptor qualityScoresForVariant:nil]) {
			if (![score.task isEqualToString:task] || ![score.benchmark isEqualToString:benchmark]) {
				continue;
			}
			[snapshots addObject:score.snapshot];
			NSMutableDictionary<NSString *, NFKModelQualityScore *> *entries = bySnapshot[score.snapshot];
			if (entries == nil) {
				entries = [NSMutableDictionary dictionary];
				bySnapshot[score.snapshot] = entries;
			}
			entries[descriptor.identifier] = score;
		}
	}
	NSString *snapshot = [self rankedSnapshotAmong:snapshots];
	if (snapshot == nil) {
		return @[];
	}
	return [self rankingsOfScores:bySnapshot[snapshot] task:task];
}

- (nullable NFKModelQualityRanking *)qualityRankingForModelIdentifier:(NSString *)identifier task:(NSString *)task
{
	for (NFKModelQualityRanking *ranking in [self qualityRankingsForTask:task]) {
		if ([ranking.modelIdentifier isEqualToString:identifier]) {
			return ranking;
		}
	}
	return nil;
}

// The snapshot the most scores report, the later one on a tie.
- (nullable NSString *)rankedSnapshotAmong:(NSCountedSet<NSString *> *)snapshots
{
	NSString *chosen = nil;
	NSUInteger chosenCount = 0;
	for (NSString *snapshot in snapshots) {
		NSUInteger count = [snapshots countForObject:snapshot];
		BOOL later = chosen != nil && [snapshot compare:chosen options:NSNumericSearch] == NSOrderedDescending;
		if (count > chosenCount || (count == chosenCount && later)) {
			chosen = snapshot;
			chosenCount = count;
		}
	}
	return chosen;
}

- (NSArray<NFKModelQualityRanking *> *)rankingsOfScores:(NSDictionary<NSString *, NFKModelQualityScore *> *)scores
												   task:(NSString *)task
{
	NSArray<NSString *> *ordered = [scores.allKeys sortedArrayUsingComparator:^NSComparisonResult(NSString *a, NSString *b) {
		NFKModelQualityScore *first = scores[a], *second = scores[b];
		if ([first isBetterThanScore:second]) {
			return NSOrderedAscending;
		}
		if ([second isBetterThanScore:first]) {
			return NSOrderedDescending;
		}
		return [a compare:b];
	}];
	NSMutableArray<NFKModelQualityRanking *> *rankings = [NSMutableArray arrayWithCapacity:ordered.count];
	NSInteger rank = 0;
	NFKModelQualityScore *previous = nil;
	for (NSUInteger index = 0; index < ordered.count; index += 1) {
		NFKModelQualityScore *score = scores[ordered[index]];
		if (previous == nil || [previous isBetterThanScore:score]) {
			rank = (NSInteger)index + 1;
		}
		previous = score;
		[rankings addObject:[[NFKModelQualityRanking alloc] initWithModelIdentifier:ordered[index] task:task rank:rank
																		rankedCount:(NSInteger)ordered.count
																			  score:score]];
	}
	return [rankings copy];
}

@end
