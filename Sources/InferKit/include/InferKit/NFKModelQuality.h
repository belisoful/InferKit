//
//  NFKModelQuality.h
//  InferKit
//

#ifndef NFKModelQuality_h
#define NFKModelQuality_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelCatalog.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelQuality
	@abstract   Published benchmark scores of a model's releases, and the rank a catalog computes from them.
	@discussion A descriptor carries the scores its publisher or a public leaderboard reports
				(NFKModelDescriptor.qualityScores). A catalog ranks the descriptors of a task on the task's
				reference benchmark, using each descriptor's default release. Introduced in InferKit 0.4.0.
*/

#pragma mark Benchmarks

/*! MMLU-Pro, multiple-choice questions across fourteen subjects, scored by accuracy. The reference
	benchmark of NFKModelTaskTextGeneration. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkMMLUPro;
/*! LiveCodeBench, programming problems published after a model's training cutoff, scored by pass@1. The
	reference benchmark of NFKModelTaskCodeGeneration. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkLiveCodeBench;
/*! The Hugging Face Open ASR Leaderboard, the mean word error rate over its English test sets. Lower is
	better. The reference benchmark of NFKModelTaskTranscription. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkOpenASRLeaderboard;
/*! FLORES-200, sentence translation across 200 languages, scored by chrF++. The reference benchmark of
	NFKModelTaskTranslation. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkFLORES200;
/*! GenEval, object-focused prompts checked by a detector, scored by the overall fraction correct. The
	reference benchmark of NFKModelTaskTextToImage. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkGenEval;
/*! TTS Arena, pairwise listener preference, scored by Elo rating. The reference benchmark of
	NFKModelTaskSpeechSynthesis. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelBenchmarkTTSArena;

/*!
	@class      NFKModelQualityScore
	@abstract   One published score of one release of a model on one benchmark.
	@discussion A score belongs to the release that was measured, at the precision that release ships.
				variantIdentifier names that release; nil names the descriptor's default variant. Another
				variant, a quantized one included, has only the scores that name it.

				Two scores compare when their benchmark and snapshot are equal. The snapshot names the
				benchmark's version or leaderboard date, because a benchmark's questions and its leaderboard
				change over time. Introduced in InferKit 0.4.0.
*/
@interface NFKModelQualityScore : NSObject <NSCopying>

/*! The task the benchmark measures, an NFKModelTask* string the descriptor lists. */
@property (nonatomic, readonly, copy) NSString *task;
/*! The benchmark, such as NFKModelBenchmarkMMLUPro. */
@property (nonatomic, readonly, copy) NSString *benchmark;
/*! The benchmark's version or leaderboard date, such as "v6" or "2026-10". */
@property (nonatomic, readonly, copy) NSString *snapshot;
/*! "benchmark@snapshot", the identity two comparable scores share. */
@property (nonatomic, readonly, copy) NSString *benchmarkIdentifier;
/*! The measure the value is in, for a user interface, such as "accuracy", "WER", or "Elo". */
@property (nonatomic, readonly, copy) NSString *metric;
/*! The score as published. A percentage is written as published, such as 69.6. */
@property (nonatomic, readonly) double value;
/*! YES when a larger value is the better score. */
@property (nonatomic, readonly) BOOL higherIsBetter;
/*! The variant the score measures, or nil for the descriptor's default variant. */
@property (nonatomic, readonly, copy, nullable) NSString *variantIdentifier;
/*! Where the score is published: the report, the model card, or the leaderboard. */
@property (nonatomic, readonly, copy) NSURL *sourceURL;
/*! The day the score was read from its source, "YYYY-MM-DD". */
@property (nonatomic, readonly, copy, nullable) NSString *retrievedDate;
/*! A sentence on how the score was measured, such as a setting that differs from the benchmark's own. */
@property (nonatomic, readonly, copy, nullable) NSString *note;
/*! The fields of the score's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A score with every field. */
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
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! Reads a score from its manifest dictionary. An absent higherIsBetter is YES. Returns nil and sets
	error when task, benchmark, snapshot, metric, value, or source is missing, or a field has the wrong
	type. */
+ (nullable instancetype)scoreWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The score as its manifest dictionary: task, benchmark, snapshot, metric, value, higherIsBetter,
	variant, source, retrieved, and note, with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

/*! YES when the score is better than other's. Scores of different benchmark identifiers do not compare,
	and return NO. */
- (BOOL)isBetterThanScore:(NFKModelQualityScore *)other;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelQualityRanking
	@abstract   Where one descriptor stands among a catalog's descriptors of a task.
	@discussion The score is the one the rank was computed from, so a user interface shows it and its
				source beside the rank. Introduced in InferKit 0.4.0.
*/
@interface NFKModelQualityRanking : NSObject

/*! The ranked descriptor's identifier. */
@property (nonatomic, readonly, copy) NSString *modelIdentifier;
/*! The task the ranking orders. */
@property (nonatomic, readonly, copy) NSString *task;
/*! The position, 1 for the best. Equal scores share a rank, and the next rank skips past them. */
@property (nonatomic, readonly) NSInteger rank;
/*! How many descriptors the ranking orders. */
@property (nonatomic, readonly) NSInteger rankedCount;
/*! The score the rank was computed from. */
@property (nonatomic, readonly, copy) NFKModelQualityScore *score;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@category   NFKModelCatalog (NFKModelQuality)
	@abstract   Ranks a catalog's descriptors of a task by their published scores.
	@discussion A ranking orders the visible descriptors that list the task and whose default variant has
				a score on the task's reference benchmark. Their scores can carry different snapshots; the
				ranking uses the snapshot the most of them report, the later snapshot on a tie, and leaves
				out a descriptor without a score at that snapshot. Descriptors with equal values order by
				identifier. A task without a reference benchmark has no ranking. Introduced in InferKit
				0.4.0.
*/
@interface NFKModelCatalog (NFKModelQuality)

/*! The benchmark a ranking of the task compares on, one of the NFKModelBenchmark* strings, or nil for a
	task the catalog does not rank. */
+ (nullable NSString *)referenceBenchmarkForTask:(NSString *)task;

/*! The rankings of the task's descriptors, best first. Empty for a task without a reference benchmark or
	without scored descriptors. */
- (NSArray<NFKModelQualityRanking *> *)qualityRankingsForTask:(NSString *)task;

/*! The ranking of one descriptor among the task's, or nil when the ranking leaves it out. */
- (nullable NFKModelQualityRanking *)qualityRankingForModelIdentifier:(NSString *)identifier task:(NSString *)task;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelQuality_h */
