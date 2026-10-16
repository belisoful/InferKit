//
//  NFKModelMemory.h
//  InferKit
//

#ifndef NFKModelMemory_h
#define NFKModelMemory_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKHardwareProfile;
@class NFKModelCatalog;

/*!
	@header     NFKModelMemory
	@abstract   What a model holds in memory under a combination of load options, the budget a machine
				plans against, and the constraints the budget puts on each option.
	@discussion A variant's NFKModelMemoryFacts record what its stages weigh at each precision. A
				descriptor's NFKModelMemoryModel names the options that change those figures. The
				estimate combines the two for one combination of options, and the constraints run the
				estimate over every combination to find which choices fit a budget. The residency
				planner (NFKModelResidencyPlan) decides how a model of several stages holds them. Every
				figure is in bytes. Introduced in InferKit 0.4.0.
*/

#pragma mark Bases

/*! The figures come from measured tensor bytes at the precisions the loader holds them in. Introduced
	in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryBasisMeasured;
/*! The figures count the stored bytes, because the loader's precision or the stored type is not known.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryBasisStoredDType;
/*! The figures come from the variant's residentBytes and peakBytes, without memory facts. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelMemoryBasisDeclared;
/*! The variant states no memory. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryBasisUnknown;

#pragma mark Components

/*! The weights the resident figure holds. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentWeights;
/*! The key-value cache at the resolved context length. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentKeyValueCache;
/*! The draft model's weights. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentDraft;
/*! The bytes a paged load's expert cache holds. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentExpertCache;
/*! The bytes the integer options with memoryBytesPerUnit add, other than the context length.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentLinearOptions;
/*! The bytes a load holds beyond the resident figure for a moment. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryComponentLoadOverhead;

/*! The userInfo key of an NFKModelCatalogErrorDoesNotFit error: the NFKModelMemoryEstimate that did
	not fit. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryEstimateKey;
/*! The userInfo key of an NFKModelCatalogErrorDoesNotFit error from a build: the load options that fit,
	nearest the requested ones (NFKModelMemoryConstraints.fittingOptions). Absent when nothing fits.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryFittingOptionsKey;
/*! The userInfo key of an NFKModelCatalogErrorDoesNotFit error from a build: the NFKModelMemoryEstimate of
	the fitting options. Absent when nothing fits. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelMemoryFittingEstimateKey;

#pragma mark - Budget

/*!
	@class      NFKModelMemoryBudget
	@abstract   The memory a machine plans a model against, and the rule it plans by.
	@discussion The budget is workingSetFraction (0.85) of the working set Metal recommends, and every
				placement keeps reserveByteCount (4 GiB) beside its weights for activations and request
				buffers. A budget of 0 is a machine that reports no working set, and the two questions
				asked of it get different answers:
				- holdsByteCount: (whether to hold stages resident, or take a wider precision) → NO.
				- admitsByteCount: (whether a stage may load at all) → YES.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryBudget : NSObject <NSCopying>

/*! The share of the recommended working set a placement plans against: 0.85. */
@property (class, nonatomic, readonly) double workingSetFraction;
/*! The bytes every placement keeps beside its weights: 4 GiB. */
@property (class, nonatomic, readonly) NSInteger reserveByteCount;

/*! The budget of this machine's recommended working set. */
+ (instancetype)currentBudget;
/*! The budget of a profile's recommended working set, such as a snapshot of another machine. */
+ (instancetype)budgetForHardwareProfile:(NFKHardwareProfile *)profile;
/*! The budget of a recommended working set. 0 is a machine that reports none. */
+ (instancetype)budgetWithWorkingSetByteCount:(NSInteger)workingSetByteCount;

/*! The recommended working set the budget was taken from. */
@property (nonatomic, readonly) NSInteger workingSetByteCount;
/*! workingSetFraction of the working set, rounded down, or 0 where the machine reports none. */
@property (nonatomic, readonly) NSInteger byteCount;
/*! YES when the machine reports a working set. */
@property (nonatomic, readonly, getter=isKnown) BOOL known;

/*! YES when bytes are known to fit beside the reserve. NO on a machine that reports no budget. */
- (BOOL)holdsByteCount:(NSInteger)bytes;
/*! YES when bytes may load at all: they fit beside the reserve, or the machine reports no budget. */
- (BOOL)admitsByteCount:(NSInteger)bytes;

/*! The least recommended working set whose budget holds bytes beside the reserve: the bytes plus the
	reserve, divided by workingSetFraction, rounded up. The figure a variant's minimumMemoryBytes
	states. */
+ (NSInteger)leastWorkingSetHoldingByteCount:(NSInteger)bytes;

/*! A byte count as the planner's sentences write it: "27.2 GiB", one decimal place. */
+ (NSString *)formattedByteCount:(NSInteger)bytes;

/*! Reads a budget from its dictionary. Returns nil and sets error when workingSetBytes is missing or is
	not a number. */
+ (nullable instancetype)budgetWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The budget as a dictionary: workingSetBytes, budgetBytes, reserveBytes, and workingSetFraction. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

#pragma mark - Facts

/*!
	@class      NFKModelStageFacts
	@abstract   What one stage of a variant weighs: its bytes at each precision, the quantizable elements,
				the routed experts a paged load leaves in the release, and its repeated layers.
	@discussion The dictionaries key a precision, an NFKModelPrecision* string, to bytes. The key
				NFKModelPrecisionCheckpoint holds the bytes as stored. A stage runs one after another in
				a staged model, in the order the facts list them. Introduced in InferKit 0.4.0.
*/
@interface NFKModelStageFacts : NSObject <NSCopying>

/*! The stage's identifier, such as "text_encoder" or "decoder", unique within its variant. */
@property (nonatomic, readonly, copy) NSString *identifier;
/*! The stage's name, for a user interface. The identifier when absent. */
@property (nonatomic, readonly, copy) NSString *title;
/*! The stage's bytes per precision. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSNumber *> *byteCountByPrecision;
/*! The elements a runtime weight quantization packs: the stage's eligible matrices. 0 for a stage that
	does not quantize. */
@property (nonatomic, readonly) NSInteger quantizableElementCount;
/*! The routed-expert bytes a paged load leaves in the release, per precision. Empty for a stage without
	routed experts. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSNumber *> *pageableByteCountByPrecision;
/*! The repeated layers a streamed load may leave in the release, from 0 to maximumLayerCount. 0 for a
	stage that cannot stream. */
@property (nonatomic, readonly) NSInteger layerCount;
/*!
	@abstract   The most repeated layers a stage states: 4096.
	@discussion A stage's dictionary whose layerCount is above it, negative, or fractional is malformed,
				and stageWithDictionary:error: refuses it with NFKModelCatalogErrorMalformedDescriptor. The
				initializer clamps layerCount to the range 0 to this maximum. The planner sizes one entry
				per layer, so the bound keeps a hostile manifest from driving that loop. Introduced in
				InferKit 0.4.0.
*/
@property (class, nonatomic, readonly) NSInteger maximumLayerCount;
/*! The bytes of the largest repeated layer, per precision. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSNumber *> *largestLayerByteCountByPrecision;
/*! A wider precision the stage takes where the budget holds it, such as a text encoder stored at
	bfloat16 and run at float32. nil for a stage with one precision. */
@property (nonatomic, readonly, copy, nullable) NSString *widenedPrecision;
/*! The fields of the stage's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A stage with every field. A nil title is the identifier. */
- (instancetype)initWithIdentifier:(NSString *)identifier
							 title:(nullable NSString *)title
			  byteCountByPrecision:(NSDictionary<NSString *, NSNumber *> *)byteCountByPrecision
		   quantizableElementCount:(NSInteger)quantizableElementCount
	  pageableByteCountByPrecision:(nullable NSDictionary<NSString *, NSNumber *> *)pageableByteCountByPrecision
						layerCount:(NSInteger)layerCount
  largestLayerByteCountByPrecision:(nullable NSDictionary<NSString *, NSNumber *> *)largestLayerByteCountByPrecision
				  widenedPrecision:(nullable NSString *)widenedPrecision
						extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A stage that only states its bytes per precision. */
+ (instancetype)stageWithIdentifier:(NSString *)identifier
			   byteCountByPrecision:(NSDictionary<NSString *, NSNumber *> *)byteCountByPrecision
	NS_SWIFT_NAME(init(identifier:byteCountByPrecision:));

/*! Reads a stage from its dictionary. Returns nil and sets error when id or bytesByPrecision is missing,
	a field has the wrong type, or layerCount is not a whole number from 0 to maximumLayerCount. */
+ (nullable instancetype)stageWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
	NS_SWIFT_NAME(init(dictionary:));
/*! The stage as its dictionary: id, title, bytesByPrecision, quantizableElements,
	pageableBytesByPrecision, layerCount, largestLayerBytesByPrecision, and widenedPrecision when
	present. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelMemoryFacts
	@abstract   What a variant's weights and caches weigh, from which every memory estimate is computed.
	@discussion keyValueByteCountPerToken is the key-value cache one position adds across the layers
				whose cache grows, at 16 bits per element. boundedKeyValueByteCount is the cache of the
				layers whose cache stops growing, such as sliding-window layers, at 16 bits per element.
				loadOverheadByteCount is what a load holds beyond the resident figure for a moment, 0
				where it is not measured. basis is NFKModelMemoryBasisMeasured or
				NFKModelMemoryBasisStoredDType. Activation memory, which depends on the request, is not
				stated: the budget's reserve covers it. Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryFacts : NSObject <NSCopying>

/*! The stages, in the order a staged load runs them. A model without stages has one. */
@property (nonatomic, readonly, copy) NSArray<NFKModelStageFacts *> *stages;
/*! The cache bytes one position adds at 16 bits, or nil for a model without a growing cache. */
@property (nonatomic, readonly, copy, nullable) NSNumber *keyValueByteCountPerToken;
/*! The cache bytes of the layers whose cache stops growing, at 16 bits, or nil. */
@property (nonatomic, readonly, copy, nullable) NSNumber *boundedKeyValueByteCount;
/*! The longest context the model reads, or nil. */
@property (nonatomic, readonly, copy, nullable) NSNumber *maximumContextLength;
/*! The tokens in the model's vocabulary, which a draft model shares with the model it drafts for, or
	nil. */
@property (nonatomic, readonly, copy, nullable) NSNumber *vocabularySize;
/*! The bytes a load holds beyond the resident figure for a moment. 0 where it is not measured. */
@property (nonatomic, readonly) NSInteger loadOverheadByteCount;
/*! Where the figures come from: an NFKModelMemoryBasis* string. */
@property (nonatomic, readonly, copy) NSString *basis;
/*! The fields of the facts' dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! Facts with every field. A nil basis is NFKModelMemoryBasisMeasured. */
- (instancetype)initWithStages:(NSArray<NFKModelStageFacts *> *)stages
	 keyValueByteCountPerToken:(nullable NSNumber *)keyValueByteCountPerToken
	  boundedKeyValueByteCount:(nullable NSNumber *)boundedKeyValueByteCount
		  maximumContextLength:(nullable NSNumber *)maximumContextLength
				vocabularySize:(nullable NSNumber *)vocabularySize
		 loadOverheadByteCount:(NSInteger)loadOverheadByteCount
						 basis:(nullable NSString *)basis
					extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! The stage with the identifier, or nil. */
- (nullable NFKModelStageFacts *)stageWithIdentifier:(NSString *)identifier;

/*! Reads facts from their dictionary. Returns nil and sets error when stages is missing, a field has
	the wrong type, a stage is malformed, or two stages share an id. */
+ (nullable instancetype)factsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The facts as their dictionary: stages, keyValueBytesPerToken, boundedKeyValueBytes,
	maximumContextLength, vocabularySize, loadOverheadBytes, and basis. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelMemoryFigure
	@abstract   The memory a model holds under one choice: resident once loaded, and at the peak of a load.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryFigure : NSObject <NSCopying>

/*! The bytes the model holds once loaded. */
@property (nonatomic, readonly) NSInteger residentByteCount;
/*! The most bytes the model holds at once while it loads or runs. */
@property (nonatomic, readonly) NSInteger peakByteCount;

/*! A figure with both byte counts. */
- (instancetype)initWithResidentByteCount:(NSInteger)residentByteCount
							peakByteCount:(NSInteger)peakByteCount NS_DESIGNATED_INITIALIZER;

/*! Reads a figure from its dictionary. Returns nil and sets error when residentBytes is missing or a
	field is not a number. An absent peakBytes is residentBytes. */
+ (nullable instancetype)figureWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The figure as its dictionary: residentBytes and peakBytes. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelMemoryModel
	@abstract   Which of a descriptor's options change its memory, and how its stages take them.
	@discussion Each OptionKey property names the declared option whose value the estimate reads; nil
				when no option drives that figure. The variant comes from NFKModelLoadOptionVariant.
				- precisionOptionKey: a stage's precision, unless stagePrecisions fixes it. Absent, the
				  checkpoint's.
				- quantizationBitsOptionKey and quantizationGroupSizeOptionKey: the runtime weight
				  quantization of every stage with quantizable elements. 0 bits is none; the group size
				  defaults to 64.
				- contextLengthOptionKey: the positions the key-value cache holds. Absent, the facts'
				  maximumContextLength.
				- cacheBitsOptionKey and cacheGroupSizeOptionKey: a quantized key-value cache.
				- draftOptionKey and draftBitsOptionKey: the draft model's identifier and its
				  quantization.
				- residencyOptionKey: how the stages are held. Absent, the descriptor's only residency, or
				  NFKModelResidencyAutomatic.
				cacheStage names the stage that holds the key-value cache and the draft model, the last
				stage when nil. draftStage names the stage of this model a draft for another model loads,
				every stage when nil. Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryModel : NSObject <NSCopying>

@property (nonatomic, readonly, copy, nullable) NSString *precisionOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *quantizationBitsOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *quantizationGroupSizeOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *contextLengthOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *cacheBitsOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *cacheGroupSizeOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *draftOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *draftBitsOptionKey;
@property (nonatomic, readonly, copy, nullable) NSString *residencyOptionKey;
/*! The stage that holds the key-value cache and the draft model. The last stage when nil. */
@property (nonatomic, readonly, copy, nullable) NSString *cacheStage;
/*! The stage a draft of this model loads. Every stage when nil. */
@property (nonatomic, readonly, copy, nullable) NSString *draftStage;
/*! The precision of each stage the precision option does not change, keyed by stage identifier. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *stagePrecisions;
/*! The fields of the model's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A memory model with every field. */
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
								extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! The keys of every option the model names, in a fixed order. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *optionKeys;

/*! Reads a memory model from its dictionary, whose fields are precision, quantizationBits,
	quantizationGroupSize, contextLength, cacheBits, cacheGroupSize, draft, draftBits, residency,
	cacheStage, draftStage, and stagePrecisions. Returns nil and sets error when a field has the wrong
	type. */
+ (nullable instancetype)memoryModelWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The memory model as its dictionary. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

#pragma mark - Residency planner

/*!
	@class      NFKModelStageFootprint
	@abstract   What one stage weighs to the residency planner.
	@discussion byteCount is every byte the stage holds resident. pageableByteCount is the routed-expert
				share of it a paged load leaves in the release. widenedByteCount is the stage at a wider
				precision it takes where the budget holds it. layerByteCounts lists each repeated layer
				a streamed load may leave in the release, counted inside byteCount, in the order a
				streamed load holds them. Introduced in InferKit 0.4.0.
*/
@interface NFKModelStageFootprint : NSObject <NSCopying>

@property (nonatomic, readonly) NSInteger byteCount;
@property (nonatomic, readonly) NSInteger pageableByteCount;
@property (nonatomic, readonly, copy, nullable) NSNumber *widenedByteCount;
@property (nonatomic, readonly, copy) NSArray<NSNumber *> *layerByteCounts;

/*! A footprint with every field. */
- (instancetype)initWithByteCount:(NSInteger)byteCount
				pageableByteCount:(NSInteger)pageableByteCount
				 widenedByteCount:(nullable NSNumber *)widenedByteCount
				  layerByteCounts:(nullable NSArray<NSNumber *> *)layerByteCounts NS_DESIGNATED_INITIALIZER;

/*! A footprint of one precision, no routed experts, and no repeated layers. */
+ (instancetype)footprintWithByteCount:(NSInteger)byteCount;

/*! What the stage holds with its routed experts paged: byteCount less pageableByteCount. */
@property (nonatomic, readonly) NSInteger unpagedByteCount;
/*! What the stage holds outside its repeated layers. */
@property (nonatomic, readonly) NSInteger unstreamableByteCount;
/*! What a streamed load of the stage needs at the least: everything outside the repeated layers, and
	the two largest layers, the one computing and the one being read. */
@property (nonatomic, readonly) NSInteger streamedMinimumByteCount;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelResidencyPlan
	@abstract   How a model of stages holds its weights against a budget: resident, staged, paged, or
				streamed, and which stages widen.
	@discussion planForStages:residency:budget:error: decides each residency as it says:
				- NFKModelResidencyResident → every stage held, no paging; refused where the stages are
				  known not to fit.
				- NFKModelResidencyStaged → stages take turns, no paging.
				- NFKModelResidencyPaged → stages take turns and routed experts are paged; refused where a
				  stage is known not to fit even with its experts paged. A model without routed experts
				  is staged.
				- NFKModelResidencyStreamed → stages take turns, and a stage known not to fit whole
				  streams its repeated layers; refused where such a stage is known not to fit even
				  streamed. With no budget every stage with layers streams and holds none of them.
				- NFKModelResidencyAutomatic → the first that is known to fit: every stage held where the
				  total fits; paged where the largest stage is known not to fit whole and has experts to
				  page; streamed where a stage with repeated layers is known not to fit whole; staged
				  otherwise. With no budget, staged.

				A paged load's cache takes half of what the largest stage leaves of the budget once its
				unpaged weights and the reserve are counted, and never more than the stage's experts; 4
				GiB with no budget. A streamed stage holds its first layers while they fit what the budget
				leaves once the reserve, everything outside the layers, and the two layers in flight are
				counted. A stage with a wider precision takes it where that is known to fit, after the
				residency is decided:
				- held resident → stages widen in order while the total, widened, still fits.
				- staged or paged → a stage widens where it fits alone, widened.
				- no budget → no stage widens.

				A refusal is an NFKModelCatalogErrorDomain error with code NFKModelCatalogErrorDoesNotFit.
				The budget is a byte count, 0 for a machine that reports none. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelResidencyPlan : NSObject <NSCopying>

/*! YES when the stages stay loaded between runs. */
@property (nonatomic, readonly) BOOL holdsStagesResident;
/*! YES when the routed experts stay in the release, materialized as they are routed to. */
@property (nonatomic, readonly) BOOL pagesExperts;
/*! The bytes a paged load's cache may hold in materialized experts; 0 where nothing is paged. */
@property (nonatomic, readonly) NSInteger expertCacheByteCount;
/*! The positions of the stages that load at their wider precision. */
@property (nonatomic, readonly, copy) NSIndexSet *widenedStageIndexes;
/*! The stages that stream their repeated layers, each mapped to the bytes it may hold in layers beyond
	the two a pass has in flight. */
@property (nonatomic, readonly, copy) NSDictionary<NSNumber *, NSNumber *> *streamedStageByteCounts;
/*! The residency the plan amounts to: resident, paged, streamed, or staged. */
@property (nonatomic, readonly, copy) NSString *residency;

/*! YES when the stage at index loads at its wider precision. */
- (BOOL)widensStageAtIndex:(NSUInteger)index;
/*! YES when the stage at index streams its repeated layers. */
- (BOOL)streamsStageAtIndex:(NSUInteger)index;
/*! The layers of the stage at index a load holds: every one where the stage does not stream, and
	otherwise the first ones, in order, while their bytes fit what the plan gave the stage. */
- (NSRange)heldLayerRangeOfStageAtIndex:(NSUInteger)index footprint:(NFKModelStageFootprint *)footprint;

/*! How a model of stages is held under residency against budget bytes, or nil with a refusal. */
+ (nullable instancetype)planForStages:(NSArray<NFKModelStageFootprint *> *)stages
							 residency:(NSString *)residency
								budget:(NSInteger)budget
								 error:(NSError **)error;

/*! YES where every stage of a staged plan loads alone at the precision the plan chose for it; NO with
	the refusal naming the first that does not. A resident plan and a streamed stage were checked when
	the plan was made. names names each stage; a missing name reads "stage N". */
+ (BOOL)verifyEachStageLoads:(NSArray<NFKModelStageFootprint *> *)stages
						plan:(NFKModelResidencyPlan *)plan
					  budget:(NSInteger)budget
					   names:(NSArray<NSString *> *)names
					   error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

#pragma mark - Estimate

/*!
	@class      NFKModelMemoryEstimate
	@abstract   What a model holds under one combination of load options.
	@discussion residentByteCount is what the model holds once loaded under the resolved residency: the
				sum of its stages when resident, the largest stage when staged. The components sum to
				it, NFKModelMemoryComponentLoadOverhead aside. peakByteCount adds the load overhead.
				minimumWorkingSetByteCount is the least recommended working set whose budget admits the
				peak. fits answers whether budget admits the peak, and reason states why when it does
				not. Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryEstimate : NSObject <NSCopying>

@property (nonatomic, readonly) NSInteger residentByteCount;
@property (nonatomic, readonly) NSInteger peakByteCount;
@property (nonatomic, readonly) NSInteger minimumWorkingSetByteCount;
/*! The residency the options asked for, which can be NFKModelResidencyAutomatic. */
@property (nonatomic, readonly, copy) NSString *requestedResidency;
/*! The residency the planner chose: resident, staged, paged, or streamed. */
@property (nonatomic, readonly, copy) NSString *residency;
/*! The bytes of each part, keyed by NFKModelMemoryComponent* strings. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSNumber *> *components;
/*! The bytes each stage holds in its turn, keyed by stage identifier. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSNumber *> *stageByteCounts;
/*! The load options the estimate was computed for, defaults filled. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *resolvedOptions;
/*! The budget the estimate was judged against, or nil for an estimate without one. */
@property (nonatomic, readonly, copy, nullable) NFKModelMemoryBudget *budget;
/*! YES when the budget admits the peak, or there is no budget. */
@property (nonatomic, readonly) BOOL fits;
/*! The sentence that says why the estimate does not fit, or nil when it fits. */
@property (nonatomic, readonly, copy, nullable) NSString *reason;
/*! Where the figures come from: an NFKModelMemoryBasis* string. */
@property (nonatomic, readonly, copy) NSString *basis;

/*! The estimate as a figure. */
@property (nonatomic, readonly) NFKModelMemoryFigure *figure;

/*! Reads an estimate from its dictionary. Returns nil and sets error when a field has the wrong type. */
+ (nullable instancetype)estimateWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The estimate as its dictionary: residentBytes, peakBytes, minimumWorkingSetBytes,
	requestedResidency, residency, components, stageBytes, options, budget, fits, reason, and basis. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

#pragma mark - Constraints

/*!
	@enum       NFKModelChoiceStatus
	@abstract   Whether a choice can be selected under a budget.
	@constant   NFKModelChoiceStatusAllowed A combination that fits includes the choice.
	@constant   NFKModelChoiceStatusRequired The choice is the only one that fits.
	@constant   NFKModelChoiceStatusRuledOut No combination that fits includes the choice, or the machine
				does not meet the choice's requirements.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKModelChoiceStatus) {
	NFKModelChoiceStatusAllowed = 0,
	NFKModelChoiceStatusRequired = 1,
	NFKModelChoiceStatusRuledOut = 2,
};

/*!
	@class      NFKModelChoiceConstraint
	@abstract   One choice's status under a budget.
	@discussion peakByteCount is the least peak among the combinations that include the choice. Introduced
				in InferKit 0.4.0.
*/
@interface NFKModelChoiceConstraint : NSObject <NSCopying>

@property (nonatomic, readonly, copy) id value;
@property (nonatomic, readonly) NFKModelChoiceStatus status;
@property (nonatomic, readonly, copy, nullable) NSString *reason;
@property (nonatomic, readonly, copy, nullable) NSNumber *peakByteCount;

/*! A choice constraint with every field. */
- (instancetype)initWithValue:(id)value
					   status:(NFKModelChoiceStatus)status
					   reason:(nullable NSString *)reason
				peakByteCount:(nullable NSNumber *)peakByteCount NS_DESIGNATED_INITIALIZER;

/*! Reads a choice constraint from its dictionary. Returns nil and sets error when value or status is
	missing. */
+ (nullable instancetype)constraintWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The choice constraint as its dictionary: value, status ("allowed", "required", or "ruled-out"),
	reason, and peakBytes. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelOptionConstraint
	@abstract   What a budget leaves of one option.
	@discussion A choice option lists each choice's status. requiredValue is set where exactly one choice
				fits; for the residency option NFKModelResidencyAutomatic is not counted, since it picks
				one of the others. An integer option whose memory is linear states maximumValue, the
				largest value that fits, when that is below the option's own maximum. reason states the
				limit as a sentence, and fullyResidentByteCount, budgetByteCount, and
				techniqueByteCount carry its numbers so an app can write its own:
				- a required residency → the bytes fully resident, the budget, and the bytes under the
				  required residency.
				- another required choice → the bytes at the option's default, the budget, and the bytes
				  under the required choice.
				- a maximum → the budget.
				available is NO for an option whose requirements the machine does not meet. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelOptionConstraint : NSObject <NSCopying>

@property (nonatomic, readonly, copy) NSString *key;
@property (nonatomic, readonly, getter=isAvailable) BOOL available;
@property (nonatomic, readonly, copy) NSArray<NFKModelChoiceConstraint *> *choices;
@property (nonatomic, readonly, copy, nullable) id requiredValue;
@property (nonatomic, readonly, copy, nullable) NSNumber *maximumValue;
@property (nonatomic, readonly, copy, nullable) NSString *reason;
@property (nonatomic, readonly, copy, nullable) NSNumber *fullyResidentByteCount;
@property (nonatomic, readonly, copy, nullable) NSNumber *budgetByteCount;
@property (nonatomic, readonly, copy, nullable) NSNumber *techniqueByteCount;

/*! An option constraint with every field. */
- (instancetype)initWithKey:(NSString *)key
				  available:(BOOL)available
					choices:(nullable NSArray<NFKModelChoiceConstraint *> *)choices
			  requiredValue:(nullable id)requiredValue
			   maximumValue:(nullable NSNumber *)maximumValue
					 reason:(nullable NSString *)reason
	 fullyResidentByteCount:(nullable NSNumber *)fullyResidentByteCount
			budgetByteCount:(nullable NSNumber *)budgetByteCount
		  techniqueByteCount:(nullable NSNumber *)techniqueByteCount NS_DESIGNATED_INITIALIZER;

/*! The constraint of the choice whose value equals value, or nil. */
- (nullable NFKModelChoiceConstraint *)constraintForChoiceValue:(id)value;

/*! Reads an option constraint from its dictionary. Returns nil and sets error when key is missing or a
	field has the wrong type. */
+ (nullable instancetype)constraintWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The option constraint as its dictionary: key, available, choices, requiredValue, maximumValue,
	reason, fullyResidentBytes, budgetBytes, and techniqueBytes. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelMemoryConstraints
	@abstract   What a budget leaves of every option of a descriptor.
	@discussion anyCombinationFits is NO when no combination of the options fits; reason then states the
				smallest combination's need. fittingOptions is the combination that fits and changes the
				fewest options from their defaults, with each linear integer option at its default or the
				largest value that fits. fittingEstimate is its estimate.

				The memory-affecting choices are enumerated in full up to 4096 combinations, and
				exhaustive is YES. Past that, each option's choices are evaluated one at a time with every
				other option at the caller's selection, else its default, so every choice still has a
				status, and exhaustive is NO: a choice ruled out alone may fit beside other changes.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelMemoryConstraints : NSObject <NSCopying>

@property (nonatomic, readonly, copy) NFKModelMemoryBudget *budget;
@property (nonatomic, readonly) BOOL anyCombinationFits;
@property (nonatomic, readonly, copy, nullable) NSString *reason;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NFKModelOptionConstraint *> *optionConstraints;
@property (nonatomic, readonly, copy, nullable) NSDictionary<NSString *, id> *fittingOptions;
@property (nonatomic, readonly, copy, nullable) NFKModelMemoryEstimate *fittingEstimate;
/*! YES when every combination of the choices was evaluated; NO past the 4096-combination cap. */
@property (nonatomic, readonly, getter=isExhaustive) BOOL exhaustive;

/*! Constraints with every field. */
- (instancetype)initWithBudget:(NFKModelMemoryBudget *)budget
			anyCombinationFits:(BOOL)anyCombinationFits
						reason:(nullable NSString *)reason
			 optionConstraints:(nullable NSDictionary<NSString *, NFKModelOptionConstraint *> *)optionConstraints
				fittingOptions:(nullable NSDictionary<NSString *, id> *)fittingOptions
			   fittingEstimate:(nullable NFKModelMemoryEstimate *)fittingEstimate
					exhaustive:(BOOL)exhaustive NS_DESIGNATED_INITIALIZER;

/*! The constraint of the option with the key, or nil. */
- (nullable NFKModelOptionConstraint *)constraintForOptionKey:(NSString *)key;

/*! Reads constraints from their dictionary. Returns nil and sets error when budget is missing or a
	field has the wrong type. */
+ (nullable instancetype)constraintsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The constraints as their dictionary: budget, anyCombinationFits, exhaustive, reason, options,
	fittingOptions, and fittingEstimate. An absent exhaustive reads as YES. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

#pragma mark - Descriptor

/*!
	@category   NFKModelDescriptor (NFKModelMemory)
	@abstract   The memory a descriptor's model holds under its load options, and the options a budget
				leaves.
	@discussion The estimate reads the selected variant's memoryFacts and the descriptor's memoryModel. Per
				stage, at precision p, with e(p) the bytes of one element:
				- weights w: the stage's bytes at p. A quantization of b bits in groups of g replaces the
				  stage's Q quantizable elements: w − Q·e(p) + Q·(b/8 + 4/g).
				- key-value cache, on the cache stage, for c positions: (perToken·c + bounded)·e(p)/2
				  unquantized, and (perToken·c + bounded)/2·(b/8 + 4/g) quantized.
				- draft, on the cache stage: the draft descriptor's draft stage at the draft's precision
				  and bits, and the draft's own key-value cache at the same context length, cache
				  precision, and cache bits.
				The residency then decides the resident figure: the sum when resident, the largest stage
				when staged, the largest unpaged stage plus the expert cache when paged, and a streamed
				stage's minimum plus the layers that fit when streamed. Automatic asks
				NFKModelResidencyPlan. The peak adds the facts' load overhead, and the estimate fits when
				the budget admits the peak.

				A variant without memoryFacts estimates its residentBytes and peakBytes as one stage, and a
				variant with neither estimates 0 bytes with basis NFKModelMemoryBasisUnknown. A draft
				model is looked up in NFKModelCatalog.sharedCatalog. Introduced in InferKit 0.4.0.
*/
@interface NFKModelDescriptor (NFKModelMemory)

/*! The estimate for the load options, defaults filled, without a budget. Returns nil and sets error
	(NFKModelCatalogErrorInvalidOptions) when the options do not validate. */
- (nullable NFKModelMemoryEstimate *)estimatedMemoryForOptions:(nullable NSDictionary<NSString *, id> *)options
														 error:(NSError **)error;

/*! The estimate for the load options, defaults filled, judged against a budget. */
- (nullable NFKModelMemoryEstimate *)estimatedMemoryForOptions:(nullable NSDictionary<NSString *, id> *)options
												  memoryBudget:(nullable NFKModelMemoryBudget *)budget
														 error:(NSError **)error;

/*! What the budget leaves of every option, with none fixed. */
- (NFKModelMemoryConstraints *)constraintsForMemoryBudget:(NFKModelMemoryBudget *)budget;

/*! What a machine leaves of every option: its budget, and the choices whose requirements it does not
	meet ruled out. */
- (NFKModelMemoryConstraints *)constraintsForHardwareProfile:(NFKHardwareProfile *)profile;

/*!
	@method     constraintsForOptions:memoryBudget:
	@abstract   What the budget leaves of every option, with the given options fixed.
	@discussion Each option's constraint fixes the other given options and lets the rest vary, so an app
				passes the user's current selections and reads what each control still allows. The
				memory-affecting choice options vary over their choices (up to 4096 combinations; past
				that, one option at a time, and the result's exhaustive is NO), and the linear integer
				options are solved for directly: the largest value is
				(budget − reserve − peak at 0) / bytes per unit, clipped to the option's range. A
				combination fits when it fits with each linear option at its given value, else its
				minimum.
*/
- (NFKModelMemoryConstraints *)constraintsForOptions:(nullable NSDictionary<NSString *, id> *)options
										memoryBudget:(NFKModelMemoryBudget *)budget;

/*!
	@method     descriptorByDerivingMemoryFigures
	@abstract   The descriptor with a memory figure on each memory-affecting choice, computed from the
				facts.
	@discussion Each choice's figure is the estimate with that choice and every other option at its
				default, without a budget; memoryByVariant repeats it for each variant of a descriptor
				with several. The residency choice NFKModelResidencyAutomatic takes no figure, since its
				memory depends on the machine. A linear integer option without memoryBytesPerUnit takes
				the cache bytes one position adds at the defaults. A variant with facts takes
				residentBytes, peakBytes, and minimumMemoryBytes from the estimate at the defaults. A
				figure the manifest states wins over the derived one. Draft models and model references
				are looked up in NFKModelCatalog.sharedCatalog.
*/
- (NFKModelDescriptor *)descriptorByDerivingMemoryFigures;

/*!
	@method     descriptorByDerivingMemoryFiguresInCatalog:
	@abstract   The descriptor with a memory figure on each memory-affecting choice, with draft models and
				model references looked up in catalog.
	@discussion The figures follow descriptorByDerivingMemoryFigures. A descriptor provider derives its
				descriptors' figures in its own catalog, because the shared catalog calls the provider
				while it loads. Introduced in InferKit 0.4.0.
*/
- (NFKModelDescriptor *)descriptorByDerivingMemoryFiguresInCatalog:(NFKModelCatalog *)catalog
	NS_SWIFT_NAME(derivingMemoryFigures(in:));

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelMemory_h */
