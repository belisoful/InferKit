//
//  NFKModelLive.h
//  InferKit
//

#ifndef NFKModelLive_h
#define NFKModelLive_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelCatalog.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelLive
	@abstract   How a model runs on a live source: the unit it takes, the keys it reads and writes per unit,
				and its measured time per unit.
	@discussion A descriptor whose model runs on a stream of video frames or audio chunks carries an
				NFKModelLiveProfile (NFKModelDescriptor.liveProfile). A client offers a model for a live
				source when it has a profile, and reads the profile's timings to tell whether the model
				keeps up on the machine at the source's rate. Introduced in InferKit 0.4.0.
*/

#pragma mark Units

/*! One video frame: an image under the profile's input keys in, the frame's results out. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelLiveUnitVideoFrame;
/*! One chunk of audio of the profile's duration, sample rate, and channel count in, the chunk's results
	out. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelLiveUnitAudioChunk;

/*!
	@class      NFKModelLiveTiming
	@abstract   The measured time one release of a model takes per unit on one machine.
	@discussion A timing belongs to the release that was measured. variantIdentifier names it; nil names the
				descriptor's default variant. A video timing names the frame size it was measured at; an
				audio timing measures the profile's chunk. The time is the median over the measured units,
				after the first units have warmed the model. Introduced in InferKit 0.4.0.
*/
@interface NFKModelLiveTiming : NSObject <NSCopying>

/*! The variant measured, or nil for the descriptor's default variant. */
@property (nonatomic, readonly, copy, nullable) NSString *variantIdentifier;
/*! The machine's chip, as NFKHardwareProfile.chipName reports it, such as "Apple M3 Max". */
@property (nonatomic, readonly, copy) NSString *chip;
/*! The frame width in pixels a video timing was measured at, or 0 for an audio timing. */
@property (nonatomic, readonly) NSInteger width;
/*! The frame height in pixels a video timing was measured at, or 0 for an audio timing. */
@property (nonatomic, readonly) NSInteger height;
/*! The median time per unit, in milliseconds. */
@property (nonatomic, readonly) double millisecondsPerUnit;
/*! The day of the measurement, "YYYY-MM-DD". */
@property (nonatomic, readonly, copy) NSString *measuredDate;
/*! A sentence on how the timing was measured, such as a setting that differs from the defaults. */
@property (nonatomic, readonly, copy, nullable) NSString *note;
/*! The fields of the timing's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A timing with every field. */
- (instancetype)initWithVariantIdentifier:(nullable NSString *)variantIdentifier
									 chip:(NSString *)chip
									width:(NSInteger)width
								   height:(NSInteger)height
					  millisecondsPerUnit:(double)millisecondsPerUnit
							 measuredDate:(NSString *)measuredDate
									 note:(nullable NSString *)note
							   extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! Reads a timing from its manifest dictionary. Absent width and height are 0. Returns nil and sets error
	when chip, milliseconds, or measured is missing, milliseconds is not positive, or a field has the
	wrong type. */
+ (nullable instancetype)timingWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The timing as its manifest dictionary: variant, chip, width, height, milliseconds, measured, and note,
	with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelLiveProfile
	@abstract   A model's declaration that it runs on a live source, one unit at a time.
	@discussion The unit is a video frame or an audio chunk. Each unit is one request: the input keys carry
				the unit, and the output keys carry its results.

				A stateful model carries state from one unit to the next, such as a recurrent matting
				decoder's hidden state or a streaming recognizer's context; its results depend on the units
				before. A stateless model treats each unit alone, so a client may drop units when it falls
				behind.

				An audio profile states the chunk the model takes (its duration, sample rate, and channel
				count) and the latency the model adds past the chunk, its lookahead. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelLiveProfile : NSObject <NSCopying>

/*! NFKModelLiveUnitVideoFrame or NFKModelLiveUnitAudioChunk. */
@property (nonatomic, readonly, copy) NSString *unit;
/*! The input keys a unit carries, keys of the descriptor's inputs, such as NFKInputImage. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *inputKeys;
/*! The output keys a unit's result carries, keys of the descriptor's outputs, such as NFKOutputMask. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *outputKeys;
/*! YES when the model carries state from unit to unit. */
@property (nonatomic, readonly, getter=isStateful) BOOL stateful;
/*! An audio chunk's duration in seconds, or 0 for video. */
@property (nonatomic, readonly) double chunkDuration;
/*! The sample rate an audio chunk is in, in hertz, or 0 for video. */
@property (nonatomic, readonly) double sampleRate;
/*! The channel count an audio chunk has, or 0 for video. */
@property (nonatomic, readonly) NSInteger channelCount;
/*! The latency the model adds past a unit, in milliseconds: the audio it reads ahead before it answers.
	0 for a model that answers from the unit alone. */
@property (nonatomic, readonly) double lookaheadMilliseconds;
/*! The measured times per unit. */
@property (nonatomic, readonly, copy) NSArray<NFKModelLiveTiming *> *timings;
/*! The fields of the profile's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A video-frame profile. */
+ (instancetype)videoFrameProfileWithInputKeys:(NSArray<NSString *> *)inputKeys
									outputKeys:(NSArray<NSString *> *)outputKeys
									  stateful:(BOOL)stateful
									   timings:(nullable NSArray<NFKModelLiveTiming *> *)timings;

/*! An audio-chunk profile. */
+ (instancetype)audioChunkProfileWithInputKeys:(NSArray<NSString *> *)inputKeys
									outputKeys:(NSArray<NSString *> *)outputKeys
									  stateful:(BOOL)stateful
								 chunkDuration:(double)chunkDuration
									sampleRate:(double)sampleRate
								  channelCount:(NSInteger)channelCount
						 lookaheadMilliseconds:(double)lookaheadMilliseconds
									   timings:(nullable NSArray<NFKModelLiveTiming *> *)timings;

/*! A profile with every field. A nil timings is empty. */
- (instancetype)initWithUnit:(NSString *)unit
				   inputKeys:(NSArray<NSString *> *)inputKeys
				  outputKeys:(NSArray<NSString *> *)outputKeys
					stateful:(BOOL)stateful
			   chunkDuration:(double)chunkDuration
				  sampleRate:(double)sampleRate
				channelCount:(NSInteger)channelCount
	   lookaheadMilliseconds:(double)lookaheadMilliseconds
					 timings:(nullable NSArray<NFKModelLiveTiming *> *)timings
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! Reads a profile from its manifest dictionary. Absent stateful is NO, and absent audio fields are 0.
	Returns nil and sets error when unit is missing or unknown, either key list is missing or empty, an
	audio profile lacks a positive chunkDuration, sampleRate, or channelCount, a video profile carries
	one, a video timing lacks its frame size, or a field has the wrong type. */
+ (nullable instancetype)profileWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The profile as its manifest dictionary: unit, inputs, outputs, stateful, the audio fields of an audio
	profile, lookahead, and timings, with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

/*! The timings of one variant, nil for the default variant. A timing that names no variant belongs to the
	default variant. */
- (NSArray<NFKModelLiveTiming *> *)timingsForVariant:(nullable NSString *)variantIdentifier;

/*!
	@method     keepsUpWithRate:variant:width:height:chip:
	@abstract   Whether a measured release keeps up with a source on a chip.
	@discussion Video: rate is the source's frames a second, and the timing is the variant's on the chip at
				the frame size, or else at the smallest measured size at least as large. It keeps up when
				its time per frame is at most 1000 / rate milliseconds. Audio: rate is ignored, width and
				height are ignored, and it keeps up when its time per chunk is below the chunk's duration.
				Returns @YES or @NO, or nil when no timing answers.
*/
- (nullable NSNumber *)keepsUpWithRate:(double)rate
							   variant:(nullable NSString *)variantIdentifier
								 width:(NSInteger)width
								height:(NSInteger)height
								  chip:(NSString *)chip
	NS_SWIFT_NAME(keepsUp(rate:variant:width:height:chip:));

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@category   NFKModelCatalog (NFKModelLive)
	@abstract   The catalog's descriptors that run on a live source.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKModelCatalog (NFKModelLive)

/*! The visible descriptors with a live profile whose unit is unit, or of either unit when unit is nil,
	ordered by identifier. */
- (NSArray<NFKModelDescriptor *> *)liveDescriptorsForUnit:(nullable NSString *)unit;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelLive_h */
