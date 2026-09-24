//
//  NFKAudioSegment.h
//  InferKit
//

#ifndef NFKAudioSegment_h
#define NFKAudioSegment_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKAudioSegment
	@abstract   A labeled span of time in an audio clip.
	@discussion A backend that locates events over time (voice-activity
				detection, sound-event detection, diarization) returns an NSArray<NFKAudioSegment *>
				under NFKOutputSegments, ordered by start time. The span is in seconds from the start
				of the clip. label names the event when the backend has a label set (a speaker id, a
				sound class), nil for a plain activity span such as "speech".

				The type archives: a consumer that records a result per frame writes an array of them
				through NSKeyedArchiver with secure coding on, and reads it back with
				unarchivedObjectOfClasses:. Conformance introduced in InferKit 0.4.0.
*/
@interface NFKAudioSegment : NSObject <NSCopying, NSSecureCoding>

/*! The span's start, in seconds from the start of the clip. */
@property (nonatomic, readonly) double startSeconds;

/*! The span's end, in seconds from the start of the clip. */
@property (nonatomic, readonly) double endSeconds;

/*! The event label, or nil for a plain activity span. */
@property (nonatomic, readonly, nullable, copy) NSString *label;

/*! The confidence in 0...1. */
@property (nonatomic, readonly) double confidence;

/*! Who speaks in the span, as the backend names the speaker ("spk_1", "SPEAKER_00"), or nil when
	the backend does not diarize. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, nullable, copy) NSString *speaker;

+ (instancetype)segmentWithStartSeconds:(double)startSeconds
							endSeconds:(double)endSeconds
								 label:(nullable NSString *)label
							confidence:(double)confidence;

/*! A span with its speaker. Introduced in InferKit 0.4.0. */
+ (instancetype)segmentWithStartSeconds:(double)startSeconds
							endSeconds:(double)endSeconds
								 label:(nullable NSString *)label
							confidence:(double)confidence
							   speaker:(nullable NSString *)speaker;

- (instancetype)initWithStartSeconds:(double)startSeconds
						 endSeconds:(double)endSeconds
							  label:(nullable NSString *)label
						 confidence:(double)confidence;

/*! Introduced in InferKit 0.4.0. */
- (instancetype)initWithStartSeconds:(double)startSeconds
						 endSeconds:(double)endSeconds
							  label:(nullable NSString *)label
						 confidence:(double)confidence
							speaker:(nullable NSString *)speaker NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKAudioSegment_h */
