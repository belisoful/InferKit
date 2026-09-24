//
//  NFKMIDINote.h
//  InferKit
//

#ifndef NFKMIDINote_h
#define NFKMIDINote_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKMIDINote
	@abstract   One transcribed note: a pitch held over a span of time.
	@discussion A music-transcription backend returns notes inside an NFKMIDISequence under
				NFKOutputMIDI. Times are seconds from the start of the clip. pitch and velocity are
				MIDI numbers, so they need no conversion on the way to a file or a synthesizer.

				pitchBend carries the note's continuous pitch in semitones relative to pitch, sampled
				evenly from startSeconds to endSeconds. A model that estimates pitch continuously
				(vibrato, a bent string, a sung slide) fills it; a model that reports a fixed pitch
				leaves it nil. The unit is semitones rather than pitch-wheel ticks because the wheel's
				range is a property of the file, not of the performance.

				Introduced in InferKit 0.4.0.

				The type archives: a consumer that records a result per frame writes an array of them
				through NSKeyedArchiver with secure coding on, and reads it back with
				unarchivedObjectOfClasses:. Conformance introduced in InferKit 0.4.0.
*/
@interface NFKMIDINote : NSObject <NSCopying, NSSecureCoding>

/*! The MIDI note number, 0...127. 60 is middle C. */
@property (nonatomic, readonly) NSInteger pitch;

/*! The onset, in seconds from the start of the clip. */
@property (nonatomic, readonly) double startSeconds;

/*! The offset, in seconds from the start of the clip. */
@property (nonatomic, readonly) double endSeconds;

/*! The MIDI velocity, 0...127. */
@property (nonatomic, readonly) NSInteger velocity;

/*! The General MIDI program the note plays on, 0...127. A model that does not identify the
	instrument reports the program its own reference writes. */
@property (nonatomic, readonly) NSInteger program;

/*! YES when the note is percussion, which a Standard MIDI File plays on channel 10. The program is
	then the percussion key rather than an instrument. */
@property (nonatomic, readonly, getter=isPercussion) BOOL percussion;

/*! The note's pitch in semitones relative to pitch, sampled evenly over its duration, or nil when
	the model reports a fixed pitch. */
@property (nonatomic, readonly, nullable, copy) NSArray<NSNumber *> *pitchBend;

/*! The note's duration in seconds. */
@property (nonatomic, readonly) double durationSeconds;

+ (instancetype)noteWithPitch:(NSInteger)pitch
				 startSeconds:(double)startSeconds
				   endSeconds:(double)endSeconds
					 velocity:(NSInteger)velocity;

- (instancetype)initWithPitch:(NSInteger)pitch
				 startSeconds:(double)startSeconds
				   endSeconds:(double)endSeconds
					 velocity:(NSInteger)velocity
					  program:(NSInteger)program
				   percussion:(BOOL)percussion
					pitchBend:(nullable NSArray<NSNumber *> *)pitchBend NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMIDINote_h */
