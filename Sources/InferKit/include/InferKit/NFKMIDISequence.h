//
//  NFKMIDISequence.h
//  InferKit
//

#ifndef NFKMIDISequence_h
#define NFKMIDISequence_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKMIDINote.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKMIDISequence
	@abstract   A transcribed performance: notes in seconds, with the tempo they are written at.
	@discussion A music-transcription backend (audio in, notes out) returns one of these under
				NFKOutputMIDI. The notes carry absolute times, so the tempo and the time signature
				only decide how a Standard MIDI File spells them; changing the tempo moves the bar
				lines and leaves every note where it sounded.

				standardMIDIFileData writes a format-1 file: a conductor track carrying the tempo and
				the time signature, then one track per distinct program. Percussion notes go to
				channel 10, as the format requires. A note's pitchBend is written as pitch-wheel
				events across the note, at the file's ±2 semitone range.

				Introduced in InferKit 0.4.0.

				The type archives: a consumer that records a result per frame writes an array of them
				through NSKeyedArchiver with secure coding on, and reads it back with
				unarchivedObjectOfClasses:. Conformance introduced in InferKit 0.4.0.
*/
@interface NFKMIDISequence : NSObject <NSCopying, NSSecureCoding>

/*! The notes, ordered by start time. */
@property (nonatomic, readonly, copy) NSArray<NFKMIDINote *> *notes;

/*! The tempo the file is written at, in beats per minute. */
@property (nonatomic, readonly) double tempoBPM;

/*! The time signature's numerator (beats per bar). */
@property (nonatomic, readonly) NSInteger beatsPerBar;

/*! The time signature's denominator (the note value that takes a beat: 4 for a quarter note). */
@property (nonatomic, readonly) NSInteger beatUnit;

/*! The file's timing resolution, in ticks per quarter note. */
@property (nonatomic, readonly) NSInteger ticksPerQuarterNote;

/*! The end of the last note, in seconds, or 0 when there are no notes. */
@property (nonatomic, readonly) double durationSeconds;

/*! A sequence at 120 BPM in 4/4, at 480 ticks per quarter note. */
+ (instancetype)sequenceWithNotes:(NSArray<NFKMIDINote *> *)notes;

+ (instancetype)sequenceWithNotes:(NSArray<NFKMIDINote *> *)notes tempoBPM:(double)tempoBPM;

- (instancetype)initWithNotes:(NSArray<NFKMIDINote *> *)notes
					 tempoBPM:(double)tempoBPM
				  beatsPerBar:(NSInteger)beatsPerBar
					 beatUnit:(NSInteger)beatUnit
		  ticksPerQuarterNote:(NSInteger)ticksPerQuarterNote NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The performance as a Standard MIDI File. */
- (NSData *)standardMIDIFileData;

/*! Writes the Standard MIDI File to a URL. Returns NO and fills error when the write fails. */
- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMIDISequence_h */
