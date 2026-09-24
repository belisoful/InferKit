//
//  NFKMusicBeat.h
//  InferKit
//

#ifndef NFKMusicBeat_h
#define NFKMusicBeat_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKMusicBeat
	@abstract   One beat, and where it falls in its bar.
	@discussion A music-structure backend returns an NSArray<NFKMusicBeat *> under NFKOutputBeats,
				ordered in time, with the estimated tempo under NFKOutputTempo and the functional
				sections (intro, verse, chorus) as NFKAudioSegments under NFKOutputSegments.

				positionInBar counts from 1, so a downbeat is position 1 and the bar length is the
				highest position a track reaches. A model that tracks beats without meter reports
				position 0, which is what isDownbeat reads.

				Introduced in InferKit 0.4.0.

				The type archives: a consumer that records a result per frame writes an array of them
				through NSKeyedArchiver with secure coding on, and reads it back with
				unarchivedObjectOfClasses:. Conformance introduced in InferKit 0.4.0.
*/
@interface NFKMusicBeat : NSObject <NSCopying, NSSecureCoding>

/*! The beat's time, in seconds from the start of the clip. */
@property (nonatomic, readonly) double timeSeconds;

/*! The beat's position in its bar, counting from 1, or 0 when the model does not track the meter. */
@property (nonatomic, readonly) NSInteger positionInBar;

/*! YES when the beat starts a bar. */
@property (nonatomic, readonly, getter=isDownbeat) BOOL downbeat;

+ (instancetype)beatWithTimeSeconds:(double)timeSeconds positionInBar:(NSInteger)positionInBar;

- (instancetype)initWithTimeSeconds:(double)timeSeconds positionInBar:(NSInteger)positionInBar NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMusicBeat_h */
