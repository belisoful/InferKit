//
//  NFKWireValueObjects.h
//  InferKit
//
//  The JSON form of InferKit's value classes, for a reader that has no Objective-C runtime. The wire
//  coding writes it when encodesValueObjectsAsJSON is set and reads it always.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	The JSON object for value, tagged under "$nfk" with its class's wire name, or nil when value is not
	one of the classes below. Points are {"x", "y"} and rectangles {"x", "y", "width", "height"}, in the
	value's own coordinates. A non-finite figure is null.

	| class | tag | fields |
	| NFKDetection | detection | label, classIndex, confidence, boundingBox, quadrilateral |
	| NFKKeypoint | keypoint | name, index, position, confidence |
	| NFKClassification | classification | label, classIndex, confidence |
	| NFKAudioSegment | audioSegment | startSeconds, endSeconds, label, confidence, speaker |
	| NFKMIDINote | midiNote | pitch, startSeconds, endSeconds, velocity, program, percussion, pitchBend |
	| NFKMIDISequence | midiSequence | notes, tempoBPM, beatsPerBar, beatUnit, ticksPerQuarterNote, durationSeconds |
	| NFKMusicBeat | musicBeat | timeSeconds, positionInBar, downbeat |
	| NFKQuadrilateral | quadrilateral | topLeft, topRight, bottomLeft, bottomRight, boundingBox |
	| NFKDecisionAnswer | decisionAnswer | type, choice, score, probability, confidence, probabilities, legend, raw, refused |

	A nil string field is left out. durationSeconds, downbeat, and a quadrilateral's boundingBox are
	derived, written for a reader's convenience and ignored on reading.
*/
NSDictionary<NSString *, id> * _Nullable NFKWireJSONForValueObject(id value);

/*! Whether tag names one of the value classes NFKWireJSONForValueObject writes. */
BOOL NFKWireIsValueObjectTag(NSString *tag);

@class NFKMesh;
/*! A mesh's JSON object, with its PLY bytes inline when `inlineBytes` is set (see NFKWireJSONForValueObject). */
NSDictionary *NFKWireMeshJSON(NFKMesh *mesh, BOOL inlineBytes);

/*! The value an entry of a value-class tag describes, or nil with a reason when a required field is
	missing or of the wrong type. */
id _Nullable NFKWireValueObjectFromJSON(NSString *tag, NSDictionary<NSString *, id> *entry, NSString * _Nullable * _Nullable outReason);

NS_ASSUME_NONNULL_END
