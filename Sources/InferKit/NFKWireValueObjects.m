//
//  NFKWireValueObjects.m
//  InferKit
//

#import "NFKWireValueObjects.h"
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKDecisionAnswer.h>
#import <InferKit/NFKDetection.h>
#import <InferKit/NFKKeypoint.h>
#import <InferKit/NFKMIDINote.h>
#import <InferKit/NFKMIDISequence.h>
#import <InferKit/NFKMusicBeat.h>
#import <InferKit/NFKQuadrilateral.h>

#import <CoreGraphics/CoreGraphics.h>

static NSString * const NFKWireValueTag = @"$nfk";

#pragma mark - Writing

static id NFKWireFigure(double value)
{
	return isfinite(value) ? @(value) : NSNull.null;
}

static NSDictionary<NSString *, id> *NFKWirePoint(CGPoint point)
{
	return @{ @"x": NFKWireFigure(point.x), @"y": NFKWireFigure(point.y) };
}

static NSDictionary<NSString *, id> *NFKWireRect(CGRect rect)
{
	return @{ @"x": NFKWireFigure(rect.origin.x), @"y": NFKWireFigure(rect.origin.y),
			  @"width": NFKWireFigure(rect.size.width), @"height": NFKWireFigure(rect.size.height) };
}

/*! The part of value JSON carries: strings, finite numbers, null, and arrays and string-keyed
	dictionaries of them. */
static id NFKWireJSONSafe(id value)
{
	if ([value isKindOfClass:NSString.class] || value == NSNull.null) {
		return value;
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return isfinite([value doubleValue]) ? value : nil;
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *safe = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			id kept = NFKWireJSONSafe(element);
			if (kept != nil) {
				[safe addObject:kept];
			}
		}
		return safe;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *safe = [NSMutableDictionary dictionaryWithCapacity:[value count]];
		for (id key in value) {
			id kept = [key isKindOfClass:NSString.class] ? NFKWireJSONSafe(value[key]) : nil;
			if (kept != nil) {
				safe[key] = kept;
			}
		}
		return safe;
	}
	return nil;
}

static NSString *NFKWireDecisionTypeName(NFKDecisionType type)
{
	switch (type) {
		case NFKDecisionTypeScore:
			return @"score";
		case NFKDecisionTypeNoul:
			return @"noul";
		default:
			return @"choice";
	}
}

static NSDictionary<NSString *, id> *NFKWireQuadrilateralJSON(NFKQuadrilateral *quadrilateral)
{
	return @{ NFKWireValueTag: @"quadrilateral", @"topLeft": NFKWirePoint(quadrilateral.topLeft),
			  @"topRight": NFKWirePoint(quadrilateral.topRight), @"bottomLeft": NFKWirePoint(quadrilateral.bottomLeft),
			  @"bottomRight": NFKWirePoint(quadrilateral.bottomRight), @"boundingBox": NFKWireRect(quadrilateral.boundingBox) };
}

static NSDictionary<NSString *, id> *NFKWireMIDINoteJSON(NFKMIDINote *note)
{
	NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"midiNote", @"pitch": @(note.pitch),
													 @"startSeconds": NFKWireFigure(note.startSeconds),
													 @"endSeconds": NFKWireFigure(note.endSeconds), @"velocity": @(note.velocity),
													 @"program": @(note.program), @"percussion": @(note.isPercussion) } mutableCopy];
	if (note.pitchBend != nil) {
		entry[@"pitchBend"] = NFKWireJSONSafe(note.pitchBend);
	}
	return entry;
}

NSDictionary<NSString *, id> * _Nullable NFKWireJSONForValueObject(id value)
{
	if ([value isKindOfClass:NFKDetection.class]) {
		NFKDetection *detection = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"detection", @"classIndex": @(detection.classIndex),
														 @"confidence": NFKWireFigure(detection.confidence),
														 @"boundingBox": NFKWireRect(detection.boundingBox) } mutableCopy];
		entry[@"label"] = detection.label;
		if (detection.quadrilateral != nil) {
			entry[@"quadrilateral"] = NFKWireQuadrilateralJSON(detection.quadrilateral);
		}
		return entry;
	}
	if ([value isKindOfClass:NFKKeypoint.class]) {
		NFKKeypoint *keypoint = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"keypoint", @"index": @(keypoint.index),
														 @"position": NFKWirePoint(keypoint.position),
														 @"confidence": NFKWireFigure(keypoint.confidence) } mutableCopy];
		entry[@"name"] = keypoint.name;
		return entry;
	}
	if ([value isKindOfClass:NFKClassification.class]) {
		NFKClassification *classification = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"classification", @"classIndex": @(classification.classIndex),
														 @"confidence": NFKWireFigure(classification.confidence) } mutableCopy];
		entry[@"label"] = classification.label;
		return entry;
	}
	if ([value isKindOfClass:NFKAudioSegment.class]) {
		NFKAudioSegment *segment = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"audioSegment",
														 @"startSeconds": NFKWireFigure(segment.startSeconds),
														 @"endSeconds": NFKWireFigure(segment.endSeconds),
														 @"confidence": NFKWireFigure(segment.confidence) } mutableCopy];
		entry[@"label"] = segment.label;
		entry[@"speaker"] = segment.speaker;
		return entry;
	}
	if ([value isKindOfClass:NFKMIDINote.class]) {
		return NFKWireMIDINoteJSON(value);
	}
	if ([value isKindOfClass:NFKMIDISequence.class]) {
		NFKMIDISequence *sequence = value;
		NSMutableArray *notes = [NSMutableArray arrayWithCapacity:sequence.notes.count];
		for (NFKMIDINote *note in sequence.notes) {
			[notes addObject:NFKWireMIDINoteJSON(note)];
		}
		return @{ NFKWireValueTag: @"midiSequence", @"notes": notes, @"tempoBPM": NFKWireFigure(sequence.tempoBPM),
				  @"beatsPerBar": @(sequence.beatsPerBar), @"beatUnit": @(sequence.beatUnit),
				  @"ticksPerQuarterNote": @(sequence.ticksPerQuarterNote), @"durationSeconds": NFKWireFigure(sequence.durationSeconds) };
	}
	if ([value isKindOfClass:NFKMusicBeat.class]) {
		NFKMusicBeat *beat = value;
		return @{ NFKWireValueTag: @"musicBeat", @"timeSeconds": NFKWireFigure(beat.timeSeconds),
				  @"positionInBar": @(beat.positionInBar), @"downbeat": @(beat.isDownbeat) };
	}
	if ([value isKindOfClass:NFKQuadrilateral.class]) {
		return NFKWireQuadrilateralJSON(value);
	}
	if ([value isKindOfClass:NFKDecisionAnswer.class]) {
		NFKDecisionAnswer *answer = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ NFKWireValueTag: @"decisionAnswer", @"type": NFKWireDecisionTypeName(answer.type),
														 @"score": NFKWireFigure(answer.score), @"probability": NFKWireFigure(answer.probability),
														 @"confidence": NFKWireFigure(answer.confidence), @"refused": @(answer.isRefused),
														 @"raw": NFKWireJSONSafe(answer.raw) ?: @{} } mutableCopy];
		entry[@"choice"] = answer.choice;
		entry[@"probabilities"] = NFKWireJSONSafe(answer.probabilities);
		entry[@"legend"] = NFKWireJSONSafe(answer.legend);
		return entry;
	}
	return nil;
}

#pragma mark - Reading

BOOL NFKWireIsValueObjectTag(NSString *tag)
{
	static NSSet<NSString *> *tags;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		tags = [NSSet setWithArray:@[ @"detection", @"keypoint", @"classification", @"audioSegment", @"midiNote",
									  @"midiSequence", @"musicBeat", @"quadrilateral", @"decisionAnswer" ]];
	});
	return [tags containsObject:tag];
}

/*! A figure as a double: a number as itself, null as NaN. NO when the field is absent or another type. */
static BOOL NFKWireReadFigure(id field, double *outValue)
{
	if ([field isKindOfClass:NSNumber.class]) {
		*outValue = [field doubleValue];
		return YES;
	}
	if (field == NSNull.null) {
		*outValue = NAN;
		return YES;
	}
	return NO;
}

static double NFKWireOptionalFigure(id field)
{
	double value = 0;
	return NFKWireReadFigure(field, &value) ? value : 0;
}

static BOOL NFKWireReadPoint(id field, CGPoint *outPoint)
{
	double x = 0;
	double y = 0;
	if (![field isKindOfClass:NSDictionary.class] || !NFKWireReadFigure(field[@"x"], &x) || !NFKWireReadFigure(field[@"y"], &y)) {
		return NO;
	}
	*outPoint = CGPointMake(x, y);
	return YES;
}

static BOOL NFKWireReadRect(id field, CGRect *outRect)
{
	double x = 0;
	double y = 0;
	double width = 0;
	double height = 0;
	BOOL read = [field isKindOfClass:NSDictionary.class] && NFKWireReadFigure(field[@"x"], &x) && NFKWireReadFigure(field[@"y"], &y)
		&& NFKWireReadFigure(field[@"width"], &width) && NFKWireReadFigure(field[@"height"], &height);
	if (read) {
		*outRect = CGRectMake(x, y, width, height);
	}
	return read;
}

static NSString * _Nullable NFKWireOptionalString(id field)
{
	return [field isKindOfClass:NSString.class] ? field : nil;
}

static NSInteger NFKWireOptionalInteger(id field, NSInteger fallback)
{
	return [field isKindOfClass:NSNumber.class] ? [field integerValue] : fallback;
}

static id _Nullable NFKWireFail(NSString * _Nullable * _Nullable outReason, NSString *reason)
{
	if (outReason != NULL) {
		*outReason = reason;
	}
	return nil;
}

static NFKQuadrilateral * _Nullable NFKWireQuadrilateralFromJSON(NSDictionary *entry, NSString * _Nullable * _Nullable outReason)
{
	CGPoint corners[4];
	NSArray<NSString *> *names = @[ @"topLeft", @"topRight", @"bottomLeft", @"bottomRight" ];
	for (NSUInteger index = 0; index < names.count; index++) {
		if (!NFKWireReadPoint(entry[names[index]], &corners[index])) {
			return NFKWireFail(outReason, [NSString stringWithFormat:@"a quadrilateral needs the point %@", names[index]]);
		}
	}
	return [NFKQuadrilateral quadrilateralWithTopLeft:corners[0] topRight:corners[1] bottomLeft:corners[2] bottomRight:corners[3]];
}

static NFKMIDINote * _Nullable NFKWireMIDINoteFromJSON(NSDictionary *entry, NSString * _Nullable * _Nullable outReason)
{
	double start = 0;
	double end = 0;
	if (![entry[@"pitch"] isKindOfClass:NSNumber.class] || !NFKWireReadFigure(entry[@"startSeconds"], &start)
		|| !NFKWireReadFigure(entry[@"endSeconds"], &end)) {
		return NFKWireFail(outReason, @"a MIDI note needs pitch, startSeconds, and endSeconds");
	}
	NSArray *bend = [entry[@"pitchBend"] isKindOfClass:NSArray.class] ? entry[@"pitchBend"] : nil;
	return [[NFKMIDINote alloc] initWithPitch:[entry[@"pitch"] integerValue] startSeconds:start endSeconds:end
									 velocity:NFKWireOptionalInteger(entry[@"velocity"], 100)
									  program:NFKWireOptionalInteger(entry[@"program"], 0)
								   percussion:[entry[@"percussion"] boolValue] pitchBend:bend];
}

static NFKDecisionType NFKWireDecisionTypeNamed(NSString *name)
{
	if ([name isEqualToString:@"score"]) {
		return NFKDecisionTypeScore;
	}
	return [name isEqualToString:@"noul"] ? NFKDecisionTypeNoul : NFKDecisionTypeChoice;
}

id _Nullable NFKWireValueObjectFromJSON(NSString *tag, NSDictionary<NSString *, id> *entry, NSString * _Nullable * _Nullable outReason)
{
	if ([tag isEqualToString:@"detection"]) {
		CGRect box = CGRectZero;
		if (!NFKWireReadRect(entry[@"boundingBox"], &box)) {
			return NFKWireFail(outReason, @"a detection needs a boundingBox");
		}
		NFKQuadrilateral *quadrilateral = nil;
		if ([entry[@"quadrilateral"] isKindOfClass:NSDictionary.class]) {
			quadrilateral = NFKWireQuadrilateralFromJSON(entry[@"quadrilateral"], outReason);
			if (quadrilateral == nil) {
				return nil;
			}
		}
		return [[NFKDetection alloc] initWithLabel:NFKWireOptionalString(entry[@"label"])
										classIndex:NFKWireOptionalInteger(entry[@"classIndex"], -1)
										confidence:NFKWireOptionalFigure(entry[@"confidence"]) boundingBox:box
									 quadrilateral:quadrilateral];
	}
	if ([tag isEqualToString:@"keypoint"]) {
		CGPoint position = CGPointZero;
		if (!NFKWireReadPoint(entry[@"position"], &position)) {
			return NFKWireFail(outReason, @"a keypoint needs a position");
		}
		return [[NFKKeypoint alloc] initWithName:NFKWireOptionalString(entry[@"name"]) index:NFKWireOptionalInteger(entry[@"index"], -1)
										position:position confidence:NFKWireOptionalFigure(entry[@"confidence"])];
	}
	if ([tag isEqualToString:@"classification"]) {
		return [[NFKClassification alloc] initWithLabel:NFKWireOptionalString(entry[@"label"])
											 classIndex:NFKWireOptionalInteger(entry[@"classIndex"], -1)
											 confidence:NFKWireOptionalFigure(entry[@"confidence"])];
	}
	if ([tag isEqualToString:@"audioSegment"]) {
		double start = 0;
		double end = 0;
		if (!NFKWireReadFigure(entry[@"startSeconds"], &start) || !NFKWireReadFigure(entry[@"endSeconds"], &end)) {
			return NFKWireFail(outReason, @"an audio segment needs startSeconds and endSeconds");
		}
		return [[NFKAudioSegment alloc] initWithStartSeconds:start endSeconds:end label:NFKWireOptionalString(entry[@"label"])
												  confidence:NFKWireOptionalFigure(entry[@"confidence"])
													 speaker:NFKWireOptionalString(entry[@"speaker"])];
	}
	if ([tag isEqualToString:@"midiNote"]) {
		return NFKWireMIDINoteFromJSON(entry, outReason);
	}
	if ([tag isEqualToString:@"midiSequence"]) {
		if (![entry[@"notes"] isKindOfClass:NSArray.class]) {
			return NFKWireFail(outReason, @"a MIDI sequence needs notes");
		}
		NSMutableArray<NFKMIDINote *> *notes = [NSMutableArray array];
		for (id note in entry[@"notes"]) {
			NFKMIDINote *read = [note isKindOfClass:NSDictionary.class] ? NFKWireMIDINoteFromJSON(note, outReason) : nil;
			if (read == nil) {
				return NFKWireFail(outReason, @"a MIDI sequence's note is malformed");
			}
			[notes addObject:read];
		}
		double tempo = 0;
		return [[NFKMIDISequence alloc] initWithNotes:notes
											 tempoBPM:NFKWireReadFigure(entry[@"tempoBPM"], &tempo) ? tempo : 120
										  beatsPerBar:NFKWireOptionalInteger(entry[@"beatsPerBar"], 4)
											 beatUnit:NFKWireOptionalInteger(entry[@"beatUnit"], 4)
								  ticksPerQuarterNote:NFKWireOptionalInteger(entry[@"ticksPerQuarterNote"], 480)];
	}
	if ([tag isEqualToString:@"musicBeat"]) {
		double time = 0;
		if (!NFKWireReadFigure(entry[@"timeSeconds"], &time)) {
			return NFKWireFail(outReason, @"a beat needs timeSeconds");
		}
		return [[NFKMusicBeat alloc] initWithTimeSeconds:time positionInBar:NFKWireOptionalInteger(entry[@"positionInBar"], 0)];
	}
	if ([tag isEqualToString:@"quadrilateral"]) {
		return NFKWireQuadrilateralFromJSON(entry, outReason);
	}
	if ([tag isEqualToString:@"decisionAnswer"]) {
		NSString *typeName = NFKWireOptionalString(entry[@"type"]);
		if (typeName == nil) {
			return NFKWireFail(outReason, @"a decision answer needs a type");
		}
		NFKDecisionType type = NFKWireDecisionTypeNamed(typeName);
		NSDictionary *raw = [entry[@"raw"] isKindOfClass:NSDictionary.class] ? entry[@"raw"] : nil;
		if ([entry[@"refused"] boolValue]) {
			return [NFKDecisionAnswer refusalForType:type raw:raw];
		}
		return [[NFKDecisionAnswer alloc] initWithType:type choice:NFKWireOptionalString(entry[@"choice"])
												 score:NFKWireOptionalFigure(entry[@"score"])
										   probability:NFKWireOptionalFigure(entry[@"probability"])
											confidence:NFKWireOptionalFigure(entry[@"confidence"])
										 probabilities:[entry[@"probabilities"] isKindOfClass:NSDictionary.class] ? entry[@"probabilities"] : nil
												legend:[entry[@"legend"] isKindOfClass:NSDictionary.class] ? entry[@"legend"] : nil
												   raw:raw];
	}
	return NFKWireFail(outReason, [NSString stringWithFormat:@"\"%@\" is no value class", tag]);
}
