//
//  NFKMIDISequence.m
//  InferKit
//

#import "NFKMIDISequence.h"

/// The pitch-wheel range a Standard MIDI File assumes without an explicit RPN: two semitones either
/// way over the wheel's 14 bits, so one semitone is 4096 ticks.
static const double NFKMIDIPitchBendTicksPerSemitone = 4096.0;
static const int32_t NFKMIDIPitchBendMinimum = -8192;
static const int32_t NFKMIDIPitchBendMaximum = 8191;
static const uint8_t NFKMIDIPercussionChannel = 9;

/// Events sort by tick, then by this order, so a voice is released and its wheel recentered before
/// anything at the same instant retriggers it.
typedef NS_ENUM(NSInteger, NFKMIDIEventOrder) {
	NFKMIDIEventOrderProgramChange = 0,
	NFKMIDIEventOrderNoteOff = 1,
	NFKMIDIEventOrderPitchBend = 2,
	NFKMIDIEventOrderNoteOn = 3,
};

@interface NFKMIDIEvent : NSObject
@property (nonatomic) uint32_t tick;
@property (nonatomic) NFKMIDIEventOrder order;
@property (nonatomic) NSUInteger sequenceIndex;
@property (nonatomic, copy) NSData *bytes;
@end

@implementation NFKMIDIEvent
@end

@implementation NFKMIDISequence

+ (instancetype)sequenceWithNotes:(NSArray<NFKMIDINote *> *)notes
{
	return [self sequenceWithNotes:notes tempoBPM:120.0];
}

+ (instancetype)sequenceWithNotes:(NSArray<NFKMIDINote *> *)notes tempoBPM:(double)tempoBPM
{
	return [[self alloc] initWithNotes:notes tempoBPM:tempoBPM beatsPerBar:4 beatUnit:4 ticksPerQuarterNote:480];
}

- (instancetype)initWithNotes:(NSArray<NFKMIDINote *> *)notes
					 tempoBPM:(double)tempoBPM
				  beatsPerBar:(NSInteger)beatsPerBar
					 beatUnit:(NSInteger)beatUnit
		  ticksPerQuarterNote:(NSInteger)ticksPerQuarterNote
{
	self = [super init];
	if (self != nil) {
		_notes = [[notes sortedArrayUsingComparator:^NSComparisonResult(NFKMIDINote *left, NFKMIDINote *right) {
			if (left.startSeconds < right.startSeconds) {
				return NSOrderedAscending;
			}
			if (left.startSeconds > right.startSeconds) {
				return NSOrderedDescending;
			}
			if (left.pitch == right.pitch) {
				return NSOrderedSame;
			}
			return left.pitch < right.pitch ? NSOrderedAscending : NSOrderedDescending;
		}] copy];
		_tempoBPM = tempoBPM > 0.0 ? tempoBPM : 120.0;
		_beatsPerBar = beatsPerBar > 0 ? beatsPerBar : 4;
		_beatUnit = beatUnit > 0 ? beatUnit : 4;
		_ticksPerQuarterNote = ticksPerQuarterNote > 0 ? ticksPerQuarterNote : 480;
	}
	return self;
}

- (double)durationSeconds
{
	double duration = 0.0;
	for (NFKMIDINote *note in self.notes) {
		duration = MAX(duration, note.endSeconds);
	}
	return duration;
}

- (id)copyWithZone:(NSZone *)zone
{
	// Immutable.
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKMIDISequence.class]) {
		return NO;
	}
	NFKMIDISequence *other = object;
	return [self.notes isEqualToArray:other.notes]
		&& self.tempoBPM == other.tempoBPM
		&& self.beatsPerBar == other.beatsPerBar
		&& self.beatUnit == other.beatUnit
		&& self.ticksPerQuarterNote == other.ticksPerQuarterNote;
}

- (NSUInteger)hash
{
	return self.notes.hash ^ (NSUInteger)(self.tempoBPM * 1000) ^ (NSUInteger)self.beatsPerBar
		^ (NSUInteger)self.beatUnit ^ (NSUInteger)self.ticksPerQuarterNote;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %lu notes, %.1f BPM, %.3fs>",
			NSStringFromClass(self.class), (unsigned long)self.notes.count, self.tempoBPM, self.durationSeconds];
}

#pragma mark Standard MIDI File

- (uint32_t)tickForSeconds:(double)seconds
{
	double ticks = seconds * (double)self.ticksPerQuarterNote * self.tempoBPM / 60.0;
	if (ticks <= 0.0) {
		return 0;
	}
	return (uint32_t)llround(ticks);
}

static void NFKMIDIAppendVariableLength(NSMutableData *data, uint32_t value)
{
	uint8_t buffer[4];
	NSUInteger count = 0;
	buffer[count++] = (uint8_t)(value & 0x7F);
	value >>= 7;
	while (value > 0) {
		buffer[count++] = (uint8_t)((value & 0x7F) | 0x80);
		value >>= 7;
	}
	// The buffer filled least-significant group first; the file wants the other order.
	while (count > 0) {
		uint8_t byte = buffer[--count];
		[data appendBytes:&byte length:1];
	}
}

static void NFKMIDIAppendUInt16(NSMutableData *data, uint16_t value)
{
	uint8_t bytes[2] = { (uint8_t)(value >> 8), (uint8_t)(value & 0xFF) };
	[data appendBytes:bytes length:2];
}

static void NFKMIDIAppendUInt32(NSMutableData *data, uint32_t value)
{
	uint8_t bytes[4] = { (uint8_t)(value >> 24), (uint8_t)(value >> 16), (uint8_t)(value >> 8), (uint8_t)value };
	[data appendBytes:bytes length:4];
}

static NSData *NFKMIDITrackChunk(NSArray<NFKMIDIEvent *> *events)
{
	NSArray<NFKMIDIEvent *> *ordered = [events sortedArrayUsingComparator:^NSComparisonResult(NFKMIDIEvent *left, NFKMIDIEvent *right) {
		if (left.tick != right.tick) {
			return left.tick < right.tick ? NSOrderedAscending : NSOrderedDescending;
		}
		if (left.order != right.order) {
			return left.order < right.order ? NSOrderedAscending : NSOrderedDescending;
		}
		if (left.sequenceIndex == right.sequenceIndex) {
			return NSOrderedSame;
		}
		return left.sequenceIndex < right.sequenceIndex ? NSOrderedAscending : NSOrderedDescending;
	}];

	NSMutableData *body = [NSMutableData data];
	uint32_t previousTick = 0;
	for (NFKMIDIEvent *event in ordered) {
		NFKMIDIAppendVariableLength(body, event.tick - previousTick);
		[body appendData:event.bytes];
		previousTick = event.tick;
	}
	const uint8_t endOfTrack[3] = { 0xFF, 0x2F, 0x00 };
	NFKMIDIAppendVariableLength(body, 0);
	[body appendBytes:endOfTrack length:3];

	NSMutableData *chunk = [NSMutableData data];
	[chunk appendBytes:"MTrk" length:4];
	NFKMIDIAppendUInt32(chunk, (uint32_t)body.length);
	[chunk appendData:body];
	return chunk;
}

static NFKMIDIEvent *NFKMIDIMakeEvent(uint32_t tick, NFKMIDIEventOrder order, NSUInteger index, const uint8_t *bytes, NSUInteger length)
{
	NFKMIDIEvent *event = [[NFKMIDIEvent alloc] init];
	event.tick = tick;
	event.order = order;
	event.sequenceIndex = index;
	event.bytes = [NSData dataWithBytes:bytes length:length];
	return event;
}

/*! The notes grouped into tracks, keyed by the program they play on, percussion apart. */
- (NSArray<NSArray<NFKMIDINote *> *> *)tracks
{
	NSMutableArray<NSNumber *> *keys = [NSMutableArray array];
	NSMutableDictionary<NSNumber *, NSMutableArray<NFKMIDINote *> *> *grouped = [NSMutableDictionary dictionary];
	for (NFKMIDINote *note in self.notes) {
		NSNumber *key = @(note.isPercussion ? -1 : note.program);
		NSMutableArray<NFKMIDINote *> *track = grouped[key];
		if (track == nil) {
			track = [NSMutableArray array];
			grouped[key] = track;
			[keys addObject:key];
		}
		[track addObject:note];
	}
	NSMutableArray<NSArray<NFKMIDINote *> *> *tracks = [NSMutableArray array];
	for (NSNumber *key in keys) {
		[tracks addObject:grouped[key]];
	}
	return tracks;
}

/*! The channel a track plays on. Percussion takes channel 10, which the format reserves for it, and
	the melodic tracks take the rest in order. A file with more than fifteen programs reuses channels,
	which costs it the per-channel pitch wheel rather than any note. */
static uint8_t NFKMIDIChannelForTrack(NSArray<NFKMIDINote *> *track, NSUInteger melodicIndex)
{
	if (track.firstObject.isPercussion) {
		return NFKMIDIPercussionChannel;
	}
	uint8_t channel = (uint8_t)(melodicIndex % 15);
	return channel < NFKMIDIPercussionChannel ? channel : (uint8_t)(channel + 1);
}

static void NFKMIDIAppendPitchBend(NSMutableArray<NFKMIDIEvent *> *events, uint32_t tick, uint8_t channel, double semitones, NSUInteger index)
{
	int32_t wheel = (int32_t)llround(semitones * NFKMIDIPitchBendTicksPerSemitone);
	wheel = MAX(NFKMIDIPitchBendMinimum, MIN(NFKMIDIPitchBendMaximum, wheel));
	uint16_t value = (uint16_t)(wheel - NFKMIDIPitchBendMinimum);
	const uint8_t bytes[3] = { (uint8_t)(0xE0 | channel), (uint8_t)(value & 0x7F), (uint8_t)((value >> 7) & 0x7F) };
	[events addObject:NFKMIDIMakeEvent(tick, NFKMIDIEventOrderPitchBend, index, bytes, 3)];
}

- (NSData *)standardMIDIFileData
{
	NSArray<NSArray<NFKMIDINote *> *> *tracks = [self tracks];

	NSMutableData *conductor = [NSMutableData data];
	uint32_t microsecondsPerQuarter = (uint32_t)llround(60000000.0 / self.tempoBPM);
	const uint8_t tempoMeta[3] = { 0xFF, 0x51, 0x03 };
	NSMutableData *tempoBytes = [NSMutableData dataWithBytes:tempoMeta length:3];
	const uint8_t tempoValue[3] = {
		(uint8_t)((microsecondsPerQuarter >> 16) & 0xFF),
		(uint8_t)((microsecondsPerQuarter >> 8) & 0xFF),
		(uint8_t)(microsecondsPerQuarter & 0xFF),
	};
	[tempoBytes appendBytes:tempoValue length:3];

	uint8_t denominatorPower = 0;
	NSInteger unit = self.beatUnit;
	while (unit > 1) {
		unit >>= 1;
		denominatorPower += 1;
	}
	const uint8_t timeSignature[7] = {
		0xFF, 0x58, 0x04, (uint8_t)self.beatsPerBar, denominatorPower, 24, 8
	};

	NSMutableArray<NFKMIDIEvent *> *conductorEvents = [NSMutableArray array];
	[conductorEvents addObject:NFKMIDIMakeEvent(0, NFKMIDIEventOrderProgramChange, 0, tempoBytes.bytes, tempoBytes.length)];
	[conductorEvents addObject:NFKMIDIMakeEvent(0, NFKMIDIEventOrderProgramChange, 1, timeSignature, 7)];
	[conductor appendData:NFKMIDITrackChunk(conductorEvents)];

	NSMutableData *trackChunks = [NSMutableData data];
	NSUInteger melodicIndex = 0;
	for (NSArray<NFKMIDINote *> *track in tracks) {
		uint8_t channel = NFKMIDIChannelForTrack(track, melodicIndex);
		if (!track.firstObject.isPercussion) {
			melodicIndex += 1;
		}

		NSMutableArray<NFKMIDIEvent *> *events = [NSMutableArray array];
		NSUInteger index = 0;
		if (!track.firstObject.isPercussion) {
			const uint8_t programChange[2] = { (uint8_t)(0xC0 | channel), (uint8_t)MAX(0, MIN(127, track.firstObject.program)) };
			[events addObject:NFKMIDIMakeEvent(0, NFKMIDIEventOrderProgramChange, index++, programChange, 2)];
		}

		for (NFKMIDINote *note in track) {
			uint8_t pitch = (uint8_t)MAX(0, MIN(127, note.pitch));
			uint8_t velocity = (uint8_t)MAX(0, MIN(127, note.velocity));
			uint32_t startTick = [self tickForSeconds:note.startSeconds];
			uint32_t endTick = [self tickForSeconds:note.endSeconds];
			if (endTick <= startTick) {
				endTick = startTick + 1;
			}

			const uint8_t noteOn[3] = { (uint8_t)(0x90 | channel), pitch, velocity };
			[events addObject:NFKMIDIMakeEvent(startTick, NFKMIDIEventOrderNoteOn, index++, noteOn, 3)];
			const uint8_t noteOff[3] = { (uint8_t)(0x80 | channel), pitch, 0 };
			[events addObject:NFKMIDIMakeEvent(endTick, NFKMIDIEventOrderNoteOff, index++, noteOff, 3)];

			NSArray<NSNumber *> *bend = note.pitchBend;
			if (bend.count == 0) {
				continue;
			}
			double span = note.endSeconds - note.startSeconds;
			for (NSUInteger i = 0; i < bend.count; i++) {
				double fraction = bend.count > 1 ? (double)i / (double)(bend.count - 1) : 0.0;
				uint32_t tick = [self tickForSeconds:note.startSeconds + fraction * span];
				NFKMIDIAppendPitchBend(events, tick, channel, bend[i].doubleValue, index++);
			}
			// The wheel is per channel, so it is recentered when the note ends; otherwise the bend
			// carries into whatever plays next on the same channel.
			NFKMIDIAppendPitchBend(events, endTick, channel, 0.0, index++);
		}
		[trackChunks appendData:NFKMIDITrackChunk(events)];
	}

	NSMutableData *file = [NSMutableData data];
	[file appendBytes:"MThd" length:4];
	NFKMIDIAppendUInt32(file, 6);
	NFKMIDIAppendUInt16(file, 1);
	NFKMIDIAppendUInt16(file, (uint16_t)(tracks.count + 1));
	NFKMIDIAppendUInt16(file, (uint16_t)self.ticksPerQuarterNote);
	[file appendData:conductor];
	[file appendData:trackChunks];
	return file;
}

- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error
{
	return [[self standardMIDIFileData] writeToURL:url options:NSDataWritingAtomic error:error];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.notes forKey:@"notes"];
	[coder encodeDouble:self.tempoBPM forKey:@"tempoBPM"];
	[coder encodeInteger:self.beatsPerBar forKey:@"beatsPerBar"];
	[coder encodeInteger:self.beatUnit forKey:@"beatUnit"];
	[coder encodeInteger:self.ticksPerQuarterNote forKey:@"ticksPerQuarterNote"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	// The notes are objects, so their classes are named as well as the array's.
	NSSet *noteClasses = [NSSet setWithObjects:NSArray.class, NFKMIDINote.class, nil];
	NSArray<NFKMIDINote *> *notes = [coder decodeObjectOfClasses:noteClasses forKey:@"notes"];
	return [self initWithNotes:notes ?: @[]
					  tempoBPM:[coder decodeDoubleForKey:@"tempoBPM"]
				   beatsPerBar:[coder decodeIntegerForKey:@"beatsPerBar"]
					  beatUnit:[coder decodeIntegerForKey:@"beatUnit"]
		   ticksPerQuarterNote:[coder decodeIntegerForKey:@"ticksPerQuarterNote"]];
}

@end
