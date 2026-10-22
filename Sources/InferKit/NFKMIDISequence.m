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

#pragma mark - Reading a Standard MIDI File

/*! One event of a file being read, at its absolute tick. kind orders events at the same tick: a tempo,
	then a program change, then a note's end, then a note's start. */
typedef NS_ENUM(NSInteger, NFKMIDIReadKind) {
	NFKMIDIReadKindTempo = 0,
	NFKMIDIReadKindProgram,
	NFKMIDIReadKindNoteOff,
	NFKMIDIReadKindNoteOn,
};

@interface NFKMIDIReadEvent : NSObject
@property (nonatomic, assign) uint64_t tick;
@property (nonatomic, assign) NFKMIDIReadKind kind;
@property (nonatomic, assign) NSUInteger sequenceIndex;
@property (nonatomic, assign) uint8_t channel;
@property (nonatomic, assign) uint32_t value;
@property (nonatomic, assign) uint8_t velocity;
@end

@implementation NFKMIDIReadEvent
@end

static BOOL NFKMIDIReadFail(NSError **error, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NSCocoaErrorDomain code:NSFileReadCorruptFileError
								 userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithFormat:@"The MIDI file cannot be read: %@.", reason] }];
	}
	return NO;
}

/*! Reads a variable-length quantity at *offset, advancing it, or returns NO past end or past four bytes. */
static BOOL NFKMIDIReadVariableLength(const uint8_t *bytes, NSUInteger end, NSUInteger *offset, uint32_t *outValue)
{
	uint32_t value = 0;
	for (NSUInteger count = 0; count < 4; count++) {
		if (*offset >= end) {
			return NO;
		}
		uint8_t byte = bytes[(*offset)++];
		value = (value << 7) | (byte & 0x7F);
		if ((byte & 0x80) == 0) {
			*outValue = value;
			return YES;
		}
	}
	return NO;
}

static uint32_t NFKMIDIReadUInt32(const uint8_t *bytes)
{
	return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | bytes[3];
}

/*! Reads one track's events into events, with their absolute ticks, and its end tick into outEnd. Reads
	the first time signature into signature when it is still unset. */
static BOOL NFKMIDIReadTrack(const uint8_t *bytes, NSUInteger start, NSUInteger end, NSMutableArray<NFKMIDIReadEvent *> *events,
							 uint64_t *outEnd, NSInteger signature[2], NSError **error)
{
	NSUInteger offset = start;
	uint64_t tick = 0;
	uint8_t status = 0;
	while (offset < end) {
		uint32_t delta = 0;
		if (!NFKMIDIReadVariableLength(bytes, end, &offset, &delta) || offset >= end) {
			return NFKMIDIReadFail(error, @"a track ends inside an event");
		}
		tick += delta;
		uint8_t byte = bytes[offset];
		if (byte & 0x80) {
			status = byte;
			offset += 1;
		} else if (status == 0 || status >= 0xF0) {
			return NFKMIDIReadFail(error, @"a data byte arrives with no running status");
		}
		if (status == 0xFF || status == 0xF0 || status == 0xF7) {
			uint8_t type = 0;
			if (status == 0xFF) {
				if (offset >= end) {
					return NFKMIDIReadFail(error, @"a meta event is cut short");
				}
				type = bytes[offset++];
			}
			uint32_t length = 0;
			if (!NFKMIDIReadVariableLength(bytes, end, &offset, &length) || length > end - offset) {
				return NFKMIDIReadFail(error, @"an event's length runs past its track");
			}
			if (status == 0xFF && type == 0x51 && length == 3) {
				NFKMIDIReadEvent *tempo = [[NFKMIDIReadEvent alloc] init];
				tempo.tick = tick;
				tempo.kind = NFKMIDIReadKindTempo;
				tempo.value = ((uint32_t)bytes[offset] << 16) | ((uint32_t)bytes[offset + 1] << 8) | bytes[offset + 2];
				[events addObject:tempo];
			} else if (status == 0xFF && type == 0x58 && length >= 2 && signature[0] == 0) {
				signature[0] = bytes[offset];
				signature[1] = bytes[offset + 1] < 8 ? (NSInteger)1 << bytes[offset + 1] : 4;
			}
			offset += length;
			if (status == 0xFF && type == 0x2F) {
				break;
			}
			// Meta and system exclusive events cancel the running status.
			status = 0;
			continue;
		}
		uint8_t command = status & 0xF0;
		NSUInteger dataLength = (command == 0xC0 || command == 0xD0) ? 1 : 2;
		if (status >= 0xF1 || dataLength > end - offset) {
			return NFKMIDIReadFail(error, @"a channel event is cut short");
		}
		uint8_t first = bytes[offset] & 0x7F;
		uint8_t second = dataLength == 2 ? (bytes[offset + 1] & 0x7F) : 0;
		offset += dataLength;
		if (command == 0x90 || command == 0x80 || command == 0xC0) {
			NFKMIDIReadEvent *event = [[NFKMIDIReadEvent alloc] init];
			event.tick = tick;
			event.channel = status & 0x0F;
			event.value = first;
			event.velocity = second;
			event.kind = command == 0xC0 ? NFKMIDIReadKindProgram
				: (command == 0x90 && second > 0 ? NFKMIDIReadKindNoteOn : NFKMIDIReadKindNoteOff);
			[events addObject:event];
		}
	}
	*outEnd = tick;
	return YES;
}

/*! The seconds at each tick under a tempo map of (tick, microseconds per quarter) pairs in tick order. */
static double NFKMIDISecondsAtTick(uint64_t tick, NSArray<NFKMIDIReadEvent *> *tempos, double ticksPerQuarter)
{
	double seconds = 0;
	uint64_t mark = 0;
	double microsecondsPerQuarter = 500000;
	for (NFKMIDIReadEvent *tempo in tempos) {
		if (tempo.tick >= tick) {
			break;
		}
		seconds += (double)(tempo.tick - mark) * microsecondsPerQuarter / ticksPerQuarter / 1e6;
		mark = tempo.tick;
		microsecondsPerQuarter = tempo.value > 0 ? tempo.value : microsecondsPerQuarter;
	}
	return seconds + (double)(tick - mark) * microsecondsPerQuarter / ticksPerQuarter / 1e6;
}

#pragma mark -

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

#pragma mark Reading

+ (nullable instancetype)sequenceWithStandardMIDIFileData:(NSData *)data error:(NSError **)error
{
	const uint8_t *bytes = data.bytes;
	NSUInteger length = data.length;
	if (length < 14 || memcmp(bytes, "MThd", 4) != 0 || NFKMIDIReadUInt32(bytes + 4) < 6) {
		NFKMIDIReadFail(error, @"it has no MThd header");
		return nil;
	}
	NSUInteger headerLength = NFKMIDIReadUInt32(bytes + 4);
	uint16_t format = (uint16_t)((bytes[8] << 8) | bytes[9]);
	uint16_t division = (uint16_t)((bytes[12] << 8) | bytes[13]);
	if (format > 1) {
		NFKMIDIReadFail(error, @"a format-2 file holds independent patterns, not one performance");
		return nil;
	}
	if ((division & 0x8000) != 0 || division == 0) {
		NFKMIDIReadFail(error, @"it is timed in SMPTE frames or names no resolution");
		return nil;
	}
	if (headerLength > length - 8) {
		NFKMIDIReadFail(error, @"its header runs past the file");
		return nil;
	}
	NSMutableArray<NFKMIDIReadEvent *> *events = [NSMutableArray array];
	NSMutableArray<NSNumber *> *trackEnds = [NSMutableArray array];
	NSMutableArray<NSNumber *> *eventTracks = [NSMutableArray array];
	NSInteger signature[2] = { 0, 0 };
	NSUInteger offset = 8 + headerLength;
	while (offset + 8 <= length) {
		NSUInteger chunkLength = NFKMIDIReadUInt32(bytes + offset + 4);
		if (chunkLength > length - offset - 8) {
			NFKMIDIReadFail(error, @"a chunk runs past the file");
			return nil;
		}
		if (memcmp(bytes + offset, "MTrk", 4) == 0) {
			NSUInteger before = events.count;
			uint64_t end = 0;
			if (!NFKMIDIReadTrack(bytes, offset + 8, offset + 8 + chunkLength, events, &end, signature, error)) {
				return nil;
			}
			for (NSUInteger index = before; index < events.count; index++) {
				[eventTracks addObject:@(trackEnds.count)];
			}
			[trackEnds addObject:@(end)];
		}
		offset += 8 + chunkLength;
	}
	[events enumerateObjectsUsingBlock:^(NFKMIDIReadEvent *event, NSUInteger index, BOOL *stop) {
		event.sequenceIndex = index;
	}];
	NSArray<NFKMIDIReadEvent *> *ordered = [events sortedArrayUsingComparator:^NSComparisonResult(NFKMIDIReadEvent *left, NFKMIDIReadEvent *right) {
		if (left.tick != right.tick) {
			return left.tick < right.tick ? NSOrderedAscending : NSOrderedDescending;
		}
		if (left.kind != right.kind) {
			return left.kind < right.kind ? NSOrderedAscending : NSOrderedDescending;
		}
		return left.sequenceIndex < right.sequenceIndex ? NSOrderedAscending : NSOrderedDescending;
	}];
	NSMutableArray<NFKMIDIReadEvent *> *tempos = [NSMutableArray array];
	for (NFKMIDIReadEvent *event in ordered) {
		if (event.kind == NFKMIDIReadKindTempo) {
			[tempos addObject:event];
		}
	}
	double ticksPerQuarter = division;
	NSInteger programs[16] = { 0 };
	NSMutableDictionary<NSNumber *, NSMutableArray<NFKMIDIReadEvent *> *> *sounding = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSNumber *, NSNumber *> *startPrograms = [NSMutableDictionary dictionary];
	NSMutableArray<NFKMIDINote *> *notes = [NSMutableArray array];
	void (^close)(NFKMIDIReadEvent *, uint64_t) = ^(NFKMIDIReadEvent *start, uint64_t endTick) {
		NSInteger program = startPrograms[@(start.sequenceIndex)].integerValue;
		[notes addObject:[[NFKMIDINote alloc] initWithPitch:start.value
											   startSeconds:NFKMIDISecondsAtTick(start.tick, tempos, ticksPerQuarter)
												 endSeconds:NFKMIDISecondsAtTick(endTick, tempos, ticksPerQuarter)
												   velocity:start.velocity
													program:program
												 percussion:start.channel == NFKMIDIPercussionChannel
												  pitchBend:nil]];
	};
	for (NFKMIDIReadEvent *event in ordered) {
		NSNumber *voice = @((event.channel << 7) | (event.value & 0x7F));
		switch (event.kind) {
			case NFKMIDIReadKindTempo:
				break;
			case NFKMIDIReadKindProgram:
				programs[event.channel] = event.value;
				break;
			case NFKMIDIReadKindNoteOn: {
				NSMutableArray<NFKMIDIReadEvent *> *open = sounding[voice] ?: [NSMutableArray array];
				sounding[voice] = open;
				startPrograms[@(event.sequenceIndex)] = @(programs[event.channel]);
				[open addObject:event];
				break;
			}
			case NFKMIDIReadKindNoteOff: {
				NFKMIDIReadEvent *start = sounding[voice].firstObject;
				if (start != nil) {
					[sounding[voice] removeObjectAtIndex:0];
					close(start, event.tick);
				}
				break;
			}
		}
	}
	for (NSNumber *voice in sounding) {
		for (NFKMIDIReadEvent *start in sounding[voice]) {
			close(start, trackEnds[eventTracks[start.sequenceIndex].unsignedIntegerValue].unsignedLongLongValue);
		}
	}
	NFKMIDIReadEvent *firstTempo = tempos.firstObject;
	double tempoBPM = firstTempo != nil && firstTempo.value > 0 ? 60000000.0 / firstTempo.value : 120.0;
	return [[self alloc] initWithNotes:notes tempoBPM:tempoBPM beatsPerBar:signature[0] > 0 ? signature[0] : 4
							  beatUnit:signature[0] > 0 ? signature[1] : 4 ticksPerQuarterNote:division];
}

+ (nullable instancetype)sequenceWithContentsOfURL:(NSURL *)url error:(NSError **)error
{
	NSData *data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:error];
	return data != nil ? [self sequenceWithStandardMIDIFileData:data error:error] : nil;
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
