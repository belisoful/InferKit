//
//  NFKMIDISequenceTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKMIDINote.h>
#import <InferKit/NFKMIDISequence.h>
#import <InferKit/NFKMusicBeat.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>

/*! One channel-voice event read back out of a written file. */
@interface NFKParsedMIDIEvent : NSObject
@property (nonatomic) NSUInteger trackIndex;
@property (nonatomic) uint32_t tick;
@property (nonatomic) uint8_t status;
@property (nonatomic) uint8_t channel;
@property (nonatomic) uint8_t data1;
@property (nonatomic) uint8_t data2;
@end

@implementation NFKParsedMIDIEvent
@end

/*! A Standard MIDI File reader written against the format rather than against the writer, so a
	round trip measures the file instead of restating how it was produced. */
@interface NFKParsedMIDIFile : NSObject
@property (nonatomic) uint16_t format;
@property (nonatomic) uint16_t trackCount;
@property (nonatomic) uint16_t division;
@property (nonatomic) uint32_t microsecondsPerQuarter;
@property (nonatomic) uint8_t beatsPerBar;
@property (nonatomic) uint8_t beatUnitPower;
@property (nonatomic, strong) NSArray<NFKParsedMIDIEvent *> *events;
+ (instancetype)fileWithData:(NSData *)data;
@end

@implementation NFKParsedMIDIFile

+ (instancetype)fileWithData:(NSData *)data
{
	const uint8_t *bytes = data.bytes;
	NSUInteger length = data.length;
	NSUInteger cursor = 0;
	NFKParsedMIDIFile *file = [[NFKParsedMIDIFile alloc] init];
	NSMutableArray<NFKParsedMIDIEvent *> *events = [NSMutableArray array];

	if (length < 14 || memcmp(bytes, "MThd", 4) != 0) {
		return nil;
	}
	uint32_t headerLength = (uint32_t)((bytes[4] << 24) | (bytes[5] << 16) | (bytes[6] << 8) | bytes[7]);
	file.format = (uint16_t)((bytes[8] << 8) | bytes[9]);
	file.trackCount = (uint16_t)((bytes[10] << 8) | bytes[11]);
	file.division = (uint16_t)((bytes[12] << 8) | bytes[13]);
	cursor = 8 + headerLength;

	NSUInteger trackIndex = 0;
	while (cursor + 8 <= length) {
		if (memcmp(bytes + cursor, "MTrk", 4) != 0) {
			return nil;
		}
		uint32_t trackLength = (uint32_t)((bytes[cursor + 4] << 24) | (bytes[cursor + 5] << 16)
										  | (bytes[cursor + 6] << 8) | bytes[cursor + 7]);
		NSUInteger trackEnd = cursor + 8 + trackLength;
		if (trackEnd > length) {
			return nil;
		}
		NSUInteger position = cursor + 8;
		uint32_t tick = 0;
		uint8_t runningStatus = 0;
		while (position < trackEnd) {
			uint32_t delta = 0;
			uint8_t byte = 0;
			do {
				byte = bytes[position++];
				delta = (delta << 7) | (byte & 0x7F);
			} while ((byte & 0x80) != 0 && position < trackEnd);
			tick += delta;

			uint8_t status = bytes[position];
			if ((status & 0x80) != 0) {
				position += 1;
				if (status < 0xF0) {
					runningStatus = status;
				}
			} else {
				status = runningStatus;
			}

			if (status == 0xFF) {
				uint8_t type = bytes[position++];
				uint32_t metaLength = 0;
				do {
					byte = bytes[position++];
					metaLength = (metaLength << 7) | (byte & 0x7F);
				} while ((byte & 0x80) != 0);
				if (type == 0x51 && metaLength == 3) {
					file.microsecondsPerQuarter = (uint32_t)((bytes[position] << 16)
															 | (bytes[position + 1] << 8) | bytes[position + 2]);
				} else if (type == 0x58 && metaLength == 4) {
					file.beatsPerBar = bytes[position];
					file.beatUnitPower = bytes[position + 1];
				}
				position += metaLength;
				continue;
			}

			uint8_t high = status & 0xF0;
			NSUInteger dataLength = (high == 0xC0 || high == 0xD0) ? 1 : 2;
			NFKParsedMIDIEvent *event = [[NFKParsedMIDIEvent alloc] init];
			event.trackIndex = trackIndex;
			event.tick = tick;
			event.status = high;
			event.channel = status & 0x0F;
			event.data1 = bytes[position];
			event.data2 = dataLength > 1 ? bytes[position + 1] : 0;
			[events addObject:event];
			position += dataLength;
		}
		cursor = trackEnd;
		trackIndex += 1;
	}

	file.events = events;
	return file;
}

@end

@interface NFKMIDISequenceTests : XCTestCase
@end

@implementation NFKMIDISequenceTests

- (NFKMIDINote *)noteWithPitch:(NSInteger)pitch start:(double)start end:(double)end
{
	return [NFKMIDINote noteWithPitch:pitch startSeconds:start endSeconds:end velocity:100];
}

- (void)testNotesSortByStartTime
{
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[
		[self noteWithPitch:64 start:1.0 end:1.5],
		[self noteWithPitch:60 start:0.0 end:0.5],
		[self noteWithPitch:62 start:0.5 end:1.0],
	]];

	XCTAssertEqual(sequence.notes[0].pitch, 60);
	XCTAssertEqual(sequence.notes[1].pitch, 62);
	XCTAssertEqual(sequence.notes[2].pitch, 64);
	XCTAssertEqualWithAccuracy(sequence.durationSeconds, 1.5, 1e-9);
}

- (void)testTheHeaderDeclaresOneTrackPerProgramBesideTheConductor
{
	NFKMIDINote *piano = [self noteWithPitch:60 start:0.0 end:0.5];
	NFKMIDINote *bass = [[NFKMIDINote alloc] initWithPitch:40 startSeconds:0.0 endSeconds:0.5
												  velocity:80 program:33 percussion:NO pitchBend:nil];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ piano, bass ]];

	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];
	XCTAssertNotNil(file);
	XCTAssertEqual(file.format, 1);
	XCTAssertEqual(file.trackCount, 3);
	XCTAssertEqual(file.division, 480);
}

- (void)testTheConductorTrackCarriesTheTempoAndTheTimeSignature
{
	NFKMIDISequence *sequence = [[NFKMIDISequence alloc] initWithNotes:@[ [self noteWithPitch:60 start:0.0 end:0.5] ]
															 tempoBPM:140.0
														  beatsPerBar:3
															 beatUnit:8
												  ticksPerQuarterNote:960];

	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];
	XCTAssertEqual(file.division, 960);
	XCTAssertEqual(file.microsecondsPerQuarter, (uint32_t)llround(60000000.0 / 140.0));
	XCTAssertEqual(file.beatsPerBar, 3);
	XCTAssertEqual(file.beatUnitPower, 3);
}

- (void)testTheNotesReadBackAtTheTimesTheyWereWrittenAt
{
	NSArray<NFKMIDINote *> *notes = @[
		[self noteWithPitch:60 start:0.0 end:0.5],
		[self noteWithPitch:67 start:0.25 end:1.75],
		[self noteWithPitch:72 start:2.0 end:2.125],
	];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:notes tempoBPM:120.0];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	double secondsPerTick = 60.0 / (120.0 * 480.0);
	NSMutableDictionary<NSNumber *, NSNumber *> *onsets = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSNumber *, NSNumber *> *offsets = [NSMutableDictionary dictionary];
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0x90) {
			onsets[@(event.data1)] = @(event.tick * secondsPerTick);
			XCTAssertEqual(event.data2, 100);
		} else if (event.status == 0x80) {
			offsets[@(event.data1)] = @(event.tick * secondsPerTick);
		}
	}

	XCTAssertEqual(onsets.count, 3u);
	for (NFKMIDINote *note in notes) {
		XCTAssertEqualWithAccuracy(onsets[@(note.pitch)].doubleValue, note.startSeconds, secondsPerTick);
		XCTAssertEqualWithAccuracy(offsets[@(note.pitch)].doubleValue, note.endSeconds, secondsPerTick);
	}
}

- (void)testAZeroLengthNoteStillSounds
{
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ [self noteWithPitch:60 start:1.0 end:1.0] ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	uint32_t onTick = 0;
	uint32_t offTick = 0;
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0x90) {
			onTick = event.tick;
		} else if (event.status == 0x80) {
			offTick = event.tick;
		}
	}
	XCTAssertGreaterThan(offTick, onTick);
}

- (void)testPercussionTakesChannelTenAndSkipsTheProgramChange
{
	NFKMIDINote *kick = [[NFKMIDINote alloc] initWithPitch:36 startSeconds:0.0 endSeconds:0.1
												  velocity:110 program:0 percussion:YES pitchBend:nil];
	NFKMIDINote *piano = [self noteWithPitch:60 start:0.0 end:0.5];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ kick, piano ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	BOOL sawPercussionNote = NO;
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0x90 && event.data1 == 36) {
			XCTAssertEqual(event.channel, 9);
			sawPercussionNote = YES;
		}
		if (event.status == 0xC0) {
			XCTAssertNotEqual(event.channel, 9);
		}
		if (event.status == 0x90 && event.data1 == 60) {
			XCTAssertNotEqual(event.channel, 9);
		}
	}
	XCTAssertTrue(sawPercussionNote);
}

- (void)testTheProgramChangeNamesTheTracksInstrument
{
	NFKMIDINote *bass = [[NFKMIDINote alloc] initWithPitch:40 startSeconds:0.0 endSeconds:0.5
												  velocity:80 program:33 percussion:NO pitchBend:nil];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ bass ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	NSUInteger programChanges = 0;
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0xC0) {
			XCTAssertEqual(event.data1, 33);
			programChanges += 1;
		}
	}
	XCTAssertEqual(programChanges, 1u);
}

- (void)testAPitchBendWritesTheWheelAcrossTheNoteAndRecentersAtItsEnd
{
	NFKMIDINote *bent = [[NFKMIDINote alloc] initWithPitch:60 startSeconds:0.0 endSeconds:1.0
												  velocity:100 program:0 percussion:NO
												 pitchBend:@[ @0.0, @0.5, @1.0 ]];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ bent ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	NSMutableArray<NSNumber *> *wheel = [NSMutableArray array];
	NSMutableArray<NSNumber *> *ticks = [NSMutableArray array];
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status != 0xE0) {
			continue;
		}
		[wheel addObject:@((event.data2 << 7 | event.data1) - 8192)];
		[ticks addObject:@(event.tick)];
	}

	// Three samples across the note, then the recentering the wheel's channel scope needs.
	XCTAssertEqual(wheel.count, 4u);
	XCTAssertEqual(wheel[0].intValue, 0);
	XCTAssertEqual(wheel[1].intValue, 2048);
	XCTAssertEqual(wheel[2].intValue, 4096);
	XCTAssertEqual(wheel[3].intValue, 0);
	// One second at 120 BPM is two quarter notes, so the note spans 960 ticks.
	XCTAssertEqual(ticks[0].unsignedIntValue, 0u);
	XCTAssertEqual(ticks[1].unsignedIntValue, 480u);
	XCTAssertEqual(ticks[2].unsignedIntValue, 960u);
}

- (void)testAPitchBendClampsToTheWheelsRange
{
	NFKMIDINote *bent = [[NFKMIDINote alloc] initWithPitch:60 startSeconds:0.0 endSeconds:1.0
												  velocity:100 program:0 percussion:NO
												 pitchBend:@[ @(-4.0), @4.0 ]];
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ bent ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	NSMutableArray<NSNumber *> *wheel = [NSMutableArray array];
	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0xE0) {
			[wheel addObject:@((event.data2 << 7 | event.data1) - 8192)];
		}
	}
	XCTAssertEqual(wheel[0].intValue, -8192);
	XCTAssertEqual(wheel[1].intValue, 8191);
}

- (void)testDeltaTimesSpanMoreThanOneVariableLengthByte
{
	// 20 seconds at 120 BPM is 19200 ticks, which needs three bytes of variable-length quantity.
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ [self noteWithPitch:60 start:20.0 end:20.5] ]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	for (NFKParsedMIDIEvent *event in file.events) {
		if (event.status == 0x90) {
			XCTAssertEqual(event.tick, 19200u);
		}
	}
}

- (void)testAnEmptySequenceWritesAPlayableFile
{
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[]];
	NFKParsedMIDIFile *file = [NFKParsedMIDIFile fileWithData:sequence.standardMIDIFileData];

	XCTAssertNotNil(file);
	XCTAssertEqual(file.trackCount, 1);
	XCTAssertEqual(file.events.count, 0u);
	XCTAssertEqualWithAccuracy(sequence.durationSeconds, 0.0, 1e-12);
}

- (void)testTheSequenceWritesToAURL
{
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ [self noteWithPitch:60 start:0.0 end:0.5] ]];
	NSString *name = [NSString stringWithFormat:@"inferkit-test-%@.mid", NSUUID.UUID.UUIDString];
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];

	NSError *error = nil;
	XCTAssertTrue([sequence writeToURL:url error:&error]);
	XCTAssertNil(error);
	NSData *written = [NSData dataWithContentsOfURL:url];
	XCTAssertEqualObjects(written, sequence.standardMIDIFileData);
	[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
}

- (void)testABeatKnowsWhetherItStartsABar
{
	NFKMusicBeat *downbeat = [NFKMusicBeat beatWithTimeSeconds:1.5 positionInBar:1];
	NFKMusicBeat *offbeat = [NFKMusicBeat beatWithTimeSeconds:2.0 positionInBar:2];
	NFKMusicBeat *unmetered = [NFKMusicBeat beatWithTimeSeconds:2.5 positionInBar:0];

	XCTAssertTrue(downbeat.isDownbeat);
	XCTAssertFalse(offbeat.isDownbeat);
	XCTAssertFalse(unmetered.isDownbeat);
	XCTAssertEqualObjects(downbeat, [NFKMusicBeat beatWithTimeSeconds:1.5 positionInBar:1]);
	XCTAssertNotEqualObjects(downbeat, offbeat);
}

- (void)testTheResultReadsTheMusicOutputsByType
{
	NFKMIDISequence *sequence = [NFKMIDISequence sequenceWithNotes:@[ [self noteWithPitch:60 start:0.0 end:0.5] ]];
	NSArray<NFKMusicBeat *> *beats = @[ [NFKMusicBeat beatWithTimeSeconds:0.0 positionInBar:1] ];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputMIDI: sequence,
		NFKOutputBeats: beats,
		NFKOutputTempo: @128.0,
	}];

	XCTAssertEqualObjects(result.midi, sequence);
	XCTAssertEqualObjects(result.beats, beats);
	XCTAssertEqualWithAccuracy([[result outputForKey:NFKOutputTempo] doubleValue], 128.0, 1e-12);

	NFKInferenceResult *textOnly = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"no music here" }];
	XCTAssertNil(textOnly.midi);
	XCTAssertNil(textOnly.beats);
}

#pragma mark Archiving

- (void)testANoteSurvivesASecureRoundTrip
{
	NFKMIDINote *note = [[NFKMIDINote alloc] initWithPitch:60
											  startSeconds:0.5
												endSeconds:1.25
												  velocity:96
												   program:42
												percussion:YES
												 pitchBend:@[ @0.0, @0.5 ]];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:note requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKMIDINote *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKMIDINote.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.pitch, 60);
	XCTAssertEqual(read.startSeconds, 0.5);
	XCTAssertEqual(read.endSeconds, 1.25);
	XCTAssertEqual(read.velocity, 96);
	XCTAssertEqual(read.program, 42);
	XCTAssertTrue(read.isPercussion);
	XCTAssertEqualObjects(read.pitchBend, (@[ @0.0, @0.5 ]));
	XCTAssertEqualObjects(read, note);
}

- (void)testANoteWithoutAPitchBendSurvivesASecureRoundTrip
{
	NFKMIDINote *note = [NFKMIDINote noteWithPitch:64 startSeconds:0 endSeconds:0.5 velocity:80];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:note requiringSecureCoding:YES error:NULL];
	NFKMIDINote *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKMIDINote.class fromData:data error:NULL];
	XCTAssertNil(read.pitchBend);
	XCTAssertEqualObjects(read, note);
}

- (void)testASequenceCarriesItsNotesThroughASecureRoundTrip
{
	// The notes are objects inside the archive, so this is the case that fails when a decode names
	// the array's class and not the element's.
	NFKMIDISequence *sequence =
		[[NFKMIDISequence alloc] initWithNotes:@[ [NFKMIDINote noteWithPitch:60 startSeconds:0 endSeconds:1 velocity:90],
												  [NFKMIDINote noteWithPitch:67 startSeconds:1 endSeconds:2 velocity:70] ]
									  tempoBPM:96
								   beatsPerBar:3
									  beatUnit:8
						   ticksPerQuarterNote:960];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:sequence requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKMIDISequence *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKMIDISequence.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.notes.count, (NSUInteger)2);
	XCTAssertEqual(read.notes.firstObject.pitch, 60);
	XCTAssertEqual(read.notes.lastObject.pitch, 67);
	XCTAssertEqual(read.tempoBPM, 96);
	XCTAssertEqual(read.beatsPerBar, 3);
	XCTAssertEqual(read.beatUnit, 8);
	XCTAssertEqual(read.ticksPerQuarterNote, 960);
	XCTAssertEqualObjects(read.notes, sequence.notes);
	XCTAssertEqualObjects(read, sequence);
	XCTAssertEqualObjects([read standardMIDIFileData], [sequence standardMIDIFileData],
						  @"a sequence read back writes the same file");
}

- (void)testABeatSurvivesASecureRoundTrip
{
	NFKMusicBeat *downbeat = [NFKMusicBeat beatWithTimeSeconds:2.5 positionInBar:1];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:downbeat requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKMusicBeat *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKMusicBeat.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.timeSeconds, 2.5);
	XCTAssertEqual(read.positionInBar, 1);
	XCTAssertTrue(read.isDownbeat, @"the position carries it, so nothing extra is archived");
	XCTAssertEqualObjects(read, downbeat);
}

- (void)testATracksBeatsArchiveAsOneArray
{
	NSArray<NFKMusicBeat *> *beats = @[ [NFKMusicBeat beatWithTimeSeconds:0.5 positionInBar:1],
										[NFKMusicBeat beatWithTimeSeconds:1.0 positionInBar:2] ];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:beats requiringSecureCoding:YES error:NULL];
	NSSet *classes = [NSSet setWithObjects:NSArray.class, NFKMusicBeat.class, nil];
	XCTAssertEqualObjects([NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:data error:NULL], beats);
}

- (void)testSequenceEqualityAndCopy
{
	NSArray<NFKMIDINote *> *notes = @[ [NFKMIDINote noteWithPitch:60 startSeconds:0 endSeconds:1 velocity:90] ];
	NFKMIDISequence *a = [NFKMIDISequence sequenceWithNotes:notes tempoBPM:120];
	NFKMIDISequence *b = [NFKMIDISequence sequenceWithNotes:notes tempoBPM:120];
	NFKMIDISequence *faster = [NFKMIDISequence sequenceWithNotes:notes tempoBPM:140];
	NFKMIDISequence *other = [NFKMIDISequence sequenceWithNotes:@[] tempoBPM:120];

	XCTAssertEqualObjects(a, b, @"the notes and the meter carry it, not the pointer");
	XCTAssertEqual(a.hash, b.hash);
	XCTAssertNotEqualObjects(a, faster, @"a different tempo is a different sequence");
	XCTAssertNotEqualObjects(a, other, @"different notes are a different sequence");
	XCTAssertEqual([a copy], a, @"immutable value copies to itself");
}

#pragma mark Reading a file

- (void)testAWrittenFileReadsBackAsTheSequence
{
	NFKMIDINote *piano = [[NFKMIDINote alloc] initWithPitch:60 startSeconds:0.25 endSeconds:1.0
												   velocity:90 program:0 percussion:NO pitchBend:nil];
	NFKMIDINote *bass = [[NFKMIDINote alloc] initWithPitch:40 startSeconds:0.5 endSeconds:2.0
												  velocity:70 program:33 percussion:NO pitchBend:nil];
	NFKMIDINote *snare = [[NFKMIDINote alloc] initWithPitch:38 startSeconds:1.0 endSeconds:1.125
												   velocity:110 program:0 percussion:YES pitchBend:nil];
	NFKMIDISequence *written = [[NFKMIDISequence alloc] initWithNotes:@[ piano, bass, snare ] tempoBPM:90 beatsPerBar:3
															 beatUnit:4 ticksPerQuarterNote:960];
	NSError *error = nil;
	NFKMIDISequence *read = [NFKMIDISequence sequenceWithStandardMIDIFileData:[written standardMIDIFileData] error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualWithAccuracy(read.tempoBPM, 90, 1e-3);
	XCTAssertEqual(read.beatsPerBar, 3);
	XCTAssertEqual(read.beatUnit, 4);
	XCTAssertEqual(read.ticksPerQuarterNote, 960);
	XCTAssertEqual(read.notes.count, 3u);
	// One tick at 90 BPM and 960 ticks per quarter note is 1/1440 of a second.
	double tick = 60.0 / 90.0 / 960.0;
	for (NSUInteger index = 0; index < 3; index++) {
		NFKMIDINote *expected = written.notes[index];
		NFKMIDINote *actual = read.notes[index];
		XCTAssertEqual(actual.pitch, expected.pitch);
		XCTAssertEqual(actual.velocity, expected.velocity);
		XCTAssertEqual(actual.program, expected.isPercussion ? 0 : expected.program);
		XCTAssertEqual(actual.isPercussion, expected.isPercussion);
		XCTAssertEqualWithAccuracy(actual.startSeconds, expected.startSeconds, tick);
		XCTAssertEqualWithAccuracy(actual.endSeconds, expected.endSeconds, tick);
		XCTAssertNil(actual.pitchBend);
	}
}

/*! A format-0 file of one track with tempo, running status, and a note-on of velocity 0 as its end. */
- (NSData *)runningStatusFile
{
	const uint8_t bytes[] = {
		'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 0, 0, 1, 0x01, 0xE0,
		'M', 'T', 'r', 'k', 0, 0, 0, 28,
		0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20,       // 500,000 microseconds per quarter: 120 BPM
		0x00, 0xC0, 0x05,                               // program 5 on channel 1
		0x00, 0x90, 0x3C, 0x64,                         // note on 60
		0x00, 0x40, 0x50,                               // running status: note on 64
		0x83, 0x60, 0x3C, 0x00,                         // 480 ticks later, note 60 off by velocity 0
		0x00, 0x40, 0x00,                               // note 64 off
		0x00, 0xFF, 0x2F, 0x00,
	};
	return [NSData dataWithBytes:bytes length:sizeof(bytes)];
}

- (void)testAFileWithRunningStatusReadsEveryNote
{
	NSError *error = nil;
	NFKMIDISequence *read = [NFKMIDISequence sequenceWithStandardMIDIFileData:[self runningStatusFile] error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.notes.count, 2u);
	XCTAssertEqual(read.notes[0].pitch, 60);
	XCTAssertEqual(read.notes[1].pitch, 64);
	XCTAssertEqual(read.notes[1].velocity, 80);
	XCTAssertEqual(read.notes[0].program, 5, @"a note takes the program its channel changed to before it");
	XCTAssertEqualWithAccuracy(read.notes[0].endSeconds, 0.5, 1e-9, @"480 ticks at 480 per quarter and 120 BPM");
	XCTAssertEqualWithAccuracy(read.tempoBPM, 120, 1e-9);
}

- (void)testATempoChangeMovesTheNotesAfterIt
{
	NSMutableData *file = [[self runningStatusFile] mutableCopy];
	// One note whose tempo falls from the default 120 BPM to 60 BPM halfway through it.
	const uint8_t track[] = {
		'M', 'T', 'r', 'k', 0, 0, 0, 21,
		0x00, 0x90, 0x3C, 0x64,                         // note on 60 at tick 0
		0x81, 0x70, 0xFF, 0x51, 0x03, 0x0F, 0x42, 0x40, // tick 240: 1,000,000 microseconds per quarter
		0x81, 0x70, 0x80, 0x3C, 0x00,                   // tick 480: note off
		0x00, 0xFF, 0x2F, 0x00,
	};
	[file replaceBytesInRange:NSMakeRange(14, file.length - 14) withBytes:track length:sizeof(track)];
	NSError *error = nil;
	NFKMIDISequence *read = [NFKMIDISequence sequenceWithStandardMIDIFileData:file error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.notes.count, 1u);
	XCTAssertEqualWithAccuracy(read.notes[0].endSeconds, 0.25 + 0.5, 1e-9, @"half a beat at 120 BPM, then half a beat at 60 BPM");
	XCTAssertEqualWithAccuracy(read.tempoBPM, 60, 1e-9, @"the sequence takes the file's first tempo");
}

- (void)testANoteTheFileNeverEndsEndsWithItsTrack
{
	const uint8_t bytes[] = {
		'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 0, 0, 1, 0x01, 0xE0,
		'M', 'T', 'r', 'k', 0, 0, 0, 12,
		0x00, 0x99, 0x24, 0x7F,                         // a kick on channel 10
		0x83, 0x60, 0xFF, 0x2F, 0x00,                   // the track ends 480 ticks later
		0x00, 0x00, 0x00,
	};
	NSError *error = nil;
	NFKMIDISequence *read = [NFKMIDISequence sequenceWithStandardMIDIFileData:[NSData dataWithBytes:bytes length:sizeof(bytes)] error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.notes.count, 1u);
	XCTAssertTrue(read.notes[0].isPercussion);
	XCTAssertEqualWithAccuracy(read.notes[0].endSeconds, 0.5, 1e-9);
}

- (void)testAFileThatIsNotOnePerformanceIsRefused
{
	NSMutableData *formatTwo = [[self runningStatusFile] mutableCopy];
	((uint8_t *)formatTwo.mutableBytes)[9] = 2;
	NSMutableData *frames = [[self runningStatusFile] mutableCopy];
	((uint8_t *)frames.mutableBytes)[12] = 0xE7;
	NSData *cut = [[self runningStatusFile] subdataWithRange:NSMakeRange(0, 30)];
	for (NSData *file in @[ formatTwo, frames, cut, [@"not midi" dataUsingEncoding:NSUTF8StringEncoding] ]) {
		NSError *error = nil;
		XCTAssertNil([NFKMIDISequence sequenceWithStandardMIDIFileData:file error:&error]);
		XCTAssertEqualObjects(error.domain, NSCocoaErrorDomain);
		XCTAssertEqual(error.code, NSFileReadCorruptFileError, @"%@", error);
	}
}
@end
