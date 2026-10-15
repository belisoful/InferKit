//
//  NFKPluginChannelTests.m
//  InferKitTests
//
//  The plugin channel's framing: messages, a frame at the size limit, a frame full of descriptors, the
//  violations that close it, its end when the peer exits, the hand-off, and the peer's identity.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "../../Sources/InferKit/NFKPluginChannel.h"
#import "../../Sources/InferKit/NFKPluginSharedMemory.h"
#import "../../Sources/InferKit/NFKPluginWire.h"
#import "../../Sources/InferKit/NFKSubprocessPrivate.h"
#import <sys/socket.h>
#import <unistd.h>

/*! Collects what a channel delivers. */
@interface NFKChannelRecorder : NSObject
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *messages;
@property (nonatomic, strong) NSMutableArray<NSArray<NFKPluginDescriptor *> *> *descriptors;
@property (nonatomic, strong, nullable) NSError *closeError;
@property (nonatomic, assign) BOOL closed;
@end

@implementation NFKChannelRecorder

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_messages = [NSMutableArray array];
		_descriptors = [NSMutableArray array];
	}
	return self;
}

- (void)attachToChannel:(NFKPluginChannel *)channel
{
	channel.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
		@synchronized (self) {
			[self.messages addObject:message];
			[self.descriptors addObject:descriptors];
		}
	};
	channel.closeHandler = ^(NSError *error) {
		@synchronized (self) {
			self.closeError = error;
			self.closed = YES;
		}
	};
}

- (NSUInteger)messageCount
{
	@synchronized (self) {
		return self.messages.count;
	}
}

- (BOOL)isClosed
{
	@synchronized (self) {
		return self.closed;
	}
}

@end

@interface NFKPluginChannelTests : XCTestCase
@end

@implementation NFKPluginChannelTests

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow < 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return YES;
}

- (void)makeChannel:(NFKPluginChannel * _Nonnull * _Nonnull)first other:(NFKPluginChannel * _Nonnull * _Nonnull)second
{
	int one = -1;
	int two = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&one second:&two error:&error], @"%@", error);
	*first = [[NFKPluginChannel alloc] initWithDescriptor:one];
	*second = [[NFKPluginChannel alloc] initWithDescriptor:two];
}

/*! A channel on one end of a socket pair, and the raw other end for writing malformed bytes. */
- (NFKPluginChannel *)channelWithRawPeer:(int *)rawPeer
{
	int one = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&one second:rawPeer error:&error], @"%@", error);
	return [[NFKPluginChannel alloc] initWithDescriptor:one];
}

- (NSData *)headerWithMagic:(uint32_t)magic length:(uint32_t)length descriptors:(uint16_t)count
{
	uint8_t header[12] = {
		(uint8_t)magic, (uint8_t)(magic >> 8), (uint8_t)(magic >> 16), (uint8_t)(magic >> 24),
		(uint8_t)length, (uint8_t)(length >> 8), (uint8_t)(length >> 16), (uint8_t)(length >> 24),
		(uint8_t)count, (uint8_t)(count >> 8), 0, 0,
	};
	return [NSData dataWithBytes:header length:sizeof header];
}

- (void)testAMessageRoundTripsWithItsFields
{
	NFKPluginChannel *host = nil;
	NFKPluginChannel *plugin = nil;
	[self makeChannel:&host other:&plugin];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:plugin];
	[plugin resume];

	NSDictionary *first = @{ @"type": @"hello", @"id": @1, @"protocol": @{ @"major": @1, @"minor": @0 }, @"capabilities": @[ @"attachments" ] };
	NSDictionary *second = @{ @"type": @"ping", @"id": @2 };
	NSError *error = nil;
	XCTAssertTrue([host sendMessage:first error:&error], @"%@", error);
	XCTAssertTrue([host sendMessage:second error:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.messageCount == 2; } timeout:5]);
	XCTAssertEqualObjects(recorder.messages[0], first);
	XCTAssertEqualObjects(recorder.messages[1], second, @"messages arrive in the order sent");
	XCTAssertEqual(recorder.descriptors[0].count, 0u);

	NSDictionary *untyped = @{ @"id": @3 };
	XCTAssertFalse([host sendMessage:untyped error:&error], @"a message without a type is refused before it is sent");
	XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);
	[host close];
	[plugin close];
}

- (void)testASixteenMebibyteFrameArrivesWhole
{
	NFKPluginChannel *host = nil;
	NFKPluginChannel *plugin = nil;
	[self makeChannel:&host other:&plugin];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:plugin];
	[plugin resume];

	NSData *envelope = [NSJSONSerialization dataWithJSONObject:@{ @"type": @"blob", @"text": @"" } options:0 error:NULL];
	NSUInteger textLength = NFKPluginFrameMaximumJSONLength - envelope.length;
	NSMutableData *bytes = [NSMutableData dataWithLength:textLength];
	memset(bytes.mutableBytes, 'x', textLength);
	((char *)bytes.mutableBytes)[textLength - 1] = 'y';
	NSString *text = [[NSString alloc] initWithData:bytes encoding:NSASCIIStringEncoding];
	NSDictionary *message = @{ @"type": @"blob", @"text": text };
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:message options:0 error:NULL];
	XCTAssertEqual(JSON.length, (NSUInteger)NFKPluginFrameMaximumJSONLength, @"the frame is exactly at the limit");

	NSError *error = nil;
	XCTAssertTrue([host sendMessage:message error:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.messageCount == 1; } timeout:30]);
	NSString *received = recorder.messages.firstObject[@"text"];
	XCTAssertEqual(received.length, textLength);
	XCTAssertTrue([received hasSuffix:@"xy"]);

	NSDictionary *oversized = @{ @"type": @"blob", @"text": [text stringByAppendingString:@"z"] };
	XCTAssertFalse([host sendMessage:oversized error:&error], @"one byte over the limit is refused");
	XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);
	[host close];
	[plugin close];
}

- (void)testSixtyFourDescriptorsArriveIntact
{
	NFKPluginChannel *host = nil;
	NFKPluginChannel *plugin = nil;
	[self makeChannel:&host other:&plugin];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:plugin];
	[plugin resume];

	NSMutableArray<NFKPluginSharedMemory *> *regions = [NSMutableArray array];
	NSMutableArray<NSNumber *> *descriptors = [NSMutableArray array];
	for (NSUInteger index = 0; index < 64; index++) {
		NSError *error = nil;
		NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithLength:4096 + index namePrefix:nil error:&error];
		XCTAssertNotNil(region, @"%@", error);
		memset(region.mutableBytes, (int)index + 1, region.length);
		[regions addObject:region];
		[descriptors addObject:@(region.descriptor)];
	}
	NSError *error = nil;
	NSDictionary *run = @{ @"type": @"run", @"id": @7 };
	XCTAssertTrue([host sendMessage:run descriptors:descriptors error:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.messageCount == 1; } timeout:5]);
	NSArray<NFKPluginDescriptor *> *received = recorder.descriptors.firstObject;
	XCTAssertEqual(received.count, 64u);
	for (NSUInteger index = 0; index < received.count; index++) {
		NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:[received[index] takeDescriptor] copiesFiles:YES error:&error];
		XCTAssertNotNil(region, @"%@", error);
		XCTAssertEqual(region.length, 4096 + index, @"descriptor %lu names its own region", (unsigned long)index);
		XCTAssertEqual(((const uint8_t *)region.mutableBytes)[region.length - 1], (uint8_t)(index + 1));
	}

	NSMutableArray<NSNumber *> *tooMany = [descriptors mutableCopy];
	[tooMany addObject:descriptors.firstObject];
	XCTAssertFalse([host sendMessage:run descriptors:tooMany error:&error], @"65 descriptors are refused");
	XCTAssertEqual(error.code, NFKPluginErrorProtocolViolation);
	[host close];
	[plugin close];
}

- (void)testABadMagicNumberClosesTheChannel
{
	int raw = -1;
	NFKPluginChannel *channel = [self channelWithRawPeer:&raw];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:channel];
	[channel resume];

	NSMutableData *frame = [[self headerWithMagic:0x50544548 length:2 descriptors:0] mutableCopy];
	[frame appendData:[@"{}" dataUsingEncoding:NSUTF8StringEncoding]];
	XCTAssertEqual(write(raw, frame.bytes, frame.length), (ssize_t)frame.length);
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.isClosed; } timeout:5]);
	XCTAssertEqualObjects(recorder.closeError.domain, NFKPluginErrorDomain);
	XCTAssertEqual(recorder.closeError.code, NFKPluginErrorProtocolViolation);
	XCTAssertEqual(recorder.messageCount, 0u);
	XCTAssertTrue(channel.isClosed);
	NSError *error = nil;
	NSDictionary *ping = @{ @"type": @"ping" };
	XCTAssertFalse([channel sendMessage:ping error:&error], @"a closed channel sends nothing");
	XCTAssertEqual(error.code, NFKPluginErrorStopped);
	close(raw);
}

- (void)testAFrameMissingItsDescriptorClosesTheChannel
{
	int raw = -1;
	NFKPluginChannel *channel = [self channelWithRawPeer:&raw];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:channel];
	[channel resume];

	NSData *JSON = [@"{\"type\":\"run\",\"id\":1}" dataUsingEncoding:NSUTF8StringEncoding];
	NSMutableData *frame = [[self headerWithMagic:NFKPluginFrameMagic length:(uint32_t)JSON.length descriptors:1] mutableCopy];
	[frame appendData:JSON];
	XCTAssertEqual(write(raw, frame.bytes, frame.length), (ssize_t)frame.length, @"a plain write carries no descriptor");
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.isClosed; } timeout:5]);
	XCTAssertEqual(recorder.closeError.code, NFKPluginErrorProtocolViolation);
	XCTAssertTrue([recorder.closeError.localizedDescription containsString:@"descriptor"], @"%@", recorder.closeError);
	XCTAssertEqual(recorder.messageCount, 0u);
	close(raw);
}

- (void)testAPeerThatExitsEndsTheChannelAtEndOfFile
{
	int ours = -1;
	int child = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&ours second:&child error:&error], @"%@", error);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"sleep 0.2" ]];
	process.ownerInheritedDescriptors = @{ @(NFKPluginChannelDescriptorNumber): @(child) };
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	close(child);

	NFKPluginChannel *channel = [[NFKPluginChannel alloc] initWithDescriptor:ours];
	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:channel];
	[channel resume];
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.isClosed; } timeout:10], @"the child's exit closes its end");
	XCTAssertNil(recorder.closeError, @"end of file is an orderly end");
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:5]);
}

- (void)testThePeerAuditTokenNamesTheProcessThatCreatedTheSocket
{
	NFKPluginChannel *host = nil;
	NFKPluginChannel *plugin = nil;
	[self makeChannel:&host other:&plugin];
	audit_token_t token;
	NSError *error = nil;
	XCTAssertTrue([host getPeerAuditToken:&token error:&error], @"%@", error);
	XCTAssertEqual(host.peerProcessIdentifier, getpid());
	[host close];
	[plugin close];

	// An inherited end still names its creator, this process, which is why the plugin hands over a
	// socket of its own.
	int ours = -1;
	int child = -1;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&ours second:&child error:&error], @"%@", error);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sleep" arguments:@[ @"5" ]];
	process.ownerInheritedDescriptors = @{ @(NFKPluginChannelDescriptorNumber): @(child) };
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	close(child);
	NFKPluginChannel *inherited = [[NFKPluginChannel alloc] initWithDescriptor:ours];
	XCTAssertNotEqual(process.processIdentifier, 0);
	XCTAssertEqual(inherited.peerProcessIdentifier, getpid());
	[inherited close];
	[process stop];
}

- (void)testTheHandOffMovesTheConversationToTheHandedSocket
{
	int hostEnd = -1;
	int pluginEnd = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&hostEnd second:&pluginEnd error:&error], @"%@", error);
	__block NFKPluginChannel *plugin = nil;
	__block NSError *handOffError = nil;
	dispatch_semaphore_t handedOff = dispatch_semaphore_create(0);
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *failure = nil;
		plugin = [NFKPluginChannel channelByHandingOffDescriptor:pluginEnd error:&failure];
		handOffError = failure;
		dispatch_semaphore_signal(handedOff);
	});
	NFKPluginChannel *host = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:5 error:&error];
	XCTAssertEqual(dispatch_semaphore_wait(handedOff, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)), 0);
	XCTAssertNotNil(plugin, @"%@", handOffError);
	XCTAssertNotNil(host, @"%@", error);
	XCTAssertEqual(host.peerProcessIdentifier, getpid(), @"the handed socket names the process that created it");

	NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
	[recorder attachToChannel:plugin];
	[plugin resume];
	NSDictionary *ping = @{ @"type": @"ping", @"id": @1 };
	XCTAssertTrue([host sendMessage:ping error:&error], @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.messageCount == 1; } timeout:5]);
	XCTAssertEqualObjects(recorder.messages.firstObject[@"type"], @"ping");
	[host close];
	XCTAssertTrue([self waitUntil:^BOOL { return recorder.isClosed; } timeout:5]);
	XCTAssertNil(recorder.closeError);
	[plugin close];
}

- (void)testAHandOffThatNeverComesTimesOut
{
	int hostEnd = -1;
	int pluginEnd = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&hostEnd second:&pluginEnd error:&error], @"%@", error);
	NFKPluginChannel *host = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:0.3 error:&error];
	XCTAssertNil(host);
	XCTAssertEqual(error.code, NFKPluginErrorHandshakeFailed);
	close(pluginEnd);
}

@end

#endif
