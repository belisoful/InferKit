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
#import <mach/vm_page_size.h>
#import <poll.h>
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

/*! Writes a frame declaring declared descriptors to a raw socket in one sendmsg that carries every
	descriptor of attached. Returns whether the whole frame went out. */
- (BOOL)writeFrameTo:(int)socket declaring:(uint16_t)declared attaching:(NSArray<NSNumber *> *)attached
{
	NSData *JSON = [@"{\"type\":\"ping\",\"id\":1}" dataUsingEncoding:NSUTF8StringEncoding];
	NSMutableData *frame = [[self headerWithMagic:NFKPluginFrameMagic length:(uint32_t)JSON.length descriptors:declared] mutableCopy];
	[frame appendData:JSON];
	NSMutableData *control = [NSMutableData dataWithLength:CMSG_SPACE(sizeof(int) * attached.count)];
	struct cmsghdr *header = (struct cmsghdr *)control.mutableBytes;
	header->cmsg_level = SOL_SOCKET;
	header->cmsg_type = SCM_RIGHTS;
	header->cmsg_len = (socklen_t)CMSG_LEN(sizeof(int) * attached.count);
	for (NSUInteger index = 0; index < attached.count; index++) {
		((int *)CMSG_DATA(header))[index] = attached[index].intValue;
	}
	struct iovec piece = { .iov_base = frame.mutableBytes, .iov_len = frame.length };
	struct msghdr message = { 0 };
	message.msg_iov = &piece;
	message.msg_iovlen = 1;
	message.msg_control = control.mutableBytes;
	message.msg_controllen = (socklen_t)control.length;
	return sendmsg(socket, &message, 0) == (ssize_t)frame.length;
}

/*! Whether descriptor, the read end of a pipe, reads end of file within timeout: every write end closed. */
- (BOOL)pipeReadsEndOfFile:(int)descriptor within:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (deadline.timeIntervalSinceNow > 0) {
		struct pollfd readable = { .fd = descriptor, .events = POLLIN, .revents = 0 };
		if (poll(&readable, 1, 50) > 0) {
			char byte;
			return read(descriptor, &byte, 1) == 0;
		}
	}
	return NO;
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
		// The frame carries no lengths, so the test states each region's, as an attachment entry would.
		NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:[received[index] takeDescriptor] length:4096 + index
																		copiesFiles:YES error:&error];
		XCTAssertNotNil(region, @"%@", error);
		if (region == nil) {
			continue;
		}
		XCTAssertEqual(region.length, 4096 + index);
		XCTAssertEqual(region.mappedLength % vm_page_size, 0u, @"the object is whole pages");
		XCTAssertEqual(((const uint8_t *)region.mutableBytes)[0], (uint8_t)(index + 1), @"descriptor %lu names its own region", (unsigned long)index);
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

- (void)testDescriptorsBeyondAFramesCountCloseTheChannelAndAreClosed
{
	NSArray<NSNumber *> *declared = @[ @0, @1 ];
	for (NSNumber *count in declared) {
		int raw = -1;
		NFKPluginChannel *channel = [self channelWithRawPeer:&raw];
		NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
		[recorder attachToChannel:channel];
		[channel resume];

		int first[2] = { -1, -1 };
		int second[2] = { -1, -1 };
		XCTAssertEqual(pipe(first), 0);
		XCTAssertEqual(pipe(second), 0);
		NSArray<NSNumber *> *attached = @[ @(first[1]), @(second[1]) ];
		XCTAssertTrue([self writeFrameTo:raw declaring:count.unsignedShortValue attaching:attached]);
		close(first[1]);
		close(second[1]);

		XCTAssertTrue([self waitUntil:^BOOL { return recorder.isClosed; } timeout:5], @"a frame declaring %@ of 2", count);
		XCTAssertEqualObjects(recorder.closeError.domain, NFKPluginErrorDomain);
		XCTAssertEqual(recorder.closeError.code, NFKPluginErrorProtocolViolation);
		XCTAssertTrue([recorder.closeError.localizedDescription containsString:@"descriptors"], @"%@", recorder.closeError);
		XCTAssertEqual(recorder.messageCount, 0u, @"the frame is not delivered");
		XCTAssertTrue([self pipeReadsEndOfFile:first[0] within:5], @"the channel closed the descriptors it received");
		XCTAssertTrue([self pipeReadsEndOfFile:second[0] within:5]);
		close(first[0]);
		close(second[0]);
		close(raw);
	}
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
	process.ownerInheritedDescriptors = nil;
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

/*! Eight channel pairs make 200 round trips each, every request carrying two fresh regions and every
	reply one, while other queues create and drop regions and launch subprocesses that inherit a
	descriptor. A descriptor closed by anything but its owner closes or corrupts some channel here. */
- (void)testManyChannelsKeepAnsweringWhileRegionsAndLaunchesChurn
{
	const NSUInteger pairCount = 8;
	const NSInteger tripCount = 200;
	const NSUInteger regionLength = 5000;
	NSMutableArray<NFKPluginChannel *> *channels = [NSMutableArray array];
	NSMutableArray<NSString *> *failures = [NSMutableArray array];
	dispatch_group_t trips = dispatch_group_create();
	for (NSUInteger pair = 0; pair < pairCount; pair++) {
		NFKPluginChannel *host = nil;
		NFKPluginChannel *plugin = nil;
		[self makeChannel:&host other:&plugin];
		__weak NFKPluginChannel *weakPlugin = plugin;
		plugin.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
			NSUInteger mapped = 0;
			for (NFKPluginDescriptor *descriptor in descriptors) {
				NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:[descriptor takeDescriptor] length:regionLength
																				copiesFiles:YES error:NULL];
				mapped += region != nil && ((const uint8_t *)region.mutableBytes)[regionLength - 1] == 0x5A ? 1 : 0;
			}
			// The region must outlive the send: its release closes the descriptor the frame names.
			NFKPluginSharedMemory *answer NS_VALID_UNTIL_END_OF_SCOPE = [NFKPluginSharedMemory regionWithLength:regionLength namePrefix:nil error:NULL];
			memset(answer.mutableBytes, [message[@"id"] intValue] & 0xFF, regionLength);
			NSDictionary *reply = @{ @"type": @"reply", @"id": message[@"id"], @"mapped": @(mapped) };
			[weakPlugin sendMessage:reply descriptors:@[ @(answer.descriptor) ] error:NULL];
		};
		NSMutableDictionary<NSNumber *, NSNumber *> *answers = [NSMutableDictionary dictionary];
		dispatch_semaphore_t answered = dispatch_semaphore_create(0);
		host.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
			NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithDescriptor:[descriptors.firstObject takeDescriptor] length:regionLength
																			copiesFiles:YES error:NULL];
			BOOL intact = region != nil && ((const uint8_t *)region.mutableBytes)[regionLength - 1] == ([message[@"id"] intValue] & 0xFF)
				&& [message[@"mapped"] integerValue] == 2;
			@synchronized (answers) {
				answers[message[@"id"]] = @(intact);
			}
			dispatch_semaphore_signal(answered);
		};
		host.closeHandler = ^(NSError *error) {
			dispatch_semaphore_signal(answered);
		};
		[plugin resume];
		[host resume];
		[channels addObjectsFromArray:@[ host, plugin ]];
		dispatch_group_async(trips, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			for (NSInteger trip = 0; trip < tripCount; trip++) {
				NFKPluginSharedMemory *first NS_VALID_UNTIL_END_OF_SCOPE = [NFKPluginSharedMemory regionWithLength:regionLength namePrefix:nil error:NULL];
				NFKPluginSharedMemory *second NS_VALID_UNTIL_END_OF_SCOPE = [NFKPluginSharedMemory regionWithLength:regionLength namePrefix:nil error:NULL];
				memset(first.mutableBytes, 0x5A, regionLength);
				memset(second.mutableBytes, 0x5A, regionLength);
				NSError *sendError = nil;
				NSDictionary *request = @{ @"type": @"ping", @"id": @(trip) };
				BOOL sent = [host sendMessage:request descriptors:@[ @(first.descriptor), @(second.descriptor) ] error:&sendError];
				BOOL arrived = sent && dispatch_semaphore_wait(answered, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0 && !host.isClosed;
				BOOL intact = NO;
				@synchronized (answers) {
					intact = [answers[@(trip)] boolValue];
				}
				if (!arrived || !intact) {
					@synchronized (failures) {
						[failures addObject:[NSString stringWithFormat:@"pair %lu trip %ld: sent %d arrived %d intact %d closed %d %@",
											 (unsigned long)pair, (long)trip, sent, arrived, intact, host.isClosed, sendError]];
					}
					return;
				}
			}
		});
	}

	__block BOOL churning = YES;
	dispatch_group_t churn = dispatch_group_create();
	dispatch_group_async(churn, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
		while (churning) {
			@autoreleasepool {
				NSMutableArray *regions = [NSMutableArray array];
				for (NSUInteger index = 0; index < 16; index++) {
					NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithLength:20000 namePrefix:nil error:NULL];
					if (region != nil) {
						[regions addObject:region];
					}
				}
			}
		}
	});
	__block NSUInteger launches = 0;
	dispatch_group_async(churn, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
		while (churning) {
			@autoreleasepool {
				int channel[2];
				if (pipe(channel) != 0) {
					continue;
				}
				NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:@"/bin/sh" arguments:@[ @"-c", @"echo ok >&3" ]];
				process.ownerInheritedDescriptors = @{ @3: @(channel[1]) };
				close(channel[1]);
				[process startWithError:NULL];
				process.ownerInheritedDescriptors = nil;
				char buffer[16];
				while (read(channel[0], buffer, sizeof buffer) > 0) {
				}
				close(channel[0]);
				[self waitUntil:^BOOL { return process.state != NFKSubprocessStateRunning && process.state != NFKSubprocessStateStarting; } timeout:5];
				launches += 1;
			}
		}
	});

	XCTAssertEqual(dispatch_group_wait(trips, dispatch_time(DISPATCH_TIME_NOW, 120 * NSEC_PER_SEC)), 0, @"every pair finishes");
	churning = NO;
	dispatch_group_wait(churn, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC));
	@synchronized (failures) {
		XCTAssertEqualObjects(failures, @[], @"every channel answers every round trip");
	}
	XCTAssertGreaterThan(launches, 0u, @"subprocesses launched during the trips");
	for (NFKPluginChannel *channel in channels) {
		XCTAssertFalse(channel.isClosed, @"no channel closed before the test closed it");
		[channel close];
	}
}

- (void)testAHandOffReadLateStillCarriesALiveSocket
{
	for (NSUInteger attempt = 0; attempt < 20; attempt++) {
		int hostEnd = -1;
		int pluginEnd = -1;
		NSError *error = nil;
		XCTAssertTrue([NFKPluginChannel createSocketPair:&hostEnd second:&pluginEnd error:&error], @"%@", error);
		NFKPluginChannel *plugin = [NFKPluginChannel channelByHandingOffDescriptor:pluginEnd error:&error];
		XCTAssertNotNil(plugin, @"%@", error);
		// A host that reads the hand-off only after the plugin side has returned, as a supervisor does
		// with an in-process launcher.
		[NSThread sleepForTimeInterval:0.005];
		NFKPluginChannel *host = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:5 error:&error];
		XCTAssertNotNil(host, @"%@", error);
		NFKChannelRecorder *recorder = [[NFKChannelRecorder alloc] init];
		[recorder attachToChannel:plugin];
		[plugin resume];
		[host resume];
		NSDictionary *ping = @{ @"type": @"ping", @"id": @(attempt) };
		XCTAssertTrue([host sendMessage:ping error:&error], @"attempt %lu: %@", (unsigned long)attempt, error);
		XCTAssertTrue([self waitUntil:^BOOL { return recorder.messageCount == 1; } timeout:5], @"attempt %lu", (unsigned long)attempt);
		XCTAssertFalse(host.isClosed, @"attempt %lu: the handed socket arrived hung up", (unsigned long)attempt);
		[host close];
		[plugin close];
	}
}

- (void)testAHandOffThatNeverComesTimesOut
{
	int hostEnd = -1;
	int pluginEnd = -1;
	NSError *error = nil;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&hostEnd second:&pluginEnd error:&error], @"%@", error);
	NFKPluginChannel *host = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:0.3 error:&error];
	XCTAssertNil(host);
	XCTAssertEqual(error.code, NFKPluginErrorTimedOut, @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:@"hand-off"], @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:@"0.3"], @"the error names the limit: %@", error);
	close(pluginEnd);
}

@end

#endif
