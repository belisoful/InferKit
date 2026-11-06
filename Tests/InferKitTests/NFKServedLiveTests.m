//
//  NFKServedLiveTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <arpa/inet.h>
#import <netinet/in.h>
#import <sys/socket.h>
#import <unistd.h>

/*! Answers with its inputs and parameters, a pixel buffer's format, size, and first byte, and a PCM
	buffer's rate and frame count; holds each unit for delay seconds. */
@interface NFKLiveRouteBackend : NSObject <NFKInferenceBackend>
@property (nonatomic) NSTimeInterval delay;
@end

@implementation NFKLiveRouteBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"live-route";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	[NSThread sleepForTimeInterval:self.delay];
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	for (NSString *key in request.inputs) {
		id value = request.inputs[key];
		if (CFGetTypeID((__bridge CFTypeRef)value) == CVPixelBufferGetTypeID()) {
			CVPixelBufferRef pixels = (__bridge CVPixelBufferRef)value;
			outputs[@"format"] = @(CVPixelBufferGetPixelFormatType(pixels));
			outputs[@"width"] = @(CVPixelBufferGetWidth(pixels));
			outputs[NFKOutputMask] = value;
		} else if ([value isKindOfClass:AVAudioPCMBuffer.class]) {
			AVAudioPCMBuffer *pcm = value;
			outputs[@"sampleRate"] = @(pcm.format.sampleRate);
			outputs[@"frames"] = @(pcm.frameLength);
			outputs[NFKOutputAudio] = pcm;
		} else {
			outputs[key] = value;
		}
	}
	[outputs addEntriesFromDictionary:request.parameters];
	return [NFKInferenceResult resultWithOutputs:outputs];
}

@end

/*! Keeps a running total across a stream's units. */
@interface NFKLiveRouteCountingBackend : NSObject <NFKInferenceBackend>
@end

@implementation NFKLiveRouteCountingBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"live-route-counting";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ @"total": [request inputForKey:@"value"] ?: @0 }];
}

- (nullable id<NFKLiveStream>)openLiveStreamWithParameters:(nullable NSDictionary<NSString *, id> *)parameters
													 error:(NSError **)error
{
	__block NSInteger total = 0;
	return [[NFKBlockLiveStream alloc] initWithParameters:parameters unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **unitError) {
		total += [[unit inputForKey:@"value"] integerValue];
		return [NFKInferenceResult resultWithOutputs:@{ @"total": @(total) }];
	} resetHandler:^{
		total = 0;
	} closeHandler:nil];
}

@end

@interface NFKServedLiveTests : XCTestCase
@end

@implementation NFKServedLiveTests

- (NFKInferenceServer *)serverHosting:(NSDictionary<NSString *, id<NFKInferenceBackend>> *)backends
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	for (NSString *name in backends) {
		[server addBackend:backends[name] forModelName:name];
	}
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	return server;
}

- (NSURLSessionWebSocketTask *)socketTo:(NFKInferenceServer *)server
{
	NSURLComponents *components = [NSURLComponents componentsWithURL:[server.localBaseURL URLByAppendingPathComponent:@"inferkit/live"]
											 resolvingAgainstBaseURL:NO];
	components.scheme = @"ws";
	NSURLSessionWebSocketTask *task = [NSURLSession.sharedSession webSocketTaskWithURL:components.URL];
	[task resume];
	[self addTeardownBlock:^{
		[task cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
	}];
	return task;
}

- (void)send:(NSURLSessionWebSocketTask *)task JSON:(NSDictionary *)message
{
	NSString *text = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:message options:0 error:NULL]
										   encoding:NSUTF8StringEncoding];
	[self send:task message:[[NSURLSessionWebSocketMessage alloc] initWithString:text]];
}

- (void)send:(NSURLSessionWebSocketTask *)task message:(NSURLSessionWebSocketMessage *)message
{
	XCTestExpectation *sent = [self expectationWithDescription:@"sent"];
	[task sendMessage:message completionHandler:^(NSError *error) {
		XCTAssertNil(error);
		[sent fulfill];
	}];
	[self waitForExpectations:@[ sent ] timeout:10];
}

/*! The next message's header and, for a binary message, the bytes after it. */
- (NSDictionary *)receive:(NSURLSessionWebSocketTask *)task payload:(NSData * _Nullable * _Nullable)outPayload
{
	__block NSURLSessionWebSocketMessage *received = nil;
	XCTestExpectation *arrived = [self expectationWithDescription:@"received"];
	[task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage *message, NSError *error) {
		XCTAssertNil(error);
		received = message;
		[arrived fulfill];
	}];
	[self waitForExpectations:@[ arrived ] timeout:10];
	if (received.type == NSURLSessionWebSocketMessageTypeString) {
		return [NSJSONSerialization JSONObjectWithData:[received.string dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	}
	uint32_t length = 0;
	[received.data getBytes:&length length:4];
	length = CFSwapInt32LittleToHost(length);
	if (outPayload != NULL) {
		*outPayload = [received.data subdataWithRange:NSMakeRange(4 + length, received.data.length - 4 - length)];
	}
	return [NSJSONSerialization JSONObjectWithData:[received.data subdataWithRange:NSMakeRange(4, length)] options:0 error:NULL];
}

- (NSURLSessionWebSocketMessage *)binaryUnit:(NSDictionary *)header payload:(NSData *)payload
{
	NSData *headerData = [NSJSONSerialization dataWithJSONObject:header options:0 error:NULL];
	uint32_t length = CFSwapInt32HostToLittle((uint32_t)headerData.length);
	NSMutableData *data = [NSMutableData dataWithBytes:&length length:4];
	[data appendData:headerData];
	[data appendData:payload];
	return [[NSURLSessionWebSocketMessage alloc] initWithData:data];
}

#pragma mark Units

- (void)testEachUnitIsAnsweredInOrderWithTheStreamsParameters
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"start", @"model": @"echo", @"parameters": @{ @"mode": @"fast" } }];
	NSDictionary *started = [self receive:task payload:NULL];
	XCTAssertEqualObjects(started[@"type"], @"started");
	XCTAssertEqualObjects(started[@"model"], @"echo");
	XCTAssertEqualObjects(started[@"live"], NSNull.null, @"a backend added without a descriptor declares no profile");
	for (NSInteger sequence = 1; sequence <= 3; sequence++) {
		[self send:task JSON:@{ @"type": @"unit", @"seq": @(sequence), @"inputs": @{ NFKInputPrompt: @"hi" },
								@"parameters": sequence == 2 ? @{ @"mode": @"exact" } : @{} }];
	}
	for (NSInteger sequence = 1; sequence <= 3; sequence++) {
		NSDictionary *result = [self receive:task payload:NULL];
		XCTAssertEqualObjects(result[@"type"], @"result");
		XCTAssertEqualObjects(result[@"seq"], @(sequence));
		XCTAssertEqualObjects(result[@"outputs"][NFKInputPrompt], @"hi");
		XCTAssertEqualObjects(result[@"outputs"][@"mode"], sequence == 2 ? @"exact" : @"fast");
		XCTAssertEqualObjects(result[@"dropped"], @[]);
		XCTAssertTrue([result[@"milliseconds"] isKindOfClass:NSNumber.class]);
	}
}

- (void)testAStatefulBackendKeepsItsStateUntilReset
{
	NFKInferenceServer *server = [self serverHosting:@{ @"counter": [[NFKLiveRouteCountingBackend alloc] init] }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"start" }];
	XCTAssertEqualObjects([self receive:task payload:NULL][@"type"], @"started", @"the only hosted model needs no name");
	NSMutableArray *totals = [NSMutableArray array];
	for (NSNumber *value in @[ @2, @3 ]) {
		[self send:task JSON:@{ @"type": @"unit", @"inputs": @{ @"value": value } }];
		[totals addObject:[self receive:task payload:NULL][@"outputs"][@"total"]];
	}
	[self send:task JSON:@{ @"type": @"reset" }];
	XCTAssertEqualObjects([self receive:task payload:NULL][@"type"], @"reset");
	[self send:task JSON:@{ @"type": @"unit", @"inputs": @{ @"value": @4 } }];
	[totals addObject:[self receive:task payload:NULL][@"outputs"][@"total"]];
	XCTAssertEqualObjects(totals, (@[ @2, @5, @4 ]));
}

- (void)testAFrameArrivesAsAPixelBufferAndAnImageReturnsAsRawBGRA
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"start" }];
	[self receive:task payload:NULL];
	NSMutableData *pixels = [NSMutableData dataWithLength:20 * 2];          // 4 × 2 BGRA, rows padded to 20 bytes
	uint8_t *bytes = pixels.mutableBytes;
	for (NSUInteger row = 0; row < 2; row++) {
		for (NSUInteger index = 0; index < 16; index++) {
			bytes[row * 20 + index] = (uint8_t)(row * 16 + index);
		}
	}
	NSDictionary *buffer = @{ @"key": NFKInputImage, @"format": @"bgra", @"width": @4, @"height": @2, @"bytesPerRow": @20,
							  @"offset": @0, @"length": @(pixels.length) };
	[self send:task message:[self binaryUnit:@{ @"type": @"unit", @"seq": @7, @"buffers": @[ buffer ] } payload:pixels]];
	NSData *payload = nil;
	NSDictionary *result = [self receive:task payload:&payload];
	XCTAssertEqualObjects(result[@"seq"], @7);
	XCTAssertEqualObjects(result[@"outputs"][@"format"], @(kCVPixelFormatType_32BGRA));
	XCTAssertEqualObjects(result[@"outputs"][@"width"], @4);
	NSDictionary *mask = [result[@"buffers"] firstObject];
	XCTAssertEqualObjects(mask[@"key"], NFKOutputMask);
	XCTAssertEqualObjects(mask[@"format"], @"bgra");
	XCTAssertEqualObjects(mask[@"bytesPerRow"], @16, @"a returned buffer's rows are tight");
	NSMutableData *expected = [NSMutableData data];
	[expected appendBytes:bytes length:16];
	[expected appendBytes:bytes + 20 length:16];
	XCTAssertEqualObjects([payload subdataWithRange:NSMakeRange([mask[@"offset"] unsignedIntegerValue], [mask[@"length"] unsignedIntegerValue])],
						  expected);
}

- (void)testAChunkArrivesAsAPCMBufferAndReturnsAsFloatPCM
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"start" }];
	[self receive:task payload:NULL];
	int16_t samples[320];
	for (NSUInteger index = 0; index < 320; index++) {
		samples[index] = (int16_t)(index * 100 - 16000);
	}
	NSData *chunk = [NSData dataWithBytes:samples length:sizeof samples];
	NSDictionary *buffer = @{ @"key": NFKInputAudio, @"format": @"pcm-s16", @"sampleRate": @16000, @"channels": @1,
							  @"offset": @0, @"length": @(chunk.length) };
	[self send:task message:[self binaryUnit:@{ @"type": @"unit", @"seq": @1, @"buffers": @[ buffer ] } payload:chunk]];
	NSData *payload = nil;
	NSDictionary *result = [self receive:task payload:&payload];
	XCTAssertEqualObjects(result[@"outputs"][@"sampleRate"], @16000);
	XCTAssertEqualObjects(result[@"outputs"][@"frames"], @320);
	NSDictionary *audio = [result[@"buffers"] firstObject];
	XCTAssertEqualObjects(audio[@"format"], @"pcm-f32");
	XCTAssertEqualObjects(audio[@"channels"], @1);
	XCTAssertEqual([audio[@"length"] unsignedIntegerValue], 320u * 4);
	float first = 0;
	[payload getBytes:&first range:NSMakeRange([audio[@"offset"] unsignedIntegerValue], 4)];
	XCTAssertEqualWithAccuracy(first, -16000.0f / 32768.0f, 1e-6f);
}

- (void)testTheNewestFrameRunsNextAndTheDropsAreReported
{
	NFKLiveRouteBackend *backend = [[NFKLiveRouteBackend alloc] init];
	backend.delay = 0.3;
	NFKInferenceServer *server = [self serverHosting:@{ @"slow": backend }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"start" }];
	[self receive:task payload:NULL];
	uint8_t pixel[4] = { 1, 2, 3, 255 };
	NSData *frame = [NSData dataWithBytes:pixel length:4];
	NSDictionary *buffer = @{ @"key": NFKInputImage, @"format": @"bgra", @"width": @1, @"height": @1, @"bytesPerRow": @4,
							  @"offset": @0, @"length": @4 };
	for (NSInteger sequence = 1; sequence <= 5; sequence++) {
		[self send:task message:[self binaryUnit:@{ @"type": @"unit", @"seq": @(sequence), @"buffers": @[ buffer ] } payload:frame]];
	}
	NSDictionary *first = [self receive:task payload:NULL];
	NSDictionary *last = [self receive:task payload:NULL];
	XCTAssertEqualObjects(first[@"seq"], @1);
	XCTAssertEqualObjects(last[@"seq"], @5, @"the frames queued behind the first are replaced by the newest");
	NSArray *dropped = [first[@"dropped"] arrayByAddingObjectsFromArray:last[@"dropped"]];
	XCTAssertEqualObjects(dropped, (@[ @2, @3, @4 ]), @"each drop is reported once, on the next answer");
}

#pragma mark The native client

- (void)testTheNativeClientStreamsAHostedModelWithItsState
{
	NFKInferenceServer *server = [self serverHosting:@{ @"counter": [[NFKLiveRouteCountingBackend alloc] init],
													   @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NFKRemoteInferKitBackend *client = [NFKRemoteInferKitBackend backendWithBaseURL:server.localBaseURL];
	client.modelName = @"counter";
	NSError *error = nil;
	id<NFKLiveStream> stream = NFKInferenceOpenLiveStream(client, nil, &error);
	XCTAssertNotNil(stream, @"%@", error);
	NSMutableArray *totals = [NSMutableArray array];
	for (NSNumber *value in @[ @1, @2, @3 ]) {
		[totals addObject:[[stream resultForUnit:[NFKInferenceRequest requestWithInputs:@{ @"value": value }] error:&error]
						   outputForKey:@"total"] ?: NSNull.null];
	}
	XCTAssertEqualObjects(totals, (@[ @1, @3, @6 ]), @"the hosted stream keeps its state: %@", error);
	[stream reset];
	XCTAssertEqualObjects([[stream resultForUnit:[NFKInferenceRequest requestWithInputs:@{ @"value": @5 }] error:&error]
						   outputForKey:@"total"], @5);
	[stream close];

	client.modelName = @"echo";
	stream = NFKInferenceOpenLiveStream(client, @{ @"mode": @"fast" }, &error);
	CVPixelBufferRef frame = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, 3, 2, kCVPixelFormatType_32BGRA, NULL, &frame);
	CVPixelBufferLockBaseAddress(frame, 0);
	memset(CVPixelBufferGetBaseAddress(frame), 0x5A, CVPixelBufferGetBytesPerRow(frame) * 2);
	CVPixelBufferUnlockBaseAddress(frame, 0);
	NFKInferenceResult *result = [stream resultForUnit:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)frame }]
												 error:&error];
	CVPixelBufferRelease(frame);
	XCTAssertEqualObjects([result outputForKey:@"mode"], @"fast", @"%@", error);
	CVPixelBufferRef mask = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputMask];
	XCTAssertEqual(CFGetTypeID(mask), CVPixelBufferGetTypeID(), @"an image returns as a pixel buffer");
	XCTAssertEqual(CVPixelBufferGetWidth(mask), 3u);
	CVPixelBufferLockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);
	XCTAssertEqual(((const uint8_t *)CVPixelBufferGetBaseAddress(mask))[5], 0x5A);
	CVPixelBufferUnlockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);
	[stream close];

	client.modelName = @"ghost";
	XCTAssertNil(NFKInferenceOpenLiveStream(client, nil, &error));
	XCTAssertEqualObjects(error.domain, NFKInferenceServerErrorDomain);
	XCTAssertEqual(error.code, NFKInferenceServerErrorModelNotFound);
}

#pragma mark Refusals
#pragma mark Refusals

- (void)testAMalformedOrEarlyUnitIsAnsweredWithAnErrorAndTheStreamGoesOn
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NSURLSessionWebSocketTask *task = [self socketTo:server];
	[self send:task JSON:@{ @"type": @"unit", @"seq": @1, @"inputs": @{} }];
	NSDictionary *early = [self receive:task payload:NULL];
	XCTAssertEqualObjects(early[@"type"], @"error");
	XCTAssertEqualObjects(early[@"seq"], @1);
	XCTAssertEqualObjects(early[@"error"][@"inferkit_code"], @(kNFKError_InferenceNotReady));
	[self send:task JSON:@{ @"type": @"start", @"model": @"ghost" }];
	XCTAssertEqualObjects([self receive:task payload:NULL][@"error"][@"inferkit_domain"], NFKInferenceServerErrorDomain);
	[self send:task JSON:@{ @"type": @"start" }];
	[self receive:task payload:NULL];
	NSDictionary *buffer = @{ @"key": NFKInputImage, @"format": @"bgra", @"width": @4, @"height": @4, @"bytesPerRow": @16,
							  @"offset": @0, @"length": @8 };
	[self send:task message:[self binaryUnit:@{ @"type": @"unit", @"seq": @2, @"buffers": @[ buffer ] } payload:[NSMutableData dataWithLength:8]]];
	NSDictionary *short_ = [self receive:task payload:NULL];
	XCTAssertEqualObjects(short_[@"type"], @"error", @"a buffer shorter than its rows is refused");
	XCTAssertEqualObjects(short_[@"seq"], @2);
	[self send:task JSON:@{ @"type": @"unit", @"seq": @3, @"inputs": @{ NFKInputPrompt: @"still here" } }];
	XCTAssertEqualObjects([self receive:task payload:NULL][@"outputs"][NFKInputPrompt], @"still here");
}

- (void)testAPlainRequestIsRefusedAndAKeyIsCheckedAtTheUpgrade
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	NSHTTPURLResponse *response = nil;
	NSURLRequest *plain = [NSURLRequest requestWithURL:[server.localBaseURL URLByAppendingPathComponent:@"inferkit/live"]];
	[NFKRemoteTransport sendRequest:plain session:NSURLSession.sharedSession response:&response error:NULL];
	XCTAssertEqual(response.statusCode, 400, @"a GET without an upgrade is refused");

	server.requiresAPIKeyOnLoopback = YES;
	server.apiKey = @"sesame";
	NSURLSessionWebSocketTask *refused = [self socketTo:server];
	XCTestExpectation *failed = [self expectationWithDescription:@"refused"];
	[refused receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage *message, NSError *error) {
		XCTAssertNotNil(error);
		[failed fulfill];
	}];
	[self waitForExpectations:@[ failed ] timeout:10];
	XCTAssertEqual([(NSHTTPURLResponse *)refused.response statusCode], 401);

	NSURLComponents *components = [NSURLComponents componentsWithURL:[server.localBaseURL URLByAppendingPathComponent:@"inferkit/live"]
											 resolvingAgainstBaseURL:NO];
	components.scheme = @"ws";
	NSMutableURLRequest *keyed = [NSMutableURLRequest requestWithURL:components.URL];
	[keyed setValue:@"Bearer sesame" forHTTPHeaderField:@"Authorization"];
	NSURLSessionWebSocketTask *task = [NSURLSession.sharedSession webSocketTaskWithRequest:keyed];
	[task resume];
	[self send:task JSON:@{ @"type": @"start" }];
	XCTAssertEqualObjects([self receive:task payload:NULL][@"type"], @"started");
	[task cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
}

#pragma mark Framing

/*! A connected socket that has completed the route's upgrade, or -1. */
- (int)upgradedSocketTo:(NFKInferenceServer *)server
{
	int descriptor = socket(AF_INET, SOCK_STREAM, 0);
	uint16_t port = (uint16_t)server.localBaseURL.port.unsignedIntValue;
	struct sockaddr_in address = { .sin_family = AF_INET, .sin_port = htons(port) };
	inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
	struct timeval timeout = { .tv_sec = 10 };
	setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof timeout);
	if (connect(descriptor, (struct sockaddr *)&address, sizeof address) != 0) {
		close(descriptor);
		return -1;
	}
	NSString *head = [NSString stringWithFormat:@"GET /v1/inferkit/live HTTP/1.1\r\nHost: 127.0.0.1:%u\r\nUpgrade: websocket\r\n"
					  "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
					  port];
	send(descriptor, head.UTF8String, strlen(head.UTF8String), 0);
	NSString *answer = [[NSString alloc] initWithData:[self readFrom:descriptor until:@"\r\n\r\n"] encoding:NSUTF8StringEncoding];
	XCTAssertTrue([answer hasPrefix:@"HTTP/1.1 101"], @"%@", answer);
	XCTAssertTrue([answer containsString:@"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="], @"RFC 6455's worked example");
	[self addTeardownBlock:^{
		close(descriptor);
	}];
	return descriptor;
}

- (NSData *)readFrom:(int)descriptor until:(NSString *)marker
{
	NSMutableData *data = [NSMutableData data];
	NSData *end = [marker dataUsingEncoding:NSUTF8StringEncoding];
	uint8_t byte = 0;
	while ([data rangeOfData:end options:0 range:NSMakeRange(0, data.length)].location == NSNotFound && recv(descriptor, &byte, 1, 0) == 1) {
		[data appendBytes:&byte length:1];
	}
	return data;
}

/*! A client frame, masked unless told otherwise. */
- (NSData *)frameWithOpcode:(uint8_t)opcode final:(BOOL)final payload:(NSData *)payload masked:(BOOL)masked
{
	NSMutableData *frame = [NSMutableData data];
	uint8_t first = (final ? 0x80 : 0) | opcode;
	[frame appendBytes:&first length:1];
	uint8_t second = (masked ? 0x80 : 0) | (uint8_t)(payload.length < 126 ? payload.length : 126);
	[frame appendBytes:&second length:1];
	if (payload.length >= 126) {
		uint16_t length = CFSwapInt16HostToBig((uint16_t)payload.length);
		[frame appendBytes:&length length:2];
	}
	uint8_t mask[4] = { 0x12, 0x34, 0x56, 0x78 };
	NSMutableData *body = [payload mutableCopy];
	if (masked) {
		[frame appendBytes:mask length:4];
		for (NSUInteger index = 0; index < body.length; index++) {
			((uint8_t *)body.mutableBytes)[index] ^= mask[index % 4];
		}
	}
	[frame appendData:body];
	return frame;
}

/*! The next server frame's opcode and payload. */
- (NSData *)readFrameFrom:(int)descriptor opcode:(uint8_t *)opcode
{
	uint8_t head[2] = { 0 };
	if (recv(descriptor, head, 2, MSG_WAITALL) != 2) {
		return nil;
	}
	*opcode = head[0] & 0x0F;
	XCTAssertEqual(head[1] & 0x80, 0, @"a server frame is not masked");
	NSUInteger length = head[1] & 0x7F;
	if (length == 126) {
		uint16_t extended = 0;
		recv(descriptor, &extended, 2, MSG_WAITALL);
		length = CFSwapInt16BigToHost(extended);
	}
	NSMutableData *payload = [NSMutableData dataWithLength:length];
	if (length > 0) {
		recv(descriptor, payload.mutableBytes, length, MSG_WAITALL);
	}
	return payload;
}

- (void)testFragmentsJoinAndAPingIsAnsweredWithItsPayload
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	int descriptor = [self upgradedSocketTo:server];
	NSData *ping = [self frameWithOpcode:0x9 final:YES payload:[@"are you there" dataUsingEncoding:NSUTF8StringEncoding] masked:YES];
	send(descriptor, ping.bytes, ping.length, 0);
	uint8_t opcode = 0;
	NSData *pong = [self readFrameFrom:descriptor opcode:&opcode];
	XCTAssertEqual(opcode, 0xA);
	XCTAssertEqualObjects(pong, [@"are you there" dataUsingEncoding:NSUTF8StringEncoding]);

	NSData *start = [@"{\"type\": \"start\"}" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *head = [self frameWithOpcode:0x1 final:NO payload:[start subdataWithRange:NSMakeRange(0, 6)] masked:YES];
	NSData *tail = [self frameWithOpcode:0x0 final:YES payload:[start subdataWithRange:NSMakeRange(6, start.length - 6)] masked:YES];
	send(descriptor, head.bytes, head.length, 0);
	send(descriptor, tail.bytes, tail.length, 0);
	NSData *started = [self readFrameFrom:descriptor opcode:&opcode];
	XCTAssertEqual(opcode, 0x1);
	XCTAssertEqualObjects([NSJSONSerialization JSONObjectWithData:started options:0 error:NULL][@"type"], @"started");
}

- (void)testAnUnmaskedOrOversizeFrameClosesTheSocketWithItsCode
{
	NFKInferenceServer *server = [self serverHosting:@{ @"echo": [[NFKLiveRouteBackend alloc] init] }];
	int descriptor = [self upgradedSocketTo:server];
	NSData *unmasked = [self frameWithOpcode:0x1 final:YES payload:[@"{}" dataUsingEncoding:NSUTF8StringEncoding] masked:NO];
	send(descriptor, unmasked.bytes, unmasked.length, 0);
	uint8_t opcode = 0;
	NSData *closing = [self readFrameFrom:descriptor opcode:&opcode];
	XCTAssertEqual(opcode, 0x8);
	XCTAssertEqual(CFSwapInt16BigToHost(*(const uint16_t *)closing.bytes), 1002);

	[server stop];
	server.maximumRequestBodyBytes = 256;
	XCTAssertTrue([server startWithError:NULL]);
	descriptor = [self upgradedSocketTo:server];
	NSData *large = [self frameWithOpcode:0x2 final:YES payload:[NSMutableData dataWithLength:300] masked:YES];
	send(descriptor, large.bytes, large.length, 0);
	closing = [self readFrameFrom:descriptor opcode:&opcode];
	XCTAssertEqual(opcode, 0x8);
	XCTAssertEqual(CFSwapInt16BigToHost(*(const uint16_t *)closing.bytes), 1009);
}

@end
