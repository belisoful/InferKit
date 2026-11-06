//
//  NFKServedLive.m
//  InferKit
//

#import "NFKServedLive.h"
#import "NFKInferenceServerPrivate.h"
#import "NFKInferenceWireCoding.h"
#import "NFKImageCoding.h"
#import "NFKLiveStream.h"
#import "NFKModelBuilding.h"
#import "NFKModelDescriptor.h"
#import "NFKModelLive.h"
#import "NFKErrors.h"
#import "NFKAudioAsset.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

/*! The most units a stream queues in order behind the one in flight. */
static const NSUInteger NFKServedLiveQueueLimit = 256;

static NSError *NFKServedLiveError(NSInteger code, NSString *reason)
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

static NSString *NFKServedLiveJSONString(id object)
{
	NSData *data = [NSJSONSerialization isValidJSONObject:object] ? [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL] : nil;
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

/*! One message's header and the bytes after it: a 4-byte little-endian header length, then the header. */
static NSData *NFKServedLiveFramed(NSDictionary *header, NSData *payload)
{
	NSData *headerData = [NSJSONSerialization dataWithJSONObject:header options:0 error:NULL] ?: [NSData data];
	uint32_t length = CFSwapInt32HostToLittle((uint32_t)headerData.length);
	NSMutableData *message = [NSMutableData dataWithBytes:&length length:4];
	[message appendData:headerData];
	[message appendData:payload];
	return message;
}

#pragma mark - Buffers in

/*! A pixel buffer of the format, copied row by row from bytes. */
static CVPixelBufferRef _Nullable NFKServedLivePixelBuffer(NSDictionary *buffer, NSData *bytes, NSError **error) CF_RETURNS_RETAINED
{
	NSString *format = buffer[@"format"];
	NSInteger width = [buffer[@"width"] integerValue], height = [buffer[@"height"] integerValue];
	NSInteger rowBytes = [buffer[@"bytesPerRow"] integerValue];
	BOOL bgra = [format isEqualToString:@"bgra"];
	OSType pixelFormat = bgra ? kCVPixelFormatType_32BGRA
		: [format isEqualToString:@"nv12"] ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
	NSInteger chromaRowBytes = buffer[@"chromaBytesPerRow"] != nil ? [buffer[@"chromaBytesPerRow"] integerValue] : rowBytes;
	NSInteger minimumRow = bgra ? width * 4 : width;
	NSInteger planeBytes = rowBytes * height + (bgra ? 0 : chromaRowBytes * ((height + 1) / 2));
	if (width <= 0 || height <= 0 || width > 16384 || height > 16384 || rowBytes < minimumRow
		|| (!bgra && chromaRowBytes < ((width + 1) / 2) * 2) || (NSInteger)bytes.length < planeBytes) {
		if (error != NULL) {
			*error = NFKServedLiveError(kNFKError_InferenceMissingInput,
										[NSString stringWithFormat:@"the %@ buffer's size, rows, or bytes do not agree", format]);
		}
		return NULL;
	}
	CVPixelBufferRef pixelBuffer = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	if (CVPixelBufferCreate(kCFAllocatorDefault, (size_t)width, (size_t)height, pixelFormat, (__bridge CFDictionaryRef)attributes,
							&pixelBuffer) != kCVReturnSuccess) {
		if (error != NULL) {
			*error = NFKServedLiveError(kNFKError_InferenceBackendFailure, @"no pixel buffer could be made for the frame");
		}
		return NULL;
	}
	CVPixelBufferLockBaseAddress(pixelBuffer, 0);
	const uint8_t *source = bytes.bytes;
	size_t planes = bgra ? 1 : 2;
	for (size_t plane = 0; plane < planes; plane++) {
		uint8_t *destination = bgra ? CVPixelBufferGetBaseAddress(pixelBuffer) : CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane);
		size_t destinationRow = bgra ? CVPixelBufferGetBytesPerRow(pixelBuffer) : CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane);
		size_t rows = plane == 0 ? (size_t)height : (size_t)(height + 1) / 2;
		size_t sourceRow = plane == 0 ? (size_t)rowBytes : (size_t)chromaRowBytes;
		size_t copied = MIN(destinationRow, plane == 0 ? (size_t)minimumRow : (size_t)((width + 1) / 2) * 2);
		for (size_t row = 0; row < rows; row++) {
			memcpy(destination + row * destinationRow, source + row * sourceRow, copied);
		}
		source += sourceRow * rows;
	}
	CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
	return pixelBuffer;
}

/*! An interleaved PCM buffer of the format. */
static AVAudioPCMBuffer * _Nullable NFKServedLivePCMBuffer(NSDictionary *buffer, NSData *bytes, NSError **error)
{
	BOOL floats = [buffer[@"format"] isEqualToString:@"pcm-f32"];
	double sampleRate = [buffer[@"sampleRate"] doubleValue];
	NSInteger channels = [buffer[@"channels"] integerValue];
	NSUInteger sampleBytes = floats ? 4 : 2;
	if (!(sampleRate > 0) || channels <= 0 || channels > 64 || bytes.length % (sampleBytes * (NSUInteger)channels) != 0) {
		if (error != NULL) {
			*error = NFKServedLiveError(kNFKError_InferenceMissingInput, @"the PCM buffer's rate, channels, or bytes do not agree");
		}
		return nil;
	}
	AVAudioFormat *format = [[AVAudioFormat alloc] initWithCommonFormat:floats ? AVAudioPCMFormatFloat32 : AVAudioPCMFormatInt16
															 sampleRate:sampleRate channels:(AVAudioChannelCount)channels interleaved:YES];
	AVAudioFrameCount frames = (AVAudioFrameCount)(bytes.length / (sampleBytes * (NSUInteger)channels));
	AVAudioPCMBuffer *pcm = [[AVAudioPCMBuffer alloc] initWithPCMFormat:format frameCapacity:MAX(frames, 1)];
	pcm.frameLength = frames;
	memcpy(floats ? (void *)pcm.floatChannelData[0] : (void *)pcm.int16ChannelData[0], bytes.bytes, bytes.length);
	return pcm;
}

#pragma mark - Buffers out

/*! A result value as a raw buffer's description and bytes, or nil for a value that travels in outputs. */
static NSDictionary * _Nullable NFKServedLiveOutputBuffer(id value, NSString *key, NSData * _Nullable * _Nonnull outBytes)
{
	CFTypeRef reference = (__bridge CFTypeRef)value;
	if ([value isKindOfClass:AVAudioPCMBuffer.class] || [value isKindOfClass:NFKAudioAsset.class]) {
		AVAudioPCMBuffer *pcm = [value isKindOfClass:AVAudioPCMBuffer.class] ? value : nil;
		if (pcm == nil) {
			NSURL *url = [(NFKAudioAsset *)value fileURL];
			AVAudioFile *file = url != nil ? [[AVAudioFile alloc] initForReading:url commonFormat:AVAudioPCMFormatFloat32
																	 interleaved:YES error:NULL] : nil;
			pcm = file != nil ? [[AVAudioPCMBuffer alloc] initWithPCMFormat:file.processingFormat
															  frameCapacity:(AVAudioFrameCount)MAX(file.length, 1)] : nil;
			if (pcm == nil || ![file readIntoBuffer:pcm error:NULL]) {
				return nil;
			}
		}
		AVAudioFormat *interleaved = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:pcm.format.sampleRate
																		channels:pcm.format.channelCount interleaved:YES];
		AVAudioPCMBuffer *converted = pcm;
		if (![pcm.format isEqual:interleaved]) {
			AVAudioConverter *converter = [[AVAudioConverter alloc] initFromFormat:pcm.format toFormat:interleaved];
			converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:interleaved frameCapacity:MAX(pcm.frameLength, 1)];
			if (converter == nil || ![converter convertToBuffer:converted fromBuffer:pcm error:NULL]) {
				return nil;
			}
		}
		NSUInteger length = (NSUInteger)converted.frameLength * converted.format.channelCount * 4;
		*outBytes = [NSData dataWithBytes:converted.floatChannelData[0] length:length];
		return @{ @"key": key, @"format": @"pcm-f32", @"sampleRate": @(converted.format.sampleRate),
				  @"channels": @(converted.format.channelCount) };
	}
	BOOL pixelBuffer = CFGetTypeID(reference) == CVPixelBufferGetTypeID();
	if (!pixelBuffer && CFGetTypeID(reference) != CGImageGetTypeID()) {
		return nil;
	}
	CVPixelBufferRef buffer = pixelBuffer ? (CVPixelBufferRef)CFRetain(reference) : [NFKImageCoding pixelBufferWithCGImage:(CGImageRef)reference];
	OSType format = buffer != NULL ? CVPixelBufferGetPixelFormatType(buffer) : 0;
	NSDictionary<NSNumber *, NSArray *> *raw = @{ @(kCVPixelFormatType_32BGRA): @[ @"bgra", @4 ],
												  @(kCVPixelFormatType_OneComponent8): @[ @"gray8", @1 ],
												  @(kCVPixelFormatType_OneComponent32Float): @[ @"gray-f32", @4 ] };
	if (buffer != NULL && raw[@(format)] == nil) {
		CGImageRef image = [NFKImageCoding CGImageForImage:(__bridge id)buffer];
		CVPixelBufferRelease(buffer);
		buffer = image != NULL ? [NFKImageCoding pixelBufferWithCGImage:image] : NULL;
		CGImageRelease(image);
		format = buffer != NULL ? CVPixelBufferGetPixelFormatType(buffer) : 0;
	}
	if (buffer == NULL || raw[@(format)] == nil) {
		if (buffer != NULL) {
			CVPixelBufferRelease(buffer);
		}
		return nil;
	}
	size_t width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer);
	size_t rowBytes = width * [raw[@(format)][1] unsignedIntegerValue];
	NSMutableData *bytes = [NSMutableData dataWithLength:rowBytes * height];
	CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t sourceRow = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t row = 0; row < height; row++) {
		memcpy((uint8_t *)bytes.mutableBytes + row * rowBytes, base + row * sourceRow, rowBytes);
	}
	CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
	CVPixelBufferRelease(buffer);
	*outBytes = bytes;
	return @{ @"key": key, @"format": raw[@(format)][0], @"width": @(width), @"height": @(height), @"bytesPerRow": @(rowBytes) };
}

#pragma mark - Session

/*! One queued unit, or a reset that waits its turn behind the units before it. */
@interface NFKServedLiveItem : NSObject
@property (nonatomic, strong, nullable) id sequence;
@property (nonatomic, strong, nullable) NFKInferenceRequest *unit;
@property (nonatomic, copy) NSArray<NSURL *> *temporaryFiles;
@property (nonatomic, assign) BOOL video;
@property (nonatomic, assign) BOOL reset;
@end

@implementation NFKServedLiveItem
@end

/*! One connection's stream: its target, the units waiting, and the stream the hosted backend opened. */
@interface NFKServedLiveSession : NSObject
@property (nonatomic, weak) NFKInferenceServer *server;
@property (nonatomic, strong) NFKHTTPWebSocket *socket;
@property (nonatomic, strong) NFKHTTPRequest *request;
@property (nonatomic, strong) NFKHTTPResponse *response;
@property (nonatomic, strong, nullable) NFKServedTarget *target;
@property (nonatomic, copy) NSDictionary<NSString *, id> *parameters;
@property (nonatomic, assign) BOOL valueObjectsAsJSON;
@property (nonatomic, strong) NSMutableArray<NFKServedLiveItem *> *queue;
@property (nonatomic, strong) NSMutableArray *dropped;
@property (nonatomic, assign) BOOL inFlight;
@property (nonatomic, assign) BOOL ended;
/*! The stream and the backend it was opened on, read and written in a unit's work, one at a time. */
@property (nonatomic, strong, nullable) id<NFKLiveStream> stream;
@property (nonatomic, weak, nullable) id<NFKInferenceBackend> streamBackend;
@property (nonatomic, assign) NSUInteger generation;
@end

@implementation NFKServedLiveSession

- (void)send:(NSDictionary *)message
{
	[self.socket sendText:NFKServedLiveJSONString(message)];
}

- (void)sendError:(NSError *)error sequence:(nullable id)sequence
{
	NSMutableDictionary *message = [@{ @"type": @"error", @"error": [NFKInferenceWireCoding JSONObjectForError:error][@"error"] ?: @{} } mutableCopy];
	message[@"seq"] = sequence;
	[self send:message];
}

- (void)receiveBinary:(nullable NSData *)binary text:(nullable NSString *)text
{
	NSData *headerData = nil, *payload = nil;
	if (text != nil) {
		headerData = [text dataUsingEncoding:NSUTF8StringEncoding];
	} else if (binary.length >= 4) {
		uint32_t length = 0;
		[binary getBytes:&length length:4];
		length = CFSwapInt32LittleToHost(length);
		if ((NSUInteger)length <= binary.length - 4) {
			headerData = [binary subdataWithRange:NSMakeRange(4, length)];
			payload = [binary subdataWithRange:NSMakeRange(4 + length, binary.length - 4 - length)];
		}
	}
	id message = headerData != nil ? [NSJSONSerialization JSONObjectWithData:headerData options:0 error:NULL] : nil;
	if (![message isKindOfClass:NSDictionary.class]) {
		[self sendError:NFKServedLiveError(kNFKError_InferenceMissingInput, @"a message is not a JSON object, or a binary message's header length runs past it") sequence:nil];
		return;
	}
	NSString *type = message[@"type"];
	if ([type isEqualToString:@"start"]) {
		[self start:message];
	} else if ([type isEqualToString:@"unit"]) {
		[self push:message payload:payload ?: [NSData data]];
	} else if ([type isEqualToString:@"reset"]) {
		NFKServedLiveItem *item = [[NFKServedLiveItem alloc] init];
		item.reset = YES;
		[self enqueue:item];
	} else if ([type isEqualToString:@"close"]) {
		[self.socket closeWithCode:1000 reason:nil];
	} else {
		[self sendError:NFKServedLiveError(kNFKError_InferenceMissingInput, @"type must be start, unit, reset, or close") sequence:message[@"seq"]];
	}
}

- (void)start:(NSDictionary *)message
{
	NSError *error = nil;
	id parameters = message[@"parameters"];
	id valueObjects = message[@"valueObjects"];
	if ((parameters != nil && ![parameters isKindOfClass:NSDictionary.class])
		|| (valueObjects != nil && ![@[ @"archive", @"json" ] containsObject:valueObjects])) {
		[self sendError:NFKServedLiveError(kNFKError_InferenceMissingInput,
										   @"parameters must be an object and valueObjects \"archive\" or \"json\"") sequence:nil];
		return;
	}
	NFKServedTarget *target = [self.server modelNamed:message[@"model"] ?: self.request.query[@"model"] error:&error];
	if (target == nil) {
		[self sendError:error sequence:nil];
		return;
	}
	@synchronized (self) {
		self.target = target;
		self.parameters = parameters ?: @{};
		self.valueObjectsAsJSON = [valueObjects isEqual:@"json"];
		self.generation += 1;
		[self.queue removeAllObjects];
	}
	NFKModelLiveProfile *profile = [self.server descriptorOfTarget:target].liveProfile;
	NSMutableDictionary *started = [@{ @"type": @"started", @"model": target.name, @"live": profile.dictionaryRepresentation ?: NSNull.null } mutableCopy];
	started[@"base"] = target.baseModelName;
	started[@"preset"] = target.presetIdentifier;
	[self send:started];
}

- (void)push:(NSDictionary *)message payload:(NSData *)payload
{
	id sequence = message[@"seq"];
	if (self.target == nil) {
		[self sendError:NFKServedLiveError(kNFKError_InferenceNotReady, @"a unit came before the stream's start") sequence:sequence];
		return;
	}
	id buffers = message[@"buffers"] ?: @[];
	id inputs = message[@"inputs"] ?: @{};
	id parameters = message[@"parameters"] ?: @{};
	if (![buffers isKindOfClass:NSArray.class] || ![inputs isKindOfClass:NSDictionary.class] || ![parameters isKindOfClass:NSDictionary.class]) {
		[self sendError:NFKServedLiveError(kNFKError_InferenceMissingInput, @"inputs and parameters must be objects and buffers an array")
			   sequence:sequence];
		return;
	}
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NSError *error = nil;
	NFKInferenceRequest *decoded = [decoder requestFromJSONObject:@{ @"inputs": inputs, @"parameters": parameters } error:&error];
	NSMutableDictionary<NSString *, id> *allInputs = [decoded.inputs mutableCopy];
	BOOL video = NO;
	for (id buffer in decoded != nil ? buffers : @[]) {
		NSUInteger offset = [buffer isKindOfClass:NSDictionary.class] ? [buffer[@"offset"] unsignedIntegerValue] : 0;
		NSUInteger length = [buffer isKindOfClass:NSDictionary.class] ? [buffer[@"length"] unsignedIntegerValue] : 0;
		NSString *key = [buffer isKindOfClass:NSDictionary.class] ? buffer[@"key"] : nil;
		NSString *format = [buffer isKindOfClass:NSDictionary.class] ? buffer[@"format"] : nil;
		if (![key isKindOfClass:NSString.class] || ![format isKindOfClass:NSString.class] || offset > payload.length
			|| length > payload.length - offset) {
			error = NFKServedLiveError(kNFKError_InferenceMissingInput, @"a buffer names no key or format, or reaches past the message");
			break;
		}
		NSData *bytes = [payload subdataWithRange:NSMakeRange(offset, length)];
		if ([@[ @"bgra", @"nv12", @"nv12-full" ] containsObject:format]) {
			CVPixelBufferRef pixels = NFKServedLivePixelBuffer(buffer, bytes, &error);
			if (pixels == NULL) {
				break;
			}
			allInputs[key] = (__bridge_transfer id)pixels;
			video = YES;
		} else if ([@[ @"pcm-f32", @"pcm-s16" ] containsObject:format]) {
			AVAudioPCMBuffer *pcm = NFKServedLivePCMBuffer(buffer, bytes, &error);
			if (pcm == nil) {
				break;
			}
			allInputs[key] = pcm;
		} else {
			error = NFKServedLiveError(kNFKError_InferenceMissingInput,
									   [NSString stringWithFormat:@"buffer format \"%@\" is not bgra, nv12, nv12-full, pcm-f32, or pcm-s16", format]);
			break;
		}
	}
	if (decoded == nil || error != nil) {
		for (NSURL *file in decoder.temporaryFileURLs) {
			[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
		}
		[self sendError:error sequence:sequence];
		return;
	}
	NFKServedLiveItem *item = [[NFKServedLiveItem alloc] init];
	item.sequence = sequence;
	item.unit = [NFKInferenceRequest requestWithInputs:allInputs parameters:decoded.parameters outputModality:decoded.outputModality];
	item.temporaryFiles = decoder.temporaryFileURLs;
	item.video = video;
	[self enqueue:item];
}

- (void)enqueue:(NFKServedLiveItem *)item
{
	@synchronized (self) {
		if (self.ended) {
			return;
		}
		if (item.video) {
			NSIndexSet *frames = [self.queue indexesOfObjectsPassingTest:^BOOL(NFKServedLiveItem *queued, NSUInteger index, BOOL *stop) {
				return queued.video;
			}];
			for (NFKServedLiveItem *frame in [self.queue objectsAtIndexes:frames]) {
				[self drop:frame];
			}
			[self.queue removeObjectsAtIndexes:frames];
		} else if (self.queue.count >= NFKServedLiveQueueLimit) {
			[self drop:self.queue.firstObject];
			[self.queue removeObjectAtIndex:0];
		}
		[self.queue addObject:item];
	}
	[self pump];
}

/*! Records a unit left out. Runs on the session's lock. */
- (void)drop:(NFKServedLiveItem *)item
{
	if (item.sequence != nil) {
		[self.dropped addObject:item.sequence];
	}
	for (NSURL *file in item.temporaryFiles) {
		[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
	}
}

- (void)pump
{
	NFKServedLiveItem *item = nil;
	NFKServedTarget *target = nil;
	NSUInteger generation = 0;
	@synchronized (self) {
		if (self.inFlight || self.ended || self.queue.count == 0) {
			return;
		}
		item = self.queue.firstObject;
		[self.queue removeObjectAtIndex:0];
		self.inFlight = YES;
		target = self.target;
		generation = self.generation;
	}
	if (item.reset) {
		[self.stream reset];
		[self send:@{ @"type": @"reset" }];
		[self finishItem];
		return;
	}
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = item.unit;
	__weak NFKServedLiveSession *weakSelf = self;
	CFAbsoluteTime enqueued = CFAbsoluteTimeGetCurrent();
	__block CFAbsoluteTime started = enqueued;
	run.work = ^NFKInferenceResult *(id<NFKInferenceBackend> backend, NSError **error) {
		NFKServedLiveSession *session = weakSelf;
		started = CFAbsoluteTimeGetCurrent();
		id<NFKLiveStream> stream = [session streamOnBackend:backend generation:generation error:error];
		return stream != nil ? [stream resultForUnit:item.unit error:error] : nil;
	};
	run.completionHandler = ^(NFKInferenceResult *result, NSError *error) {
		NFKServedLiveSession *session = weakSelf;
		@try {
			[session answer:item result:result error:error milliseconds:(CFAbsoluteTimeGetCurrent() - started) * 1000];
		} @catch (NSException *exception) {
			[session sendError:NFKServedLiveError(kNFKError_InferenceBackendFailure, exception.reason ?: exception.name) sequence:item.sequence];
		}
		for (NSURL *file in item.temporaryFiles) {
			[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
		}
		[session finishItem];
	};
	[self.server enqueue:run on:target request:self.request response:self.response];
}

- (void)finishItem
{
	@synchronized (self) {
		self.inFlight = NO;
	}
	[self pump];
}

/*! The stream on the backend the unit's run reached, opened with the session's parameters, and opened
	again when a start changed the stream or the model was built again. */
- (nullable id<NFKLiveStream>)streamOnBackend:(id<NFKInferenceBackend>)backend generation:(NSUInteger)generation error:(NSError **)error
{
	id<NFKInferenceBackend> hosted = [backend isKindOfClass:NFKDefaultedBackend.class] ? ((NFKDefaultedBackend *)backend).backend : backend;
	@synchronized (self) {
		if (self.stream != nil && self.streamBackend == hosted && self.generation == generation) {
			return self.stream;
		}
		[self.stream close];
		self.stream = NFKInferenceOpenLiveStream(backend, self.parameters, error);
		self.streamBackend = hosted;
		return self.stream;
	}
}

- (void)answer:(NFKServedLiveItem *)item result:(nullable NFKInferenceResult *)result error:(nullable NSError *)error milliseconds:(double)milliseconds
{
	NSArray *dropped = nil;
	BOOL json = NO;
	@synchronized (self) {
		dropped = [self.dropped copy];
		[self.dropped removeAllObjects];
		json = self.valueObjectsAsJSON;
	}
	if (result == nil) {
		[self sendError:error ?: NFKServedLiveError(kNFKError_InferenceBackendFailure, @"the unit failed and reported no error") sequence:item.sequence];
		return;
	}
	NSMutableDictionary<NSString *, id> *remaining = [NSMutableDictionary dictionary];
	NSMutableArray<NSDictionary *> *buffers = [NSMutableArray array];
	NSMutableData *payload = [NSMutableData data];
	for (NSString *key in [result.outputs.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSData *bytes = nil;
		NSDictionary *buffer = NFKServedLiveOutputBuffer(result.outputs[key], key, &bytes);
		if (buffer == nil) {
			remaining[key] = result.outputs[key];
			continue;
		}
		NSMutableDictionary *placed = [buffer mutableCopy];
		placed[@"offset"] = @(payload.length);
		placed[@"length"] = @(bytes.length);
		[buffers addObject:placed];
		[payload appendData:bytes];
	}
	NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	encoder.encodesValueObjectsAsJSON = json;
	NSError *encodeError = nil;
	id encoded = [encoder JSONObjectForResult:[NFKInferenceResult resultWithOutputs:remaining] error:&encodeError];
	if (encoded == nil) {
		[self sendError:encodeError sequence:item.sequence];
		return;
	}
	NSMutableDictionary *message = [@{ @"type": @"result", @"outputs": encoded[@"outputs"] ?: @{}, @"milliseconds": @(milliseconds),
									   @"dropped": dropped } mutableCopy];
	message[@"seq"] = item.sequence;
	if (buffers.count == 0) {
		[self send:message];
		return;
	}
	message[@"buffers"] = buffers;
	[self.socket sendBinary:NFKServedLiveFramed(message, payload)];
}

- (void)end
{
	NSArray<NFKServedLiveItem *> *abandoned = nil;
	id<NFKLiveStream> stream = nil;
	@synchronized (self) {
		self.ended = YES;
		abandoned = [self.queue copy];
		[self.queue removeAllObjects];
		stream = self.stream;
		self.stream = nil;
	}
	for (NFKServedLiveItem *item in abandoned) {
		for (NSURL *file in item.temporaryFiles) {
			[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
		}
	}
	[stream close];
}

@end

@implementation NFKServedLive

+ (void)serveRequest:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server
{
	NFKHTTPWebSocket *socket = [response acceptWebSocketForRequest:request];
	if (socket == nil) {
		return;
	}
	NFKServedLiveSession *session = [[NFKServedLiveSession alloc] init];
	session.server = server;
	session.socket = socket;
	session.request = request;
	session.response = response;
	session.parameters = @{};
	session.queue = [NSMutableArray array];
	session.dropped = [NSMutableArray array];
	// The handlers hold the session for the life of the socket, which releases them when it closes.
	socket.messageHandler = ^(NSData *binary, NSString *text) {
		[session receiveBinary:binary text:text];
	};
	socket.closeHandler = ^{
		[session end];
	};
}

@end
