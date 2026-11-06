//
//  NFKHTTPServer.m
//  InferKit
//

#import "NFKHTTPServer.h"
#import <CommonCrypto/CommonDigest.h>
#import <Network/Network.h>
#import <arpa/inet.h>
#import <netinet/in.h>

static const NSUInteger NFKHTTPMaximumHeadBytes = 64 * 1024;

/*! Marks a listener's queue, where its cancelled state arrives, so stop does not wait for it there. */
static char NFKHTTPListenerQueueKey;

static NSString *NFKHTTPReasonPhrase(NSInteger status)
{
	NSDictionary<NSNumber *, NSString *> *phrases = @{ @200: @"OK", @201: @"Created", @202: @"Accepted",
													   @204: @"No Content", @301: @"Moved Permanently", @302: @"Found",
													   @304: @"Not Modified", @400: @"Bad Request", @409: @"Conflict",
													   @401: @"Unauthorized", @403: @"Forbidden", @404: @"Not Found",
													   @405: @"Method Not Allowed", @408: @"Request Timeout",
													   @413: @"Payload Too Large", @415: @"Unsupported Media Type", @421: @"Misdirected Request",
													   @422: @"Unprocessable Content", @429: @"Too Many Requests",
													   @431: @"Request Header Fields Too Large",
													   @500: @"Internal Server Error", @501: @"Not Implemented",
													   @502: @"Bad Gateway", @503: @"Service Unavailable" };
	return phrases[@(status)] ?: @"Status";
}

static NSData *NFKHTTPLineBreak(void)
{
	return [NSData dataWithBytes:"\r\n" length:2];
}

static NSData *NFKHTTPHeadEnd(void)
{
	return [NSData dataWithBytes:"\r\n\r\n" length:4];
}

#pragma mark - Request

@interface NFKHTTPRequest ()
@property (nonatomic, copy, readwrite) NSString *method;
@property (nonatomic, copy, readwrite) NSString *path;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSString *> *query;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, NSString *> *headers;
@property (nonatomic, copy, readwrite) NSData *body;
@property (nonatomic, assign, readwrite) BOOL fromLoopback;
@property (nonatomic, copy, readwrite) NSString *clientAddress;
@property (nonatomic, assign, readwrite) BOOL firstOnConnection;
@end

@implementation NFKHTTPRequest

- (nullable NSString *)valueForHeader:(NSString *)name
{
	return self.headers[name.lowercaseString];
}

@end

#pragma mark - Form parts

@interface NFKHTTPFormPart ()
@property (nonatomic, copy, readwrite) NSString *name;
@property (nonatomic, copy, readwrite, nullable) NSString *filename;
@property (nonatomic, copy, readwrite, nullable) NSString *contentType;
@property (nonatomic, copy, readwrite) NSData *data;
@end

@implementation NFKHTTPFormPart

- (NSString *)stringValue
{
	return [[NSString alloc] initWithData:self.data encoding:NSUTF8StringEncoding] ?: @"";
}

@end

/*! A header parameter's value (boundary=…, name="…"), quoted or bare. */
static NSString * _Nullable NFKHTTPHeaderParameter(NSString *header, NSString *parameter)
{
	for (NSString *piece in [header componentsSeparatedByString:@";"]) {
		NSString *trimmed = [piece stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		NSString *prefix = [parameter stringByAppendingString:@"="];
		if ([trimmed.lowercaseString hasPrefix:prefix]) {
			NSString *value = [trimmed substringFromIndex:prefix.length];
			if (value.length >= 2 && [value hasPrefix:@"\""] && [value hasSuffix:@"\""]) {
				value = [value substringWithRange:NSMakeRange(1, value.length - 2)];
			}
			return value;
		}
	}
	return nil;
}

NSArray<NFKHTTPFormPart *> * _Nullable NFKHTTPFormParts(NFKHTTPRequest *request)
{
	NSString *contentType = [request valueForHeader:@"content-type"] ?: @"";
	if (![contentType.lowercaseString hasPrefix:@"multipart/form-data"]) {
		return nil;
	}
	NSString *boundary = NFKHTTPHeaderParameter(contentType, @"boundary");
	if (boundary.length == 0) {
		return nil;
	}
	NSData *body = request.body;
	NSData *delimiter = [[@"--" stringByAppendingString:boundary] dataUsingEncoding:NSUTF8StringEncoding];
	NSRange found = [body rangeOfData:delimiter options:0 range:NSMakeRange(0, body.length)];
	NSMutableArray<NFKHTTPFormPart *> *parts = [NSMutableArray array];
	while (found.location != NSNotFound) {
		NSUInteger start = NSMaxRange(found);
		if (start + 2 <= body.length && memcmp((const char *)body.bytes + start, "--", 2) == 0) {
			return parts;
		}
		start += 2;
		NSRange next = start <= body.length
			? [body rangeOfData:delimiter options:0 range:NSMakeRange(start, body.length - start)]
			: NSMakeRange(NSNotFound, 0);
		NSRange headEnd = start <= body.length
			? [body rangeOfData:NFKHTTPHeadEnd() options:0 range:NSMakeRange(start, body.length - start)]
			: NSMakeRange(NSNotFound, 0);
		if (next.location == NSNotFound || headEnd.location == NSNotFound || headEnd.location > next.location) {
			return nil;
		}
		NSString *head = [[NSString alloc] initWithData:[body subdataWithRange:NSMakeRange(start, headEnd.location - start)]
											   encoding:NSUTF8StringEncoding] ?: @"";
		NFKHTTPFormPart *part = [[NFKHTTPFormPart alloc] init];
		for (NSString *line in [head componentsSeparatedByString:@"\r\n"]) {
			NSRange colon = [line rangeOfString:@":"];
			if (colon.location == NSNotFound) {
				continue;
			}
			NSString *name = [line substringToIndex:colon.location].lowercaseString;
			NSString *value = [[line substringFromIndex:colon.location + 1] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
			if ([name isEqualToString:@"content-disposition"]) {
				part.name = NFKHTTPHeaderParameter(value, @"name") ?: @"";
				part.filename = NFKHTTPHeaderParameter(value, @"filename");
			} else if ([name isEqualToString:@"content-type"]) {
				part.contentType = value;
			}
		}
		NSUInteger dataStart = NSMaxRange(headEnd);
		NSUInteger dataEnd = next.location >= 2 ? next.location - 2 : next.location;
		part.data = dataEnd > dataStart ? [body subdataWithRange:NSMakeRange(dataStart, dataEnd - dataStart)] : [NSData data];
		if (part.name != nil) {
			[parts addObject:part];
		}
		found = next;
	}
	return parts;
}

#pragma mark - Chunked request bodies

typedef NS_ENUM(NSInteger, NFKHTTPChunkedState) {
	NFKHTTPChunkedComplete,
	NFKHTTPChunkedIncomplete,
	NFKHTTPChunkedMalformed,
	NFKHTTPChunkedTooLarge,
};

static NFKHTTPChunkedState NFKHTTPDecodeChunked(NSData *buffer, NSUInteger start, NSUInteger limit,
												NSData * _Nullable * _Nonnull outBody, NSUInteger *outEnd)
{
	NSMutableData *body = [NSMutableData data];
	NSUInteger position = start;
	while (YES) {
		NSRange lineEnd = position <= buffer.length
			? [buffer rangeOfData:NFKHTTPLineBreak() options:0 range:NSMakeRange(position, buffer.length - position)]
			: NSMakeRange(NSNotFound, 0);
		if (lineEnd.location == NSNotFound) {
			return NFKHTTPChunkedIncomplete;
		}
		NSString *sizeLine = [[NSString alloc] initWithData:[buffer subdataWithRange:NSMakeRange(position, lineEnd.location - position)]
												   encoding:NSASCIIStringEncoding];
		sizeLine = [[sizeLine componentsSeparatedByString:@";"].firstObject stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		NSScanner *scanner = [NSScanner scannerWithString:sizeLine ?: @""];
		unsigned long long size = 0;
		if (sizeLine.length == 0 || ![scanner scanHexLongLong:&size] || !scanner.isAtEnd) {
			return NFKHTTPChunkedMalformed;
		}
		position = NSMaxRange(lineEnd);
		if (size == 0) {
			if (position + 2 <= buffer.length && memcmp((const char *)buffer.bytes + position, "\r\n", 2) == 0) {
				*outEnd = position + 2;
			} else {
				NSRange trailerEnd = [buffer rangeOfData:NFKHTTPHeadEnd() options:0 range:NSMakeRange(position, buffer.length - position)];
				if (trailerEnd.location == NSNotFound) {
					return NFKHTTPChunkedIncomplete;
				}
				*outEnd = NSMaxRange(trailerEnd);
			}
			*outBody = body;
			return NFKHTTPChunkedComplete;
		}
		if (body.length + size > limit) {
			return NFKHTTPChunkedTooLarge;
		}
		if (position + size + 2 > buffer.length) {
			return NFKHTTPChunkedIncomplete;
		}
		[body appendBytes:(const char *)buffer.bytes + position length:(NSUInteger)size];
		if (memcmp((const char *)buffer.bytes + position + size, "\r\n", 2) != 0) {
			return NFKHTTPChunkedMalformed;
		}
		position += (NSUInteger)size + 2;
	}
}

#pragma mark - Connection

@class NFKHTTPConnection;

@interface NFKHTTPListener ()
@property (nonatomic, copy) NFKHTTPRequestHandler handler;
@property (nonatomic, strong, nullable) nw_listener_t listener;
/*! Signalled when the running listener reaches the cancelled state, which is when its port is free. */
@property (nonatomic, strong, nullable) dispatch_semaphore_t listenerCancelled;
@property (nonatomic, strong) dispatch_queue_t listenerQueue;
@property (nonatomic, strong) NSMutableSet<NFKHTTPConnection *> *connections;
@property (nonatomic, assign, readwrite) uint16_t boundPort;
- (void)connectionDidClose:(NFKHTTPConnection *)connection;
@end

@interface NFKHTTPResponse ()
@property (nonatomic, strong) NFKHTTPConnection *owner;
@property (nonatomic, assign) BOOL keepAlive;
@property (nonatomic, assign, readwrite) BOOL headersSent;
@property (nonatomic, assign, readwrite) BOOL finished;
@property (nonatomic, assign, readwrite) BOOL disconnected;
/*! The handler disconnectHandler reads and writes on the reply's lock. */
@property (nonatomic, copy, nullable) void (^storedDisconnectHandler)(void);
- (instancetype)initWithOwner:(NFKHTTPConnection *)owner keepAlive:(BOOL)keepAlive;
- (void)connectionDidDrop;
@end

@interface NFKHTTPConnection : NSObject
@property (nonatomic, strong) nw_connection_t connection;
@property (nonatomic, strong) dispatch_queue_t queue;
@property (nonatomic, weak) NFKHTTPListener *listener;
@property (nonatomic, copy) NFKHTTPRequestHandler handler;
@property (nonatomic, strong) NSMutableData *buffer;
@property (nonatomic, strong, nullable) NFKHTTPResponse *current;
@property (nonatomic, strong, nullable) dispatch_source_t idleTimer;
@property (nonatomic, assign) BOOL closed;
@property (nonatomic, assign) BOOL sentContinue;
@property (nonatomic, assign) BOOL fromLoopback;
@property (nonatomic, copy) NSString *clientAddress;
@property (nonatomic, assign) NSUInteger requestCount;
@property (nonatomic, assign) NSUInteger maximumBodyBytes;
@property (nonatomic, assign) NSTimeInterval idleTimeout;
@property (nonatomic, assign) CFAbsoluteTime lastActivity;
/*! The socket the connection became, after which the buffer holds WebSocket frames. */
@property (nonatomic, strong, nullable) NFKHTTPWebSocket *webSocket;
@end

#pragma mark - WebSocket

typedef NS_ENUM(uint8_t, NFKWebSocketOpcode) {
	NFKWebSocketOpcodeContinuation = 0x0,
	NFKWebSocketOpcodeText = 0x1,
	NFKWebSocketOpcodeBinary = 0x2,
	NFKWebSocketOpcodeClose = 0x8,
	NFKWebSocketOpcodePing = 0x9,
	NFKWebSocketOpcodePong = 0xA,
};

@interface NFKHTTPWebSocket ()
@property (nonatomic, weak) NFKHTTPConnection *owner;
@property (nonatomic, assign, readwrite) BOOL closed;
@property (nonatomic, assign) BOOL closeSent;
/*! The opcode and bytes of a fragmented message still arriving, or 0 and nil. */
@property (nonatomic, assign) NFKWebSocketOpcode fragmentOpcode;
@property (nonatomic, strong, nullable) NSMutableData *fragments;
- (instancetype)initWithOwner:(NFKHTTPConnection *)owner;
- (void)receiveFrameWithOpcode:(NFKWebSocketOpcode)opcode final:(BOOL)final payload:(NSData *)payload;
- (void)connectionDidDrop;
@end

/*! An unmasked server frame: FIN set, the opcode, and the payload. */
static NSData *NFKWebSocketFrame(NFKWebSocketOpcode opcode, NSData *payload)
{
	NSMutableData *frame = [NSMutableData dataWithCapacity:payload.length + 10];
	uint8_t first = 0x80 | opcode;
	[frame appendBytes:&first length:1];
	if (payload.length < 126) {
		uint8_t length = (uint8_t)payload.length;
		[frame appendBytes:&length length:1];
	} else if (payload.length <= UINT16_MAX) {
		uint8_t marker = 126;
		uint16_t length = CFSwapInt16HostToBig((uint16_t)payload.length);
		[frame appendBytes:&marker length:1];
		[frame appendBytes:&length length:2];
	} else {
		uint8_t marker = 127;
		uint64_t length = CFSwapInt64HostToBig(payload.length);
		[frame appendBytes:&marker length:1];
		[frame appendBytes:&length length:8];
	}
	[frame appendData:payload];
	return frame;
}

/*! The Sec-WebSocket-Accept value for a client's key. */
static NSString *NFKWebSocketAcceptValue(NSString *key)
{
	NSData *joined = [[key stringByAppendingString:@"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"] dataUsingEncoding:NSUTF8StringEncoding];
	unsigned char digest[CC_SHA1_DIGEST_LENGTH];
	CC_SHA1(joined.bytes, (CC_LONG)joined.length, digest);
	return [[NSData dataWithBytes:digest length:sizeof digest] base64EncodedStringWithOptions:0];
}

/*! Whether an address endpoint is a loopback address, IPv4-mapped IPv6 included. */
static BOOL NFKHTTPEndpointIsLoopback(nw_endpoint_t _Nullable endpoint)
{
	if (endpoint == nil || nw_endpoint_get_type(endpoint) != nw_endpoint_type_address) {
		return NO;
	}
	const struct sockaddr *address = nw_endpoint_get_address(endpoint);
	if (address == NULL) {
		return NO;
	}
	if (address->sa_family == AF_INET) {
		in_addr_t host = ntohl(((const struct sockaddr_in *)(const void *)address)->sin_addr.s_addr);
		return (host >> 24) == 127;
	}
	if (address->sa_family == AF_INET6) {
		const struct in6_addr *host = &((const struct sockaddr_in6 *)(const void *)address)->sin6_addr;
		return IN6_IS_ADDR_LOOPBACK(host) || (IN6_IS_ADDR_V4MAPPED(host) && host->s6_addr[12] == 127);
	}
	return NO;
}

/*! An address endpoint's IP address as text, an IPv4-mapped IPv6 address as its IPv4 form; empty for
	any other endpoint. */
static NSString *NFKHTTPEndpointAddress(nw_endpoint_t _Nullable endpoint)
{
	if (endpoint == nil || nw_endpoint_get_type(endpoint) != nw_endpoint_type_address) {
		return @"";
	}
	const struct sockaddr *address = nw_endpoint_get_address(endpoint);
	char text[INET6_ADDRSTRLEN] = { 0 };
	if (address != NULL && address->sa_family == AF_INET) {
		inet_ntop(AF_INET, &((const struct sockaddr_in *)(const void *)address)->sin_addr, text, sizeof(text));
	} else if (address != NULL && address->sa_family == AF_INET6) {
		const struct in6_addr *host = &((const struct sockaddr_in6 *)(const void *)address)->sin6_addr;
		if (IN6_IS_ADDR_V4MAPPED(host)) {
			inet_ntop(AF_INET, &host->s6_addr[12], text, sizeof(text));
		} else {
			inet_ntop(AF_INET6, host, text, sizeof(text));
		}
	}
	return [NSString stringWithUTF8String:text] ?: @"";
}

@implementation NFKHTTPConnection

- (void)start
{
	self.buffer = [NSMutableData data];
	self.lastActivity = CFAbsoluteTimeGetCurrent();
	nw_endpoint_t peer = nw_connection_copy_endpoint(self.connection);
	self.fromLoopback = NFKHTTPEndpointIsLoopback(peer);
	self.clientAddress = NFKHTTPEndpointAddress(peer);
	nw_connection_set_queue(self.connection, self.queue);
	nw_connection_set_state_changed_handler(self.connection, ^(nw_connection_state_t state, nw_error_t error) {
		if (state == nw_connection_state_failed || state == nw_connection_state_cancelled) {
			[self dropConnection];
		}
	});
	nw_connection_start(self.connection);
	[self startIdleTimer];
	[self receive];
}

- (void)startIdleTimer
{
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, self.queue);
	uint64_t interval = (uint64_t)(MAX(1.0, self.idleTimeout / 4.0) * NSEC_PER_SEC);
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)interval), interval, NSEC_PER_SEC / 4);
	__weak NFKHTTPConnection *weakSelf = self;
	dispatch_source_set_event_handler(timer, ^{
		NFKHTTPConnection *connection = weakSelf;
		if (connection != nil && connection.current == nil
			&& CFAbsoluteTimeGetCurrent() - connection.lastActivity > connection.idleTimeout) {
			[connection closeGracefully];
		}
	});
	dispatch_resume(timer);
	self.idleTimer = timer;
}

- (void)receive
{
	nw_connection_receive(self.connection, 1, 64 * 1024, ^(dispatch_data_t content, nw_content_context_t context, bool isComplete, nw_error_t error) {
		if (self.closed) {
			return;
		}
		if (content != nil) {
			[self.buffer appendData:(NSData *)content];
			self.lastActivity = CFAbsoluteTimeGetCurrent();
		}
		if (error != nil) {
			[self dropConnection];
			return;
		}
		[self processBuffer];
		if (!isComplete) {
			[self receive];
			return;
		}
		// The client closed its side. A reply still pending is abandoned; a client that reads
		// after shutting its writes down is not one this server serves.
		if (self.current != nil) {
			[self dropConnection];
		} else {
			[self closeGracefully];
		}
	});
}

- (void)processBuffer
{
	if (self.webSocket != nil) {
		[self processFrames];
		return;
	}
	while (!self.closed && self.current == nil && self.buffer.length > 0) {
		NSRange headEnd = [self.buffer rangeOfData:NFKHTTPHeadEnd() options:0 range:NSMakeRange(0, self.buffer.length)];
		if (headEnd.location == NSNotFound || headEnd.location > NFKHTTPMaximumHeadBytes) {
			if (headEnd.location != NSNotFound || self.buffer.length > NFKHTTPMaximumHeadBytes) {
				[self rejectWithStatus:431 message:@"the request head is too large"];
			}
			return;
		}
		NSString *head = [[NSString alloc] initWithData:[self.buffer subdataWithRange:NSMakeRange(0, headEnd.location)]
											   encoding:NSUTF8StringEncoding]
			?: [[NSString alloc] initWithData:[self.buffer subdataWithRange:NSMakeRange(0, headEnd.location)]
									 encoding:NSISOLatin1StringEncoding];
		NSArray<NSString *> *lines = [head componentsSeparatedByString:@"\r\n"];
		NSArray<NSString *> *requestLine = [lines.firstObject componentsSeparatedByString:@" "];
		if (requestLine.count != 3 || ![requestLine[2] hasPrefix:@"HTTP/1."]) {
			[self rejectWithStatus:400 message:@"the request line is malformed"];
			return;
		}
		NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
		for (NSString *line in [lines subarrayWithRange:NSMakeRange(1, lines.count - 1)]) {
			NSRange colon = [line rangeOfString:@":"];
			if (colon.location == NSNotFound) {
				continue;
			}
			NSString *name = [line substringToIndex:colon.location].lowercaseString;
			NSString *value = [[line substringFromIndex:colon.location + 1] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
			headers[name] = headers[name] != nil ? [NSString stringWithFormat:@"%@, %@", headers[name], value] : value;
		}

		NSUInteger bodyStart = NSMaxRange(headEnd);
		NSData *body = [NSData data];
		NSUInteger consumed = bodyStart;
		if ([headers[@"transfer-encoding"].lowercaseString containsString:@"chunked"]) {
			NSData *decoded = nil;
			switch (NFKHTTPDecodeChunked(self.buffer, bodyStart, self.maximumBodyBytes, &decoded, &consumed)) {
				case NFKHTTPChunkedIncomplete:
					[self sendContinueIfExpected:headers];
					return;
				case NFKHTTPChunkedMalformed:
					[self rejectWithStatus:400 message:@"the chunked body is malformed"];
					return;
				case NFKHTTPChunkedTooLarge:
					[self rejectWithStatus:413 message:@"the request body exceeds the server's limit"];
					return;
				case NFKHTTPChunkedComplete:
					body = decoded;
					break;
			}
		} else if (headers[@"content-length"] != nil) {
			NSScanner *scanner = [NSScanner scannerWithString:headers[@"content-length"]];
			unsigned long long length = 0;
			if (![scanner scanUnsignedLongLong:&length] || !scanner.isAtEnd) {
				[self rejectWithStatus:400 message:@"the Content-Length is malformed"];
				return;
			}
			if (length > self.maximumBodyBytes) {
				[self rejectWithStatus:413 message:@"the request body exceeds the server's limit"];
				return;
			}
			if (self.buffer.length < bodyStart + length) {
				[self sendContinueIfExpected:headers];
				return;
			}
			body = [self.buffer subdataWithRange:NSMakeRange(bodyStart, (NSUInteger)length)];
			consumed = bodyStart + (NSUInteger)length;
		}
		[self.buffer replaceBytesInRange:NSMakeRange(0, consumed) withBytes:NULL length:0];
		self.sentContinue = NO;
		[self dispatchRequestWithLine:requestLine headers:headers body:body];
	}
	if (self.webSocket != nil) {
		[self processFrames];
	}
}

/*! Reads every whole frame in the buffer and hands its message to the socket. */
- (void)processFrames
{
	NFKHTTPWebSocket *socket = self.webSocket;
	while (!self.closed && !socket.closed && self.buffer.length >= 2) {
		const uint8_t *bytes = self.buffer.bytes;
		BOOL final = (bytes[0] & 0x80) != 0;
		NFKWebSocketOpcode opcode = bytes[0] & 0x0F;
		BOOL masked = (bytes[1] & 0x80) != 0;
		uint64_t length = bytes[1] & 0x7F;
		NSUInteger offset = 2;
		if (length == 126) {
			if (self.buffer.length < 4) {
				return;
			}
			length = ((uint64_t)bytes[2] << 8) | bytes[3];
			offset = 4;
		} else if (length == 127) {
			if (self.buffer.length < 10) {
				return;
			}
			length = 0;
			for (NSUInteger index = 0; index < 8; index++) {
				length = (length << 8) | bytes[2 + index];
			}
			offset = 10;
		}
		if ((bytes[0] & 0x70) != 0 || !masked) {
			[socket closeWithCode:1002 reason:(bytes[0] & 0x70) != 0 ? @"a reserved bit is set" : @"a client frame is not masked"];
			return;
		}
		BOOL control = (opcode & 0x08) != 0;
		BOOL known = opcode <= NFKWebSocketOpcodeBinary || opcode == NFKWebSocketOpcodeClose || opcode == NFKWebSocketOpcodePing
			|| opcode == NFKWebSocketOpcodePong;
		if (!known || (control && (!final || length > 125))) {
			[socket closeWithCode:1002 reason:@"the frame breaks the protocol"];
			return;
		}
		if (length + socket.fragments.length > self.maximumBodyBytes) {
			[socket closeWithCode:1009 reason:@"the message exceeds the server's limit"];
			return;
		}
		if (self.buffer.length < offset + 4 + length) {
			return;
		}
		uint8_t mask[4];
		memcpy(mask, bytes + offset, 4);
		NSMutableData *payload = [NSMutableData dataWithBytes:bytes + offset + 4 length:(NSUInteger)length];
		uint8_t *unmasked = payload.mutableBytes;
		for (NSUInteger index = 0; index < payload.length; index++) {
			unmasked[index] ^= mask[index % 4];
		}
		[self.buffer replaceBytesInRange:NSMakeRange(0, offset + 4 + (NSUInteger)length) withBytes:NULL length:0];
		[socket receiveFrameWithOpcode:opcode final:final payload:payload];
	}
}

- (void)sendContinueIfExpected:(NSDictionary<NSString *, NSString *> *)headers
{
	if (self.sentContinue || ![headers[@"expect"].lowercaseString isEqualToString:@"100-continue"]) {
		return;
	}
	self.sentContinue = YES;
	[self write:[@"HTTP/1.1 100 Continue\r\n\r\n" dataUsingEncoding:NSASCIIStringEncoding] completion:nil];
}

- (void)dispatchRequestWithLine:(NSArray<NSString *> *)requestLine
						headers:(NSDictionary<NSString *, NSString *> *)headers
						   body:(NSData *)body
{
	NSString *target = requestLine[1];
	if ([target containsString:@"://"]) {
		NSURLComponents *absolute = [NSURLComponents componentsWithString:target];
		target = absolute.percentEncodedQuery.length > 0
			? [NSString stringWithFormat:@"%@?%@", absolute.percentEncodedPath, absolute.percentEncodedQuery]
			: absolute.percentEncodedPath ?: @"/";
	}
	NSRange question = [target rangeOfString:@"?"];
	NSString *rawPath = question.location == NSNotFound ? target : [target substringToIndex:question.location];
	NSMutableDictionary<NSString *, NSString *> *query = [NSMutableDictionary dictionary];
	if (question.location != NSNotFound) {
		for (NSString *pair in [[target substringFromIndex:question.location + 1] componentsSeparatedByString:@"&"]) {
			// A value keeps every "=" after the first, as a base64 key's padding does.
			NSRange equals = [pair rangeOfString:@"="];
			NSString *rawName = equals.location == NSNotFound ? pair : [pair substringToIndex:equals.location];
			NSString *rawValue = equals.location == NSNotFound ? @"" : [pair substringFromIndex:NSMaxRange(equals)];
			NSString *name = [[rawName stringByReplacingOccurrencesOfString:@"+" withString:@" "] stringByRemovingPercentEncoding];
			NSString *value = [[rawValue stringByReplacingOccurrencesOfString:@"+" withString:@" "] stringByRemovingPercentEncoding];
			if (name.length > 0) {
				query[name] = value ?: @"";
			}
		}
	}

	NFKHTTPRequest *request = [[NFKHTTPRequest alloc] init];
	request.method = requestLine[0].uppercaseString;
	request.path = [rawPath stringByRemovingPercentEncoding] ?: rawPath;
	request.query = query;
	request.headers = headers;
	request.body = body;
	request.fromLoopback = self.fromLoopback;
	request.clientAddress = self.clientAddress;
	request.firstOnConnection = self.requestCount == 0;
	self.requestCount += 1;

	NSString *connectionHeader = headers[@"connection"].lowercaseString ?: @"";
	BOOL keepAlive = [requestLine[2] isEqualToString:@"HTTP/1.1"]
		? ![connectionHeader containsString:@"close"]
		: [connectionHeader containsString:@"keep-alive"];
	NFKHTTPResponse *response = [[NFKHTTPResponse alloc] initWithOwner:self keepAlive:keepAlive];
	self.current = response;
	self.handler(request, response);
}

- (void)rejectWithStatus:(NSInteger)status message:(NSString *)message
{
	NFKHTTPResponse *response = [[NFKHTTPResponse alloc] initWithOwner:self keepAlive:NO];
	self.current = response;
	[response sendStatus:status JSONObject:@{ @"error": @{ @"message": message, @"type": @"invalid_request_error" } } headers:nil];
}

- (void)write:(NSData *)data completion:(void (^ _Nullable)(void))completion
{
	dispatch_async(self.queue, ^{
		if (self.closed) {
			return;
		}
		dispatch_data_t payload = dispatch_data_create(data.bytes, data.length, self.queue, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
		nw_connection_send(self.connection, payload, NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, false, ^(nw_error_t error) {
			if (error != nil) {
				[self dropConnection];
				return;
			}
			self.lastActivity = CFAbsoluteTimeGetCurrent();
			if (completion != nil) {
				completion();
			}
		});
	});
}

- (void)responseDidFinish:(NFKHTTPResponse *)response
{
	dispatch_async(self.queue, ^{
		if (self.current != response) {
			return;
		}
		self.current = nil;
		if (!response.keepAlive) {
			[self closeGracefully];
			return;
		}
		[self processBuffer];
	});
}

- (void)closeGracefully
{
	if (self.closed) {
		return;
	}
	self.closed = YES;
	[self cancelTimer];
	nw_connection_t connection = self.connection;
	nw_connection_send(connection, NULL, NW_CONNECTION_FINAL_MESSAGE_CONTEXT, true, ^(nw_error_t error) {
		nw_connection_cancel(connection);
	});
	[self.listener connectionDidClose:self];
	[self.webSocket connectionDidDrop];
}

/*! The peer went away, the network failed, or the server is stopping: the pending reply, if any,
	learns of it and the connection is torn down. Runs on the connection's queue. */
- (void)dropConnection
{
	if (!self.closed) {
		self.closed = YES;
		[self cancelTimer];
		nw_connection_cancel(self.connection);
		[self.listener connectionDidClose:self];
	}
	NFKHTTPResponse *pending = self.current;
	self.current = nil;
	[pending connectionDidDrop];
	[self.webSocket connectionDidDrop];
}

- (void)cancelTimer
{
	if (self.idleTimer != nil) {
		dispatch_source_cancel(self.idleTimer);
		self.idleTimer = nil;
	}
}

@end

#pragma mark - Response

@implementation NFKHTTPResponse

- (instancetype)initWithOwner:(NFKHTTPConnection *)owner keepAlive:(BOOL)keepAlive
{
	self = [super init];
	if (self != nil) {
		_owner = owner;
		_keepAlive = keepAlive;
	}
	return self;
}

- (NSMutableData *)headWithStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers extra:(NSDictionary<NSString *, NSString *> *)extra
{
	NSMutableString *head = [NSMutableString stringWithFormat:@"HTTP/1.1 %ld %@\r\n", (long)status, NFKHTTPReasonPhrase(status)];
	NSDictionary<NSString *, NSString *> * _Nullable (^provider)(void) = self.headerProvider;
	NSMutableDictionary<NSString *, NSString *> *fields = [NSMutableDictionary dictionaryWithDictionary:(provider != nil ? provider() : nil) ?: @{}];
	[fields addEntriesFromDictionary:headers ?: @{}];
	[fields addEntriesFromDictionary:extra];
	fields[@"Server"] = @"InferKit";
	fields[@"Connection"] = self.keepAlive ? @"keep-alive" : @"close";
	for (NSString *name in fields) {
		[head appendFormat:@"%@: %@\r\n", name, fields[name]];
	}
	[head appendString:@"\r\n"];
	return [[head dataUsingEncoding:NSUTF8StringEncoding] mutableCopy];
}

- (BOOL)claimHeaders
{
	@synchronized (self) {
		if (self.headersSent || self.finished || self.disconnected) {
			return NO;
		}
		self.headersSent = YES;
		return YES;
	}
}

- (BOOL)claimFinish
{
	@synchronized (self) {
		if (self.finished || self.disconnected) {
			return NO;
		}
		self.finished = YES;
		// A finished reply never fires the handler; releasing it frees a handler that captures this reply.
		self.storedDisconnectHandler = nil;
		return YES;
	}
}

- (void)sendStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers body:(nullable NSData *)body
{
	if (![self claimHeaders] || ![self claimFinish]) {
		return;
	}
	NSMutableData *message = [self headWithStatus:status headers:headers
											extra:@{ @"Content-Length": [NSString stringWithFormat:@"%lu", (unsigned long)body.length] }];
	if (body != nil) {
		[message appendData:body];
	}
	NFKHTTPConnection *owner = self.owner;
	[owner write:message completion:^{
		[owner responseDidFinish:self];
	}];
}

- (void)sendStatus:(NSInteger)status JSONObject:(id)object headers:(nullable NSDictionary<NSString *, NSString *> *)headers
{
	// dataWithJSONObject: raises on a value JSON cannot carry, such as a non-finite number a backend returned.
	if (![NSJSONSerialization isValidJSONObject:object]) {
		status = 500;
		object = @{ @"error": @{ @"message": @"the reply holds a value JSON cannot carry, such as a non-finite number",
								 @"type": @"server_error" } };
	}
	NSData *body = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL] ?: [NSData data];
	NSMutableDictionary<NSString *, NSString *> *fields = [NSMutableDictionary dictionaryWithDictionary:headers ?: @{}];
	fields[@"Content-Type"] = @"application/json";
	[self sendStatus:status headers:fields body:body];
}

- (void)beginStreamWithStatus:(NSInteger)status headers:(nullable NSDictionary<NSString *, NSString *> *)headers
{
	if (![self claimHeaders]) {
		return;
	}
	[self.owner write:[self headWithStatus:status headers:headers extra:@{ @"Transfer-Encoding": @"chunked", @"Cache-Control": @"no-cache" }]
		   completion:nil];
}

- (void)sendStreamData:(NSData *)data
{
	@synchronized (self) {
		if (!self.headersSent || self.finished || self.disconnected || data.length == 0) {
			return;
		}
	}
	NSMutableData *chunk = [[[NSString stringWithFormat:@"%lx\r\n", (unsigned long)data.length] dataUsingEncoding:NSASCIIStringEncoding] mutableCopy];
	[chunk appendData:data];
	[chunk appendData:NFKHTTPLineBreak()];
	[self.owner write:chunk completion:nil];
}

- (void)sendEventData:(NSString *)payload
{
	[self sendStreamData:[[NSString stringWithFormat:@"data: %@\n\n", payload] dataUsingEncoding:NSUTF8StringEncoding]];
}

- (void)sendEventNamed:(NSString *)name data:(NSString *)payload
{
	[self sendStreamData:[[NSString stringWithFormat:@"event: %@\ndata: %@\n\n", name, payload] dataUsingEncoding:NSUTF8StringEncoding]];
}

- (void)endStream
{
	if (!self.headersSent || ![self claimFinish]) {
		return;
	}
	NFKHTTPConnection *owner = self.owner;
	[owner write:[@"0\r\n\r\n" dataUsingEncoding:NSASCIIStringEncoding] completion:^{
		[owner responseDidFinish:self];
	}];
}

- (nullable NFKHTTPWebSocket *)acceptWebSocketForRequest:(NFKHTTPRequest *)request
{
	NSString *key = [request valueForHeader:@"sec-websocket-key"];
	BOOL upgrade = [request.method isEqualToString:@"GET"]
		&& [[request valueForHeader:@"upgrade"].lowercaseString isEqualToString:@"websocket"]
		&& [[request valueForHeader:@"connection"].lowercaseString containsString:@"upgrade"]
		&& [[request valueForHeader:@"sec-websocket-version"] isEqualToString:@"13"]
		&& [[NSData alloc] initWithBase64EncodedString:key ?: @"" options:0].length == 16;
	if (!upgrade) {
		[self sendStatus:400 JSONObject:@{ @"error": @{ @"message": @"the route takes a WebSocket upgrade (RFC 6455, version 13)",
														 @"type": @"invalid_request_error" } }
				 headers:@{ @"Sec-WebSocket-Version": @"13" }];
		return nil;
	}
	if (![self claimHeaders]) {
		return nil;
	}
	NFKHTTPConnection *owner = self.owner;
	NFKHTTPWebSocket *socket = [[NFKHTTPWebSocket alloc] initWithOwner:owner];
	owner.webSocket = socket;
	NSString *head = [NSString stringWithFormat:@"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
					  "Sec-WebSocket-Accept: %@\r\nServer: InferKit\r\n\r\n", NFKWebSocketAcceptValue(key)];
	[owner write:[head dataUsingEncoding:NSUTF8StringEncoding] completion:nil];
	return socket;
}

- (void)connectionDidDrop
{
	void (^handler)(void) = nil;
	@synchronized (self) {
		if (self.finished || self.disconnected) {
			return;
		}
		self.disconnected = YES;
		handler = self.storedDisconnectHandler;
		self.storedDisconnectHandler = nil;
	}
	if (handler != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), handler);
	}
}

- (nullable void (^)(void))disconnectHandler
{
	@synchronized (self) {
		return self.storedDisconnectHandler;
	}
}

- (void)setDisconnectHandler:(nullable void (^)(void))disconnectHandler
{
	BOOL alreadyGone = NO;
	@synchronized (self) {
		alreadyGone = self.disconnected;
		self.storedDisconnectHandler = alreadyGone || self.finished ? nil : disconnectHandler;
	}
	if (alreadyGone && disconnectHandler != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), disconnectHandler);
	}
}

@end

#pragma mark - WebSocket

@implementation NFKHTTPWebSocket

- (instancetype)initWithOwner:(NFKHTTPConnection *)owner
{
	self = [super init];
	if (self != nil) {
		_owner = owner;
	}
	return self;
}

/*! Runs on the connection's queue. */
- (void)receiveFrameWithOpcode:(NFKWebSocketOpcode)opcode final:(BOOL)final payload:(NSData *)payload
{
	switch (opcode) {
		case NFKWebSocketOpcodePing:
			[self send:NFKWebSocketOpcodePong payload:payload];
			return;
		case NFKWebSocketOpcodePong:
			return;
		case NFKWebSocketOpcodeClose: {
			uint16_t code = 1000;
			if (payload.length >= 2) {
				const uint8_t *bytes = payload.bytes;
				code = (uint16_t)((bytes[0] << 8) | bytes[1]);
			}
			[self closeWithCode:code reason:nil];
			return;
		}
		case NFKWebSocketOpcodeContinuation:
			if (self.fragments == nil) {
				[self closeWithCode:1002 reason:@"a continuation frame continues no message"];
				return;
			}
			[self.fragments appendData:payload];
			break;
		case NFKWebSocketOpcodeText:
		case NFKWebSocketOpcodeBinary:
			if (self.fragments != nil) {
				[self closeWithCode:1002 reason:@"a new message began inside a fragmented one"];
				return;
			}
			self.fragmentOpcode = opcode;
			self.fragments = [payload mutableCopy];
			break;
	}
	if (!final) {
		return;
	}
	NSData *message = [self.fragments copy];
	NFKWebSocketOpcode kind = self.fragmentOpcode;
	self.fragments = nil;
	NSString *text = nil;
	if (kind == NFKWebSocketOpcodeText) {
		text = [[NSString alloc] initWithData:message encoding:NSUTF8StringEncoding];
		if (text == nil) {
			[self closeWithCode:1007 reason:@"a text message is not UTF-8"];
			return;
		}
	}
	void (^handler)(NSData * _Nullable, NSString * _Nullable) = self.messageHandler;
	if (handler != nil) {
		handler(kind == NFKWebSocketOpcodeBinary ? message : nil, text);
	}
}

- (void)send:(NFKWebSocketOpcode)opcode payload:(NSData *)payload
{
	@synchronized (self) {
		if (self.closeSent) {
			return;
		}
		if (opcode == NFKWebSocketOpcodeClose) {
			self.closeSent = YES;
		}
	}
	NFKHTTPConnection *owner = self.owner;
	[owner write:NFKWebSocketFrame(opcode, payload) completion:opcode == NFKWebSocketOpcodeClose ? ^{
		dispatch_async(owner.queue, ^{
			[owner closeGracefully];
		});
	} : nil];
}

- (void)sendText:(NSString *)text
{
	[self send:NFKWebSocketOpcodeText payload:[text dataUsingEncoding:NSUTF8StringEncoding] ?: [NSData data]];
}

- (void)sendBinary:(NSData *)data
{
	[self send:NFKWebSocketOpcodeBinary payload:data];
}

- (void)closeWithCode:(uint16_t)code reason:(nullable NSString *)reason
{
	NSMutableData *payload = [NSMutableData dataWithCapacity:2 + reason.length];
	uint16_t bigEndian = CFSwapInt16HostToBig(code);
	[payload appendBytes:&bigEndian length:2];
	NSData *text = [reason dataUsingEncoding:NSUTF8StringEncoding];
	[payload appendData:[text subdataWithRange:NSMakeRange(0, MIN(text.length, (NSUInteger)123))]];
	[self send:NFKWebSocketOpcodeClose payload:payload];
	[self connectionDidDrop];
}

/*! Marks the socket closed and runs closeHandler once. */
- (void)connectionDidDrop
{
	void (^handler)(void) = nil;
	@synchronized (self) {
		if (self.closed) {
			return;
		}
		self.closed = YES;
		handler = self.closeHandler;
		self.closeHandler = nil;
		self.messageHandler = nil;
	}
	if (handler != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), handler);
	}
}

@end

#pragma mark - Listener

@implementation NFKHTTPListener

- (instancetype)initWithHandler:(NFKHTTPRequestHandler)handler
{
	self = [super init];
	if (self != nil) {
		_handler = [handler copy];
		_connections = [NSMutableSet set];
		_maximumBodyBytes = 256 * 1024 * 1024;
		_idleTimeout = 120.0;
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
		_listenerQueue = dispatch_queue_create("com.inferkit.http.listener", attributes);
		dispatch_queue_set_specific(_listenerQueue, &NFKHTTPListenerQueueKey, &NFKHTTPListenerQueueKey, NULL);
	}
	return self;
}

- (nw_parameters_t)parameters
{
	id identity = self.TLSIdentity;
	nw_parameters_configure_protocol_block_t configureTLS = NW_PARAMETERS_DISABLE_PROTOCOL;
	if (identity != nil) {
		configureTLS = ^(nw_protocol_options_t options) {
			sec_protocol_options_t security = nw_tls_copy_sec_protocol_options(options);
			sec_identity_t secIdentity = sec_identity_create((__bridge SecIdentityRef)identity);
			if (secIdentity != nil) {
				sec_protocol_options_set_local_identity(security, secIdentity);
			}
		};
	}
	nw_parameters_t parameters = nw_parameters_create_secure_tcp(configureTLS, NW_PARAMETERS_DEFAULT_CONFIGURATION);
	nw_parameters_set_reuse_local_address(parameters, true);
	if (self.loopbackOnly) {
		nw_parameters_set_required_interface_type(parameters, nw_interface_type_loopback);
	}
	return parameters;
}

- (BOOL)startWithError:(NSError * _Nullable *)outError
{
	char port[8];
	snprintf(port, sizeof(port), "%u", (unsigned)self.port);
	nw_listener_t listener = nw_listener_create_with_port(port, [self parameters]);
	if (listener == nil) {
		return [self failWithReason:@"the listener could not be created" underlying:nil error:outError];
	}
	if (self.serviceType.length > 0) {
		nw_advertise_descriptor_t advertisement = nw_advertise_descriptor_create_bonjour_service(self.serviceName.UTF8String, self.serviceType.UTF8String, NULL);
		nw_txt_record_t record = nw_txt_record_create_dictionary();
		[self.TXTRecord enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSString *value, BOOL *stop) {
			const char *bytes = value.UTF8String;
			nw_txt_record_set_key(record, key.UTF8String, (const uint8_t *)bytes, strlen(bytes));
		}];
		nw_advertise_descriptor_set_txt_record_object(advertisement, record);
		nw_listener_set_advertise_descriptor(listener, advertisement);
	}
	nw_listener_set_queue(listener, self.listenerQueue);

	dispatch_semaphore_t settled = dispatch_semaphore_create(0);
	dispatch_semaphore_t cancelled = dispatch_semaphore_create(0);
	__block BOOL ready = NO;
	__block BOOL signalled = NO;
	__block nw_error_t failure = nil;
	nw_listener_set_state_changed_handler(listener, ^(nw_listener_state_t state, nw_error_t error) {
		if (state == nw_listener_state_cancelled) {
			dispatch_semaphore_signal(cancelled);
			return;
		}
		if (signalled || (state != nw_listener_state_ready && state != nw_listener_state_failed)) {
			return;
		}
		ready = state == nw_listener_state_ready;
		failure = error;
		signalled = YES;
		dispatch_semaphore_signal(settled);
	});
	__weak NFKHTTPListener *weakSelf = self;
	nw_listener_set_new_connection_handler(listener, ^(nw_connection_t connection) {
		NFKHTTPListener *strongSelf = weakSelf;
		if (strongSelf == nil) {
			nw_connection_cancel(connection);
			return;
		}
		[strongSelf acceptConnection:connection];
	});
	nw_listener_start(listener);
	if (dispatch_semaphore_wait(settled, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(10 * NSEC_PER_SEC))) != 0 || !ready) {
		nw_listener_cancel(listener);
		return [self failWithReason:@"the server could not listen" underlying:failure error:outError];
	}
	self.listener = listener;
	self.listenerCancelled = cancelled;
	self.boundPort = nw_listener_get_port(listener);
	return YES;
}

- (void)acceptConnection:(nw_connection_t)nwConnection
{
	NFKHTTPConnection *connection = [[NFKHTTPConnection alloc] init];
	connection.connection = nwConnection;
	dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
	connection.queue = dispatch_queue_create("com.inferkit.http.connection", attributes);
	connection.listener = self;
	connection.handler = self.handler;
	connection.maximumBodyBytes = self.maximumBodyBytes;
	connection.idleTimeout = self.idleTimeout;
	@synchronized (self.connections) {
		[self.connections addObject:connection];
	}
	dispatch_async(connection.queue, ^{
		[connection start];
	});
}

- (void)connectionDidClose:(NFKHTTPConnection *)connection
{
	@synchronized (self.connections) {
		[self.connections removeObject:connection];
	}
}

- (void)stop
{
	if (self.listener != nil) {
		nw_listener_cancel(self.listener);
		self.listener = nil;
		// Cancellation is asynchronous; waiting for it frees the port before stop returns, so a start
		// on the same port right after succeeds. On a listener queue the state could not arrive.
		if (dispatch_get_specific(&NFKHTTPListenerQueueKey) == NULL) {
			dispatch_semaphore_wait(self.listenerCancelled, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)));
		}
		self.listenerCancelled = nil;
	}
	NSArray<NFKHTTPConnection *> *open = nil;
	@synchronized (self.connections) {
		open = self.connections.allObjects;
	}
	for (NFKHTTPConnection *connection in open) {
		dispatch_async(connection.queue, ^{
			[connection dropConnection];
		});
	}
}

- (BOOL)failWithReason:(NSString *)reason underlying:(nullable nw_error_t)underlying error:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
	NSString *detail = nil;
	if (underlying != nil && nw_error_get_error_domain(underlying) == nw_error_domain_posix) {
		int code = nw_error_get_error_code(underlying);
		detail = [NSString stringWithUTF8String:strerror(code)];
		userInfo[NSUnderlyingErrorKey] = [NSError errorWithDomain:NSPOSIXErrorDomain code:code userInfo:nil];
	}
	userInfo[NSLocalizedDescriptionKey] = detail != nil
		? [NSString stringWithFormat:@"%@ on port %u: %@", reason, (unsigned)self.port, detail]
		: [NSString stringWithFormat:@"%@ on port %u", reason, (unsigned)self.port];
	*outError = [NSError errorWithDomain:@"NFKInferenceServerErrorDomain" code:1 userInfo:userInfo];
	return NO;
}

@end
