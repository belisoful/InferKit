//
//  NFKHTTPServer.m
//  InferKit
//

#import "NFKHTTPServer.h"
#import <Network/Network.h>
#import <arpa/inet.h>
#import <netinet/in.h>

static const NSUInteger NFKHTTPMaximumHeadBytes = 64 * 1024;

static NSString *NFKHTTPReasonPhrase(NSInteger status)
{
	NSDictionary<NSNumber *, NSString *> *phrases = @{ @200: @"OK", @204: @"No Content", @400: @"Bad Request",
													   @401: @"Unauthorized", @403: @"Forbidden", @404: @"Not Found",
													   @405: @"Method Not Allowed", @408: @"Request Timeout",
													   @413: @"Payload Too Large", @415: @"Unsupported Media Type",
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
@property (nonatomic, assign) NSUInteger maximumBodyBytes;
@property (nonatomic, assign) NSTimeInterval idleTimeout;
@property (nonatomic, assign) CFAbsoluteTime lastActivity;
@end

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

@implementation NFKHTTPConnection

- (void)start
{
	self.buffer = [NSMutableData data];
	self.lastActivity = CFAbsoluteTimeGetCurrent();
	self.fromLoopback = NFKHTTPEndpointIsLoopback(nw_connection_copy_endpoint(self.connection));
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
			NSArray<NSString *> *sides = [pair componentsSeparatedByString:@"="];
			NSString *name = [[sides.firstObject stringByReplacingOccurrencesOfString:@"+" withString:@" "] stringByRemovingPercentEncoding];
			NSString *value = sides.count > 1 ? [[sides[1] stringByReplacingOccurrencesOfString:@"+" withString:@" "] stringByRemovingPercentEncoding] : @"";
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
	NSMutableDictionary<NSString *, NSString *> *fields = [NSMutableDictionary dictionaryWithDictionary:headers ?: @{}];
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

- (void)connectionDidDrop
{
	void (^handler)(void) = nil;
	@synchronized (self) {
		if (self.finished || self.disconnected) {
			return;
		}
		self.disconnected = YES;
		handler = self.disconnectHandler;
		self.disconnectHandler = nil;
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
	__block BOOL ready = NO;
	__block BOOL signalled = NO;
	__block nw_error_t failure = nil;
	nw_listener_set_state_changed_handler(listener, ^(nw_listener_state_t state, nw_error_t error) {
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
