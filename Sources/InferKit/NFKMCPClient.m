//
//  NFKMCPClient.m
//  InferKit
//

#import "NFKMCPClient.h"
#import "NFKInferenceKeys.h"
#import "NFKInferKit.h"
#import "NFKAudioAsset.h"
#import "NFKVideoAsset.h"
#import "NFKImageCoding.h"
#import "NFKErrors.h"
#import "NFKRemoteTransport.h"
#import "NFKRemoteMediaSupport.h"
#import "NFKServedMCP.h"
#import "NFKSubprocessPrivate.h"

/*! A tool a server lists, which runs by calling the server. */
@interface NFKMCPClientTool : NFKInferenceTool
@property (nonatomic, strong) NFKMCPClient *client;
@end

@implementation NFKMCPClientTool

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	return [self.client resultOfToolNamed:self.name arguments:arguments error:error];
}

@end

/*! A request sent to a stdio server, waiting for the reply the reader hands it. */
@interface NFKMCPPendingRequest : NSObject
@property (nonatomic, strong) dispatch_semaphore_t answered;
@property (nonatomic, copy, nullable) NSDictionary *reply;
@property (nonatomic, strong, nullable) NSError *failure;
@end

@implementation NFKMCPPendingRequest

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_answered = dispatch_semaphore_create(0);
	}
	return self;
}

@end

@interface NFKMCPClient ()
@property (nonatomic, copy, readwrite, nullable) NSDictionary<NSString *, id> *serverInfo;
@property (nonatomic, copy, readwrite, nullable) NSString *instructions;
@property (nonatomic, copy, readwrite, nullable) NSString *protocolVersion;
@property (nonatomic, copy, nullable) NSString *sessionIdentifier;
@property (nonatomic, assign) NSInteger nextIdentifier;
@property (nonatomic, strong) NSLock *lock;
@property (nonatomic, strong) NSLock *pendingLock;
@property (nonatomic, strong) NSMutableDictionary<id, NFKMCPPendingRequest *> *pendingRequests;
@property (nonatomic, strong) dispatch_queue_t answerQueue;
@property (atomic, assign) BOOL stdioSessionEnded;
@property (nonatomic, assign) BOOL launchesOnNextCall;
@end

@implementation NFKMCPClient

#if !TARGET_OS_OSX
@dynamic serverProcess;
#endif

- (instancetype)initWithEndpointURL:(NSURL *)endpointURL apiKey:(nullable NSString *)apiKey
{
	self = [super init];
	if (self != nil) {
		_transport = NFKMCPTransportStreamableHTTP;
		_endpointURL = [endpointURL copy];
		_apiKey = [apiKey copy];
		[self setUpRequestState];
	}
	return self;
}

#if TARGET_OS_OSX

- (instancetype)initWithCommand:(NSString *)command
					  arguments:(NSArray<NSString *> *)arguments
					environment:(nullable NSDictionary<NSString *, NSString *> *)environment
			workingDirectoryURL:(nullable NSURL *)workingDirectoryURL
{
	self = [super init];
	if (self != nil) {
		_transport = NFKMCPTransportStdio;
		_serverProcess = [[NFKSubprocess alloc] initWithCommand:command arguments:arguments];
		_serverProcess.environment = environment ?: @{};
		_serverProcess.workingDirectoryURL = workingDirectoryURL;
		_serverProcess.inheritedEnvironmentVariables = NFKMCPClient.defaultInheritedEnvironmentVariables;
		[self setUpRequestState];
		[self observeServerProcess];
	}
	return self;
}

- (void)setInheritsFullEnvironment:(BOOL)inheritsFullEnvironment
{
	_inheritsFullEnvironment = inheritsFullEnvironment;
	self.serverProcess.inheritedEnvironmentVariables = inheritsFullEnvironment ? nil : NFKMCPClient.defaultInheritedEnvironmentVariables;
}

#else
@dynamic inheritsFullEnvironment;
#endif

+ (NSArray<NSString *> *)defaultInheritedEnvironmentVariables
{
	return @[ @"HOME", @"LOGNAME", @"PATH", @"SHELL", @"TERM", @"USER", @"TMPDIR", @"LANG", @"LC_ALL" ];
}

- (void)setUpRequestState
{
	self.additionalHeaders = @{};
	self.timeout = 600;
	self.nextIdentifier = 1;
	self.launchesOnNextCall = YES;
	self.lock = [[NSLock alloc] init];
	self.pendingLock = [[NSLock alloc] init];
	self.pendingRequests = [NSMutableDictionary dictionary];
	dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
	self.answerQueue = dispatch_queue_create("com.inferkit.mcp-client.answers", attributes);
}

- (void)dealloc
{
#if TARGET_OS_OSX
	[_serverProcess stopWithCompletionHandler:nil];
#endif
}

#pragma mark Calls

- (BOOL)connectWithError:(NSError **)error
{
	[self.lock lock];
	self.launchesOnNextCall = YES;
	BOOL connected = [self connectLockedWithError:error];
	[self.lock unlock];
	return connected;
}

- (BOOL)connectLockedWithError:(NSError **)error
{
	if (![self prepareTransportWithError:error]) {
		return NO;
	}
	if (self.protocolVersion != nil) {
		return YES;
	}
	NSDictionary *params = @{ @"protocolVersion": NFKMCPProtocolVersions().firstObject,
							  @"capabilities": @{},
							  @"clientInfo": @{ @"name": @"InferKit", @"version": NFKInferKit.version } };
	NSDictionary *result = [self resultOfMethod:@"initialize" params:params error:error];
	if (result == nil) {
		return NO;
	}
	NSString *version = [result[@"protocolVersion"] isKindOfClass:NSString.class] ? result[@"protocolVersion"] : nil;
	if (![NFKMCPProtocolVersions() containsObject:version ?: @""]) {
		return [self fail:error reason:[NSString stringWithFormat:@"the server speaks MCP %@, which this client does not", version ?: @"(none)"]];
	}
	self.protocolVersion = version;
	self.serverInfo = [result[@"serverInfo"] isKindOfClass:NSDictionary.class] ? result[@"serverInfo"] : nil;
	self.instructions = [result[@"instructions"] isKindOfClass:NSString.class] ? result[@"instructions"] : nil;
	return [self sendMessage:@{ @"jsonrpc": @"2.0", @"method": @"notifications/initialized" } expectingReplyTo:nil error:error] != nil;
}

- (BOOL)pingWithError:(NSError **)error
{
	[self.lock lock];
	BOOL answered = [self connectLockedWithError:error] && [self resultOfMethod:@"ping" params:@{} error:error] != nil;
	[self.lock unlock];
	return answered;
}

- (void)disconnect
{
	[self.lock lock];
#if TARGET_OS_OSX
	[self.serverProcess stop];
#endif
	self.launchesOnNextCall = YES;
	if (self.transport == NFKMCPTransportStreamableHTTP && self.sessionIdentifier != nil) {
		NSMutableURLRequest *request = [self HTTPRequestWithMethod:@"DELETE"];
		[NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:NULL error:NULL];
	}
	[self forgetSession];
	[self.lock unlock];
}

- (void)forgetSession
{
	self.protocolVersion = nil;
	self.serverInfo = nil;
	self.instructions = nil;
	self.sessionIdentifier = nil;
}

- (nullable NSArray<NFKInferenceTool *> *)toolsWithError:(NSError **)error
{
	[self.lock lock];
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	NSString *cursor = nil;
	BOOL failed = ![self connectLockedWithError:error];
	while (!failed) {
		NSDictionary *result = [self resultOfMethod:@"tools/list" params:cursor != nil ? @{ @"cursor": cursor } : @{} error:error];
		if (result == nil) {
			failed = YES;
			break;
		}
		for (NSDictionary *listed in [result[@"tools"] isKindOfClass:NSArray.class] ? result[@"tools"] : @[]) {
			if (![listed isKindOfClass:NSDictionary.class] || ![listed[@"name"] isKindOfClass:NSString.class]) {
				continue;
			}
			NSDictionary *schema = [listed[@"inputSchema"] isKindOfClass:NSDictionary.class] ? listed[@"inputSchema"] : @{ @"type": @"object" };
			NSString *description = [listed[@"description"] isKindOfClass:NSString.class] ? listed[@"description"] : @"";
			NFKMCPClientTool *tool = [[NFKMCPClientTool alloc] initWithName:listed[@"name"] description:description parameters:schema];
			tool.client = self;
			[tools addObject:tool];
		}
		cursor = [result[@"nextCursor"] isKindOfClass:NSString.class] && [result[@"nextCursor"] length] > 0 ? result[@"nextCursor"] : nil;
		if (cursor == nil) {
			break;
		}
	}
	[self.lock unlock];
	return failed ? nil : tools;
}

- (nullable NFKToolResult *)resultOfToolNamed:(NSString *)name arguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	[self.lock lock];
	NSDictionary *result = [self connectLockedWithError:error]
		? [self resultOfMethod:@"tools/call" params:@{ @"name": name, @"arguments": arguments ?: @{} } error:error] : nil;
	[self.lock unlock];
	return result != nil ? [self toolResultFrom:result error:error] : nil;
}

#pragma mark Results

- (nullable NFKToolResult *)toolResultFrom:(NSDictionary *)result error:(NSError **)error
{
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSData *> *pngs = [NSMutableArray array];
	NSMutableArray *images = [NSMutableArray array];
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	for (NSDictionary *item in [result[@"content"] isKindOfClass:NSArray.class] ? result[@"content"] : @[]) {
		if (![item isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *type = [item[@"type"] isKindOfClass:NSString.class] ? item[@"type"] : @"";
		NSData *bytes = [item[@"data"] isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:item[@"data"] options:0] : nil;
		NSString *mimeType = [item[@"mimeType"] isKindOfClass:NSString.class] ? item[@"mimeType"] : @"";
		if ([type isEqualToString:@"text"] && [item[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:item[@"text"]];
		} else if ([type isEqualToString:@"image"] && bytes != nil) {
			CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithImageData:bytes];
			if (pixelBuffer != NULL) {
				[images addObject:(__bridge id)pixelBuffer];
				NSData *png = [mimeType isEqualToString:@"image/png"] ? bytes : [NFKImageCoding PNGDataForImage:(__bridge id)pixelBuffer];
				CVPixelBufferRelease(pixelBuffer);
				if (png != nil) {
					[pngs addObject:png];
				}
			}
		} else if ([type isEqualToString:@"audio"] && bytes != nil) {
			NSURL *file = NFKRemoteWriteMediaFile(bytes, @"mcp-audio", [self extensionForMediaType:mimeType fallback:@"wav"], nil, error);
			if (file == nil) {
				return nil;
			}
			outputs[NFKOutputAudio] = [NFKAudioAsset audioAssetWithFileURL:file];
		} else if ([type isEqualToString:@"resource"] && [item[@"resource"] isKindOfClass:NSDictionary.class]) {
			if (![self readResource:item[@"resource"] into:outputs texts:texts error:error]) {
				return nil;
			}
		} else if ([type isEqualToString:@"resource_link"] && [item[@"uri"] isKindOfClass:NSString.class]) {
			[texts addObject:item[@"uri"]];
		}
	}
	if (images.count > 0) {
		outputs[NFKOutputImage] = images.firstObject;
	}
	if (images.count > 1) {
		outputs[NFKOutputImages] = images;
	}
	if ([result[@"structuredContent"] isKindOfClass:NSDictionary.class]) {
		outputs[NFKOutputStructured] = result[@"structuredContent"];
	}
	return [[NFKToolResult alloc] initWithText:[texts componentsJoinedByString:@"\n"] outputs:outputs imagePNGs:pngs
									   isError:[result[@"isError"] boolValue]];
}

// An embedded resource is text the model reads, or a file: a video becomes the result's clip.
- (BOOL)readResource:(NSDictionary *)resource into:(NSMutableDictionary *)outputs texts:(NSMutableArray<NSString *> *)texts error:(NSError **)error
{
	if ([resource[@"text"] isKindOfClass:NSString.class]) {
		[texts addObject:resource[@"text"]];
		return YES;
	}
	NSString *mimeType = [resource[@"mimeType"] isKindOfClass:NSString.class] ? resource[@"mimeType"] : @"";
	NSData *blob = [resource[@"blob"] isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:resource[@"blob"] options:0] : nil;
	if (blob == nil || ![mimeType hasPrefix:@"video/"]) {
		return YES;
	}
	NSURL *file = NFKRemoteWriteMediaFile(blob, @"mcp-video", [self extensionForMediaType:mimeType fallback:@"mp4"], nil, error);
	if (file == nil) {
		return NO;
	}
	outputs[NFKOutputVideo] = [NFKVideoAsset videoAssetWithFileURL:file];
	return YES;
}

- (NSString *)extensionForMediaType:(NSString *)mimeType fallback:(NSString *)fallback
{
	NSDictionary<NSString *, NSString *> *extensions = @{ @"audio/wav": @"wav", @"audio/x-wav": @"wav", @"audio/wave": @"wav",
														  @"audio/mpeg": @"mp3", @"audio/mp3": @"mp3", @"audio/mp4": @"m4a",
														  @"audio/aac": @"aac", @"audio/flac": @"flac", @"audio/aiff": @"aiff",
														  @"audio/x-caf": @"caf", @"audio/ogg": @"ogg", @"video/mp4": @"mp4",
														  @"video/quicktime": @"mov", @"video/webm": @"webm" };
	return extensions[mimeType.lowercaseString] ?: fallback;
}

#pragma mark Transport

- (nullable NSDictionary *)resultOfMethod:(NSString *)method params:(NSDictionary *)params error:(NSError **)error
{
	NSNumber *identifier = @(self.nextIdentifier++);
	NSDictionary *reply = [self sendMessage:@{ @"jsonrpc": @"2.0", @"id": identifier, @"method": method, @"params": params }
						   expectingReplyTo:identifier error:error];
	if (reply == nil) {
		return nil;
	}
	NSDictionary *failure = [reply[@"error"] isKindOfClass:NSDictionary.class] ? reply[@"error"] : nil;
	if (failure != nil) {
		NSString *message = [failure[@"message"] isKindOfClass:NSString.class] ? failure[@"message"] : @"the server reported an error";
		[self fail:error reason:[NSString stringWithFormat:@"%@: %@", method, message]];
		return nil;
	}
	return [reply[@"result"] isKindOfClass:NSDictionary.class] ? reply[@"result"] : @{};
}

/*! Sends one message. With an identifier, returns the reply naming it; without one, returns an empty
	dictionary once the server accepts the notification. */
- (nullable NSDictionary *)sendMessage:(NSDictionary *)message expectingReplyTo:(nullable NSNumber *)identifier error:(NSError **)error
{
#if TARGET_OS_OSX
	if (self.transport == NFKMCPTransportStdio) {
		return [self sendStdioMessage:message expectingReplyTo:identifier error:error];
	}
#endif
	return [self sendHTTPMessage:message expectingReplyTo:identifier error:error];
}

- (BOOL)prepareTransportWithError:(NSError **)error
{
#if TARGET_OS_OSX
	if (self.transport != NFKMCPTransportStdio) {
		return YES;
	}
	NFKSubprocess *process = self.serverProcess;
	NFKSubprocessState state = process.state;
	if (state != NFKSubprocessStateRunning && state != NFKSubprocessStateStarting) {
		if (!self.launchesOnNextCall) {
			NSError *ended = [self errorForEndedProcess:process state:state];
			if (error != NULL) {
				*error = ended;
			}
			return NO;
		}
		self.launchesOnNextCall = NO;
		NSError *launchError = nil;
		if (![process startWithError:&launchError]) {
			return [self unreachable:error reason:[NSString stringWithFormat:@"the MCP server could not be launched: %@", launchError.localizedDescription]
						  underlying:launchError];
		}
	}
	if (self.stdioSessionEnded) {
		self.stdioSessionEnded = NO;
		[self forgetSession];
	}
#endif
	return YES;
}

#pragma mark Streamable HTTP

- (NSMutableURLRequest *)HTTPRequestWithMethod:(NSString *)method
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.endpointURL];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	[self.additionalHeaders enumerateKeysAndObjectsUsingBlock:^(NSString *field, NSString *value, BOOL *stop) {
		[request setValue:value forHTTPHeaderField:field];
	}];
	if (self.apiKey.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:self.apiKey] forHTTPHeaderField:@"Authorization"];
	}
	if (self.protocolVersion != nil) {
		[request setValue:self.protocolVersion forHTTPHeaderField:@"MCP-Protocol-Version"];
	}
	if (self.sessionIdentifier != nil) {
		[request setValue:self.sessionIdentifier forHTTPHeaderField:@"Mcp-Session-Id"];
	}
	return request;
}

- (nullable NSDictionary *)sendHTTPMessage:(NSDictionary *)message expectingReplyTo:(nullable NSNumber *)identifier error:(NSError **)error
{
	NSMutableURLRequest *request = [self HTTPRequestWithMethod:@"POST"];
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:message options:0 error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[request setValue:@"application/json, text/event-stream" forHTTPHeaderField:@"Accept"];
	NSHTTPURLResponse *response = nil;
	NSData *data = [NFKRemoteTransport sendRequest:request session:NSURLSession.sharedSession response:&response error:error];
	if (data == nil) {
		return nil;
	}
	NSString *session = [response valueForHTTPHeaderField:@"Mcp-Session-Id"];
	if (session.length > 0) {
		self.sessionIdentifier = session;
	}
	if (response.statusCode >= 400) {
		NSError *failure = [NFKRemoteTransport errorForResponse:response data:data];
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	if (identifier == nil) {
		return @{};
	}
	NSString *type = [response valueForHTTPHeaderField:@"Content-Type"].lowercaseString ?: @"";
	NSArray *messages = [type hasPrefix:@"text/event-stream"] ? [self messagesInEventStream:data] : [self messagesInJSON:data];
	for (NSDictionary *reply in messages) {
		if ([reply isKindOfClass:NSDictionary.class] && [reply[@"id"] isEqual:identifier]) {
			return reply;
		}
	}
	[self fail:error reason:@"the server's reply named no response to the request"];
	return nil;
}

- (NSArray *)messagesInJSON:(NSData *)data
{
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [object isKindOfClass:NSArray.class] ? object : (object != nil ? @[ object ] : @[]);
}

// Each event's data lines join with newlines into one JSON-RPC message.
- (NSArray *)messagesInEventStream:(NSData *)data
{
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
	NSMutableArray *messages = [NSMutableArray array];
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	void (^flush)(void) = ^{
		if (lines.count > 0) {
			NSData *payload = [[lines componentsJoinedByString:@"\n"] dataUsingEncoding:NSUTF8StringEncoding];
			[messages addObjectsFromArray:[self messagesInJSON:payload]];
			[lines removeAllObjects];
		}
	};
	for (NSString *raw in [text componentsSeparatedByString:@"\n"]) {
		NSString *line = [raw hasSuffix:@"\r"] ? [raw substringToIndex:raw.length - 1] : raw;
		if (line.length == 0) {
			flush();
		} else if ([line hasPrefix:@"data:"]) {
			NSString *value = [line substringFromIndex:5];
			[lines addObject:[value hasPrefix:@" "] ? [value substringFromIndex:1] : value];
		}
	}
	flush();
	return messages;
}

#pragma mark stdio

#if TARGET_OS_OSX

- (void)observeServerProcess
{
	__weak NFKMCPClient *weakSelf = self;
	self.serverProcess.ownerLineHandler = ^(NSData *line) {
		[weakSelf receiveLine:line];
	};
	self.serverProcess.ownerStateHandler = ^(NFKSubprocess *subprocess, NFKSubprocessState state) {
		[weakSelf serverProcess:subprocess didEnterState:state];
	};
}

// Runs on the process's event queue, so it takes only the pending lock.
- (void)serverProcess:(NFKSubprocess *)process didEnterState:(NFKSubprocessState)state
{
	if (state == NFKSubprocessStateStarting) {
		self.stdioSessionEnded = YES;
		return;
	}
	if (state != NFKSubprocessStateExited && state != NFKSubprocessStateFailedToLaunch) {
		return;
	}
	self.stdioSessionEnded = YES;
	NSError *failure = [self errorForEndedProcess:process state:state];
	[self.pendingLock lock];
	NSArray<NFKMCPPendingRequest *> *waiting = self.pendingRequests.allValues;
	[self.pendingRequests removeAllObjects];
	for (NFKMCPPendingRequest *pending in waiting) {
		pending.failure = failure;
	}
	[self.pendingLock unlock];
	for (NFKMCPPendingRequest *pending in waiting) {
		dispatch_semaphore_signal(pending.answered);
	}
}

- (NSError *)errorForEndedProcess:(NFKSubprocess *)process state:(NFKSubprocessState)state
{
	NSString *reason = @"the MCP server exited";
	if (state == NFKSubprocessStateFailedToLaunch) {
		reason = [NSString stringWithFormat:@"the MCP server could not be launched: %@", process.launchError.localizedDescription];
	} else if (process.terminationReason == NFKSubprocessTerminationReasonSignal) {
		reason = [NSString stringWithFormat:@"the MCP server ended on signal %d", process.terminationStatus];
	} else if (process.terminationReason == NFKSubprocessTerminationReasonExit) {
		reason = [NSString stringWithFormat:@"the MCP server exited with status %d", process.terminationStatus];
	}
	NSString *lastLogLine = [self lastLineOf:process.recentStandardError];
	if (lastLogLine.length > 0) {
		reason = [reason stringByAppendingFormat:@"; its log ends: %@", lastLogLine];
	}
	NSError *failure = nil;
	[self unreachable:&failure reason:reason underlying:process.launchError];
	return failure;
}

- (nullable NSString *)lastLineOf:(NSString *)text
{
	NSArray<NSString *> *lines = [text componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
	for (NSString *line in lines.reverseObjectEnumerator) {
		NSString *trimmed = [line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		if (trimmed.length > 0) {
			return trimmed;
		}
	}
	return nil;
}

// Runs on the process's event queue. A line that is not JSON is skipped: the transport reserves
// standard output for messages, and a server's stray print is not one.
- (void)receiveLine:(NSData *)line
{
	if (line.length == 0) {
		return;
	}
	id object = [NSJSONSerialization JSONObjectWithData:line options:0 error:NULL];
	NSArray *messages = [object isKindOfClass:NSArray.class] ? object : (object != nil ? @[ object ] : @[]);
	for (NSDictionary *message in messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		id identifier = message[@"id"];
		BOOL identified = identifier != nil && identifier != NSNull.null;
		if ([message[@"method"] isKindOfClass:NSString.class]) {
			if (identified) {
				[self answerServerRequest:message];
			}
			continue;
		}
		if (identified) {
			[self completeRequest:identifier withReply:message];
		}
	}
}

- (void)completeRequest:(id)identifier withReply:(NSDictionary *)reply
{
	// The reply is stored under the lock, so a caller whose wait times out at the same moment reads it.
	[self.pendingLock lock];
	NFKMCPPendingRequest *pending = self.pendingRequests[identifier];
	[self.pendingRequests removeObjectForKey:identifier];
	pending.reply = reply;
	[self.pendingLock unlock];
	if (pending != nil) {
		dispatch_semaphore_signal(pending.answered);
	}
}

/*! Answers ping, and refuses every other request: the client declares no capabilities, so it
	serves no sampling, roots, or elicitation. The answer is written off the reading queue, so a
	server blocked writing to the client cannot block the client's answer. */
- (void)answerServerRequest:(NSDictionary *)request
{
	NSString *method = request[@"method"];
	NSDictionary *answer = [method isEqualToString:@"ping"]
		? @{ @"jsonrpc": @"2.0", @"id": request[@"id"], @"result": @{} }
		: @{ @"jsonrpc": @"2.0", @"id": request[@"id"],
			 @"error": @{ @"code": @(-32601), @"message": [NSString stringWithFormat:@"InferKit's MCP client does not handle %@", method] } };
	NSData *line = [NSJSONSerialization dataWithJSONObject:answer options:NSJSONWritingWithoutEscapingSlashes error:NULL];
	NFKSubprocess *process = self.serverProcess;
	if (line == nil) {
		return;
	}
	dispatch_async(self.answerQueue, ^{
		[process writeLine:line error:NULL];
	});
}

- (nullable NSDictionary *)sendStdioMessage:(NSDictionary *)message expectingReplyTo:(nullable NSNumber *)identifier error:(NSError **)error
{
	NSError *encodingError = nil;
	NSData *line = [NSJSONSerialization dataWithJSONObject:message options:NSJSONWritingWithoutEscapingSlashes error:&encodingError];
	if (line == nil) {
		[self fail:error reason:[NSString stringWithFormat:@"the message is not JSON: %@", encodingError.localizedDescription]];
		return nil;
	}
	NFKMCPPendingRequest *pending = nil;
	if (identifier != nil) {
		pending = [[NFKMCPPendingRequest alloc] init];
		[self.pendingLock lock];
		self.pendingRequests[identifier] = pending;
		[self.pendingLock unlock];
	}
	NSError *writeError = nil;
	if (![self.serverProcess writeLine:line error:&writeError]) {
		[self removePendingRequest:identifier];
		[self unreachable:error reason:[NSString stringWithFormat:@"%@: %@", message[@"method"], writeError.localizedDescription] underlying:writeError];
		return nil;
	}
	if (identifier == nil) {
		return @{};
	}
	NSTimeInterval timeout = self.timeout;
	dispatch_time_t deadline = timeout > 0 ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)) : DISPATCH_TIME_FOREVER;
	if (dispatch_semaphore_wait(pending.answered, deadline) != 0 && [self removePendingRequest:identifier]) {
		NSDictionary *cancellation = @{ @"jsonrpc": @"2.0", @"method": @"notifications/cancelled",
										@"params": @{ @"requestId": identifier, @"reason": @"The request timed out." } };
		[self sendStdioMessage:cancellation expectingReplyTo:nil error:NULL];
		[self unreachable:error reason:[NSString stringWithFormat:@"%@: the MCP server did not answer within %g seconds", message[@"method"], timeout]
			   underlying:nil];
		return nil;
	}
	if (pending.failure != nil) {
		if (error != NULL) {
			*error = pending.failure;
		}
		return nil;
	}
	return pending.reply;
}

/*! Removes the request and returns whether it was still waiting. */
- (BOOL)removePendingRequest:(nullable NSNumber *)identifier
{
	if (identifier == nil) {
		return NO;
	}
	[self.pendingLock lock];
	BOOL waiting = self.pendingRequests[identifier] != nil;
	[self.pendingRequests removeObjectForKey:identifier];
	[self.pendingLock unlock];
	return waiting;
}

#endif

- (BOOL)unreachable:(NSError **)error reason:(NSString *)reason underlying:(nullable NSError *)underlying
{
	if (error != NULL) {
		NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
		if (underlying != nil) {
			userInfo[NSUnderlyingErrorKey] = underlying;
		}
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable userInfo:userInfo];
	}
	return NO;
}

- (BOOL)fail:(NSError **)error reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
	}
	return NO;
}

@end
