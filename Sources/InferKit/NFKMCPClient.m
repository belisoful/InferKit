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

@interface NFKMCPClient ()
@property (nonatomic, copy, readwrite, nullable) NSDictionary<NSString *, id> *serverInfo;
@property (nonatomic, copy, readwrite, nullable) NSString *instructions;
@property (nonatomic, copy, readwrite, nullable) NSString *protocolVersion;
@property (nonatomic, copy, nullable) NSString *sessionIdentifier;
@property (nonatomic, assign) NSInteger nextIdentifier;
@property (nonatomic, strong) NSLock *lock;
@end

@implementation NFKMCPClient

- (instancetype)initWithEndpointURL:(NSURL *)endpointURL apiKey:(nullable NSString *)apiKey
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [endpointURL copy];
		_apiKey = [apiKey copy];
		_additionalHeaders = @{};
		_timeout = 600;
		_nextIdentifier = 1;
		_lock = [[NSLock alloc] init];
	}
	return self;
}

#pragma mark Calls

- (BOOL)connectWithError:(NSError **)error
{
	[self.lock lock];
	BOOL connected = [self connectLockedWithError:error];
	[self.lock unlock];
	return connected;
}

- (BOOL)connectLockedWithError:(NSError **)error
{
	if (self.protocolVersion != nil) {
		return YES;
	}
	NSDictionary *params = @{ @"protocolVersion": NFKMCPProtocolVersions().firstObject,
							  @"capabilities": @{},
							  @"clientInfo": @{ @"name": @"InferKit", @"version": NFKInferKit.releaseVersion } };
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
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.endpointURL];
	request.HTTPMethod = @"POST";
	request.timeoutInterval = self.timeout;
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:message options:0 error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[request setValue:@"application/json, text/event-stream" forHTTPHeaderField:@"Accept"];
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

- (BOOL)fail:(NSError **)error reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:reason];
	}
	return NO;
}

@end
