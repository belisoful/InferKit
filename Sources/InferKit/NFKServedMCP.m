//
//  NFKServedMCP.m
//  InferKit
//

#import "NFKServedMCP.h"
#import <InferKit/NFKInferKit.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>

static NSInteger const NFKJSONRPCParseError		= -32700;
static NSInteger const NFKJSONRPCInvalidRequest	= -32600;
static NSInteger const NFKJSONRPCMethodNotFound	= -32601;
static NSInteger const NFKJSONRPCInvalidParams	= -32602;

NSArray<NSString *> *NFKMCPProtocolVersions(void)
{
	return @[ @"2025-11-25", @"2025-06-18", @"2025-03-26", @"2024-11-05" ];
}

static NSString *NFKMCPMediaType(NSURL *fileURL, NSString *fallback)
{
	NSDictionary<NSString *, NSString *> *types = @{ @"wav": @"audio/wav", @"mp3": @"audio/mpeg", @"m4a": @"audio/mp4",
													 @"aac": @"audio/aac", @"flac": @"audio/flac", @"aif": @"audio/aiff",
													 @"aiff": @"audio/aiff", @"caf": @"audio/x-caf", @"ogg": @"audio/ogg",
													 @"mp4": @"video/mp4", @"mov": @"video/quicktime", @"m4v": @"video/mp4",
													 @"webm": @"video/webm" };
	return types[fileURL.pathExtension.lowercaseString] ?: fallback;
}

NSDictionary<NSString *, id> *NFKMCPCallResult(NFKToolResult *result)
{
	NSMutableArray *content = [NSMutableArray arrayWithObject:@{ @"type": @"text", @"text": result.text }];
	for (NSData *png in result.imagePNGs) {
		[content addObject:@{ @"type": @"image", @"data": [png base64EncodedStringWithOptions:0], @"mimeType": @"image/png" }];
	}
	id audio = result.outputs[NFKOutputAudio];
	NSURL *audioURL = [audio isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)audio fileURL] : nil;
	NSData *audioData = audioURL != nil ? [NSData dataWithContentsOfURL:audioURL] : nil;
	if (audioData != nil) {
		[content addObject:@{ @"type": @"audio", @"data": [audioData base64EncodedStringWithOptions:0],
							  @"mimeType": NFKMCPMediaType(audioURL, @"audio/wav") }];
	}
	id video = result.outputs[NFKOutputVideo];
	NSURL *videoURL = [video isKindOfClass:NFKVideoAsset.class] ? [(NFKVideoAsset *)video fileURL] : nil;
	NSData *videoData = videoURL != nil ? [NSData dataWithContentsOfURL:videoURL] : nil;
	if (videoData != nil) {
		[content addObject:@{ @"type": @"resource",
							  @"resource": @{ @"uri": [@"inferkit:///" stringByAppendingString:videoURL.lastPathComponent],
											  @"mimeType": NFKMCPMediaType(videoURL, @"video/mp4"),
											  @"blob": [videoData base64EncodedStringWithOptions:0] } }];
	}
	NSMutableDictionary *reply = [NSMutableDictionary dictionaryWithDictionary:@{ @"content": content, @"isError": @(result.isError) }];
	id structured = result.outputs[NFKOutputStructured];
	if ([structured isKindOfClass:NSDictionary.class] && [NSJSONSerialization isValidJSONObject:structured]) {
		reply[@"structuredContent"] = structured;
	}
	return reply;
}

@implementation NFKServedMCP

+ (void)serveRequest:(NFKHTTPRequest *)request
			response:(NFKHTTPResponse *)response
			   tools:(NSArray<NFKInferenceTool *> *)tools
		instructions:(nullable NSString *)instructions
{
	if (![self originIsAllowedFor:request]) {
		[response sendStatus:403 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
															  message:@"the Origin header names another host"] headers:nil];
		return;
	}
	if (![request.method isEqualToString:@"POST"]) {
		[response sendStatus:405 headers:@{ @"Allow": @"POST" } body:nil];
		return;
	}
	NSString *version = [request valueForHeader:@"mcp-protocol-version"];
	if (version != nil && ![NFKMCPProtocolVersions() containsObject:version]) {
		[response sendStatus:400 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest
															  message:[@"unsupported MCP-Protocol-Version " stringByAppendingString:version]] headers:nil];
		return;
	}
	id body = request.body.length > 0 ? [NSJSONSerialization JSONObjectWithData:request.body options:0 error:NULL] : nil;
	if (body == nil) {
		[response sendStatus:400 JSONObject:[self errorWithIdentifier:nil code:NFKJSONRPCParseError message:@"the body is not JSON"] headers:nil];
		return;
	}
	BOOL batch = [body isKindOfClass:NSArray.class];
	NSArray *messages = batch ? body : @[ body ];
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSMutableArray *replies = [NSMutableArray array];
		for (id message in messages) {
			NSDictionary *reply = [self replyTo:message tools:tools instructions:instructions];
			if (reply != nil) {
				[replies addObject:reply];
			}
		}
		if (replies.count == 0) {
			[response sendStatus:202 headers:nil body:nil];
			return;
		}
		[response sendStatus:200 JSONObject:batch ? replies : replies.firstObject headers:nil];
	});
}

// A browser sends Origin; a page on another host that resolves to this machine names its own.
+ (BOOL)originIsAllowedFor:(NFKHTTPRequest *)request
{
	NSString *origin = [request valueForHeader:@"origin"];
	if (origin.length == 0 || [origin isEqualToString:@"null"]) {
		return origin.length == 0;
	}
	NSString *host = [NSURL URLWithString:origin].host.lowercaseString;
	if ([@[ @"localhost", @"127.0.0.1", @"::1" ] containsObject:host ?: @""]) {
		return YES;
	}
	NSString *served = [[[request valueForHeader:@"host"] ?: @"" componentsSeparatedByString:@":"].firstObject lowercaseString];
	return host.length > 0 && [host isEqualToString:served];
}

/*! The reply to one message, or nil for a notification or a response, which take none. */
+ (nullable NSDictionary *)replyTo:(id)message tools:(NSArray<NFKInferenceTool *> *)tools instructions:(nullable NSString *)instructions
{
	if (![message isKindOfClass:NSDictionary.class] || ![message[@"jsonrpc"] isEqual:@"2.0"]) {
		return [self errorWithIdentifier:nil code:NFKJSONRPCInvalidRequest message:@"not a JSON-RPC 2.0 message"];
	}
	id identifier = message[@"id"];
	NSString *method = [message[@"method"] isKindOfClass:NSString.class] ? message[@"method"] : nil;
	if (identifier == nil || method == nil) {
		return nil;
	}
	NSDictionary *params = [message[@"params"] isKindOfClass:NSDictionary.class] ? message[@"params"] : @{};
	if ([method isEqualToString:@"initialize"]) {
		return [self resultWithIdentifier:identifier result:[self initializeResult:params instructions:instructions]];
	}
	if ([method isEqualToString:@"ping"]) {
		return [self resultWithIdentifier:identifier result:@{}];
	}
	if ([method isEqualToString:@"tools/list"]) {
		NSMutableArray *listed = [NSMutableArray arrayWithCapacity:tools.count];
		for (NFKInferenceTool *tool in tools) {
			[listed addObject:@{ @"name": tool.name, @"description": tool.toolDescription, @"inputSchema": tool.parameters }];
		}
		return [self resultWithIdentifier:identifier result:@{ @"tools": listed }];
	}
	if ([method isEqualToString:@"tools/call"]) {
		NSString *name = [params[@"name"] isKindOfClass:NSString.class] ? params[@"name"] : nil;
		NFKInferenceTool *tool = nil;
		for (NFKInferenceTool *candidate in tools) {
			if ([candidate.name isEqualToString:name]) {
				tool = candidate;
				break;
			}
		}
		if (tool == nil) {
			return [self errorWithIdentifier:identifier code:NFKJSONRPCInvalidParams
									 message:[NSString stringWithFormat:@"no tool named %@", name ?: @"(none)"]];
		}
		NSDictionary *arguments = [params[@"arguments"] isKindOfClass:NSDictionary.class] ? params[@"arguments"] : @{};
		return [self resultWithIdentifier:identifier result:NFKMCPCallResult([tool invokeWithArguments:arguments])];
	}
	return [self errorWithIdentifier:identifier code:NFKJSONRPCMethodNotFound message:[@"no method " stringByAppendingString:method]];
}

// The client's version when the server speaks it, otherwise the newest the server speaks, which a
// client that cannot speak it disconnects from.
+ (NSDictionary *)initializeResult:(NSDictionary *)params instructions:(nullable NSString *)instructions
{
	NSString *requested = [params[@"protocolVersion"] isKindOfClass:NSString.class] ? params[@"protocolVersion"] : nil;
	NSString *version = requested != nil && [NFKMCPProtocolVersions() containsObject:requested] ? requested : NFKMCPProtocolVersions()[0];
	NSMutableDictionary *result = [NSMutableDictionary dictionaryWithDictionary:@{
		@"protocolVersion": version,
		@"capabilities": @{ @"tools": @{ @"listChanged": @NO } },
		@"serverInfo": @{ @"name": @"InferKit", @"version": NFKInferKit.version } }];
	if (instructions.length > 0) {
		result[@"instructions"] = instructions;
	}
	return result;
}

+ (NSDictionary *)resultWithIdentifier:(id)identifier result:(NSDictionary *)result
{
	return @{ @"jsonrpc": @"2.0", @"id": identifier ?: NSNull.null, @"result": result };
}

+ (NSDictionary *)errorWithIdentifier:(nullable id)identifier code:(NSInteger)code message:(NSString *)message
{
	return @{ @"jsonrpc": @"2.0", @"id": identifier ?: NSNull.null, @"error": @{ @"code": @(code), @"message": message } };
}

@end
