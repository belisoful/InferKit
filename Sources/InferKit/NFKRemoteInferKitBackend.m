//
//  NFKRemoteInferKitBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteInferKitBackend.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKHardwareProfile.h>
#import <InferKit/NFKModelCatalog.h>
#import <InferKit/NFKModelMemory.h>
#import "NFKInferenceWireCoding.h"
#import "NFKJobGuard.h"
#import "NFKLiveStream.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

@interface NFKRemoteInferKitBackend ()
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedInputKeys;
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedParameterKeys;
@property (nonatomic, copy, nullable) NSSet<NSString *> *preparedOutputKeys;
@property (atomic, readwrite, strong, nullable) NFKServerLoad *lastReportedLoad;
@end


/*! A result buffer as the value it describes: a pixel buffer for an image, a PCM buffer for audio. */
static id _Nullable NFKRemoteLiveBufferValue(NSDictionary *buffer, NSData * _Nullable payload)
{
	NSUInteger offset = [buffer[@"offset"] unsignedIntegerValue], length = [buffer[@"length"] unsignedIntegerValue];
	if (![buffer[@"key"] isKindOfClass:NSString.class] || offset > payload.length || length > payload.length - offset) {
		return nil;
	}
	const uint8_t *bytes = (const uint8_t *)payload.bytes + offset;
	NSString *format = buffer[@"format"];
	if ([format isEqualToString:@"pcm-f32"]) {
		NSInteger channels = [buffer[@"channels"] integerValue];
		double sampleRate = [buffer[@"sampleRate"] doubleValue];
		if (channels <= 0 || !(sampleRate > 0) || length % ((NSUInteger)channels * 4) != 0) {
			return nil;
		}
		AVAudioFormat *pcmFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32 sampleRate:sampleRate
																	  channels:(AVAudioChannelCount)channels interleaved:YES];
		AVAudioFrameCount frames = (AVAudioFrameCount)(length / ((NSUInteger)channels * 4));
		AVAudioPCMBuffer *pcm = [[AVAudioPCMBuffer alloc] initWithPCMFormat:pcmFormat frameCapacity:MAX(frames, 1)];
		pcm.frameLength = frames;
		memcpy(pcm.floatChannelData[0], bytes, length);
		return pcm;
	}
	NSDictionary<NSString *, NSArray<NSNumber *> *> *pixelFormats = @{ @"bgra": @[ @(kCVPixelFormatType_32BGRA), @4 ],
																		@"gray8": @[ @(kCVPixelFormatType_OneComponent8), @1 ],
																		@"gray-f32": @[ @(kCVPixelFormatType_OneComponent32Float), @4 ] };
	NSArray<NSNumber *> *pixelFormat = pixelFormats[format];
	size_t width = [buffer[@"width"] unsignedIntegerValue], height = [buffer[@"height"] unsignedIntegerValue];
	size_t rowBytes = [buffer[@"bytesPerRow"] unsignedIntegerValue];
	if (pixelFormat == nil || width == 0 || height == 0 || rowBytes < width * pixelFormat[1].unsignedIntegerValue || length < rowBytes * height) {
		return nil;
	}
	CVPixelBufferRef pixels = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	if (CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat[0].unsignedIntValue, (__bridge CFDictionaryRef)attributes,
							&pixels) != kCVReturnSuccess) {
		return nil;
	}
	CVPixelBufferLockBaseAddress(pixels, 0);
	uint8_t *destination = CVPixelBufferGetBaseAddress(pixels);
	size_t destinationRow = CVPixelBufferGetBytesPerRow(pixels);
	for (size_t row = 0; row < height; row++) {
		memcpy(destination + row * destinationRow, bytes + row * rowBytes, width * pixelFormat[1].unsignedIntegerValue);
	}
	CVPixelBufferUnlockBaseAddress(pixels, 0);
	return (__bridge_transfer id)pixels;
}

/*! A live route connection that sends a message and waits for the answer of a type. */
@interface NFKRemoteLiveSocket : NSObject
- (instancetype)initWithTask:(NSURLSessionWebSocketTask *)task timeout:(NSTimeInterval)timeout;
/*! Sends message and returns the next message of type with the sequence, or nil with the error the
	server answered, the transport's, or a timeout. A binary answer's bytes after its header land in
	payload. */
- (nullable NSDictionary *)send:(NSDictionary *)message
					  untilType:(NSString *)type
					   sequence:(nullable NSNumber *)sequence
						payload:(NSData * _Nullable * _Nullable)outPayload
						  error:(NSError * _Nullable *)outError;
- (void)close;
@end

@implementation NFKRemoteLiveSocket {
	NSURLSessionWebSocketTask *_task;
	NSTimeInterval _timeout;
}

- (instancetype)initWithTask:(NSURLSessionWebSocketTask *)task timeout:(NSTimeInterval)timeout
{
	self = [super init];
	if (self != nil) {
		_task = task;
		_timeout = timeout > 0 ? timeout : 60;
		[task resume];
	}
	return self;
}

- (nullable NSDictionary *)send:(NSDictionary *)message
					  untilType:(NSString *)type
					   sequence:(nullable NSNumber *)sequence
						payload:(NSData * _Nullable * _Nullable)outPayload
						  error:(NSError * _Nullable *)outError
{
	NSString *text = [[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:message options:0 error:NULL] ?: [NSData data]
										   encoding:NSUTF8StringEncoding];
	__block NSError *failure = nil;
	dispatch_semaphore_t sent = dispatch_semaphore_create(0);
	[_task sendMessage:[[NSURLSessionWebSocketMessage alloc] initWithString:text] completionHandler:^(NSError *error) {
		failure = error;
		dispatch_semaphore_signal(sent);
	}];
	if (dispatch_semaphore_wait(sent, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_timeout * NSEC_PER_SEC))) != 0 || failure != nil) {
		return [self fail:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the live route did not take the message in time"]
					error:outError];
	}
	for (;;) {
		__block NSURLSessionWebSocketMessage *received = nil;
		dispatch_semaphore_t arrived = dispatch_semaphore_create(0);
		[_task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage *answer, NSError *error) {
			received = answer;
			failure = error;
			dispatch_semaphore_signal(arrived);
		}];
		if (dispatch_semaphore_wait(arrived, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_timeout * NSEC_PER_SEC))) != 0) {
			return [self fail:[NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the live route did not answer in time"]
						error:outError];
		}
		if (received == nil) {
			return [self fail:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the live route closed"]
						error:outError];
		}
		NSData *headerData = nil, *payload = nil;
		if (received.type == NSURLSessionWebSocketMessageTypeString) {
			headerData = [received.string dataUsingEncoding:NSUTF8StringEncoding];
		} else if (received.data.length >= 4) {
			uint32_t length = 0;
			[received.data getBytes:&length length:4];
			length = CFSwapInt32LittleToHost(length);
			if ((NSUInteger)length <= received.data.length - 4) {
				headerData = [received.data subdataWithRange:NSMakeRange(4, length)];
				payload = [received.data subdataWithRange:NSMakeRange(4 + length, received.data.length - 4 - length)];
			}
		}
		NSDictionary *answer = headerData != nil ? [NSJSONSerialization JSONObjectWithData:headerData options:0 error:NULL] : nil;
		if (![answer isKindOfClass:NSDictionary.class]) {
			continue;
		}
		BOOL answersThis = sequence == nil || [answer[@"seq"] isEqual:sequence];
		if ([answer[@"type"] isEqualToString:@"error"] && (answer[@"seq"] == nil || answersThis)) {
			return [self fail:[NFKInferenceWireCoding errorFromJSONObject:answer]
						?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the live route failed the unit"]
						error:outError];
		}
		if ([answer[@"type"] isEqualToString:type] && answersThis) {
			if (outPayload != NULL) {
				*outPayload = payload;
			}
			return answer;
		}
	}
}

- (nullable NSDictionary *)fail:(NSError *)error error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = error;
	}
	return nil;
}

- (void)close
{
	[_task cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
}

@end

@implementation NFKRemoteInferKitBackend

@synthesize session = _session;

+ (instancetype)backendWithBaseURL:(NSURL *)baseURL
{
	return [self backendWithEndpointURL:[[baseURL URLByAppendingPathComponent:@"inferkit"] URLByAppendingPathComponent:@"run"]];
}

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteInferKitBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 600.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

- (nullable NSString *)pinnedCertificateSHA256
{
	return [NFKRemoteTransport pinnedCertificateSHA256ForSession:self.session];
}

- (void)setPinnedCertificateSHA256:(nullable NSString *)pinnedCertificateSHA256
{
	self.session = pinnedCertificateSHA256 != nil ? [NFKRemoteTransport sessionWithPinnedCertificateSHA256:pinnedCertificateSHA256]
												   : NSURLSession.sharedSession;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"inferkit-remote";
}

// The key sets are the hosted backend's, so they are declared only once prepareWithError: has read
// them; before that the backend answers as one that declares nothing.
- (BOOL)respondsToSelector:(SEL)selector
{
	if (selector == @selector(supportedInputKeys)) {
		return self.preparedInputKeys != nil;
	}
	if (selector == @selector(supportedParameterKeys)) {
		return self.preparedParameterKeys != nil;
	}
	if (selector == @selector(supportedOutputKeys)) {
		return self.preparedOutputKeys != nil;
	}
	return [super respondsToSelector:selector];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return self.preparedInputKeys ?: [NSSet set];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return self.preparedParameterKeys ?: [NSSet set];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return self.preparedOutputKeys ?: [NSSet set];
}

- (BOOL)prepareWithError:(NSError * _Nullable *)outError
{
	NSURL *base = self.endpointURL.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent;
	if (base == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSURL *url = [base URLByAppendingPathComponent:@"models"];
	if (self.modelName.length > 0) {
		url = [url URLByAppendingPathComponent:self.modelName];
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = MIN(self.timeout, 30.0);
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	NSDictionary *entry = [self JSONObjectForRequest:request error:outError];
	if (entry == nil) {
		return NO;
	}
	if (self.modelName.length == 0) {
		NSArray *models = [entry[@"data"] isKindOfClass:NSArray.class] ? entry[@"data"] : @[];
		if (models.count != 1) {
			return [self failWithCode:kNFKError_InferenceNotReady
							   reason:@"the server hosts more than one model; set modelName" error:outError];
		}
		entry = models.firstObject;
	}
	NSDictionary *description = [entry isKindOfClass:NSDictionary.class] && [entry[@"inferkit"] isKindOfClass:NSDictionary.class]
		? entry[@"inferkit"] : @{};
	if ([description[@"inputs"] isKindOfClass:NSArray.class]) {
		self.preparedInputKeys = [NSSet setWithArray:description[@"inputs"]];
	}
	if ([description[@"parameters"] isKindOfClass:NSArray.class]) {
		self.preparedParameterKeys = [NSSet setWithArray:description[@"parameters"]];
	}
	if ([description[@"outputs"] isKindOfClass:NSArray.class]) {
		self.preparedOutputKeys = [NSSet setWithArray:description[@"outputs"]];
	}
	if (![description[@"ready"] boolValue]) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"the served model reports it is not ready" error:outError];
	}
	return YES;
}

/*! A request for a route under the server's base URL, authorized, or nil when no endpoint is set. */
- (nullable NSMutableURLRequest *)serverRequestForPath:(NSString *)path error:(NSError * _Nullable *)outError
{
	NSURL *base = self.endpointURL.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent;
	if (base == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
		return nil;
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[base URLByAppendingPathComponent:path]];
	request.timeoutInterval = MIN(self.timeout, 30.0);
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return request;
}

- (nullable NFKServerStatus *)fetchServerStatusWithError:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self serverRequestForPath:@"inferkit/status" error:outError];
	id reply = request != nil ? [self JSONObjectForRequest:request error:outError] : nil;
	if (reply == nil) {
		return nil;
	}
	NFKServerStatus *status = [NFKServerStatus statusWithJSONObject:reply];
	if (status == nil) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the server's reply is not a status" error:outError];
	}
	return status;
}

- (nullable NFKHardwareProfile *)fetchServerHardwareProfileWithError:(NSError * _Nullable *)outError
{
	NFKServerStatus *status = [self fetchServerStatusWithError:outError];
	if (status == nil) {
		return nil;
	}
	if (status.host.hardwareProfile == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"the server leaves its machine's details out of its status"
					 error:outError];
	}
	return status.host.hardwareProfile;
}

- (nullable NFKModelCatalog *)fetchModelCatalogWithError:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self serverRequestForPath:@"inferkit/catalog" error:outError];
	NSData *manifest = request != nil ? [self replyDataForRequest:request error:outError] : nil;
	if (manifest == nil) {
		return nil;
	}
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	return [catalog addDescriptorsFromManifestData:manifest error:outError] ? catalog : nil;
}

- (nullable NFKModelMemoryEstimate *)fetchMemoryEstimateForModelNamed:(nullable NSString *)modelName
															  options:(nullable NSDictionary<NSString *, id> *)options
														  constraints:(NFKModelMemoryConstraints * _Nullable * _Nullable)constraints
																error:(NSError * _Nullable *)outError
{
	NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:options ?: @{} forKey:@"options"];
	body[@"model"] = modelName;
	if (![NSJSONSerialization isValidJSONObject:body]) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"the options are not JSON values" error:outError];
		return nil;
	}
	NSMutableURLRequest *request = [self serverRequestForPath:@"inferkit/estimate" error:outError];
	NSData *data = request != nil ? [NSJSONSerialization dataWithJSONObject:body options:0 error:outError] : nil;
	if (data == nil) {
		return nil;
	}
	request.HTTPMethod = @"POST";
	request.HTTPBody = data;
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	NSDictionary *reply = [self JSONObjectForRequest:request error:outError];
	NSDictionary *estimated = [reply[@"estimate"] isKindOfClass:NSDictionary.class] ? reply[@"estimate"] : nil;
	NFKModelMemoryEstimate *estimate = estimated != nil ? [NFKModelMemoryEstimate estimateWithDictionary:estimated error:outError] : nil;
	if (estimate == nil) {
		if (reply != nil && estimated == nil) {
			[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the server's reply carries no estimate" error:outError];
		}
		return nil;
	}
	if (constraints != NULL) {
		NSDictionary *limits = [reply[@"constraints"] isKindOfClass:NSDictionary.class] ? reply[@"constraints"] : nil;
		*constraints = limits != nil ? [NFKModelMemoryConstraints constraintsWithDictionary:limits error:NULL] : nil;
	}
	return estimate;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self URLRequestForRequest:request streaming:NO error:outError];
	id reply = urlRequest != nil ? [self JSONObjectForRequest:urlRequest error:outError] : nil;
	if (reply == nil) {
		return nil;
	}
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	return [decoder resultFromJSONObject:[reply isKindOfClass:NSDictionary.class] ? reply[@"result"] : nil error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self URLRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error];
		return job;
	}
	[job reportProgress:-1.0];
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (finished || ![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
		if ([type isEqualToString:@"progress"]) {
			double progress = [event[@"progress"] isKindOfClass:NSNumber.class] ? [event[@"progress"] doubleValue] : -1.0;
			NFKInferenceResult *partial = event[@"partial"] != nil ? [decoder resultFromJSONObject:event[@"partial"] error:NULL] : nil;
			[job reportProgress:progress partialResult:partial];
		} else if ([type isEqualToString:@"result"]) {
			finished = YES;
			NSError *decodeError = nil;
			NFKInferenceResult *result = [decoder resultFromJSONObject:event[@"result"] error:&decodeError];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:decodeError];
			}
		} else if ([type isEqualToString:@"error"]) {
			finished = YES;
			[job finishWithError:[NFKInferenceWireCoding errorFromJSONObject:@{ @"error": event[@"error"] ?: @{} }]
				?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the served run failed"]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		NFKJobRunGuarded(job, ^{
			[self recordLoadFromResponse:response];
			if (finished || job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			finished = YES;
			NSError *failure = streamError ?: [self errorForResponse:response data:errorBody];
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
																	   reason:@"the server closed the stream without a result"]];
		});
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Requests

- (nullable NSMutableURLRequest *)URLRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
		return nil;
	}
	NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	id encoded = [encoder JSONObjectForRequest:request error:outError];
	if (encoded == nil) {
		return nil;
	}
	NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:encoded forKey:@"request"];
	body[@"model"] = self.modelName;
	body[@"stream"] = @(streaming);
	NSData *data = [NSJSONSerialization dataWithJSONObject:body options:0 error:outError];
	if (data == nil) {
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.HTTPBody = data;
	urlRequest.timeoutInterval = self.timeout;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

/*! The body of a successful reply, or nil with the server's or the transport's error. */
- (nullable NSData *)replyDataForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	[self recordLoadFromResponse:response];
	NSError *failure = data == nil ? sendError : [self errorForResponse:response data:data];
	if (failure != nil) {
		if (outError != NULL) {
			*outError = failure;
		}
		return nil;
	}
	return data;
}

- (nullable id)JSONObjectForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSData *data = [self replyDataForRequest:request error:outError];
	if (data == nil) {
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the server's reply is not a JSON object" error:outError];
		return nil;
	}
	return object;
}

/*! Keeps the load a reply's headers carry; a reply without them leaves the last one in place. */
- (void)recordLoadFromResponse:(nullable NSHTTPURLResponse *)response
{
	NFKServerLoad *load = response != nil ? [NFKServerLoad loadWithHTTPHeaders:response.allHeaderFields] : nil;
	if (load != nil) {
		self.lastReportedLoad = load;
	}
}

/*! The server's own error when its body carries one, domain and code intact, keeping the transport's
	Retry-After date; otherwise the transport's reading of the status. */
- (nullable NSError *)errorForResponse:(nullable NSHTTPURLResponse *)response data:(nullable NSData *)data
{
	NSError *transportError = [NFKRemoteTransport errorForResponse:response data:data];
	if (transportError == nil || data.length == 0) {
		return transportError;
	}
	NSError *served = [NFKInferenceWireCoding errorFromJSONObject:[NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]];
	if (served == nil) {
		return transportError;
	}
	NSMutableDictionary *userInfo = [served.userInfo mutableCopy];
	userInfo[NFKRemoteErrorStatusCodeKey] = transportError.userInfo[NFKRemoteErrorStatusCodeKey];
	userInfo[NFKRemoteErrorRetryAfterKey] = transportError.userInfo[NFKRemoteErrorRetryAfterKey];
	return [NSError errorWithDomain:served.domain code:served.code userInfo:userInfo];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable, NSData * _Nullable, NSError * _Nullable))completionHandler
{
	return [NFKRemoteTransport streamRequest:request session:self.session lineHandler:lineHandler completionHandler:completionHandler];
}

#pragma mark Errors

#pragma mark Live streams

- (nullable id<NFKLiveStream>)openLiveStreamWithParameters:(nullable NSDictionary<NSString *, id> *)parameters
													 error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *request = [self serverRequestForPath:@"inferkit/live" error:outError];
	if (request == nil) {
		return nil;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:request.URL resolvingAgainstBaseURL:NO];
	components.scheme = [components.scheme.lowercaseString isEqualToString:@"https"] ? @"wss" : @"ws";
	request.URL = components.URL;
	NFKRemoteLiveSocket *socket = [[NFKRemoteLiveSocket alloc] initWithTask:[self.session webSocketTaskWithRequest:request]
																	timeout:self.timeout];
	NSMutableDictionary *start = [@{ @"type": @"start", @"parameters": parameters ?: @{} } mutableCopy];
	start[@"model"] = self.modelName;
	NSError *error = nil;
	NSDictionary *started = [socket send:start untilType:@"started" sequence:nil payload:NULL error:&error];
	if (started == nil) {
		[socket close];
		if (outError != NULL) {
			*outError = error;
		}
		return nil;
	}
	__block NSInteger sequence = 0;
	return [[NFKBlockLiveStream alloc] initWithParameters:nil unitHandler:^NFKInferenceResult *(NFKInferenceRequest *unit, NSError **unitError) {
		NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
		NSDictionary *encoded = [coder JSONObjectForRequest:unit error:unitError];
		if (encoded == nil) {
			return nil;
		}
		sequence += 1;
		NSMutableDictionary *message = [@{ @"type": @"unit", @"seq": @(sequence) } mutableCopy];
		message[@"inputs"] = encoded[@"inputs"];
		message[@"parameters"] = encoded[@"parameters"];
		NSData *payload = nil;
		NSDictionary *answer = [socket send:message untilType:@"result" sequence:@(sequence) payload:&payload error:unitError];
		if (answer == nil) {
			return nil;
		}
		NFKInferenceResult *result = [coder resultFromJSONObject:@{ @"outputs": answer[@"outputs"] ?: @{} } error:unitError];
		if (result == nil) {
			return nil;
		}
		NSMutableDictionary<NSString *, id> *outputs = [result.outputs mutableCopy];
		for (NSDictionary *buffer in [answer[@"buffers"] isKindOfClass:NSArray.class] ? answer[@"buffers"] : @[]) {
			id value = NFKRemoteLiveBufferValue(buffer, payload);
			if (value == nil) {
				if (unitError != NULL) {
					*unitError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
															reason:@"a result buffer does not agree with its description"];
				}
				return nil;
			}
			outputs[buffer[@"key"]] = value;
		}
		return [NFKInferenceResult resultWithOutputs:outputs];
	} resetHandler:^{
		[socket send:@{ @"type": @"reset" } untilType:@"reset" sequence:nil payload:NULL error:NULL];
	} closeHandler:^{
		[socket close];
	}];
}

- (BOOL)failWithCode:(NSInteger)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:(NFKInferenceError)code reason:reason];
	}
	return NO;
}

@end
