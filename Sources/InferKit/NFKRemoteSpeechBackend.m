//
//  NFKRemoteSpeechBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteSpeechBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteSpeechBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteSpeechBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
									  voice:(nullable NSString *)voice
{
	if (provider.apiStyle != NFKRemoteAPIStyleOpenAIChat) {
		return nil;
	}
	NFKRemoteSpeechBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"audio/speech"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	backend.voice = voice;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_responseFormat = @"wav";
		_timeout = 120.0;
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

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote-speech";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSString *text = [self textForRequest:request];
	if (text == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries neither a prompt nor messages" error:outError];
	}

	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	body[@"input"] = text;
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	if (self.voice.length > 0) {
		body[@"voice"] = self.voice;
	}
	body[@"response_format"] = self.responseFormat;
	// Parameters fold into the body so a caller sets voice, speed, instructions, and similar.
	for (NSString *key in request.parameters) {
		body[key] = request.parameters[key];
	}
	if ([body[@"voice"] length] == 0) {
		return [self failWithCode:kNFKError_InferenceNotReady
						   reason:@"no voice is set; every speech service requires one" error:outError];
	}

	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *audio = [self sendRequest:urlRequest response:&response error:&sendError];
	if (audio == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:audio];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	if (audio.length == 0) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the endpoint returned no audio" error:outError];
	}

	NSURL *fileURL = [self writeAudio:audio format:[body[@"response_format"] description] error:outError];
	if (fileURL == nil) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputAudio: [NFKAudioAsset audioAssetWithFileURL:fileURL] }];
}

#pragma mark Text and file

- (nullable NSString *)textForRequest:(NFKInferenceRequest *)request
{
	NSString *prompt = request.prompt;
	if (prompt.length > 0) {
		return prompt;
	}
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	for (NSDictionary *message in request.messages) {
		if ([message isKindOfClass:NSDictionary.class] && [message[@"content"] isKindOfClass:NSString.class]) {
			[pieces addObject:message[@"content"]];
		}
	}
	return pieces.count > 0 ? [pieces componentsJoinedByString:@"\n"] : nil;
}

- (nullable NSURL *)writeAudio:(NSData *)audio format:(NSString *)format error:(NSError * _Nullable *)outError
{
	NSURL *directory = self.outputDirectoryURL
		?: [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES];
	NSError *error = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSString *extension = format.length > 0 ? format : @"wav";
	NSURL *fileURL = [directory URLByAppendingPathComponent:[NSString stringWithFormat:@"speech-%@.%@", NSUUID.UUID.UUIDString, extension]];
	if (![audio writeToURL:fileURL options:NSDataWritingAtomic error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	return fileURL;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable NFKInferenceResult *)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
