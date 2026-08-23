//
//  NFKRemoteTranscriptionBackend.m
//  InferKit
//

#import "NFKRemoteTranscriptionBackend.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceKeys.h"
#import "NFKAudioAsset.h"
#import "NFKErrors.h"

@implementation NFKRemoteTranscriptionBackend

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteTranscriptionBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 60.0;
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
	return @"remote-transcription";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
		return nil;
	}

	NSString *filename = nil;
	NSData *audio = [self audioDataForRequest:request filename:&filename];
	if (audio == nil) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"no audio is set under NFKInputAudio"];
		return nil;
	}

	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSData *body = [self multipartBodyForAudio:audio filename:filename request:request boundary:boundary];

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = body;
	NSString *contentType = [NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary];
	[urlRequest setValue:contentType forHTTPHeaderField:@"Content-Type"];
	if (self.apiKey.length > 0) {
		[urlRequest setValue:[@"Bearer " stringByAppendingString:self.apiKey] forHTTPHeaderField:@"Authorization"];
	}

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *responseData = [self sendRequest:urlRequest response:&response error:&sendError];
	if (responseData == nil) {
		[self propagateError:sendError to:outError];
		return nil;
	}
	if (response != nil && (response.statusCode < 200 || response.statusCode >= 300)) {
		NSString *reason = [NSString stringWithFormat:@"the endpoint returned HTTP %ld", (long)response.statusCode];
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:reason];
		return nil;
	}
	return [self resultFromResponseData:responseData error:outError];
}

#pragma mark Audio and body

- (nullable NSData *)audioDataForRequest:(NFKInferenceRequest *)request filename:(NSString * _Nullable *)outFilename
{
	id audio = [request inputForKey:NFKInputAudio];
	if ([audio isKindOfClass:NFKAudioAsset.class]) {
		NFKAudioAsset *asset = audio;
		if (asset.fileURL == nil) {
			return nil;
		}
		if (outFilename != NULL) {
			NSString *name = asset.fileURL.lastPathComponent;
			*outFilename = name.length > 0 ? name : @"audio.wav";
		}
		return [NSData dataWithContentsOfURL:asset.fileURL];
	}
	if ([audio isKindOfClass:NSData.class]) {
		if (outFilename != NULL) {
			*outFilename = @"audio.wav";
		}
		return audio;
	}
	return nil;
}

- (NSData *)multipartBodyForAudio:(NSData *)audio
						 filename:(NSString *)filename
						  request:(NFKInferenceRequest *)request
						 boundary:(NSString *)boundary
{
	NSMutableData *body = [NSMutableData data];
	NSString *dashBoundary = [NSString stringWithFormat:@"--%@\r\n", boundary];

	void (^appendField)(NSString *, NSString *) = ^(NSString *name, NSString *value) {
		[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
		NSString *disposition = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, value];
		[body appendData:[disposition dataUsingEncoding:NSUTF8StringEncoding]];
	};

	if (self.modelName.length > 0) {
		appendField(@"model", self.modelName);
	}
	// Parameters fold in as form fields so a caller sets language, prompt, response_format, temperature.
	for (NSString *key in request.parameters) {
		appendField(key, [request.parameters[key] description]);
	}

	[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
	NSString *fileHeader = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\nContent-Type: application/octet-stream\r\n\r\n", filename];
	[body appendData:[fileHeader dataUsingEncoding:NSUTF8StringEncoding]];
	[body appendData:audio];
	[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];

	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
	return body;
}

- (nullable NFKInferenceResult *)resultFromResponseData:(NSData *)data error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if ([object isKindOfClass:NSDictionary.class]) {
		NSDictionary *responseBody = object;
		if ([responseBody[@"text"] isKindOfClass:NSString.class]) {
			outputs[NFKOutputText] = responseBody[@"text"];
		}
		outputs[NFKOutputStructured] = responseBody;
	} else {
		// response_format=text/srt/vtt returns a plain body, not JSON; take it as the transcript.
		NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
		if (text == nil) {
			[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the response is neither JSON nor text"];
			return nil;
		}
		outputs[NFKOutputText] = text;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError
{
	__block NSData *resultData = nil;
	__block NSURLResponse *resultResponse = nil;
	__block NSError *resultError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request
												completionHandler:^(NSData *data, NSURLResponse *response, NSError *taskError) {
		resultData = data;
		resultResponse = response;
		resultError = taskError;
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (outResponse != NULL && [resultResponse isKindOfClass:NSHTTPURLResponse.class]) {
		*outResponse = (NSHTTPURLResponse *)resultResponse;
	}
	if (resultData == nil && outError != NULL) {
		*outError = resultError != nil ? resultError
									  : [NSError errorWithDomain:NFKInferenceErrorDomain
															code:kNFKError_InferenceBackendFailure
														userInfo:@{ NSLocalizedDescriptionKey: @"the request returned no data" }];
	}
	return resultData;
}

#pragma mark Errors

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	*outError = error != nil ? error
							: [NSError errorWithDomain:NFKInferenceErrorDomain
												  code:kNFKError_InferenceBackendFailure
											  userInfo:@{ NSLocalizedDescriptionKey: @"the transcription call failed" }];
	return NO;
}

@end
