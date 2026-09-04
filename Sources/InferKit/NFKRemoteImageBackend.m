//
//  NFKRemoteImageBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteImageBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteImageBackend

@synthesize session = _session;

+ (instancetype)backendWithGenerationsURL:(nullable NSURL *)generationsURL editsURL:(nullable NSURL *)editsURL
{
	NFKRemoteImageBackend *backend = [[self alloc] init];
	backend.generationsURL = generationsURL;
	backend.editsURL = editsURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (provider.apiStyle != NFKRemoteAPIStyleOpenAIChat) {
		return nil;
	}
	NFKRemoteImageBackend *backend = [self backendWithGenerationsURL:[provider URLForPath:@"images/generations"]
															editsURL:[provider URLForPath:@"images/edits"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 180.0;
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
	return self.generationsURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote-image";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSString *prompt = request.prompt;
	if (prompt.length == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries no prompt" error:outError];
	}
	id sourceImage = [request inputForKey:NFKInputImage];
	NSMutableURLRequest *urlRequest = sourceImage != nil
		? [self editRequestForPrompt:prompt image:sourceImage mask:[request inputForKey:NFKInputMask]
						  parameters:request.parameters error:outError]
		: [self generationRequestForPrompt:prompt parameters:request.parameters error:outError];
	if (urlRequest == nil) {
		return nil;
	}

	NSDictionary *body = [self JSONObjectForRequest:urlRequest error:outError];
	if (body == nil) {
		return nil;
	}
	CVPixelBufferRef pixelBuffer = [self firstImageInBody:body error:outError];
	if (pixelBuffer == NULL) {
		return nil;
	}
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: (__bridge id)pixelBuffer,
																		   NFKRemoteBackendRawKey: body }];
	CVPixelBufferRelease(pixelBuffer);
	return result;
}

#pragma mark Requests

// The fields the contract names map onto the service's; everything else passes by name.
- (NSMutableDictionary<NSString *, id> *)fieldsForPrompt:(NSString *)prompt parameters:(NSDictionary<NSString *, id> *)parameters
{
	NSMutableDictionary<NSString *, id> *fields = [NSMutableDictionary dictionary];
	fields[@"prompt"] = prompt;
	if (self.modelName.length > 0) {
		fields[@"model"] = self.modelName;
	}
	NSNumber *width = parameters[NFKParameterWidth];
	NSNumber *height = parameters[NFKParameterHeight];
	if ([width isKindOfClass:NSNumber.class] && [height isKindOfClass:NSNumber.class]) {
		fields[@"size"] = [NSString stringWithFormat:@"%ldx%ld", (long)width.integerValue, (long)height.integerValue];
	}
	NSNumber *seed = parameters[NFKParameterSeed];
	if ([seed isKindOfClass:NSNumber.class]) {
		fields[@"seed"] = seed;
	}
	NSNumber *steps = parameters[NFKParameterSteps];
	if ([steps isKindOfClass:NSNumber.class]) {
		fields[@"steps"] = steps;
	}
	NSSet<NSString *> *mapped = [NSSet setWithArray:@[ NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps ]];
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			fields[key] = parameters[key];
		}
	}
	return fields;
}

- (nullable NSMutableURLRequest *)generationRequestForPrompt:(NSString *)prompt
												  parameters:(NSDictionary<NSString *, id> *)parameters
													   error:(NSError * _Nullable *)outError
{
	if (self.generationsURL == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no generations URL is set" error:outError];
		return nil;
	}
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:[self fieldsForPrompt:prompt parameters:parameters]
													  options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *request = [self requestForURL:self.generationsURL];
	request.HTTPBody = payload;
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	return request;
}

- (nullable NSMutableURLRequest *)editRequestForPrompt:(NSString *)prompt
												 image:(id)image
												  mask:(nullable id)mask
											parameters:(NSDictionary<NSString *, id> *)parameters
												 error:(NSError * _Nullable *)outError
{
	if (self.editsURL == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"the service has no edit endpoint" error:outError];
		return nil;
	}
	NSData *imagePNG = [NFKImageCoding PNGDataForImage:image];
	if (imagePNG == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:@"the image under NFKInputImage is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture" error:outError];
		return nil;
	}
	NSData *maskPNG = nil;
	if (mask != nil) {
		maskPNG = [NFKImageCoding PNGDataForImage:mask];
		if (maskPNG == nil) {
			[self failWithCode:kNFKError_InferenceMissingInput
						reason:@"the mask under NFKInputMask is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture" error:outError];
			return nil;
		}
	}

	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSString *dashBoundary = [NSString stringWithFormat:@"--%@\r\n", boundary];
	void (^appendField)(NSString *, NSString *) = ^(NSString *name, NSString *value) {
		[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
		NSString *part = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, value];
		[body appendData:[part dataUsingEncoding:NSUTF8StringEncoding]];
	};
	void (^appendFile)(NSString *, NSString *, NSData *) = ^(NSString *name, NSString *filename, NSData *data) {
		[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
		NSString *header = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"; filename=\"%@\"\r\nContent-Type: image/png\r\n\r\n", name, filename];
		[body appendData:[header dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:data];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	};

	NSDictionary<NSString *, id> *fields = [self fieldsForPrompt:prompt parameters:parameters];
	for (NSString *key in fields) {
		appendField(key, [fields[key] description]);
	}
	appendFile(@"image", @"image.png", imagePNG);
	if (maskPNG != nil) {
		appendFile(@"mask", @"mask.png", maskPNG);
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

	NSMutableURLRequest *request = [self requestForURL:self.editsURL];
	request.HTTPBody = body;
	[request setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	return request;
}

- (NSMutableURLRequest *)requestForURL:(NSURL *)url
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"POST";
	request.timeoutInterval = self.timeout;
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return request;
}

#pragma mark Response

- (nullable NSDictionary *)JSONObjectForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:request response:&response error:&error];
	if (data == nil) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object" error:outError];
		return nil;
	}
	return object;
}

// The envelope is data[] of {b64_json} or {url}; the first entry is the image.
- (nullable CVPixelBufferRef)firstImageInBody:(NSDictionary *)body error:(NSError * _Nullable *)outError CF_RETURNS_RETAINED
{
	NSArray *entries = body[@"data"];
	NSDictionary *entry = [entries isKindOfClass:NSArray.class] && entries.count > 0 ? entries.firstObject : nil;
	if (![entry isKindOfClass:NSDictionary.class]) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no image" error:outError];
		return NULL;
	}
	NSData *bytes = nil;
	if ([entry[@"b64_json"] isKindOfClass:NSString.class]) {
		bytes = [[NSData alloc] initWithBase64EncodedString:entry[@"b64_json"] options:NSDataBase64DecodingIgnoreUnknownCharacters];
	} else if ([entry[@"url"] isKindOfClass:NSString.class]) {
		bytes = [self fetchImageAtURL:[NSURL URLWithString:entry[@"url"]] error:outError];
		if (bytes == nil) {
			return NULL;
		}
	}
	CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
	if (pixelBuffer == NULL) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response's image could not be decoded" error:outError];
	}
	return pixelBuffer;
}

- (nullable NSData *)fetchImageAtURL:(nullable NSURL *)url error:(NSError * _Nullable *)outError
{
	if (url == nil) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response names no usable image URL" error:outError];
		return nil;
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = self.timeout;
	NSHTTPURLResponse *response = nil;
	NSError *error = nil;
	NSData *data = [self sendRequest:request response:&response error:&error];
	if (data == nil) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	return data;
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
