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
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! The contract keys this backend translates; every other parameter goes out under its own name. */
static NSSet<NSString *> *NFKImageMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterWidth, NFKParameterHeight, NFKParameterSeed, NFKParameterSteps,
										NFKParameterGuidanceScale, NFKParameterSampleCount, NFKParameterOutputFormat,
										NFKParameterAspectRatio, NFKParameterResolution ]];
	});
	return mapped;
}

static NSString *NFKImageDataURI(NSData *png)
{
	return [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
}

static NSInteger NFKImageGreatestCommonDivisor(NSInteger a, NSInteger b)
{
	while (b != 0) {
		NSInteger remainder = a % b;
		a = b;
		b = remainder;
	}
	return a;
}

/*! The images a request edits, as PNG: NFKInputImage first, then NFKInputImages. */
@interface NFKImageSources : NSObject
@property (nonatomic, copy) NSArray<NSData *> *images;
@property (nonatomic, copy, nullable) NSData *mask;
@end

@implementation NFKImageSources
@end

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
	NSString *identifier = provider.identifier;
	NSURL *generations = [provider URLForPath:@"images/generations"];
	NFKRemoteImageBackend *backend = nil;
	if ([identifier isEqualToString:@"openai"]) {
		backend = [self backendWithGenerationsURL:generations editsURL:[provider URLForPath:@"images/edits"]];
	} else if ([identifier isEqualToString:@"gemini"]) {
		backend = [self backendWithGenerationsURL:generations editsURL:nil];
	} else if ([identifier isEqualToString:@"xai"]) {
		backend = [self backendWithGenerationsURL:generations editsURL:[provider URLForPath:@"images/edits"]];
		backend.apiStyle = NFKRemoteImageAPIStyleXAI;
	} else if ([identifier isEqualToString:@"together"]) {
		backend = [self backendWithGenerationsURL:generations editsURL:generations];
		backend.apiStyle = NFKRemoteImageAPIStyleTogether;
	} else if ([identifier isEqualToString:@"openrouter"]) {
		NSURL *images = [provider URLForPath:@"images"];
		backend = [self backendWithGenerationsURL:images editsURL:images];
		backend.apiStyle = NFKRemoteImageAPIStyleOpenRouter;
	}
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
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *body = [self JSONObjectForRequest:urlRequest error:outError];
	if (body == nil) {
		return nil;
	}
	return [self resultFromBody:body error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	BOOL streamable = self.streams && (self.apiStyle == NFKRemoteImageAPIStyleOpenAI || self.apiStyle == NFKRemoteImageAPIStyleOpenRouter);
	if (!streamable) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			if (job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			[job reportProgress:-1.0];
			NSError *error = nil;
			NFKInferenceResult *result = [self runInferenceForRequest:request error:&error];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the image call failed"]];
			}
		});
		return job;
	}
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the image call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NSMutableArray<NSDictionary *> *completed = [NSMutableArray array];
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (finished || ![event isKindOfClass:NSDictionary.class] || ![event[@"b64_json"] isKindOfClass:NSString.class]) {
			return;
		}
		NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
		if ([type hasSuffix:@".completed"]) {
			[completed addObject:@{ @"b64_json": event[@"b64_json"] }];
			return;
		}
		NSData *bytes = [[NSData alloc] initWithBase64EncodedString:event[@"b64_json"] options:NSDataBase64DecodingIgnoreUnknownCharacters];
		CVPixelBufferRef partial = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
		if (partial != NULL) {
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: (__bridge id)partial }]];
			CVPixelBufferRelease(partial);
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		NFKInferenceResult *result = failure == nil ? [self resultFromBody:@{ @"data": completed } error:&failure] : nil;
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the image call failed"]];
		}
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Requests

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	NSString *prompt = request.prompt;
	if (prompt.length == 0) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries no prompt" error:outError];
		return nil;
	}
	if (self.generationsURL == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no generations URL is set" error:outError];
		return nil;
	}
	NFKImageSources *sources = [self sourcesForRequest:request error:outError];
	if (sources == nil) {
		return nil;
	}
	BOOL edits = sources.images.count > 0;
	if (edits && self.editsURL == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"the service has no edit endpoint" error:outError];
		return nil;
	}
	if (sources.mask != nil && self.apiStyle != NFKRemoteImageAPIStyleOpenAI) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"only the OpenAI-style edit takes a mask" error:outError];
		return nil;
	}
	NSMutableDictionary<NSString *, id> *fields = [self fieldsForPrompt:prompt request:request sources:sources];
	if (streaming) {
		fields[@"stream"] = @YES;
		if (self.apiStyle == NFKRemoteImageAPIStyleOpenAI && fields[@"partial_images"] == nil) {
			fields[@"partial_images"] = @2;
		}
	}
	if (edits && self.apiStyle == NFKRemoteImageAPIStyleOpenAI) {
		return [self multipartRequestToURL:self.editsURL fields:fields sources:sources];
	}
	return [self JSONRequestToURL:edits ? self.editsURL : self.generationsURL fields:fields error:outError];
}

- (nullable NFKImageSources *)sourcesForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSMutableArray *inputs = [NSMutableArray array];
	id first = [request inputForKey:NFKInputImage];
	if (first != nil) {
		[inputs addObject:first];
	}
	id more = [request inputForKey:NFKInputImages];
	if ([more isKindOfClass:NSArray.class]) {
		[inputs addObjectsFromArray:more];
	}
	NSMutableArray<NSData *> *images = [NSMutableArray array];
	for (id input in inputs) {
		NSData *png = [NFKImageCoding PNGDataForImage:input];
		if (png == nil) {
			[self failWithCode:kNFKError_InferenceMissingInput
						reason:@"an image under NFKInputImage or NFKInputImages is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture" error:outError];
			return nil;
		}
		[images addObject:png];
	}
	NFKImageSources *sources = [[NFKImageSources alloc] init];
	sources.images = images;
	id mask = [request inputForKey:NFKInputMask];
	if (mask != nil) {
		sources.mask = [NFKImageCoding PNGDataForImage:mask];
		if (sources.mask == nil) {
			[self failWithCode:kNFKError_InferenceMissingInput
						reason:@"the mask under NFKInputMask is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture" error:outError];
			return nil;
		}
	}
	return sources;
}

// The contract's fields in each style's spelling; everything else passes by name.
- (NSMutableDictionary<NSString *, id> *)fieldsForPrompt:(NSString *)prompt request:(NFKInferenceRequest *)request sources:(NFKImageSources *)sources
{
	NSDictionary *parameters = request.parameters;
	NSMutableDictionary<NSString *, id> *fields = [NSMutableDictionary dictionary];
	fields[@"prompt"] = prompt;
	if (self.modelName.length > 0) {
		fields[@"model"] = self.modelName;
	}
	NSNumber *width = [parameters[NFKParameterWidth] isKindOfClass:NSNumber.class] ? parameters[NFKParameterWidth] : nil;
	NSNumber *height = [parameters[NFKParameterHeight] isKindOfClass:NSNumber.class] ? parameters[NFKParameterHeight] : nil;
	NSString *size = width != nil && height != nil
		? [NSString stringWithFormat:@"%ldx%ld", (long)width.integerValue, (long)height.integerValue] : nil;
	fields[@"n"] = parameters[NFKParameterSampleCount];
	fields[@"output_format"] = parameters[NFKParameterOutputFormat];
	id negative = [request inputForKey:NFKInputNegativePrompt];

	switch (self.apiStyle) {
		case NFKRemoteImageAPIStyleOpenAI:
			fields[@"size"] = size;
			fields[@"seed"] = parameters[NFKParameterSeed];
			fields[@"steps"] = parameters[NFKParameterSteps];
			fields[@"aspect_ratio"] = parameters[NFKParameterAspectRatio];
			break;
		case NFKRemoteImageAPIStyleXAI: {
			fields[@"aspect_ratio"] = parameters[NFKParameterAspectRatio] ?: [self ratioForWidth:width height:height];
			fields[@"resolution"] = parameters[NFKParameterResolution];
			fields[@"response_format"] = @"b64_json";
			NSMutableArray *images = [NSMutableArray array];
			for (NSData *png in sources.images) {
				[images addObject:@{ @"url": NFKImageDataURI(png) }];
			}
			if (images.count == 1) {
				fields[@"image"] = images.firstObject;
			} else if (images.count > 1) {
				fields[@"images"] = images;
			}
			break;
		}
		case NFKRemoteImageAPIStyleTogether: {
			fields[@"width"] = width;
			fields[@"height"] = height;
			fields[@"aspect_ratio"] = parameters[NFKParameterAspectRatio];
			fields[@"steps"] = parameters[NFKParameterSteps];
			fields[@"seed"] = parameters[NFKParameterSeed];
			fields[@"guidance_scale"] = parameters[NFKParameterGuidanceScale];
			fields[@"negative_prompt"] = negative;
			fields[@"response_format"] = @"base64";
			if (sources.images.count > 0) {
				fields[@"image_url"] = NFKImageDataURI(sources.images.firstObject);
			}
			if (sources.images.count > 1) {
				NSMutableArray<NSString *> *references = [NSMutableArray array];
				for (NSData *png in [sources.images subarrayWithRange:NSMakeRange(1, sources.images.count - 1)]) {
					[references addObject:NFKImageDataURI(png)];
				}
				fields[@"reference_images"] = references;
			}
			break;
		}
		case NFKRemoteImageAPIStyleOpenRouter: {
			fields[@"size"] = size;
			fields[@"aspect_ratio"] = parameters[NFKParameterAspectRatio];
			fields[@"resolution"] = parameters[NFKParameterResolution];
			fields[@"seed"] = parameters[NFKParameterSeed];
			NSMutableArray<NSString *> *references = [NSMutableArray array];
			for (NSData *png in sources.images) {
				[references addObject:NFKImageDataURI(png)];
			}
			if (references.count > 0) {
				fields[@"input_references"] = references;
			}
			break;
		}
	}
	NSSet<NSString *> *mapped = NFKImageMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			fields[key] = parameters[key];
		}
	}
	return fields;
}

- (nullable NSString *)ratioForWidth:(nullable NSNumber *)width height:(nullable NSNumber *)height
{
	if (width.integerValue <= 0 || height.integerValue <= 0) {
		return nil;
	}
	NSInteger divisor = NFKImageGreatestCommonDivisor(width.integerValue, height.integerValue);
	return [NSString stringWithFormat:@"%ld:%ld", (long)(width.integerValue / divisor), (long)(height.integerValue / divisor)];
}

- (nullable NSMutableURLRequest *)JSONRequestToURL:(NSURL *)url fields:(NSDictionary *)fields error:(NSError * _Nullable *)outError
{
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:fields options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *request = [self requestForURL:url];
	request.HTTPBody = payload;
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	return request;
}

// One source goes as image, several as repeated image[] files, the spelling OpenAI's edit reads.
- (NSMutableURLRequest *)multipartRequestToURL:(NSURL *)url fields:(NSDictionary *)fields sources:(NFKImageSources *)sources
{
	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSData *dashBoundary = [[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding];
	void (^appendField)(NSString *, NSString *) = ^(NSString *name, NSString *value) {
		[body appendData:dashBoundary];
		NSString *part = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, value];
		[body appendData:[part dataUsingEncoding:NSUTF8StringEncoding]];
	};
	void (^appendFile)(NSString *, NSString *, NSData *) = ^(NSString *name, NSString *filename, NSData *data) {
		[body appendData:dashBoundary];
		NSString *header = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"; filename=\"%@\"\r\nContent-Type: image/png\r\n\r\n", name, filename];
		[body appendData:[header dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:data];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	};
	for (NSString *key in fields) {
		appendField(key, [fields[key] description]);
	}
	NSString *imageField = sources.images.count > 1 ? @"image[]" : @"image";
	[sources.images enumerateObjectsUsingBlock:^(NSData *png, NSUInteger index, BOOL *stop) {
		NSString *filename = sources.images.count > 1 ? [NSString stringWithFormat:@"image%lu.png", (unsigned long)index] : @"image.png";
		appendFile(imageField, filename, png);
	}];
	if (sources.mask != nil) {
		appendFile(@"mask", @"mask.png", sources.mask);
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

	NSMutableURLRequest *request = [self requestForURL:url];
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

// The envelope is data[] of {b64_json} or {url}; every entry decodes, the first is NFKOutputImage.
- (nullable NFKInferenceResult *)resultFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
{
	NSArray *entries = [body[@"data"] isKindOfClass:NSArray.class] ? body[@"data"] : @[];
	NSMutableArray *images = [NSMutableArray array];
	for (NSDictionary *entry in entries) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		CVPixelBufferRef pixelBuffer = [self imageInEntry:entry error:outError];
		if (pixelBuffer == NULL) {
			return nil;
		}
		[images addObject:(__bridge id)pixelBuffer];
		CVPixelBufferRelease(pixelBuffer);
	}
	if (images.count == 0) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no image" error:outError];
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObjectsAndKeys:
													images.firstObject, NFKOutputImage, body, NFKRemoteBackendRawKey, nil];
	if (images.count > 1) {
		outputs[NFKOutputImages] = images;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (nullable CVPixelBufferRef)imageInEntry:(NSDictionary *)entry error:(NSError * _Nullable *)outError CF_RETURNS_RETAINED
{
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

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable, NSData * _Nullable, NSError * _Nullable))completionHandler
{
	return [NFKRemoteTransport streamRequest:request session:self.session lineHandler:lineHandler completionHandler:completionHandler];
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
