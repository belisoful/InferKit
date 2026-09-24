//
//  NFKRemoteVideoBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteVideoBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"

/*! The contract keys this backend translates; every other parameter goes out under its own name. */
static NSSet<NSString *> *NFKRemoteVideoMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterDurationSeconds, NFKParameterWidth, NFKParameterHeight,
										NFKParameterAspectRatio, NFKParameterResolution, NFKParameterSeed,
										NFKParameterSteps, NFKParameterGuidanceScale, NFKParameterFramesPerSecond,
										NFKParameterGenerateAudio, NFKParameterOutputFormat, NFKParameterSampleCount,
										NFKParameterVideoOperation, NFKParameterSourceVideoIdentifier ]];
	});
	return mapped;
}

static NSString *NFKRemoteVideoImageDataURI(NSData *png)
{
	return [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
}

/*! Whether the bytes open with the EBML signature a WebM file starts with; a service that renders
	WebM on request says so nowhere else, and its download URL carries no extension. */
static BOOL NFKRemoteVideoIsWebM(NSData *bytes)
{
	static const uint8_t signature[] = { 0x1A, 0x45, 0xDF, 0xA3 };
	return bytes.length >= sizeof(signature) && memcmp(bytes.bytes, signature, sizeof(signature)) == 0;
}

static NSInteger NFKRemoteVideoGreatestCommonDivisor(NSInteger a, NSInteger b)
{
	while (b != 0) {
		NSInteger remainder = a % b;
		a = b;
		b = remainder;
	}
	return a;
}

@implementation NFKRemoteVideoBackend

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSNumber *style = [self defaultAPIStyleForProvider:provider];
	if (style == nil) {
		return nil;
	}
	return [self backendForProvider:provider apiStyle:style.integerValue apiKey:apiKey modelName:modelName];
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
								   apiStyle:(NFKRemoteVideoAPIStyle)apiStyle
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSURL *submitURL = [self submitURLForProvider:provider apiStyle:apiStyle];
	if (submitURL == nil) {
		return nil;
	}
	NFKRemoteVideoBackend *backend = [[self alloc] init];
	backend.apiStyle = apiStyle;
	backend.submitURL = submitURL;
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

+ (nullable NSNumber *)defaultAPIStyleForProvider:(NFKRemoteProvider *)provider
{
	NSDictionary<NSString *, NSNumber *> *styles = @{ @"openai": @(NFKRemoteVideoAPIStyleOpenAI),
													  @"gemini": @(NFKRemoteVideoAPIStyleGeminiSoraCompatible),
													  @"xai": @(NFKRemoteVideoAPIStyleXAI),
													  @"together": @(NFKRemoteVideoAPIStyleTogether),
													  @"openrouter": @(NFKRemoteVideoAPIStyleOpenRouter) };
	return styles[provider.identifier];
}

// Gemini's native API sits beside its OpenAI layer, and Together's videos are on its v2 API.
+ (nullable NSURL *)submitURLForProvider:(NFKRemoteProvider *)provider apiStyle:(NFKRemoteVideoAPIStyle)apiStyle
{
	NSString *identifier = provider.identifier;
	NSURL *versionRoot = provider.baseURL.URLByDeletingLastPathComponent;
	switch (apiStyle) {
		case NFKRemoteVideoAPIStyleOpenAI:
			return [identifier isEqualToString:@"openai"] ? [provider URLForPath:@"videos"] : nil;
		case NFKRemoteVideoAPIStyleGeminiSoraCompatible:
			return [identifier isEqualToString:@"gemini"] ? [provider URLForPath:@"videos"] : nil;
		case NFKRemoteVideoAPIStyleGeminiVeo:
			return [identifier isEqualToString:@"gemini"] ? [versionRoot URLByAppendingPathComponent:@"models"] : nil;
		case NFKRemoteVideoAPIStyleXAI:
			return [identifier isEqualToString:@"xai"] ? [provider URLForPath:@"videos/generations"] : nil;
		case NFKRemoteVideoAPIStyleTogether:
			return [identifier isEqualToString:@"together"] ? [versionRoot URLByAppendingPathComponent:@"v2/videos"] : nil;
		case NFKRemoteVideoAPIStyleOpenRouter:
			return [identifier isEqualToString:@"openrouter"] ? [provider URLForPath:@"videos"] : nil;
	}
	return nil;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 300.0;
		self.pollInterval = 5.0;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return @"remote-video";
}

#pragma mark Validation

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NSString *refusal = [self refusalForRequest:request];
	if (refusal == nil) {
		return [super submitInferenceJobForRequest:request];
	}
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	[job finishWithError:[NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
										 userInfo:@{ NSLocalizedDescriptionKey: refusal }]];
	return job;
}

// What the chosen style cannot express, said before anything is sent.
- (nullable NSString *)refusalForRequest:(NFKInferenceRequest *)request
{
	if (self.apiStyle == NFKRemoteVideoAPIStyleGeminiVeo && self.modelName.length == 0) {
		return @"Gemini's native Veo API names the model in its path, so the backend needs a model name";
	}
	NSString *operation = [self operationForRequest:request];
	if (operation == nil) {
		return nil;
	}
	if (![operation isEqualToString:NFKVideoOperationEdit] && ![operation isEqualToString:NFKVideoOperationExtend]) {
		return @"NFKParameterVideoOperation is NFKVideoOperationEdit or NFKVideoOperationExtend";
	}
	BOOL edits = [operation isEqualToString:NFKVideoOperationEdit];
	NSString *identifier = [self sourceIdentifierForRequest:request];
	NSString *sourceURL = [self sourceVideoURLStringForRequest:request allowingInlineData:NO];
	switch (self.apiStyle) {
		case NFKRemoteVideoAPIStyleOpenAI:
		case NFKRemoteVideoAPIStyleOpenRouter:
			return identifier != nil ? nil : @"this service edits and extends a clip it generated, named by NFKParameterSourceVideoIdentifier";
		case NFKRemoteVideoAPIStyleGeminiSoraCompatible:
			if (edits) {
				return @"Veo extends a clip but does not edit one";
			}
			return identifier != nil ? nil : @"Veo extends a clip it generated, named by NFKParameterSourceVideoIdentifier";
		case NFKRemoteVideoAPIStyleGeminiVeo:
			if (edits) {
				return @"Veo extends a clip but does not edit one";
			}
			return identifier != nil || sourceURL != nil ? nil : @"Veo extends a clip it generated, named by its uri under NFKParameterSourceVideoIdentifier";
		case NFKRemoteVideoAPIStyleXAI:
			return identifier != nil || [self sourceVideoURLStringForRequest:request allowingInlineData:YES] != nil
				? nil : @"xAI edits and extends the clip under NFKInputVideo";
		case NFKRemoteVideoAPIStyleTogether:
			return sourceURL != nil ? nil : @"Together reads the source clip from a URL: NFKInputVideo must be a hosted clip";
	}
	return nil;
}

#pragma mark Request inputs

- (nullable NSString *)operationForRequest:(NFKInferenceRequest *)request
{
	id operation = request.parameters[NFKParameterVideoOperation];
	return [operation isKindOfClass:NSString.class] && [operation length] > 0 ? operation : nil;
}

- (nullable NSString *)sourceIdentifierForRequest:(NFKInferenceRequest *)request
{
	id identifier = request.parameters[NFKParameterSourceVideoIdentifier];
	return [identifier isKindOfClass:NSString.class] && [identifier length] > 0 ? identifier : nil;
}

// A hosted clip goes by its URL; a local file goes inline as a data URI where the service reads one.
- (nullable NSString *)sourceVideoURLStringForRequest:(NFKInferenceRequest *)request allowingInlineData:(BOOL)allowingInlineData
{
	NFKVideoAsset *video = [request inputForKey:NFKInputVideo];
	NSURL *url = [video isKindOfClass:NFKVideoAsset.class] ? video.fileURL : nil;
	if (url == nil) {
		return nil;
	}
	if (!url.isFileURL) {
		return url.absoluteString;
	}
	if (!allowingInlineData) {
		return nil;
	}
	NSData *bytes = [NSData dataWithContentsOfURL:url];
	return bytes != nil ? [@"data:video/mp4;base64," stringByAppendingString:[bytes base64EncodedStringWithOptions:0]] : nil;
}

- (nullable NSData *)PNGForInputKey:(NSString *)key request:(NFKInferenceRequest *)request
{
	id image = [request inputForKey:key];
	return image != nil ? [NFKImageCoding PNGDataForImage:image] : nil;
}

- (NSArray<NSData *> *)referencePNGsForRequest:(NFKInferenceRequest *)request
{
	NSArray *images = [request inputForKey:NFKInputImages];
	NSMutableArray<NSData *> *pngs = [NSMutableArray array];
	if (![images isKindOfClass:NSArray.class]) {
		return pngs;
	}
	for (id image in images) {
		NSData *png = [NFKImageCoding PNGDataForImage:image];
		if (png != nil) {
			[pngs addObject:png];
		}
	}
	return pngs;
}

- (nullable NSNumber *)numberForKey:(NSString *)key request:(NFKInferenceRequest *)request
{
	id value = request.parameters[key];
	return [value isKindOfClass:NSNumber.class] ? value : nil;
}

- (nullable NSString *)stringForKey:(NSString *)key request:(NFKInferenceRequest *)request
{
	id value = request.parameters[key];
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

- (nullable NSString *)sizeForRequest:(NFKInferenceRequest *)request
{
	NSNumber *width = [self numberForKey:NFKParameterWidth request:request];
	NSNumber *height = [self numberForKey:NFKParameterHeight request:request];
	if (width == nil || height == nil) {
		return nil;
	}
	return [NSString stringWithFormat:@"%ldx%ld", (long)width.integerValue, (long)height.integerValue];
}

// The ratio the caller named, or the one the size reduces to.
- (nullable NSString *)aspectRatioForRequest:(NFKInferenceRequest *)request
{
	NSString *named = [self stringForKey:NFKParameterAspectRatio request:request];
	if (named != nil) {
		return named;
	}
	NSInteger width = [self numberForKey:NFKParameterWidth request:request].integerValue;
	NSInteger height = [self numberForKey:NFKParameterHeight request:request].integerValue;
	if (width <= 0 || height <= 0) {
		return nil;
	}
	NSInteger divisor = NFKRemoteVideoGreatestCommonDivisor(width, height);
	return [NSString stringWithFormat:@"%ld:%ld", (long)(width / divisor), (long)(height / divisor)];
}

// The tier the caller named, or the one the size's short side names.
- (nullable NSString *)resolutionForRequest:(NFKInferenceRequest *)request
{
	NSString *named = [self stringForKey:NFKParameterResolution request:request];
	if (named != nil) {
		return named;
	}
	NSInteger width = [self numberForKey:NFKParameterWidth request:request].integerValue;
	NSInteger height = [self numberForKey:NFKParameterHeight request:request].integerValue;
	NSInteger shortSide = MIN(width, height);
	if (shortSide <= 0) {
		return nil;
	}
	return shortSide >= 2160 ? @"4k" : [NSString stringWithFormat:@"%ldp", (long)shortSide];
}

- (void)addUnmappedParametersOfRequest:(NFKInferenceRequest *)request to:(NSMutableDictionary<NSString *, id> *)fields
{
	NSSet<NSString *> *mapped = NFKRemoteVideoMappedParameters();
	for (NSString *key in request.parameters) {
		if (![mapped containsObject:key]) {
			fields[key] = request.parameters[key];
		}
	}
}

#pragma mark Submit bodies

- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request
{
	switch (self.apiStyle) {
		case NFKRemoteVideoAPIStyleOpenAI:
			return [self openAIBodyForRequest:request];
		case NFKRemoteVideoAPIStyleGeminiSoraCompatible:
			return [self geminiSoraFieldsForRequest:request];
		case NFKRemoteVideoAPIStyleGeminiVeo:
			return [self geminiVeoBodyForRequest:request];
		case NFKRemoteVideoAPIStyleXAI:
			return [self xAIBodyForRequest:request];
		case NFKRemoteVideoAPIStyleTogether:
			return [self togetherBodyForRequest:request];
		case NFKRemoteVideoAPIStyleOpenRouter:
			return [self openRouterBodyForRequest:request];
	}
	return @{};
}

- (NSMutableDictionary<NSString *, id> *)openAIBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"prompt"] = request.prompt ?: @"";
	NSNumber *seconds = [self numberForKey:NFKParameterDurationSeconds request:request];
	if (seconds != nil) {
		body[@"seconds"] = [NSString stringWithFormat:@"%ld", (long)seconds.integerValue];
	}
	NSString *size = [self sizeForRequest:request];
	if (size != nil) {
		body[@"size"] = size;
	}
	NSString *identifier = [self sourceIdentifierForRequest:request];
	if ([self operationForRequest:request] != nil && identifier != nil) {
		body[@"video"] = @{ @"id": identifier };
	}
	[self addUnmappedParametersOfRequest:request to:body];
	return body;
}

// The Sora-compatible layer takes model and prompt as fields and every Veo option beside them as
// further form fields, which is where the OpenAI SDKs put extra_body on a multipart request.
- (NSMutableDictionary<NSString *, id> *)geminiSoraFieldsForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *fields = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		fields[@"model"] = self.modelName;
	}
	fields[@"prompt"] = request.prompt ?: @"";
	NSNumber *seconds = [self numberForKey:NFKParameterDurationSeconds request:request];
	if (seconds != nil) {
		fields[@"duration_seconds"] = @(seconds.integerValue);
	}
	fields[@"size"] = [self sizeForRequest:request];
	fields[@"aspect_ratio"] = [self stringForKey:NFKParameterAspectRatio request:request];
	fields[@"resolution"] = [self stringForKey:NFKParameterResolution request:request];
	NSNumber *fps = [self numberForKey:NFKParameterFramesPerSecond request:request];
	if (fps != nil) {
		fields[@"frame_rate"] = fps.stringValue;
	}
	fields[@"negative_prompt"] = [request inputForKey:NFKInputNegativePrompt];
	fields[@"seed"] = [self numberForKey:NFKParameterSeed request:request];
	NSData *first = [self PNGForInputKey:NFKInputImage request:request];
	if (first != nil) {
		fields[@"image"] = [first base64EncodedStringWithOptions:0];
	}
	NSData *last = [self PNGForInputKey:NFKInputLastFrame request:request];
	if (last != nil) {
		fields[@"last_frame"] = [last base64EncodedStringWithOptions:0];
	}
	NSMutableArray<NSString *> *references = [NSMutableArray array];
	for (NSData *png in [self referencePNGsForRequest:request]) {
		[references addObject:[png base64EncodedStringWithOptions:0]];
	}
	if (references.count > 0) {
		fields[@"reference_images"] = references;
	}
	if ([[self operationForRequest:request] isEqualToString:NFKVideoOperationExtend]) {
		fields[@"extend_video_id"] = [self sourceIdentifierForRequest:request];
	}
	[self addUnmappedParametersOfRequest:request to:fields];
	return fields;
}

// The native API splits the request into what to generate from (an instance) and how (parameters).
- (NSDictionary<NSString *, id> *)geminiVeoBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *instance = [NSMutableDictionary dictionary];
	instance[@"prompt"] = request.prompt ?: @"";
	NSData *first = [self PNGForInputKey:NFKInputImage request:request];
	if (first != nil) {
		instance[@"image"] = [self veoInlineImage:first];
	}
	NSData *last = [self PNGForInputKey:NFKInputLastFrame request:request];
	if (last != nil) {
		instance[@"lastFrame"] = [self veoInlineImage:last];
	}
	NSMutableArray *references = [NSMutableArray array];
	for (NSData *png in [self referencePNGsForRequest:request]) {
		[references addObject:@{ @"image": [self veoInlineImage:png], @"referenceType": @"asset" }];
	}
	if (references.count > 0) {
		instance[@"referenceImages"] = references;
	}
	NSString *sourceURI = [self sourceIdentifierForRequest:request] ?: [self sourceVideoURLStringForRequest:request allowingInlineData:NO];
	if ([[self operationForRequest:request] isEqualToString:NFKVideoOperationExtend] && sourceURI != nil) {
		instance[@"video"] = @{ @"uri": sourceURI };
	}

	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	parameters[@"durationSeconds"] = [self numberForKey:NFKParameterDurationSeconds request:request];
	parameters[@"aspectRatio"] = [self aspectRatioForRequest:request];
	parameters[@"resolution"] = [self resolutionForRequest:request];
	parameters[@"numberOfVideos"] = [self numberForKey:NFKParameterSampleCount request:request];
	parameters[@"negativePrompt"] = [request inputForKey:NFKInputNegativePrompt];
	parameters[@"seed"] = [self numberForKey:NFKParameterSeed request:request];
	[self addUnmappedParametersOfRequest:request to:parameters];
	return @{ @"instances": @[ instance ], @"parameters": parameters };
}

- (NSDictionary *)veoInlineImage:(NSData *)png
{
	return @{ @"inlineData": @{ @"mimeType": @"image/png", @"data": [png base64EncodedStringWithOptions:0] } };
}

- (NSDictionary<NSString *, id> *)xAIBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"prompt"] = request.prompt ?: @"";
	body[@"duration"] = [self numberForKey:NFKParameterDurationSeconds request:request];
	body[@"aspect_ratio"] = [self aspectRatioForRequest:request];
	body[@"resolution"] = [self resolutionForRequest:request];
	body[@"generate_audio"] = [self numberForKey:NFKParameterGenerateAudio request:request];
	NSData *first = [self PNGForInputKey:NFKInputImage request:request];
	if (first != nil) {
		body[@"image"] = @{ @"url": NFKRemoteVideoImageDataURI(first) };
	}
	NSData *last = [self PNGForInputKey:NFKInputLastFrame request:request];
	if (last != nil) {
		body[@"last_frame"] = @{ @"url": NFKRemoteVideoImageDataURI(last) };
	}
	NSMutableArray *references = [NSMutableArray array];
	for (NSData *png in [self referencePNGsForRequest:request]) {
		[references addObject:@{ @"url": NFKRemoteVideoImageDataURI(png) }];
	}
	if (references.count > 0) {
		body[@"reference_images"] = references;
	}
	if ([self operationForRequest:request] != nil) {
		NSString *identifier = [self sourceIdentifierForRequest:request];
		NSString *sourceURL = identifier == nil ? [self sourceVideoURLStringForRequest:request allowingInlineData:YES] : nil;
		if (identifier != nil) {
			body[@"video"] = @{ @"file_id": identifier };
		} else if (sourceURL != nil) {
			body[@"video"] = @{ @"url": sourceURL };
		}
		if ([[self operationForRequest:request] isEqualToString:NFKVideoOperationEdit]) {
			[body removeObjectForKey:@"duration"];
		}
	}
	[self addUnmappedParametersOfRequest:request to:body];
	return body;
}

// Together reads keyframes as raw base64 under media, and a source clip only as a hosted URL.
- (NSDictionary<NSString *, id> *)togetherBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"prompt"] = request.prompt ?: @"";
	body[@"width"] = [self numberForKey:NFKParameterWidth request:request];
	body[@"height"] = [self numberForKey:NFKParameterHeight request:request];
	NSNumber *seconds = [self numberForKey:NFKParameterDurationSeconds request:request];
	if (seconds != nil) {
		body[@"seconds"] = [NSString stringWithFormat:@"%ld", (long)seconds.integerValue];
	}
	body[@"fps"] = [self numberForKey:NFKParameterFramesPerSecond request:request];
	body[@"steps"] = [self numberForKey:NFKParameterSteps request:request];
	body[@"seed"] = [self numberForKey:NFKParameterSeed request:request];
	body[@"guidance_scale"] = [self numberForKey:NFKParameterGuidanceScale request:request];
	body[@"output_format"] = [self stringForKey:NFKParameterOutputFormat request:request].uppercaseString;
	body[@"negative_prompt"] = [request inputForKey:NFKInputNegativePrompt];
	body[@"generate_audio"] = [self numberForKey:NFKParameterGenerateAudio request:request];
	body[@"resolution"] = [self stringForKey:NFKParameterResolution request:request].uppercaseString;
	body[@"ratio"] = [self stringForKey:NFKParameterAspectRatio request:request];

	NSMutableDictionary<NSString *, id> *media = [NSMutableDictionary dictionary];
	NSMutableArray *frames = [NSMutableArray array];
	NSData *first = [self PNGForInputKey:NFKInputImage request:request];
	if (first != nil) {
		[frames addObject:@{ @"input_image": [first base64EncodedStringWithOptions:0], @"frame": @"first" }];
	}
	NSData *last = [self PNGForInputKey:NFKInputLastFrame request:request];
	if (last != nil) {
		[frames addObject:@{ @"input_image": [last base64EncodedStringWithOptions:0], @"frame": @"last" }];
	}
	if (frames.count > 0) {
		media[@"frame_images"] = frames;
	}
	NSMutableArray<NSString *> *references = [NSMutableArray array];
	for (NSData *png in [self referencePNGsForRequest:request]) {
		[references addObject:[png base64EncodedStringWithOptions:0]];
	}
	if (references.count > 0) {
		media[@"reference_images"] = references;
	}
	NSString *operation = [self operationForRequest:request];
	NSString *source = [self sourceVideoURLStringForRequest:request allowingInlineData:NO];
	if ([operation isEqualToString:NFKVideoOperationEdit]) {
		media[@"source_video"] = source;
	} else if ([operation isEqualToString:NFKVideoOperationExtend] && source != nil) {
		media[@"frame_videos"] = @[ @{ @"video": source } ];
	}
	if (media.count > 0) {
		body[@"media"] = media;
	}
	[self addUnmappedParametersOfRequest:request to:body];
	return body;
}

- (NSDictionary<NSString *, id> *)openRouterBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"prompt"] = request.prompt ?: @"";
	body[@"duration"] = [self numberForKey:NFKParameterDurationSeconds request:request];
	body[@"size"] = [self sizeForRequest:request];
	body[@"aspect_ratio"] = [self stringForKey:NFKParameterAspectRatio request:request];
	body[@"resolution"] = [self stringForKey:NFKParameterResolution request:request];
	body[@"seed"] = [self numberForKey:NFKParameterSeed request:request];
	body[@"generate_audio"] = [self numberForKey:NFKParameterGenerateAudio request:request];
	NSMutableArray *frames = [NSMutableArray array];
	NSData *first = [self PNGForInputKey:NFKInputImage request:request];
	if (first != nil) {
		[frames addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": NFKRemoteVideoImageDataURI(first) },
							 @"frame_type": @"first_frame" }];
	}
	NSData *last = [self PNGForInputKey:NFKInputLastFrame request:request];
	if (last != nil) {
		[frames addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": NFKRemoteVideoImageDataURI(last) },
							 @"frame_type": @"last_frame" }];
	}
	if (frames.count > 0) {
		body[@"frame_images"] = frames;
	}
	NSMutableArray *references = [NSMutableArray array];
	for (NSData *png in [self referencePNGsForRequest:request]) {
		[references addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": NFKRemoteVideoImageDataURI(png) } }];
	}
	if (references.count > 0) {
		body[@"input_references"] = references;
	}
	if ([self operationForRequest:request] != nil) {
		body[@"previous_job_id"] = [self sourceIdentifierForRequest:request];
	}
	[self addUnmappedParametersOfRequest:request to:body];
	return body;
}

#pragma mark Submit requests

- (NSURLRequest *)submitRequestForRequest:(NFKInferenceRequest *)request
{
	switch (self.apiStyle) {
		case NFKRemoteVideoAPIStyleOpenAI:
			return [self openAISubmitRequestForRequest:request];
		case NFKRemoteVideoAPIStyleGeminiSoraCompatible:
			return [self multipartRequestToURL:self.submitURL fields:[self geminiSoraFieldsForRequest:request] file:nil];
		case NFKRemoteVideoAPIStyleGeminiVeo: {
			NSString *action = [self.modelName stringByAppendingString:@":predictLongRunning"];
			return [self JSONRequestToURL:[self.submitURL URLByAppendingPathComponent:action]
									 body:[self geminiVeoBodyForRequest:request]];
		}
		case NFKRemoteVideoAPIStyleXAI:
			return [self JSONRequestToURL:[self xAIURLForOperation:[self operationForRequest:request]]
									 body:[self xAIBodyForRequest:request]];
		case NFKRemoteVideoAPIStyleTogether:
		case NFKRemoteVideoAPIStyleOpenRouter:
			return [self JSONRequestToURL:self.submitURL body:[self submitBodyForRequest:request]];
	}
	return [self JSONRequestToURL:self.submitURL body:[self submitBodyForRequest:request]];
}

// An edit or an extension has its own path beside the create path; a reference image makes the
// create multipart, the image as a PNG file beside the fields.
- (NSURLRequest *)openAISubmitRequestForRequest:(NFKInferenceRequest *)request
{
	NSString *operation = [self operationForRequest:request];
	if ([operation isEqualToString:NFKVideoOperationEdit]) {
		return [self JSONRequestToURL:[self.submitURL URLByAppendingPathComponent:@"edits"] body:[self openAIBodyForRequest:request]];
	}
	if ([operation isEqualToString:NFKVideoOperationExtend]) {
		return [self JSONRequestToURL:[self.submitURL URLByAppendingPathComponent:@"extensions"] body:[self openAIBodyForRequest:request]];
	}
	NSData *png = [self PNGForInputKey:NFKInputImage request:request];
	if (png == nil) {
		return [self JSONRequestToURL:self.submitURL body:[self openAIBodyForRequest:request]];
	}
	return [self multipartRequestToURL:self.submitURL fields:[self openAIBodyForRequest:request] file:png];
}

- (NSURL *)xAIURLForOperation:(nullable NSString *)operation
{
	NSURL *videos = self.submitURL.URLByDeletingLastPathComponent;
	if ([operation isEqualToString:NFKVideoOperationEdit]) {
		return [videos URLByAppendingPathComponent:@"edits"];
	}
	if ([operation isEqualToString:NFKVideoOperationExtend]) {
		return [videos URLByAppendingPathComponent:@"extensions"];
	}
	return self.submitURL;
}

- (NSURLRequest *)JSONRequestToURL:(NSURL *)url body:(NSDictionary<NSString *, id> *)body
{
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[self authorize:urlRequest];
	return urlRequest;
}

// A list goes out as repeated key[] fields, the way the OpenAI SDKs serialize one into a form.
- (NSURLRequest *)multipartRequestToURL:(NSURL *)url fields:(NSDictionary<NSString *, id> *)fields file:(nullable NSData *)png
{
	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSData *dashBoundary = [[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding];
	void (^appendField)(NSString *, NSString *) = ^(NSString *name, NSString *value) {
		[body appendData:dashBoundary];
		NSString *part = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, value];
		[body appendData:[part dataUsingEncoding:NSUTF8StringEncoding]];
	};
	for (NSString *key in [fields.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		id value = fields[key];
		if ([value isKindOfClass:NSArray.class]) {
			for (id element in value) {
				appendField([key stringByAppendingString:@"[]"], [element description]);
			}
			continue;
		}
		appendField(key, [value description]);
	}
	if (png != nil) {
		[body appendData:dashBoundary];
		[body appendData:[@"Content-Disposition: form-data; name=\"input_reference\"; filename=\"reference.png\"\r\nContent-Type: image/png\r\n\r\n"
						  dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:png];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = body;
	[urlRequest setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	[self authorize:urlRequest];
	return urlRequest;
}

// Gemini's native API reads the key from its own header; every other style takes a bearer token.
- (void)authorize:(NSMutableURLRequest *)request
{
	if (self.apiKey.length == 0) {
		return;
	}
	if (self.apiStyle == NFKRemoteVideoAPIStyleGeminiVeo) {
		[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
		return;
	}
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
}

#pragma mark Status

- (NSURLRequest *)statusRequestForURL:(NSURL *)url
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[self authorize:request];
	return request;
}

- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response
{
	NSString *key = self.apiStyle == NFKRemoteVideoAPIStyleXAI ? @"request_id"
				  : self.apiStyle == NFKRemoteVideoAPIStyleGeminiVeo ? @"name" : @"id";
	id identifier = response[key];
	return [identifier isKindOfClass:NSString.class] ? identifier : nil;
}

// xAI polls beside its generations path; a Veo operation's name is its path under the API version.
- (nullable NSURL *)statusURLForJobIdentifier:(NSString *)jobIdentifier
{
	switch (self.apiStyle) {
		case NFKRemoteVideoAPIStyleXAI:
			return [self.submitURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:jobIdentifier];
		case NFKRemoteVideoAPIStyleGeminiVeo:
			return [self.submitURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:jobIdentifier];
		default:
			return [self.submitURL URLByAppendingPathComponent:jobIdentifier];
	}
}

// The services that report progress report it as a percentage.
- (double)progressFromStatusResponse:(NSDictionary *)response
{
	id progress = response[@"progress"];
	if (![progress isKindOfClass:NSNumber.class]) {
		return -1.0;
	}
	double value = [progress doubleValue];
	return value > 1.0 ? value / 100.0 : value;
}

- (nullable NSString *)statusInResponse:(NSDictionary *)response
{
	id status = response[@"status"];
	return [status isKindOfClass:NSString.class] ? status : nil;
}

- (BOOL)isSucceededStatusResponse:(NSDictionary *)response
{
	if (self.apiStyle == NFKRemoteVideoAPIStyleGeminiVeo) {
		return [response[@"done"] boolValue] && response[@"error"] == nil;
	}
	NSString *status = [self statusInResponse:response];
	return [status isEqualToString:@"completed"] || [status isEqualToString:@"succeeded"] || [status isEqualToString:@"done"];
}

- (BOOL)isFailedStatusResponse:(NSDictionary *)response
{
	if (self.apiStyle == NFKRemoteVideoAPIStyleGeminiVeo) {
		return response[@"error"] != nil;
	}
	NSString *status = [self statusInResponse:response];
	return [@[ @"failed", @"error", @"cancelled", @"expired" ] containsObject:status ?: @""];
}

#pragma mark Result

- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response
											 outputModality:(NFKModality)outputModality
{
	NSArray<NSURL *> *clipURLs = [self clipURLsInStatusResponse:response];
	NSMutableArray<NFKVideoAsset *> *clips = [NSMutableArray array];
	for (NSURL *url in clipURLs) {
		NFKVideoAsset *clip = [self downloadClipAtURL:url];
		if (clip == nil) {
			return nil;
		}
		[clips addObject:clip];
	}
	if (clips.count == 0) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:clips.firstObject forKey:NFKOutputVideo];
	if (clips.count > 1) {
		outputs[NFKOutputVideos] = clips;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// Where each style leaves the finished clips: a content path, a url on the job, or a list.
- (NSArray<NSURL *> *)clipURLsInStatusResponse:(NSDictionary *)response
{
	NSMutableArray<NSString *> *locations = [NSMutableArray array];
	switch (self.apiStyle) {
		case NFKRemoteVideoAPIStyleOpenAI: {
			NSString *identifier = [self jobIdentifierFromResponse:response];
			if (identifier.length > 0) {
				NSURL *content = [[self.submitURL URLByAppendingPathComponent:identifier] URLByAppendingPathComponent:@"content"];
				return @[ content ];
			}
			break;
		}
		case NFKRemoteVideoAPIStyleGeminiSoraCompatible:
			[self addString:response[@"url"] to:locations];
			[self addString:[self dictionary:response[@"video"]][@"url"] to:locations];
			break;
		case NFKRemoteVideoAPIStyleGeminiVeo: {
			NSDictionary *generated = [self dictionary:[self dictionary:response[@"response"]][@"generateVideoResponse"]];
			NSArray *samples = [generated[@"generatedSamples"] isKindOfClass:NSArray.class] ? generated[@"generatedSamples"] : @[];
			for (id sample in samples) {
				[self addString:[self dictionary:[self dictionary:sample][@"video"]][@"uri"] to:locations];
			}
			break;
		}
		case NFKRemoteVideoAPIStyleXAI:
			[self addString:[self dictionary:response[@"video"]][@"url"] to:locations];
			break;
		case NFKRemoteVideoAPIStyleTogether:
			[self addString:[self dictionary:response[@"outputs"]][@"video_url"] to:locations];
			break;
		case NFKRemoteVideoAPIStyleOpenRouter: {
			NSArray *urls = [response[@"unsigned_urls"] isKindOfClass:NSArray.class] ? response[@"unsigned_urls"] : @[];
			for (id url in urls) {
				[self addString:url to:locations];
			}
			break;
		}
	}
	NSMutableArray<NSURL *> *urls = [NSMutableArray array];
	for (NSString *location in locations) {
		NSURL *url = [NSURL URLWithString:location relativeToURL:self.submitURL].absoluteURL;
		if (url != nil) {
			[urls addObject:url];
		}
	}
	return urls;
}

- (NSDictionary *)dictionary:(id)value
{
	return [value isKindOfClass:NSDictionary.class] ? value : @{};
}

- (void)addString:(id)value to:(NSMutableArray<NSString *> *)strings
{
	if ([value isKindOfClass:NSString.class] && [value length] > 0) {
		[strings addObject:value];
	}
}

// The key goes only to the service's own host: a clip on a storage host is fetched without it.
- (nullable NFKVideoAsset *)downloadClipAtURL:(NSURL *)url
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	if ([url.host isEqualToString:self.submitURL.host]) {
		[self authorize:request];
	}
	NSHTTPURLResponse *httpResponse = nil;
	NSData *bytes = [self sendRequest:request response:&httpResponse error:NULL];
	if (bytes == nil || [NFKRemoteTransport errorForResponse:httpResponse data:bytes] != nil) {
		return nil;
	}
	NSString *extension = NFKRemoteVideoIsWebM(bytes) ? @"webm" : @"mp4";
	NSURL *fileURL = NFKRemoteWriteMediaFile(bytes, @"video", extension, self.outputDirectoryURL, NULL);
	return fileURL != nil ? [NFKVideoAsset videoAssetWithFileURL:fileURL] : nil;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

@end
