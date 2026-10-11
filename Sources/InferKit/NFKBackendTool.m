//
//  NFKBackendTool.m
//  InferKit
//

#import "NFKBackendTool.h"
#import "NFKInferenceKeys.h"
#import "NFKAudioAsset.h"
#import "NFKVideoAsset.h"
#import "NFKImageCoding.h"
#import <CoreVideo/CoreVideo.h>

@implementation NFKBackendTool

+ (instancetype)toolWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
				 outputModality:(NFKModality)outputModality
						purpose:(nullable NSString *)purpose
{
	return [self toolWithBackend:backend name:name outputModality:outputModality modelName:nil purpose:purpose];
}

+ (instancetype)toolWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
				 outputModality:(NFKModality)outputModality
					  modelName:(nullable NSString *)modelName
						purpose:(nullable NSString *)purpose
{
	NSString *named = modelName.length > 0 ? modelName : [self modelNameOfBackend:backend];
	return [[self alloc] initWithBackend:backend
									name:name
							 description:[self defaultDescriptionForOutputModality:outputModality modelName:named purpose:purpose]
							  parameters:[self defaultParametersForOutputModality:outputModality]
						  outputModality:outputModality];
}

- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
					description:(NSString *)toolDescription
					 parameters:(NSDictionary<NSString *, id> *)parameters
				 outputModality:(NFKModality)outputModality
{
	self = [super initWithName:name description:toolDescription parameters:parameters];
	if (self != nil) {
		_backend = backend;
		_outputModality = outputModality;
		_fixedParameters = @{};
	}
	return self;
}

+ (NSString *)modelNameOfBackend:(id<NFKInferenceBackend>)backend
{
	if ([backend respondsToSelector:@selector(modelName)]) {
		id modelName = [(id)backend valueForKey:@"modelName"];
		if ([modelName isKindOfClass:NSString.class] && [modelName length] > 0) {
			return modelName;
		}
	}
	return backend.backendIdentifier;
}

#pragma mark Defaults

+ (NSString *)defaultDescriptionForOutputModality:(NFKModality)outputModality
										modelName:(NSString *)modelName
										  purpose:(nullable NSString *)purpose
{
	switch (outputModality) {
		case NFKModalityAudio:
			return [NSString stringWithFormat:
				@"Generates audio with the model \"%@\" from a text prompt. Use it when %@. "
				@"For a speech model the prompt is the exact words to speak; otherwise it describes the sound or music to make. "
				@"Returns a short description of the audio produced: its length, sample rate, and channel count. "
				@"The audio itself goes to the application, which plays or saves it. You do not hear it, so do not describe its sound.",
				modelName, purpose.length > 0 ? purpose : @"the user asks to hear speech, music, or a sound effect"];
		case NFKModalityImage:
			return [NSString stringWithFormat:
				@"Generates an image with the model \"%@\" from a text description. Use it when %@. "
				@"Describe the subject, style, composition, and lighting in the prompt, and list what to leave out in negativePrompt. "
				@"Returns the image's size in pixels and shows you the image, so you can check it against the request. "
				@"The image also goes to the application, which displays or saves it.",
				modelName, purpose.length > 0 ? purpose : @"the user asks for a picture, an illustration, or another image"];
		case NFKModalityVideo:
			return [NSString stringWithFormat:
				@"Generates a short video clip with the model \"%@\" from a text description. Use it when %@. "
				@"Describe the subject, the motion, and the camera in the prompt. "
				@"Returns the clip's length and frame size. The clip goes to the application, which plays or saves it; you do not see it.",
				modelName, purpose.length > 0 ? purpose : @"the user asks for a video or an animation"];
		case NFKModalityText:
		default:
			return [NSString stringWithFormat:
				@"Sends a prompt to \"%@\", a separate language model, and returns its reply. Use it when %@. "
				@"The model has no access to this conversation. It sees only the prompt and the optional system instruction, "
				@"so include every fact and passage it needs. "
				@"Returns the model's reply as plain text.",
				modelName, purpose.length > 0 ? purpose : @"a task suits a second model, such as drafting, rewriting, summarizing, or answering from text you supply"];
	}
}

+ (NSDictionary<NSString *, id> *)defaultParametersForOutputModality:(NFKModality)outputModality
{
	NSDictionary *seed = @{ @"type": @"integer", @"description": @"Optional seed. The same seed and prompt reproduce a result." };
	switch (outputModality) {
		case NFKModalityAudio:
			return @{ @"type": @"object",
					  @"properties": @{ NFKInputPrompt: @{ @"type": @"string", @"description": @"The words to speak, or a description of the sound or music to make." },
										NFKParameterDurationSeconds: @{ @"type": @"number", @"description": @"Optional length of the audio, in seconds." } },
					  @"required": @[ NFKInputPrompt ] };
		case NFKModalityImage:
			return @{ @"type": @"object",
					  @"properties": @{ NFKInputPrompt: @{ @"type": @"string", @"description": @"What the image shows: subject, style, composition, and lighting." },
										NFKInputNegativePrompt: @{ @"type": @"string", @"description": @"Optional things the image must not contain." },
										NFKParameterWidth: @{ @"type": @"integer", @"description": @"Optional width, in pixels." },
										NFKParameterHeight: @{ @"type": @"integer", @"description": @"Optional height, in pixels." },
										NFKParameterSeed: seed },
					  @"required": @[ NFKInputPrompt ] };
		case NFKModalityVideo:
			return @{ @"type": @"object",
					  @"properties": @{ NFKInputPrompt: @{ @"type": @"string", @"description": @"What the clip shows: subject, motion, and camera." },
										NFKParameterDurationSeconds: @{ @"type": @"number", @"description": @"Optional length of the clip, in seconds." },
										NFKParameterSeed: seed },
					  @"required": @[ NFKInputPrompt ] };
		case NFKModalityText:
		default:
			return @{ @"type": @"object",
					  @"properties": @{ NFKInputPrompt: @{ @"type": @"string", @"description": @"The complete instruction or question for the model, with any text it should work from." },
										@"system": @{ @"type": @"string", @"description": @"Optional instructions that set the model's role, tone, or rules." } },
					  @"required": @[ NFKInputPrompt ] };
	}
}

#pragma mark Running

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	NFKBackendToolRequestBuilder builder = self.requestBuilder;
	NFKInferenceRequest *request = builder != nil ? builder(arguments, error) : [self defaultRequestForArguments:arguments];
	if (request == nil) {
		return nil;
	}
	if (!self.backend.isReady && !NFKInferencePrepare(self.backend, error)) {
		return nil;
	}
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:error];
	if (result == nil) {
		return nil;
	}
	NFKBackendToolResultDescriber describer = self.resultDescriber;
	return describer != nil ? describer(result) : [self defaultToolResultForResult:result];
}

+ (NSSet<NSString *> *)argumentInputKeys
{
	static NSSet<NSString *> *keys;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		keys = [NSSet setWithArray:@[ NFKInputPrompt, NFKInputNegativePrompt, NFKInputLyrics, NFKInputSuffix ]];
	});
	return keys;
}

- (NFKInferenceRequest *)defaultRequestForArguments:(NSDictionary<NSString *, id> *)arguments
{
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSString *, id> *parameters = [self.fixedParameters mutableCopy];
	NSString *system = nil;
	for (NSString *name in arguments) {
		id value = arguments[name];
		if (value == NSNull.null) {
			continue;
		}
		if ([name isEqualToString:@"system"] && [value isKindOfClass:NSString.class]) {
			system = value;
		} else if ([[self.class argumentInputKeys] containsObject:name]) {
			inputs[name] = value;
		} else {
			parameters[name] = value;
		}
	}
	NSString *prompt = [inputs[NFKInputPrompt] isKindOfClass:NSString.class] ? inputs[NFKInputPrompt] : nil;
	if (system.length > 0 && prompt != nil) {
		inputs[NFKInputMessages] = @[ @{ @"role": @"system", @"content": system }, @{ @"role": @"user", @"content": prompt } ];
	}
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:self.outputModality];
}

- (NFKToolResult *)defaultToolResultForResult:(NFKInferenceResult *)result
{
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	NSMutableArray<NSData *> *images = [NSMutableArray array];

	NSDictionary *structured = result.structured;
	if (structured != nil && [NSJSONSerialization isValidJSONObject:structured]) {
		NSData *JSON = [NSJSONSerialization dataWithJSONObject:structured options:NSJSONWritingSortedKeys error:NULL];
		[lines addObject:[[NSString alloc] initWithData:JSON encoding:NSUTF8StringEncoding]];
	} else if (result.text.length > 0) {
		[lines addObject:result.text];
	}

	id audio = [result outputForKey:NFKOutputAudio];
	if ([audio isKindOfClass:NFKAudioAsset.class]) {
		NFKAudioAsset *asset = audio;
		[lines addObject:[NSString stringWithFormat:@"Generated %.1f seconds of audio at %.0f Hz, %ld channel%@.",
						  asset.durationSeconds, asset.sampleRate, (long)asset.channelCount, asset.channelCount == 1 ? @"" : @"s"]];
	} else if (audio != nil) {
		[lines addObject:@"Generated audio."];
	}

	NSArray *generatedImages = [result outputForKey:NFKOutputImages];
	id firstImage = [result outputForKey:NFKOutputImage];
	NSArray *allImages = [generatedImages isKindOfClass:NSArray.class] && generatedImages.count > 0 ? generatedImages
		: (firstImage != nil ? @[ firstImage ] : @[]);
	for (id image in allImages) {
		CGSize size = [self pixelSizeOfImage:image];
		[lines addObject:size.width > 0 ? [NSString stringWithFormat:@"Generated a %.0f by %.0f image.", size.width, size.height]
										: @"Generated an image."];
		NSData *png = [NFKImageCoding PNGDataForImage:image];
		if (png != nil) {
			[images addObject:png];
		}
	}

	id video = [result outputForKey:NFKOutputVideo];
	if ([video isKindOfClass:NFKVideoAsset.class]) {
		NFKVideoAsset *asset = video;
		[lines addObject:[NSString stringWithFormat:@"Generated a %.1f-second clip, %.0f by %.0f.",
						  asset.durationSeconds, asset.dimensions.width, asset.dimensions.height]];
	} else if (video != nil) {
		[lines addObject:@"Generated a video clip."];
	}

	NSArray *embedding = result.embedding;
	if (embedding.count > 0 && lines.count == 0) {
		[lines addObject:[NSString stringWithFormat:@"Computed an embedding of %lu dimensions.", (unsigned long)embedding.count]];
	}
	if (lines.count == 0) {
		[lines addObject:@"The model returned no text."];
	}
	return [NFKToolResult resultWithText:[lines componentsJoinedByString:@"\n"] outputs:result.outputs imagePNGs:images];
}

- (CGSize)pixelSizeOfImage:(id)image
{
	if (CFGetTypeID((__bridge CFTypeRef)image) == CVPixelBufferGetTypeID()) {
		CVPixelBufferRef pixelBuffer = (__bridge CVPixelBufferRef)image;
		return CGSizeMake(CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer));
	}
	CGImageRef cgImage = [NFKImageCoding CGImageForImage:image];
	if (cgImage == NULL) {
		return CGSizeZero;
	}
	CGSize size = CGSizeMake(CGImageGetWidth(cgImage), CGImageGetHeight(cgImage));
	CGImageRelease(cgImage);
	return size;
}

@end
