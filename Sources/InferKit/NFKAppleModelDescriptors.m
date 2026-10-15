//
//  NFKAppleModelDescriptors.m
//  InferKit
//

#import "NFKAppleModelDescriptors.h"
#import "NFKModelBuilding.h"
#import "NFKModelDescriptor.h"
#import "NFKModelVocabulary.h"
#import "NFKVisionTextBackend.h"
#import "NFKVisionSegmentationBackend.h"
#import "NFKVisionPoseBackend.h"
#import "NFKVisionFaceBackend.h"
#import "NFKVisionFeaturePrintBackend.h"
#import "NFKVisionClassificationBackend.h"
#import "NFKVisionAnimalBackend.h"
#import "NFKVisionRectangleBackend.h"
#import "NFKVisionMeasurementBackend.h"
#import "NFKVisionContourBackend.h"
#import "NFKVisionRegistrationBackend.h"
#import "NFKVisionTrackingBackend.h"
#import "NFKVideoToolboxBackend.h"
#import "NFKSpeechRecognitionBackend.h"
#import "NFKSpeechSynthesisBackend.h"
#import "NFKSoundClassificationBackend.h"
#import "NFKTextEmbeddingBackend.h"

/*! One engine the registry builds and the catalog describes. */
@interface NFKAppleEngine : NSObject
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, copy) NSString *displayName;
@property (nonatomic, copy) NSString *family;
@property (nonatomic, copy) NSString *summary;
@property (nonatomic, copy) NSArray<NSString *> *tasks;
@property (nonatomic, copy, nullable) NSString *providerClassName;
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *minimumOSVersions;
@property (nonatomic, assign) BOOL requiresAppleSilicon;
@property (nonatomic, copy) id<NFKInferenceBackend> (^builder)(void);
@end

@implementation NFKAppleEngine
@end

@implementation NFKAppleModelDescriptors

#pragma mark Engines

/*! The core's own floor, where every engine below it runs. */
+ (NSDictionary<NSString *, NSString *> *)coreFloor
{
	return @{ @"macOS": @"11.0", @"iOS": @"14.0", @"tvOS": @"14.0" };
}

+ (NFKAppleEngine *)engineWithIdentifier:(NSString *)identifier
							 displayName:(NSString *)displayName
								  family:(NSString *)family
								 summary:(NSString *)summary
								   tasks:(NSArray<NSString *> *)tasks
						providerClassName:(nullable NSString *)providerClassName
					   minimumOSVersions:(NSDictionary<NSString *, NSString *> *)minimumOSVersions
								 builder:(id<NFKInferenceBackend> (^)(void))builder
{
	NFKAppleEngine *engine = [[NFKAppleEngine alloc] init];
	engine.identifier = identifier;
	engine.displayName = displayName;
	engine.family = family;
	engine.summary = summary;
	engine.tasks = tasks;
	engine.providerClassName = providerClassName;
	engine.minimumOSVersions = minimumOSVersions;
	engine.builder = builder;
	return engine;
}

+ (NSArray<NFKAppleEngine *> *)visionEngines
{
	NSDictionary<NSString *, NSString *> *floor = [self coreFloor];
	return @[
		[self engineWithIdentifier:@"vision-text" displayName:@"Vision Text Recognition" family:@"apple-vision"
						   summary:@"Reads the lines of text in an image, with a box around each line."
							 tasks:@[ NFKModelTaskTextRecognition ] providerClassName:@"NFKVisionTextProvider"
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionTextBackend backend];
		}],
		[self engineWithIdentifier:@"vision-segmentation" displayName:@"Vision Segmentation" family:@"apple-vision"
						   summary:@"Masks the subject, the people, or the salient region of an image."
							 tasks:@[ NFKModelTaskSegmentation ] providerClassName:@"NFKVisionSegmentationProvider"
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionSegmentationBackend backend];
		}],
		[self engineWithIdentifier:@"vision-pose" displayName:@"Vision Pose" family:@"apple-vision"
						   summary:@"Finds the body, hand, or animal joints in an image."
							 tasks:@[ NFKModelTaskPoseEstimation ] providerClassName:@"NFKVisionPoseProvider"
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionPoseBackend backend];
		}],
		[self engineWithIdentifier:@"vision-face" displayName:@"Vision Faces" family:@"apple-vision"
						   summary:@"Finds the faces in an image and their landmarks."
							 tasks:@[ NFKModelTaskFaceDetection ] providerClassName:@"NFKVisionFaceProvider"
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionFaceBackend backend];
		}],
		[self engineWithIdentifier:@"vision-feature-print" displayName:@"Vision Feature Print" family:@"apple-vision"
						   summary:@"Embeds an image as a vector for comparing images."
							 tasks:@[ NFKModelTaskImageEmbedding ] providerClassName:@"NFKVisionFeaturePrintProvider"
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionFeaturePrintBackend backend];
		}],
		[self engineWithIdentifier:@"vision-classification" displayName:@"Vision Classification" family:@"apple-vision"
						   summary:@"Labels an image from Apple's built-in taxonomy."
							 tasks:@[ NFKModelTaskClassification ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionClassificationBackend backend];
		}],
		[self engineWithIdentifier:@"vision-animal" displayName:@"Vision Animals" family:@"apple-vision"
						   summary:@"Finds the cats and dogs in an image."
							 tasks:@[ NFKModelTaskObjectDetection ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionAnimalBackend backend];
		}],
		[self engineWithIdentifier:@"vision-rectangle" displayName:@"Vision Rectangles" family:@"apple-vision"
						   summary:@"Finds rectangles, document pages, and barcodes in an image, each with its four corners."
							 tasks:@[ NFKModelTaskObjectDetection ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionRectangleBackend backend];
		}],
		[self engineWithIdentifier:@"vision-measurement" displayName:@"Vision Measurement" family:@"apple-vision"
						   summary:@"Scores an image's composition or measures the angle of its horizon."
							 tasks:@[ NFKModelTaskClassification ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionMeasurementBackend backend];
		}],
		[self engineWithIdentifier:@"vision-contour" displayName:@"Vision Contours" family:@"apple-vision"
						   summary:@"Traces the outlines of the shapes in an image."
							 tasks:@[ NFKModelTaskSegmentation ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionContourBackend backend];
		}],
		[self engineWithIdentifier:@"vision-registration" displayName:@"Vision Registration" family:@"apple-vision"
						   summary:@"Measures the shift or the homography that aligns one image with another."
							 tasks:@[ NFKModelTaskOpticalFlow ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionRegistrationBackend backend];
		}],
		[self engineWithIdentifier:@"vision-tracking" displayName:@"Vision Object Tracking" family:@"apple-vision"
						   summary:@"Follows a region the caller names from frame to frame."
							 tasks:@[ NFKModelTaskObjectDetection ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionTrackingBackend backendWithKind:NFKVisionTrackingKindObject];
		}],
		[self engineWithIdentifier:@"vision-tracking-trajectory" displayName:@"Vision Trajectories"
							family:@"apple-vision"
						   summary:@"Finds objects moving on a parabola across frames."
							 tasks:@[ NFKModelTaskObjectDetection ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKVisionTrackingBackend backendWithKind:NFKVisionTrackingKindTrajectory];
		}],
	];
}

// The processors postdate the core's floor, and only interpolation exists on tvOS.
+ (NSArray<NFKAppleEngine *> *)videoToolboxEngines
{
	NSArray<NFKAppleEngine *> *engines = @[
		[self engineWithIdentifier:@"videotoolbox-super-resolution" displayName:@"VideoToolbox Super Resolution"
							family:@"apple-videotoolbox"
						   summary:@"Upscales a frame with the system's super-resolution model."
							 tasks:@[ NFKModelTaskUpscaling ] providerClassName:@"NFKVideoToolboxUpscalingProvider"
				 minimumOSVersions:@{ @"macOS": @"26.0", @"iOS": @"26.0" } builder:^id<NFKInferenceBackend> {
			return [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskSuperResolution];
		}],
		[self engineWithIdentifier:@"videotoolbox-frame-interpolation" displayName:@"VideoToolbox Frame Interpolation"
							family:@"apple-videotoolbox"
						   summary:@"Synthesizes the frame between two frames."
							 tasks:@[ NFKModelTaskFrameInterpolation ] providerClassName:nil
				 minimumOSVersions:@{ @"macOS": @"26.0", @"iOS": @"26.0", @"tvOS": @"26.0" }
						   builder:^id<NFKInferenceBackend> {
			return [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
		}],
		[self engineWithIdentifier:@"videotoolbox-optical-flow" displayName:@"VideoToolbox Optical Flow"
							family:@"apple-videotoolbox"
						   summary:@"Estimates the motion between two frames as a packed flow map."
							 tasks:@[ NFKModelTaskOpticalFlow ] providerClassName:@"NFKVideoToolboxOpticalFlowProvider"
				 minimumOSVersions:@{ @"macOS": @"15.4", @"iOS": @"26.0" } builder:^id<NFKInferenceBackend> {
			return [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
		}],
	];
	for (NFKAppleEngine *engine in engines) {
		engine.requiresAppleSilicon = YES;
	}
	return engines;
}

+ (NSArray<NFKAppleEngine *> *)audioAndLanguageEngines
{
	NSDictionary<NSString *, NSString *> *floor = [self coreFloor];
	return @[
		[self engineWithIdentifier:@"apple-speech" displayName:@"Apple Speech Recognition" family:@"apple-speech"
						   summary:@"Transcribes speech on device with Apple's recognizer."
							 tasks:@[ NFKModelTaskTranscription ] providerClassName:@"NFKSpeechRecognitionProvider"
				 minimumOSVersions:@{ @"macOS": @"11.0", @"iOS": @"14.0" } builder:^id<NFKInferenceBackend> {
			return [NFKSpeechRecognitionBackend backend];
		}],
		[self engineWithIdentifier:@"apple-speech-synthesis" displayName:@"Apple Voices" family:@"apple-avfoundation"
						   summary:@"Speaks text aloud with the system's voices."
							 tasks:@[ NFKModelTaskSpeechSynthesis ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKSpeechSynthesisBackend backend];
		}],
		[self engineWithIdentifier:@"sound-analysis" displayName:@"Sound Analysis" family:@"apple-sound-analysis"
						   summary:@"Labels the sounds in a recording from Apple's built-in taxonomy, window by window."
							 tasks:@[ NFKModelTaskClassification ] providerClassName:nil
				 minimumOSVersions:@{ @"macOS": @"12.0", @"iOS": @"15.0", @"tvOS": @"15.0" }
						   builder:^id<NFKInferenceBackend> {
			return [NFKSoundClassificationBackend backend];
		}],
		[self engineWithIdentifier:@"natural-language-embedding" displayName:@"Natural Language Sentence Embedding"
							family:@"apple-natural-language"
						   summary:@"Embeds a sentence as a vector for comparing text."
							 tasks:@[ NFKModelTaskEmbedding ] providerClassName:nil
				 minimumOSVersions:floor builder:^id<NFKInferenceBackend> {
			return [NFKTextEmbeddingBackend backend];
		}],
	];
}

/*! Every engine, keyed by registry name. */
+ (NSDictionary<NSString *, NFKAppleEngine *> *)engines
{
	static NSDictionary<NSString *, NFKAppleEngine *> *engines;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		NSMutableDictionary<NSString *, NFKAppleEngine *> *byName = [NSMutableDictionary dictionary];
		NSArray *all = [[[self visionEngines] arrayByAddingObjectsFromArray:[self videoToolboxEngines]]
						arrayByAddingObjectsFromArray:[self audioAndLanguageEngines]];
		for (NFKAppleEngine *engine in all) {
			byName[engine.identifier] = engine;
		}
		engines = [byName copy];
	});
	return engines;
}

#pragma mark Registry

+ (NSArray<NSString *> *)registryNames
{
	return [[self engines].allKeys sortedArrayUsingSelector:@selector(compare:)];
}

+ (nullable id<NFKInferenceBackend>)backendNamed:(NSString *)registryName
{
	NFKAppleEngine *engine = [self engines][registryName];
	return engine != nil ? engine.builder() : nil;
}

+ (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	NSString *name = request.variant.registryName ?: request.descriptor.registryName ?: request.descriptor.identifier;
	id<NFKInferenceBackend> backend = [self backendNamed:name];
	if (backend == nil) {
		if (error != NULL) {
			NSString *reason = [NSString stringWithFormat:@"The Apple engines have no backend named \"%@\".", name];
			*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorUnknownModel
									 userInfo:@{ NSLocalizedDescriptionKey: reason }];
		}
		return nil;
	}
	if (![request applyPropertyOptionsToBackend:backend error:error]) {
		return nil;
	}
	return backend;
}

#pragma mark Descriptors

+ (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	NSMutableArray<NFKModelDescriptor *> *descriptors = [NSMutableArray array];
	for (NSString *name in [self registryNames]) {
		[descriptors addObject:[self descriptorOfEngine:[self engines][name]]];
	}
	return [descriptors copy];
}

// The ports and options come from the backend the registry builds, so a descriptor states exactly
// the keys its backend declares.
+ (NFKModelDescriptor *)descriptorOfEngine:(NFKAppleEngine *)engine
{
	id<NFKInferenceBackend> backend = engine.builder();
	NSSet<NSString *> *inputKeys = [backend respondsToSelector:@selector(supportedInputKeys)] ? backend.supportedInputKeys : [NSSet set];
	NSSet<NSString *> *outputKeys = [backend respondsToSelector:@selector(supportedOutputKeys)] ? backend.supportedOutputKeys : [NSSet set];
	NSSet<NSString *> *parameterKeys = [backend respondsToSelector:@selector(supportedParameterKeys)]
		? backend.supportedParameterKeys : [NSSet set];

	NSMutableArray<NFKModelPort *> *inputs = [NSMutableArray array];
	for (NSString *key in [inputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
		[inputs addObject:[NFKModelPort portWithKey:key required:inputKeys.count == 1]];
	}
	// Every Apple engine writes each key it declares on every result.
	NSMutableArray<NFKModelPort *> *outputs = [NSMutableArray array];
	for (NSString *key in [outputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
		[outputs addObject:[NFKModelPort outputPortWithKey:key required:YES]];
	}
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	for (NSString *key in [parameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
		[options addObject:[NFKModelOption optionWithKey:key]];
	}

	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindSystem repository:nil revision:nil
															  URL:nil layout:NFKModelSourceLayoutNone files:nil
											   requiresConversion:NO extensions:nil];
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"system" displayName:@"System model"
																 isDefault:YES registryName:engine.identifier
																	source:source downloadByteCount:nil
														 residentByteCount:nil peakByteCount:nil
															parameterCount:nil precision:nil
													minimumMemoryByteCount:nil extensions:nil];
	NFKModelLicense *license = [[NFKModelLicense alloc] initWithIdentifier:nil name:@"Apple system framework" URL:nil
																	 gated:NO restrictsCommercialUse:NO
																	  note:@"The model is part of the operating system and runs under its license."
																extensions:nil];
	NFKModelRequirements *requirements = [[NFKModelRequirements alloc] initWithMinimumOSVersions:engine.minimumOSVersions
																			requiresAppleSilicon:engine.requiresAppleSilicon
																		  minimumMemoryByteCount:nil
																					  extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:engine.identifier displayName:engine.displayName
												   family:engine.family summary:engine.summary engine:NFKModelEngineApple
											 registryName:engine.identifier providerClassName:engine.providerClassName
													tasks:engine.tasks inputModalities:nil outputModalities:nil
												   inputs:inputs outputs:outputs options:options variants:@[ variant ]
												  license:license requirements:requirements residencies:nil standIn:NO
										introducedVersion:@"0.4.0" extensions:nil];
}

@end
