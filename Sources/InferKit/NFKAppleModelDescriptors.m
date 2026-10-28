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
#import "NFKInferenceKeys.h"
#import <AVFoundation/AVFoundation.h>

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
/*! The engine's options, described for a user interface; nil states each parameter key the backend
	declares by its key alone. */
@property (nonatomic, copy, nullable) NSArray<NFKModelOption *> * (^options)(void);
/*! The output keys a load option can switch off, which the descriptor marks optional. */
@property (nonatomic, copy, nullable) NSSet<NSString *> *optionalOutputKeys;
@end

@implementation NFKAppleEngine
@end

@implementation NFKAppleModelDescriptors

static BOOL NFKAppleIncludesMachineChoices = YES;

+ (BOOL)includesMachineChoices
{
	return NFKAppleIncludesMachineChoices;
}

+ (void)setIncludesMachineChoices:(BOOL)includesMachineChoices
{
	NFKAppleIncludesMachineChoices = includesMachineChoices;
}

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
		[self speechSynthesisEngine],
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

+ (NFKAppleEngine *)speechSynthesisEngine
{
	NFKAppleEngine *engine = [self engineWithIdentifier:@"apple-speech-synthesis" displayName:@"Apple Voices"
												 family:@"apple-avfoundation"
												summary:@"Speaks text aloud with the system's voices."
												  tasks:@[ NFKModelTaskSpeechSynthesis ] providerClassName:nil
									  minimumOSVersions:[self coreFloor] builder:^id<NFKInferenceBackend> {
		return [NFKSpeechSynthesisBackend backend];
	}];
	engine.options = ^NSArray<NFKModelOption *> *{
		return [NFKAppleModelDescriptors speechSynthesisOptions];
	};
	return engine;
}

/*! The voice, language, rate, pitch, and volume a request sets, with AVFoundation's ranges and defaults. */
+ (NSArray<NFKModelOption *> *)speechSynthesisOptions
{
	NSString *voiceHelp = @"The voice to speak in. Without one, the voice follows the language, and without that, the "
		@"user's setting. The choices are the voices this machine had when the catalog was built: a voice installed "
		@"later in the system's Spoken Content settings is listed once the catalog is built again.";
	NFKModelOption *voice = [[NFKModelOption alloc] initWithKey:NFKParameterVoice kind:NFKModelOptionKindChoice title:@"Voice"
														   help:voiceHelp defaultValue:nil minimum:nil maximum:nil step:nil
														choices:[self installedVoiceChoices] stage:nil group:nil extensions:nil];
	NFKModelOption *language = [[NFKModelOption alloc] initWithKey:NFKParameterSourceLanguage kind:NFKModelOptionKindText
															 title:@"Language"
															  help:@"The language to choose a voice for, a BCP-47 tag such as en-US, when no voice is chosen. "
																   @"Without one, the user's setting."
													  defaultValue:nil minimum:nil maximum:nil step:nil choices:nil stage:nil
															 group:nil extensions:nil];
	NFKModelOption *rate = [[NFKModelOption alloc] initWithKey:NFKSpeechSynthesisParameterRate kind:NFKModelOptionKindNumber
														 title:@"Rate"
														  help:@"How fast the voice speaks, from 0 to 1. 0.5 is the system's default rate."
												  defaultValue:@(AVSpeechUtteranceDefaultSpeechRate) minimum:@0 maximum:@1 step:@0.05
													   choices:nil stage:nil group:nil extensions:nil];
	NFKModelOption *pitch = [[NFKModelOption alloc] initWithKey:NFKSpeechSynthesisParameterPitch kind:NFKModelOptionKindNumber
														  title:@"Pitch"
														   help:@"The pitch multiplier, from 0.5 to 2."
												   defaultValue:@1 minimum:@0.5 maximum:@2 step:@0.05
														choices:nil stage:nil group:nil extensions:nil];
	NFKModelOption *volume = [[NFKModelOption alloc] initWithKey:NFKSpeechSynthesisParameterVolume kind:NFKModelOptionKindNumber
														   title:@"Volume"
															help:@"The volume, from 0 to 1."
													defaultValue:@1 minimum:@0 maximum:@1 step:@0.05
														 choices:nil stage:nil group:nil extensions:nil];
	return @[ voice, language, rate, pitch, volume ];
}

/*! The machine's voices, by language then name, each titled with its name, language, and a quality above
	the default; none when includesMachineChoices is NO. */
+ (NSArray<NFKModelOptionChoice *> *)installedVoiceChoices
{
	if (!self.includesMachineChoices) {
		return @[];
	}
	NSArray<AVSpeechSynthesisVoice *> *voices = [AVSpeechSynthesisVoice.speechVoices sortedArrayUsingComparator:
		^NSComparisonResult(AVSpeechSynthesisVoice *first, AVSpeechSynthesisVoice *second) {
		NSComparisonResult language = [first.language compare:second.language];
		return language != NSOrderedSame ? language : [first.name localizedCompare:second.name];
	}];
	NSMutableArray<NFKModelOptionChoice *> *choices = [NSMutableArray arrayWithCapacity:voices.count];
	for (AVSpeechSynthesisVoice *voice in voices) {
		NSString *quality = voice.quality == AVSpeechSynthesisVoiceQualityEnhanced ? @", enhanced" : @"";
		if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
			if (voice.quality == AVSpeechSynthesisVoiceQualityPremium) {
				quality = @", premium";
			}
		}
		NSString *title = [NSString stringWithFormat:@"%@ (%@%@)", voice.name, voice.language, quality];
		[choices addObject:[NFKModelOptionChoice choiceWithValue:voice.identifier title:title]];
	}
	return choices;
}

#pragma mark Options

/*! The options of every engine but the voices, which set theirs where the engine is made, by identifier.
	A load option's key is the backend property it sets. */
+ (NSDictionary<NSString *, NSArray<NFKModelOption *> * (^)(void)> *)engineOptions
{
	return @{
		@"vision-text": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionTextOptions]; },
		@"vision-segmentation": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionSegmentationOptions]; },
		@"vision-pose": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionPoseOptions]; },
		@"vision-face": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionFaceOptions]; },
		@"vision-classification": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionClassificationOptions]; },
		@"vision-rectangle": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionRectangleOptions]; },
		@"vision-measurement": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionMeasurementOptions]; },
		@"vision-contour": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionContourOptions]; },
		@"vision-registration": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors visionRegistrationOptions]; },
		@"vision-tracking": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors objectTrackingOptions]; },
		@"vision-tracking-trajectory": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors trajectoryOptions]; },
		@"videotoolbox-super-resolution": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors superResolutionOptions]; },
		@"videotoolbox-optical-flow": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors opticalFlowOptions]; },
		@"apple-speech": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors speechRecognitionOptions]; },
		@"sound-analysis": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors soundAnalysisOptions]; },
		@"natural-language-embedding": ^NSArray<NFKModelOption *> *{ return [NFKAppleModelDescriptors sentenceEmbeddingOptions]; },
	};
}

/*! A load option that sets the backend property its key names. */
+ (NFKModelOption *)loadOption:(NSString *)property
						  kind:(NSString *)kind
						 title:(NSString *)title
						  help:(NSString *)help
				  defaultValue:(nullable id)defaultValue
					   minimum:(nullable NSNumber *)minimum
					   maximum:(nullable NSNumber *)maximum
						  step:(nullable NSNumber *)step
{
	return [[NFKModelOption alloc] initWithKey:property kind:kind title:title help:help defaultValue:defaultValue
									   minimum:minimum maximum:maximum step:step choices:nil
										 stage:NFKModelOptionStageLoad group:nil requirements:nil
							   propertyKeyPath:property memoryBytesPerUnit:nil extensions:nil];
}

+ (NFKModelOption *)loadOption:(NSString *)property title:(NSString *)title help:(NSString *)help
						 flag:(BOOL)defaultValue
{
	return [self loadOption:property kind:NFKModelOptionKindBoolean title:title help:help defaultValue:@(defaultValue)
					minimum:nil maximum:nil step:nil];
}

/*! A load option whose choices are the values of an enumeration property, the first the default. */
+ (NFKModelOption *)loadOption:(NSString *)property title:(NSString *)title help:(NSString *)help
					   choices:(NSArray<NFKModelOptionChoice *> *)choices
{
	return [[NFKModelOption alloc] initWithKey:property kind:NFKModelOptionKindChoice title:title help:help
								  defaultValue:choices.firstObject.value minimum:nil maximum:nil step:nil choices:choices
										 stage:NFKModelOptionStageLoad group:nil requirements:nil
							   propertyKeyPath:property memoryBytesPerUnit:nil extensions:nil];
}

/*! A choice that needs an OS later than the core's floor, or every machine when versions is nil. */
+ (NFKModelOptionChoice *)choice:(NSInteger)value title:(NSString *)title
					   minimumOS:(nullable NSDictionary<NSString *, NSString *> *)versions
{
	NFKModelRequirements *requirements = versions == nil ? nil
		: [[NFKModelRequirements alloc] initWithMinimumOSVersions:versions requiresAppleSilicon:NO
										   minimumMemoryByteCount:nil extensions:nil];
	return [[NFKModelOptionChoice alloc] initWithValue:@(value) title:title requirements:requirements memory:nil
									   memoryByVariant:nil extensions:nil];
}

+ (NFKModelOption *)unitOption:(NSString *)property title:(NSString *)title help:(NSString *)help
				  defaultValue:(double)defaultValue
{
	return [self loadOption:property kind:NFKModelOptionKindNumber title:title help:help defaultValue:@(defaultValue)
					minimum:@0 maximum:@1 step:@0.05];
}

+ (NSArray<NFKModelOption *> *)visionTextOptions
{
	return @[
		[self loadOption:@"usesAccurateRecognition" title:@"Accurate recognition"
					help:@"Vision's accurate recognizer. Off, its faster one." flag:YES],
		[self loadOption:@"correctsLanguage" title:@"Language correction"
					help:@"Corrects the reading with a language model." flag:YES],
		[self loadOption:@"languages" kind:NFKModelOptionKindTextList title:@"Languages"
					help:@"The languages to read, BCP-47 tags such as en-US in order of preference. Without them, "
						 @"Vision's own order, which follows the user's preferred languages."
			defaultValue:nil minimum:nil maximum:nil step:nil],
		[self loadOption:@"customWords" kind:NFKModelOptionKindTextList title:@"Custom words"
					help:@"Words to add to the vocabulary, such as names and terms a general model corrects away."
			defaultValue:nil minimum:nil maximum:nil step:nil],
		[self loadOption:@"minimumTextHeight" kind:NFKModelOptionKindNumber title:@"Minimum text height"
					help:@"The smallest text to read, as a fraction of the image height. 0 uses Vision's default."
			defaultValue:@0 minimum:@0 maximum:@1 step:@0.01],
	];
}

+ (NSArray<NFKModelOption *> *)visionSegmentationOptions
{
	NSDictionary<NSString *, NSString *> *sonoma = @{ @"macOS": @"14.0", @"iOS": @"17.0", @"tvOS": @"17.0" };
	NSArray<NFKModelOptionChoice *> *kinds = @[
		[self choice:NFKVisionSegmentationKindForegroundInstance title:@"Subject" minimumOS:sonoma],
		[self choice:NFKVisionSegmentationKindPerson title:@"People"
		   minimumOS:@{ @"macOS": @"12.0", @"iOS": @"15.0", @"tvOS": @"15.0" }],
		[self choice:NFKVisionSegmentationKindAttentionSaliency title:@"Attention saliency" minimumOS:nil],
		[self choice:NFKVisionSegmentationKindObjectnessSaliency title:@"Objectness saliency" minimumOS:nil],
		[self choice:NFKVisionSegmentationKindPersonInstances title:@"Up to four people" minimumOS:sonoma],
	];
	NSArray<NFKModelOptionChoice *> *qualities = @[
		[self choice:NFKVisionSegmentationQualityAccurate title:@"Accurate" minimumOS:nil],
		[self choice:NFKVisionSegmentationQualityBalanced title:@"Balanced" minimumOS:nil],
		[self choice:NFKVisionSegmentationQualityFast title:@"Fast" minimumOS:nil],
	];
	return @[ [self loadOption:@"kind" title:@"Mask" help:@"What the mask covers." choices:kinds],
			  [self loadOption:@"quality" title:@"Quality"
						  help:@"The person mask's quality, traded against speed. The other masks ignore it."
					   choices:qualities] ];
}

+ (NSArray<NFKModelOption *> *)visionPoseOptions
{
	NSArray<NFKModelOptionChoice *> *kinds = @[
		[self choice:NFKVisionPoseKindBody title:@"Body" minimumOS:nil],
		[self choice:NFKVisionPoseKindHand title:@"Hand" minimumOS:nil],
		[self choice:NFKVisionPoseKindAnimal title:@"Animal"
		   minimumOS:@{ @"macOS": @"14.0", @"iOS": @"17.0", @"tvOS": @"17.0" }],
	];
	return @[ [self loadOption:@"kind" title:@"Joints"
						  help:@"Nineteen joints per body, twenty-one per hand, or twenty-five per cat or dog."
					   choices:kinds],
			  [self loadOption:@"maximumHandCount" kind:NFKModelOptionKindInteger title:@"Most hands"
						  help:@"The most hands to find. The body and animal joints ignore it."
				  defaultValue:@2 minimum:@1 maximum:nil step:@1] ];
}

+ (NSArray<NFKModelOption *> *)visionFaceOptions
{
	return @[ [self loadOption:@"detectsLandmarks" title:@"Landmarks"
						  help:@"Reads each face's landmarks under the pose output as well as its box." flag:YES] ];
}

+ (NSArray<NFKModelOption *> *)visionClassificationOptions
{
	return @[ [self unitOption:@"minimumConfidence" title:@"Minimum confidence"
						  help:@"Labels below this confidence are dropped." defaultValue:0.1],
			  [self unitOption:@"minimumRecallForPrecision" title:@"Minimum recall"
						  help:@"Above 0, keeps the labels that reach this recall at the precision Vision measured, "
							   @"and ignores the minimum confidence."
				  defaultValue:0] ];
}

+ (NSArray<NFKModelOption *> *)visionRectangleOptions
{
	NSArray<NFKModelOptionChoice *> *kinds = @[
		[self choice:NFKVisionRectangleKindRectangle title:@"Rectangle" minimumOS:nil],
		[self choice:NFKVisionRectangleKindDocument title:@"Document page"
		   minimumOS:@{ @"macOS": @"12.0", @"iOS": @"15.0", @"tvOS": @"15.0" }],
		[self choice:NFKVisionRectangleKindBarcode title:@"Barcode" minimumOS:nil],
	];
	return @[
		[self loadOption:@"kind" title:@"Shape"
					help:@"Any rectangle, the page in a photograph, or a barcode, which is read as well as located."
				 choices:kinds],
		[self loadOption:@"maximumCount" kind:NFKModelOptionKindInteger title:@"Most shapes"
					help:@"The most shapes to report. Without it, 16, and 1 for a document page."
			defaultValue:nil minimum:@1 maximum:nil step:@1],
		[self unitOption:@"minimumSize" title:@"Minimum size"
					help:@"The smallest side of a rectangle, as a fraction of the image." defaultValue:0.2],
		[self loadOption:@"quadratureTolerance" kind:NFKModelOptionKindNumber title:@"Corner tolerance"
					help:@"How far from square a rectangle's corner may be, in degrees."
			defaultValue:@30 minimum:@0 maximum:@45 step:@1],
		[self loadOption:@"symbologies" kind:NFKModelOptionKindTextList title:@"Symbologies"
					help:@"The barcode symbologies to look for, by Vision's names such as VNBarcodeSymbologyQR. "
						 @"Without them, every symbology the installed Vision reads."
			defaultValue:nil minimum:nil maximum:nil step:nil],
	];
}

+ (NSArray<NFKModelOption *> *)visionMeasurementOptions
{
	NSArray<NFKModelOptionChoice *> *kinds = @[
		[self choice:NFKVisionMeasurementKindAesthetics title:@"Aesthetics"
		   minimumOS:@{ @"macOS": @"15.0", @"iOS": @"18.0", @"tvOS": @"18.0" }],
		[self choice:NFKVisionMeasurementKindHorizon title:@"Horizon" minimumOS:nil],
	];
	return @[ [self loadOption:@"kind" title:@"Reading"
						  help:@"The image's aesthetic score, or the angle of its horizon." choices:kinds] ];
}

+ (NSArray<NFKModelOption *> *)visionContourOptions
{
	return @[
		[self unitOption:@"detail" title:@"Detail" help:@"How finely to trace. Higher keeps more points." defaultValue:0.5],
		[self loadOption:@"contrastAdjustment" kind:NFKModelOptionKindNumber title:@"Contrast"
					help:@"The contrast applied before tracing." defaultValue:@2 minimum:@0 maximum:@3 step:@0.1],
		[self loadOption:@"detectsDarkOnLight" title:@"Dark on light"
					help:@"Traces dark shapes on a light field. Off, light shapes on a dark field." flag:NO],
		[self loadOption:@"maximumPointsPerContour" kind:NFKModelOptionKindInteger title:@"Most points per contour"
					help:@"The most points to report per contour. 0 reports them all."
			defaultValue:@0 minimum:@0 maximum:nil step:@1],
	];
}

+ (NSArray<NFKModelOption *> *)visionRegistrationOptions
{
	NSArray<NFKModelOptionChoice *> *kinds = @[
		[self choice:NFKVisionRegistrationKindTranslational title:@"Translation" minimumOS:nil],
		[self choice:NFKVisionRegistrationKindHomographic title:@"Homography" minimumOS:nil],
	];
	return @[ [self loadOption:@"kind" title:@"Alignment"
						  help:@"A shift, reported as an affine transform, or a homography of nine numbers."
					   choices:kinds] ];
}

+ (NSArray<NFKModelOption *> *)objectTrackingOptions
{
	NFKModelOption *region = [[NFKModelOption alloc] initWithKey:NFKVisionTrackingParameterRegion kind:NFKModelOptionKindJSON
														   title:@"Region"
															help:@"The region to follow, an array [x, y, width, height] normalized 0 to 1 with "
																 @"the origin at the top left. The first frame needs one. A different region "
																 @"starts following it from that frame; the same region continues."
													defaultValue:nil minimum:nil maximum:nil step:nil choices:nil stage:nil
														   group:nil extensions:nil];
	NFKModelOption *level = [[NFKModelOption alloc] initWithKey:NFKVisionTrackingParameterLevel kind:NFKModelOptionKindChoice
														  title:@"Tracking level"
														   help:@"Vision's accurate tracker, or its faster one."
												   defaultValue:@"accurate" minimum:nil maximum:nil step:nil
														choices:@[ [NFKModelOptionChoice choiceWithValue:@"accurate" title:@"Accurate"],
																   [NFKModelOptionChoice choiceWithValue:@"fast" title:@"Fast"] ]
														  stage:nil group:nil extensions:nil];
	return @[ region, level, [self restartTrackingOption] ];
}

+ (NSArray<NFKModelOption *> *)trajectoryOptions
{
	return @[
		[self restartTrackingOption],
		[self loadOption:@"trajectoryLength" kind:NFKModelOptionKindInteger title:@"Trajectory length"
					help:@"The frames a trajectory needs before it is reported. Vision's minimum is 5."
			defaultValue:@5 minimum:@5 maximum:nil step:@1],
		[self loadOption:@"framesPerSecond" kind:NFKModelOptionKindNumber title:@"Frame rate"
					help:@"The rate the submitted frames are timed at, in frames per second. Vision times a "
						 @"trajectory, and a still frame carries no timestamp."
			defaultValue:@30 minimum:@1 maximum:@240 step:@1],
	];
}

+ (NFKModelOption *)restartTrackingOption
{
	return [[NFKModelOption alloc] initWithKey:NFKVisionTrackingParameterRestart kind:NFKModelOptionKindBoolean
										 title:@"Restart"
										  help:@"Ends the sequence before this frame, as when the clip changes."
								  defaultValue:@NO minimum:nil maximum:nil step:nil choices:nil stage:nil group:nil
									extensions:nil];
}

+ (NSArray<NFKModelOption *> *)superResolutionOptions
{
	return @[ [self loadOption:@"scaleFactor" kind:NFKModelOptionKindInteger title:@"Scale factor"
						  help:@"The upscaling factor. 0 takes the smallest factor the machine offers; a factor it "
							   @"does not offer is refused by name."
				  defaultValue:@0 minimum:@0 maximum:nil step:@1] ];
}

+ (NSArray<NFKModelOption *> *)opticalFlowOptions
{
	return @[ [self loadOption:@"flowScale" kind:NFKModelOptionKindNumber title:@"Flow scale"
						  help:@"The motion a packed flow channel covers, in pixels: a channel reads "
							   @"0.5 + component / (2 × scale)."
				  defaultValue:@32 minimum:@1 maximum:nil step:@1] ];
}

+ (NSArray<NFKModelOption *> *)speechRecognitionOptions
{
	NFKModelOption *language = [[NFKModelOption alloc] initWithKey:NFKParameterSourceLanguage kind:NFKModelOptionKindText
															 title:@"Language"
															  help:@"The language to transcribe, a locale identifier such as en-US. Without "
																   @"one, the user's current locale."
													  defaultValue:nil minimum:nil maximum:nil step:nil choices:nil stage:nil
															 group:nil extensions:nil];
	NFKModelRequirements *ventura = [[NFKModelRequirements alloc] initWithMinimumOSVersions:@{ @"macOS": @"13.0", @"iOS": @"16.0" }
																	   requiresAppleSilicon:NO minimumMemoryByteCount:nil
																				 extensions:nil];
	NFKModelOption *punctuation = [[NFKModelOption alloc] initWithKey:@"addsPunctuation" kind:NFKModelOptionKindBoolean
																title:@"Punctuation" help:@"Punctuates the transcription."
														 defaultValue:@YES minimum:nil maximum:nil step:nil choices:nil
																stage:NFKModelOptionStageLoad group:nil requirements:ventura
													  propertyKeyPath:@"addsPunctuation" memoryBytesPerUnit:nil extensions:nil];
	return @[ language,
			  [self loadOption:@"requiresOnDeviceRecognition" title:@"On device only"
						  help:@"Refuses recognition that would send the audio off the machine." flag:YES],
			  punctuation ];
}

+ (NSArray<NFKModelOption *> *)soundAnalysisOptions
{
	return @[ [self unitOption:@"minimumConfidence" title:@"Minimum confidence"
						  help:@"Sounds below this confidence are dropped." defaultValue:0.3],
			  [self loadOption:@"windowSeconds" kind:NFKModelOptionKindNumber title:@"Window"
						  help:@"The window the classifier reads, in seconds. 0 uses the classifier's own window."
				  defaultValue:@0 minimum:@0 maximum:nil step:@0.1] ];
}

+ (NSArray<NFKModelOption *> *)sentenceEmbeddingOptions
{
	return @[ [self loadOption:@"language" kind:NFKModelOptionKindText title:@"Language"
						  help:@"The language to embed for, a BCP-47 tag such as en. Without one, each request's "
							   @"language is detected from its text."
				  defaultValue:nil minimum:nil maximum:nil step:nil] ];
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
		NSDictionary<NSString *, NSArray<NFKModelOption *> * (^)(void)> *options = [self engineOptions];
		for (NFKAppleEngine *engine in all) {
			byName[engine.identifier] = engine;
			if (engine.options == nil) {
				engine.options = options[engine.identifier];
			}
		}
		byName[@"vision-face"].optionalOutputKeys = [NSSet setWithObject:NFKOutputPose];
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
	// Every Apple engine writes each key it declares on every result, except a key a load option
	// switches off.
	NSMutableArray<NFKModelPort *> *outputs = [NSMutableArray array];
	for (NSString *key in [outputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
		[outputs addObject:[NFKModelPort outputPortWithKey:key required:![engine.optionalOutputKeys containsObject:key]]];
	}
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	if (engine.options != nil) {
		[options addObjectsFromArray:engine.options()];
	} else {
		for (NSString *key in [parameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
			[options addObject:[NFKModelOption optionWithKey:key]];
		}
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
