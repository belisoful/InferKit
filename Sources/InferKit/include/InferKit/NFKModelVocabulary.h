//
//  NFKModelVocabulary.h
//  InferKit
//

#ifndef NFKModelVocabulary_h
#define NFKModelVocabulary_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelVocabulary
	@abstract   The strings a model descriptor uses for value types, tasks, engines, residencies, option
				kinds and stages, file roles, and source kinds and layouts.
	@discussion Each vocabulary is a set of plain string constants. A descriptor field holds any string,
				so a plugin names a task, a value type, or an engine the core does not list, and a reader
				passes an unknown string through unchanged. The constants are the values the core's own
				descriptors use and the values a reader can rely on. Introduced in InferKit 0.4.0.
*/

#pragma mark Value types

/*! A string (NSString). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeText;
/*! A chat conversation in the OpenAI shape NFKInputMessages takes (NSArray of NSDictionary).
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeMessages;
/*! One image (CGImage, CVPixelBuffer, or texture). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeImage;
/*! Several images (NSArray of images). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeImages;
/*! A mask or matte over an image (CGImage, CVPixelBuffer, or texture). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeMask;
/*! One clip (NFKVideoAsset). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeVideo;
/*! Several clips (NSArray of NFKVideoAsset). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeVideos;
/*! One recording (NFKAudioAsset, or NSData PCM). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeAudio;
/*! Several recordings (NSArray of NFKAudioAsset or NSData). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeAudios;
/*! One document for a text model to read (NSURL, NSData, or NSString). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeDocument;
/*! Several documents (NSArray of NSURL or NSData). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeDocuments;
/*! A real number (NSNumber). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeNumber;
/*! A whole number (NSNumber). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeInteger;
/*! A boolean (NSNumber holding YES or NO). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeBoolean;
/*! A list of strings (NSArray of NSString). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeTextList;
/*! Any JSON value (NSDictionary, NSArray, NSString, or NSNumber). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeJSON;
/*! A feature vector (NSArray of NSNumber). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeEmbedding;
/*! Detected objects (NSArray of NFKDetection). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeDetections;
/*! Located landmarks (NSArray of NFKKeypoint). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeKeypoints;
/*! Predicted classes (NSArray of NFKClassification). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeClassifications;
/*! Time spans (NSArray of NFKAudioSegment). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeSegments;
/*! Timed words (NSArray of NFKAudioSegment, the word as the label). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeWords;
/*! A note sequence (NFKMIDISequence). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeMIDI;
/*! The beats of a track (NSArray of NFKMusicBeat). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeBeats;
/*! Typed questions for a decision model (NSDictionary of NFKDecisionQuestion). Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeQuestions;
/*! Typed answers from a decision model (NSDictionary of NFKDecisionAnswer). Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeAnswers;
/*! The tool calls a text model made (NSArray of NSDictionary). Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeToolCalls;
/*! Named readings keyed by field name (NSDictionary), the shape of NFKOutputStructured. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelValueTypeStructured;

#pragma mark Tasks

/*! Writing text from a prompt or a conversation. Equal to NFKCapabilityTextGeneration. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelTaskTextGeneration;
/*! Answering a question about an image or a clip. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskVisionQuestionAnswering;
/*! Turning speech into text. Equal to NFKCapabilityTranscription. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskTranscription;
/*! Turning text from one language into another. Equal to NFKCapabilityTranslation. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelTaskTranslation;
/*! Reading the text in an image. Equal to NFKCapabilityTextRecognition. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskTextRecognition;
/*! Speaking text aloud. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskSpeechSynthesis;
/*! Making an image from a prompt. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskTextToImage;
/*! Making an image from a source image and a prompt. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskImageToImage;
/*! Filling a masked region of an image. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskInpainting;
/*! Generating an image under a control map. Equal to NFKCapabilityControlNet. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelTaskControlNet;
/*! Making a clip from a prompt, an image, or both. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskVideoGeneration;
/*! Making the frames between two frames. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskFrameInterpolation;
/*! Enlarging an image or a frame. Equal to NFKCapabilityUpscaling. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskUpscaling;
/*! Removing noise, blur, or damage from an image. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskImageRestoration;
/*! Estimating the distance of each pixel. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskDepthEstimation;
/*! A mask over a region or a subject of an image. Equal to NFKCapabilitySegmentation. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelTaskSegmentation;
/*! A soft alpha matte that separates a subject from its background. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskMatting;
/*! Boxes around the objects in an image. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskObjectDetection;
/*! Body or hand joints. Equal to NFKCapabilityPose. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskPoseEstimation;
/*! Faces and their landmarks. Equal to NFKCapabilityFaceDetection. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskFaceDetection;
/*! Motion between two frames. Equal to NFKCapabilityOpticalFlow. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskOpticalFlow;
/*! Labels for an image, a clip, or a recording. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskClassification;
/*! A vector for comparing images. Equal to NFKCapabilityImageEmbedding. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskImageEmbedding;
/*! A vector for comparing text, or text with images. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskEmbedding;
/*! Ordering documents by relevance to a query. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskReranking;
/*! Predicting the next values of a time series. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskForecasting;
/*! Splitting a recording into its sources: voices, instruments, stems. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskSourceSeparation;
/*! Removing noise and reverberation from speech. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskSpeechEnhancement;
/*! Finding the spans of a recording that hold speech. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskVoiceActivityDetection;
/*! Encoding audio into tokens and decoding it back. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskAudioCodec;
/*! Making music from a prompt or lyrics. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskMusicGeneration;
/*! Turning a performance into notes. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskMusicTranscription;
/*! The beats and sections of a track. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelTaskMusicStructure;

#pragma mark Engines

/*! The InferKitMLX companion. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelEngineMLX;
/*! The core's Apple-framework engines: Vision, VideoToolbox, Speech, SoundAnalysis, AVFoundation, and
	NaturalLanguage. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelEngineApple;
/*! The InferKitFoundationModels companion. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelEngineFoundationModels;
/*! The InferKitAppleSwift companion. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelEngineAppleSwift;
/*! A backend a plugin or an app brings. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelEnginePlugin;

#pragma mark Residencies

/*! Every stage stays loaded between runs. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelResidencyResident;
/*! Each stage loads for its turn and is released after it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelResidencyStaged;
/*! Stages take turns, and each stage's routed experts load as a step routes to them. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelResidencyPaged;
/*! Stages take turns, and a stage's repeated layers that do not fit load in their turn on each pass.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelResidencyStreamed;

#pragma mark Option kinds

/*! An on-off switch. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindBoolean;
/*! A whole number. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindInteger;
/*! A real number. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindNumber;
/*! One of the option's choices. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindChoice;
/*! A string. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindText;
/*! A list of strings. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindTextList;
/*! Any JSON value. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindJSON;
/*! A path to a file or a folder on disk. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindFileOrFolder;
/*! The identifier of another descriptor in the catalog. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionKindModelReference;

#pragma mark Option stages

/*! The option rides on each request as a parameter. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionStageRequest;
/*! The option is fixed when the backend is built, and changing it builds the backend again.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelOptionStageLoad;

#pragma mark File roles

/*! A file holding the model's weights. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelFileRoleWeights;
/*! A file holding the model's configuration. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelFileRoleConfig;
/*! A file holding the model's tokenizer or vocabulary. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelFileRoleTokenizer;
/*! Any other file the model reads. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelFileRoleOther;

#pragma mark Source kinds

/*! A Hugging Face repository. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceKindHuggingFace;
/*! A download URL. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceKindURL;
/*! A model the operating system provides. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceKindSystem;
/*! No download: the model needs no files, or the app supplies them. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceKindNone;

#pragma mark Source layouts

/*! The model loads from one file. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceLayoutFile;
/*! The model loads from a directory of files. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceLayoutDirectory;
/*! The model loads from no files of its own. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelSourceLayoutNone;

#pragma mark Key types

/*!
	@function   NFKModelValueTypeForKey
	@abstract   The value type a core request key carries, or nil for a key the core does not define.
	@discussion The table covers every NFKInput*, NFKParameter*, and NFKOutput* key. A request key
				and a result key that share a string resolve to the request key's type, so
				"citations" is NFKParameterCitations, a boolean. NFKModelValueTypeForOutputKey resolves
				the result key's type. Introduced in InferKit 0.4.0.
*/
NSString * _Nullable NFKModelValueTypeForKey(NSString *key);

/*!
	@function   NFKModelValueTypeForOutputKey
	@abstract   The value type a core result key carries, or nil for a key the core does not define.
	@discussion The result side of NFKModelValueTypeForKey: an NFKOutput* key resolves to its own type,
				and any other core key resolves as NFKModelValueTypeForKey resolves it. Introduced in
				InferKit 0.4.0.
*/
NSString * _Nullable NFKModelValueTypeForOutputKey(NSString *key);

NS_ASSUME_NONNULL_END

#endif /* NFKModelVocabulary_h */
