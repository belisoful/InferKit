//
//  NFKModelVocabulary.m
//  InferKit
//

#import "NFKModelVocabulary.h"
#import "NFKInferenceKeys.h"

NSString * const NFKModelValueTypeText				= @"text";
NSString * const NFKModelValueTypeMessages			= @"messages";
NSString * const NFKModelValueTypeImage				= @"image";
NSString * const NFKModelValueTypeImages			= @"images";
NSString * const NFKModelValueTypeMask				= @"mask";
NSString * const NFKModelValueTypeVideo				= @"video";
NSString * const NFKModelValueTypeVideos			= @"videos";
NSString * const NFKModelValueTypeAudio				= @"audio";
NSString * const NFKModelValueTypeAudios			= @"audios";
NSString * const NFKModelValueTypeDocument			= @"document";
NSString * const NFKModelValueTypeDocuments			= @"documents";
NSString * const NFKModelValueTypeNumber			= @"number";
NSString * const NFKModelValueTypeInteger			= @"integer";
NSString * const NFKModelValueTypeBoolean			= @"boolean";
NSString * const NFKModelValueTypeTextList			= @"text-list";
NSString * const NFKModelValueTypeJSON				= @"json";
NSString * const NFKModelValueTypeEmbedding			= @"embedding";
NSString * const NFKModelValueTypeDetections		= @"detections";
NSString * const NFKModelValueTypeKeypoints			= @"keypoints";
NSString * const NFKModelValueTypeClassifications	= @"classifications";
NSString * const NFKModelValueTypeSegments			= @"segments";
NSString * const NFKModelValueTypeWords				= @"words";
NSString * const NFKModelValueTypeMIDI				= @"midi";
NSString * const NFKModelValueTypeBeats				= @"beats";
NSString * const NFKModelValueTypeQuestions			= @"questions";
NSString * const NFKModelValueTypeAnswers			= @"answers";
NSString * const NFKModelValueTypeToolCalls			= @"tool-calls";
NSString * const NFKModelValueTypeStructured		= @"structured";

// The tasks a capability already names keep the capability's string, so a task resolves through
// NFKDynamicBackend as it is.
NSString * const NFKModelTaskTextGeneration			= @"text-generation";
NSString * const NFKModelTaskVisionQuestionAnswering = @"vision-question-answering";
NSString * const NFKModelTaskTranscription			= @"transcription";
NSString * const NFKModelTaskTranslation			= @"translation";
NSString * const NFKModelTaskTextRecognition		= @"text-recognition";
NSString * const NFKModelTaskSpeechSynthesis		= @"speech-synthesis";
NSString * const NFKModelTaskTextToImage			= @"text-to-image";
NSString * const NFKModelTaskImageToImage			= @"image-to-image";
NSString * const NFKModelTaskInpainting				= @"inpainting";
NSString * const NFKModelTaskControlNet				= @"controlnet";
NSString * const NFKModelTaskVideoGeneration		= @"video-generation";
NSString * const NFKModelTaskFrameInterpolation		= @"frame-interpolation";
NSString * const NFKModelTaskUpscaling				= @"upscaling";
NSString * const NFKModelTaskImageRestoration		= @"image-restoration";
NSString * const NFKModelTaskDepthEstimation		= @"depth-estimation";
NSString * const NFKModelTaskSegmentation			= @"segmentation";
NSString * const NFKModelTaskMatting				= @"matting";
NSString * const NFKModelTaskObjectDetection		= @"object-detection";
NSString * const NFKModelTaskPoseEstimation			= @"pose";
NSString * const NFKModelTaskFaceDetection			= @"face-detection";
NSString * const NFKModelTaskOpticalFlow			= @"optical-flow";
NSString * const NFKModelTaskClassification			= @"classification";
NSString * const NFKModelTaskImageEmbedding			= @"image-embedding";
NSString * const NFKModelTaskEmbedding				= @"embedding";
NSString * const NFKModelTaskReranking				= @"reranking";
NSString * const NFKModelTaskForecasting			= @"forecasting";
NSString * const NFKModelTaskSourceSeparation		= @"source-separation";
NSString * const NFKModelTaskSpeechEnhancement		= @"speech-enhancement";
NSString * const NFKModelTaskVoiceActivityDetection	= @"voice-activity-detection";
NSString * const NFKModelTaskAudioCodec				= @"audio-codec";
NSString * const NFKModelTaskMusicGeneration		= @"music-generation";
NSString * const NFKModelTaskMusicTranscription		= @"music-transcription";
NSString * const NFKModelTaskMusicStructure			= @"music-structure";

NSString * const NFKModelEngineMLX					= @"mlx";
NSString * const NFKModelEngineApple				= @"apple";
NSString * const NFKModelEngineFoundationModels		= @"foundation-models";
NSString * const NFKModelEngineAppleSwift			= @"apple-swift";
NSString * const NFKModelEnginePlugin				= @"plugin";

NSString * const NFKModelResidencyResident			= @"resident";
NSString * const NFKModelResidencyStaged			= @"staged";
NSString * const NFKModelResidencyPaged				= @"paged";
NSString * const NFKModelResidencyStreamed			= @"streamed";

NSString * const NFKModelOptionKindBoolean			= @"boolean";
NSString * const NFKModelOptionKindInteger			= @"integer";
NSString * const NFKModelOptionKindNumber			= @"number";
NSString * const NFKModelOptionKindChoice			= @"choice";
NSString * const NFKModelOptionKindText				= @"text";
NSString * const NFKModelOptionKindTextList			= @"text-list";
NSString * const NFKModelOptionKindJSON				= @"json";
NSString * const NFKModelOptionKindFileOrFolder		= @"file-or-folder";
NSString * const NFKModelOptionKindModelReference	= @"model-reference";

NSString * const NFKModelOptionStageRequest			= @"request";
NSString * const NFKModelOptionStageLoad			= @"load";

NSString * const NFKModelFileRoleWeights			= @"weights";
NSString * const NFKModelFileRoleConfig				= @"config";
NSString * const NFKModelFileRoleTokenizer			= @"tokenizer";
NSString * const NFKModelFileRoleOther				= @"other";

NSString * const NFKModelSourceKindHuggingFace		= @"huggingface";
NSString * const NFKModelSourceKindURL				= @"url";
NSString * const NFKModelSourceKindSystem			= @"system";
NSString * const NFKModelSourceKindNone				= @"none";

NSString * const NFKModelSourceLayoutFile			= @"file";
NSString * const NFKModelSourceLayoutDirectory		= @"directory";
NSString * const NFKModelSourceLayoutNone			= @"none";

static NSDictionary<NSString *, NSString *> *NFKRequestKeyValueTypes(void)
{
	static NSDictionary *types;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		types = @{
			NFKInputPrompt:							NFKModelValueTypeText,
			NFKInputNegativePrompt:					NFKModelValueTypeText,
			NFKInputImage:							NFKModelValueTypeImage,
			NFKInputMask:							NFKModelValueTypeMask,
			NFKInputImages:							NFKModelValueTypeImages,
			NFKInputDocument:						NFKModelValueTypeDocument,
			NFKInputDocuments:						NFKModelValueTypeDocuments,
			NFKInputControl:						NFKModelValueTypeImage,
			NFKInputVideo:							NFKModelValueTypeVideo,
			NFKInputLastFrame:						NFKModelValueTypeImage,
			NFKInputVoiceReference:					NFKModelValueTypeAudio,
			NFKInputSuffix:							NFKModelValueTypeText,
			NFKInputAudio:							NFKModelValueTypeAudio,
			NFKInputAudios:							NFKModelValueTypeAudios,
			NFKInputMessages:						NFKModelValueTypeMessages,
			NFKInputLyrics:							NFKModelValueTypeText,
			NFKInputState:							NFKModelValueTypeJSON,
			NFKInputQuestions:						NFKModelValueTypeQuestions,

			NFKParameterSeed:						NFKModelValueTypeInteger,
			NFKParameterSteps:						NFKModelValueTypeInteger,
			NFKParameterGuidanceScale:				NFKModelValueTypeNumber,
			NFKParameterStrength:					NFKModelValueTypeNumber,
			NFKParameterWidth:						NFKModelValueTypeInteger,
			NFKParameterHeight:						NFKModelValueTypeInteger,
			NFKParameterTools:						NFKModelValueTypeJSON,
			NFKParameterMCPServers:					NFKModelValueTypeJSON,
			NFKParameterJSONSchema:					NFKModelValueTypeJSON,
			NFKParameterOutputFormat:				NFKModelValueTypeText,
			NFKParameterChoices:					NFKModelValueTypeTextList,
			NFKParameterVideoFrameCount:			NFKModelValueTypeInteger,
			NFKParameterAudioOutput:				NFKModelValueTypeJSON,
			NFKParameterSampleCount:				NFKModelValueTypeInteger,
			NFKParameterFrameCount:					NFKModelValueTypeInteger,
			NFKParameterFramesPerSecond:			NFKModelValueTypeNumber,
			NFKParameterDurationSeconds:			NFKModelValueTypeNumber,
			NFKParameterAspectRatio:				NFKModelValueTypeText,
			NFKParameterResolution:					NFKModelValueTypeText,
			NFKParameterGenerateAudio:				NFKModelValueTypeBoolean,
			NFKParameterVideoOperation:				NFKModelValueTypeText,
			NFKParameterSourceVideoIdentifier:		NFKModelValueTypeText,
			NFKParameterSpeakerDiarization:			NFKModelValueTypeBoolean,
			NFKParameterWordTimestamps:				NFKModelValueTypeBoolean,
			NFKParameterVocabulary:					NFKModelValueTypeTextList,
			NFKParameterCitations:					NFKModelValueTypeBoolean,
			NFKParameterPreviousResponseIdentifier:	NFKModelValueTypeText,
			NFKParameterConversationKey:			NFKModelValueTypeText,
			NFKParameterMotionScale:				NFKModelValueTypeNumber,
			NFKParameterSampleRate:					NFKModelValueTypeInteger,
			NFKParameterChannelCount:				NFKModelValueTypeInteger,
			NFKParameterSourceLanguage:				NFKModelValueTypeText,
			NFKParameterTargetLanguage:				NFKModelValueTypeText,
			NFKParameterTemperature:				NFKModelValueTypeNumber,
			NFKParameterTopP:						NFKModelValueTypeNumber,
			NFKParameterTopK:						NFKModelValueTypeInteger,
			NFKParameterMaxTokens:					NFKModelValueTypeInteger,
			NFKParameterRepetitionPenalty:			NFKModelValueTypeNumber,
			NFKParameterStopSequences:				NFKModelValueTypeTextList,
			NFKParameterReasoningEffort:			NFKModelValueTypeText,
		};
	});
	return types;
}

static NSDictionary<NSString *, NSString *> *NFKResultKeyValueTypes(void)
{
	static NSDictionary *types;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		types = @{
			NFKOutputImage:							NFKModelValueTypeImage,
			NFKOutputImages:						NFKModelValueTypeImages,
			NFKOutputVideo:							NFKModelValueTypeVideo,
			NFKOutputVideos:						NFKModelValueTypeVideos,
			NFKOutputWords:							NFKModelValueTypeWords,
			NFKOutputCitations:						NFKModelValueTypeJSON,
			NFKOutputServerToolResults:				NFKModelValueTypeJSON,
			NFKOutputResponseIdentifier:			NFKModelValueTypeText,
			NFKOutputAudio:							NFKModelValueTypeAudio,
			NFKOutputMask:							NFKModelValueTypeMask,
			NFKOutputText:							NFKModelValueTypeText,
			NFKOutputStructured:					NFKModelValueTypeStructured,
			NFKOutputEmbedding:						NFKModelValueTypeEmbedding,
			NFKOutputToolCalls:						NFKModelValueTypeToolCalls,
			NFKOutputAssistantMessage:				NFKModelValueTypeJSON,
			NFKOutputMessages:						NFKModelValueTypeMessages,
			NFKOutputToolInvocations:				NFKModelValueTypeJSON,
			NFKOutputToolTurnLimitReached:			NFKModelValueTypeBoolean,
			NFKOutputDetections:					NFKModelValueTypeDetections,
			NFKOutputPose:							NFKModelValueTypeKeypoints,
			NFKOutputClassifications:				NFKModelValueTypeClassifications,
			NFKOutputSegments:						NFKModelValueTypeSegments,
			NFKOutputMIDI:							NFKModelValueTypeMIDI,
			NFKOutputBeats:							NFKModelValueTypeBeats,
			NFKOutputTempo:							NFKModelValueTypeNumber,
			NFKOutputAnswers:						NFKModelValueTypeAnswers,
			NFKOutputReasoning:						NFKModelValueTypeText,
			NFKOutputUsage:							NFKModelValueTypeJSON,
		};
	});
	return types;
}

NSString * _Nullable NFKModelValueTypeForKey(NSString *key)
{
	return NFKRequestKeyValueTypes()[key] ?: NFKResultKeyValueTypes()[key];
}

NSString * _Nullable NFKModelValueTypeForOutputKey(NSString *key)
{
	return NFKResultKeyValueTypes()[key] ?: NFKRequestKeyValueTypes()[key];
}
