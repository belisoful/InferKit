//
//  NFKInferenceKeys.m
//  InferKit
//

#import "NFKInferenceKeys.h"

NSString * const NFKInputPrompt				= @"prompt";
NSString * const NFKInputNegativePrompt		= @"negativePrompt";
NSString * const NFKInputImage				= @"image";
NSString * const NFKInputMask				= @"mask";
NSString * const NFKInputControl			= @"control";
NSString * const NFKInputVideo				= @"video";
NSString * const NFKInputAudio				= @"audio";
NSString * const NFKInputAudios				= @"audios";
NSString * const NFKInputMessages			= @"messages";
NSString * const NFKInputImages				= @"images";
NSString * const NFKInputDocument			= @"document";
NSString * const NFKInputDocuments			= @"documents";
NSString * const NFKInputLyrics				= @"lyrics";
NSString * const NFKInputLastFrame			= @"lastFrame";
NSString * const NFKInputVoiceReference		= @"voiceReference";
NSString * const NFKInputSuffix				= @"suffix";
NSString * const NFKInputState				= @"state";
NSString * const NFKInputQuestions			= @"questions";

NSString * const NFKParameterSeed			= @"seed";
NSString * const NFKParameterTools			= @"tools";
NSString * const NFKParameterJSONSchema		= @"jsonSchema";
NSString * const NFKParameterOutputFormat		= @"outputFormat";
NSString * const NFKParameterChoices			= @"choices";
NSString * const NFKParameterVideoFrameCount	= @"videoFrameCount";
NSString * const NFKParameterAudioOutput		= @"audioOutput";
NSString * const NFKParameterSteps			= @"steps";
NSString * const NFKParameterGuidanceScale	= @"guidanceScale";
NSString * const NFKParameterStrength		= @"strength";
NSString * const NFKParameterWidth			= @"width";
NSString * const NFKParameterHeight			= @"height";
NSString * const NFKParameterSampleCount	= @"sampleCount";
NSString * const NFKParameterFrameCount		= @"frameCount";
NSString * const NFKParameterFramesPerSecond = @"framesPerSecond";
NSString * const NFKParameterDurationSeconds = @"durationSeconds";
NSString * const NFKParameterMotionScale	= @"motionScale";
NSString * const NFKParameterAspectRatio	= @"aspectRatio";
NSString * const NFKParameterResolution		= @"resolution";
NSString * const NFKParameterGenerateAudio	= @"generateAudio";
NSString * const NFKParameterVideoOperation	= @"videoOperation";
NSString * const NFKParameterSourceVideoIdentifier = @"sourceVideoIdentifier";
NSString * const NFKParameterSpeakerDiarization = @"speakerDiarization";
NSString * const NFKParameterWordTimestamps	= @"wordTimestamps";
NSString * const NFKParameterVocabulary		= @"vocabulary";
NSString * const NFKParameterCitations		= @"citations";
NSString * const NFKParameterPreviousResponseIdentifier = @"previousResponseIdentifier";
NSString * const NFKVideoOperationEdit		= @"edit";
NSString * const NFKVideoOperationExtend	= @"extend";
NSString * const NFKParameterSampleRate		= @"sampleRate";
NSString * const NFKParameterChannelCount	= @"channelCount";

NSString * const NFKParameterSourceLanguage	= @"sourceLanguage";
NSString * const NFKParameterTargetLanguage	= @"targetLanguage";
NSString * const NFKParameterTemperature		= @"temperature";
NSString * const NFKParameterTopP				= @"topP";
NSString * const NFKParameterTopK				= @"topK";
NSString * const NFKParameterMaxTokens			= @"maxTokens";
NSString * const NFKParameterRepetitionPenalty	= @"repetitionPenalty";
NSString * const NFKParameterStopSequences		= @"stopSequences";
NSString * const NFKParameterReasoningEffort		= @"reasoningEffort";

NSString * const NFKReasoningEffortLight		= @"light";
NSString * const NFKReasoningEffortModerate		= @"moderate";
NSString * const NFKReasoningEffortDeep			= @"deep";

NSString * const NFKOutputImage				= @"image";
NSString * const NFKOutputVideo				= @"video";
NSString * const NFKOutputVideos			= @"videos";
NSString * const NFKOutputImages			= @"images";
NSString * const NFKOutputWords				= @"words";
NSString * const NFKOutputCitations			= @"citations";
NSString * const NFKOutputServerToolResults	= @"serverToolResults";
NSString * const NFKOutputResponseIdentifier	= @"responseIdentifier";
NSString * const NFKOutputAudio				= @"audio";
NSString * const NFKOutputMask				= @"mask";
NSString * const NFKOutputText				= @"text";
NSString * const NFKOutputStructured		= @"structured";
NSString * const NFKOutputEmbedding			= @"embedding";
NSString * const NFKOutputToolCalls			= @"toolCalls";
NSString * const NFKOutputDetections		= @"detections";
NSString * const NFKOutputPose				= @"pose";
NSString * const NFKOutputClassifications	= @"classifications";
NSString * const NFKOutputSegments			= @"segments";
NSString * const NFKOutputMIDI				= @"midi";
NSString * const NFKOutputBeats				= @"beats";
NSString * const NFKOutputTempo				= @"tempo";
NSString * const NFKOutputReasoning			= @"reasoning";
NSString * const NFKOutputAnswers			= @"answers";
NSString * const NFKOutputUsage				= @"usage";

NSString * const NFKUsageInputTokens		= @"inputTokens";
NSString * const NFKUsageCachedTokens		= @"cachedTokens";
NSString * const NFKUsageOutputTokens		= @"outputTokens";
NSString * const NFKUsageReasoningTokens	= @"reasoningTokens";
