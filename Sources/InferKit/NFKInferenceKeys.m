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
NSString * const NFKInputMessages			= @"messages";
NSString * const NFKInputImages				= @"images";
NSString * const NFKInputDocument			= @"document";
NSString * const NFKInputDocuments			= @"documents";
NSString * const NFKInputLyrics				= @"lyrics";

NSString * const NFKParameterSeed			= @"seed";
NSString * const NFKParameterTools			= @"tools";
NSString * const NFKParameterJSONSchema		= @"jsonSchema";
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
NSString * const NFKParameterSampleRate		= @"sampleRate";
NSString * const NFKParameterChannelCount	= @"channelCount";

NSString * const NFKParameterTemperature		= @"temperature";
NSString * const NFKParameterTopP				= @"topP";
NSString * const NFKParameterTopK				= @"topK";
NSString * const NFKParameterMaxTokens			= @"maxTokens";
NSString * const NFKParameterRepetitionPenalty	= @"repetitionPenalty";
NSString * const NFKParameterStopSequences		= @"stopSequences";

NSString * const NFKOutputImage				= @"image";
NSString * const NFKOutputVideo				= @"video";
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
