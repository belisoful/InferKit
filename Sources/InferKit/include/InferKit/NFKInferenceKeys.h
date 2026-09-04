//
//  NFKInferenceKeys.h
//  InferKit
//

#ifndef NFKInferenceKeys_h
#define NFKInferenceKeys_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKInferenceKeys
	@abstract   Well-known request input, parameter, and result output keys for generative models.
	@discussion A request's inputs and parameters are open dictionaries,
				so any control works, but a consumer and a backend must agree on names. These
				constants are the shared vocabulary for the common text-to-image, image-to-image,
				video, audio, and text-generation controls. A backend maps them to its provider's own names (steps to
				num_inference_steps, maxTokens to max_tokens, and so on) and ignores what it does not use.

				Prompts are inputs because they are conditioning content; scalar controls are
				parameters. Values are the natural types: prompts are NSString, counts and scales
				are NSNumber, an image or video input is a CVPixelBuffer, a texture, or an
				NFKVideoAsset, and an audio input is an NFKAudioAsset.
*/

#pragma mark Input keys (conditioning and media)

/*! The positive prompt (NSString). */
extern NSString * const NFKInputPrompt;
/*! The negative prompt, describing what to avoid (NSString). */
extern NSString * const NFKInputNegativePrompt;
/*! The source or reference image (CVPixelBuffer or texture). */
extern NSString * const NFKInputImage;
/*! The inpaint or region mask (CVPixelBuffer or texture). */
extern NSString * const NFKInputMask;
/*! Further images beside NFKInputImage (NSArray of CGImage, CVPixelBuffer, or texture), for a question
	over several frames. A vision backend attaches them in order after NFKInputImage. Introduced in
	InferKit 0.3.0. */
extern NSString * const NFKInputImages;
/*! A document for a text model to read (an NSURL to a PDF file, or NSData holding one). A chat backend
	attaches it to the user turn in its provider's document shape. Introduced in InferKit 0.3.0. */
extern NSString * const NFKInputDocument;
/*! Further documents beside NFKInputDocument (NSArray of NSURL or NSData). Introduced in InferKit 0.3.0. */
extern NSString * const NFKInputDocuments;
/*! A control map that conditions generation (a ControlNet input: edges, depth, pose, a scribble),
	CVPixelBuffer or texture. Distinct from NFKInputImage (the base/source) and NFKInputMask. */
extern NSString * const NFKInputControl;
/*! The source clip (NFKVideoAsset). */
extern NSString * const NFKInputVideo;
/*! The source audio (NFKAudioAsset, or NSData PCM / AVAudioPCMBuffer for in-memory samples). */
extern NSString * const NFKInputAudio;
/*! The chat messages, an OpenAI-style array of {role, content} dictionaries (NSArray). A text
	backend uses this when present, otherwise it wraps NFKInputPrompt as one user message. */
extern NSString * const NFKInputMessages;
/*! The lyrics a music-generation backend sings (NSString). Structure tags such as [verse] or
	[chorus] each go on their own line. Distinct from NFKInputPrompt, which describes the music.
	Introduced in InferKit 0.2.0. */
extern NSString * const NFKInputLyrics;

#pragma mark Parameter keys (scalar controls)

/*! The random seed for reproducibility (NSNumber). */
extern NSString * const NFKParameterSeed;
/*! The number of diffusion steps (NSNumber). */
extern NSString * const NFKParameterSteps;
/*! The classifier-free guidance scale (NSNumber). */
extern NSString * const NFKParameterGuidanceScale;
/*! The image-to-image denoising strength, 0 to 1 (NSNumber). */
extern NSString * const NFKParameterStrength;
/*! The output width in pixels (NSNumber). */
extern NSString * const NFKParameterWidth;
/*! The output height in pixels (NSNumber). */
extern NSString * const NFKParameterHeight;
/*! The tools a text model may call (NSArray of NSDictionary), each {name, description, parameters}
	with parameters a JSON Schema object. A backend wraps them in its provider's wire shape and returns
	what the model called under NFKOutputToolCalls. Introduced in InferKit 0.3.0. */
extern NSString * const NFKParameterTools;
/*! A JSON Schema (NSDictionary) the text reply must conform to. The backend asks the provider for it
	in the provider's way and returns the parsed reply under NFKOutputStructured. Introduced in
	InferKit 0.3.0. */
extern NSString * const NFKParameterJSONSchema;
/*! How many frames a chat backend samples from NFKInputVideo, evenly spaced, to show a vision model
	(NSNumber; default 8). Introduced in InferKit 0.3.0. */
extern NSString * const NFKParameterVideoFrameCount;
/*! Asks a chat model to answer in speech as well as text (NSDictionary {voice, format}; format wav by
	default). The spoken reply comes back as an NFKAudioAsset under NFKOutputAudio beside the text.
	Introduced in InferKit 0.3.0. */
extern NSString * const NFKParameterAudioOutput;
/*! The number of outputs to generate (NSNumber). */
extern NSString * const NFKParameterSampleCount;
/*! The number of video frames to generate (NSNumber). */
extern NSString * const NFKParameterFrameCount;
/*! The output frame rate (NSNumber). */
extern NSString * const NFKParameterFramesPerSecond;
/*! The output duration in seconds (NSNumber). */
extern NSString * const NFKParameterDurationSeconds;
/*! The amount of motion for image-to-video, 0 to 1 (NSNumber). */
extern NSString * const NFKParameterMotionScale;
/*! The output audio sample rate in hertz (NSNumber). */
extern NSString * const NFKParameterSampleRate;
/*! The output audio channel count (NSNumber), e.g. 1 for mono, 2 for stereo. */
extern NSString * const NFKParameterChannelCount;

#pragma mark Parameter keys (text generation)

/*! The sampling temperature (NSNumber). 0 selects the most likely token (greedy). */
extern NSString * const NFKParameterTemperature;
/*! The nucleus-sampling probability mass to keep, 0 to 1 (NSNumber). Sampling draws from the
	smallest set of tokens whose probabilities sum to this value. */
extern NSString * const NFKParameterTopP;
/*! The top-k sampling cutoff (NSNumber). Sampling draws from the k most likely tokens; 0 disables
	the cutoff. When both topK and topP are set, topK applies first. */
extern NSString * const NFKParameterTopK;
/*! The maximum number of tokens to generate (NSNumber). */
extern NSString * const NFKParameterMaxTokens;
/*! The penalty dividing the score of tokens already generated, discouraging repetition (NSNumber).
	1 disables it; values above 1 penalize more. */
extern NSString * const NFKParameterRepetitionPenalty;
/*! The stop sequences (NSArray<NSString *>). Generation ends when the output ends with one of them. */
extern NSString * const NFKParameterStopSequences;

#pragma mark Output keys

/*! The generated image (CVPixelBuffer or texture). */
extern NSString * const NFKOutputImage;
/*! The generated clip (NFKVideoAsset). */
extern NSString * const NFKOutputVideo;
/*! The generated audio (NFKAudioAsset, or NSData PCM / AVAudioPCMBuffer for in-memory samples). */
extern NSString * const NFKOutputAudio;
/*! A generated alpha matte or mask (CVPixelBuffer, texture, or CGImage), separate from the image. */
extern NSString * const NFKOutputMask;
/*! The generated text (NSString). */
extern NSString * const NFKOutputText;
/*! A structured result keyed by field name (NSDictionary), for a backend that generates to a schema. */
extern NSString * const NFKOutputStructured;
/*! A feature embedding vector (NSArray<NSNumber *> of floats), for a model that encodes an image or
    text into a shared representation. An encoder that L2-normalizes returns a unit vector, so a
    consumer compares two embeddings by dot product. */
extern NSString * const NFKOutputEmbedding;
/*! The tool calls a text model made (NSArray of NSDictionary), each {id, name, arguments} with
	arguments the parsed argument object, beside argumentsJSON, the provider's own text of it. Present
	only when the model called a tool. Introduced in InferKit 0.3.0. */
extern NSString * const NFKOutputToolCalls;
/*! Detected objects (NSArray<NFKDetection *>), for an object-detection backend. Each box is normalized
    to the input image. */
extern NSString * const NFKOutputDetections;
/*! Located landmarks (NSArray<NFKKeypoint *>), for a pose-estimation backend. Each position is
    normalized to the input image. */
extern NSString * const NFKOutputPose;
/*! Predicted classes (NSArray<NFKClassification *>), for a classification or tagging backend, ordered
    most-confident first. */
extern NSString * const NFKOutputClassifications;
/*! Time spans (NSArray<NFKAudioSegment *>), for a backend that locates events over time (voice-activity
    or sound-event detection). */
extern NSString * const NFKOutputSegments;

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceKeys_h */
