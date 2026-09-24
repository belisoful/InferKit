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
/*! A document for a text model to read: an NSURL to a PDF or text file, NSData holding a PDF, an
	NSString of text, or an NFKRemoteFile the service already keeps, which rides as the service's file
	reference. A chat backend attaches it to the user turn in its provider's document shape.
	Introduced in InferKit 0.3.0. */
extern NSString * const NFKInputDocument;
/*! Further documents beside NFKInputDocument (NSArray of NSURL or NSData). Introduced in InferKit 0.3.0. */
extern NSString * const NFKInputDocuments;
/*! A control map that conditions generation (a ControlNet input: edges, depth, pose, a scribble),
	CVPixelBuffer or texture. Distinct from NFKInputImage (the base/source) and NFKInputMask. */
extern NSString * const NFKInputControl;
/*! The source clip (NFKVideoAsset). */
extern NSString * const NFKInputVideo;
/*! The image a generated clip ends on (CGImage, CVPixelBuffer, or texture). NFKInputImage is the
	first frame; the two together ask for an interpolation. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputLastFrame;
/*! A recording of the voice a speech service should speak in (NFKAudioAsset, or NSData holding an
	encoded file), for a service that clones a voice from a clip. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputVoiceReference;
/*! The text after the gap a fill-in-the-middle model writes into (NSString); NFKInputPrompt is the
	text before it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputSuffix;
/*! The source audio (NFKAudioAsset, or NSData PCM / AVAudioPCMBuffer for in-memory samples). */
extern NSString * const NFKInputAudio;
/*! Further clips beside NFKInputAudio (NSArray of NFKAudioAsset or NSData), for a question over several
	recordings. An audio backend attaches them in order after NFKInputAudio. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputAudios;
/*! The chat messages, an OpenAI-style array of {role, content} dictionaries (NSArray). A text
	backend uses this when present, otherwise it wraps NFKInputPrompt as one user message. */
extern NSString * const NFKInputMessages;
/*! The lyrics a music-generation backend sings (NSString). Structure tags such as [verse] or
	[chorus] each go on their own line. Distinct from NFKInputPrompt, which describes the music.
	Introduced in InferKit 0.2.0. */
extern NSString * const NFKInputLyrics;
/*! The state a decision model judges (NSString, or a JSON-serializable NSDictionary or NSArray): a
	message, a record, a conversation. A decision backend reads NFKInputPrompt, then NFKInputMessages,
	when this key is absent. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputState;
/*! The typed questions a decision model answers about NFKInputState (NSDictionary keyed by the
	caller's question identifiers, each an NFKDecisionQuestion or its dictionaryRepresentation). The
	answers come back under NFKOutputAnswers under the same keys. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInputQuestions;

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
/*! The shape a text reply must take (NSString): "json" constrains sampling to well-formed JSON with
	an object or array root, "json-object" to an object, "json-array" to an array. An on-device
	backend enforces it through a grammar mask (NFKJSONConstraint); the parsed reply rides under
	NFKOutputStructured. Introduced in InferKit 0.3.1. */
extern NSString * const NFKParameterOutputFormat;
/*! The strings a text reply must be exactly one of (NSArray of NSString), enforced through a grammar
	mask on device (NFKChoiceConstraint). Introduced in InferKit 0.3.1. */
extern NSString * const NFKParameterChoices;
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
/*! The output aspect ratio as width:height (NSString), for example "16:9". A backend whose service
	takes a ratio rather than a size reads it; one that takes a size derives the ratio from
	NFKParameterWidth and NFKParameterHeight when it is absent. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterAspectRatio;
/*! The output resolution tier (NSString), for example "720p" or "1080p", for a service that names
	tiers rather than sizes. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterResolution;
/*! Whether a video service generates a soundtrack with the clip (NSNumber, BOOL). Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKParameterGenerateAudio;
/*! What a video request does with its source clip (NSString): NFKVideoOperationEdit or
	NFKVideoOperationExtend. Absent, the request generates a new clip. The source is NFKInputVideo or
	NFKParameterSourceVideoIdentifier. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterVideoOperation;
/*! The service's identifier of a clip it generated earlier (NSString), the source an edit or an
	extension reads where the service takes an identifier rather than a file. Introduced in InferKit
	0.4.0. */
extern NSString * const NFKParameterSourceVideoIdentifier;

/*! Whether a transcription names who speaks in each segment (NSNumber, BOOL). The speaker lands on
	NFKAudioSegment.speaker. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterSpeakerDiarization;
/*! Whether a transcription times each word (NSNumber, BOOL). The words come back under NFKOutputWords.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterWordTimestamps;
/*! Terms a transcription should favor: names, jargon, product words (NSArray of NSString). Each
	backend sends them as its service's keyword or context-bias list. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterVocabulary;

/*! Whether a text model cites the documents it was given (NSNumber, BOOL). The citations come back
	under NFKOutputCitations. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterCitations;

/*! The identifier of an earlier reply the request continues from (NSString): the Responses API's
	previous_response_id, Gemini's previous_interaction_id. The service keeps the history, so the
	request carries only the new turn. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterPreviousResponseIdentifier;

/*! Edits the source clip according to the prompt. Introduced in InferKit 0.4.0. */
extern NSString * const NFKVideoOperationEdit;
/*! Continues the source clip according to the prompt. Introduced in InferKit 0.4.0. */
extern NSString * const NFKVideoOperationExtend;
/*! The amount of motion for image-to-video, 0 to 1 (NSNumber). */
extern NSString * const NFKParameterMotionScale;
/*! The output audio sample rate in hertz (NSNumber). */
extern NSString * const NFKParameterSampleRate;
/*! The output audio channel count (NSNumber), e.g. 1 for mono, 2 for stereo. */
extern NSString * const NFKParameterChannelCount;

/*! The language the input is in (NSString, a BCP-47 tag such as "en", "de", or "pt-BR"). Absent
	means the engine detects it, which every translator here can do. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterSourceLanguage;
/*! The language to produce (NSString, a BCP-47 tag). A translation backend requires it. Introduced
	in InferKit 0.4.0. */
extern NSString * const NFKParameterTargetLanguage;

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
/*! How much the model reasons before it answers (NSString): NFKReasoningEffortLight,
	NFKReasoningEffortModerate, or NFKReasoningEffortDeep. A backend maps the three to its provider's
	control, which may be a named level or a token budget. Any other string passes through, so a
	caller reaches a level only one provider names. A backend that cannot reason refuses the key
	rather than answering without it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKParameterReasoningEffort;

/*! The least reasoning: a short chain, for a question that needs little of it. */
extern NSString * const NFKReasoningEffortLight;
/*! The middle level, between NFKReasoningEffortLight and NFKReasoningEffortDeep. */
extern NSString * const NFKReasoningEffortModerate;
/*! The most reasoning: a long chain, for a question worth the tokens. */
extern NSString * const NFKReasoningEffortDeep;

#pragma mark Output keys

/*! The generated image (CVPixelBuffer or texture). */
extern NSString * const NFKOutputImage;
/*! Every image a request generated (NSArray of CVPixelBuffer or texture), present when there is more
	than one; NFKOutputImage holds the first. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputImages;
/*! The generated clip (NFKVideoAsset). */
extern NSString * const NFKOutputVideo;
/*! Every clip a request generated (NSArray of NFKVideoAsset), present when there is more than one;
	NFKOutputVideo holds the first. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputVideos;
/*! Each transcribed word with its time span (NSArray of NFKAudioSegment, the word as the label).
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputWords;
/*! The sources a reply cites (NSArray of NSDictionary). Each entry carries "text" (the cited span
	or the claim it supports) and whichever of "url", "title", "documentIndex", "start", and "end"
	the service reports, with the service's own record under "raw". Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputCitations;
/*! The tools the service ran itself while answering: web search, code execution, a fetch (NSArray of
	NSDictionary {name, input, output}). Distinct from NFKOutputToolCalls, which the caller runs.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputServerToolResults;
/*! The service's identifier for the reply (NSString), which a later request names under
	NFKParameterPreviousResponseIdentifier to continue from it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputResponseIdentifier;
/*! The generated audio (NFKAudioAsset, or NSData PCM / AVAudioPCMBuffer for in-memory samples). */
extern NSString * const NFKOutputAudio;
/*! A generated alpha matte or mask (CVPixelBuffer, texture, or CGImage), separate from the image. */
extern NSString * const NFKOutputMask;
/*! The generated text (NSString). */
extern NSString * const NFKOutputText;
/*! A structured result keyed by field name (NSDictionary): the fields a backend generating to a
	schema filled, or the named readings an engine produced that have no key of their own. */
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

/*! A transcribed performance (NFKMIDISequence), for a music-transcription backend that turns audio
    into notes. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputMIDI;
/*! The beats of a track (NSArray<NFKMusicBeat *>), ordered in time, for a music-structure backend.
    Its functional sections come back as NFKAudioSegments under NFKOutputSegments. Introduced in
    InferKit 0.4.0. */
extern NSString * const NFKOutputBeats;
/*! The estimated tempo in beats per minute (NSNumber), beside NFKOutputBeats. Introduced in
    InferKit 0.4.0. */
extern NSString * const NFKOutputTempo;

/*! A decision model's typed answers (NSDictionary of NFKDecisionAnswer keyed as NFKInputQuestions
	was), for a backend that judges a state rather than generating text. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputAnswers;

/*! The reasoning a model showed before its answer (NSString), separate from the answer under
	NFKOutputText. Present only where the provider returns it; a provider that hides its reasoning
	reports the tokens it spent under NFKOutputUsage and nothing here. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputReasoning;

/*! What the turn cost in tokens (NSDictionary of NSNumber keyed by NFKUsageInputTokens,
	NFKUsageCachedTokens, NFKUsageOutputTokens, and NFKUsageReasoningTokens). A backend fills the
	counts its provider reports and leaves the rest out, so a caller reads a key that is present
	rather than trusting a zero. Introduced in InferKit 0.4.0. */
extern NSString * const NFKOutputUsage;

/*! The tokens the request cost, the cached ones included (NSNumber). */
extern NSString * const NFKUsageInputTokens;
/*! How many of the input tokens were served from the provider's cache (NSNumber). */
extern NSString * const NFKUsageCachedTokens;
/*! The tokens the reply cost, the reasoning ones included (NSNumber). */
extern NSString * const NFKUsageOutputTokens;
/*! How many of the output tokens went to reasoning (NSNumber). */
extern NSString * const NFKUsageReasoningTokens;

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceKeys_h */
