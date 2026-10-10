//
//  NFKGeminiVoiceLibrary.h
//  InferKit
//

#ifndef NFKGeminiVoiceLibrary_h
#define NFKGeminiVoiceLibrary_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKRemoteSpeechBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKGeminiVoiceLibrary
	@abstract   The Gemini API's voice library: lists the prebuilt and custom voices, designs a voice from
				a description, replicates one from a recording, and deletes a stored one.
	@discussion /v1beta/voices. Every voice comes back as an NFKRemoteVoice whose identifier is what
				NFKGeminiInteractionsBackend takes as its voice (or a "voice" request parameter) to
				speak with it on the TTS models:

				- a prebuilt voice → its name, for example Kore
				- a stored custom voice → its voice_… id, kept by Google for a year after its last use
				- a replicated voice made with stores off → its voicekey_… key, which the caller keeps
				  and which expires after seven days

				A designed voice and a voice looked up by identifier carry a WAV preview under
				sampleAudioData; a listing omits it. A project keeps at most 200 stored custom voices.
				Gemini speaks a custom voice in a single-speaker reply only; two speakers take
				prebuilt voices. Every call blocks for its round trip, so run it off the main or render
				thread. Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiVoiceLibrary : NSObject

/*! The voices collection. Defaults to https://generativelanguage.googleapis.com/v1beta/voices. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as x-goog-api-key. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model that designs or replicates a voice, for example gemini-3.8-flash-tts. nil lets the
	service choose its latest voice-design model. A created voice speaks on every TTS model. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Seconds before a request times out. Defaults to 60, because designing a voice renders its preview. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A library at the gemini preset's voices collection, at its own base or a base it was re-pointed at;
	every other preset returns nil. */
+ (nullable instancetype)voiceLibraryForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*!
	@method     voicesMatchingFilters:error:
	@abstract   The voices that match the filters, every page of them, blocking.
	@discussion Each filter key is a query parameter the service reads, and its value is a string or
				an array of strings, which match as alternatives:

				- type → prebuilt, prompted, or replicated
				- language_code, region_code, accent, gender, pitch, persona, context
				- search → a case-insensitive substring of the name or description
				- page_size → voices a page, 50 unless given, at most 1000

				Different keys must all match. Custom voices come first, newest first, then the
				prebuilt ones. nil filters list every voice.
*/
- (nullable NSArray<NFKRemoteVoice *> *)voicesMatchingFilters:(nullable NSDictionary<NSString *, id> *)filters
														error:(NSError * _Nullable *)outError;

/*! One voice by its identifier, with its preview where the service keeps one. */
- (nullable NFKRemoteVoice *)voiceWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*!
	@method     designVoiceWithDescription:displayName:attributes:error:
	@abstract   Creates and stores a voice from a description of how it sounds.
	@discussion description is the prompt, for example "a warm astronomer in his late sixties with a
				gentle British accent". attributes carries the voice's optional discovery fields by
				their wire names (gender, language_code, accent, persona, pitch, context,
				region_code, description); type, prompted, and replicated are the library's own and
				are ignored there. The voice is always stored, because the service stores every
				designed voice. The result carries its id and a WAV preview.
*/
- (nullable NFKRemoteVoice *)designVoiceWithDescription:(NSString *)description
											displayName:(nullable NSString *)displayName
											 attributes:(nullable NSDictionary<NSString *, id> *)attributes
												  error:(NSError * _Nullable *)outError;

/*!
	@method     replicateVoiceFromAudio:consentAudio:displayName:stores:error:
	@abstract   Creates a voice that sounds like the speaker in a recording.
	@discussion sourceAudio is 10 to 30 seconds of the speaker's clean, natural speech. consentAudio is
				the same adult speaker reading the service's consent statement for their language;
				the service checks that the two voices match. Each is an NSData or a file NSURL in
				WAV, MP3, or FLAC; 24 kHz mono 16-bit WAV, recorded on one microphone in one room, is
				what the service recommends. With stores set the voice is kept by Google and the
				result's identifier is its voice_… id; with it off the identifier is a voicekey_… key
				the caller keeps, and displayName is not sent.
*/
- (nullable NFKRemoteVoice *)replicateVoiceFromAudio:(id)sourceAudio
										consentAudio:(id)consentAudio
										 displayName:(nullable NSString *)displayName
											  stores:(BOOL)stores
											   error:(NSError * _Nullable *)outError;

/*! Deletes a stored custom voice. */
- (BOOL)deleteVoiceWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGeminiVoiceLibrary_h */
