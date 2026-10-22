//
//  NFKValueCoding.h
//  InferKit
//

#ifndef NFKValueCoding_h
#define NFKValueCoding_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKValueCoding
	@abstract   Requests, results, and single values as JSON, inline or with their large values in files.
	@discussion The JSON is the tagged form the native route (/v1/inferkit/run) carries. A JSON string,
				number, boolean, null, array, or dictionary passes as itself, and every other value is an
				object tagged under "$nfk": images, pixel buffers, and textures; PCM buffers; audio, video,
				and other files; Core ML multi-arrays; packed float arrays; and InferKit's value classes
				(NFKDetection, NFKKeypoint, NFKClassification, NFKAudioSegment, NFKMIDISequence,
				NFKMIDINote, NFKMusicBeat, NFKQuadrilateral, NFKDecisionAnswer, NFKDecisionQuestion, tool
				results, and tool invocations). The value classes are written as JSON objects of their
				fields. A value the form does not carry fails the encode, naming its key and class.

				Two forms:
				- init → the inline form. Every value's bytes travel inside the JSON, base64 where they are
				  not text. Decoding writes a file value's bytes to a temporary file that
				  temporaryFileURLs lists.
				- initWithDirectoryURL: → the file form. Each value a person or a program opens as a file
				  is written into the directory, and the JSON names it by its path relative to the
				  directory:
				  - an image, mask, or other pixel buffer → a PNG (exact for 8-bit BGRA, ARGB, and
				    one-channel buffers; a buffer of another format → a .bin of its planes);
				  - a PCM buffer → a WAV of 16- or 32-bit integer or 32- or 64-bit float samples;
				  - an audio, video, or other file → that file, copied in when it lies outside;
				  - an NFKMIDISequence → its JSON object, beside a Standard MIDI File of it;
				  - data, a float array, or a multi-array above inlineByteLimit → a .bin, or a NumPy .npy.
				  Files are named after their key, such as "image.png" and "image-2.png". Decoding reads
				  them in place and refuses an absolute path or one that leaves the directory.

				The inline form opens any absolute path its JSON names as a file value, so an app decodes JSON
				it did not write in the file form, which reads only inside its directory. Either form refuses
				an image that declares more than 8192 × 8192 pixels before decoding it.

				valueOfType:fromFileURL:error: and valueOfType:fromJSONValue:error: turn a file or a JSON
				value into what a model port of that NFKModelValueType takes, so a person's file becomes a
				request input without a converter per type. Introduced in InferKit 0.4.0.
*/
@interface NFKValueCoding : NSObject

/*! A coder of the inline form. Introduced in InferKit 0.4.0. */
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/*! A coder of the file form, which writes and reads its files in directoryURL. The directory must exist.
	Introduced in InferKit 0.4.0. */
- (instancetype)initWithDirectoryURL:(NSURL *)directoryURL NS_DESIGNATED_INITIALIZER;

/*! The file form's directory, or nil for the inline form. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy, nullable) NSURL *directoryURL;

/*! The largest data, float array, or multi-array the file form keeps in the JSON, in bytes. Defaults to
	65536. Images, PCM buffers, files, and MIDI sequences are written to files whatever their size.
	Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSUInteger inlineByteLimit;

/*! The temporary files the inline form wrote while decoding file values, which the caller removes when
	it is done with the values. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSURL *> *temporaryFileURLs;

/*! A request as {"inputs", "parameters", "outputModality"}, or nil with an error. Introduced in InferKit 0.4.0. */
- (nullable NSDictionary<NSString *, id> *)JSONObjectForRequest:(NFKInferenceRequest *)request
														  error:(NSError * _Nullable *)error;

/*! The request a JSON object describes, or nil with an error. Introduced in InferKit 0.4.0. */
- (nullable NFKInferenceRequest *)requestFromJSONObject:(id)object error:(NSError * _Nullable *)error;

/*! A result as {"outputs"}, or nil with an error. Introduced in InferKit 0.4.0. */
- (nullable NSDictionary<NSString *, id> *)JSONObjectForResult:(NFKInferenceResult *)result
														 error:(NSError * _Nullable *)error;

/*! The result a JSON object describes, or nil with an error. Introduced in InferKit 0.4.0. */
- (nullable NFKInferenceResult *)resultFromJSONObject:(id)object error:(NSError * _Nullable *)error;

/*! One value in the tagged form. key names the value in an error and its file in the file form.
	Introduced in InferKit 0.4.0. */
- (nullable id)JSONValueForValue:(id)value key:(NSString *)key error:(NSError * _Nullable *)error;

/*! The value a tagged JSON value describes. key names the value in an error. Introduced in InferKit 0.4.0. */
- (nullable id)valueFromJSONValue:(id)JSONValue key:(NSString *)key error:(NSError * _Nullable *)error;

/*!
	@method     valueOfType:fromFileURL:error:
	@abstract   The value a port of valueType takes, read from the file at url.
	@discussion A file whose extension is "json" is read as JSON and converted by
				valueOfType:fromJSONValue:error:. Otherwise:
				- image, mask → a 32BGRA pixel buffer of any image ImageIO reads; a PNG's 8-bit samples
				  are read as stored. images → an array of one.
				- audio → an NFKAudioAsset over the file, with the duration, sample rate, and channel count
				  AVAudioFile reads. audios → an array of one.
				- video → an NFKVideoAsset over the file. videos → an array of one.
				- document → the file URL. documents → an array of one.
				- text → the file's UTF-8 text. text-list → its non-empty lines.
				- number, integer, boolean → the file's text read as one.
				- embedding → a vector .npy of floats, as an array of numbers.
				- midi → an NFKMIDISequence read from a Standard MIDI File.
				Any other type takes JSON only. A file that is not of the type fails with
				kNFKError_InferenceUnsupported in NFKInferenceErrorDomain. Introduced in InferKit 0.4.0.
*/
- (nullable id)valueOfType:(NSString *)valueType fromFileURL:(NSURL *)url error:(NSError * _Nullable *)error;

/*!
	@method     valueOfType:fromJSONValue:error:
	@abstract   The value a port of valueType takes, from a JSON value.
	@discussion A tagged value decodes as valueFromJSONValue:key:error: decodes it. Untagged JSON converts by
				type:
				- text → a string. text-list → an array of strings.
				- number, integer, boolean → a number, from a JSON number, a boolean, or text.
				- embedding → an array of numbers.
				- detections, keypoints, classifications, segments, words, beats → an array of the value
				  class, each from the JSON object of its fields; midi → an NFKMIDISequence from its
				  object.
				- questions → NFKDecisionQuestion values from their dictionaries, by name; answers →
				  NFKDecisionAnswer values from their objects, by name.
				- messages, json, structured, tool-calls → the JSON as it is.
				- image, mask, images → a pixel buffer from a "data:" URL string.
				- audio, video, document and their lists → a file URL from a path, relative to the file
				  form's directory or absolute in the inline form.
				Introduced in InferKit 0.4.0.
*/
- (nullable id)valueOfType:(NSString *)valueType fromJSONValue:(id)JSONValue error:(NSError * _Nullable *)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKValueCoding_h */
