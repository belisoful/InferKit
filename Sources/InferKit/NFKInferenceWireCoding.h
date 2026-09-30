//
//  NFKInferenceWireCoding.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>

NS_ASSUME_NONNULL_BEGIN

/*! A client-named file extension made safe to write: one to eight letters and digits, lowercased, or
	"bin". A client's "../x" would otherwise leave the temporary directory. */
NSString *NFKServedFileExtension(NSString * _Nullable extension);

/*!
	@class      NFKInferenceWireCoding
	@abstract   Requests, results, and errors as JSON, for InferKit's native serving route.
	@discussion NFKInferenceServer and NFKRemoteInferKitBackend both speak this form, so the two ends
				agree by construction. JSON strings, numbers, booleans, null, arrays, and dictionaries
				with string keys pass as themselves. Every other value is an object tagged under
				"$nfk":
				- data, date, url → the bytes, the seconds since 1970, the absolute string.
				- file (a file NSURL), audio (NFKAudioAsset), video (NFKVideoAsset) → the file's bytes
				  and name, with the asset's metadata; the decoder writes the bytes to a temporary
				  file and records its URL in temporaryFileURLs.
				- pixelBuffer (CVPixelBuffer) → the pixel format, the dimensions, and each plane's rows
				  packed, so a float depth map crosses exactly. cgImage → PNG. A Metal texture → PNG,
				  decoded as a 32BGRA pixel buffer, since the far side has no device to make one on.
				- multiArray (MLMultiArray) → the data type, the shape, and the elements in row-major
				  order. pcmBuffer (AVAudioPCMBuffer) → the stream description and each buffer's bytes.
				- decisionQuestion → NFKDecisionQuestion's dictionary representation.
				- archive → a keyed archive of one of the core's secure-coding value types (NFKDetection,
				  NFKKeypoint, NFKClassification, NFKAudioSegment, NFKMIDISequence, NFKMIDINote,
				  NFKMusicBeat, NFKQuadrilateral, NFKDecisionAnswer), decoded against that allowlist only.
				- float32Array, float64Array → an array whose every element is a finite float (or
				  double), packed as little-endian words, so an embedding crosses bit for bit.
				- double → a floating value that NSJSONSerialization would not return identical, as its
				  17 significant digits. number → a non-finite double, which JSON cannot spell.
				- dictionary → a dictionary that itself carries an "$nfk" key, so it is never read as a tag.
				A value of any other class fails the encode, naming its key and class.
*/
@interface NFKInferenceWireCoding : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! A coder whose decoded files land in directory, or in the InferKit temporary directory when nil. */
- (instancetype)initWithTemporaryDirectory:(nullable NSURL *)directory NS_DESIGNATED_INITIALIZER;

/*! The files this coder wrote while decoding, for the caller that owns their lifetime. */
@property (nonatomic, readonly, copy) NSArray<NSURL *> *temporaryFileURLs;

- (nullable id)JSONObjectForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError;
- (nullable NFKInferenceRequest *)requestFromJSONObject:(nullable id)object error:(NSError * _Nullable *)outError;
- (nullable id)JSONObjectForResult:(NFKInferenceResult *)result error:(NSError * _Nullable *)outError;
- (nullable NFKInferenceResult *)resultFromJSONObject:(nullable id)object error:(NSError * _Nullable *)outError;

/*! One value in the tagged form, for a caller that encodes outside a request or result. */
- (nullable id)JSONValueForValue:(id)value key:(NSString *)key error:(NSError * _Nullable *)outError;
- (nullable id)valueFromJSONValue:(id)JSONValue key:(NSString *)key error:(NSError * _Nullable *)outError;

/*! The error envelope both serving surfaces answer with: an OpenAI-shaped {error: {message, type,
	code}} that also carries the NSError's domain, code, and its string and number userInfo entries. */
+ (NSDictionary<NSString *, id> *)JSONObjectForError:(NSError *)error;

/*! The NSError an envelope describes, or nil when the object is not one. An envelope from another
	server, with no domain in it, answers nil. */
+ (nullable NSError *)errorFromJSONObject:(nullable id)object;

@end

NS_ASSUME_NONNULL_END
