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

@class NFKInferenceWireCoding;

/*!
	@protocol   NFKWireAttachmentSink
	@abstract   Where an encoder in the attachment mode puts the bytes of a value too large to inline.
	@discussion The plugin channel implements it with shared-memory regions. An index names the
				attachment in the entry, and the frame carries the attachments in index order.
*/
@protocol NFKWireAttachmentSink <NSObject>

/*! A writable region of length bytes for one new attachment, with its index in outIndex, or NULL with
	an error. The region stays valid while the sink lives. */
- (nullable void *)attachmentRegionOfLength:(NSUInteger)length
									  index:(NSInteger *)outIndex
									  error:(NSError * _Nullable *)outError;

@optional

/*! Attaches an open descriptor of the file at url and sets outIndex to its index, or sets outIndex to
	-1 to pass the file by path alone. Returns NO with an error when the file cannot be opened. */
- (BOOL)attachFileAtURL:(NSURL *)url index:(NSInteger *)outIndex error:(NSError * _Nullable *)outError;

@end

/*!
	@protocol   NFKWireAttachmentSource
	@abstract   Where a decoder in the attachment mode reads the attachments an entry indexes.
*/
@protocol NFKWireAttachmentSource <NSObject>

/*! The region that holds the attachment at index, valid while the returned data lives, or nil with an
	error. The region may run past the attachment, as a shared-memory object runs to a page boundary; the
	decoder reads the entry's "length" bytes from its start. */
- (nullable NSData *)attachmentDataAtIndex:(NSInteger)index error:(NSError * _Nullable *)outError;

@optional

/*! A file URL with the contents of the file attachment at index, named after name, or nil with an
	error. */
- (nullable NSURL *)fileURLForAttachmentAtIndex:(NSInteger)index
										   name:(NSString *)name
										  error:(NSError * _Nullable *)outError;

/*! The file URL a reference by path names, or nil with an error when the source refuses the path. */
- (nullable NSURL *)fileURLForPath:(NSString *)path error:(NSError * _Nullable *)outError;

@end

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
				  decoded as a 32BGRA pixel buffer, since the far side has no device to make one on. A
				  texture PNG cannot hold, such as a half or float image → a pixelBuffer of its format,
				  over its IOSurface or read from the CPU.
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
				- numberKeyedDictionary → a dictionary whose every key is a number, such as a Core ML
				  classifier's probabilities by class index, with each key as its decimal text.
				- An MLFeatureValue → the value it holds (a number, a string, a pixel buffer, a
				  multi-array, a dictionary, or an array), which is what the decoder returns.
				- detection, keypoint, classification, audioSegment, midiSequence, midiNote, musicBeat,
				  quadrilateral, decisionAnswer → InferKit's value classes as JSON objects, written in
				  place of archive while encodesValueObjectsAsJSON is set (NFKWireValueObjects.h).
				A value of any other class fails the encode, naming its key and class.

				The attachment mode carries a large value's bytes beside the JSON. It is on for encoding
				while attachmentSink is set and for decoding while attachmentSource is set; the HTTP
				route sets neither, so its JSON is unchanged.
				- A value whose bytes exceed inlineByteLimit → {"$nfk": "buffer", "index", "as",
				  "length", …}, with "as" one of data, float32Array, float64Array, multiArray,
				  pixelBuffer, pcm, cgImage, and image. A pixel buffer keeps each plane's row padding
				  and names each plane's offset; a PCM buffer names each buffer's offset.
				- A file, audio, or video value → its tag with "path" and "name" in place of "base64",
				  and "index" when the sink attaches an open descriptor of the file.
				The decoder wraps each attachment's bytes without a copy where the value type allows.
*/
@interface NFKInferenceWireCoding : NSObject

- (instancetype)init NS_UNAVAILABLE;

/*! A coder whose decoded files land in directory, or in the InferKit temporary directory when nil. */
- (instancetype)initWithTemporaryDirectory:(nullable NSURL *)directory NS_DESIGNATED_INITIALIZER;

/*! The sink that turns on the attachment mode for encoding, or nil. */
@property (nonatomic, strong, nullable) id<NFKWireAttachmentSink> attachmentSink;

/*! The source that turns on the attachment mode for decoding, or nil. */
@property (nonatomic, strong, nullable) id<NFKWireAttachmentSource> attachmentSource;

/*! The largest value the attachment mode inlines, in bytes. Defaults to 65536. */
@property (nonatomic, assign) NSUInteger inlineByteLimit;

/*! Writes InferKit's value classes (NFKDetection, NFKKeypoint, NFKClassification, NFKAudioSegment,
	NFKMIDISequence, NFKMIDINote, NFKMusicBeat, NFKQuadrilateral, NFKDecisionAnswer) as JSON objects
	tagged by class, the form NFKWireValueObjects.h lists, in place of keyed archives. The C interface
	and the plugin channel set it. Off by default, so the HTTP route keeps its archives. The decoder
	reads both forms whatever the setting. */
@property (nonatomic, assign) BOOL encodesValueObjectsAsJSON;

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
