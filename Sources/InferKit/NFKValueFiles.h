//
//  NFKValueFiles.h
//  InferKit
//
//  The file formats NFKValueCoding's file form writes and reads: WAV for PCM buffers, NumPy's .npy for
//  packed arrays, and PNG for 8-bit pixel buffers, each read back to the value it was written from.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/*! A new, empty file in directory named after stem and extension: stem made safe (letters, digits, "-",
	and "_", at most 40 characters, "value" when nothing is left), and "-2", "-3", … added until the name
	is free. The file is created exclusively, so two coders writing into one directory never share a
	name. */
NSURL * _Nullable NFKValueFileCreate(NSURL *directory, NSString *stem, NSString *extension, NSError * _Nullable * _Nullable error);

/*! The file a path names inside directory, or nil with an error when the path is absolute, the file is
	missing, or the path resolves outside directory, through links included. */
NSURL * _Nullable NFKValueFileResolve(NSURL *directory, NSString *path, NSError * _Nullable * _Nullable error);

/*! The path of url relative to directory when url lies inside it, through links included, or nil. */
NSString * _Nullable NFKValueFileRelativePath(NSURL *directory, NSURL *url);

/*! Whether WAV holds the buffer's samples as they are: 16- or 32-bit integers, or 32- or 64-bit floats,
	little-endian. */
BOOL NFKValueFileWAVHoldsFormat(AVAudioFormat *format);

/*! The buffer's frames as a WAV file, interleaved, or nil with an error. */
NSData * _Nullable NFKValueFileWAVData(AVAudioPCMBuffer *buffer, NSError * _Nullable * _Nullable error);

/*! The frames of a WAV file in a buffer of format, which the file's sample format, rate, and channel
	count must match, or nil with an error. A non-interleaved format takes the frames apart per channel. */
AVAudioPCMBuffer * _Nullable NFKValueFilePCMBufferFromWAV(NSData *data, AVAudioFormat *format, NSError * _Nullable * _Nullable error);

/*! The NumPy type string of a little-endian element: "<f4", "<f8", "<f2", "<i4", "|i1", or nil for a kind
	and size .npy does not name. */
NSString * _Nullable NFKValueFileNPYDescription(BOOL floating, NSUInteger elementSize);

/*! An .npy file of packed row-major elements of the type description names, in shape. */
NSData *NFKValueFileNPYData(NSData *elements, NSString *description, NSArray<NSNumber *> *shape);

/*! The elements, type string, and shape of an .npy file in C order, or NO with an error. The elements
	share the file's storage. */
BOOL NFKValueFileReadNPY(NSData *data, NSData * _Nullable * _Nonnull outElements, NSString * _Nullable * _Nonnull outDescription,
						 NSArray<NSNumber *> * _Nullable * _Nonnull outShape, NSError * _Nullable * _Nullable error);

/*! Whether a pixel buffer of format crosses as a PNG with every byte kept: 32BGRA, 32ARGB, and
	OneComponent8. 32RGBA is left out because CVPixelBufferCreate refuses it on macOS, so no buffer of it
	could be made to read the PNG into; it crosses as its planes. */
BOOL NFKValueFilePNGHoldsPixelFormat(OSType format);

/*! The pixel buffer as a PNG holding its bytes exactly, alpha unpremultiplied as stored, or nil. */
NSData * _Nullable NFKValueFilePNGData(CVPixelBufferRef buffer);

/*! A pixel buffer of format holding a PNG's pixels. A PNG this form wrote comes back byte for byte; any
	other PNG of 8-bit samples is read as stored. Returns NULL with an error for a PNG of another layout. */
CVPixelBufferRef _Nullable NFKValueFilePixelBufferFromPNG(NSData *data, OSType format, NSError * _Nullable * _Nullable error) CF_RETURNS_RETAINED;

NS_ASSUME_NONNULL_END
