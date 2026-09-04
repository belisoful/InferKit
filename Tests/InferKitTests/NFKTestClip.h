//
//  NFKTestClip.h
//  InferKitTests
//
//  A short H.264 clip of solid-colour frames, written on the fly, for the tests that read video.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKTestClip : NSObject

/*! Writes a 64×64 clip at two frames a second, one frame per colour given as {r, g, b} in 0…1, to a
	unique file in the temporary directory, and returns its URL or nil with an error. */
+ (nullable NSURL *)writeClipWithColors:(NSArray<NSArray<NSNumber *> *> *)colors error:(NSError * _Nullable *)outError;

/*! The mean {r, g, b} of an image in 0…1, for asserting which frame came back. */
+ (NSArray<NSNumber *> *)meanColorOfImage:(CGImageRef)image;

@end

NS_ASSUME_NONNULL_END
