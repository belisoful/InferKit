//
//  NFKGaussianSplats.h
//  InferKit
//

#ifndef NFKGaussianSplats_h
#define NFKGaussianSplats_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKMesh.h>

NS_ASSUME_NONNULL_BEGIN

/*! A splat's orientation as a unit quaternion: w the scalar part. Introduced in InferKit 0.4.0. */
typedef struct NFKGaussianRotation {
	float w;
	float x;
	float y;
	float z;
} NFKGaussianRotation;

/*!
	@class      NFKGaussianSplats
	@abstract   A 3D Gaussian splatting scene: anisotropic Gaussians with a position, scale, rotation,
				opacity, and color each.
	@discussion A 3D-generation backend that returns Gaussians puts one of these under
				NFKOutputGaussianSplats. Positions share NFKMesh's frame: right-handed, y up, an object
				reconstructed from a picture facing +z.
				Each attribute is one packed little-endian float32 array in splat order: three floats a
				position, three a scale (the standard deviations along the splat's own axes, in the
				positions' units), four a rotation (w, x, y, z, unit length), one an opacity in 0...1, and
				three a color as degree-0 spherical-harmonic coefficients (the displayed color is
				0.5 + 0.28209479 × coefficient per channel).
				The scene writes itself as the binary little-endian PLY that 3D Gaussian splatting tools
				read: properties x, y, z, nx, ny, nz, f_dc_0...2, opacity (as its logit), scale_0...2 (as
				natural logarithms), and rot_0...3, in that format's frame, which turns this one half a turn
				about x (y down, z away from the viewer). It reads that layout back, ignoring further
				properties such as f_rest_*. The type archives with secure coding.
				Introduced in InferKit 0.4.0.
*/
@interface NFKGaussianSplats : NSObject <NSCopying, NSSecureCoding>

/*! The number of Gaussians. */
@property (nonatomic, readonly) NSUInteger count;

/*! Three float32 a Gaussian: its center. */
@property (nonatomic, readonly, copy) NSData *positions;

/*! Three float32 a Gaussian: its standard deviations along its own axes. */
@property (nonatomic, readonly, copy) NSData *scales;

/*! Four float32 a Gaussian: its rotation (w, x, y, z), unit length. */
@property (nonatomic, readonly, copy) NSData *rotations;

/*! One float32 a Gaussian: its opacity in 0...1. */
@property (nonatomic, readonly, copy) NSData *opacities;

/*! Three float32 a Gaussian: its color's degree-0 spherical-harmonic coefficients. */
@property (nonatomic, readonly, copy) NSData *colors;

/*!
	@method     initWithPositions:scales:rotations:opacities:colors:error:
	@abstract   Gaussians from packed attribute arrays.
	@discussion Returns nil and fills error with kNFKError_InferenceUnsupported in
				NFKInferenceErrorDomain when an array is not a whole number of elements or its count
				differs from the positions'.
*/
- (nullable instancetype)initWithPositions:(NSData *)positions
									scales:(NSData *)scales
								 rotations:(NSData *)rotations
								 opacities:(NSData *)opacities
									colors:(NSData *)colors
									 error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The center of Gaussian index. */
- (NFKMeshVector)positionAtIndex:(NSUInteger)index;

/*! The standard deviations of Gaussian index along its own axes. */
- (NFKMeshVector)scaleAtIndex:(NSUInteger)index;

/*! The rotation of Gaussian index. */
- (NFKGaussianRotation)rotationAtIndex:(NSUInteger)index;

/*! The opacity of Gaussian index. */
- (float)opacityAtIndex:(NSUInteger)index;

/*! The displayed color of Gaussian index in 0...1 (red, green, blue), from its coefficients, clamped. */
- (NFKMeshVector)colorAtIndex:(NSUInteger)index;

/*! The Gaussians as a binary little-endian 3D Gaussian splatting PLY file. */
- (NSData *)PLYData;

/*! Writes the PLY file to url. Returns NO and fills error for a failed write. */
- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error;

/*! Reads a binary little-endian 3D Gaussian splatting PLY file. A file in another layout or form fails
	with NSFileReadCorruptFileError in NSCocoaErrorDomain. */
+ (nullable instancetype)splatsWithPLYData:(NSData *)data error:(NSError **)error;

/*! Reads the PLY file at url. */
+ (nullable instancetype)splatsWithContentsOfURL:(NSURL *)url error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGaussianSplats_h */
