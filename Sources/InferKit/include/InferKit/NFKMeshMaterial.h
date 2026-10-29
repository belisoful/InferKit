//
//  NFKMeshMaterial.h
//  InferKit
//

#ifndef NFKMeshMaterial_h
#define NFKMeshMaterial_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The MIME type of a PNG texture. Introduced in InferKit 0.4.0. */
FOUNDATION_EXPORT NSString * const NFKMeshTextureMIMETypePNG;
/*! The MIME type of a JPEG texture. Introduced in InferKit 0.4.0. */
FOUNDATION_EXPORT NSString * const NFKMeshTextureMIMETypeJPEG;

/*!
	@class      NFKMeshTexture
	@abstract   One texture image of a mesh material: an encoded image file and its MIME type.
	@discussion The bytes are a whole PNG or JPEG file, the two image types glTF 2.0 embeds. The texture
				is read through the mesh's texture coordinates, u across the image from its left edge and v
				up it from its bottom edge. The type decodes nothing; a consumer reads the image through
				ImageIO (CGImageSourceCreateWithData).

				Introduced in InferKit 0.4.0.
*/
@interface NFKMeshTexture : NSObject <NSCopying, NSSecureCoding>

/*! The encoded image file. */
@property (nonatomic, readonly, copy) NSData *data;

/*! The image's MIME type: NFKMeshTextureMIMETypePNG or NFKMeshTextureMIMETypeJPEG. */
@property (nonatomic, readonly, copy) NSString *MIMEType;

/*! A texture from an encoded image file. Returns nil and fills error with kNFKError_InferenceUnsupported
	in NFKInferenceErrorDomain when data is empty or MIMEType is neither PNG nor JPEG. */
- (nullable instancetype)initWithData:(NSData *)data MIMEType:(NSString *)MIMEType error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKMeshMaterial
	@abstract   A physically based material: glTF 2.0's metallic-roughness model.
	@discussion The base color is baseColorFactor times the base color texture's sample, when it has one.
				Metalness and roughness are metallicFactor and roughnessFactor, each times the blue and the
				green channel of the metallic-roughness texture when it has one. The normal texture holds a
				tangent-space normal per texel, each component mapped from -1...1 to 0...1, as glTF reads
				it. The mesh's GLB writes the material with its textures embedded, and the GLB reader reads
				it back; OBJ and PLY carry no material.

				Introduced in InferKit 0.4.0.
*/
@interface NFKMeshMaterial : NSObject <NSCopying, NSSecureCoding>

/*! The base color texture, or nil when the base color is the factor alone. */
@property (nonatomic, readonly, copy, nullable) NFKMeshTexture *baseColorTexture;

/*! The tangent-space normal texture, or nil when the surface uses its vertex normals alone. */
@property (nonatomic, readonly, copy, nullable) NFKMeshTexture *normalTexture;

/*! The metallic-roughness texture (green roughness, blue metalness), or nil when the factors hold
	over the whole surface. */
@property (nonatomic, readonly, copy, nullable) NFKMeshTexture *metallicRoughnessTexture;

/*! The base color's red, green, blue, and alpha factors in 0...1, four float32. */
@property (nonatomic, readonly, copy) NSArray<NSNumber *> *baseColorFactor;

/*! The metalness in 0...1. */
@property (nonatomic, readonly) float metallicFactor;

/*! The roughness in 0...1. */
@property (nonatomic, readonly) float roughnessFactor;

/*!
	@method     initWithBaseColorTexture:normalTexture:metallicRoughnessTexture:baseColorFactor:metallicFactor:roughnessFactor:error:
	@abstract   A material from its textures and factors.
	@discussion A nil baseColorFactor is white and opaque. Returns nil and fills error with
				kNFKError_InferenceUnsupported in NFKInferenceErrorDomain when baseColorFactor does not
				hold four numbers or a factor lies outside 0...1.
*/
- (nullable instancetype)initWithBaseColorTexture:(nullable NFKMeshTexture *)baseColorTexture
									normalTexture:(nullable NFKMeshTexture *)normalTexture
						 metallicRoughnessTexture:(nullable NFKMeshTexture *)metallicRoughnessTexture
								  baseColorFactor:(nullable NSArray<NSNumber *> *)baseColorFactor
								   metallicFactor:(float)metallicFactor
								  roughnessFactor:(float)roughnessFactor
											error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMeshMaterial_h */
