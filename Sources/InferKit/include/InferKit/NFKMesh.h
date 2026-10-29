//
//  NFKMesh.h
//  InferKit
//

#ifndef NFKMesh_h
#define NFKMesh_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! A position, normal, or color: three floats. Plain C so the header imports no simd, whose overloads
	would reach every Swift file that imports InferKit. Introduced in InferKit 0.4.0. */
typedef struct NFKMeshVector {
	float x;
	float y;
	float z;
} NFKMeshVector;

/*! A texture coordinate: u across the image, v up it. Introduced in InferKit 0.4.0. */
typedef struct NFKMeshTextureCoordinate {
	float u;
	float v;
} NFKMeshTextureCoordinate;

/*! A triangle's three vertex indices, counter-clockwise seen from outside. Introduced in InferKit 0.4.0. */
typedef struct NFKMeshTriangle {
	uint32_t a;
	uint32_t b;
	uint32_t c;
} NFKMeshTriangle;

/*! An NFKMeshVector from its components. Introduced in InferKit 0.4.0. */
NS_INLINE NFKMeshVector NFKMeshVectorMake(float x, float y, float z)
{
	NFKMeshVector vector = { x, y, z };
	return vector;
}

/*!
	@class      NFKMesh
	@abstract   A triangle mesh: vertex positions and triangle indices, with optional normals, vertex
				colors, and texture coordinates.
	@discussion A 3D-generation backend (an image or a prompt in, an object out) returns one of these
				under NFKOutputMesh. Positions are in the backend's object space in glTF's frame:
				right-handed, y up, and an object reconstructed from a picture facing +z, toward the
				camera that took it.
				Each attribute is one packed little-endian float32 array in vertex order: three floats a
				position, three a normal, three a color in 0...1 (red, green, blue), two a texture
				coordinate (u, v with v up, the glTF and OBJ convention). The indices are three uint32 a
				triangle, counter-clockwise seen from outside.

				The mesh writes itself as a Wavefront OBJ (text, vertex colors as the common "v x y z r g
				b" extension), a binary little-endian PLY (colors as 8-bit red, green, blue), or a glTF
				2.0 binary (GLB, one primitive, colors as COLOR_0 floats), and reads each of the three.
				A PLY is read in its text and both binary forms; an OBJ's polygons are fanned into
				triangles and its per-face attribute indices are resolved to one attribute set per vertex.

				Introduced in InferKit 0.4.0.

				The type archives: a consumer that records a result writes it through NSKeyedArchiver with
				secure coding on, and reads it back with unarchivedObjectOfClasses:.
*/
@interface NFKMesh : NSObject <NSCopying, NSSecureCoding>

/*! The number of vertices. */
@property (nonatomic, readonly) NSUInteger vertexCount;

/*! The number of triangles. */
@property (nonatomic, readonly) NSUInteger faceCount;

/*! The positions, three float32 a vertex. */
@property (nonatomic, readonly, copy) NSData *positions;

/*! The unit normals, three float32 a vertex, or nil when the mesh carries none. */
@property (nonatomic, readonly, copy, nullable) NSData *normals;

/*! The vertex colors, three float32 a vertex in 0...1 (red, green, blue), or nil when the mesh carries
	none. */
@property (nonatomic, readonly, copy, nullable) NSData *colors;

/*! The texture coordinates, two float32 a vertex (u, v), or nil when the mesh carries none. */
@property (nonatomic, readonly, copy, nullable) NSData *textureCoordinates;

/*! The triangle indices, three uint32 a face, each below vertexCount. */
@property (nonatomic, readonly, copy) NSData *indices;

/*! The smallest corner of the axis-aligned box around the positions, or zero for an empty mesh. */
@property (nonatomic, readonly) NFKMeshVector boundingBoxMinimum;

/*! The largest corner of the axis-aligned box around the positions, or zero for an empty mesh. */
@property (nonatomic, readonly) NFKMeshVector boundingBoxMaximum;

/*! A mesh of positions and triangles alone. Returns nil and fills error when a length is not a whole
	number of elements or an index is out of range. */
+ (nullable instancetype)meshWithPositions:(NSData *)positions indices:(NSData *)indices error:(NSError **)error;

/*!
	@method     initWithPositions:normals:colors:textureCoordinates:indices:error:
	@abstract   A mesh from packed attribute arrays.
	@discussion positions holds three float32 a vertex and indices three uint32 a triangle. Each optional
				attribute is nil or holds one entry per vertex. Returns nil and fills error with
				kNFKError_InferenceUnsupported in NFKInferenceErrorDomain when a length is not a whole
				number of elements, an attribute's vertex count differs from the positions', or an index
				reaches vertexCount.
*/
- (nullable instancetype)initWithPositions:(NSData *)positions
								   normals:(nullable NSData *)normals
									colors:(nullable NSData *)colors
						textureCoordinates:(nullable NSData *)textureCoordinates
								   indices:(NSData *)indices
									 error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The position of vertex index. */
- (NFKMeshVector)positionAtIndex:(NSUInteger)index;

/*! The normal of vertex index, or zero when the mesh carries no normals. */
- (NFKMeshVector)normalAtIndex:(NSUInteger)index;

/*! The color of vertex index, or zero when the mesh carries no colors. */
- (NFKMeshVector)colorAtIndex:(NSUInteger)index;

/*! The texture coordinate of vertex index, or zero when the mesh carries none. */
- (NFKMeshTextureCoordinate)textureCoordinateAtIndex:(NSUInteger)index;

/*! The three vertex indices of triangle index. */
- (NFKMeshTriangle)faceAtIndex:(NSUInteger)index;

/*! The mesh as a Wavefront OBJ file. */
- (NSData *)OBJData;

/*! The mesh as a binary little-endian PLY file. */
- (NSData *)PLYData;

/*! The mesh as a glTF 2.0 binary (GLB) file. */
- (NSData *)GLBData;

/*! Writes the mesh in the format url's extension names: "obj", "ply", or "glb". Returns NO and fills
	error for another extension or a failed write. */
- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error;

/*! Reads a Wavefront OBJ file. A malformed file fails with NSFileReadCorruptFileError in NSCocoaErrorDomain. */
+ (nullable instancetype)meshWithOBJData:(NSData *)data error:(NSError **)error;

/*! Reads a PLY file in its text, binary little-endian, or binary big-endian form. Vertex properties x,
	y, z, nx, ny, nz, red, green, blue (8-bit or float), and s, t (or u, v) are read; the face element's
	list property gives the triangles, with larger polygons fanned. A malformed file fails with
	NSFileReadCorruptFileError in NSCocoaErrorDomain. */
+ (nullable instancetype)meshWithPLYData:(NSData *)data error:(NSError **)error;

/*! Reads a glTF 2.0 binary (GLB) file: every triangle primitive of every mesh, with POSITION, NORMAL,
	COLOR_0, and TEXCOORD_0 where present, concatenated into one mesh in the order the file lists them.
	A file without a binary chunk, with an unsupported primitive mode, or with an attribute of another
	component type fails with NSFileReadCorruptFileError in NSCocoaErrorDomain. */
+ (nullable instancetype)meshWithGLBData:(NSData *)data error:(NSError **)error;

/*! Reads the mesh at url in the format its extension names: "obj", "ply", or "glb". */
+ (nullable instancetype)meshWithContentsOfURL:(NSURL *)url error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKMesh_h */
