//
//  NFKMesh.m
//  InferKit
//

#import "NFKMesh.h"
#import "NFKMeshMaterial.h"
#import "NFKErrors.h"
#import <simd/simd.h>

static NSString * const NFKMeshGLBGenerator = @"InferKit";

static BOOL NFKMeshFail(NSError **error, NSInteger code, NSString *domain, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

static BOOL NFKMeshInvalid(NSError **error, NSString *reason)
{
	return NFKMeshFail(error, kNFKError_InferenceUnsupported, NFKInferenceErrorDomain,
					   [NSString stringWithFormat:@"The mesh is not well formed: %@.", reason]);
}

static BOOL NFKMeshReadFail(NSError **error, NSString *format, NSString *reason)
{
	return NFKMeshFail(error, NSFileReadCorruptFileError, NSCocoaErrorDomain,
					   [NSString stringWithFormat:@"The %@ file cannot be read: %@.", format, reason]);
}

/*! Appends a little-endian 32-bit value. */
static void NFKMeshAppendUInt32(NSMutableData *data, uint32_t value)
{
	uint32_t little = CFSwapInt32HostToLittle(value);
	[data appendBytes:&little length:4];
}

static void NFKMeshAppendFloat(NSMutableData *data, float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4);
	NFKMeshAppendUInt32(data, bits);
}

/*! Pads data with `pad` to a multiple of four bytes. */
static void NFKMeshPadToFour(NSMutableData *data, uint8_t pad)
{
	while (data.length % 4 != 0) {
		[data appendBytes:&pad length:1];
	}
}

#pragma mark - PLY structure

typedef NS_ENUM(NSInteger, NFKPLYType) {
	NFKPLYTypeInt8, NFKPLYTypeUInt8, NFKPLYTypeInt16, NFKPLYTypeUInt16, NFKPLYTypeInt32, NFKPLYTypeUInt32,
	NFKPLYTypeFloat32, NFKPLYTypeFloat64, NFKPLYTypeUnknown
};

static NFKPLYType NFKPLYTypeNamed(NSString *name)
{
	static NSDictionary<NSString *, NSNumber *> *types;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		types = @{ @"char": @(NFKPLYTypeInt8), @"int8": @(NFKPLYTypeInt8), @"uchar": @(NFKPLYTypeUInt8), @"uint8": @(NFKPLYTypeUInt8),
				   @"short": @(NFKPLYTypeInt16), @"int16": @(NFKPLYTypeInt16), @"ushort": @(NFKPLYTypeUInt16), @"uint16": @(NFKPLYTypeUInt16),
				   @"int": @(NFKPLYTypeInt32), @"int32": @(NFKPLYTypeInt32), @"uint": @(NFKPLYTypeUInt32), @"uint32": @(NFKPLYTypeUInt32),
				   @"float": @(NFKPLYTypeFloat32), @"float32": @(NFKPLYTypeFloat32), @"double": @(NFKPLYTypeFloat64), @"float64": @(NFKPLYTypeFloat64) };
	});
	NSNumber *type = types[name];
	return type != nil ? type.integerValue : NFKPLYTypeUnknown;
}

static NSUInteger NFKPLYTypeSize(NFKPLYType type)
{
	switch (type) {
		case NFKPLYTypeInt8: case NFKPLYTypeUInt8: return 1;
		case NFKPLYTypeInt16: case NFKPLYTypeUInt16: return 2;
		case NFKPLYTypeInt32: case NFKPLYTypeUInt32: case NFKPLYTypeFloat32: return 4;
		case NFKPLYTypeFloat64: return 8;
		case NFKPLYTypeUnknown: return 0;
	}
	return 0;
}

/*! A PLY property: scalar, or a list whose count has its own type. */
@interface NFKPLYProperty : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic) NFKPLYType type;
@property (nonatomic) BOOL isList;
@property (nonatomic) NFKPLYType countType;
@end
@implementation NFKPLYProperty
@end

@interface NFKPLYElement : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic) NSUInteger count;
@property (nonatomic, strong) NSMutableArray<NFKPLYProperty *> *properties;
@end
@implementation NFKPLYElement
@end

/*! A cursor over PLY body values, binary of either byte order or whitespace-separated text. */
@interface NFKPLYCursor : NSObject
@property (nonatomic, strong) NSData *data;
@property (nonatomic) NSUInteger offset;
@property (nonatomic) BOOL binary;
@property (nonatomic) BOOL bigEndian;
@property (nonatomic, strong) NSArray<NSString *> *tokens;
@property (nonatomic) NSUInteger tokenIndex;
- (BOOL)read:(NFKPLYType)type into:(double *)outValue;
@end

@implementation NFKPLYCursor

- (BOOL)read:(NFKPLYType)type into:(double *)outValue
{
	if (!self.binary) {
		if (self.tokenIndex >= self.tokens.count) {
			return NO;
		}
		*outValue = self.tokens[self.tokenIndex++].doubleValue;
		return YES;
	}
	NSUInteger size = NFKPLYTypeSize(type);
	if (size == 0 || self.offset + size > self.data.length) {
		return NO;
	}
	uint8_t raw[8] = { 0 };
	memcpy(raw, (const uint8_t *)self.data.bytes + self.offset, size);
	self.offset += size;
	if (self.bigEndian) {
		for (NSUInteger i = 0; i < size / 2; i++) {
			uint8_t swap = raw[i];
			raw[i] = raw[size - 1 - i];
			raw[size - 1 - i] = swap;
		}
	}
	switch (type) {
		case NFKPLYTypeInt8: *outValue = *(int8_t *)raw; break;
		case NFKPLYTypeUInt8: *outValue = *(uint8_t *)raw; break;
		case NFKPLYTypeInt16: { int16_t v; memcpy(&v, raw, 2); *outValue = v; break; }
		case NFKPLYTypeUInt16: { uint16_t v; memcpy(&v, raw, 2); *outValue = v; break; }
		case NFKPLYTypeInt32: { int32_t v; memcpy(&v, raw, 4); *outValue = v; break; }
		case NFKPLYTypeUInt32: { uint32_t v; memcpy(&v, raw, 4); *outValue = v; break; }
		case NFKPLYTypeFloat32: { float v; memcpy(&v, raw, 4); *outValue = v; break; }
		case NFKPLYTypeFloat64: { double v; memcpy(&v, raw, 8); *outValue = v; break; }
		case NFKPLYTypeUnknown: return NO;
	}
	return YES;
}

@end

@interface NFKMesh ()
@property (nonatomic, copy) NSData *positions;
@property (nonatomic, copy, nullable) NSData *normals;
@property (nonatomic, copy, nullable) NSData *colors;
@property (nonatomic, copy, nullable) NSData *textureCoordinates;
@property (nonatomic, copy) NSData *indices;
@property (nonatomic, copy, nullable) NFKMeshMaterial *material;
@end

@implementation NFKMesh

#pragma mark - Construction

+ (nullable instancetype)meshWithPositions:(NSData *)positions indices:(NSData *)indices error:(NSError **)error
{
	return [[self alloc] initWithPositions:positions normals:nil colors:nil textureCoordinates:nil indices:indices error:error];
}

- (nullable instancetype)initWithPositions:(NSData *)positions
								   normals:(nullable NSData *)normals
									colors:(nullable NSData *)colors
						textureCoordinates:(nullable NSData *)textureCoordinates
								   indices:(NSData *)indices
									 error:(NSError **)error
{
	return [self initWithPositions:positions normals:normals colors:colors textureCoordinates:textureCoordinates
						   indices:indices material:nil error:error];
}

- (nullable instancetype)initWithPositions:(NSData *)positions
								   normals:(nullable NSData *)normals
									colors:(nullable NSData *)colors
						textureCoordinates:(nullable NSData *)textureCoordinates
								   indices:(NSData *)indices
								  material:(nullable NFKMeshMaterial *)material
									 error:(NSError **)error
{
	self = [super init];
	if (self == nil) {
		return nil;
	}
	if (positions.length % 12 != 0) {
		NFKMeshInvalid(error, @"the positions are not a whole number of float32 triples");
		return nil;
	}
	if (indices.length % 12 != 0) {
		NFKMeshInvalid(error, @"the indices are not a whole number of uint32 triples");
		return nil;
	}
	NSUInteger vertexCount = positions.length / 12;
	if (normals != nil && normals.length != positions.length) {
		NFKMeshInvalid(error, @"the normals do not number one per vertex");
		return nil;
	}
	if (colors != nil && colors.length != positions.length) {
		NFKMeshInvalid(error, @"the colors do not number one per vertex");
		return nil;
	}
	if (textureCoordinates != nil && textureCoordinates.length != vertexCount * 8) {
		NFKMeshInvalid(error, @"the texture coordinates do not number one per vertex");
		return nil;
	}
	const uint32_t *faces = indices.bytes;
	NSUInteger indexCount = indices.length / 4;
	for (NSUInteger i = 0; i < indexCount; i++) {
		if (faces[i] >= vertexCount) {
			NFKMeshInvalid(error, [NSString stringWithFormat:@"index %u reaches past the %lu vertices", faces[i], (unsigned long)vertexCount]);
			return nil;
		}
	}
	BOOL textured = material.baseColorTexture != nil || material.normalTexture != nil || material.metallicRoughnessTexture != nil;
	if (textured && textureCoordinates == nil) {
		NFKMeshInvalid(error, @"the material has a texture and the mesh no texture coordinates to read it through");
		return nil;
	}
	_material = material;
	_positions = [positions copy];
	_normals = [normals copy];
	_colors = [colors copy];
	_textureCoordinates = [textureCoordinates copy];
	_indices = [indices copy];
	return self;
}

- (NSUInteger)vertexCount
{
	return self.positions.length / 12;
}

- (NSUInteger)faceCount
{
	return self.indices.length / 12;
}

#pragma mark - Element access

static simd_float3 NFKMeshFloat3(NSData *data, NSUInteger index)
{
	if (data == nil || (index + 1) * 12 > data.length) {
		return simd_make_float3(0, 0, 0);
	}
	const float *values = (const float *)data.bytes + index * 3;
	return simd_make_float3(values[0], values[1], values[2]);
}

static NFKMeshVector NFKMeshVectorFrom(simd_float3 v)
{
	return NFKMeshVectorMake(v.x, v.y, v.z);
}

- (NFKMeshVector)positionAtIndex:(NSUInteger)index
{
	return NFKMeshVectorFrom(NFKMeshFloat3(self.positions, index));
}

- (NFKMeshVector)normalAtIndex:(NSUInteger)index
{
	return NFKMeshVectorFrom(NFKMeshFloat3(self.normals, index));
}

- (NFKMeshVector)colorAtIndex:(NSUInteger)index
{
	return NFKMeshVectorFrom(NFKMeshFloat3(self.colors, index));
}

- (NFKMeshTextureCoordinate)textureCoordinateAtIndex:(NSUInteger)index
{
	NFKMeshTextureCoordinate coordinate = { 0, 0 };
	NSData *data = self.textureCoordinates;
	if (data == nil || (index + 1) * 8 > data.length) {
		return coordinate;
	}
	const float *values = (const float *)data.bytes + index * 2;
	coordinate.u = values[0];
	coordinate.v = values[1];
	return coordinate;
}

- (NFKMeshTriangle)faceAtIndex:(NSUInteger)index
{
	NFKMeshTriangle triangle = { 0, 0, 0 };
	if ((index + 1) * 12 > self.indices.length) {
		return triangle;
	}
	const uint32_t *values = (const uint32_t *)self.indices.bytes + index * 3;
	triangle.a = values[0];
	triangle.b = values[1];
	triangle.c = values[2];
	return triangle;
}

- (void)boundsMinimum:(simd_float3 *)outMinimum maximum:(simd_float3 *)outMaximum
{
	NSUInteger count = self.vertexCount;
	if (count == 0) {
		*outMinimum = simd_make_float3(0, 0, 0);
		*outMaximum = simd_make_float3(0, 0, 0);
		return;
	}
	simd_float3 minimum = NFKMeshFloat3(self.positions, 0);
	simd_float3 maximum = minimum;
	for (NSUInteger i = 1; i < count; i++) {
		simd_float3 position = NFKMeshFloat3(self.positions, i);
		minimum = simd_min(minimum, position);
		maximum = simd_max(maximum, position);
	}
	*outMinimum = minimum;
	*outMaximum = maximum;
}

- (NFKMeshVector)boundingBoxMinimum
{
	simd_float3 minimum, maximum;
	[self boundsMinimum:&minimum maximum:&maximum];
	return NFKMeshVectorFrom(minimum);
}

- (NFKMeshVector)boundingBoxMaximum
{
	simd_float3 minimum, maximum;
	[self boundsMinimum:&minimum maximum:&maximum];
	return NFKMeshVectorFrom(maximum);
}

#pragma mark - Identity

- (id)copyWithZone:(NSZone *)zone
{
	// Immutable.
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKMesh.class]) {
		return NO;
	}
	NFKMesh *other = object;
	BOOL (^same)(NSData *, NSData *) = ^BOOL(NSData *a, NSData *b) {
		return (a == nil && b == nil) || [a isEqualToData:b];
	};
	return same(self.positions, other.positions) && same(self.normals, other.normals) && same(self.colors, other.colors)
		&& same(self.textureCoordinates, other.textureCoordinates) && same(self.indices, other.indices)
		&& ((self.material == nil && other.material == nil) || [self.material isEqual:other.material]);
}

- (NSUInteger)hash
{
	return self.positions.hash ^ self.indices.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %lu vertices, %lu faces%@%@%@%@>", NSStringFromClass(self.class), (unsigned long)self.vertexCount,
			(unsigned long)self.faceCount, self.normals != nil ? @", normals" : @"", self.colors != nil ? @", colors" : @"",
			self.textureCoordinates != nil ? @", texture coordinates" : @"", self.material != nil ? @", material" : @""];
}

#pragma mark - Writing

- (NSData *)OBJData
{
	NSMutableString *text = [NSMutableString stringWithString:@"# Wavefront OBJ written by InferKit\n"];
	NSUInteger vertexCount = self.vertexCount;
	BOOL colors = self.colors != nil, normals = self.normals != nil, textures = self.textureCoordinates != nil;
	for (NSUInteger i = 0; i < vertexCount; i++) {
		simd_float3 p = NFKMeshFloat3(self.positions, i);
		if (colors) {
			simd_float3 c = NFKMeshFloat3(self.colors, i);
			[text appendFormat:@"v %.9g %.9g %.9g %.9g %.9g %.9g\n", p.x, p.y, p.z, c.x, c.y, c.z];
		} else {
			[text appendFormat:@"v %.9g %.9g %.9g\n", p.x, p.y, p.z];
		}
	}
	for (NSUInteger i = 0; textures && i < vertexCount; i++) {
		NFKMeshTextureCoordinate t = [self textureCoordinateAtIndex:i];
		[text appendFormat:@"vt %.9g %.9g\n", t.u, t.v];
	}
	for (NSUInteger i = 0; normals && i < vertexCount; i++) {
		simd_float3 n = NFKMeshFloat3(self.normals, i);
		[text appendFormat:@"vn %.9g %.9g %.9g\n", n.x, n.y, n.z];
	}
	NSUInteger faceCount = self.faceCount;
	for (NSUInteger i = 0; i < faceCount; i++) {
		NFKMeshTriangle f = [self faceAtIndex:i];
		[text appendString:@"f"];
		uint32_t corners[3] = { f.a, f.b, f.c };
		for (int k = 0; k < 3; k++) {
			unsigned long index = corners[k] + 1;
			if (textures && normals) {
				[text appendFormat:@" %lu/%lu/%lu", index, index, index];
			} else if (textures) {
				[text appendFormat:@" %lu/%lu", index, index];
			} else if (normals) {
				[text appendFormat:@" %lu//%lu", index, index];
			} else {
				[text appendFormat:@" %lu", index];
			}
		}
		[text appendString:@"\n"];
	}
	return [text dataUsingEncoding:NSUTF8StringEncoding];
}

- (NSData *)PLYData
{
	NSUInteger vertexCount = self.vertexCount, faceCount = self.faceCount;
	BOOL colors = self.colors != nil, normals = self.normals != nil, textures = self.textureCoordinates != nil;
	NSMutableString *header = [NSMutableString stringWithFormat:@"ply\nformat binary_little_endian 1.0\ncomment written by InferKit\nelement vertex %lu\n"
								"property float x\nproperty float y\nproperty float z\n", (unsigned long)vertexCount];
	if (normals) {
		[header appendString:@"property float nx\nproperty float ny\nproperty float nz\n"];
	}
	if (colors) {
		[header appendString:@"property uchar red\nproperty uchar green\nproperty uchar blue\n"];
	}
	if (textures) {
		[header appendString:@"property float s\nproperty float t\n"];
	}
	[header appendFormat:@"element face %lu\nproperty list uchar uint vertex_indices\nend_header\n", (unsigned long)faceCount];

	NSMutableData *data = [[header dataUsingEncoding:NSASCIIStringEncoding] mutableCopy];
	for (NSUInteger i = 0; i < vertexCount; i++) {
		simd_float3 p = NFKMeshFloat3(self.positions, i);
		NFKMeshAppendFloat(data, p.x); NFKMeshAppendFloat(data, p.y); NFKMeshAppendFloat(data, p.z);
		if (normals) {
			simd_float3 n = NFKMeshFloat3(self.normals, i);
			NFKMeshAppendFloat(data, n.x); NFKMeshAppendFloat(data, n.y); NFKMeshAppendFloat(data, n.z);
		}
		if (colors) {
			simd_float3 c = simd_clamp(NFKMeshFloat3(self.colors, i), 0, 1);
			uint8_t bytes[3] = { (uint8_t)lrintf(c.x * 255), (uint8_t)lrintf(c.y * 255), (uint8_t)lrintf(c.z * 255) };
			[data appendBytes:bytes length:3];
		}
		if (textures) {
			NFKMeshTextureCoordinate t = [self textureCoordinateAtIndex:i];
			NFKMeshAppendFloat(data, t.u); NFKMeshAppendFloat(data, t.v);
		}
	}
	for (NSUInteger i = 0; i < faceCount; i++) {
		NFKMeshTriangle f = [self faceAtIndex:i];
		uint8_t three = 3;
		[data appendBytes:&three length:1];
		NFKMeshAppendUInt32(data, f.a); NFKMeshAppendUInt32(data, f.b); NFKMeshAppendUInt32(data, f.c);
	}
	return data;
}

- (NSData *)GLBData
{
	NSMutableData *binary = [NSMutableData data];
	NSMutableArray<NSDictionary *> *bufferViews = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *accessors = [NSMutableArray array];
	NSMutableDictionary<NSString *, NSNumber *> *attributes = [NSMutableDictionary dictionary];

	NSNumber *(^addView)(NSData *, NSNumber *) = ^NSNumber *(NSData *bytes, NSNumber *target) {
		NSMutableDictionary *view = [@{ @"buffer": @0, @"byteOffset": @(binary.length), @"byteLength": @(bytes.length) } mutableCopy];
		view[@"target"] = target;
		[binary appendData:bytes];
		NFKMeshPadToFour(binary, 0);
		[bufferViews addObject:view];
		return @(bufferViews.count - 1);
	};
	NSUInteger vertexCount = self.vertexCount;
	NSNumber *(^addAttribute)(NSData *, NSString *, BOOL) = ^NSNumber *(NSData *bytes, NSString *type, BOOL bounds) {
		NSMutableDictionary *accessor = [@{ @"bufferView": addView(bytes, @34962), @"componentType": @5126,
											@"count": @(vertexCount), @"type": type } mutableCopy];
		if (bounds) {
			simd_float3 minimum, maximum;
			[self boundsMinimum:&minimum maximum:&maximum];
			accessor[@"min"] = @[ @(minimum.x), @(minimum.y), @(minimum.z) ];
			accessor[@"max"] = @[ @(maximum.x), @(maximum.y), @(maximum.z) ];
		}
		[accessors addObject:accessor];
		return @(accessors.count - 1);
	};
	attributes[@"POSITION"] = addAttribute(self.positions, @"VEC3", YES);
	if (self.normals != nil) {
		attributes[@"NORMAL"] = addAttribute(self.normals, @"VEC3", NO);
	}
	if (self.colors != nil) {
		attributes[@"COLOR_0"] = addAttribute(self.colors, @"VEC3", NO);
	}
	if (self.textureCoordinates != nil) {
		// glTF's v runs down the image; OBJ's and this type's run up.
		NSMutableData *flipped = [self.textureCoordinates mutableCopy];
		float *values = flipped.mutableBytes;
		for (NSUInteger i = 0; i < vertexCount; i++) {
			values[i * 2 + 1] = 1.0f - values[i * 2 + 1];
		}
		attributes[@"TEXCOORD_0"] = addAttribute(flipped, @"VEC2", NO);
	}
	[accessors addObject:@{ @"bufferView": addView(self.indices, @34963), @"componentType": @5125,
							@"count": @(self.faceCount * 3), @"type": @"SCALAR" }];
	NSNumber *indexAccessor = @(accessors.count - 1);

	NSMutableDictionary *primitive = [@{ @"attributes": attributes, @"indices": indexAccessor, @"mode": @4 } mutableCopy];
	NSMutableDictionary *document = [@{
		@"asset": @{ @"version": @"2.0", @"generator": NFKMeshGLBGenerator },
		@"scene": @0,
		@"scenes": @[ @{ @"nodes": @[ @0 ] } ],
		@"nodes": @[ @{ @"mesh": @0 } ],
		@"meshes": @[ @{ @"primitives": @[ primitive ] } ],
		@"accessors": accessors,
		@"bufferViews": bufferViews,
	} mutableCopy];
	NFKMeshMaterial *material = self.material;
	if (material != nil) {
		NSMutableArray<NSDictionary *> *images = [NSMutableArray array];
		NSMutableArray<NSDictionary *> *textures = [NSMutableArray array];
		NSDictionary *(^addTexture)(NFKMeshTexture *) = ^NSDictionary *(NFKMeshTexture *texture) {
			[images addObject:@{ @"bufferView": addView(texture.data, nil), @"mimeType": texture.MIMEType }];
			[textures addObject:@{ @"sampler": @0, @"source": @(images.count - 1) }];
			return @{ @"index": @(textures.count - 1), @"texCoord": @0 };
		};
		NSMutableDictionary *pbr = [@{ @"baseColorFactor": material.baseColorFactor,
									   @"metallicFactor": @(material.metallicFactor),
									   @"roughnessFactor": @(material.roughnessFactor) } mutableCopy];
		NSMutableDictionary *entry = [@{ @"pbrMetallicRoughness": pbr } mutableCopy];
		if (material.baseColorTexture != nil) {
			pbr[@"baseColorTexture"] = addTexture(material.baseColorTexture);
		}
		if (material.metallicRoughnessTexture != nil) {
			pbr[@"metallicRoughnessTexture"] = addTexture(material.metallicRoughnessTexture);
		}
		if (material.normalTexture != nil) {
			entry[@"normalTexture"] = addTexture(material.normalTexture);
		}
		document[@"materials"] = @[ entry ];
		if (textures.count > 0) {
			document[@"images"] = images;
			document[@"textures"] = textures;
			// Linear filtering with mipmaps, repeating, as glTF's defaults read.
			document[@"samplers"] = @[ @{ @"magFilter": @9729, @"minFilter": @9987, @"wrapS": @10497, @"wrapT": @10497 } ];
		}
		primitive[@"material"] = @0;
	}
	document[@"buffers"] = @[ @{ @"byteLength": @(binary.length) } ];
	NSMutableData *json = [[NSJSONSerialization dataWithJSONObject:document options:NSJSONWritingSortedKeys error:NULL] mutableCopy];
	NFKMeshPadToFour(json, ' ');

	NSMutableData *file = [NSMutableData data];
	NFKMeshAppendUInt32(file, 0x46546C67);                                  // "glTF"
	NFKMeshAppendUInt32(file, 2);
	NFKMeshAppendUInt32(file, (uint32_t)(12 + 8 + json.length + 8 + binary.length));
	NFKMeshAppendUInt32(file, (uint32_t)json.length);
	NFKMeshAppendUInt32(file, 0x4E4F534A);                                  // "JSON"
	[file appendData:json];
	NFKMeshAppendUInt32(file, (uint32_t)binary.length);
	NFKMeshAppendUInt32(file, 0x004E4942);                                  // "BIN\0"
	[file appendData:binary];
	return file;
}

- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error
{
	NSString *extension = url.pathExtension.lowercaseString;
	NSData *data = nil;
	if ([extension isEqualToString:@"obj"]) {
		data = self.OBJData;
	} else if ([extension isEqualToString:@"ply"]) {
		data = self.PLYData;
	} else if ([extension isEqualToString:@"glb"]) {
		data = self.GLBData;
	} else {
		return NFKMeshFail(error, NSFileWriteUnsupportedSchemeError, NSCocoaErrorDomain,
						   [NSString stringWithFormat:@"A mesh is written as .obj, .ply, or .glb, not .%@.", extension]);
	}
	return [data writeToURL:url options:NSDataWritingAtomic error:error];
}

#pragma mark - Reading

+ (nullable instancetype)meshWithContentsOfURL:(NSURL *)url error:(NSError **)error
{
	NSData *data = [NSData dataWithContentsOfURL:url options:0 error:error];
	if (data == nil) {
		return nil;
	}
	NSString *extension = url.pathExtension.lowercaseString;
	if ([extension isEqualToString:@"obj"]) {
		return [self meshWithOBJData:data error:error];
	}
	if ([extension isEqualToString:@"ply"]) {
		return [self meshWithPLYData:data error:error];
	}
	if ([extension isEqualToString:@"glb"]) {
		return [self meshWithGLBData:data error:error];
	}
	NFKMeshFail(error, NSFileReadUnsupportedSchemeError, NSCocoaErrorDomain,
				[NSString stringWithFormat:@"A mesh is read from .obj, .ply, or .glb, not .%@.", extension]);
	return nil;
}

#pragma mark OBJ

/*! One "a/b/c" corner of an OBJ face: 1-based indices resolved against the counts so far, 0 for absent. */
static BOOL NFKMeshOBJCorner(NSString *token, NSUInteger positionCount, NSUInteger textureCount, NSUInteger normalCount,
							 NSUInteger *outPosition, NSUInteger *outTexture, NSUInteger *outNormal)
{
	NSArray<NSString *> *parts = [token componentsSeparatedByString:@"/"];
	NSUInteger counts[3] = { positionCount, textureCount, normalCount };
	NSUInteger *outputs[3] = { outPosition, outTexture, outNormal };
	for (NSUInteger k = 0; k < 3; k++) {
		*outputs[k] = 0;
		if (k >= parts.count || parts[k].length == 0) {
			continue;
		}
		NSInteger value = parts[k].integerValue;
		if (value < 0) {
			value += (NSInteger)counts[k] + 1;                             // relative to the entries so far
		}
		if (value <= 0 || (NSUInteger)value > counts[k]) {
			return NO;
		}
		*outputs[k] = (NSUInteger)value;
	}
	return *outPosition != 0;
}

+ (nullable instancetype)meshWithOBJData:(NSData *)data error:(NSError **)error
{
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]
		?: [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding];
	if (text == nil) {
		NFKMeshReadFail(error, @"OBJ", @"the file is not text");
		return nil;
	}
	NSMutableArray<NSArray<NSNumber *> *> *positions = [NSMutableArray array];
	NSMutableArray<NSArray<NSNumber *> *> *vertexColors = [NSMutableArray array];
	NSMutableArray<NSArray<NSNumber *> *> *textures = [NSMutableArray array];
	NSMutableArray<NSArray<NSNumber *> *> *normals = [NSMutableArray array];
	NSMutableArray<NSArray<NSNumber *> *> *faces = [NSMutableArray array];       // triples of (p, t, n) keys per corner
	BOOL anyColor = NO;
	__block BOOL malformed = NO;
	__block NSString *problem = nil;
	NSCharacterSet *whitespace = NSCharacterSet.whitespaceCharacterSet;

	NSArray<NSString *> *lines = [text componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
	for (NSString *rawLine in lines) {
		NSString *line = [rawLine stringByTrimmingCharactersInSet:whitespace];
		if (line.length == 0 || [line hasPrefix:@"#"]) {
			continue;
		}
		NSMutableArray<NSString *> *tokens = [[line componentsSeparatedByCharactersInSet:whitespace] mutableCopy];
		[tokens removeObject:@""];
		NSString *keyword = tokens.firstObject;
		NSArray<NSString *> *values = [tokens subarrayWithRange:NSMakeRange(1, tokens.count - 1)];
		NSArray<NSNumber *> *(^numbers)(NSUInteger) = ^NSArray<NSNumber *> *(NSUInteger minimum) {
			if (values.count < minimum) {
				return nil;
			}
			NSMutableArray<NSNumber *> *parsed = [NSMutableArray arrayWithCapacity:values.count];
			for (NSString *value in values) {
				[parsed addObject:@(value.doubleValue)];
			}
			return parsed;
		};
		if ([keyword isEqualToString:@"v"]) {
			NSArray<NSNumber *> *parsed = numbers(3);
			if (parsed == nil) {
				problem = @"a vertex has fewer than three coordinates";
				malformed = YES;
				break;
			}
			[positions addObject:[parsed subarrayWithRange:NSMakeRange(0, 3)]];
			BOOL colored = parsed.count >= 6;
			anyColor = anyColor || colored;
			[vertexColors addObject:colored ? [parsed subarrayWithRange:NSMakeRange(3, 3)] : @[ @0, @0, @0 ]];
		} else if ([keyword isEqualToString:@"vt"]) {
			NSArray<NSNumber *> *parsed = numbers(1);
			if (parsed == nil) {
				problem = @"a texture coordinate is empty";
				malformed = YES;
				break;
			}
			[textures addObject:@[ parsed[0], parsed.count > 1 ? parsed[1] : @0 ]];
		} else if ([keyword isEqualToString:@"vn"]) {
			NSArray<NSNumber *> *parsed = numbers(3);
			if (parsed == nil) {
				problem = @"a normal has fewer than three components";
				malformed = YES;
				break;
			}
			[normals addObject:[parsed subarrayWithRange:NSMakeRange(0, 3)]];
		} else if ([keyword isEqualToString:@"f"]) {
			if (values.count < 3) {
				problem = @"a face has fewer than three corners";
				malformed = YES;
				break;
			}
			NSMutableArray<NSArray<NSNumber *> *> *corners = [NSMutableArray arrayWithCapacity:values.count];
			for (NSString *token in values) {
				NSUInteger p, t, n;
				if (!NFKMeshOBJCorner(token, positions.count, textures.count, normals.count, &p, &t, &n)) {
					problem = [NSString stringWithFormat:@"the face corner \"%@\" names no vertex read so far", token];
					malformed = YES;
					break;
				}
				[corners addObject:@[ @(p), @(t), @(n) ]];
			}
			if (malformed) {
				break;
			}
			for (NSUInteger k = 1; k + 1 < corners.count; k++) {                // fan a polygon
				[faces addObject:corners[0]];
				[faces addObject:corners[k]];
				[faces addObject:corners[k + 1]];
			}
		}
	}
	if (malformed) {
		NFKMeshReadFail(error, @"OBJ", problem);
		return nil;
	}

	// A file whose corners index one attribute set per position keeps its vertex order; otherwise one
	// vertex is made per distinct (position, texture, normal) corner.
	BOOL hasTextures = textures.count > 0, hasNormals = normals.count > 0;
	BOOL aligned = (!hasTextures || textures.count == positions.count) && (!hasNormals || normals.count == positions.count);
	for (NSArray<NSNumber *> *corner in faces) {
		NSUInteger p = corner[0].unsignedIntegerValue, t = corner[1].unsignedIntegerValue, n = corner[2].unsignedIntegerValue;
		aligned = aligned && (t == 0 || t == p) && (n == 0 || n == p);
		if (!aligned) {
			break;
		}
	}
	if (aligned) {
		NSMutableData *alignedPositions = [NSMutableData data], *alignedColors = [NSMutableData data];
		NSMutableData *alignedTextures = [NSMutableData data], *alignedNormals = [NSMutableData data], *alignedIndices = [NSMutableData data];
		for (NSUInteger i = 0; i < positions.count; i++) {
			for (NSNumber *number in positions[i]) { float value = number.floatValue; [alignedPositions appendBytes:&value length:4]; }
			for (NSNumber *number in vertexColors[i]) { float value = number.floatValue; [alignedColors appendBytes:&value length:4]; }
			for (NSNumber *number in hasTextures ? textures[i] : @[]) { float value = number.floatValue; [alignedTextures appendBytes:&value length:4]; }
			for (NSNumber *number in hasNormals ? normals[i] : @[]) { float value = number.floatValue; [alignedNormals appendBytes:&value length:4]; }
		}
		BOOL usesTextures = NO, usesNormals = NO;
		for (NSArray<NSNumber *> *corner in faces) {
			uint32_t index = (uint32_t)(corner[0].unsignedIntegerValue - 1);
			[alignedIndices appendBytes:&index length:4];
			usesTextures = usesTextures || corner[1].unsignedIntegerValue > 0;
			usesNormals = usesNormals || corner[2].unsignedIntegerValue > 0;
		}
		return [[self alloc] initWithPositions:alignedPositions
									   normals:hasNormals && usesNormals ? alignedNormals : nil
										colors:anyColor ? alignedColors : nil
							textureCoordinates:hasTextures && usesTextures ? alignedTextures : nil
									   indices:alignedIndices
										 error:error];
	}
	NSMutableDictionary<NSArray<NSNumber *> *, NSNumber *> *vertexForCorner = [NSMutableDictionary dictionary];
	NSMutableData *outPositions = [NSMutableData data], *outColors = [NSMutableData data];
	NSMutableData *outTextures = [NSMutableData data], *outNormals = [NSMutableData data], *outIndices = [NSMutableData data];
	void (^appendFloats)(NSMutableData *, NSArray<NSNumber *> *) = ^(NSMutableData *out, NSArray<NSNumber *> *array) {
		for (NSNumber *number in array) {
			float value = number.floatValue;
			[out appendBytes:&value length:4];
		}
	};
	BOOL usesTextures = NO, usesNormals = NO;
	for (NSArray<NSNumber *> *corner in faces) {
		NSNumber *vertex = vertexForCorner[corner];
		if (vertex == nil) {
			NSUInteger p = corner[0].unsignedIntegerValue, t = corner[1].unsignedIntegerValue, n = corner[2].unsignedIntegerValue;
			appendFloats(outPositions, positions[p - 1]);
			appendFloats(outColors, vertexColors[p - 1]);
			appendFloats(outTextures, t > 0 ? textures[t - 1] : @[ @0, @0 ]);
			appendFloats(outNormals, n > 0 ? normals[n - 1] : @[ @0, @0, @0 ]);
			usesTextures = usesTextures || t > 0;
			usesNormals = usesNormals || n > 0;
			vertex = @(vertexForCorner.count);
			vertexForCorner[corner] = vertex;
		}
		uint32_t index = vertex.unsignedIntValue;
		[outIndices appendBytes:&index length:4];
	}
	if (faces.count == 0) {
		// A point cloud, or a file with vertices only: keep every vertex.
		for (NSUInteger i = 0; i < positions.count; i++) {
			appendFloats(outPositions, positions[i]);
			appendFloats(outColors, vertexColors[i]);
		}
	}
	return [[self alloc] initWithPositions:outPositions
								   normals:hasNormals && usesNormals ? outNormals : nil
									colors:anyColor ? outColors : nil
						textureCoordinates:hasTextures && usesTextures ? outTextures : nil
								   indices:outIndices
									 error:error];
}

#pragma mark PLY

+ (nullable instancetype)meshWithPLYData:(NSData *)data error:(NSError **)error
{
	// The header is ASCII up to and including "end_header\n".
	NSData *terminator = [@"end_header" dataUsingEncoding:NSASCIIStringEncoding];
	NSRange end = [data rangeOfData:terminator options:0 range:NSMakeRange(0, data.length)];
	if (end.location == NSNotFound) {
		NFKMeshReadFail(error, @"PLY", @"the header has no end_header line");
		return nil;
	}
	NSUInteger bodyStart = NSMaxRange(end);
	const uint8_t *bytes = data.bytes;
	while (bodyStart < data.length && bytes[bodyStart] != '\n') {
		bodyStart++;
	}
	bodyStart = MIN(bodyStart + 1, data.length);
	NSString *header = [[NSString alloc] initWithData:[data subdataWithRange:NSMakeRange(0, end.location)] encoding:NSASCIIStringEncoding];
	NSArray<NSString *> *lines = [header componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
	if (lines.count == 0 || ![[lines[0] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] isEqualToString:@"ply"]) {
		NFKMeshReadFail(error, @"PLY", @"the file does not start with \"ply\"");
		return nil;
	}

	NFKPLYCursor *cursor = [[NFKPLYCursor alloc] init];
	cursor.data = data;
	cursor.offset = bodyStart;
	BOOL formatSeen = NO;
	NSMutableArray<NFKPLYElement *> *elements = [NSMutableArray array];
	for (NSString *rawLine in lines) {
		NSMutableArray<NSString *> *tokens = [[rawLine componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceCharacterSet] mutableCopy];
		[tokens removeObject:@""];
		if (tokens.count == 0) {
			continue;
		}
		NSString *keyword = tokens[0];
		if ([keyword isEqualToString:@"format"] && tokens.count >= 2) {
			formatSeen = YES;
			cursor.binary = ![tokens[1] isEqualToString:@"ascii"];
			cursor.bigEndian = [tokens[1] isEqualToString:@"binary_big_endian"];
			if (cursor.binary && ![tokens[1] hasPrefix:@"binary_"]) {
				NFKMeshReadFail(error, @"PLY", [NSString stringWithFormat:@"the format \"%@\" is unknown", tokens[1]]);
				return nil;
			}
		} else if ([keyword isEqualToString:@"element"] && tokens.count >= 3) {
			NFKPLYElement *element = [[NFKPLYElement alloc] init];
			element.name = tokens[1];
			element.count = (NSUInteger)MAX(0, tokens[2].integerValue);
			element.properties = [NSMutableArray array];
			[elements addObject:element];
		} else if ([keyword isEqualToString:@"property"] && tokens.count >= 3) {
			if (elements.count == 0) {
				NFKMeshReadFail(error, @"PLY", @"a property precedes every element");
				return nil;
			}
			NFKPLYProperty *property = [[NFKPLYProperty alloc] init];
			if ([tokens[1] isEqualToString:@"list"]) {
				if (tokens.count < 5) {
					NFKMeshReadFail(error, @"PLY", @"a list property names a count type, an item type, and a name");
					return nil;
				}
				property.isList = YES;
				property.countType = NFKPLYTypeNamed(tokens[2]);
				property.type = NFKPLYTypeNamed(tokens[3]);
				property.name = tokens[4];
			} else {
				property.type = NFKPLYTypeNamed(tokens[1]);
				property.name = tokens[2];
			}
			if (property.type == NFKPLYTypeUnknown || (property.isList && property.countType == NFKPLYTypeUnknown)) {
				NFKMeshReadFail(error, @"PLY", [NSString stringWithFormat:@"the property \"%@\" has an unknown type", property.name]);
				return nil;
			}
			[elements.lastObject.properties addObject:property];
		}
	}
	if (!formatSeen) {
		NFKMeshReadFail(error, @"PLY", @"the header names no format");
		return nil;
	}
	if (!cursor.binary) {
		NSString *body = [[NSString alloc] initWithData:[data subdataWithRange:NSMakeRange(bodyStart, data.length - bodyStart)] encoding:NSASCIIStringEncoding];
		NSMutableArray<NSString *> *tokens = [[body componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet] mutableCopy];
		[tokens removeObject:@""];
		cursor.tokens = tokens;
	}

	NSMutableData *positions = [NSMutableData data], *normals = [NSMutableData data], *colors = [NSMutableData data];
	NSMutableData *textures = [NSMutableData data], *indices = [NSMutableData data];
	BOOL hasNormals = NO, hasColors = NO, hasTextures = NO;
	NSSet<NSString *> *normalNames = [NSSet setWithArray:@[ @"nx", @"ny", @"nz" ]];
	NSSet<NSString *> *colorNames = [NSSet setWithArray:@[ @"red", @"green", @"blue", @"r", @"g", @"b" ]];
	NSSet<NSString *> *textureNames = [NSSet setWithArray:@[ @"s", @"t", @"u", @"v", @"texture_u", @"texture_v" ]];

	for (NFKPLYElement *element in elements) {
		BOOL isVertex = [element.name isEqualToString:@"vertex"];
		BOOL isFace = [element.name isEqualToString:@"face"];
		if (isVertex) {
			NSMutableSet<NSString *> *names = [NSMutableSet set];
			for (NFKPLYProperty *property in element.properties) {
				[names addObject:property.name];
			}
			hasNormals = [normalNames isSubsetOfSet:names];
			hasColors = ([names containsObject:@"red"] && [names containsObject:@"green"] && [names containsObject:@"blue"])
				|| ([names containsObject:@"r"] && [names containsObject:@"g"] && [names containsObject:@"b"]);
			hasTextures = ([names containsObject:@"s"] && [names containsObject:@"t"]) || ([names containsObject:@"u"] && [names containsObject:@"v"])
				|| ([names containsObject:@"texture_u"] && [names containsObject:@"texture_v"]);
			if (![names containsObject:@"x"] || ![names containsObject:@"y"] || ![names containsObject:@"z"]) {
				NFKMeshReadFail(error, @"PLY", @"the vertex element has no x, y, z");
				return nil;
			}
		}
		for (NSUInteger i = 0; i < element.count; i++) {
			float position[3] = { 0 }, normal[3] = { 0 }, color[3] = { 0 }, texture[2] = { 0 };
			for (NFKPLYProperty *property in element.properties) {
				if (property.isList) {
					double count = 0;
					if (![cursor read:property.countType into:&count]) {
						NFKMeshReadFail(error, @"PLY", @"the body ends inside a list");
						return nil;
					}
					NSUInteger items = (NSUInteger)MAX(0, count);
					uint32_t polygon[items > 0 ? items : 1];
					for (NSUInteger k = 0; k < items; k++) {
						double value = 0;
						if (![cursor read:property.type into:&value]) {
							NFKMeshReadFail(error, @"PLY", @"the body ends inside a list");
							return nil;
						}
						polygon[k] = (uint32_t)MAX(0, value);
					}
					BOOL faceIndices = isFace && ([property.name isEqualToString:@"vertex_indices"] || [property.name isEqualToString:@"vertex_index"]);
					for (NSUInteger k = 1; faceIndices && k + 1 < items; k++) {
						uint32_t triangle[3] = { polygon[0], polygon[k], polygon[k + 1] };
						[indices appendBytes:triangle length:12];
					}
					continue;
				}
				double value = 0;
				if (![cursor read:property.type into:&value]) {
					NFKMeshReadFail(error, @"PLY", @"the body ends before every element is read");
					return nil;
				}
				if (!isVertex) {
					continue;
				}
				NSString *name = property.name;
				if ([name isEqualToString:@"x"]) { position[0] = (float)value; }
				else if ([name isEqualToString:@"y"]) { position[1] = (float)value; }
				else if ([name isEqualToString:@"z"]) { position[2] = (float)value; }
				else if ([normalNames containsObject:name]) { normal[[name characterAtIndex:1] - 'x'] = (float)value; }
				else if ([colorNames containsObject:name]) {
					float scaled = property.type == NFKPLYTypeFloat32 || property.type == NFKPLYTypeFloat64 ? (float)value : (float)(value / 255.0);
					unichar initial = [name characterAtIndex:0];
					color[initial == 'r' ? 0 : initial == 'g' ? 1 : 2] = scaled;
				} else if ([textureNames containsObject:name]) {
					unichar last = [name characterAtIndex:name.length - 1];
					texture[last == 's' || last == 'u' ? 0 : 1] = (float)value;
				}
			}
			if (isVertex) {
				[positions appendBytes:position length:12];
				[normals appendBytes:normal length:12];
				[colors appendBytes:color length:12];
				[textures appendBytes:texture length:8];
			}
		}
	}
	return [[self alloc] initWithPositions:positions normals:hasNormals ? normals : nil colors:hasColors ? colors : nil
						textureCoordinates:hasTextures ? textures : nil indices:indices error:error];
}

#pragma mark GLB

static uint32_t NFKMeshReadUInt32(const uint8_t *bytes)
{
	uint32_t value;
	memcpy(&value, bytes, 4);
	return CFSwapInt32LittleToHost(value);
}

/*! The image a glTF texture-info entry names, from the binary chunk, or nil when the entry, its texture,
	its image, or the image's buffer view is missing or malformed, or the image is not PNG or JPEG. */
static NFKMeshTexture *NFKMeshGLBTexture(NSDictionary *document, NSData *binary, id info)
{
	if (![info isKindOfClass:NSDictionary.class] || ![info[@"index"] isKindOfClass:NSNumber.class]) {
		return nil;
	}
	NSArray *textures = document[@"textures"], *images = document[@"images"], *views = document[@"bufferViews"];
	NSUInteger textureIndex = [info[@"index"] unsignedIntegerValue];
	if (![textures isKindOfClass:NSArray.class] || textureIndex >= textures.count) {
		return nil;
	}
	NSDictionary *texture = textures[textureIndex];
	if (![texture isKindOfClass:NSDictionary.class] || ![texture[@"source"] isKindOfClass:NSNumber.class]) {
		return nil;
	}
	NSUInteger imageIndex = [texture[@"source"] unsignedIntegerValue];
	if (![images isKindOfClass:NSArray.class] || imageIndex >= images.count) {
		return nil;
	}
	NSDictionary *image = images[imageIndex];
	if (![image isKindOfClass:NSDictionary.class] || ![image[@"bufferView"] isKindOfClass:NSNumber.class]
		|| ![image[@"mimeType"] isKindOfClass:NSString.class]) {
		return nil;
	}
	NSUInteger viewIndex = [image[@"bufferView"] unsignedIntegerValue];
	if (![views isKindOfClass:NSArray.class] || viewIndex >= views.count || ![views[viewIndex] isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSDictionary *view = views[viewIndex];
	NSUInteger offset = [view[@"byteOffset"] unsignedIntegerValue], length = [view[@"byteLength"] unsignedIntegerValue];
	if ([view[@"buffer"] unsignedIntegerValue] != 0 || offset + length > binary.length) {
		return nil;
	}
	return [[NFKMeshTexture alloc] initWithData:[binary subdataWithRange:NSMakeRange(offset, length)]
									   MIMEType:image[@"mimeType"] error:NULL];
}

/*! Material index of a glTF document as an NFKMeshMaterial, or nil when it is missing or malformed. A
	factor outside 0...1 is clamped into it, and a texture this type cannot hold is left out. */
static NFKMeshMaterial *NFKMeshGLBMaterial(NSDictionary *document, NSData *binary, NSUInteger index)
{
	NSArray *materials = document[@"materials"];
	if (![materials isKindOfClass:NSArray.class] || index >= materials.count || ![materials[index] isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSDictionary *material = materials[index];
	NSDictionary *pbr = [material[@"pbrMetallicRoughness"] isKindOfClass:NSDictionary.class] ? material[@"pbrMetallicRoughness"] : @{};
	float (^factor)(id, float) = ^float(id value, float fallback) {
		float number = [value isKindOfClass:NSNumber.class] ? [value floatValue] : fallback;
		return fminf(fmaxf(number, 0), 1);
	};
	NSArray<NSNumber *> *baseColorFactor = nil;
	if ([pbr[@"baseColorFactor"] isKindOfClass:NSArray.class] && [pbr[@"baseColorFactor"] count] == 4) {
		NSMutableArray<NSNumber *> *components = [NSMutableArray arrayWithCapacity:4];
		for (id component in pbr[@"baseColorFactor"]) {
			[components addObject:@(factor(component, 1))];
		}
		baseColorFactor = components;
	}
	return [[NFKMeshMaterial alloc] initWithBaseColorTexture:NFKMeshGLBTexture(document, binary, pbr[@"baseColorTexture"])
											   normalTexture:NFKMeshGLBTexture(document, binary, material[@"normalTexture"])
									metallicRoughnessTexture:NFKMeshGLBTexture(document, binary, pbr[@"metallicRoughnessTexture"])
											 baseColorFactor:baseColorFactor
											  metallicFactor:factor(pbr[@"metallicFactor"], 1)
											 roughnessFactor:factor(pbr[@"roughnessFactor"], 1)
													   error:NULL];
}

+ (nullable instancetype)meshWithGLBData:(NSData *)data error:(NSError **)error
{
	const uint8_t *bytes = data.bytes;
	if (data.length < 20 || NFKMeshReadUInt32(bytes) != 0x46546C67) {
		NFKMeshReadFail(error, @"GLB", @"the file does not start with the glTF magic");
		return nil;
	}
	NSDictionary *document = nil;
	NSData *binary = nil;
	NSUInteger cursor = 12;
	while (cursor + 8 <= data.length) {
		uint32_t length = NFKMeshReadUInt32(bytes + cursor), type = NFKMeshReadUInt32(bytes + cursor + 4);
		cursor += 8;
		if (cursor + length > data.length) {
			NFKMeshReadFail(error, @"GLB", @"a chunk runs past the end of the file");
			return nil;
		}
		NSData *chunk = [data subdataWithRange:NSMakeRange(cursor, length)];
		if (type == 0x4E4F534A) {
			document = [NSJSONSerialization JSONObjectWithData:chunk options:0 error:NULL];
		} else if (type == 0x004E4942 && binary == nil) {
			binary = chunk;
		}
		cursor += length;
	}
	if (![document isKindOfClass:NSDictionary.class]) {
		NFKMeshReadFail(error, @"GLB", @"the JSON chunk is missing or malformed");
		return nil;
	}
	if (binary == nil) {
		NFKMeshReadFail(error, @"GLB", @"the file has no binary chunk");
		return nil;
	}
	NSArray *accessors = document[@"accessors"], *bufferViews = document[@"bufferViews"], *meshes = document[@"meshes"];
	if (![accessors isKindOfClass:NSArray.class] || ![bufferViews isKindOfClass:NSArray.class] || ![meshes isKindOfClass:NSArray.class]) {
		NFKMeshReadFail(error, @"GLB", @"the document has no meshes, accessors, or buffer views");
		return nil;
	}
	static NSDictionary<NSString *, NSNumber *> *componentCounts;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		componentCounts = @{ @"SCALAR": @1, @"VEC2": @2, @"VEC3": @3, @"VEC4": @4 };
	});

	// Reads accessor `index` as doubles, `components` per element. Returns nil when malformed.
	__block NSString *problem = nil;
	NSArray<NSNumber *> *(^readAccessor)(NSNumber *, NSUInteger *) = ^NSArray<NSNumber *> *(NSNumber *index, NSUInteger *outComponents) {
		NSDictionary *accessor = index.unsignedIntegerValue < accessors.count ? accessors[index.unsignedIntegerValue] : nil;
		if (![accessor isKindOfClass:NSDictionary.class] || accessor[@"bufferView"] == nil) {
			problem = @"an accessor is missing or has no buffer view";
			return nil;
		}
		NSUInteger viewIndex = [accessor[@"bufferView"] unsignedIntegerValue];
		NSDictionary *view = viewIndex < bufferViews.count ? bufferViews[viewIndex] : nil;
		if (![view isKindOfClass:NSDictionary.class] || [view[@"buffer"] unsignedIntegerValue] != 0) {
			problem = @"an accessor's buffer view is missing or names a second buffer";
			return nil;
		}
		NSUInteger components = componentCounts[accessor[@"type"] ?: @""].unsignedIntegerValue;
		NSUInteger componentType = [accessor[@"componentType"] unsignedIntegerValue];
		NSUInteger componentSize = componentType == 5126 || componentType == 5125 ? 4 : componentType == 5123 || componentType == 5122 ? 2
			: componentType == 5121 || componentType == 5120 ? 1 : 0;
		if (components == 0 || componentSize == 0) {
			problem = @"an accessor has an unsupported type";
			return nil;
		}
		NSUInteger count = [accessor[@"count"] unsignedIntegerValue];
		NSUInteger elementSize = components * componentSize;
		NSUInteger stride = [view[@"byteStride"] unsignedIntegerValue] ?: elementSize;
		NSUInteger start = [view[@"byteOffset"] unsignedIntegerValue] + [accessor[@"byteOffset"] unsignedIntegerValue];
		if (count > 0 && start + (count - 1) * stride + elementSize > binary.length) {
			problem = @"an accessor reads past the binary chunk";
			return nil;
		}
		const uint8_t *base = binary.bytes;
		NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:count * components];
		for (NSUInteger i = 0; i < count; i++) {
			const uint8_t *element = base + start + i * stride;
			for (NSUInteger c = 0; c < components; c++) {
				const uint8_t *at = element + c * componentSize;
				double value = 0;
				switch (componentType) {
					case 5126: { float v; memcpy(&v, at, 4); value = v; break; }
					case 5125: { uint32_t v; memcpy(&v, at, 4); value = CFSwapInt32LittleToHost(v); break; }
					case 5123: { uint16_t v; memcpy(&v, at, 2); value = CFSwapInt16LittleToHost(v); break; }
					case 5122: { int16_t v; memcpy(&v, at, 2); value = (int16_t)CFSwapInt16LittleToHost((uint16_t)v); break; }
					case 5121: value = *at; break;
					case 5120: value = *(const int8_t *)at; break;
				}
				BOOL normalized = [accessor[@"normalized"] boolValue];
				if (normalized && componentType == 5121) { value /= 255.0; }
				if (normalized && componentType == 5123) { value /= 65535.0; }
				[values addObject:@(value)];
			}
		}
		*outComponents = components;
		return values;
	};

	NSMutableData *positions = [NSMutableData data], *normals = [NSMutableData data], *colors = [NSMutableData data];
	NSMutableData *textures = [NSMutableData data], *indices = [NSMutableData data];
	BOOL anyNormals = NO, anyColors = NO, anyTextures = NO, missingNormals = NO, missingColors = NO, missingTextures = NO;
	void (^appendFloats)(NSMutableData *, NSArray<NSNumber *> *, NSUInteger, NSUInteger, NSUInteger) =
		^(NSMutableData *out, NSArray<NSNumber *> *values, NSUInteger components, NSUInteger keep, NSUInteger count) {
			for (NSUInteger i = 0; i < count; i++) {
				for (NSUInteger c = 0; c < keep; c++) {
					float value = values != nil && c < components ? values[i * components + c].floatValue : 0;
					[out appendBytes:&value length:4];
				}
			}
		};
	NSNumber *materialIndex = nil;
	for (NSDictionary *mesh in meshes) {
		for (NSDictionary *primitive in [mesh[@"primitives"] isKindOfClass:NSArray.class] ? mesh[@"primitives"] : @[]) {
			if (materialIndex == nil && [primitive[@"material"] isKindOfClass:NSNumber.class]) {
				materialIndex = primitive[@"material"];
			}
			NSUInteger mode = primitive[@"mode"] != nil ? [primitive[@"mode"] unsignedIntegerValue] : 4;
			if (mode != 4) {
				NFKMeshReadFail(error, @"GLB", @"a primitive is not a triangle list");
				return nil;
			}
			NSDictionary *attributes = primitive[@"attributes"];
			NSUInteger positionComponents = 0;
			NSArray<NSNumber *> *positionValues = attributes[@"POSITION"] != nil ? readAccessor(attributes[@"POSITION"], &positionComponents) : nil;
			if (positionValues == nil || positionComponents != 3) {
				NFKMeshReadFail(error, @"GLB", problem ?: @"a primitive has no POSITION");
				return nil;
			}
			NSUInteger count = positionValues.count / 3;
			NSUInteger base = positions.length / 12;
			appendFloats(positions, positionValues, 3, 3, count);

			NSUInteger components = 0;
			NSArray<NSNumber *> *values = attributes[@"NORMAL"] != nil ? readAccessor(attributes[@"NORMAL"], &components) : nil;
			if (attributes[@"NORMAL"] != nil && values == nil) {
				NFKMeshReadFail(error, @"GLB", problem);
				return nil;
			}
			anyNormals = anyNormals || values != nil;
			missingNormals = missingNormals || values == nil;
			appendFloats(normals, values, components, 3, count);

			values = attributes[@"COLOR_0"] != nil ? readAccessor(attributes[@"COLOR_0"], &components) : nil;
			if (attributes[@"COLOR_0"] != nil && values == nil) {
				NFKMeshReadFail(error, @"GLB", problem);
				return nil;
			}
			anyColors = anyColors || values != nil;
			missingColors = missingColors || values == nil;
			appendFloats(colors, values, components, 3, count);

			values = attributes[@"TEXCOORD_0"] != nil ? readAccessor(attributes[@"TEXCOORD_0"], &components) : nil;
			if (attributes[@"TEXCOORD_0"] != nil && values == nil) {
				NFKMeshReadFail(error, @"GLB", problem);
				return nil;
			}
			anyTextures = anyTextures || values != nil;
			missingTextures = missingTextures || values == nil;
			NSUInteger before = textures.length;
			appendFloats(textures, values, components, 2, count);
			if (values != nil) {
				float *uv = (float *)((uint8_t *)textures.mutableBytes + before);
				for (NSUInteger i = 0; i < count; i++) {
					uv[i * 2 + 1] = 1.0f - uv[i * 2 + 1];                   // glTF's v runs down; this type's runs up
				}
			}

			NSArray<NSNumber *> *indexValues = nil;
			if (primitive[@"indices"] != nil) {
				indexValues = readAccessor(primitive[@"indices"], &components);
				if (indexValues == nil || components != 1) {
					NFKMeshReadFail(error, @"GLB", problem ?: @"the indices accessor is not scalar");
					return nil;
				}
			} else {
				NSMutableArray<NSNumber *> *sequence = [NSMutableArray arrayWithCapacity:count];
				for (NSUInteger i = 0; i < count; i++) {
					[sequence addObject:@(i)];
				}
				indexValues = sequence;
			}
			for (NSUInteger i = 0; i + 2 < indexValues.count; i += 3) {
				uint32_t triangle[3] = { (uint32_t)(indexValues[i].unsignedIntegerValue + base),
										 (uint32_t)(indexValues[i + 1].unsignedIntegerValue + base),
										 (uint32_t)(indexValues[i + 2].unsignedIntegerValue + base) };
				[indices appendBytes:triangle length:12];
			}
		}
	}
	NSData *textureCoordinates = anyTextures && !missingTextures ? textures : nil;
	NFKMeshMaterial *material = materialIndex != nil ? NFKMeshGLBMaterial(document, binary, materialIndex.unsignedIntegerValue) : nil;
	BOOL textured = material.baseColorTexture != nil || material.normalTexture != nil || material.metallicRoughnessTexture != nil;
	if (textured && textureCoordinates == nil) {
		material = nil;
	}
	// An attribute some primitives lack is dropped rather than invented as zeros.
	return [[self alloc] initWithPositions:positions
								   normals:anyNormals && !missingNormals ? normals : nil
									colors:anyColors && !missingColors ? colors : nil
						textureCoordinates:textureCoordinates
								   indices:indices
								  material:material
									 error:error];
}

#pragma mark - NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.positions forKey:@"positions"];
	[coder encodeObject:self.normals forKey:@"normals"];
	[coder encodeObject:self.colors forKey:@"colors"];
	[coder encodeObject:self.textureCoordinates forKey:@"textureCoordinates"];
	[coder encodeObject:self.indices forKey:@"indices"];
	[coder encodeObject:self.material forKey:@"material"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSData *positions = [coder decodeObjectOfClass:NSData.class forKey:@"positions"];
	NSData *indices = [coder decodeObjectOfClass:NSData.class forKey:@"indices"];
	return [self initWithPositions:positions ?: [NSData data]
						   normals:[coder decodeObjectOfClass:NSData.class forKey:@"normals"]
							colors:[coder decodeObjectOfClass:NSData.class forKey:@"colors"]
				textureCoordinates:[coder decodeObjectOfClass:NSData.class forKey:@"textureCoordinates"]
						   indices:indices ?: [NSData data]
						  material:[coder decodeObjectOfClass:NFKMeshMaterial.class forKey:@"material"]
							 error:NULL];
}

@end
