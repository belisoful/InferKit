//
//  NFKMeshTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKMesh.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKModelVocabulary.h>
#import <InferKit/NFKValueCoding.h>
#import <InferKit/NFKErrors.h>

static BOOL NFKVectorEqual(NFKMeshVector a, NFKMeshVector b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
static BOOL NFKUVEqual(NFKMeshTextureCoordinate a, NFKMeshTextureCoordinate b) { return a.u == b.u && a.v == b.v; }
static BOOL NFKTriangleEqual(NFKMeshTriangle a, NFKMeshTriangle b) { return a.a == b.a && a.b == b.b && a.c == b.c; }
static NFKMeshTextureCoordinate NFKUV(float u, float v) { NFKMeshTextureCoordinate t = { u, v }; return t; }
static NFKMeshTriangle NFKTri(uint32_t a, uint32_t b, uint32_t c) { NFKMeshTriangle t = { a, b, c }; return t; }
static float NFKVectorMaxDifference(NFKMeshVector a, NFKMeshVector b)
{
	return MAX(fabsf(a.x - b.x), MAX(fabsf(a.y - b.y), fabsf(a.z - b.z)));
}

@interface NFKMeshTests : XCTestCase
@end

@implementation NFKMeshTests

/*! A tetrahedron with every attribute: four vertices, four faces. */
- (NFKMesh *)tetrahedron
{
	float positions[12] = { 0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1 };
	float normals[12] = { -0.577f, -0.577f, -0.577f,  1, 0, 0,  0, 1, 0,  0, 0, 1 };
	float colors[12] = { 1, 0, 0,  0, 1, 0,  0, 0, 1,  0.5f, 0.25f, 0.75f };
	float textures[8] = { 0, 0,  1, 0,  0, 1,  0.5f, 0.5f };
	uint32_t indices[12] = { 0, 2, 1,  0, 1, 3,  0, 3, 2,  1, 2, 3 };
	NSError *error = nil;
	NFKMesh *mesh = [[NFKMesh alloc] initWithPositions:[NSData dataWithBytes:positions length:sizeof positions]
											  normals:[NSData dataWithBytes:normals length:sizeof normals]
											   colors:[NSData dataWithBytes:colors length:sizeof colors]
								   textureCoordinates:[NSData dataWithBytes:textures length:sizeof textures]
											  indices:[NSData dataWithBytes:indices length:sizeof indices]
												error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	return mesh;
}

- (NFKMesh *)bareTriangle
{
	float positions[9] = { 0, 0, 0,  1, 0, 0,  0, 1, 0 };
	uint32_t indices[3] = { 0, 1, 2 };
	NSError *error = nil;
	NFKMesh *mesh = [NFKMesh meshWithPositions:[NSData dataWithBytes:positions length:sizeof positions]
									   indices:[NSData dataWithBytes:indices length:sizeof indices] error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	return mesh;
}

static void NFKAssertMeshesClose(XCTestCase *test, NFKMesh *a, NFKMesh *b, float colorTolerance)
{
	XCTAssertEqual(a.vertexCount, b.vertexCount);
	XCTAssertEqual(a.faceCount, b.faceCount);
	XCTAssertEqual(a.normals != nil, b.normals != nil);
	XCTAssertEqual(a.colors != nil, b.colors != nil);
	XCTAssertEqual(a.textureCoordinates != nil, b.textureCoordinates != nil);
	for (NSUInteger i = 0; i < MIN(a.vertexCount, b.vertexCount); i++) {
		XCTAssertTrue(NFKVectorEqual([a positionAtIndex:i], [b positionAtIndex:i]), @"position %lu", (unsigned long)i);
		XCTAssertTrue(NFKVectorEqual([a normalAtIndex:i], [b normalAtIndex:i]), @"normal %lu", (unsigned long)i);
		XCTAssertTrue(NFKUVEqual([a textureCoordinateAtIndex:i], [b textureCoordinateAtIndex:i]), @"texture coordinate %lu", (unsigned long)i);
		XCTAssertLessThanOrEqual(NFKVectorMaxDifference([a colorAtIndex:i], [b colorAtIndex:i]), colorTolerance, @"color %lu", (unsigned long)i);
	}
	for (NSUInteger i = 0; i < MIN(a.faceCount, b.faceCount); i++) {
		XCTAssertTrue(NFKTriangleEqual([a faceAtIndex:i], [b faceAtIndex:i]), @"face %lu", (unsigned long)i);
	}
}

#pragma mark - Construction

- (void)testCountsAttributesAndBoundsReadBack
{
	NFKMesh *mesh = self.tetrahedron;
	XCTAssertEqual(mesh.vertexCount, 4u);
	XCTAssertEqual(mesh.faceCount, 4u);
	XCTAssertTrue(NFKVectorEqual([mesh positionAtIndex:3], NFKMeshVectorMake(0, 0, 1)));
	XCTAssertTrue(NFKVectorEqual([mesh colorAtIndex:3], NFKMeshVectorMake(0.5f, 0.25f, 0.75f)));
	XCTAssertTrue(NFKUVEqual([mesh textureCoordinateAtIndex:1], NFKUV(1, 0)));
	XCTAssertTrue(NFKTriangleEqual([mesh faceAtIndex:1], NFKTri(0, 1, 3)));
	XCTAssertTrue(NFKVectorEqual(mesh.boundingBoxMinimum, NFKMeshVectorMake(0, 0, 0)));
	XCTAssertTrue(NFKVectorEqual(mesh.boundingBoxMaximum, NFKMeshVectorMake(1, 1, 1)));

	NFKMesh *bare = self.bareTriangle;
	XCTAssertNil(bare.normals);
	XCTAssertNil(bare.colors);
	XCTAssertTrue(NFKVectorEqual([bare normalAtIndex:0], NFKMeshVectorMake(0, 0, 0)));
	XCTAssertTrue(NFKVectorEqual([bare positionAtIndex:99], NFKMeshVectorMake(0, 0, 0)), @"an index past the end reads as zero");
}

- (void)testMalformedArraysAreRefused
{
	float positions[9] = { 0, 0, 0,  1, 0, 0,  0, 1, 0 };
	uint32_t indices[3] = { 0, 1, 2 };
	uint32_t outOfRange[3] = { 0, 1, 3 };
	NSData *positionData = [NSData dataWithBytes:positions length:sizeof positions];
	NSData *indexData = [NSData dataWithBytes:indices length:sizeof indices];
	NSError *error = nil;

	XCTAssertNil([NFKMesh meshWithPositions:[positionData subdataWithRange:NSMakeRange(0, 10)] indices:indexData error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertNil([NFKMesh meshWithPositions:positionData indices:[indexData subdataWithRange:NSMakeRange(0, 8)] error:NULL]);
	XCTAssertNil([NFKMesh meshWithPositions:positionData indices:[NSData dataWithBytes:outOfRange length:sizeof outOfRange] error:NULL]);
	XCTAssertNil([[NFKMesh alloc] initWithPositions:positionData normals:[positionData subdataWithRange:NSMakeRange(0, 24)] colors:nil
								 textureCoordinates:nil indices:indexData error:NULL]);
	XCTAssertNil([[NFKMesh alloc] initWithPositions:positionData normals:nil colors:nil
								 textureCoordinates:[positionData subdataWithRange:NSMakeRange(0, 16)] indices:indexData error:NULL]);
	XCTAssertNotNil([NFKMesh meshWithPositions:[NSData data] indices:[NSData data] error:NULL], @"an empty mesh is well formed");
}

- (void)testEqualityCopyingAndArchiving
{
	NFKMesh *mesh = self.tetrahedron;
	XCTAssertEqualObjects(mesh, self.tetrahedron);
	XCTAssertEqual(mesh.hash, self.tetrahedron.hash);
	XCTAssertNotEqualObjects(mesh, self.bareTriangle);
	XCTAssertEqual([mesh copy], mesh);

	NSError *error = nil;
	NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:mesh requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(archive, @"%@", error);
	NFKMesh *restored = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKMesh.class fromData:archive error:&error];
	XCTAssertEqualObjects(restored, mesh, @"%@", error);
}

- (void)testTheResultConvenienceAndTheVocabularyNameTheMesh
{
	NFKMesh *mesh = self.bareTriangle;
	NFKInferenceResult *result = [[NFKInferenceResult alloc] initWithOutputs:@{ NFKOutputMesh: mesh }];
	XCTAssertEqualObjects(result.mesh, mesh);
	XCTAssertNil([[NFKInferenceResult alloc] initWithOutputs:@{ NFKOutputMesh: @"not a mesh" }].mesh);
	XCTAssertEqualObjects(NFKModelValueTypeForOutputKey(NFKOutputMesh), NFKModelValueTypeMesh);
	XCTAssertEqualObjects(NFKModelTaskImageTo3D, @"image-to-3d");
}

#pragma mark - Files

- (void)testOBJRoundTripKeepsEveryAttribute
{
	NFKMesh *mesh = self.tetrahedron;
	NSError *error = nil;
	NFKMesh *read = [NFKMesh meshWithOBJData:mesh.OBJData error:&error];
	XCTAssertNotNil(read, @"%@", error);
	NFKAssertMeshesClose(self, mesh, read, 1e-6f);

	NSString *text = [[NSString alloc] initWithData:mesh.OBJData encoding:NSUTF8StringEncoding];
	XCTAssertTrue([text containsString:@"\nv 0 0 1 0.5 0.25 0.75\n"], @"vertex colors ride on the v line");
	XCTAssertTrue([text containsString:@"\nf 1/1/1 3/3/3 2/2/2\n"], @"faces index 1-based with texture and normal");

	read = [NFKMesh meshWithOBJData:self.bareTriangle.OBJData error:&error];
	XCTAssertEqualObjects(read, self.bareTriangle, @"%@", error);
}

- (void)testOBJReadingResolvesPerFaceAttributeIndicesAndFansPolygons
{
	NSString *text = @"# a quad whose corners share positions but not texture coordinates\n"
		"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
		"vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
		"vn 0 0 1\n"
		"f 1/1/1 2/2/1 3/3/1 4/4/1\n"
		"f -4/5/1 -3/5/1 -2/5/1\n";
	NSError *error = nil;
	NFKMesh *mesh = [NFKMesh meshWithOBJData:[text dataUsingEncoding:NSUTF8StringEncoding] error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	XCTAssertEqual(mesh.faceCount, 3u, @"the quad fans into two triangles");
	XCTAssertEqual(mesh.vertexCount, 7u, @"the second face's corners carry a fifth texture coordinate, so they are new vertices");
	XCTAssertNotNil(mesh.normals);
	XCTAssertNil(mesh.colors);
	XCTAssertTrue(NFKUVEqual([mesh textureCoordinateAtIndex:6], NFKUV(0.5f, 0.5f)));
	XCTAssertTrue(NFKVectorEqual([mesh positionAtIndex:6], NFKMeshVectorMake(1, 1, 0)));

	XCTAssertNil([NFKMesh meshWithOBJData:[@"v 0 0 0\nf 1 2 3\n" dataUsingEncoding:NSUTF8StringEncoding] error:&error]);
	XCTAssertEqualObjects(error.domain, NSCocoaErrorDomain);
	XCTAssertEqual(error.code, NSFileReadCorruptFileError);
}

- (void)testPLYRoundTripQuantizesColorsToEightBits
{
	NFKMesh *mesh = self.tetrahedron;
	NSError *error = nil;
	NFKMesh *read = [NFKMesh meshWithPLYData:mesh.PLYData error:&error];
	XCTAssertNotNil(read, @"%@", error);
	NFKAssertMeshesClose(self, mesh, read, 0.5f / 255 + 1e-6f);

	NSString *header = [[NSString alloc] initWithData:[mesh.PLYData subdataWithRange:NSMakeRange(0, 60)] encoding:NSASCIIStringEncoding];
	XCTAssertTrue([header hasPrefix:@"ply\nformat binary_little_endian 1.0\n"]);
	XCTAssertEqualObjects([NFKMesh meshWithPLYData:self.bareTriangle.PLYData error:&error], self.bareTriangle, @"%@", error);
}

- (void)testPLYReadsTextAndBigEndianFormsWithOtherPropertyNames
{
	NSString *text = @"ply\nformat ascii 1.0\ncomment made by hand\n"
		"element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
		"property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
		"property float u\nproperty float v\n"
		"element face 1\nproperty list uchar int vertex_index\n"
		"element edge 1\nproperty int vertex1\nproperty int vertex2\n"
		"end_header\n"
		"0 0 0 255 0 0 255 0 0\n1 0 0 0 255 0 255 1 0\n0 1 0 0 0 255 128 0 1\n"
		"3 0 1 2\n"
		"0 1\n";
	NSError *error = nil;
	NFKMesh *mesh = [NFKMesh meshWithPLYData:[text dataUsingEncoding:NSASCIIStringEncoding] error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	XCTAssertEqual(mesh.vertexCount, 3u);
	XCTAssertEqual(mesh.faceCount, 1u);
	XCTAssertNil(mesh.normals);
	XCTAssertTrue(NFKVectorEqual([mesh colorAtIndex:2], NFKMeshVectorMake(0, 0, 1)));
	XCTAssertTrue(NFKUVEqual([mesh textureCoordinateAtIndex:1], NFKUV(1, 0)));
	XCTAssertTrue(NFKTriangleEqual([mesh faceAtIndex:0], NFKTri(0, 1, 2)));

	// The same triangle, binary big-endian, with double positions and a quad face.
	NSMutableData *big = [[@"ply\nformat binary_big_endian 1.0\nelement vertex 4\nproperty double x\nproperty double y\nproperty double z\n"
							"element face 1\nproperty list uchar uint vertex_indices\nend_header\n" dataUsingEncoding:NSASCIIStringEncoding] mutableCopy];
	double corners[12] = { 0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0 };
	for (int i = 0; i < 12; i++) {
		uint64_t bits;
		memcpy(&bits, &corners[i], 8);
		bits = CFSwapInt64HostToBig(bits);
		[big appendBytes:&bits length:8];
	}
	uint8_t four = 4;
	[big appendBytes:&four length:1];
	for (uint32_t index = 0; index < 4; index++) {
		uint32_t swapped = CFSwapInt32HostToBig(index);
		[big appendBytes:&swapped length:4];
	}
	mesh = [NFKMesh meshWithPLYData:big error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	XCTAssertEqual(mesh.faceCount, 2u, @"the quad fans");
	XCTAssertTrue(NFKVectorEqual([mesh positionAtIndex:2], NFKMeshVectorMake(1, 1, 0)));
	XCTAssertTrue(NFKTriangleEqual([mesh faceAtIndex:1], NFKTri(0, 2, 3)));

	XCTAssertNil([NFKMesh meshWithPLYData:[big subdataWithRange:NSMakeRange(0, big.length - 3)] error:&error], @"a truncated body is refused");
	XCTAssertEqual(error.code, NSFileReadCorruptFileError);
	XCTAssertNil([NFKMesh meshWithPLYData:[@"not a ply" dataUsingEncoding:NSASCIIStringEncoding] error:&error]);
}

- (void)testGLBRoundTripAndLayout
{
	NFKMesh *mesh = self.tetrahedron;
	NSData *glb = mesh.GLBData;
	const uint8_t *bytes = glb.bytes;
	XCTAssertEqual(memcmp(bytes, "glTF", 4), 0);
	uint32_t length;
	memcpy(&length, bytes + 8, 4);
	XCTAssertEqual(CFSwapInt32LittleToHost(length), (uint32_t)glb.length);
	XCTAssertEqual(glb.length % 4, 0u);

	uint32_t jsonLength;
	memcpy(&jsonLength, bytes + 12, 4);
	NSError *error = nil;
	NSDictionary *document = [NSJSONSerialization JSONObjectWithData:[glb subdataWithRange:NSMakeRange(20, CFSwapInt32LittleToHost(jsonLength))]
															options:0 error:&error];
	XCTAssertNotNil(document, @"%@", error);
	XCTAssertEqualObjects(document[@"asset"][@"version"], @"2.0");
	NSDictionary *primitive = document[@"meshes"][0][@"primitives"][0];
	XCTAssertEqualObjects([NSSet setWithArray:[primitive[@"attributes"] allKeys]],
						  ([NSSet setWithArray:@[ @"POSITION", @"NORMAL", @"COLOR_0", @"TEXCOORD_0" ]]));
	NSDictionary *positions = document[@"accessors"][[primitive[@"attributes"][@"POSITION"] unsignedIntegerValue]];
	XCTAssertEqualObjects(positions[@"max"], (@[ @1, @1, @1 ]));
	XCTAssertEqualObjects(positions[@"componentType"], @5126);
	NSDictionary *indices = document[@"accessors"][[primitive[@"indices"] unsignedIntegerValue]];
	XCTAssertEqualObjects(indices[@"count"], @12);
	XCTAssertEqualObjects(indices[@"componentType"], @5125);

	NFKMesh *read = [NFKMesh meshWithGLBData:glb error:&error];
	XCTAssertNotNil(read, @"%@", error);
	NFKAssertMeshesClose(self, mesh, read, 1e-6f);
	XCTAssertEqualObjects([NFKMesh meshWithGLBData:self.bareTriangle.GLBData error:&error], self.bareTriangle, @"%@", error);

	XCTAssertNil([NFKMesh meshWithGLBData:[glb subdataWithRange:NSMakeRange(0, 20 + CFSwapInt32LittleToHost(jsonLength))] error:&error],
				 @"a file with no binary chunk is refused");
	XCTAssertEqual(error.code, NSFileReadCorruptFileError);
}

- (void)testWritingAndReadingByExtension
{
	NFKMesh *mesh = self.tetrahedron;
	NSURL *directory = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	NSError *error = nil;
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error], @"%@", error);
	for (NSString *extension in @[ @"obj", @"ply", @"glb" ]) {
		NSURL *url = [directory URLByAppendingPathComponent:[@"object." stringByAppendingString:extension]];
		XCTAssertTrue([mesh writeToURL:url error:&error], @"%@", error);
		NFKMesh *read = [NFKMesh meshWithContentsOfURL:url error:&error];
		XCTAssertNotNil(read, @"%@", error);
		XCTAssertEqual(read.vertexCount, 4u, @"%@", extension);
		XCTAssertEqual(read.faceCount, 4u, @"%@", extension);
	}
	NSURL *other = [directory URLByAppendingPathComponent:@"object.stl"];
	XCTAssertFalse([mesh writeToURL:other error:&error]);
	XCTAssertEqualObjects(error.domain, NSCocoaErrorDomain);
	XCTAssertNil([NFKMesh meshWithContentsOfURL:other error:&error]);
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];
}

#pragma mark - Wire and value coding

- (void)testTheInlineFormCarriesThePLYAndTheFileFormWritesIt
{
	NFKMesh *mesh = self.tetrahedron;
	NFKValueCoding *inlineForm = [[NFKValueCoding alloc] init];
	NSError *error = nil;
	NSDictionary *entry = [inlineForm JSONValueForValue:mesh key:NFKOutputMesh error:&error];
	XCTAssertEqualObjects(entry[@"$nfk"], @"mesh", @"%@", error);
	XCTAssertEqualObjects(entry[@"vertexCount"], @4);
	XCTAssertEqualObjects(entry[@"faceCount"], @4);
	XCTAssertEqualObjects(entry[@"colors"], @YES);
	XCTAssertTrue([entry[@"ply"] isKindOfClass:NSString.class]);
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:entry]);
	NFKMesh *decoded = [inlineForm valueFromJSONValue:entry key:NFKOutputMesh error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	NFKAssertMeshesClose(self, mesh, decoded, 0.5f / 255 + 1e-6f);
	XCTAssertEqualObjects([inlineForm valueOfType:NFKModelValueTypeMesh fromJSONValue:entry error:&error], decoded, @"%@", error);

	NSURL *directory = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error], @"%@", error);
	NFKValueCoding *files = [[NFKValueCoding alloc] initWithDirectoryURL:directory];
	entry = [files JSONValueForValue:mesh key:NFKOutputMesh error:&error];
	XCTAssertEqualObjects(entry[@"$nfk"], @"mesh", @"%@", error);
	XCTAssertNil(entry[@"ply"], @"the file form names the PLY instead of carrying it");
	XCTAssertEqualObjects(entry[@"path"], @"mesh.ply");
	XCTAssertEqualObjects(entry[@"vertexCount"], @4);
	NFKMesh *fromFile = [NFKMesh meshWithContentsOfURL:[directory URLByAppendingPathComponent:entry[@"path"]] error:&error];
	XCTAssertNotNil(fromFile, @"%@", error);
	decoded = [files valueFromJSONValue:entry key:NFKOutputMesh error:&error];
	XCTAssertNotNil(decoded, @"%@", error);
	NFKAssertMeshesClose(self, mesh, decoded, 0.5f / 255 + 1e-6f);

	NSURL *obj = [directory URLByAppendingPathComponent:@"object.obj"];
	XCTAssertTrue([mesh writeToURL:obj error:&error], @"%@", error);
	XCTAssertEqualObjects([files valueOfType:NFKModelValueTypeMesh fromFileURL:obj error:&error], mesh, @"%@", error);
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];

	NSDictionary *withoutBytes = @{ @"$nfk": @"mesh", @"vertexCount": @4 };
	XCTAssertNil([inlineForm valueFromJSONValue:withoutBytes key:NFKOutputMesh error:&error], @"an object without bytes is refused");
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
}

@end
