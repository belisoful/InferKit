//
//  NFKMeshMaterialTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKMesh.h>
#import <InferKit/NFKMeshMaterial.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKValueCoding.h>
#import <InferKit/NFKErrors.h>

@interface NFKMeshMaterialTests : XCTestCase
@end

@implementation NFKMeshMaterialTests

/*! Bytes standing in for an image file. The types carry an image without decoding it. */
- (NSData *)imageBytes:(uint8_t)seed
{
	uint8_t bytes[24] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
	for (NSUInteger i = 8; i < sizeof bytes; i++) {
		bytes[i] = (uint8_t)(seed + i * 7);
	}
	return [NSData dataWithBytes:bytes length:sizeof bytes];
}

- (NFKMeshTexture *)textureWithSeed:(uint8_t)seed MIMEType:(NSString *)MIMEType
{
	NSError *error = nil;
	NFKMeshTexture *texture = [[NFKMeshTexture alloc] initWithData:[self imageBytes:seed] MIMEType:MIMEType error:&error];
	XCTAssertNotNil(texture, @"%@", error);
	return texture;
}

- (NFKMeshMaterial *)texturedMaterial
{
	NSError *error = nil;
	NSArray<NSNumber *> *factor = @[ @1.0f, @0.5f, @0.25f, @1.0f ];
	NFKMeshMaterial *material = [[NFKMeshMaterial alloc] initWithBaseColorTexture:[self textureWithSeed:3 MIMEType:NFKMeshTextureMIMETypeJPEG]
																	 normalTexture:[self textureWithSeed:9 MIMEType:NFKMeshTextureMIMETypePNG]
														  metallicRoughnessTexture:nil
																   baseColorFactor:factor
																	metallicFactor:0.125f
																   roughnessFactor:0.75f
																			 error:&error];
	XCTAssertNotNil(material, @"%@", error);
	return material;
}

/*! A unit quad of two triangles with texture coordinates. */
- (NFKMesh *)quadWithMaterial:(NFKMeshMaterial *)material
{
	float positions[12] = { 0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0 };
	float textures[8] = { 0, 0,  1, 0,  1, 1,  0, 1 };
	uint32_t indices[6] = { 0, 1, 2,  0, 2, 3 };
	NSError *error = nil;
	NFKMesh *mesh = [[NFKMesh alloc] initWithPositions:[NSData dataWithBytes:positions length:sizeof positions]
											  normals:nil
											   colors:nil
								   textureCoordinates:[NSData dataWithBytes:textures length:sizeof textures]
											  indices:[NSData dataWithBytes:indices length:sizeof indices]
											 material:material
												error:&error];
	XCTAssertNotNil(mesh, @"%@", error);
	return mesh;
}

#pragma mark - Construction

- (void)testATextureKeepsItsBytesAndRefusesAnEmptyOrUnknownImage
{
	NFKMeshTexture *texture = [self textureWithSeed:1 MIMEType:NFKMeshTextureMIMETypePNG];
	XCTAssertEqualObjects(texture.data, [self imageBytes:1]);
	XCTAssertEqualObjects(texture.MIMEType, @"image/png");
	XCTAssertEqualObjects(NFKMeshTextureMIMETypeJPEG, @"image/jpeg");

	NSError *error = nil;
	XCTAssertNil([[NFKMeshTexture alloc] initWithData:[NSData data] MIMEType:NFKMeshTextureMIMETypePNG error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	error = nil;
	XCTAssertNil([[NFKMeshTexture alloc] initWithData:[self imageBytes:1] MIMEType:@"image/webp" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

- (void)testAMaterialDefaultsToAWhiteBaseAndRefusesFactorsOutsideTheUnitRange
{
	NSError *error = nil;
	NFKMeshMaterial *plain = [[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
															   baseColorFactor:nil metallicFactor:0 roughnessFactor:1 error:&error];
	XCTAssertNotNil(plain, @"%@", error);
	NSArray<NSNumber *> *white = @[ @1.0f, @1.0f, @1.0f, @1.0f ];
	XCTAssertEqualObjects(plain.baseColorFactor, white);
	XCTAssertNil(plain.baseColorTexture);
	XCTAssertEqual(plain.metallicFactor, 0.0f);
	XCTAssertEqual(plain.roughnessFactor, 1.0f);

	NSArray<NSNumber *> *three = @[ @1, @1, @1 ];
	XCTAssertNil([[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
												  baseColorFactor:three metallicFactor:0 roughnessFactor:1 error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	NSArray<NSNumber *> *bright = @[ @1, @2, @1, @1 ];
	XCTAssertNil([[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
												  baseColorFactor:bright metallicFactor:0 roughnessFactor:1 error:&error]);
	XCTAssertNil([[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
												  baseColorFactor:nil metallicFactor:-0.5f roughnessFactor:1 error:&error]);
	XCTAssertNil([[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
												  baseColorFactor:nil metallicFactor:0 roughnessFactor:NAN error:&error]);
}

- (void)testATexturedMaterialNeedsTextureCoordinates
{
	float positions[9] = { 0, 0, 0,  1, 0, 0,  0, 1, 0 };
	uint32_t indices[3] = { 0, 1, 2 };
	NSError *error = nil;
	NFKMesh *mesh = [[NFKMesh alloc] initWithPositions:[NSData dataWithBytes:positions length:sizeof positions]
											  normals:nil colors:nil textureCoordinates:nil
											  indices:[NSData dataWithBytes:indices length:sizeof indices]
											 material:self.texturedMaterial error:&error];
	XCTAssertNil(mesh);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

- (void)testEqualityCopyingAndArchivingIncludeTheMaterial
{
	NFKMesh *mesh = [self quadWithMaterial:self.texturedMaterial];
	XCTAssertEqualObjects(mesh, [self quadWithMaterial:self.texturedMaterial]);
	XCTAssertNotEqualObjects(mesh, [self quadWithMaterial:nil], @"a material is part of the value");
	XCTAssertEqualObjects(mesh.material, [mesh.material copy]);
	XCTAssertTrue([mesh.description containsString:@"material"]);

	NSError *error = nil;
	NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:mesh requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(archive, @"%@", error);
	NSSet *classes = [NSSet setWithObjects:NFKMesh.class, NFKMeshMaterial.class, NFKMeshTexture.class, nil];
	NFKMesh *restored = [NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:archive error:&error];
	XCTAssertEqualObjects(restored, mesh, @"%@", error);
	XCTAssertEqualObjects(restored.material.normalTexture.MIMEType, NFKMeshTextureMIMETypePNG);
}

#pragma mark - GLB

- (void)testGLBEmbedsTheTexturesAndReadsTheMaterialBack
{
	NFKMesh *mesh = [self quadWithMaterial:self.texturedMaterial];
	NSData *glb = mesh.GLBData;
	uint32_t jsonLength = 0;
	[glb getBytes:&jsonLength range:NSMakeRange(12, 4)];
	NSDictionary *document = [NSJSONSerialization JSONObjectWithData:[glb subdataWithRange:NSMakeRange(20, jsonLength)] options:0 error:NULL];
	XCTAssertEqualObjects(document[@"meshes"][0][@"primitives"][0][@"material"], @0);
	XCTAssertEqual([document[@"images"] count], (NSUInteger)2);
	XCTAssertEqual([document[@"textures"] count], (NSUInteger)2);
	XCTAssertEqual([document[@"samplers"] count], (NSUInteger)1);
	NSDictionary *material = document[@"materials"][0];
	XCTAssertEqualObjects(material[@"pbrMetallicRoughness"][@"baseColorTexture"][@"index"], @0);
	XCTAssertEqualObjects(material[@"normalTexture"][@"index"], @1);
	XCTAssertEqualObjects(document[@"images"][0][@"mimeType"], @"image/jpeg");
	XCTAssertEqual([material[@"pbrMetallicRoughness"][@"roughnessFactor"] floatValue], 0.75f);

	NSError *error = nil;
	NFKMesh *read = [NFKMesh meshWithGLBData:glb error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read.material, mesh.material);
	XCTAssertEqualObjects(read.material.baseColorTexture.data, [self imageBytes:3]);
	XCTAssertEqualObjects(read, mesh);
}

- (void)testGLBCarriesAMaterialOfFactorsAlone
{
	NSError *error = nil;
	NSArray<NSNumber *> *factor = @[ @0.5f, @0.5f, @0.5f, @1.0f ];
	NFKMeshMaterial *flat = [[NFKMeshMaterial alloc] initWithBaseColorTexture:nil normalTexture:nil metallicRoughnessTexture:nil
															  baseColorFactor:factor metallicFactor:1 roughnessFactor:0.25f error:&error];
	XCTAssertNotNil(flat, @"%@", error);
	NFKMesh *mesh = [self quadWithMaterial:flat];
	NFKMesh *read = [NFKMesh meshWithGLBData:mesh.GLBData error:&error];
	XCTAssertEqualObjects(read.material, flat, @"%@", error);
	XCTAssertNil(read.material.baseColorTexture);
}

- (void)testOBJAndPLYCarryNoMaterial
{
	NFKMesh *mesh = [self quadWithMaterial:self.texturedMaterial];
	NSError *error = nil;
	XCTAssertNil([NFKMesh meshWithOBJData:mesh.OBJData error:&error].material, @"%@", error);
	XCTAssertNil([NFKMesh meshWithPLYData:mesh.PLYData error:&error].material, @"%@", error);
}

#pragma mark - Wire coding

- (void)testTheWireFormsCarryAMaterialAsGLB
{
	NFKMesh *mesh = [self quadWithMaterial:self.texturedMaterial];
	NFKValueCoding *inlineForm = [[NFKValueCoding alloc] init];
	NSError *error = nil;
	NSDictionary *entry = [inlineForm JSONValueForValue:mesh key:NFKOutputMesh error:&error];
	XCTAssertEqualObjects(entry[@"material"], @YES, @"%@", error);
	XCTAssertTrue([entry[@"glb"] isKindOfClass:NSString.class]);
	XCTAssertNil(entry[@"ply"], @"a PLY would drop the material");
	XCTAssertEqualObjects([inlineForm valueFromJSONValue:entry key:NFKOutputMesh error:&error], mesh, @"%@", error);

	NSURL *directory = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error], @"%@", error);
	NFKValueCoding *files = [[NFKValueCoding alloc] initWithDirectoryURL:directory];
	entry = [files JSONValueForValue:mesh key:NFKOutputMesh error:&error];
	XCTAssertEqualObjects(entry[@"path"], @"mesh.glb", @"%@", error);
	XCTAssertNil(entry[@"glb"], @"the file form names the GLB instead of carrying it");
	XCTAssertEqualObjects([files valueFromJSONValue:entry key:NFKOutputMesh error:&error], mesh, @"%@", error);
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];
}

@end
