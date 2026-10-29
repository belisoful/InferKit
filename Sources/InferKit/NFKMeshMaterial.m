//
//  NFKMeshMaterial.m
//  InferKit
//

#import "NFKMeshMaterial.h"
#import "NFKErrors.h"

NSString * const NFKMeshTextureMIMETypePNG = @"image/png";
NSString * const NFKMeshTextureMIMETypeJPEG = @"image/jpeg";

static BOOL NFKMeshMaterialInvalid(NSError **error, NSString *reason)
{
	if (error != NULL) {
		NSString *description = [NSString stringWithFormat:@"The mesh material is not well formed: %@.", reason];
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey: description }];
	}
	return NO;
}

static BOOL NFKMeshMaterialSame(id a, id b)
{
	return (a == nil && b == nil) || [a isEqual:b];
}

@interface NFKMeshTexture ()
@property (nonatomic, copy) NSData *data;
@property (nonatomic, copy) NSString *MIMEType;
@end

@implementation NFKMeshTexture

- (nullable instancetype)initWithData:(NSData *)data MIMEType:(NSString *)MIMEType error:(NSError **)error
{
	self = [super init];
	if (self == nil) {
		return nil;
	}
	if (data.length == 0) {
		NFKMeshMaterialInvalid(error, @"a texture holds no image");
		return nil;
	}
	if (![MIMEType isEqualToString:NFKMeshTextureMIMETypePNG] && ![MIMEType isEqualToString:NFKMeshTextureMIMETypeJPEG]) {
		NFKMeshMaterialInvalid(error, [NSString stringWithFormat:@"a texture is %@, not PNG or JPEG", MIMEType]);
		return nil;
	}
	_data = [data copy];
	_MIMEType = [MIMEType copy];
	return self;
}

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
	if (![object isKindOfClass:NFKMeshTexture.class]) {
		return NO;
	}
	NFKMeshTexture *other = object;
	return [self.MIMEType isEqualToString:other.MIMEType] && [self.data isEqualToData:other.data];
}

- (NSUInteger)hash
{
	return self.data.hash ^ self.MIMEType.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@, %lu bytes>", NSStringFromClass(self.class), self.MIMEType,
			(unsigned long)self.data.length];
}

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.data forKey:@"data"];
	[coder encodeObject:self.MIMEType forKey:@"MIMEType"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	return [self initWithData:[coder decodeObjectOfClass:NSData.class forKey:@"data"] ?: [NSData data]
					 MIMEType:[coder decodeObjectOfClass:NSString.class forKey:@"MIMEType"] ?: @""
						error:NULL];
}

@end

@interface NFKMeshMaterial ()
@property (nonatomic, copy, nullable) NFKMeshTexture *baseColorTexture;
@property (nonatomic, copy, nullable) NFKMeshTexture *normalTexture;
@property (nonatomic, copy, nullable) NFKMeshTexture *metallicRoughnessTexture;
@property (nonatomic, copy) NSArray<NSNumber *> *baseColorFactor;
@property (nonatomic) float metallicFactor;
@property (nonatomic) float roughnessFactor;
@end

@implementation NFKMeshMaterial

- (nullable instancetype)initWithBaseColorTexture:(nullable NFKMeshTexture *)baseColorTexture
									normalTexture:(nullable NFKMeshTexture *)normalTexture
						 metallicRoughnessTexture:(nullable NFKMeshTexture *)metallicRoughnessTexture
								  baseColorFactor:(nullable NSArray<NSNumber *> *)baseColorFactor
								   metallicFactor:(float)metallicFactor
								  roughnessFactor:(float)roughnessFactor
											error:(NSError **)error
{
	self = [super init];
	if (self == nil) {
		return nil;
	}
	NSArray<NSNumber *> *factor = baseColorFactor ?: @[ @1.0f, @1.0f, @1.0f, @1.0f ];
	if (factor.count != 4) {
		NFKMeshMaterialInvalid(error, @"the base color factor does not hold four numbers");
		return nil;
	}
	for (id component in factor) {
		if (![component isKindOfClass:NSNumber.class] || [component floatValue] < 0 || [component floatValue] > 1) {
			NFKMeshMaterialInvalid(error, @"a base color factor lies outside 0...1");
			return nil;
		}
	}
	if (!(metallicFactor >= 0 && metallicFactor <= 1) || !(roughnessFactor >= 0 && roughnessFactor <= 1)) {
		NFKMeshMaterialInvalid(error, @"the metallic or roughness factor lies outside 0...1");
		return nil;
	}
	NSMutableArray<NSNumber *> *floats = [NSMutableArray arrayWithCapacity:4];
	for (NSNumber *component in factor) {
		[floats addObject:@(component.floatValue)];
	}
	_baseColorTexture = baseColorTexture;
	_normalTexture = normalTexture;
	_metallicRoughnessTexture = metallicRoughnessTexture;
	_baseColorFactor = [floats copy];
	_metallicFactor = metallicFactor;
	_roughnessFactor = roughnessFactor;
	return self;
}

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
	if (![object isKindOfClass:NFKMeshMaterial.class]) {
		return NO;
	}
	NFKMeshMaterial *other = object;
	return NFKMeshMaterialSame(self.baseColorTexture, other.baseColorTexture)
		&& NFKMeshMaterialSame(self.normalTexture, other.normalTexture)
		&& NFKMeshMaterialSame(self.metallicRoughnessTexture, other.metallicRoughnessTexture)
		&& [self.baseColorFactor isEqualToArray:other.baseColorFactor]
		&& self.metallicFactor == other.metallicFactor && self.roughnessFactor == other.roughnessFactor;
}

- (NSUInteger)hash
{
	return self.baseColorTexture.hash ^ self.normalTexture.hash ^ self.baseColorFactor.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ metallic %g, roughness %g%@%@%@>", NSStringFromClass(self.class),
			self.metallicFactor, self.roughnessFactor, self.baseColorTexture != nil ? @", base color texture" : @"",
			self.normalTexture != nil ? @", normal texture" : @"",
			self.metallicRoughnessTexture != nil ? @", metallic-roughness texture" : @""];
}

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.baseColorTexture forKey:@"baseColorTexture"];
	[coder encodeObject:self.normalTexture forKey:@"normalTexture"];
	[coder encodeObject:self.metallicRoughnessTexture forKey:@"metallicRoughnessTexture"];
	[coder encodeObject:self.baseColorFactor forKey:@"baseColorFactor"];
	[coder encodeFloat:self.metallicFactor forKey:@"metallicFactor"];
	[coder encodeFloat:self.roughnessFactor forKey:@"roughnessFactor"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSSet *factorClasses = [NSSet setWithObjects:NSArray.class, NSNumber.class, nil];
	return [self initWithBaseColorTexture:[coder decodeObjectOfClass:NFKMeshTexture.class forKey:@"baseColorTexture"]
							normalTexture:[coder decodeObjectOfClass:NFKMeshTexture.class forKey:@"normalTexture"]
				 metallicRoughnessTexture:[coder decodeObjectOfClass:NFKMeshTexture.class forKey:@"metallicRoughnessTexture"]
						  baseColorFactor:[coder decodeObjectOfClasses:factorClasses forKey:@"baseColorFactor"]
						   metallicFactor:[coder decodeFloatForKey:@"metallicFactor"]
						  roughnessFactor:[coder decodeFloatForKey:@"roughnessFactor"]
									error:NULL];
}

@end
