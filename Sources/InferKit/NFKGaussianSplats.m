//
//  NFKGaussianSplats.m
//  InferKit
//

#import "NFKGaussianSplats.h"
#import "NFKErrors.h"
#import <math.h>

/*! The constant degree-0 spherical-harmonic basis function, 1 / (2√π). */
static const float NFKSplatsHarmonicDC = 0.28209479177387814f;

static BOOL NFKSplatsFail(NSError **error, NSInteger code, NSString *domain, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

static BOOL NFKSplatsInvalid(NSError **error, NSString *reason)
{
	return NFKSplatsFail(error, kNFKError_InferenceUnsupported, NFKInferenceErrorDomain,
						 [NSString stringWithFormat:@"The Gaussians are not well formed: %@.", reason]);
}

static BOOL NFKSplatsReadFail(NSError **error, NSString *reason)
{
	return NFKSplatsFail(error, NSFileReadCorruptFileError, NSCocoaErrorDomain,
						 [NSString stringWithFormat:@"The Gaussian splatting PLY file cannot be read: %@.", reason]);
}

/*! The PLY vertex properties in the order the file lists them. */
static NSArray<NSString *> *NFKSplatsPLYProperties(void)
{
	return @[ @"x", @"y", @"z", @"nx", @"ny", @"nz", @"f_dc_0", @"f_dc_1", @"f_dc_2", @"opacity",
			  @"scale_0", @"scale_1", @"scale_2", @"rot_0", @"rot_1", @"rot_2", @"rot_3" ];
}

/*! Half a turn about x, which carries this type's frame to the splatting format's and back: (x, −y, −z)
	for a point, and for a rotation q the product (0, 1, 0, 0) ⊗ q, kept with w ≥ 0. */
static NFKGaussianRotation NFKSplatsTurnedRotation(NFKGaussianRotation q)
{
	NFKGaussianRotation turned = { -q.x, q.w, -q.z, q.y };
	if (turned.w < 0) {
		turned = (NFKGaussianRotation){ -turned.w, -turned.x, -turned.y, -turned.z };
	}
	return turned;
}

@interface NFKGaussianSplats ()
@property (nonatomic, readwrite) NSUInteger count;
@property (nonatomic, readwrite, copy) NSData *positions;
@property (nonatomic, readwrite, copy) NSData *scales;
@property (nonatomic, readwrite, copy) NSData *rotations;
@property (nonatomic, readwrite, copy) NSData *opacities;
@property (nonatomic, readwrite, copy) NSData *colors;
@end

@implementation NFKGaussianSplats

- (nullable instancetype)initWithPositions:(NSData *)positions
									scales:(NSData *)scales
								 rotations:(NSData *)rotations
								 opacities:(NSData *)opacities
									colors:(NSData *)colors
									 error:(NSError **)error
{
	if (positions.length % (3 * sizeof(float)) != 0) {
		NFKSplatsInvalid(error, @"the positions are not three floats a Gaussian");
		return nil;
	}
	NSUInteger count = positions.length / (3 * sizeof(float));
	NSDictionary<NSString *, NSArray *> *attributes = @{ @"scales": @[ scales, @3 ], @"rotations": @[ rotations, @4 ],
														 @"opacities": @[ opacities, @1 ], @"colors": @[ colors, @3 ] };
	for (NSString *name in attributes) {
		NSData *data = attributes[name][0];
		NSUInteger width = [attributes[name][1] unsignedIntegerValue];
		if (data.length != count * width * sizeof(float)) {
			NFKSplatsInvalid(error, [NSString stringWithFormat:@"the %@ do not hold %lu floats for each of %lu Gaussians", name,
																(unsigned long)width, (unsigned long)count]);
			return nil;
		}
	}
	self = [super init];
	if (self != nil) {
		self.count = count;
		self.positions = positions;
		self.scales = scales;
		self.rotations = rotations;
		self.opacities = opacities;
		self.colors = colors;
	}
	return self;
}

#pragma mark - Access

static NFKMeshVector NFKSplatsVector(NSData *data, NSUInteger index)
{
	const float *values = (const float *)data.bytes + index * 3;
	return NFKMeshVectorMake(values[0], values[1], values[2]);
}

- (NFKMeshVector)positionAtIndex:(NSUInteger)index
{
	return NFKSplatsVector(self.positions, index);
}

- (NFKMeshVector)scaleAtIndex:(NSUInteger)index
{
	return NFKSplatsVector(self.scales, index);
}

- (NFKGaussianRotation)rotationAtIndex:(NSUInteger)index
{
	const float *values = (const float *)self.rotations.bytes + index * 4;
	return (NFKGaussianRotation){ values[0], values[1], values[2], values[3] };
}

- (float)opacityAtIndex:(NSUInteger)index
{
	return ((const float *)self.opacities.bytes)[index];
}

- (NFKMeshVector)colorAtIndex:(NSUInteger)index
{
	NFKMeshVector coefficients = NFKSplatsVector(self.colors, index);
	float (^shown)(float) = ^float(float coefficient) {
		return fminf(fmaxf(0.5f + NFKSplatsHarmonicDC * coefficient, 0), 1);
	};
	return NFKMeshVectorMake(shown(coefficients.x), shown(coefficients.y), shown(coefficients.z));
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
	if (![object isKindOfClass:NFKGaussianSplats.class]) {
		return NO;
	}
	NFKGaussianSplats *other = object;
	return [self.positions isEqualToData:other.positions] && [self.scales isEqualToData:other.scales]
		&& [self.rotations isEqualToData:other.rotations] && [self.opacities isEqualToData:other.opacities]
		&& [self.colors isEqualToData:other.colors];
}

- (NSUInteger)hash
{
	return self.positions.hash ^ self.colors.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %lu Gaussians>", NSStringFromClass(self.class), (unsigned long)self.count];
}

#pragma mark - PLY

- (NSData *)PLYData
{
	NSArray<NSString *> *properties = NFKSplatsPLYProperties();
	NSMutableString *header = [NSMutableString stringWithFormat:@"ply\nformat binary_little_endian 1.0\nelement vertex %lu\n",
										   (unsigned long)self.count];
	for (NSString *property in properties) {
		[header appendFormat:@"property float %@\n", property];
	}
	[header appendString:@"end_header\n"];
	NSMutableData *data = [[header dataUsingEncoding:NSASCIIStringEncoding] mutableCopy];
	NSMutableData *body = [NSMutableData dataWithLength:self.count * properties.count * sizeof(float)];
	float *row = body.mutableBytes;
	for (NSUInteger index = 0; index < self.count; index++, row += properties.count) {
		NFKMeshVector position = [self positionAtIndex:index];
		NFKMeshVector scale = [self scaleAtIndex:index];
		NFKMeshVector color = NFKSplatsVector(self.colors, index);
		NFKGaussianRotation rotation = NFKSplatsTurnedRotation([self rotationAtIndex:index]);
		float opacity = [self opacityAtIndex:index];
		float values[17] = { position.x, -position.y, -position.z, 0, 0, 0, color.x, color.y, color.z,
							 logf(opacity / (1 - opacity)), logf(scale.x), logf(scale.y), logf(scale.z),
							 rotation.w, rotation.x, rotation.y, rotation.z };
		memcpy(row, values, sizeof(values));
	}
	// The format is little-endian, as every supported platform is.
	[data appendData:body];
	return data;
}

- (BOOL)writeToURL:(NSURL *)url error:(NSError **)error
{
	return [self.PLYData writeToURL:url options:NSDataWritingAtomic error:error];
}

+ (nullable instancetype)splatsWithContentsOfURL:(NSURL *)url error:(NSError **)error
{
	NSData *data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:error];
	return data != nil ? [self splatsWithPLYData:data error:error] : nil;
}

+ (nullable instancetype)splatsWithPLYData:(NSData *)data error:(NSError **)error
{
	NSData *terminator = [@"end_header\n" dataUsingEncoding:NSASCIIStringEncoding];
	NSRange end = [data rangeOfData:terminator options:0 range:NSMakeRange(0, MIN(data.length, (NSUInteger)65536))];
	if (end.location == NSNotFound) {
		NFKSplatsReadFail(error, @"it has no end_header line");
		return nil;
	}
	NSString *header = [[NSString alloc] initWithData:[data subdataWithRange:NSMakeRange(0, end.location)] encoding:NSASCIIStringEncoding];
	NSArray<NSString *> *lines = [header componentsSeparatedByString:@"\n"];
	if (lines.count < 2 || ![lines[0] isEqualToString:@"ply"] || ![lines[1] isEqualToString:@"format binary_little_endian 1.0"]) {
		NFKSplatsReadFail(error, @"it is not a binary little-endian PLY file");
		return nil;
	}
	NSUInteger count = 0;
	BOOL inVertex = NO;
	NSMutableArray<NSString *> *names = [NSMutableArray array];
	for (NSString *line in lines) {
		NSArray<NSString *> *words = [line componentsSeparatedByString:@" "];
		if ([words.firstObject isEqualToString:@"element"]) {
			inVertex = words.count == 3 && [words[1] isEqualToString:@"vertex"];
			if (!inVertex) {
				NFKSplatsReadFail(error, @"it holds an element other than vertex");
				return nil;
			}
			count = (NSUInteger)words[2].longLongValue;
		} else if ([words.firstObject isEqualToString:@"property"]) {
			if (words.count != 3 || ![words[1] isEqualToString:@"float"]) {
				NFKSplatsReadFail(error, [NSString stringWithFormat:@"the property \"%@\" is not a float", line]);
				return nil;
			}
			[names addObject:words[2]];
		}
	}
	NSMutableDictionary<NSString *, NSNumber *> *column = [NSMutableDictionary dictionary];
	[names enumerateObjectsUsingBlock:^(NSString *name, NSUInteger index, BOOL *stop) {
		column[name] = @(index);
	}];
	for (NSString *required in NFKSplatsPLYProperties()) {
		if (column[required] == nil && ![required hasPrefix:@"n"]) {
			NFKSplatsReadFail(error, [NSString stringWithFormat:@"it has no %@ property", required]);
			return nil;
		}
	}
	NSUInteger start = NSMaxRange(end), stride = names.count * sizeof(float);
	if (data.length < start + count * stride) {
		NFKSplatsReadFail(error, @"it is shorter than its header says");
		return nil;
	}
	NSMutableData *positions = [NSMutableData dataWithLength:count * 3 * sizeof(float)];
	NSMutableData *scales = [NSMutableData dataWithLength:count * 3 * sizeof(float)];
	NSMutableData *rotations = [NSMutableData dataWithLength:count * 4 * sizeof(float)];
	NSMutableData *opacities = [NSMutableData dataWithLength:count * sizeof(float)];
	NSMutableData *colors = [NSMutableData dataWithLength:count * 3 * sizeof(float)];
	float *p = positions.mutableBytes, *s = scales.mutableBytes, *r = rotations.mutableBytes, *o = opacities.mutableBytes,
		  *c = colors.mutableBytes;
	static NSString * const read[14] = { @"x", @"y", @"z", @"f_dc_0", @"f_dc_1", @"f_dc_2", @"opacity", @"scale_0", @"scale_1",
										 @"scale_2", @"rot_0", @"rot_1", @"rot_2", @"rot_3" };
	NSUInteger at[14] = { 0 };
	for (NSUInteger k = 0; k < 14; k++) {
		at[k] = column[read[k]].unsignedIntegerValue;
	}
	NSMutableData *rowStorage = [NSMutableData dataWithLength:stride];
	float *row = rowStorage.mutableBytes;
	const uint8_t *bytes = (const uint8_t *)data.bytes + start;
	for (NSUInteger index = 0; index < count; index++) {
		memcpy(row, bytes + index * stride, stride);
		NFKGaussianRotation rotation = NFKSplatsTurnedRotation((NFKGaussianRotation){ row[at[10]], row[at[11]], row[at[12]], row[at[13]] });
		float length = sqrtf(rotation.w * rotation.w + rotation.x * rotation.x + rotation.y * rotation.y + rotation.z * rotation.z);
		if (length > 0) {
			rotation = (NFKGaussianRotation){ rotation.w / length, rotation.x / length, rotation.y / length, rotation.z / length };
		}
		float point[3] = { row[at[0]], -row[at[1]], -row[at[2]] };
		float scale[3] = { expf(row[at[7]]), expf(row[at[8]]), expf(row[at[9]]) };
		float quaternion[4] = { rotation.w, rotation.x, rotation.y, rotation.z };
		float color[3] = { row[at[3]], row[at[4]], row[at[5]] };
		memcpy(p + index * 3, point, sizeof(point));
		memcpy(s + index * 3, scale, sizeof(scale));
		memcpy(r + index * 4, quaternion, sizeof(quaternion));
		memcpy(c + index * 3, color, sizeof(color));
		o[index] = 1 / (1 + expf(-row[at[6]]));
	}
	return [[self alloc] initWithPositions:positions scales:scales rotations:rotations opacities:opacities colors:colors error:error];
}

#pragma mark - NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.positions forKey:@"positions"];
	[coder encodeObject:self.scales forKey:@"scales"];
	[coder encodeObject:self.rotations forKey:@"rotations"];
	[coder encodeObject:self.opacities forKey:@"opacities"];
	[coder encodeObject:self.colors forKey:@"colors"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSData * (^read)(NSString *) = ^NSData *(NSString *key) {
		return [coder decodeObjectOfClass:NSData.class forKey:key] ?: [NSData data];
	};
	return [self initWithPositions:read(@"positions") scales:read(@"scales") rotations:read(@"rotations")
						 opacities:read(@"opacities") colors:read(@"colors") error:NULL];
}

@end
