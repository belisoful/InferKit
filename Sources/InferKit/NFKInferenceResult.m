//
//  NFKInferenceResult.m
//  InferKit
//

#import "NFKInferenceResult.h"
#import "NFKInferenceKeys.h"
#import "NFK_ARC.h"

@implementation NFKInferenceResult

+ (instancetype)resultWithOutputs:(NSDictionary<NSString *, id> *)outputs
{
	return NARC_AUTORELEASE([[self alloc] initWithOutputs:outputs]);
}

- (instancetype)initWithOutputs:(NSDictionary<NSString *, id> *)outputs
{
	self = [super init];
	if (self != nil) {
		_outputs = [(outputs ?: @{}) copy];
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_outputs);
	SUPER_DEALLOC();
}

- (nullable id)outputForKey:(NSString *)key
{
	return key != nil ? _outputs[key] : nil;
}

- (nullable NSString *)text
{
	id value = _outputs[NFKOutputText];
	return [value isKindOfClass:NSString.class] ? value : nil;
}

- (nullable NSDictionary<NSString *, id> *)structured
{
	id value = _outputs[NFKOutputStructured];
	return [value isKindOfClass:NSDictionary.class] ? value : nil;
}

- (nullable NSArray<NSDictionary<NSString *, id> *> *)toolCalls
{
	id value = _outputs[NFKOutputToolCalls];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (nullable NSArray<NSNumber *> *)embedding
{
	id value = _outputs[NFKOutputEmbedding];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (nullable NSArray<NFKDetection *> *)detections
{
	id value = _outputs[NFKOutputDetections];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (nullable NSArray<NFKKeypoint *> *)pose
{
	id value = _outputs[NFKOutputPose];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (nullable NSArray<NFKClassification *> *)classifications
{
	id value = _outputs[NFKOutputClassifications];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (nullable NSArray<NFKAudioSegment *> *)segments
{
	id value = _outputs[NFKOutputSegments];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (id)copyWithZone:(NSZone *)zone
{
	// Immutable.
	return NARC_RETAIN(self);
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKInferenceResult.class]) {
		return NO;
	}
	NFKInferenceResult *other = object;
	return [other->_outputs isEqualToDictionary:_outputs];
}

- (NSUInteger)hash
{
	return _outputs.hash;
}

@end
