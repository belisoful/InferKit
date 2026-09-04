//
//  NFKRemoteModel.m
//  InferKit
//

#import <InferKit/NFKRemoteModel.h>

@interface NFKRemoteModel ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite) NSString *displayName;
@property (nonatomic, copy, readwrite, nullable) NSString *ownedBy;
@property (nonatomic, copy, readwrite, nullable) NSDate *createdAt;
@property (nonatomic, copy, readwrite, nullable) NSNumber *contextLength;
@property (nonatomic, copy, readwrite, nullable) NSNumber *sizeBytes;
@property (nonatomic, copy, readwrite, nullable) NSString *quantization;
@property (nonatomic, copy, readwrite, nullable) NSArray<NSString *> *capabilities;
@property (nonatomic, copy, readwrite) NSDictionary<NSString *, id> *raw;
@end

@implementation NFKRemoteModel

+ (nullable instancetype)modelWithEntry:(NSDictionary<NSString *, id> *)entry
{
	if (![entry isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *identifier = [self stringIn:entry forKey:@"id"]
		?: [self stringIn:entry forKey:@"model"]
		?: [self stringIn:entry forKey:@"name"];
	if (identifier == nil) {
		return nil;
	}
	NSDictionary *details = [entry[@"details"] isKindOfClass:NSDictionary.class] ? entry[@"details"] : @{};

	NFKRemoteModel *model = [[self alloc] initPrivate];
	model.identifier = identifier;
	model.displayName = [self stringIn:entry forKey:@"display_name"]
		?: [self stringIn:entry forKey:@"name"]
		?: identifier;
	model.ownedBy = [self stringIn:entry forKey:@"owned_by"];
	model.createdAt = [self creationDateIn:entry];
	model.contextLength = [self numberIn:entry forKey:@"context_length"]
		?: [self numberIn:entry forKey:@"max_context_length"]
		?: [self numberIn:details forKey:@"context_length"];
	model.sizeBytes = [self numberIn:entry forKey:@"size"];
	model.quantization = [self stringIn:entry forKey:@"quantization"]
		?: [self stringIn:details forKey:@"quantization_level"];
	model.capabilities = [self stringsIn:entry forKey:@"capabilities"];
	model.raw = entry;
	return model;
}

- (instancetype)initPrivate
{
	return [super init];
}

#pragma mark Fields

+ (nullable NSString *)stringIn:(NSDictionary *)entry forKey:(NSString *)key
{
	id value = entry[key];
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

+ (nullable NSNumber *)numberIn:(NSDictionary *)entry forKey:(NSString *)key
{
	id value = entry[key];
	return [value isKindOfClass:NSNumber.class] ? value : nil;
}

+ (nullable NSArray<NSString *> *)stringsIn:(NSDictionary *)entry forKey:(NSString *)key
{
	id value = entry[key];
	if (![value isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableArray<NSString *> *strings = [NSMutableArray array];
	for (id item in value) {
		if ([item isKindOfClass:NSString.class]) {
			[strings addObject:item];
		}
	}
	return strings;
}

// OpenAI writes a Unix time under `created`; Anthropic an ISO-8601 string under `created_at`.
+ (nullable NSDate *)creationDateIn:(NSDictionary *)entry
{
	NSNumber *seconds = [self numberIn:entry forKey:@"created"];
	if (seconds != nil) {
		return [NSDate dateWithTimeIntervalSince1970:seconds.doubleValue];
	}
	NSString *stamp = [self stringIn:entry forKey:@"created_at"];
	if (stamp == nil) {
		return nil;
	}
	return [[[NSISO8601DateFormatter alloc] init] dateFromString:stamp];
}

#pragma mark NSObject

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;			// immutable
}

- (BOOL)isEqual:(id)object
{
	if (![object isKindOfClass:NFKRemoteModel.class]) {
		return NO;
	}
	NFKRemoteModel *other = object;
	return [self.identifier isEqualToString:other.identifier] && [self.raw isEqualToDictionary:other.raw];
}

- (NSUInteger)hash
{
	return self.identifier.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@>", self.class, self.identifier];
}

@end
