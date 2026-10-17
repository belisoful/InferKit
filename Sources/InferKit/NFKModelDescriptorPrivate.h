//
//  NFKModelDescriptorPrivate.h
//  InferKit
//
//  The manifest reading the descriptor, memory, preset, and catalog files share. Not a public header.
//

#ifndef NFKModelDescriptorPrivate_h
#define NFKModelDescriptorPrivate_h

#import "NFKModelDescriptor.h"
#import "NFKModelCatalog.h"
#import "NFKModelMemory.h"
#import "NFKModelOptions.h"
#import "NFKModelPreset.h"

NS_ASSUME_NONNULL_BEGIN

/*! Sets error to an NFKModelCatalogErrorDomain error with the code and description; returns NO. */
BOOL NFKModelCatalogFail(NFKModelCatalogError code, NSString *reason, NSError **error);

/*! The dictionary without the known fields: what an object keeps as its extensions. */
NSDictionary<NSString *, id> *NFKModelExtensionsWithout(NSDictionary<NSString *, id> * _Nullable extensions,
														NSArray<NSString *> *knownFields);

/*! Each value's dictionaryRepresentation, in order. */
NSArray *NFKModelDictionariesOf(NSArray *values);

/*! YES when other is an expectedClass whose dictionaryRepresentation equals object's. */
BOOL NFKModelIsEqualDescription(id object, id other, Class expectedClass);

/*! YES for an NSNumber that holds a JSON boolean. */
BOOL NFKModelIsBoolean(id _Nullable value);

typedef id _Nullable (^NFKModelElementReader)(id element, NSString *path, NSError **error);

/*! Reads typed fields from one manifest dictionary and names a failing field by its path. */
@interface NFKModelFieldReader : NSObject

@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *dictionary;
@property (nonatomic, readonly, copy) NSString *path;

+ (nullable instancetype)readerWithObject:(id)object path:(NSString *)path error:(NSError **)error;
- (NSString *)pathForField:(NSString *)field;
- (BOOL)failField:(NSString *)field reason:(NSString *)reason error:(NSError **)error;
- (NSDictionary<NSString *, id> *)extensionsWithoutFields:(NSArray<NSString *> *)knownFields;
- (BOOL)readString:(NSString *)field required:(BOOL)required into:(NSString * _Nullable * _Nonnull)value error:(NSError **)error;
- (BOOL)readNumber:(NSString *)field into:(NSNumber * _Nullable * _Nonnull)value error:(NSError **)error;
- (BOOL)readBoolean:(NSString *)field into:(BOOL *)value error:(NSError **)error;
- (BOOL)readURL:(NSString *)field into:(NSURL * _Nullable * _Nonnull)value error:(NSError **)error;
- (BOOL)readStringArray:(NSString *)field into:(NSArray<NSString *> * _Nullable * _Nonnull)value error:(NSError **)error;
- (BOOL)readStringDictionary:(NSString *)field
						into:(NSDictionary<NSString *, NSString *> * _Nullable * _Nonnull)value
					   error:(NSError **)error;
- (BOOL)readNumberDictionary:(NSString *)field
						into:(NSDictionary<NSString *, NSNumber *> * _Nullable * _Nonnull)value
					   error:(NSError **)error;
- (BOOL)readJSONDictionary:(NSString *)field
					  into:(NSDictionary<NSString *, id> * _Nullable * _Nonnull)value
					 error:(NSError **)error;
- (BOOL)readObject:(NSString *)field
		withReader:(NFKModelElementReader)reader
			  into:(id _Nullable * _Nonnull)value
			 error:(NSError **)error;
- (BOOL)readObjects:(NSString *)field
		 withReader:(NFKModelElementReader)reader
			   into:(NSArray * _Nullable * _Nonnull)value
			  error:(NSError **)error;
- (BOOL)readObjectDictionary:(NSString *)field
				  withReader:(NFKModelElementReader)reader
						into:(NSDictionary<NSString *, id> * _Nullable * _Nonnull)value
					   error:(NSError **)error;
/*! Fails with NFKModelCatalogErrorMalformedDescriptor when two of the elements read from field share the
	value at keyPath, naming both by their JSON member name. */
- (BOOL)requireUniqueElements:(nullable NSArray *)elements
					  ofField:(NSString *)field
					  keyPath:(NSString *)keyPath
						 name:(NSString *)name
						error:(NSError **)error;

@end

@interface NFKModelDescriptor (NFKModelCatalogReading)

/*! Reads a descriptor, naming fields in an error from path, such as "models[2]". */
+ (nullable instancetype)descriptorFromObject:(id)object path:(NSString *)path error:(NSError **)error;

@end

@interface NFKModelRequirements (NFKModelCatalogReading)
+ (nullable instancetype)requirementsFromObject:(id)object path:(NSString *)path error:(NSError **)error;
@end

@interface NFKModelMemoryFigure (NFKModelCatalogReading)
+ (nullable instancetype)figureFromObject:(id)object path:(NSString *)path error:(NSError **)error;
@end

@interface NFKModelMemoryFacts (NFKModelCatalogReading)
+ (nullable instancetype)factsFromObject:(id)object path:(NSString *)path error:(NSError **)error;
@end

@interface NFKModelMemoryModel (NFKModelCatalogReading)
+ (nullable instancetype)memoryModelFromObject:(id)object path:(NSString *)path error:(NSError **)error;
@end

@interface NFKModelPreset (NFKModelCatalogReading)
+ (nullable instancetype)presetFromObject:(id)object path:(NSString *)path error:(NSError **)error;
@end

/*! The descriptor's option and memory work against a given catalog and machine. The public methods pass
	NFKModelCatalog.sharedCatalog and NFKHardwareProfile.currentProfile. */
@interface NFKModelDescriptor (NFKModelCatalogInternal)

/*! The keys a request may set although their options are load-stage: the memory model's context length,
	cache bits, and cache group size. */
@property (nonatomic, readonly) NSSet<NSString *> *requestOverridableOptionKeys;

- (NSArray<NFKModelOptionProblem *> *)problemsWithOptions:(nullable NSDictionary<NSString *, id> *)options
													stage:(nullable NSString *)stage
												  catalog:(NFKModelCatalog *)catalog
												  profile:(nullable NFKHardwareProfile *)profile;

- (nullable NSDictionary<NSString *, id> *)resolvedOptions:(nullable NSDictionary<NSString *, id> *)options
													 stage:(nullable NSString *)stage
												   catalog:(NFKModelCatalog *)catalog
												   profile:(nullable NFKHardwareProfile *)profile
													 error:(NSError **)error;

@end

@interface NFKModelDescriptor (NFKModelMemoryInternal)

/*! The estimate for load options already resolved, defaults filled. */
- (NFKModelMemoryEstimate *)estimateForResolvedOptions:(NSDictionary<NSString *, id> *)resolved
										  memoryBudget:(nullable NFKModelMemoryBudget *)budget
											   catalog:(NFKModelCatalog *)catalog;

/*! The value of the option the memory model names under key, or nil. */
- (nullable id)memoryOptionValue:(nullable NSString *)key in:(NSDictionary<NSString *, id> *)resolved;

@end

@interface NFKModelDescriptor (NFKModelMemoryConstraintsInternal)

/*! The constraints with options fixed, against a catalog and a machine. fittingOptions is the fitting
	combination nearest preferred (over the defaults) rather than nearest the defaults. */
- (NFKModelMemoryConstraints *)constraintsForOptions:(nullable NSDictionary<NSString *, id> *)options
										memoryBudget:(NFKModelMemoryBudget *)budget
											 profile:(nullable NFKHardwareProfile *)profile
											 catalog:(NFKModelCatalog *)catalog
										   preferred:(nullable NSDictionary<NSString *, id> *)preferred;

@end

/*! The option value as a message writes it: a string in quotes, a number or a boolean as text. */
NSString *NFKModelOptionValueText(id _Nullable value);

@interface NFKModelCatalog ()

/*! The provider class that supplied the descriptor with the identifier, or nil for a descriptor added
	any other way. */
- (nullable NSString *)providerClassNameForDescriptorIdentifier:(NSString *)identifier;

/*! The memory budget an app set, or nil for this machine's. */
@property (atomic, strong, nullable) NFKModelMemoryBudget *storedMemoryBudget;

/*! The locator an app set, or nil for the default. */
@property (atomic, copy, nullable) NSURL * _Nullable (^storedLocator)(NFKModelDescriptor *descriptor,
																	   NFKModelVariant * _Nullable variant,
																	   NSError **error);

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelDescriptorPrivate_h */
