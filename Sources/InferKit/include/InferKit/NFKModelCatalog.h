//
//  NFKModelCatalog.h
//  InferKit
//

#ifndef NFKModelCatalog_h
#define NFKModelCatalog_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKModelBuildRequest;
@class NFKModelRepository;

/*! The error domain for reading model descriptors and catalog manifests. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelCatalogErrorDomain;

/*!
	@enum       NFKModelCatalogError
	@abstract   Error codes in NFKModelCatalogErrorDomain.
	@constant   NFKModelCatalogErrorUnsupportedSchemaVersion The manifest's schemaVersion is newer than
				NFKModelCatalog.schemaVersion.
	@constant   NFKModelCatalogErrorMalformedManifest The data is not valid JSON or not a JSON object,
				its schemaVersion, kind, or models field is missing or of the wrong type, kind is not
				"inferkit.model-catalog", or two models share an id. For data that does not parse,
				NSUnderlyingErrorKey holds the parser's error.
	@constant   NFKModelCatalogErrorMalformedDescriptor A descriptor or a value inside one is missing a
				field it needs, has a field of the wrong type, or holds a value JSON cannot carry. The
				error's description names the field.
	@constant   NFKModelCatalogErrorInvalidOptions An option value does not resolve against the
				descriptor. NFKModelOptionProblemsKey holds the problems, and the description is the
				first problem's message.
	@constant   NFKModelCatalogErrorUnknownModel The catalog has no descriptor with the identifier.
	@constant   NFKModelCatalogErrorNoBuilder No builder builds the descriptor's engine, and the
				descriptor names no provider class.
	@constant   NFKModelCatalogErrorUnavailable This machine does not meet the requirements of the model
				or of a chosen choice.
	@constant   NFKModelCatalogErrorDoesNotFit The memory estimate does not fit the budget, or the
				residency planner refuses the residency. NFKModelMemoryEstimateKey holds the estimate
				where there is one.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKModelCatalogErrorDomain, NFKModelCatalogError) {
	NFKModelCatalogErrorUnsupportedSchemaVersion = 1,
	NFKModelCatalogErrorMalformedManifest = 2,
	NFKModelCatalogErrorMalformedDescriptor = 3,
	NFKModelCatalogErrorInvalidOptions = 4,
	NFKModelCatalogErrorUnknownModel = 5,
	NFKModelCatalogErrorNoBuilder = 6,
	NFKModelCatalogErrorUnavailable = 7,
	NFKModelCatalogErrorDoesNotFit = 8,
};

/*! The kind a catalog manifest names: "inferkit.model-catalog". Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelCatalogManifestKind;

/*!
	@protocol   NFKModelDescriptorProvider
	@abstract   A class that supplies model descriptors, discovered at runtime by name.
	@discussion A package or a plugin adopts this protocol on a class and lists the class name with
				+[NFKModelCatalog registerProviderClassName:]. The shared catalog looks the class up with
				NSClassFromString, so the core never references the package's symbols, and a package
				that is not linked contributes nothing. Introduced in InferKit 0.4.0.
*/
@protocol NFKModelDescriptorProvider <NSObject>

/*! The descriptors of the models the package ships. The catalog calls this on the class. */
+ (NSArray<NFKModelDescriptor *> *)modelDescriptors;

@optional

/*! A backend for one of the class's descriptors, or nil with an error. The catalog's build entry
	(NFKModelBuilding.h) asks the class that supplied a descriptor before an engine's builder. Introduced
	in InferKit 0.4.0. */
+ (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error;

/*!
	@method     descriptorForRepository:matchingDescriptor:error:
	@abstract   A descriptor for a repository whose configuration names a model type one of the class's
				descriptors lists, or nil with the reason its loader does not read the repository.
	@discussion A resolver (NFKModelResolver.h) asks the class that supplied descriptor, one whose
				modelTypes holds repository.modelType. The returned descriptor builds the repository's model
				through descriptor's loader, with one variant whose source is the repository at
				repository.revision and whose memory facts come from the repository's own files. The
				error is in NFKModelResolutionErrorDomain. Introduced in InferKit 0.4.0.
*/
+ (nullable NFKModelDescriptor *)descriptorForRepository:(NFKModelRepository *)repository
									  matchingDescriptor:(NFKModelDescriptor *)descriptor
												   error:(NSError **)error;

@end

/*!
	@class      NFKModelCatalog
	@abstract   The models InferKit and its linked packages can run, as descriptors an app lists and
				configures.
	@discussion NFKRemoteModelCatalog lists what a remote provider serves at the moment; this catalog
				describes the models an engine in the process runs.

				The shared catalog loads its descriptors once, on first use, from each provider class
				name that is linked into the process: the four the core names (NFKAppleModelDescriptors,
				NFKMLXModelDescriptors, NFKFoundationModelsDescriptors, NFKAppleSwiftModelDescriptors),
				then each name registerProviderClassName: adds, in order. A later provider's descriptor
				replaces an earlier one with the same identifier.

				A manifest is the JSON form of a catalog: {"schemaVersion", "kind", "models"}. A plugin
				ships its descriptors as one, and an app adds it with
				addDescriptorsFromManifestData:error:. Every method is safe to call from any thread.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelCatalog : NSObject

/*! The catalog of every linked provider's descriptors. */
@property (class, nonatomic, readonly) NFKModelCatalog *sharedCatalog;

/*! The manifest schema version this release writes: 2. It reads 1 and 2; version 2 adds a descriptor's modelTypes. */
@property (class, nonatomic, readonly) NSInteger schemaVersion;

/*! The provider class names the shared catalog loads from, in load order. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *providerClassNames;

/*! Adds a provider class name after the known ones. When the shared catalog has loaded, it adds the
	class's descriptors at once, so registering a name again reloads that provider. */
+ (void)registerProviderClassName:(NSString *)className;

/*! An empty catalog, for an app that keeps its own list and for tests. */
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/*! Every descriptor, sorted by identifier. */
@property (nonatomic, readonly, copy) NSArray<NFKModelDescriptor *> *descriptors;

/*! The descriptors that are not stand-ins, sorted by identifier. */
@property (nonatomic, readonly, copy) NSArray<NFKModelDescriptor *> *visibleDescriptors;

/*! The descriptor with the identifier, or nil. */
- (nullable NFKModelDescriptor *)descriptorWithIdentifier:(NSString *)identifier;

/*! The descriptors that list the task among their tasks, stand-ins included, sorted by identifier. */
- (NSArray<NFKModelDescriptor *> *)descriptorsForTask:(NSString *)task;

/*! The descriptors of the family, stand-ins included, sorted by identifier. */
- (NSArray<NFKModelDescriptor *> *)descriptorsInFamily:(NSString *)family;

/*! Adds a descriptor, replacing one with the same identifier. */
- (void)addDescriptor:(NFKModelDescriptor *)descriptor;

/*! Adds descriptors in order, each replacing one with the same identifier. */
- (void)addDescriptors:(NSArray<NFKModelDescriptor *> *)descriptors;

/*! Removes the descriptor with the identifier, if there is one. */
- (void)removeDescriptorWithIdentifier:(NSString *)identifier;

/*! Adds the descriptors of a provider class. Returns NO, and adds nothing, when no class of that name
	is linked or the class does not adopt NFKModelDescriptorProvider. */
- (BOOL)addDescriptorsFromProviderClassName:(NSString *)className;

/*!
	@method     addDescriptorsFromManifestData:error:
	@abstract   Adds every descriptor in a JSON manifest, or none.
	@discussion A manifest whose schemaVersion is newer than schemaVersion fails with
				NFKModelCatalogErrorUnsupportedSchemaVersion. A manifest without kind
				"inferkit.model-catalog", data that is not valid JSON, and a manifest in which two models
				share an id fail with NFKModelCatalogErrorMalformedManifest. A malformed descriptor fails
				with NFKModelCatalogErrorMalformedDescriptor. On any failure the catalog is left as it
				was.
*/
- (BOOL)addDescriptorsFromManifestData:(NSData *)data error:(NSError **)error;

/*! The catalog as a JSON manifest: pretty-printed with sorted keys and the descriptors sorted by
	identifier, so the same catalog always writes the same bytes. Returns nil and sets error when a
	descriptor's extensions hold a value JSON cannot carry. */
- (nullable NSData *)manifestDataWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelCatalog_h */
