//
//  NFKModelBuilding.h
//  InferKit
//

#ifndef NFKModelBuilding_h
#define NFKModelBuilding_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKModelCatalog.h>
#import <InferKit/NFKModelMemory.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelBuilding
	@abstract   Building a backend from a catalog entry and a set of options.
	@discussion -[NFKModelCatalog backendForModelIdentifier:options:error:] takes a model's identifier,
				or model@preset, and the option values an app collected, and returns a backend:
				1. The descriptor is found, and a preset's options are merged under the given ones.
				2. The options resolve against the descriptor (NFKModelCatalogErrorInvalidOptions).
				3. The model and the chosen choices are checked against this machine
				   (NFKModelCatalogErrorUnavailable).
				4. The memory estimate is checked against the catalog's memoryBudget, this machine's by
				   default (NFKModelCatalogErrorDoesNotFit), unless NFKModelLoadOptionIgnoreMemoryBudget
				   is YES.
				   The error carries the estimate under NFKModelMemoryEstimateKey. Where another
				   combination fits, the one nearest the requested options is under
				   NFKModelMemoryFittingOptionsKey with its estimate under NFKModelMemoryFittingEstimateKey,
				   and the description names it: "… Choosing residency paged fits: it holds 9.0 GiB."
				   Where nothing fits, the description adds the smallest combination's need.
				5. The variant's files are located, by the given URL or by the catalog's locator.
				6. A builder builds the backend: the descriptor's provider class when it implements
				   +backendForBuildRequest:error:, else the builder registered for the engine, else the
				   descriptor's NFKDynamicBackendProvider class with the property options set on what it
				   returns (NFKModelCatalogErrorNoBuilder when there is none).
				7. Request defaults wrap the backend in an NFKDefaultedBackend.
				Introduced in InferKit 0.4.0.
*/

@class NFKModelBuildRequest;

/*!
	@typedef    NFKModelLocator
	@abstract   Finds the local copy of a variant's files: a directory for a directory layout, the file
				for a file layout.
	@discussion Returns nil when the files are not present; sets error only when the lookup itself
				failed. Introduced in InferKit 0.4.0.
*/
typedef NSURL * _Nullable (^NFKModelLocator)(NFKModelDescriptor *descriptor,
											 NFKModelVariant * _Nullable variant,
											 NSError **error);

/*!
	@protocol   NFKModelBackendBuilding
	@abstract   An object that builds the backends of one engine's descriptors.
	@discussion The builder reads the build request's argumentOptions as construction arguments, and
				calls applyPropertyOptionsToBackend:error: on what it constructs, so a builder that builds
				the backend again sets the property options again. Introduced in InferKit 0.4.0.
*/
@protocol NFKModelBackendBuilding <NSObject>

/*! A backend for the request, or nil with an error. */
- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error;

@end

/*!
	@class      NFKModelBuildRequest
	@abstract   Everything a builder needs to build one backend: the descriptor, the variant, the
				resolved options, and where the files are.
	@discussion loadOptions holds every load option, defaults filled, the variant under
				NFKModelLoadOptionVariant. argumentOptions holds the load options without a
				propertyKeyPath, and propertyOptions those with one. requestDefaults holds the request
				parameters the given options and the preset set, and the load-stage context length and
				cache settings a request may override. Introduced in InferKit 0.4.0.
*/
@interface NFKModelBuildRequest : NSObject

@property (nonatomic, readonly) NFKModelDescriptor *descriptor;
@property (nonatomic, readonly, nullable) NFKModelVariant *variant;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *loadOptions;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *argumentOptions;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *propertyOptions;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *requestDefaults;
/*! The local copy of the variant's files, or nil when there is none, such as for a system model. */
@property (nonatomic, readonly, copy, nullable) NSURL *location;
/*! The estimate the build was checked against, or nil when the build was not checked. */
@property (nonatomic, readonly, nullable) NFKModelMemoryEstimate *memoryEstimate;
/*! The catalog the request was made from. */
@property (nonatomic, readonly) NFKModelCatalog *catalog;

/*! A request with every field. */
- (instancetype)initWithDescriptor:(NFKModelDescriptor *)descriptor
						   variant:(nullable NFKModelVariant *)variant
					   loadOptions:(nullable NSDictionary<NSString *, id> *)loadOptions
				   requestDefaults:(nullable NSDictionary<NSString *, id> *)requestDefaults
						  location:(nullable NSURL *)location
					memoryEstimate:(nullable NFKModelMemoryEstimate *)memoryEstimate
						   catalog:(NFKModelCatalog *)catalog NS_DESIGNATED_INITIALIZER;

/*! The local copy of another model's files through the catalog's locator, such as a draft model's. A
	nil variant is the model's default. Returns nil with UnknownModel when the catalog has no such
	model, and nil without an error when the files are not present. */
- (nullable NSURL *)locationForModelIdentifier:(NSString *)identifier
									   variant:(nullable NSString *)variant
										 error:(NSError **)error;

/*! Sets each property option on the backend by key-value coding. Returns NO with
	NFKModelCatalogErrorInvalidOptions, setting nothing, when the backend has no setter for a key
	path. */
- (BOOL)applyPropertyOptionsToBackend:(id)backend error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKDefaultedBackend
	@abstract   A backend that fills each request's missing parameters from a set of defaults and
				forwards everything else to the backend it wraps.
	@discussion A request's own parameters win. The optional members, such as modelInfo, prepare,
				unload, isLoaded, and the byte counts, answer as the wrapped backend answers them, and
				respondsToSelector: reports what the wrapped backend implements. loadOptionValues is the
				wrapped backend's when it reports them, else the values the wrapper was given.
				Introduced in InferKit 0.4.0.
*/
@interface NFKDefaultedBackend : NSObject <NFKInferenceBackend>

@property (nonatomic, readonly) id<NFKInferenceBackend> backend;
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *requestDefaults;

/*! A wrapper that fills the defaults into each request and reports loadOptionValues. */
- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
				requestDefaults:(NSDictionary<NSString *, id> *)requestDefaults
			   loadOptionValues:(nullable NSDictionary<NSString *, id> *)loadOptionValues NS_DESIGNATED_INITIALIZER;

/*! The request with the defaults filled where it leaves a parameter out. */
- (NFKInferenceRequest *)requestByFillingDefaults:(NFKInferenceRequest *)request;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@category   NFKModelCatalog (NFKModelBuilding)
	@abstract   Builds a backend from a catalog entry and a set of options.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKModelCatalog (NFKModelBuilding)

/*! Registers a builder for the descriptors of an engine, replacing any before it. A nil builder removes
	the engine's. The registry is shared by every catalog. */
+ (void)registerBackendBuilder:(nullable id<NFKModelBackendBuilding>)builder forEngine:(NSString *)engine;

/*! The builder registered for an engine, or nil. */
+ (nullable id<NFKModelBackendBuilding>)backendBuilderForEngine:(NSString *)engine;

/*! The locator that finds a variant's Hugging Face files in NFKHFHub's default cache: the snapshot
	directory of a directory layout, the first weights file of a file layout. */
@property (class, nonatomic, readonly) NFKModelLocator hubCacheLocator;

/*! The locator the build uses when no location is given. A download store sets its own. Setting nil
	restores hubCacheLocator. */
@property (nonatomic, copy, null_resettable) NFKModelLocator locator;

/*!
	@property   memoryBudget
	@abstract   The budget the build checks a model's memory estimate against (step 4).
	@discussion The default is +[NFKModelMemoryBudget currentBudget], read at each build. An app sets a
				budget to plan for a known machine, and a test sets one so its builds do not depend on the
				machine that runs it. Setting nil restores the default. The budget governs the fit check
				and the refusal's sentences; an engine places the backend it builds against its own
				reading of the machine. NFKInferenceServer's estimate route reports the budget of its
				modelCatalog. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, strong, null_resettable) NFKModelMemoryBudget *memoryBudget;

/*! The request a build makes, after steps 1 to 5. Returns nil with UnknownModel, InvalidOptions,
	Unavailable, or DoesNotFit. */
- (nullable NFKModelBuildRequest *)buildRequestForModelIdentifier:(NSString *)identifier
														   options:(nullable NSDictionary<NSString *, id> *)options
														  location:(nullable NSURL *)location
															 error:(NSError **)error;

/*! A backend for the model, its files found by the locator. */
- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)identifier
													  options:(nullable NSDictionary<NSString *, id> *)options
														error:(NSError **)error;

/*! A backend for the model, its files at location. A nil location asks the locator. */
- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)identifier
													  options:(nullable NSDictionary<NSString *, id> *)options
													 location:(nullable NSURL *)location
														error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelBuilding_h */
