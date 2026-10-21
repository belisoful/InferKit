//
//  NFKModelResolver.h
//  InferKit
//

#ifndef NFKModelResolver_h
#define NFKModelResolver_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelCatalog.h>
#import <InferKit/NFKModelMemory.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKHFHub;
@class NFKDownloadPlanFile;

/*!
	@header     NFKModelResolver
	@abstract   Finding how InferKit runs a Hugging Face repository, before any weights download.
	@discussion A resolver reads a repository's metadata: the commit its revision names, its model card's
				license, its file listing, its config.json, and the headers of its safetensors files. It
				then tries each route in order and records why each one it passes over does not apply:
				1. NFKModelRouteCatalog: a shipped variant reads the repository. Its figures are measured,
				   and a repository named at the variant's revision resolves with no network request.
				2. NFKModelRouteFamily: a shipped model's loader reads the repository's configuration. The
				   catalog's descriptors list the config.json model_type values their loaders read
				   (modelTypes), and the class that supplied a matching descriptor builds a descriptor for
				   the repository (+[NFKModelDescriptorProvider descriptorForRepository:matchingDescriptor:error:]).
				   Its memory figures come from the repository's own headers and configuration, and no
				   reference measured its output. This route resolves only when allowsUnverifiedRoutes
				   is YES.
				The resolution carries the descriptor, which an app adds to a catalog and builds as any
				other, and its memory estimate against a budget, this machine's by default. A
				descriptor's dictionaryRepresentation saves it, and +[NFKModelDescriptor
				descriptorWithDictionary:error:] restores it with no network request. A resolved
				descriptor's identifier is the repository name, so a catalog holds one revision of a
				repository at a time.
				Introduced in InferKit 0.4.0.
*/

/*! The error domain of a route the resolver passes over. Introduced in InferKit 0.4.0. */
extern NSErrorDomain const NFKModelResolutionErrorDomain;

/*!
	@enum       NFKModelResolutionError
	@abstract   Why a route does not apply, in NFKModelResolutionErrorDomain. Introduced in InferKit 0.4.0.
	@constant   NFKModelResolutionErrorUnverifiedRouteNotAllowed A route applies whose output no reference
				measured, and the resolver's allowsUnverifiedRoutes is NO.
	@constant   NFKModelResolutionErrorCustomCode The configuration names code the repository ships
				(auto_map), so its model type does not describe its computation.
	@constant   NFKModelResolutionErrorUnsupportedWeightFormat The repository holds no safetensors weights:
				pickled weights only, a GGUF file, or none.
	@constant   NFKModelResolutionErrorGated The hub withholds the repository without an access token
				whose account may read it: a gated repository's files, or a private repository's metadata.
				Without such a token the Hugging Face hub answers a private repository and a missing one
				alike, so this refusal covers a misspelled name too. NSUnderlyingErrorKey holds the hub's
				error.
	@constant   NFKModelResolutionErrorNoConfiguration The repository has no readable config.json.
	@constant   NFKModelResolutionErrorUnsupportedArchitecture No shipped model reads the configuration's
				model type.
	@constant   NFKModelResolutionErrorUnsupportedConfiguration A shipped loader reads the model type and
				refuses this configuration: a layer kind, an expert routing, or a size it does not build.
	@constant   NFKModelResolutionErrorUnsupportedTokenizer The loader does not read the repository's
				tokenizer as the release defines it.
	@constant   NFKModelResolutionErrorUnsupportedQuantization The loader does not read the repository's
				quantized weights.
*/
typedef NS_ERROR_ENUM(NFKModelResolutionErrorDomain, NFKModelResolutionError) {
	NFKModelResolutionErrorUnverifiedRouteNotAllowed	= 1,
	NFKModelResolutionErrorCustomCode					= 2,
	NFKModelResolutionErrorUnsupportedWeightFormat		= 3,
	NFKModelResolutionErrorGated						= 4,
	NFKModelResolutionErrorNoConfiguration				= 5,
	NFKModelResolutionErrorUnsupportedArchitecture		= 6,
	NFKModelResolutionErrorUnsupportedConfiguration		= 7,
	NFKModelResolutionErrorUnsupportedTokenizer			= 8,
	NFKModelResolutionErrorUnsupportedQuantization		= 9,
};

/*!
	@enum       NFKModelRoute
	@abstract   How InferKit runs a repository. Introduced in InferKit 0.4.0.
	@constant   NFKModelRouteNone No route applies; the resolution's reasons say why.
	@constant   NFKModelRouteCatalog A shipped variant reads the repository.
	@constant   NFKModelRouteFamily A shipped model's loader reads the repository's configuration.
*/
typedef NS_ENUM(NSInteger, NFKModelRoute) {
	NFKModelRouteNone		= 0,
	NFKModelRouteCatalog	= 1,
	NFKModelRouteFamily		= 2,
} NS_SWIFT_NAME(NFKModelResolution.Route);

/*! How far a route's output is verified, one of the NFKModelVerification* strings. Introduced in
	InferKit 0.4.0. */
typedef NSString *NFKModelVerification NS_TYPED_EXTENSIBLE_ENUM;
/*! The output was measured against the model's reference implementation. */
extern NFKModelVerification const NFKModelVerificationMeasured;
/*! The loader reads the configuration, and no reference measured this repository's output. */
extern NFKModelVerification const NFKModelVerificationConfigCompatible;

/*! The userInfo key of a refusal: the NFKModelRoute (an NSNumber) the refusal concerns. Introduced in
	InferKit 0.4.0. */
extern NSErrorUserInfoKey const NFKModelResolutionRouteKey;
/*! The userInfo key of a refusal: the identifier of the shipped descriptor whose loader refused, where
	one did. Introduced in InferKit 0.4.0. */
extern NSErrorUserInfoKey const NFKModelResolutionDescriptorKey;

/*! The extension key a resolved descriptor carries: a dictionary of "verification", "repository",
	"revision", and "basedOn", the shipped descriptor whose loader reads it. It survives the descriptor's
	dictionary round trip. Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelResolutionExtensionKey;

/*! The route's name in a resolution's dictionary: "none", "catalog", or "family". Introduced in InferKit
	0.4.0. */
FOUNDATION_EXPORT NSString *NFKModelRouteName(NFKModelRoute route) NS_SWIFT_NAME(getter:NFKModelRoute.name(self:));

/*!
	@class      NFKModelRepository
	@abstract   What a resolver read of a repository, handed to a loader that judges it.
	@discussion files lists every file at the commit with its size. The file reads go through the hub: a
				whole file lands in its cache, where a download of the variant finds it again, and a
				safetensors header is read by range requests without the tensors. Each read blocks on the
				network. Introduced in InferKit 0.4.0.
*/
@interface NFKModelRepository : NSObject

/*! The repository, such as "org/name". */
@property (nonatomic, readonly, copy) NSString *repository;
/*! The commit the files are read at. */
@property (nonatomic, readonly, copy) NSString *revision;
/*! The revision the caller named, or nil for the default branch. */
@property (nonatomic, readonly, copy, nullable) NSString *requestedRevision;
/*! The parsed config.json. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *configuration;
/*! The configuration's model_type, or nil when it states none. */
@property (nonatomic, readonly, copy, nullable) NSString *modelType;
/*! Every file at the commit, with its size. */
@property (nonatomic, readonly, copy) NSArray<NFKDownloadPlanFile *> *files;
/*! The license the model card states, or nil when it states none. gated is YES when the repository
	asks its users to accept terms. */
@property (nonatomic, readonly, copy, nullable) NFKModelLicense *license;
/*! The hub the files are read through. */
@property (nonatomic, readonly) NFKHFHub *hub;

/*! A repository read at revision. */
- (instancetype)initWithRepository:(NSString *)repository
						  revision:(NSString *)revision
				 requestedRevision:(nullable NSString *)requestedRevision
					 configuration:(NSDictionary<NSString *, id> *)configuration
							 files:(NSArray<NFKDownloadPlanFile *> *)files
						   license:(nullable NFKModelLicense *)license
							   hub:(NFKHFHub *)hub NS_DESIGNATED_INITIALIZER;

/*! The listed file at path, or nil. */
- (nullable NFKDownloadPlanFile *)fileAtPath:(NSString *)path;

/*! The paths of the listed files whose last path component has the suffix, such as ".safetensors", in
	listing order. */
- (NSArray<NSString *> *)pathsWithSuffix:(NSString *)suffix;

/*! The parsed JSON of the file at path, fetched into the hub's cache. Returns nil and sets error when
	the file is not listed, the fetch fails, or the contents are not JSON. */
- (nullable id)JSONObjectOfFileAtPath:(NSString *)path error:(NSError **)error;

/*! The header of the safetensors file at path, as -[NFKHFHub safetensorsHeaderOfRepo:revision:path:error:]
	reads it. */
- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfFileAtPath:(NSString *)path error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelResolution
	@abstract   How InferKit runs a repository, or why it does not.
	@discussion route is NFKModelRouteNone when no route resolves, and reasons then holds one error per
				route the resolver passed over, in NFKModelResolutionErrorDomain, in the order tried. A
				resolved route keeps the reasons of the routes before it. descriptor is the shipped
				descriptor for NFKModelRouteCatalog and a descriptor built for the repository for
				NFKModelRouteFamily, and variant is the one that reads the repository. memoryEstimate is
				the estimate at the descriptor's default options, judged against the resolver's budget.
				Introduced in InferKit 0.4.0.
*/
@interface NFKModelResolution : NSObject <NSCopying>

/*! The repository resolved. */
@property (nonatomic, readonly, copy) NSString *repository;
/*! The commit the resolution read, or the revision the caller named when the catalog answered without a
	request; nil for the default branch read with no request. */
@property (nonatomic, readonly, copy, nullable) NSString *revision;
/*! The route that runs the repository. */
@property (nonatomic, readonly) NFKModelRoute route;
/*! How far the route's output is verified, or nil for NFKModelRouteNone. */
@property (nonatomic, readonly, copy, nullable) NFKModelVerification verification;
/*! The descriptor that builds the repository's model, or nil. */
@property (nonatomic, readonly, copy, nullable) NFKModelDescriptor *descriptor;
/*! The descriptor's variant that reads the repository, or nil. */
@property (nonatomic, readonly, nullable) NFKModelVariant *variant;
/*! The memory estimate at the descriptor's default options against the budget, or nil. */
@property (nonatomic, readonly, copy, nullable) NFKModelMemoryEstimate *memoryEstimate;
/*! The bytes the variant's download transfers, or nil when unknown. */
@property (nonatomic, readonly, copy, nullable) NSNumber *downloadByteCount;
/*! Why each route passed over does not apply, in the order tried. */
@property (nonatomic, readonly, copy) NSArray<NSError *> *reasons;
/*! YES when a route applies. */
@property (nonatomic, readonly, getter=isResolved) BOOL resolved;

/*! A resolution with every field. A nil reasons is empty. */
- (instancetype)initWithRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							 route:(NFKModelRoute)route
					  verification:(nullable NFKModelVerification)verification
						descriptor:(nullable NFKModelDescriptor *)descriptor
				 variantIdentifier:(nullable NSString *)variantIdentifier
					memoryEstimate:(nullable NFKModelMemoryEstimate *)memoryEstimate
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
						   reasons:(nullable NSArray<NSError *> *)reasons NS_DESIGNATED_INITIALIZER;

/*! The resolution as JSON values: repository, revision, route (its name), verification, descriptor (its
	dictionary), variant (its identifier), memoryEstimate, downloadBytes, and reasons, each reason as
	{"domain", "code", "description", "route", "descriptor"}. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

/*! Reads a resolution from its dictionary. Returns nil and sets error when repository is missing or a
	field is malformed. */
+ (nullable instancetype)resolutionWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelResolver
	@abstract   Resolves Hugging Face repositories against a catalog.
	@discussion The resolver reads metadata only: no weights download. Every method is safe to call from
				any thread. The synchronous method blocks on the network, and the asynchronous one runs it
				on a background queue at user-initiated quality of service. A failure to read the
				repository fails the call with the hub's or the URL system's error: a transport failure,
				a revision the hub does not have, or a file a judgment reads that the hub does not serve.
				A refused read is the gated refusal. Every other outcome is a resolution. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelResolver : NSObject

/*! A resolver over the shared catalog and a hub on the default cache. */
- (instancetype)init;

/*! A resolver over catalog, reading through hub. A nil catalog is the shared catalog, and a nil hub a hub
	on +[NFKHFHub defaultCacheDirectoryURL]. */
- (instancetype)initWithCatalog:(nullable NFKModelCatalog *)catalog hub:(nullable NFKHFHub *)hub NS_DESIGNATED_INITIALIZER;

/*! The catalog whose descriptors the resolver matches. */
@property (nonatomic, readonly) NFKModelCatalog *catalog;
/*! The hub the resolver reads through. */
@property (nonatomic, readonly) NFKHFHub *hub;
/*! YES lets a route whose output no reference measured resolve. NO by default, which leaves such a route
	as a refusal with NFKModelResolutionErrorUnverifiedRouteNotAllowed. */
@property (atomic) BOOL allowsUnverifiedRoutes;
/*! The budget the memory estimate is judged against; nil, the default, is the catalog's memoryBudget,
	which is this machine's unless an app set another. */
@property (atomic, copy, nullable) NFKModelMemoryBudget *memoryBudget;

/*! The resolution of a repository at a revision; nil for the default branch. Blocks on the network.
	Returns nil and sets error when the metadata cannot be read. */
- (nullable NFKModelResolution *)resolutionForRepository:(NSString *)repository
												revision:(nullable NSString *)revision
												   error:(NSError **)error;

/*! The asynchronous form of resolutionForRepository:revision:error:. The handler runs on a background
	queue. */
- (void)resolveRepository:(NSString *)repository
				 revision:(nullable NSString *)revision
		completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelResolution * _Nullable resolution,
													  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelResolver_h */
