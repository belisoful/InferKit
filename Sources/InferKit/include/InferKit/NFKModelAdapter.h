//
//  NFKModelAdapter.h
//  InferKit
//

#ifndef NFKModelAdapter_h
#define NFKModelAdapter_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelCatalog.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKHFHub;

/*!
	@header     NFKModelAdapter
	@abstract   Inspects a LoRA adapter file without loading a model: its tensor layout, the base model
				it was trained on, and the catalog models that take it.
	@discussion An inspection reads the safetensors header only, so a repository adapter costs two
				HTTP range requests. A model takes adapters through NFKModelLoadOptionAdapters, and its
				descriptor's adapterBases lists the bases it takes. Introduced in InferKit 0.4.0.
*/

/*!
	@enum       NFKModelAdapterFit
	@abstract   Which catalog models take an inspected adapter. Introduced in InferKit 0.4.0.
	@constant   NFKModelAdapterFitFits The model the inspection names takes the adapter.
	@constant   NFKModelAdapterFitFitsOtherModels The named model does not take the adapter, or the
				inspection names none, and the models in fittingModelIdentifiers do.
	@constant   NFKModelAdapterFitUnsupported No catalog model takes the adapter. The report's error
				states why.
*/
typedef NS_ENUM(NSInteger, NFKModelAdapterFit) {
	NFKModelAdapterFitFits = 0,
	NFKModelAdapterFitFitsOtherModels = 1,
	NFKModelAdapterFitUnsupported = 2,
};

/*! The error domain of an adapter inspection. Introduced in InferKit 0.4.0. */
extern NSErrorDomain const NFKModelAdapterErrorDomain;

/*!
	@enum       NFKModelAdapterError
	@abstract   Why an adapter is unreadable or unsupported, in NFKModelAdapterErrorDomain. Introduced in
				InferKit 0.4.0.
	@constant   NFKModelAdapterErrorUnreadable The file is missing or is not a safetensors file, or a
				repository holds no single safetensors file. NSUnderlyingErrorKey holds the reading error
				where there is one.
	@constant   NFKModelAdapterErrorPickle The file is a pickle file (.bin, .pt, .pth, .ckpt, or .pkl),
				refused without reading, because unpickling runs code the file carries.
	@constant   NFKModelAdapterErrorUnknownLayout The tensors are in no layout InferKit reads, such as LoHa
				(hada_), LoKr (lokr_), or DoRA (dora_scale, lora_magnitude_vector).
	@constant   NFKModelAdapterErrorUnknownBase The layout reads, and neither the tensors nor the metadata
				tell the base model.
	@constant   NFKModelAdapterErrorNoModel The base is known, and no catalog model takes adapters of that
				base.
*/
typedef NS_ERROR_ENUM(NFKModelAdapterErrorDomain, NFKModelAdapterError) {
	NFKModelAdapterErrorUnreadable		= 1,
	NFKModelAdapterErrorPickle			= 2,
	NFKModelAdapterErrorUnknownLayout	= 3,
	NFKModelAdapterErrorUnknownBase		= 4,
	NFKModelAdapterErrorNoModel			= 5,
};

/*!
	@class      NFKModelAdapterReport
	@abstract   What an inspection found in one adapter file.
	@discussion Fields the inspection could not read are nil, empty, or 0. A pickle file's report holds
				only its file name, its size where the file exists, its fit, and its error. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelAdapterReport : NSObject <NSCopying>

/*! The file's name. */
@property (nonatomic, readonly, copy) NSString *fileName;
/*! The tensor layout, an NFKModelAdapterLayout* string, or nil when no layout InferKit reads matches. */
@property (nonatomic, readonly, copy, nullable) NSString *layout;
/*! The base model, an NFKModelAdapterBase* string, or the base model's name an adapter_config.json or
	the metadata states for a base InferKit does not list; nil when unknown. */
@property (nonatomic, readonly, copy, nullable) NSString *base;
/*! The base_model_name_or_path of the adapter_config.json beside the file, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *baseModelName;
/*! The most common rank of the adapter's low-rank pairs; 0 when unknown. */
@property (nonatomic, readonly) NSInteger rank;
/*! The parts of the model the adapter changes, NFKModelAdapterTarget* strings in a fixed order. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *targets;
/*! The words that call up what the adapter learned: modelspec.trigger_phrase, else the five most
	frequent training tags of ss_tag_frequency. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *triggerWords;
/*! The bytes of the stored tensors. */
@property (nonatomic, readonly) NSInteger tensorByteCount;
/*! The bytes of the whole file; 0 when unknown. */
@property (nonatomic, readonly) long long fileByteCount;
/*! The SHA-256 of the whole file in lowercase hexadecimal: hashed from a local file, or the hub's LFS
	checksum for a repository file. nil for a pickle file and for a repository file the hub states none for. */
@property (nonatomic, readonly, copy, nullable) NSString *sha256;
/*! The safetensors __metadata__ strings. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *metadata;
/*! Which catalog models take the adapter. */
@property (nonatomic, readonly) NFKModelAdapterFit fit;
/*! The identifiers of the catalog descriptors whose adapterBases contain base, sorted. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *fittingModelIdentifiers;
/*! Why the adapter is unsupported, in NFKModelAdapterErrorDomain; nil otherwise. */
@property (nonatomic, readonly, copy, nullable) NSError *error;

/*! The report as JSON-safe values: fileName, layout, base, baseModelName, rank, targets, triggerWords,
	tensorByteCount, fileByteCount, sha256, metadata, fit ("fits", "fits-other-models", or "unsupported"),
	fittingModelIdentifiers, and error {domain, code, message}. An absent value is left out. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*! Adapter inspection against the receiver's descriptors. Introduced in InferKit 0.4.0. */
@interface NFKModelCatalog (NFKModelAdapter)

/*!
	@method     inspectAdapterAtURL:forModelIdentifier:error:
	@abstract   Inspects the adapter file at url against the catalog.
	@discussion Reads the safetensors header and an adapter_config.json in the same folder, and hashes the
				file in chunks for sha256. A pickle file is reported unsupported without being read.
				identifier names the model the adapter is meant for; an identifier the catalog does not hold
				fits nothing. Returns nil and sets error, NFKModelAdapterErrorUnreadable, only for a file
				that cannot be read; every other outcome is a report.
*/
- (nullable NFKModelAdapterReport *)inspectAdapterAtURL:(NSURL *)url
									 forModelIdentifier:(nullable NSString *)identifier
												  error:(NSError **)error;

/*!
	@method     inspectAdapterInRepository:revision:path:forModelIdentifier:completionHandler:
	@abstract   Inspects an adapter in a Hugging Face repository by reading its header with HTTP range
				requests.
	@discussion path names the file. A nil path lists the repository and takes its one safetensors file;
				a repository with none or several fails with NFKModelAdapterErrorUnreadable, and one whose
				only adapter files are pickle files is reported unsupported. An adapter_config.json in the
				file's folder is read too. The file's size and SHA-256 come from the repository listing,
				with nothing downloaded. A nil revision is "main". The hub is a default one, with
				NFKHFHub's access-token rules. A gated or private repository the token does not open fails
				with the hub's NFKDownloadErrorAuthenticationRequired error, unchanged. The handler runs
				on a background queue, with a report or an error as
				inspectAdapterAtURL:forModelIdentifier:error: gives them.
*/
- (void)inspectAdapterInRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							  path:(nullable NSString *)path
				forModelIdentifier:(nullable NSString *)identifier
				 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelAdapterReport * _Nullable report,
															   NSError * _Nullable error))completionHandler;

/*! inspectAdapterInRepository:revision:path:forModelIdentifier:completionHandler: through the given hub,
	which supplies the endpoint, the session, and the access token. */
- (void)inspectAdapterInRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							  path:(nullable NSString *)path
				forModelIdentifier:(nullable NSString *)identifier
							   hub:(NFKHFHub *)hub
				 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelAdapterReport * _Nullable report,
															   NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelAdapter_h */
