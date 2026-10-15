//
//  NFKModelDescriptor.h
//  InferKit
//

#ifndef NFKModelDescriptor_h
#define NFKModelDescriptor_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKModelVocabulary.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelDescriptor
	@abstract   Immutable value types that describe a model: what it takes, what it returns, the options
				it reads, where its files come from, and what it needs to run.
	@discussion Each type reads from and writes to the JSON shape of a model-catalog manifest
				(Docs/model-catalog.md). A reader keeps every field it does not know in extensions, and
				dictionaryRepresentation writes those fields back, so a manifest a newer writer made
				passes through an older reader unchanged. A value that is absent is left out of the
				dictionary. A malformed dictionary fails with NFKModelCatalogErrorDomain and
				NFKModelCatalogErrorMalformedDescriptor, and the error's description names the field.
				Introduced in InferKit 0.4.0.
*/

/*!
	@class      NFKModelPort
	@abstract   One input a model reads or one output it writes, named by its request or result key.
	@discussion valueType is one of the NFKModelValueType* strings. A port built without one takes the
				type the core key carries, from NFKModelValueTypeForKey for an input and
				NFKModelValueTypeForOutputKey for an output. Introduced in InferKit 0.4.0.
*/
@interface NFKModelPort : NSObject <NSCopying>

/*! The NFKInput* or NFKOutput* key, or a key of the backend's own. */
@property (nonatomic, readonly, copy) NSString *key;
/*! The kind of value under the key, or nil when neither the descriptor nor the core table names one. */
@property (nonatomic, readonly, copy, nullable) NSString *valueType;
/*! On an input, YES when every request must carry it. On an output, YES when every result carries it. */
@property (nonatomic, readonly, getter=isRequired) BOOL required;
/*! A short name for the port, for a user interface. */
@property (nonatomic, readonly, copy, nullable) NSString *title;
/*! A sentence that explains the port, for a user interface. */
@property (nonatomic, readonly, copy, nullable) NSString *help;
/*! The fields of the port's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A port with every field. A nil valueType takes the type NFKModelValueTypeForKey gives the key. */
- (instancetype)initWithKey:(NSString *)key
				  valueType:(nullable NSString *)valueType
				   required:(BOOL)required
					  title:(nullable NSString *)title
					   help:(nullable NSString *)help
				 extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! An input port typed from NFKModelValueTypeForKey. */
+ (instancetype)portWithKey:(NSString *)key required:(BOOL)required;

/*! An output port typed from NFKModelValueTypeForOutputKey. */
+ (instancetype)outputPortWithKey:(NSString *)key required:(BOOL)required;

/*! Reads an input port from its manifest dictionary. Returns nil and sets error when key is missing or
	a field has the wrong type. */
+ (nullable instancetype)portWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The port as its manifest dictionary: key, valueType, required, title, and help, with the
	extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelOptionChoice
	@abstract   One value an option of kind NFKModelOptionKindChoice takes, with its name.
	@discussion The value is a JSON scalar: an NSString or an NSNumber. Introduced in InferKit 0.4.0.
*/
@interface NFKModelOptionChoice : NSObject <NSCopying>

/*! The value a request carries when the choice is selected: an NSString or an NSNumber. */
@property (nonatomic, readonly, copy) id value;
/*! The name a user interface shows for the choice. */
@property (nonatomic, readonly, copy) NSString *title;
/*! The fields of the choice's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A choice with every field. A nil title is the value's description. */
- (instancetype)initWithValue:(id)value
						title:(nullable NSString *)title
				   extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A choice with a value and a title. */
+ (instancetype)choiceWithValue:(id)value title:(nullable NSString *)title;

/*! Reads a choice from its manifest dictionary. Returns nil and sets error when value is missing or is
	neither a string nor a number. */
+ (nullable instancetype)choiceWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The choice as its manifest dictionary: value and title, with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelOption
	@abstract   One setting a model reads, described for a user interface that edits it.
	@discussion A request-stage option rides on each request as the parameter named by key. A
				load-stage option is fixed when the backend is built. kind is one of the
				NFKModelOptionKind* strings and decides the control a user interface shows: minimum,
				maximum, and step bound a number or an integer, and choices list the values of a
				choice. Introduced in InferKit 0.4.0.
*/
@interface NFKModelOption : NSObject <NSCopying>

/*! The NFKParameter* key, or a key of the backend's own. */
@property (nonatomic, readonly, copy) NSString *key;
/*! The kind of value the option takes, one of the NFKModelOptionKind* strings. */
@property (nonatomic, readonly, copy) NSString *kind;
/*! A short name for the option, for a user interface. */
@property (nonatomic, readonly, copy, nullable) NSString *title;
/*! A sentence that explains the option, for a user interface. */
@property (nonatomic, readonly, copy, nullable) NSString *help;
/*! The value the backend uses when a request leaves the option out: an NSString, an NSNumber, an
	NSArray, or an NSDictionary of JSON values. */
@property (nonatomic, readonly, copy, nullable) id defaultValue;
/*! The least value a number or an integer takes. */
@property (nonatomic, readonly, copy, nullable) NSNumber *minimum;
/*! The greatest value a number or an integer takes. */
@property (nonatomic, readonly, copy, nullable) NSNumber *maximum;
/*! The increment a user interface steps a number or an integer by. */
@property (nonatomic, readonly, copy, nullable) NSNumber *step;
/*! The values a choice takes, in the order a user interface lists them. Empty for any other kind. */
@property (nonatomic, readonly, copy) NSArray<NFKModelOptionChoice *> *choices;
/*! When the option applies, one of the NFKModelOptionStage* strings. NFKModelOptionStageRequest when
	not given. */
@property (nonatomic, readonly, copy) NSString *stage;
/*! The name of the group a user interface shows the option in. */
@property (nonatomic, readonly, copy, nullable) NSString *group;
/*! The fields of the option's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! An option with every field. A nil stage is NFKModelOptionStageRequest. */
- (instancetype)initWithKey:(NSString *)key
					   kind:(NSString *)kind
					  title:(nullable NSString *)title
					   help:(nullable NSString *)help
			   defaultValue:(nullable id)defaultValue
					minimum:(nullable NSNumber *)minimum
					maximum:(nullable NSNumber *)maximum
					   step:(nullable NSNumber *)step
					choices:(nullable NSArray<NFKModelOptionChoice *> *)choices
					  stage:(nullable NSString *)stage
					  group:(nullable NSString *)group
				 extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A request-stage option of the given kind. */
+ (instancetype)optionWithKey:(NSString *)key kind:(NSString *)kind;

/*! A request-stage option whose kind follows the value type NFKModelValueTypeForKey gives the key: a
	boolean, integer, number, text, or text-list type gives the kind of the same name, and any other
	type or an unknown key gives NFKModelOptionKindJSON. */
+ (instancetype)optionWithKey:(NSString *)key;

/*! Reads an option from its manifest dictionary. An absent kind follows the key as optionWithKey:
	does. Returns nil and sets error when key is missing or a field has the wrong type. */
+ (nullable instancetype)optionWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The option as its manifest dictionary: key, kind, stage, and each field that is present (the
	default under "default", choices only when there are some), with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelLicense
	@abstract   The license of a model's weights.
	@discussion The license covers the weights a variant downloads. InferKit's own code is under its
				own license, which this does not describe. Introduced in InferKit 0.4.0.
*/
@interface NFKModelLicense : NSObject <NSCopying>

/*! The SPDX identifier, such as "Apache-2.0", when the license has one. */
@property (nonatomic, readonly, copy, nullable) NSString *identifier;
/*! The license's name, for a user interface. */
@property (nonatomic, readonly, copy) NSString *name;
/*! Where the license text is published. */
@property (nonatomic, readonly, copy, nullable) NSURL *URL;
/*! YES when the publisher asks a user to accept terms before the weights download, so a download
	needs an access token whose account accepted them. */
@property (nonatomic, readonly, getter=isGated) BOOL gated;
/*! YES when the license forbids or limits commercial use. */
@property (nonatomic, readonly) BOOL restrictsCommercialUse;
/*! A sentence on a term a user should know, such as an attribution requirement. */
@property (nonatomic, readonly, copy, nullable) NSString *note;
/*! The fields of the license's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A license with every field. */
- (instancetype)initWithIdentifier:(nullable NSString *)identifier
							  name:(NSString *)name
							   URL:(nullable NSURL *)URL
							 gated:(BOOL)gated
			restrictsCommercialUse:(BOOL)restrictsCommercialUse
							  note:(nullable NSString *)note
						extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! An ungated license that allows commercial use. */
+ (instancetype)licenseWithIdentifier:(nullable NSString *)identifier name:(NSString *)name URL:(nullable NSURL *)URL;

/*! Reads a license from its manifest dictionary. An absent name is the identifier. Returns nil and
	sets error when both are missing or a field has the wrong type. */
+ (nullable instancetype)licenseWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The license as its manifest dictionary: id, name, url, gated, restrictsCommercialUse, and note,
	with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelFile
	@abstract   One file of a model's source.
	@discussion path is relative to the source: a path inside a repository or a directory, or the file
				name of a single-file source. Introduced in InferKit 0.4.0.
*/
@interface NFKModelFile : NSObject <NSCopying>

/*! The file's path relative to the source. */
@property (nonatomic, readonly, copy) NSString *path;
/*! The file's size in bytes, when known. */
@property (nonatomic, readonly, copy, nullable) NSNumber *byteCount;
/*! The SHA-256 of the file's contents as lowercase hexadecimal digits, when known. */
@property (nonatomic, readonly, copy, nullable) NSString *sha256;
/*! What the file holds, one of the NFKModelFileRole* strings. NFKModelFileRoleOther when not given. */
@property (nonatomic, readonly, copy) NSString *role;
/*! YES when the model loads without the file. */
@property (nonatomic, readonly, getter=isOptional) BOOL optional;
/*! The fields of the file's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A file with every field. A nil role is NFKModelFileRoleOther. */
- (instancetype)initWithPath:(NSString *)path
				   byteCount:(nullable NSNumber *)byteCount
					  sha256:(nullable NSString *)sha256
						role:(nullable NSString *)role
					optional:(BOOL)optional
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A required file with a role and no size or digest. */
+ (instancetype)fileWithPath:(NSString *)path role:(NSString *)role;

/*! Reads a file from its manifest dictionary. Returns nil and sets error when path is missing or a
	field has the wrong type. */
+ (nullable instancetype)fileWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The file as its manifest dictionary: path, bytes, sha256, role, and optional, with the extensions
	beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelSource
	@abstract   Where a variant's files come from and how they are laid out.
	@discussion kind is one of the NFKModelSourceKind* strings and layout one of the
				NFKModelSourceLayout* strings. A Hugging Face source names its repository and, for a
				reproducible download, the revision. A URL source names the URL. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelSource : NSObject <NSCopying>

/*! Where the files come from, one of the NFKModelSourceKind* strings. */
@property (nonatomic, readonly, copy) NSString *kind;
/*! The repository, such as "org/name", for a Hugging Face source. */
@property (nonatomic, readonly, copy, nullable) NSString *repository;
/*! The commit or tag the files are read at. */
@property (nonatomic, readonly, copy, nullable) NSString *revision;
/*! The download URL for a URL source, or the page that describes the release. */
@property (nonatomic, readonly, copy, nullable) NSURL *URL;
/*! How the model loads its files, one of the NFKModelSourceLayout* strings. NFKModelSourceLayoutNone
	when not given. */
@property (nonatomic, readonly, copy) NSString *layout;
/*! The files the source holds, in the order the manifest lists them. */
@property (nonatomic, readonly, copy) NSArray<NFKModelFile *> *files;
/*! YES when the original release needs a converter before InferKit loads it. */
@property (nonatomic, readonly) BOOL requiresConversion;
/*! The fields of the source's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A source with every field. A nil layout is NFKModelSourceLayoutNone. */
- (instancetype)initWithKind:(NSString *)kind
				  repository:(nullable NSString *)repository
					revision:(nullable NSString *)revision
						 URL:(nullable NSURL *)URL
					  layout:(nullable NSString *)layout
					   files:(nullable NSArray<NFKModelFile *> *)files
		  requiresConversion:(BOOL)requiresConversion
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A Hugging Face source with no listed files. */
+ (instancetype)huggingFaceSourceWithRepository:(NSString *)repository
									   revision:(nullable NSString *)revision
										 layout:(NSString *)layout;

/*! Reads a source from its manifest dictionary. Returns nil and sets error when kind is missing or a
	field has the wrong type. */
+ (nullable instancetype)sourceWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The source as its manifest dictionary: kind, repository, revision, url, layout,
	requiresConversion, and files, with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelVariant
	@abstract   One release of a model: a size, a precision, or a fine-tune, with its own files.
	@discussion registryName is the name the variant registers under in its engine's registry, which
				a caller passes to that registry to build the backend. The byte counts are the download
				size, the bytes the weights hold in memory, the most memory the variant holds at once,
				and the least memory the variant runs in, as NFKModelRequirements defines it. Introduced in InferKit 0.4.0.
*/
@interface NFKModelVariant : NSObject <NSCopying>

/*! The variant's identifier, unique within its descriptor. */
@property (nonatomic, readonly, copy) NSString *identifier;
/*! The variant's name, for a user interface. */
@property (nonatomic, readonly, copy) NSString *displayName;
/*! YES for the variant a caller gets without choosing one. */
@property (nonatomic, readonly) BOOL isDefault;
/*! The name the variant registers under in its engine's registry. */
@property (nonatomic, readonly, copy, nullable) NSString *registryName;
/*! Where the variant's files come from. */
@property (nonatomic, readonly, copy, nullable) NFKModelSource *source;
/*! The bytes a download of the variant transfers. */
@property (nonatomic, readonly, copy, nullable) NSNumber *downloadByteCount;
/*! The bytes the variant's weights hold in memory once loaded. */
@property (nonatomic, readonly, copy, nullable) NSNumber *residentByteCount;
/*! The most bytes the variant holds at once while it loads or runs at its default options, which can
	exceed residentByteCount. */
@property (nonatomic, readonly, copy, nullable) NSNumber *peakByteCount;
/*! The variant's parameter count. */
@property (nonatomic, readonly, copy, nullable) NSNumber *parameterCount;
/*! The type the weights are stored in, such as "float32", "float16", "bfloat16", or "int4". */
@property (nonatomic, readonly, copy, nullable) NSString *precision;
/*! The least memory, in bytes, the variant runs in, as NFKModelRequirements.minimumMemoryByteCount
	defines it. */
@property (nonatomic, readonly, copy, nullable) NSNumber *minimumMemoryByteCount;
/*! The fields of the variant's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A variant with every field. A nil displayName is the identifier. */
- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
						 isDefault:(BOOL)isDefault
					  registryName:(nullable NSString *)registryName
							source:(nullable NFKModelSource *)source
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
				 residentByteCount:(nullable NSNumber *)residentByteCount
					 peakByteCount:(nullable NSNumber *)peakByteCount
					parameterCount:(nullable NSNumber *)parameterCount
						 precision:(nullable NSString *)precision
			minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
						extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! A variant with an identifier and a name and nothing else. */
+ (instancetype)variantWithIdentifier:(NSString *)identifier displayName:(nullable NSString *)displayName;

/*! Reads a variant from its manifest dictionary. An absent displayName is the identifier. Returns nil
	and sets error when id is missing or a field has the wrong type. */
+ (nullable instancetype)variantWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The variant as its manifest dictionary: id, displayName, isDefault, and each field that is present
	(the byte counts as downloadBytes, residentBytes, peakBytes, and minimumMemoryBytes), with the
	extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKModelRequirements
	@abstract   What a machine needs to run a model.
	@discussion minimumOSVersions maps a platform name ("macOS", "iOS", "tvOS", "visionOS") to the
				least version that runs the model, such as "14.0". A platform the dictionary leaves out
				does not run the model. An empty dictionary runs everywhere InferKit runs. Introduced in
				InferKit 0.4.0.
*/
@interface NFKModelRequirements : NSObject <NSCopying>

/*! The least operating-system version per platform name. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *minimumOSVersions;
/*! YES when the model runs only on Apple silicon. */
@property (nonatomic, readonly) BOOL requiresAppleSilicon;
/*!
	@property   minimumMemoryByteCount
	@abstract   The least memory, in bytes, the model runs in: InferKit's residency budget for it.
	@discussion The figure is the smallest recommended Metal working set
				(NFKHardwareProfile.recommendedWorkingSetSize) whose residency budget holds the model's
				largest resident stage. The budget is the fraction of the working set the residency
				planner plans against, less the reserve the planner keeps for activations. So the
				figure is the bytes the largest resident stage holds plus the reserve, divided by that
				fraction. A model that stages, pages, or streams counts only what one stage holds at
				once. The MLX companion supplies the figures for its models.
*/
@property (nonatomic, readonly, copy, nullable) NSNumber *minimumMemoryByteCount;
/*! The fields of the requirements' dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! No requirements: every platform InferKit runs on, any processor, and no stated memory. */
- (instancetype)init;

/*! Requirements with every field. */
- (instancetype)initWithMinimumOSVersions:(nullable NSDictionary<NSString *, NSString *> *)minimumOSVersions
					 requiresAppleSilicon:(BOOL)requiresAppleSilicon
				   minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
							   extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! Reads requirements from their manifest dictionary. Returns nil and sets error when a field has the
	wrong type. */
+ (nullable instancetype)requirementsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The requirements as their manifest dictionary: minimumOS, requiresAppleSilicon, and
	minimumMemoryBytes when present, with the extensions beside them. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

@end

/*!
	@class      NFKModelDescriptor
	@abstract   Everything an app needs to list, explain, download, and configure one model.
	@discussion A descriptor names its engine and how that engine builds the model: registryName for an
				engine registry, providerClassName for an NFKDynamicBackendProvider. tasks lists what
				the model does, primary first. inputs, outputs, and parameters list the keys a request
				and a result use, typed for a user interface. variants lists the releases.

				Fields left nil take a value from the others:
				- inputModalities and outputModalities: the value types of inputs and outputs, in port
				  order without repeats
				- displayName: the identifier
				- requirements: no requirements

				standIn marks a reference or test stand-in that a user interface hides; the catalog's
				visibleDescriptors leaves it out. Introduced in InferKit 0.4.0.
*/
@interface NFKModelDescriptor : NSObject <NSCopying>

/*! The descriptor's identifier, unique within a catalog. */
@property (nonatomic, readonly, copy) NSString *identifier;
/*! The model's name, for a user interface. */
@property (nonatomic, readonly, copy) NSString *displayName;
/*! The model family, such as "qwen3" or "whisper", shared by the descriptors of one architecture. */
@property (nonatomic, readonly, copy) NSString *family;
/*! A sentence on what the model does, for a user interface. */
@property (nonatomic, readonly, copy, nullable) NSString *summary;
/*! The engine that runs the model, one of the NFKModelEngine* strings. */
@property (nonatomic, readonly, copy) NSString *engine;
/*! The name the model registers under in its engine's registry. */
@property (nonatomic, readonly, copy, nullable) NSString *registryName;
/*! The NFKDynamicBackendProvider class that builds the model's backend. */
@property (nonatomic, readonly, copy, nullable) NSString *providerClassName;
/*! What the model does, NFKModelTask* strings with the primary task first. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *tasks;
/*! The kinds of value the model reads, NFKModelValueType* strings. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *inputModalities;
/*! The kinds of value the model writes, NFKModelValueType* strings. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *outputModalities;
/*! The inputs a request carries. */
@property (nonatomic, readonly, copy) NSArray<NFKModelPort *> *inputs;
/*! The outputs a result carries. */
@property (nonatomic, readonly, copy) NSArray<NFKModelPort *> *outputs;
/*! The settings the model reads. */
@property (nonatomic, readonly, copy) NSArray<NFKModelOption *> *parameters;
/*! The model's releases. */
@property (nonatomic, readonly, copy) NSArray<NFKModelVariant *> *variants;
/*! The variant whose isDefault is YES, else the first variant, else nil. */
@property (nonatomic, readonly, nullable) NFKModelVariant *defaultVariant;
/*! The license of the model's weights. */
@property (nonatomic, readonly, copy, nullable) NFKModelLicense *license;
/*! What a machine needs to run the model. */
@property (nonatomic, readonly, copy) NFKModelRequirements *requirements;
/*! How the model can hold its weights, NFKModelResidency* strings. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *residencies;
/*! YES for a reference or test stand-in that a user interface hides. */
@property (nonatomic, readonly, getter=isStandIn) BOOL standIn;
/*! The InferKit version that introduced the model, such as "0.4.0". */
@property (nonatomic, readonly, copy, nullable) NSString *introducedVersion;
/*! The fields of the descriptor's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A descriptor with every field. A nil collection is empty, and the derived fields follow the class
	discussion. */
- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
							family:(NSString *)family
						   summary:(nullable NSString *)summary
							engine:(NSString *)engine
					  registryName:(nullable NSString *)registryName
				 providerClassName:(nullable NSString *)providerClassName
							 tasks:(nullable NSArray<NSString *> *)tasks
				   inputModalities:(nullable NSArray<NSString *> *)inputModalities
				  outputModalities:(nullable NSArray<NSString *> *)outputModalities
							inputs:(nullable NSArray<NFKModelPort *> *)inputs
						   outputs:(nullable NSArray<NFKModelPort *> *)outputs
						parameters:(nullable NSArray<NFKModelOption *> *)parameters
						  variants:(nullable NSArray<NFKModelVariant *> *)variants
						   license:(nullable NFKModelLicense *)license
					  requirements:(nullable NFKModelRequirements *)requirements
					   residencies:(nullable NSArray<NSString *> *)residencies
						   standIn:(BOOL)standIn
				 introducedVersion:(nullable NSString *)introducedVersion
						extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*!
	@method     descriptorWithIdentifier:derivedFromBackend:
	@abstract   A descriptor read from what a backend declares, for a backend an app or a plugin builds.
	@discussion The inputs come from supportedInputKeys, the parameters from supportedParameterKeys,
				and the outputs from supportedOutputKeys, each sorted by key and typed from the core
				table. A backend that does not declare a set contributes no entries for it. No port is
				required. The family is the NFKModelInfoArchitecture the backend's modelInfo reports,
				else its backendIdentifier. The engine is NFKModelEnginePlugin.
*/
+ (instancetype)descriptorWithIdentifier:(NSString *)identifier
					  derivedFromBackend:(id<NFKInferenceBackend>)backend
	NS_SWIFT_NAME(init(identifier:derivedFrom:));

/*! Reads a descriptor from its manifest dictionary. Returns nil and sets error when id, family, or
	engine is missing, or a field or a nested value is malformed. */
+ (nullable instancetype)descriptorWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;

/*! The descriptor as its manifest dictionary, every nested value included, with the extensions
	beside the fields. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

/*! The variant with the identifier, or nil. */
- (nullable NFKModelVariant *)variantWithIdentifier:(NSString *)identifier;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelDescriptor_h */
