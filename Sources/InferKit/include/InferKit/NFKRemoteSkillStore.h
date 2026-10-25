//
//  NFKRemoteSkillStore.h
//  InferKit
//

#ifndef NFKRemoteSkillStore_h
#define NFKRemoteSkillStore_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteSkillStoreAPIStyle
	@abstract   The wire shape of a hosted skill store.
	@constant   NFKRemoteSkillStoreAPIStyleAnthropic Anthropic's Skills API (/v1/skills with
				x-api-key): a bundle goes as files[] parts named by their path, versions are
				skver_ identifiers listed under /versions, and each version can be deleted.
	@constant   NFKRemoteSkillStoreAPIStylePerplexity Perplexity's Skills API (/v1/skills with a
				Bearer token): a bundle goes as one zip under file, revisions are opaque tokens, an
				update or a delete names the revision it expects, and a revision downloads.
	@discussion Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteSkillStoreAPIStyle) {
	NFKRemoteSkillStoreAPIStyleAnthropic = 0,
	NFKRemoteSkillStoreAPIStylePerplexity = 1,
};

/*!
	@class      NFKRemoteSkill
	@abstract   A skill a service keeps: its identifier, its name, and the version this record describes.
	@discussion version is the version or revision the record describes, and latestVersion the newest
				one; Anthropic's records describe the newest, so both carry its latest_version_id.
				raw keeps the service's record. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteSkill : NSObject <NSCopying>

/*! The identifier a request and the store name the skill by: skill_… on both services. */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! The skill's name: Anthropic's display_name, Perplexity's SKILL.md name. */
@property (nonatomic, copy, readonly, nullable) NSString *name;

/*! The SKILL.md description, where the record carries one (Perplexity's). */
@property (nonatomic, copy, readonly, nullable) NSString *skillDescription;

/*! The version or revision this record describes. */
@property (nonatomic, copy, readonly, nullable) NSString *version;

/*! The newest version or revision, the one an update or a delete on Perplexity expects. */
@property (nonatomic, copy, readonly, nullable) NSString *latestVersion;

/*! Where the skill comes from: custom (the account's own), or on Anthropic anthropic,
	anthropic_example, or plugin. */
@property (nonatomic, copy, readonly) NSString *source;

@property (nonatomic, copy, readonly, nullable) NSDate *createdAt;
@property (nonatomic, copy, readonly, nullable) NSDate *updatedAt;

/*! The service's record as it came. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! The skill as an NFKParameterSkills entry: {id, source}, which loads its newest version. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, NSString *> *skillReference;

/*! A skill read from a service's record, or nil for one without an identifier. */
+ (nullable instancetype)skillWithRecord:(NSDictionary<NSString *, id> *)record apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteSkillVersion
	@abstract   One immutable version of a skill.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteSkillVersion : NSObject <NSCopying>

/*! The version's identifier: Anthropic's skver_… id, Perplexity's revision token. */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! The SKILL.md name at this version. */
@property (nonatomic, copy, readonly, nullable) NSString *name;

/*! The SKILL.md description at this version, where the record carries one (Anthropic's). */
@property (nonatomic, copy, readonly, nullable) NSString *skillDescription;

@property (nonatomic, copy, readonly, nullable) NSDate *createdAt;

/*! The service's record as it came. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! A version read from a service's record, or nil for one without an identifier. */
+ (nullable instancetype)versionWithRecord:(NSDictionary<NSString *, id> *)record apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteSkillStore
	@abstract   Uploads, versions, lists, and deletes the skills a hosted agent loads.
	@discussion A skill is an Agent Skills bundle: one top-level folder holding a SKILL.md, whose
				YAML frontmatter names the skill (name) and says when to use it (description), with
				the instructions after it, and any reference files and scripts beside it. The service
				reads the name and description from the file. The model sees each skill's name and
				description and loads the rest when it decides the skill applies.

				A bundle is a zip (createSkillWithBundle:name:error:) or a folder on disk
				(createSkillWithDirectoryAtURL:name:error:), which Perplexity takes zipped; the store
				zips it through NSFileCoordinator. A request reaches a skill through
				NFKParameterSkills on NFKAnthropicBackend and on NFKRemoteResponsesBackend in its
				Perplexity dialect.

				The services differ:
				- Anthropic: a new version needs no expected version; a single version deletes
				  (deleteVersion:ofSkillWithIdentifier:error:); no bundle downloads.
				- Perplexity: an update and a delete send the revision they expect and fail with
				  HTTP 409 (kNFKError_InferenceBackendFailure) when the skill moved past it; a
				  revision downloads through a short-lived URL; a single revision does not delete.
				A call the style does not serve fails with kNFKError_InferenceUnsupported. Every call
				blocks; run it off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteSkillStore : NSObject

/*! The skills collection, for example https://api.anthropic.com/v1/skills. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape. Defaults to NFKRemoteSkillStoreAPIStyleAnthropic. */
@property (nonatomic, assign) NFKRemoteSkillStoreAPIStyle apiStyle;

/*! The key: x-api-key on Anthropic's style, a Bearer token on Perplexity's. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The request timeout in seconds. Defaults to 120, since an upload can be tens of megabytes. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)skillStoreWithEndpointURL:(nullable NSURL *)endpointURL apiStyle:(NFKRemoteSkillStoreAPIStyle)apiStyle;

/*! A store pointed at the provider's skills: anthropic and perplexity serve one, and every other
	preset returns nil. */
+ (nullable instancetype)skillStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*!
	@method     createSkillWithBundle:name:error:
	@abstract   Creates a skill from a zip of its folder, or nil with an error.
	@discussion name is Anthropic's display_name, a label that need not be unique; Perplexity takes
				the name from SKILL.md alone and ignores it.
*/
- (nullable NFKRemoteSkill *)createSkillWithBundle:(NSData *)zip
											  name:(nullable NSString *)name
											 error:(NSError * _Nullable *)outError;

/*! Creates a skill from its folder on disk, or nil with an error. Anthropic receives each file
	under its path inside the folder, and Perplexity the folder zipped. */
- (nullable NFKRemoteSkill *)createSkillWithDirectoryAtURL:(NSURL *)directoryURL
													  name:(nullable NSString *)name
													 error:(NSError * _Nullable *)outError;

/*! Every skill the account keeps, across the service's pages, newest first, or nil with an error. */
- (nullable NSArray<NFKRemoteSkill *> *)skillsWithError:(NSError * _Nullable *)outError;

/*!
	@method     skillWithIdentifier:version:error:
	@abstract   One skill, or nil with an error.
	@discussion version nil describes the active version. Perplexity reads a named revision through
				?revision=; Anthropic's record always describes the newest, so a named version is
				read through versionsOfSkillWithIdentifier:error: there.
*/
- (nullable NFKRemoteSkill *)skillWithIdentifier:(NSString *)identifier
										 version:(nullable NSString *)version
										   error:(NSError * _Nullable *)outError;

/*! The skill's versions, newest first, across the service's pages, or nil with an error. */
- (nullable NSArray<NFKRemoteSkillVersion *> *)versionsOfSkillWithIdentifier:(NSString *)identifier
																	   error:(NSError * _Nullable *)outError;

/*!
	@method     addVersionToSkillWithIdentifier:bundle:expectedVersion:error:
	@abstract   Uploads a new version of the skill from a zip, or nil with an error.
	@discussion A version is a whole snapshot, so a file the bundle leaves out is gone from it, and
				its SKILL.md name must match the skill's. On Perplexity expectedVersion is the revision
				the skill must still be at (its latestVersion); nil reads that first, which gives up
				the guard against a concurrent upload. Anthropic answers with the version, and the
				store reads the skill back.
*/
- (nullable NFKRemoteSkill *)addVersionToSkillWithIdentifier:(NSString *)identifier
													  bundle:(NSData *)zip
											 expectedVersion:(nullable NSString *)expectedVersion
													   error:(NSError * _Nullable *)outError;

/*! The same with the skill's folder on disk. */
- (nullable NFKRemoteSkill *)addVersionToSkillWithIdentifier:(NSString *)identifier
											directoryAtURL:(NSURL *)directoryURL
										   expectedVersion:(nullable NSString *)expectedVersion
													 error:(NSError * _Nullable *)outError;

/*! Deletes the skill and every version of it. On Perplexity expectedVersion is the revision the
	skill must still be at, and nil reads it first. */
- (BOOL)deleteSkillWithIdentifier:(NSString *)identifier
				  expectedVersion:(nullable NSString *)expectedVersion
							error:(NSError * _Nullable *)outError;

/*! Deletes one version of the skill. Anthropic serves it; Perplexity does not. */
- (BOOL)deleteVersion:(NSString *)version
ofSkillWithIdentifier:(NSString *)identifier
				error:(NSError * _Nullable *)outError;

/*! A short-lived URL of the zip of a revision, nil meaning the active one. Perplexity serves it;
	Anthropic does not. */
- (nullable NSURL *)downloadURLForSkillWithIdentifier:(NSString *)identifier
											  version:(nullable NSString *)version
												error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteSkillStore_h */
