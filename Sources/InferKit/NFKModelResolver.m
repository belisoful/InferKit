//
//  NFKModelResolver.m
//  InferKit
//

#import "NFKModelResolver.h"
#import "NFKModelBuilding.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelVocabulary.h"
#import "NFKHFHub.h"
#import "NFKHFHubTransfer.h"
#import "NFKDownload.h"
#import "NFKDownloadPlan.h"
#import "NFKLocalModelRunner.h"
#import "NFKRemoteModel.h"
#import "NFKRemoteModelCatalog.h"
#import "NFKRemoteProvider.h"
#import "NFKInferenceKeys.h"

NSErrorDomain const NFKModelResolutionErrorDomain = @"NFKModelResolutionErrorDomain";
NFKModelVerification const NFKModelVerificationMeasured = @"measured";
NFKModelVerification const NFKModelVerificationConfigCompatible = @"configCompatible";
NFKModelVerification const NFKModelVerificationExternal = @"external";
NSErrorUserInfoKey const NFKModelResolutionRouteKey = @"NFKModelResolutionRoute";
NSErrorUserInfoKey const NFKModelResolutionDescriptorKey = @"NFKModelResolutionDescriptor";
NSString * const NFKModelResolutionExtensionKey = @"resolution";

static NSString * const NFKModelResolutionConfigPath = @"config.json";
static NSString * const NFKModelResolutionManifestPath = @"manifest.json";
/*! How long the local-server route waits for a server to answer. */
static const NSTimeInterval NFKModelResolutionServerTimeout = 2.0;

NSString *NFKModelRouteName(NFKModelRoute route)
{
	switch (route) {
		case NFKModelRouteCatalog:
			return @"catalog";
		case NFKModelRouteFamily:
			return @"family";
		case NFKModelRouteCoreMLPackage:
			return @"coreml-package";
		case NFKModelRouteLocalServer:
			return @"local-server";
		case NFKModelRouteNone:
			break;
	}
	return @"none";
}

static NFKModelRoute NFKModelRouteNamed(id name)
{
	if ([name isEqual:@"catalog"]) {
		return NFKModelRouteCatalog;
	}
	if ([name isEqual:@"family"]) {
		return NFKModelRouteFamily;
	}
	if ([name isEqual:@"coreml-package"]) {
		return NFKModelRouteCoreMLPackage;
	}
	if ([name isEqual:@"local-server"]) {
		return NFKModelRouteLocalServer;
	}
	return NFKModelRouteNone;
}

/*! A refusal of a route, with the shipped descriptor it concerns when there is one. */
static NSError *NFKModelResolutionRefusal(NFKModelResolutionError code, NFKModelRoute route,
										  NSString * _Nullable descriptorIdentifier, NSString *reason,
										  NSError * _Nullable underlying)
{
	NSMutableDictionary<NSErrorUserInfoKey, id> *userInfo = [NSMutableDictionary dictionary];
	userInfo[NSLocalizedDescriptionKey] = reason;
	userInfo[NFKModelResolutionRouteKey] = @(route);
	userInfo[NFKModelResolutionDescriptorKey] = descriptorIdentifier;
	userInfo[NSUnderlyingErrorKey] = underlying;
	return [NSError errorWithDomain:NFKModelResolutionErrorDomain code:code userInfo:userInfo];
}

/*! Sets the out-parameter when the caller passed one. */
static void NFKModelResolutionSet(NSError * _Nullable * _Nullable outError, NSError * _Nullable error)
{
	if (outError != NULL) {
		*outError = error;
	}
}

static BOOL NFKModelResolutionIsAuthenticationError(NSError * _Nullable error)
{
	return [error.domain isEqualToString:NFKDownloadErrorDomain] && error.code == NFKDownloadErrorAuthenticationRequired;
}

#pragma mark - NFKModelRepository

@interface NFKModelRepository ()
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NFKDownloadPlanFile *> *filesByPath;
@end

@implementation NFKModelRepository

- (instancetype)initWithRepository:(NSString *)repository
						  revision:(NSString *)revision
				 requestedRevision:(nullable NSString *)requestedRevision
					 configuration:(NSDictionary<NSString *, id> *)configuration
							 files:(NSArray<NFKDownloadPlanFile *> *)files
						   license:(nullable NFKModelLicense *)license
							   hub:(NFKHFHub *)hub
{
	self = [super init];
	if (self != nil) {
		_repository = [repository copy];
		_revision = [revision copy];
		_requestedRevision = [requestedRevision copy];
		_configuration = [configuration copy];
		_files = [files copy];
		_license = [license copy];
		_hub = hub;
		NSMutableDictionary<NSString *, NFKDownloadPlanFile *> *byPath = [NSMutableDictionary dictionary];
		for (NFKDownloadPlanFile *file in files) {
			byPath[file.path] = file;
		}
		_filesByPath = [byPath copy];
	}
	return self;
}

- (nullable NSString *)modelType
{
	id type = self.configuration[@"model_type"];
	return [type isKindOfClass:NSString.class] ? type : nil;
}

- (nullable NFKDownloadPlanFile *)fileAtPath:(NSString *)path
{
	return self.filesByPath[path];
}

- (NSArray<NSString *> *)pathsWithSuffix:(NSString *)suffix
{
	NSMutableArray<NSString *> *paths = [NSMutableArray array];
	for (NFKDownloadPlanFile *file in self.files) {
		if ([file.path.lastPathComponent hasSuffix:suffix]) {
			[paths addObject:file.path];
		}
	}
	return paths;
}

- (nullable id)JSONObjectOfFileAtPath:(NSString *)path error:(NSError **)error
{
	NFKDownloadPlanFile *file = [self fileAtPath:path];
	if (file == nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ does not list %@ at %@", self.repository, path, self.revision];
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKDownloadErrorDomain code:NFKDownloadErrorNotFound
									 userInfo:@{ NSLocalizedDescriptionKey: reason }];
		}
		return nil;
	}
	NSURL *localURL = [self.hub downloadRepo:self.repository revision:self.revision path:path sha256:file.sha256
									   error:error];
	if (localURL == nil) {
		return nil;
	}
	NSData *data = [NSData dataWithContentsOfURL:localURL options:0 error:error];
	if (data == nil) {
		return nil;
	}
	return [NSJSONSerialization JSONObjectWithData:data options:0 error:error];
}

- (nullable NSDictionary<NSString *, id> *)safetensorsHeaderOfFileAtPath:(NSString *)path error:(NSError **)error
{
	return [self.hub safetensorsHeaderOfRepo:self.repository revision:self.revision path:path error:error];
}

@end

#pragma mark - NFKModelResolutionAction

@implementation NFKModelResolutionAction

- (instancetype)initWithKind:(NFKModelResolutionActionKind)kind
		  providerIdentifier:(NSString *)providerIdentifier
					   model:(NSString *)model
						path:(nullable NSString *)path
				   byteCount:(nullable NSNumber *)byteCount
{
	self = [super init];
	if (self != nil) {
		_kind = kind;
		_providerIdentifier = [providerIdentifier copy];
		_model = [model copy];
		_path = [path copy];
		_byteCount = [byteCount copy];
	}
	return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (![object isKindOfClass:NFKModelResolutionAction.class]) {
		return NO;
	}
	return [self.dictionaryRepresentation isEqualToDictionary:[object dictionaryRepresentation]];
}

- (NSUInteger)hash
{
	return self.model.hash ^ (NSUInteger)self.kind;
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"kind"] = self.kind == NFKModelResolutionActionKindPull ? @"pull" : @"use";
	dictionary[@"provider"] = self.providerIdentifier;
	dictionary[@"model"] = self.model;
	dictionary[@"path"] = self.path;
	dictionary[@"bytes"] = self.byteCount;
	return [dictionary copy];
}

+ (nullable instancetype)actionWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	id kind = dictionary[@"kind"];
	id provider = dictionary[@"provider"];
	id model = dictionary[@"model"];
	id path = dictionary[@"path"];
	id bytes = dictionary[@"bytes"];
	BOOL known = [kind isEqual:@"use"] || [kind isEqual:@"pull"];
	if (!known || ![provider isKindOfClass:NSString.class] || ![model isKindOfClass:NSString.class]
		|| (path != nil && ![path isKindOfClass:NSString.class]) || (bytes != nil && ![bytes isKindOfClass:NSNumber.class])) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorMalformedDescriptor
									 userInfo:@{ NSLocalizedDescriptionKey: @"The resolution's action lacks its kind, "
												 "provider, or model, or a field has the wrong type" }];
		}
		return nil;
	}
	NFKModelResolutionActionKind actionKind = [kind isEqual:@"pull"] ? NFKModelResolutionActionKindPull
																	 : NFKModelResolutionActionKindUse;
	return [[self alloc] initWithKind:actionKind providerIdentifier:provider model:model path:path byteCount:bytes];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ on %@>", NSStringFromClass(self.class),
			self.kind == NFKModelResolutionActionKindPull ? @"pull" : @"use", self.model, self.providerIdentifier];
}

@end

#pragma mark - NFKModelResolution

@implementation NFKModelResolution

- (instancetype)initWithRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							 route:(NFKModelRoute)route
					  verification:(nullable NFKModelVerification)verification
						descriptor:(nullable NFKModelDescriptor *)descriptor
				 variantIdentifier:(nullable NSString *)variantIdentifier
					memoryEstimate:(nullable NFKModelMemoryEstimate *)memoryEstimate
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
						   reasons:(nullable NSArray<NSError *> *)reasons
{
	return [self initWithRepository:repository revision:revision route:route verification:verification
						 descriptor:descriptor variantIdentifier:variantIdentifier memoryEstimate:memoryEstimate
				  downloadByteCount:downloadByteCount reasons:reasons action:nil];
}

- (instancetype)initWithRepository:(NSString *)repository
						  revision:(nullable NSString *)revision
							 route:(NFKModelRoute)route
					  verification:(nullable NFKModelVerification)verification
						descriptor:(nullable NFKModelDescriptor *)descriptor
				 variantIdentifier:(nullable NSString *)variantIdentifier
					memoryEstimate:(nullable NFKModelMemoryEstimate *)memoryEstimate
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
						   reasons:(nullable NSArray<NSError *> *)reasons
							action:(nullable NFKModelResolutionAction *)action
{
	self = [super init];
	if (self != nil) {
		_repository = [repository copy];
		_revision = [revision copy];
		_route = route;
		_verification = [verification copy];
		_descriptor = [descriptor copy];
		_variant = variantIdentifier != nil ? [descriptor variantWithIdentifier:variantIdentifier] : descriptor.defaultVariant;
		_memoryEstimate = [memoryEstimate copy];
		_downloadByteCount = [downloadByteCount copy];
		_reasons = [reasons copy] ?: @[];
		_action = [action copy];
	}
	return self;
}

- (BOOL)isResolved
{
	return self.route != NFKModelRouteNone;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"repository"] = self.repository;
	dictionary[@"revision"] = self.revision;
	dictionary[@"route"] = NFKModelRouteName(self.route);
	dictionary[@"verification"] = self.verification;
	dictionary[@"descriptor"] = self.descriptor.dictionaryRepresentation;
	dictionary[@"variant"] = self.variant.identifier;
	dictionary[@"memoryEstimate"] = self.memoryEstimate.dictionaryRepresentation;
	dictionary[@"downloadBytes"] = self.downloadByteCount;
	dictionary[@"action"] = self.action.dictionaryRepresentation;
	NSMutableArray<NSDictionary *> *reasons = [NSMutableArray arrayWithCapacity:self.reasons.count];
	for (NSError *reason in self.reasons) {
		[reasons addObject:[NFKModelResolution dictionaryOfReason:reason]];
	}
	dictionary[@"reasons"] = reasons;
	return [dictionary copy];
}

+ (NSDictionary<NSString *, id> *)dictionaryOfReason:(NSError *)reason
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"domain"] = reason.domain;
	dictionary[@"code"] = @(reason.code);
	dictionary[@"description"] = reason.localizedDescription;
	NSNumber *route = reason.userInfo[NFKModelResolutionRouteKey];
	if ([route isKindOfClass:NSNumber.class]) {
		dictionary[@"route"] = NFKModelRouteName(route.integerValue);
	}
	dictionary[@"descriptor"] = reason.userInfo[NFKModelResolutionDescriptorKey];
	return [dictionary copy];
}

+ (nullable NSError *)reasonFromObject:(id)object
{
	if (![object isKindOfClass:NSDictionary.class] || ![object[@"domain"] isKindOfClass:NSString.class]
		|| ![object[@"code"] isKindOfClass:NSNumber.class]) {
		return nil;
	}
	NSMutableDictionary<NSErrorUserInfoKey, id> *userInfo = [NSMutableDictionary dictionary];
	if ([object[@"description"] isKindOfClass:NSString.class]) {
		userInfo[NSLocalizedDescriptionKey] = object[@"description"];
	}
	if ([object[@"route"] isKindOfClass:NSString.class]) {
		userInfo[NFKModelResolutionRouteKey] = @(NFKModelRouteNamed(object[@"route"]));
	}
	if ([object[@"descriptor"] isKindOfClass:NSString.class]) {
		userInfo[NFKModelResolutionDescriptorKey] = object[@"descriptor"];
	}
	return [NSError errorWithDomain:object[@"domain"] code:[object[@"code"] integerValue] userInfo:userInfo];
}

+ (nullable instancetype)resolutionWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	id repository = dictionary[@"repository"];
	if (![repository isKindOfClass:NSString.class]) {
		[self fail:@"repository is missing or not a string" error:error];
		return nil;
	}
	id revision = dictionary[@"revision"];
	id verification = dictionary[@"verification"];
	id variant = dictionary[@"variant"];
	id downloadBytes = dictionary[@"downloadBytes"];
	if ((revision != nil && ![revision isKindOfClass:NSString.class])
		|| (verification != nil && ![verification isKindOfClass:NSString.class])
		|| (variant != nil && ![variant isKindOfClass:NSString.class])
		|| (downloadBytes != nil && ![downloadBytes isKindOfClass:NSNumber.class])) {
		[self fail:@"revision, verification, variant, or downloadBytes has the wrong type" error:error];
		return nil;
	}
	NFKModelDescriptor *descriptor = nil;
	if (dictionary[@"descriptor"] != nil) {
		descriptor = [self readObject:dictionary[@"descriptor"] field:@"descriptor" error:error
							   reader:^id (NSDictionary *object, NSError **readError) {
			return [NFKModelDescriptor descriptorWithDictionary:object error:readError];
		}];
		if (descriptor == nil) {
			return nil;
		}
	}
	NFKModelMemoryEstimate *estimate = nil;
	if (dictionary[@"memoryEstimate"] != nil) {
		estimate = [self readObject:dictionary[@"memoryEstimate"] field:@"memoryEstimate" error:error
							 reader:^id (NSDictionary *object, NSError **readError) {
			return [NFKModelMemoryEstimate estimateWithDictionary:object error:readError];
		}];
		if (estimate == nil) {
			return nil;
		}
	}
	NFKModelResolutionAction *action = nil;
	if (dictionary[@"action"] != nil) {
		action = [self readObject:dictionary[@"action"] field:@"action" error:error
						   reader:^id (NSDictionary *object, NSError **readError) {
			return [NFKModelResolutionAction actionWithDictionary:object error:readError];
		}];
		if (action == nil) {
			return nil;
		}
	}
	NSMutableArray<NSError *> *reasons = [NSMutableArray array];
	id reasonObjects = dictionary[@"reasons"] ?: @[];
	if (![reasonObjects isKindOfClass:NSArray.class]) {
		[self fail:@"reasons is not an array" error:error];
		return nil;
	}
	for (id object in reasonObjects) {
		NSError *reason = [self reasonFromObject:object];
		if (reason == nil) {
			[self fail:@"a reason lacks its domain or code" error:error];
			return nil;
		}
		[reasons addObject:reason];
	}
	return [[self alloc] initWithRepository:repository revision:revision route:NFKModelRouteNamed(dictionary[@"route"])
							   verification:verification descriptor:descriptor variantIdentifier:variant
							 memoryEstimate:estimate downloadByteCount:downloadBytes reasons:reasons
									 action:action];
}

+ (nullable id)readObject:(id)object
					field:(NSString *)field
					error:(NSError **)error
				   reader:(id _Nullable (^)(NSDictionary *object, NSError **readError))reader
{
	if (![object isKindOfClass:NSDictionary.class]) {
		[self fail:[field stringByAppendingString:@" is not an object"] error:error];
		return nil;
	}
	return reader(object, error);
}

+ (BOOL)fail:(NSString *)reason error:(NSError **)error
{
	if (error != NULL) {
		NSString *description = [@"The resolution's " stringByAppendingString:reason];
		*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorMalformedDescriptor
								 userInfo:@{ NSLocalizedDescriptionKey: description }];
	}
	return NO;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ route=%@ verification=%@ descriptor=%@ reasons=%lu>",
			NSStringFromClass(self.class), self.repository, NFKModelRouteName(self.route), self.verification,
			self.descriptor.identifier, (unsigned long)self.reasons.count];
}

@end

#pragma mark - NFKModelResolver

@implementation NFKModelResolver

- (instancetype)init
{
	return [self initWithCatalog:nil hub:nil];
}

- (instancetype)initWithCatalog:(nullable NFKModelCatalog *)catalog hub:(nullable NFKHFHub *)hub
{
	self = [super init];
	if (self != nil) {
		_catalog = catalog ?: NFKModelCatalog.sharedCatalog;
		_hub = hub ?: [NFKHFHub hubWithCacheDirectoryURL:NFKHFHub.defaultCacheDirectoryURL];
		_localServers = NFKRemoteProvider.localProviders;
	}
	return self;
}

- (void)resolveRepository:(NSString *)repository
				 revision:(nullable NSString *)revision
		completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKModelResolution * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKModelResolution *resolution = [self resolutionForRepository:repository revision:revision error:&error];
		completionHandler(resolution, resolution != nil ? nil : error);
	});
}

- (nullable NFKModelResolution *)resolutionForRepository:(NSString *)repository
												revision:(nullable NSString *)revision
												   error:(NSError **)error
{
	NFKModelResolution *shipped = [self catalogResolutionForRepository:repository revision:revision commit:nil];
	if (shipped != nil) {
		return shipped;
	}
	NSError *informationError = nil;
	NSDictionary<NSString *, id> *information = [self informationOfRepository:repository revision:revision
																		error:&informationError];
	if (information == nil) {
		if (NFKModelResolutionIsAuthenticationError(informationError)) {
			return [self refusedResolutionOfRepository:repository revision:revision
											   reasons:@[ [self unreadableRefusalOfRepository:repository
																						error:informationError] ]];
		}
		NFKModelResolutionSet(error, informationError);
		return nil;
	}
	NSString *commit = [information[@"sha"] isKindOfClass:NSString.class] ? information[@"sha"] : (revision ?: @"main");
	shipped = [self catalogResolutionForRepository:repository revision:revision commit:commit];
	if (shipped != nil) {
		return shipped;
	}
	NSError *planError = nil;
	NFKDownloadPlan *plan = [self.hub planDownloadOfRepo:repository revision:commit paths:nil error:&planError];
	if (plan == nil) {
		return [self gatedResolutionOfRepository:repository revision:commit error:planError failure:error];
	}
	NFKModelLicense *license = [self licenseOfRepository:repository information:information];
	NFKModelRepository *listed = [[NFKModelRepository alloc] initWithRepository:repository revision:commit
															  requestedRevision:revision configuration:@{}
																		  files:plan.files license:license hub:self.hub];
	NSMutableArray<NSError *> *reasons = [NSMutableArray array];
	NSError *refusal = nil;
	NSError *failure = nil;
	NFKModelRepository *read = [self configuredRepository:listed refusal:&refusal error:&failure];
	if (failure != nil) {
		NFKModelResolutionSet(error, failure);
		return nil;
	}
	if (read == nil) {
		if (refusal.code == NFKModelResolutionErrorGated) {
			return [self refusedResolutionOfRepository:repository revision:commit reasons:@[ refusal ]];
		}
		[reasons addObject:refusal];
	} else {
		NFKModelResolution *family = [self familyResolutionOfRepository:read error:error];
		if (family == nil || family.isResolved || [self matchedWithoutConsent:family]) {
			return family;
		}
		[reasons addObjectsFromArray:family.reasons];
	}
	NFKModelResolution *package = [self packageResolutionOfRepository:listed reasons:reasons failure:&failure];
	if (package != nil || failure != nil) {
		NFKModelResolutionSet(error, failure);
		return package;
	}
	NFKModelResolution *served = [self serverResolutionOfRepository:listed gated:[self isGated:information] reasons:reasons];
	if (served != nil) {
		return served;
	}
	return [self refusedResolutionOfRepository:repository revision:commit reasons:reasons];
}

/*! YES for a refused resolution whose last reason is a route that applies without the caller's consent to
	unverified routes, which ends the resolution as the caller set it. */
- (BOOL)matchedWithoutConsent:(NFKModelResolution *)resolution
{
	NSError *last = resolution.reasons.lastObject;
	return [last.domain isEqualToString:NFKModelResolutionErrorDomain] && last.code == NFKModelResolutionErrorUnverifiedRouteNotAllowed;
}

/*! The refusal of a route that applies while allowsUnverifiedRoutes is NO, ending the resolution. */
- (NFKModelResolution *)unverifiedResolutionOfRepository:(NFKModelRepository *)repository
												   route:(NFKModelRoute)route
												 reasons:(NSMutableArray<NSError *> *)reasons
{
	NSString *reason = [NSString stringWithFormat:@"the %@ route reads %@, and no reference measured its output; "
						"allowsUnverifiedRoutes is NO", NFKModelRouteName(route), repository.repository];
	[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorUnverifiedRouteNotAllowed, route, nil, reason, nil)];
	return [self refusedResolutionOfRepository:repository.repository revision:repository.revision reasons:reasons];
}

#pragma mark Catalog route

- (nullable NFKModelResolution *)catalogResolutionForRepository:(NSString *)repository
													   revision:(nullable NSString *)revision
														 commit:(nullable NSString *)commit
{
	for (NFKModelDescriptor *descriptor in self.catalog.descriptors) {
		if (descriptor.isStandIn) {
			continue;
		}
		for (NFKModelVariant *variant in descriptor.variants) {
			if ([self variant:variant readsRepository:repository revision:revision commit:commit]) {
				return [self resolutionOfRepository:repository revision:commit ?: revision route:NFKModelRouteCatalog
									   verification:NFKModelVerificationMeasured descriptor:descriptor
											variant:variant reasons:nil];
			}
		}
	}
	return nil;
}

/*! YES when the variant's source is the repository at the revision named: any revision when the caller
	named none, else the source's revision or the commit it was measured at. */
- (BOOL)variant:(NFKModelVariant *)variant
	readsRepository:(NSString *)repository
		   revision:(nullable NSString *)revision
			 commit:(nullable NSString *)commit
{
	NFKModelSource *source = variant.source;
	if (![source.kind isEqualToString:NFKModelSourceKindHuggingFace] || source.repository == nil
		|| [source.repository caseInsensitiveCompare:repository] != NSOrderedSame) {
		return NO;
	}
	if (revision == nil) {
		return YES;
	}
	NSMutableSet<NSString *> *pinned = [NSMutableSet set];
	if (source.revision != nil) {
		[pinned addObject:source.revision];
	}
	if ([source.extensions[@"measuredRevision"] isKindOfClass:NSString.class]) {
		[pinned addObject:source.extensions[@"measuredRevision"]];
	}
	return [pinned containsObject:revision] || (commit != nil && [pinned containsObject:commit]);
}

#pragma mark Metadata

/*! The hub's answer for the repository at the revision: its commit (sha), model card (cardData), and
	gating (gated). */
- (nullable NSDictionary<NSString *, id> *)informationOfRepository:(NSString *)repository
														  revision:(nullable NSString *)revision
															 error:(NSError **)error
{
	NSMutableCharacterSet *segment = [NSCharacterSet.URLPathAllowedCharacterSet mutableCopy];
	[segment removeCharactersInString:@"/"];
	NSString *encodedRevision = [revision ?: @"main" stringByAddingPercentEncodingWithAllowedCharacters:segment];
	NSString *base = self.hub.endpointURL.absoluteString;
	while ([base hasSuffix:@"/"]) {
		base = [base substringToIndex:base.length - 1];
	}
	NSURL *URL = [NSURL URLWithString:[NSString stringWithFormat:@"%@/api/models/%@/revision/%@", base, repository,
									   encodedRevision]];
	if (URL == nil || [repository componentsSeparatedByString:@"/"].count != 2) {
		NSString *reason = [NSString stringWithFormat:@"\"%@\" is not a repository name of the form owner/name", repository];
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKDownloadErrorDomain code:NFKDownloadErrorUnsafePath
									 userInfo:@{ NSLocalizedDescriptionKey: reason }];
		}
		return nil;
	}
	NSHTTPURLResponse *response = nil;
	NSData *body = [self.hub fetchMetadataFromURL:URL HTTPMethod:@"GET" response:&response error:error];
	if (body == nil) {
		return nil;
	}
	NSInteger status = response != nil ? response.statusCode : 200;
	if (status < 200 || status >= 300) {
		if (error != NULL) {
			NSString *subject = [NSString stringWithFormat:@"%@ at %@", repository, revision ?: @"main"];
			*error = [NFKHFHubTransfer errorForHTTPStatus:status subject:subject
												 hadToken:self.hub.accessToken.length > 0];
		}
		return nil;
	}
	id information = [NSJSONSerialization JSONObjectWithData:body options:0 error:NULL];
	return [information isKindOfClass:NSDictionary.class] ? information : @{};
}

/*! YES when the hub's answer marks the repository gated: its users accept terms before its files serve. */
- (BOOL)isGated:(NSDictionary *)information
{
	id gated = information[@"gated"];
	return gated != nil && ![gated isEqual:@NO] && ![gated isEqual:@"false"];
}

/*! The license the model card states. */
- (nullable NFKModelLicense *)licenseOfRepository:(NSString *)repository information:(NSDictionary *)information
{
	NSDictionary *card = [information[@"cardData"] isKindOfClass:NSDictionary.class] ? information[@"cardData"] : nil;
	id identifier = card[@"license"];
	if ([identifier isKindOfClass:NSArray.class]) {
		identifier = [identifier firstObject];
	}
	if (![identifier isKindOfClass:NSString.class]) {
		return nil;
	}
	NSString *name = [card[@"license_name"] isKindOfClass:NSString.class] ? card[@"license_name"] : identifier;
	NSString *link = [card[@"license_link"] isKindOfClass:NSString.class] ? card[@"license_link"] : nil;
	NSURL *URL = link != nil ? [NSURL URLWithString:link relativeToURL:[self.hub.endpointURL URLByAppendingPathComponent:repository]]
							 : [self.hub.endpointURL URLByAppendingPathComponent:repository];
	return [[NFKModelLicense alloc] initWithIdentifier:identifier name:name URL:URL.absoluteURL gated:[self isGated:information]
								restrictsCommercialUse:NO note:nil extensions:nil];
}

/*! The listed repository with its configuration, as a loader judges it, or nil with the family route's
	refusal or an error (a failed read). */
- (nullable NFKModelRepository *)configuredRepository:(NFKModelRepository *)listed
											  refusal:(NSError **)refusal
												error:(NSError **)error
{
	NSString *repository = listed.repository;
	if ([listed fileAtPath:NFKModelResolutionConfigPath] == nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ has no config.json", repository];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorNoConfiguration, NFKModelRouteFamily, nil, reason, nil));
		return nil;
	}
	NSError *readError = nil;
	id configuration = [listed JSONObjectOfFileAtPath:NFKModelResolutionConfigPath error:&readError];
	if (NFKModelResolutionIsAuthenticationError(readError)) {
		NFKModelResolutionSet(refusal, [self gatedRefusalOfRepository:repository error:readError]);
		return nil;
	}
	if (configuration == nil && ![readError.domain isEqualToString:NSCocoaErrorDomain]) {
		NFKModelResolutionSet(error, readError);
		return nil;
	}
	if (![configuration isKindOfClass:NSDictionary.class]) {
		NSString *reason = [NSString stringWithFormat:@"the config.json of %@ is not a JSON object", repository];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorNoConfiguration, NFKModelRouteFamily, nil, reason, readError));
		return nil;
	}
	NFKModelRepository *read = [[NFKModelRepository alloc] initWithRepository:repository revision:listed.revision
															requestedRevision:listed.requestedRevision
																configuration:configuration files:listed.files
																	  license:listed.license hub:self.hub];
	NSError *judgment = [self refusalOfRepository:read];
	NFKModelResolutionSet(refusal, judgment);
	return judgment == nil ? read : nil;
}

/*! The reason no loader reads the repository whatever its model type, or nil. */
- (nullable NSError *)refusalOfRepository:(NFKModelRepository *)repository
{
	if (repository.configuration[@"auto_map"] != nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ runs code it ships (auto_map), so its model type does not "
							"describe its computation", repository.repository];
		return NFKModelResolutionRefusal(NFKModelResolutionErrorCustomCode, NFKModelRouteFamily, nil, reason, nil);
	}
	if ([repository pathsWithSuffix:@".safetensors"].count == 0) {
		NSArray<NSString *> *others = [[[repository pathsWithSuffix:@".bin"] arrayByAddingObjectsFromArray:
										[repository pathsWithSuffix:@".pt"]]
									   arrayByAddingObjectsFromArray:[repository pathsWithSuffix:@".gguf"]];
		NSString *reason = others.count > 0
			? [NSString stringWithFormat:@"%@ holds no safetensors weights, only %@", repository.repository,
			   [others componentsJoinedByString:@", "]]
			: [NSString stringWithFormat:@"%@ holds no weights", repository.repository];
		return NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedWeightFormat, NFKModelRouteFamily, nil, reason, nil);
	}
	if (repository.modelType == nil) {
		NSString *reason = [NSString stringWithFormat:@"the config.json of %@ states no model_type", repository.repository];
		return NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedArchitecture, NFKModelRouteFamily, nil, reason, nil);
	}
	return nil;
}

/*! The refusal of a repository whose metadata the hub withholds. Without an accepted access token the hub
	answers a private repository and a missing one alike (HTTP 401 with no X-Error-Code), so the refusal
	names both. */
- (NSError *)unreadableRefusalOfRepository:(NSString *)repository error:(nullable NSError *)error
{
	NSString *reason = [NSString stringWithFormat:@"the hub withholds %@: it is private to an account whose token "
						"was not sent, or it does not exist", repository];
	return NFKModelResolutionRefusal(NFKModelResolutionErrorGated, NFKModelRouteNone, nil, reason, error);
}

- (NSError *)gatedRefusalOfRepository:(NSString *)repository error:(nullable NSError *)error
{
	NSString *reason = [NSString stringWithFormat:@"%@ refuses its files without an access token whose account "
						"accepted its terms", repository];
	return NFKModelResolutionRefusal(NFKModelResolutionErrorGated, NFKModelRouteNone, nil, reason, error);
}

/*! A resolution refusing a gated repository, or nil with the error for any other failed listing. */
- (nullable NFKModelResolution *)gatedResolutionOfRepository:(NSString *)repository
													revision:(NSString *)revision
													   error:(nullable NSError *)planError
													 failure:(NSError **)error
{
	if (NFKModelResolutionIsAuthenticationError(planError)) {
		return [self refusedResolutionOfRepository:repository revision:revision
										   reasons:@[ [self gatedRefusalOfRepository:repository error:planError] ]];
	}
	if (error != NULL) {
		*error = planError;
	}
	return nil;
}

#pragma mark Family route

/*! The family route: the first matching descriptor whose provider class builds a descriptor for the
	repository. A matching descriptor whose provider class and family were judged already is skipped, since
	a provider judges a repository by its family's loader. Returns nil and sets error when a judgment fails
	to read the repository. */
- (nullable NFKModelResolution *)familyResolutionOfRepository:(NFKModelRepository *)repository error:(NSError **)error
{
	NSMutableArray<NSError *> *reasons = [NSMutableArray array];
	NSMutableSet<NSString *> *judged = [NSMutableSet set];
	for (NFKModelDescriptor *candidate in self.catalog.descriptors) {
		if (candidate.isStandIn || ![candidate.modelTypes containsObject:repository.modelType]) {
			continue;
		}
		NSString *providerName = [self.catalog providerClassNameForDescriptorIdentifier:candidate.identifier];
		NSString *judgment = [NSString stringWithFormat:@"%@\n%@", providerName ?: @"", candidate.family];
		if ([judged containsObject:judgment]) {
			continue;
		}
		[judged addObject:judgment];
		NSError *refusal = nil;
		NSError *failure = nil;
		NFKModelDescriptor *descriptor = [self descriptorForRepository:repository matching:candidate
														  providerName:providerName refusal:&refusal failure:&failure];
		if (failure != nil) {
			NFKModelResolutionSet(error, failure);
			return nil;
		}
		if (descriptor == nil) {
			if (refusal != nil) {
				[reasons addObject:refusal];
			}
			continue;
		}
		if (!self.allowsUnverifiedRoutes) {
			NSString *reason = [NSString stringWithFormat:@"the loader of %@ reads %@, and no reference measured its "
								"output; allowsUnverifiedRoutes is NO", candidate.identifier, repository.repository];
			[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorUnverifiedRouteNotAllowed,
														 NFKModelRouteFamily, candidate.identifier, reason, nil)];
			return [self refusedResolutionOfRepository:repository.repository revision:repository.revision reasons:reasons];
		}
		NFKModelDescriptor *stamped = [self descriptor:descriptor resolvedFrom:repository basedOn:candidate];
		return [self resolutionOfRepository:repository.repository revision:repository.revision route:NFKModelRouteFamily
							   verification:NFKModelVerificationConfigCompatible descriptor:stamped
									variant:stamped.defaultVariant reasons:reasons];
	}
	if (reasons.count == 0) {
		NSString *reason = [NSString stringWithFormat:@"no model in the catalog reads the model type \"%@\" of %@",
							repository.modelType, repository.repository];
		[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedArchitecture, NFKModelRouteFamily,
													 nil, reason, nil)];
	}
	return [self refusedResolutionOfRepository:repository.repository revision:repository.revision reasons:reasons];
}

/*! YES for an error that says the repository could not be read, not that the loader refuses it: the hub's
	and the URL system's. */
static BOOL NFKModelResolutionIsReadFailure(NSError *error)
{
	return [error.domain isEqualToString:NFKDownloadErrorDomain] || [error.domain isEqualToString:NSURLErrorDomain];
}

/*! The candidate's provider class's descriptor for the repository, or nil with its refusal, or nil with
	the failure that kept the class from reading the repository. An authentication failure is a gated
	refusal. A candidate whose class does not judge repositories gives nil with neither. */
- (nullable NFKModelDescriptor *)descriptorForRepository:(NFKModelRepository *)repository
												matching:(NFKModelDescriptor *)candidate
											providerName:(nullable NSString *)providerName
												 refusal:(NSError **)refusal
												 failure:(NSError **)failure
{
	Class provider = providerName != nil ? NSClassFromString(providerName) : Nil;
	if (![provider respondsToSelector:@selector(descriptorForRepository:matchingDescriptor:error:)]) {
		return nil;
	}
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [(Class<NFKModelDescriptorProvider>)provider descriptorForRepository:repository
																					  matchingDescriptor:candidate
																								   error:&error];
	if (descriptor != nil) {
		return descriptor;
	}
	if (NFKModelResolutionIsAuthenticationError(error)) {
		NSError *gated = [self gatedRefusalOfRepository:repository.repository error:error];
		NSMutableDictionary<NSErrorUserInfoKey, id> *userInfo = [gated.userInfo mutableCopy];
		userInfo[NFKModelResolutionRouteKey] = @(NFKModelRouteFamily);
		userInfo[NFKModelResolutionDescriptorKey] = candidate.identifier;
		NFKModelResolutionSet(refusal, [NSError errorWithDomain:gated.domain code:gated.code userInfo:userInfo]);
		return nil;
	}
	if (error != nil && NFKModelResolutionIsReadFailure(error)) {
		NFKModelResolutionSet(failure, error);
		return nil;
	}
	NSMutableDictionary<NSErrorUserInfoKey, id> *userInfo = [error.userInfo mutableCopy] ?: [NSMutableDictionary dictionary];
	userInfo[NFKModelResolutionRouteKey] = @(NFKModelRouteFamily);
	userInfo[NFKModelResolutionDescriptorKey] = candidate.identifier;
	if (userInfo[NSLocalizedDescriptionKey] == nil) {
		userInfo[NSLocalizedDescriptionKey] = [NSString stringWithFormat:@"the loader of %@ does not read %@",
											   candidate.identifier, repository.repository];
	}
	BOOL resolutionError = [error.domain isEqualToString:NFKModelResolutionErrorDomain];
	if (!resolutionError && error != nil) {
		userInfo[NSUnderlyingErrorKey] = error;
	}
	NFKModelResolutionSet(refusal, [NSError errorWithDomain:NFKModelResolutionErrorDomain
													   code:resolutionError ? error.code
																			: NFKModelResolutionErrorUnsupportedConfiguration
												   userInfo:userInfo]);
	return nil;
}

/*! The descriptor with the resolution extension beside its own extensions. */
- (NFKModelDescriptor *)descriptor:(NFKModelDescriptor *)descriptor
					  resolvedFrom:(NFKModelRepository *)repository
						   basedOn:(NFKModelDescriptor *)candidate
{
	NSMutableDictionary<NSString *, id> *extensions = [descriptor.extensions mutableCopy];
	extensions[NFKModelResolutionExtensionKey] = @{ @"verification": NFKModelVerificationConfigCompatible,
													@"repository": repository.repository,
													@"revision": repository.revision,
													@"basedOn": candidate.identifier };
	return [[NFKModelDescriptor alloc] initWithIdentifier:descriptor.identifier displayName:descriptor.displayName
												   family:descriptor.family summary:descriptor.summary
												   engine:descriptor.engine registryName:descriptor.registryName
										providerClassName:descriptor.providerClassName tasks:descriptor.tasks
										  inputModalities:descriptor.inputModalities
										 outputModalities:descriptor.outputModalities inputs:descriptor.inputs
												  outputs:descriptor.outputs options:descriptor.options
												 variants:descriptor.variants license:descriptor.license
											 requirements:descriptor.requirements residencies:descriptor.residencies
											  memoryModel:descriptor.memoryModel presets:descriptor.presets
											   modelTypes:descriptor.modelTypes standIn:descriptor.isStandIn
										introducedVersion:descriptor.introducedVersion extensions:extensions];
}

#pragma mark Core ML package route

/*! The Core ML package route: a manifest.json Tools/inferkit-convert writes, the package it names, and its
	tokenizer files. Returns nil and appends the refusal when the repository holds a manifest the runner does
	not read, nil alone when it holds none, and nil with failure set when a read fails. */
- (nullable NFKModelResolution *)packageResolutionOfRepository:(NFKModelRepository *)listed
													   reasons:(NSMutableArray<NSError *> *)reasons
													   failure:(NSError **)failure
{
	if ([listed fileAtPath:NFKModelResolutionManifestPath] == nil) {
		return nil;
	}
	NSError *readError = nil;
	id manifest = [listed JSONObjectOfFileAtPath:NFKModelResolutionManifestPath error:&readError];
	if (NFKModelResolutionIsAuthenticationError(readError)) {
		[reasons addObject:[self gatedRefusalOfRepository:listed.repository error:readError]];
		return nil;
	}
	if (manifest == nil && ![readError.domain isEqualToString:NSCocoaErrorDomain]) {
		NFKModelResolutionSet(failure, readError);
		return nil;
	}
	NSError *refusal = nil;
	NSArray<NFKDownloadPlanFile *> *files = [self packageFilesOfRepository:listed manifest:manifest refusal:&refusal];
	if (files == nil) {
		[reasons addObject:refusal];
		return nil;
	}
	if (!self.allowsUnverifiedRoutes) {
		return [self unverifiedResolutionOfRepository:listed route:NFKModelRouteCoreMLPackage reasons:reasons];
	}
	NSError *descriptorError = nil;
	NFKModelDescriptor *descriptor = [self packageDescriptorOfRepository:listed manifest:manifest files:files
																   error:&descriptorError];
	if (descriptor == nil) {
		NSString *reason = [NSString stringWithFormat:@"the Core ML package of %@ does not describe a model: %@",
							listed.repository, descriptorError.localizedDescription];
		[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedPackage, NFKModelRouteCoreMLPackage,
													 nil, reason, descriptorError)];
		return nil;
	}
	return [self resolutionOfRepository:listed.repository revision:listed.revision route:NFKModelRouteCoreMLPackage
						   verification:NFKModelVerificationExternal descriptor:descriptor
								variant:descriptor.defaultVariant reasons:reasons action:nil];
}

/*! The files NFKCoreMLLanguageBackend reads, as NFKTokenizer and -loadModelFromDirectory: name them: the
	manifest, every file of the model it names (model.mlmodelc, model.mlpackage, or model.mlmodel when it
	names none), and the tokenizer's files. Returns nil with a refusal for a manifest the runner does not
	read. */
- (nullable NSArray<NFKDownloadPlanFile *> *)packageFilesOfRepository:(NFKModelRepository *)listed
															 manifest:(id)manifest
															  refusal:(NSError **)refusal
{
	NSString *name = listed.repository;
	if (![manifest isKindOfClass:NSDictionary.class]) {
		NSString *reason = [NSString stringWithFormat:@"the manifest.json of %@ is not a JSON object", name];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedPackage, NFKModelRouteCoreMLPackage, nil, reason, nil));
		return nil;
	}
	id modelType = manifest[@"modelType"];
	if (modelType != nil && ![modelType isEqual:@"causal-lm"]) {
		NSString *reason = [NSString stringWithFormat:@"the Core ML package of %@ is a %@, and the runner runs a causal "
							"language model", name, modelType];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedPackage, NFKModelRouteCoreMLPackage, nil, reason, nil));
		return nil;
	}
	NSArray<NSString *> *candidates = [manifest[@"model"] isKindOfClass:NSString.class]
		? @[ manifest[@"model"] ] : @[ @"model.mlmodelc", @"model.mlpackage", @"model.mlmodel" ];
	NSArray<NFKDownloadPlanFile *> *modelFiles = @[];
	for (NSString *candidate in candidates) {
		modelFiles = [self filesOfRepository:listed underPath:candidate];
		if (modelFiles.count > 0) {
			break;
		}
	}
	if (modelFiles.count == 0) {
		NSString *reason = [NSString stringWithFormat:@"%@ lists no %@, the model its manifest.json names", name,
							[candidates componentsJoinedByString:@" or "]];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedPackage, NFKModelRouteCoreMLPackage, nil, reason, nil));
		return nil;
	}
	NSArray<NSString *> *tokenizerPaths = [self tokenizerPathsOfManifest:manifest];
	NSString *missing = nil;
	for (NSString *path in tokenizerPaths) {
		if ([listed fileAtPath:path] == nil) {
			missing = path;
			break;
		}
	}
	if (tokenizerPaths == nil || missing != nil) {
		NSString *reason = tokenizerPaths == nil
			? [NSString stringWithFormat:@"the manifest.json of %@ names a tokenizer the runtime does not read", name]
			: [NSString stringWithFormat:@"%@ lists no %@, which its manifest.json's tokenizer reads", name, missing];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedTokenizer, NFKModelRouteCoreMLPackage, nil, reason, nil));
		return nil;
	}
	NSMutableArray<NFKDownloadPlanFile *> *files = [NSMutableArray arrayWithObject:[listed fileAtPath:NFKModelResolutionManifestPath]];
	[files addObjectsFromArray:modelFiles];
	for (NSString *path in tokenizerPaths) {
		[files addObject:[listed fileAtPath:path]];
	}
	return files;
}

/*! The listed files at path or under it. */
- (NSArray<NFKDownloadPlanFile *> *)filesOfRepository:(NFKModelRepository *)listed underPath:(NSString *)path
{
	NSString *prefix = [path stringByAppendingString:@"/"];
	NSMutableArray<NFKDownloadPlanFile *> *files = [NSMutableArray array];
	for (NFKDownloadPlanFile *file in listed.files) {
		if ([file.path isEqualToString:path] || [file.path hasPrefix:prefix]) {
			[files addObject:file];
		}
	}
	return files;
}

/*! The files the manifest's tokenizer reads, with NFKTokenizer's defaults, or nil for a type it does not
	read. */
- (nullable NSArray<NSString *> *)tokenizerPathsOfManifest:(NSDictionary *)manifest
{
	NSDictionary *spec = [manifest[@"tokenizer"] isKindOfClass:NSDictionary.class] ? manifest[@"tokenizer"] : nil;
	NSString *type = [spec[@"type"] isKindOfClass:NSString.class] ? spec[@"type"] : nil;
	NSString *(^named)(NSString *, NSString *) = ^NSString *(NSString *key, NSString *fallback) {
		return [spec[key] isKindOfClass:NSString.class] ? spec[key] : fallback;
	};
	if ([@[ @"clip", @"bpe-clip", @"bpe-bytelevel", @"bytelevel-bpe", @"bpe" ] containsObject:type ?: @""]) {
		return @[ named(@"vocab", @"vocab.json"), named(@"merges", @"merges.txt") ];
	}
	if ([type isEqualToString:@"unigram"] || [type isEqualToString:@"sentencepiece"]) {
		return @[ named(@"vocab", @"unigram.json") ];
	}
	if ([type isEqualToString:@"wordpiece"]) {
		return @[ named(@"vocab", @"wordpiece.json") ];
	}
	return nil;
}

/*! The descriptor of a Core ML package: a text decoder NFKCoreMLLanguageBackend runs on macOS 15, iOS 18,
	and tvOS 18, whose memory is the package's bytes. */
- (nullable NFKModelDescriptor *)packageDescriptorOfRepository:(NFKModelRepository *)listed
													  manifest:(NSDictionary *)manifest
														 files:(NSArray<NFKDownloadPlanFile *> *)files
														 error:(NSError **)error
{
	long long modelBytes = 0;
	long long downloadBytes = 0;
	NSMutableArray<NSDictionary *> *fileObjects = [NSMutableArray array];
	NSString *manifestPath = NFKModelResolutionManifestPath;
	NSSet<NSString *> *tokenizerPaths = [NSSet setWithArray:[self tokenizerPathsOfManifest:manifest] ?: @[]];
	for (NFKDownloadPlanFile *file in files) {
		NSString *role = [file.path isEqualToString:manifestPath] ? NFKModelFileRoleConfig
			: [tokenizerPaths containsObject:file.path] ? NFKModelFileRoleTokenizer : NFKModelFileRoleWeights;
		if ([role isEqualToString:NFKModelFileRoleWeights]) {
			modelBytes += MAX(file.byteCount, 0);
		}
		downloadBytes += MAX(file.byteCount, 0);
		NSMutableDictionary *object = [@{ @"path": file.path, @"role": role } mutableCopy];
		object[@"bytes"] = file.byteCount >= 0 ? @(file.byteCount) : nil;
		object[@"sha256"] = file.sha256;
		[fileObjects addObject:object];
	}
	NSMutableDictionary *facts = [@{ @"stages": @[ @{ @"id": @"model", @"bytesByPrecision": @{ NFKModelPrecisionCheckpoint: @(modelBytes) } } ] } mutableCopy];
	facts[@"maximumContextLength"] = [manifest[@"contextLength"] isKindOfClass:NSNumber.class] ? manifest[@"contextLength"] : nil;
	facts[@"vocabularySize"] = [manifest[@"vocabSize"] isKindOfClass:NSNumber.class] ? manifest[@"vocabSize"] : nil;
	NSDictionary *spec = [manifest[@"tokenizer"] isKindOfClass:NSDictionary.class] ? manifest[@"tokenizer"] : @{};
	NSMutableArray *inputs = [NSMutableArray arrayWithObject:@{ @"key": NFKInputPrompt, @"required": @YES }];
	if (spec[@"chatTemplate"] != nil) {
		[inputs addObject:@{ @"key": NFKInputMessages }];
	}
	NSDictionary *variant = @{ @"id": [self variantIdentifierOfRepository:listed.repository], @"isDefault": @YES,
							   @"source": @{ @"kind": NFKModelSourceKindHuggingFace, @"repository": listed.repository,
											 @"revision": listed.revision, @"layout": NFKModelSourceLayoutDirectory,
											 @"files": fileObjects },
							   @"downloadBytes": @(downloadBytes), @"residentBytes": @(modelBytes), @"memoryFacts": facts };
	NSMutableDictionary *object = [@{
		@"id": listed.repository, @"displayName": listed.repository, @"family": NFKModelEngineCoreML,
		@"summary": [NSString stringWithFormat:@"%@, a Core ML language model NFKCoreMLLanguageBackend runs.", listed.repository],
		@"engine": NFKModelEngineCoreML, @"tasks": @[ NFKModelTaskTextGeneration ], @"inputs": inputs,
		@"outputs": @[ @{ @"key": NFKOutputText, @"required": @YES } ], @"variants": @[ variant ],
		@"requirements": @{ @"minimumOS": @{ @"macOS": @"15.0", @"iOS": @"18.0", @"tvOS": @"18.0" } },
		NFKModelResolutionExtensionKey: @{ @"verification": NFKModelVerificationExternal, @"repository": listed.repository,
										   @"revision": listed.revision },
	} mutableCopy];
	object[@"license"] = listed.license.dictionaryRepresentation;
	return [NFKModelDescriptor descriptorWithDictionary:object error:error];
}

#pragma mark Local-server route

/*! The local-server route: a server among localServers that holds the repository already, else one whose
	runner pulls (Ollama) for a repository holding a single-file GGUF. Appends a refusal only for a GGUF
	repository no server pulls; a server that answers nothing is not a refusal of another repository. */
- (nullable NFKModelResolution *)serverResolutionOfRepository:(NFKModelRepository *)listed
														gated:(BOOL)gated
													  reasons:(NSMutableArray<NSError *> *)reasons
{
	NSArray<NFKRemoteProvider *> *servers = self.localServers;
	if (servers.count == 0) {
		return nil;
	}
	NFKDownloadPlanFile *gguf = [self preferredGGUFOfRepository:listed];
	NSArray<NFKRemoteProvider *> *available = [NFKRemoteProvider availableProvidersAmong:servers
																			   timeout:NFKModelResolutionServerTimeout];
	for (NFKRemoteProvider *provider in available) {
		NFKRemoteModel *held = [self modelOfServer:provider holdingRepository:listed.repository];
		if (held != nil) {
			NFKModelResolutionAction *action = [[NFKModelResolutionAction alloc] initWithKind:NFKModelResolutionActionKindUse
																		  providerIdentifier:provider.identifier
																					   model:held.identifier path:nil
																				   byteCount:nil];
			return [self serverResolutionOfRepository:listed action:action byteCount:held.sizeBytes reasons:reasons];
		}
	}
	if (gguf == nil) {
		return nil;
	}
	if (gated) {
		NSString *reason = [NSString stringWithFormat:@"%@ is gated, and a server pulls it without the account's token",
							listed.repository];
		[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorGated, NFKModelRouteLocalServer, nil, reason, nil)];
		return nil;
	}
	NFKRemoteProvider *puller = nil;
	for (NFKRemoteProvider *provider in available) {
		if ([provider.localRunner respondsToSelector:@selector(pullModel:)]) {
			puller = provider;
			break;
		}
	}
	if (puller == nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ holds GGUF weights, and no server that pulls them answers "
							"on this machine (Ollama does)", listed.repository];
		[reasons addObject:NFKModelResolutionRefusal(NFKModelResolutionErrorNoServer, NFKModelRouteLocalServer, nil, reason, nil)];
		return nil;
	}
	NSString *model = [NSString stringWithFormat:@"hf.co/%@:%@", listed.repository, [self quantizationOfGGUF:gguf.path]];
	NFKModelResolutionAction *action = [[NFKModelResolutionAction alloc] initWithKind:NFKModelResolutionActionKindPull
																  providerIdentifier:puller.identifier model:model
																				path:gguf.path
																		   byteCount:gguf.byteCount >= 0 ? @(gguf.byteCount) : nil];
	return [self serverResolutionOfRepository:listed action:action byteCount:action.byteCount reasons:reasons];
}

- (nullable NFKModelResolution *)serverResolutionOfRepository:(NFKModelRepository *)listed
													   action:(NFKModelResolutionAction *)action
													byteCount:(nullable NSNumber *)byteCount
													  reasons:(NSMutableArray<NSError *> *)reasons
{
	if (!self.allowsUnverifiedRoutes) {
		return [self unverifiedResolutionOfRepository:listed route:NFKModelRouteLocalServer reasons:reasons];
	}
	NSDictionary *variant = byteCount != nil
		? @{ @"id": [self variantIdentifierOfRepository:listed.repository], @"isDefault": @YES, @"registryName": action.model,
			 @"source": @{ @"kind": NFKModelSourceKindNone }, @"residentBytes": byteCount,
			 @"memoryFacts": @{ @"stages": @[ @{ @"id": @"model", @"bytesByPrecision": @{ NFKModelPrecisionCheckpoint: byteCount } } ] } }
		: @{ @"id": [self variantIdentifierOfRepository:listed.repository], @"isDefault": @YES, @"registryName": action.model,
			 @"source": @{ @"kind": NFKModelSourceKindNone } };
	NSMutableDictionary *object = [@{
		@"id": listed.repository, @"displayName": listed.repository, @"family": NFKModelEngineLocalServer,
		@"summary": [NSString stringWithFormat:@"%@, served as %@ by the %@ server on this machine.", listed.repository,
					 action.model, action.providerIdentifier],
		@"engine": NFKModelEngineLocalServer, @"registryName": action.model, @"tasks": @[ NFKModelTaskTextGeneration ],
		@"inputs": @[ @{ @"key": NFKInputPrompt }, @{ @"key": NFKInputMessages } ],
		@"outputs": @[ @{ @"key": NFKOutputText, @"required": @YES } ], @"variants": @[ variant ],
		NFKModelResolutionExtensionKey: @{ @"verification": NFKModelVerificationExternal, @"repository": listed.repository,
										   @"revision": listed.revision },
		@"localServer": @{ @"provider": action.providerIdentifier, @"model": action.model },
	} mutableCopy];
	object[@"license"] = listed.license.dictionaryRepresentation;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:object error:NULL];
	return [self resolutionOfRepository:listed.repository revision:listed.revision route:NFKModelRouteLocalServer
						   verification:NFKModelVerificationExternal descriptor:descriptor
								variant:descriptor.defaultVariant reasons:reasons action:action];
}

/*! The model a server holds for the repository: its own name for it, the repository's name, or a
	Hugging Face pull of it (hf.co/<repository>:<tag>, huggingface.co/<repository>:<tag>), compared without
	case. An Ollama or LM Studio server lists its installed models, another its served ones. */
- (nullable NFKRemoteModel *)modelOfServer:(NFKRemoteProvider *)provider holdingRepository:(NSString *)repository
{
	id<NFKLocalModelRunner> runner = provider.localRunner;
	NSArray<NFKRemoteModel *> *models = runner != nil ? [runner installedModelsWithError:NULL]
		: [[NFKRemoteModelCatalog catalogForProvider:provider apiKey:nil] modelsWithError:NULL];
	NSString *name = repository.lowercaseString;
	for (NFKRemoteModel *model in models) {
		NSString *identifier = model.identifier.lowercaseString;
		for (NSString *host in @[ @"hf.co/", @"huggingface.co/" ]) {
			if ([identifier hasPrefix:host]) {
				identifier = [identifier substringFromIndex:host.length];
			}
		}
		NSRange tag = [identifier rangeOfString:@":" options:NSBackwardsSearch];
		NSString *bare = tag.location != NSNotFound ? [identifier substringToIndex:tag.location] : identifier;
		if ([identifier isEqualToString:name] || [bare isEqualToString:name]) {
			return model;
		}
	}
	return nil;
}

/*! The GGUF a pull fetches: the first by quantization preference, else the smallest. A file split into
	parts (name-00001-of-00003.gguf) is not one a pull names. */
- (nullable NFKDownloadPlanFile *)preferredGGUFOfRepository:(NFKModelRepository *)listed
{
	NSRegularExpression *split = [NSRegularExpression regularExpressionWithPattern:@"-\\d{5}-of-\\d{5}\\.gguf$" options:0 error:NULL];
	NSMutableArray<NFKDownloadPlanFile *> *candidates = [NSMutableArray array];
	for (NFKDownloadPlanFile *file in listed.files) {
		NSString *path = file.path;
		if ([path.lowercaseString hasSuffix:@".gguf"]
			&& [split firstMatchInString:path options:0 range:NSMakeRange(0, path.length)] == nil
			&& [self quantizationOfGGUF:path].length > 0) {
			[candidates addObject:file];
		}
	}
	for (NSString *preferred in @[ @"Q4_K_M", @"Q4_K_S", @"Q5_K_M", @"Q4_0", @"Q6_K", @"Q8_0" ]) {
		for (NFKDownloadPlanFile *file in candidates) {
			if ([[self quantizationOfGGUF:file.path] caseInsensitiveCompare:preferred] == NSOrderedSame) {
				return file;
			}
		}
	}
	return [candidates sortedArrayUsingComparator:^NSComparisonResult(NFKDownloadPlanFile *a, NFKDownloadPlanFile *b) {
		return [@(a.byteCount) compare:@(b.byteCount)];
	}].firstObject;
}

/*! The quantization a GGUF file name states, the tag an Ollama pull names: the last of Q4_K_M, IQ3_XXS,
	Q8_0, F16, BF16, F32, in upper case; empty where the name states none. */
- (NSString *)quantizationOfGGUF:(NSString *)path
{
	NSString *name = path.lastPathComponent.stringByDeletingPathExtension;
	NSRegularExpression *pattern = [NSRegularExpression regularExpressionWithPattern:
									@"(?i)(?:^|[-_.])((?:I?Q\\d(?:_[A-Z0-9]+)*)|BF16|F16|F32)$" options:0 error:NULL];
	NSTextCheckingResult *match = [pattern firstMatchInString:name options:0 range:NSMakeRange(0, name.length)];
	return match != nil ? [name substringWithRange:[match rangeAtIndex:1]].uppercaseString : @"";
}

/*! A resolved variant's identifier: the repository's own name, lowercased. */
- (NSString *)variantIdentifierOfRepository:(NSString *)repository
{
	return (repository.lastPathComponent ?: repository).lowercaseString;
}

#pragma mark Results

- (NFKModelResolution *)resolutionOfRepository:(NSString *)repository
									  revision:(nullable NSString *)revision
										 route:(NFKModelRoute)route
								  verification:(NFKModelVerification)verification
									descriptor:(NFKModelDescriptor *)descriptor
									   variant:(nullable NFKModelVariant *)variant
									   reasons:(nullable NSArray<NSError *> *)reasons
{
	return [self resolutionOfRepository:repository revision:revision route:route verification:verification
							 descriptor:descriptor variant:variant reasons:reasons action:nil];
}

- (NFKModelResolution *)resolutionOfRepository:(NSString *)repository
									  revision:(nullable NSString *)revision
										 route:(NFKModelRoute)route
								  verification:(NFKModelVerification)verification
									descriptor:(NFKModelDescriptor *)descriptor
									   variant:(nullable NFKModelVariant *)variant
									   reasons:(nullable NSArray<NSError *> *)reasons
										action:(nullable NFKModelResolutionAction *)action
{
	NSDictionary<NSString *, id> *options = descriptor.variants.count > 1 && variant != nil
		? @{ NFKModelLoadOptionVariant: variant.identifier } : nil;
	NFKModelMemoryBudget *budget = self.memoryBudget ?: self.catalog.memoryBudget;
	NFKModelMemoryEstimate *estimate = [descriptor estimatedMemoryForOptions:options memoryBudget:budget error:NULL];
	return [[NFKModelResolution alloc] initWithRepository:repository revision:revision route:route
											 verification:verification descriptor:descriptor
										variantIdentifier:variant.identifier memoryEstimate:estimate
										downloadByteCount:variant.downloadByteCount reasons:reasons action:action];
}

- (NFKModelResolution *)refusedResolutionOfRepository:(NSString *)repository
											 revision:(nullable NSString *)revision
											  reasons:(NSArray<NSError *> *)reasons
{
	return [[NFKModelResolution alloc] initWithRepository:repository revision:revision route:NFKModelRouteNone
											 verification:nil descriptor:nil variantIdentifier:nil memoryEstimate:nil
										downloadByteCount:nil reasons:reasons];
}

@end
