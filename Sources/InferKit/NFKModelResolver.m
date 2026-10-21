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

NSErrorDomain const NFKModelResolutionErrorDomain = @"NFKModelResolutionErrorDomain";
NFKModelVerification const NFKModelVerificationMeasured = @"measured";
NFKModelVerification const NFKModelVerificationConfigCompatible = @"configCompatible";
NSErrorUserInfoKey const NFKModelResolutionRouteKey = @"NFKModelResolutionRoute";
NSErrorUserInfoKey const NFKModelResolutionDescriptorKey = @"NFKModelResolutionDescriptor";
NSString * const NFKModelResolutionExtensionKey = @"resolution";

static NSString * const NFKModelResolutionConfigPath = @"config.json";

NSString *NFKModelRouteName(NFKModelRoute route)
{
	switch (route) {
		case NFKModelRouteCatalog:
			return @"catalog";
		case NFKModelRouteFamily:
			return @"family";
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
							 memoryEstimate:estimate downloadByteCount:downloadBytes reasons:reasons];
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
	NSError *refusal = nil;
	NFKModelRepository *read = [self repository:repository revision:commit requestedRevision:revision plan:plan
									information:information refusal:&refusal error:error];
	if (read == nil) {
		return refusal != nil ? [self refusedResolutionOfRepository:repository revision:commit reasons:@[ refusal ]] : nil;
	}
	return [self familyResolutionOfRepository:read error:error];
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
	id gated = information[@"gated"];
	BOOL isGated = gated != nil && ![gated isEqual:@NO] && ![gated isEqual:@"false"];
	return [[NFKModelLicense alloc] initWithIdentifier:identifier name:name URL:URL.absoluteURL gated:isGated
								restrictsCommercialUse:NO note:nil extensions:nil];
}

/*! The repository as a loader judges it, or nil with a refusal (a route-independent reason) or an error
	(a failed read). */
- (nullable NFKModelRepository *)repository:(NSString *)repository
								   revision:(NSString *)commit
						  requestedRevision:(nullable NSString *)requestedRevision
									   plan:(NFKDownloadPlan *)plan
								information:(NSDictionary *)information
									refusal:(NSError **)refusal
									  error:(NSError **)error
{
	NFKModelLicense *license = [self licenseOfRepository:repository information:information];
	NFKModelRepository *listed = [[NFKModelRepository alloc] initWithRepository:repository revision:commit
															  requestedRevision:requestedRevision configuration:@{}
																		  files:plan.files license:license hub:self.hub];
	if ([listed fileAtPath:NFKModelResolutionConfigPath] == nil) {
		NSString *reason = [NSString stringWithFormat:@"%@ has no config.json", repository];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorNoConfiguration, NFKModelRouteNone, nil, reason, nil));
		return nil;
	}
	NSError *readError = nil;
	id configuration = [listed JSONObjectOfFileAtPath:NFKModelResolutionConfigPath error:&readError];
	if (NFKModelResolutionIsAuthenticationError(readError)) {
		NFKModelResolutionSet(refusal, [self gatedRefusalOfRepository:repository error:readError]);
		return nil;
	}
	if (configuration == nil && ![readError.domain isEqualToString:NSCocoaErrorDomain]) {
		if (error != NULL) {
			*error = readError;
		}
		return nil;
	}
	if (![configuration isKindOfClass:NSDictionary.class]) {
		NSString *reason = [NSString stringWithFormat:@"the config.json of %@ is not a JSON object", repository];
		NFKModelResolutionSet(refusal, NFKModelResolutionRefusal(NFKModelResolutionErrorNoConfiguration, NFKModelRouteNone, nil, reason, readError));
		return nil;
	}
	NFKModelRepository *read = [[NFKModelRepository alloc] initWithRepository:repository revision:commit
															requestedRevision:requestedRevision
																configuration:configuration files:plan.files
																	  license:license hub:self.hub];
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
		return NFKModelResolutionRefusal(NFKModelResolutionErrorCustomCode, NFKModelRouteNone, nil, reason, nil);
	}
	if ([repository pathsWithSuffix:@".safetensors"].count == 0) {
		NSArray<NSString *> *others = [[[repository pathsWithSuffix:@".bin"] arrayByAddingObjectsFromArray:
										[repository pathsWithSuffix:@".pt"]]
									   arrayByAddingObjectsFromArray:[repository pathsWithSuffix:@".gguf"]];
		NSString *reason = others.count > 0
			? [NSString stringWithFormat:@"%@ holds no safetensors weights, only %@", repository.repository,
			   [others componentsJoinedByString:@", "]]
			: [NSString stringWithFormat:@"%@ holds no weights", repository.repository];
		return NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedWeightFormat, NFKModelRouteNone, nil, reason, nil);
	}
	if (repository.modelType == nil) {
		NSString *reason = [NSString stringWithFormat:@"the config.json of %@ states no model_type", repository.repository];
		return NFKModelResolutionRefusal(NFKModelResolutionErrorUnsupportedArchitecture, NFKModelRouteNone, nil, reason, nil);
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

#pragma mark Results

- (NFKModelResolution *)resolutionOfRepository:(NSString *)repository
									  revision:(nullable NSString *)revision
										 route:(NFKModelRoute)route
								  verification:(NFKModelVerification)verification
									descriptor:(NFKModelDescriptor *)descriptor
									   variant:(nullable NFKModelVariant *)variant
									   reasons:(nullable NSArray<NSError *> *)reasons
{
	NSDictionary<NSString *, id> *options = descriptor.variants.count > 1 && variant != nil
		? @{ NFKModelLoadOptionVariant: variant.identifier } : nil;
	NFKModelMemoryBudget *budget = self.memoryBudget ?: self.catalog.memoryBudget;
	NFKModelMemoryEstimate *estimate = [descriptor estimatedMemoryForOptions:options memoryBudget:budget error:NULL];
	return [[NFKModelResolution alloc] initWithRepository:repository revision:revision route:route
											 verification:verification descriptor:descriptor
										variantIdentifier:variant.identifier memoryEstimate:estimate
										downloadByteCount:variant.downloadByteCount reasons:reasons];
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
