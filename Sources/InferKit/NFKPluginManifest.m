//
//  NFKPluginManifest.m
//  InferKit
//

#import "NFKPluginManifest.h"
#import "NFKPluginManifestPrivate.h"
#import "NFKPluginWire.h"
#import "NFKInferKit.h"
#import "NFKModelVocabulary.h"
#import <limits.h>
#import <stdlib.h>
#import <sys/stat.h>

#if __has_include(<InferKitC/NFKPluginABI.h>)
#import <InferKitC/NFKPluginABI.h>
#elif __has_include(<InferKit/NFKPluginABI.h>)
#import <InferKit/NFKPluginABI.h>
#else
#import "../InferKitC/include/InferKitC/NFKPluginABI.h"
#endif

NFKPluginEntryKind const NFKPluginEntryKindObjC = @"objc";
NFKPluginEntryKind const NFKPluginEntryKindCABI = @"c-abi";
NFKPluginEntryKind const NFKPluginEntryKindExecutable = @"executable";

NSString * const NFKPluginManifestRelativePath = @"Resources/inferkit-plugin.json";

static NSString * const NFKPluginManifestKind = @"inferkit.plugin";
static const NSTimeInterval NFKPluginManifestDefaultIdleExitInterval = 300;

static BOOL NFKPluginManifestFail(NSError **error, NFKPluginError code, NSString *reason)
{
	return NFKPluginSetError(error, code, reason);
}

static BOOL NFKPluginManifestIsBoolean(id value)
{
	return [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

static BOOL NFKPluginManifestIsInteger(id value)
{
	if (![value isKindOfClass:NSNumber.class] || NFKPluginManifestIsBoolean(value)) {
		return NO;
	}
	double number = [value doubleValue];
	return isfinite(number) && number == floor(number);
}

/*! Whether string matches pattern in full. */
static BOOL NFKPluginManifestMatches(NSString *string, NSString *pattern)
{
	NSRange range = [string rangeOfString:pattern options:NSRegularExpressionSearch];
	return range.location == 0 && range.length == string.length;
}

/*! Reads "major.minor" (a third component is allowed and ignored). Returns NO for anything else. */
static BOOL NFKPluginManifestParseVersion(NSString * _Nullable version, NSInteger *major, NSInteger *minor)
{
	if (version == nil || !NFKPluginManifestMatches(version, @"[0-9]{1,6}\\.[0-9]{1,6}(\\.[0-9]{1,6})?")) {
		return NO;
	}
	NSArray<NSString *> *components = [version componentsSeparatedByString:@"."];
	*major = components[0].integerValue;
	*minor = components[1].integerValue;
	return YES;
}

/*! A relative path inside Contents/MacOS: no leading slash, no "." or ".." component. */
static BOOL NFKPluginManifestIsContainedPath(NSString *path)
{
	if (path.length == 0 || path.isAbsolutePath) {
		return NO;
	}
	for (NSString *component in [path componentsSeparatedByString:@"/"]) {
		if (component.length == 0 || [component isEqualToString:@"."] || [component isEqualToString:@".."]) {
			return NO;
		}
	}
	return YES;
}

/*! The path of an existing file with every symbolic link resolved, or nil. */
static NSString * _Nullable NFKPluginManifestResolvedPath(NSURL *url)
{
	char resolved[PATH_MAX];
	return realpath(url.fileSystemRepresentation, resolved) != NULL ? [NSString stringWithUTF8String:resolved] : nil;
}

/*! Reads the manifest's fields from its JSON object, reporting the first problem by its path. */
@interface NFKPluginManifestReader : NSObject
@property (nonatomic, copy) NSDictionary<NSString *, id> *object;
@property (nonatomic, copy) NSString *path;
@end

@implementation NFKPluginManifestReader

+ (instancetype)readerWithObject:(NSDictionary<NSString *, id> *)object path:(NSString *)path
{
	NFKPluginManifestReader *reader = [[self alloc] init];
	reader.object = object;
	reader.path = path;
	return reader;
}

- (NSString *)nameOfField:(NSString *)field
{
	return self.path.length > 0 ? [NSString stringWithFormat:@"%@.%@", self.path, field] : field;
}

- (BOOL)failField:(NSString *)field reason:(NSString *)reason error:(NSError **)error
{
	return NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								 [NSString stringWithFormat:@"the manifest's \"%@\" %@", [self nameOfField:field], reason]);
}

- (BOOL)readString:(NSString *)field required:(BOOL)required into:(NSString * _Nullable * _Nonnull)value error:(NSError **)error
{
	id raw = self.object[field];
	*value = nil;
	if (raw == nil || raw == NSNull.null) {
		return required ? [self failField:field reason:@"is missing" error:error] : YES;
	}
	if (![raw isKindOfClass:NSString.class] || [raw length] == 0) {
		return [self failField:field reason:@"is not a nonempty string" error:error];
	}
	*value = raw;
	return YES;
}

- (BOOL)readObject:(NSString *)field required:(BOOL)required into:(NSDictionary * _Nullable * _Nonnull)value error:(NSError **)error
{
	id raw = self.object[field];
	*value = nil;
	if (raw == nil || raw == NSNull.null) {
		return required ? [self failField:field reason:@"is missing" error:error] : YES;
	}
	if (![raw isKindOfClass:NSDictionary.class]) {
		return [self failField:field reason:@"is not an object" error:error];
	}
	*value = raw;
	return YES;
}

- (BOOL)readArray:(NSString *)field of:(Class)elementClass into:(NSArray * _Nonnull * _Nonnull)value error:(NSError **)error
{
	id raw = self.object[field];
	*value = @[];
	if (raw == nil || raw == NSNull.null) {
		return YES;
	}
	if (![raw isKindOfClass:NSArray.class]) {
		return [self failField:field reason:@"is not an array" error:error];
	}
	for (id element in raw) {
		if (![element isKindOfClass:elementClass]) {
			return [self failField:field reason:@"holds an element of the wrong type" error:error];
		}
	}
	*value = raw;
	return YES;
}

- (BOOL)readVersion:(NSString *)field required:(BOOL)required into:(NSString * _Nullable * _Nonnull)value error:(NSError **)error
{
	if (![self readString:field required:required into:value error:error]) {
		return NO;
	}
	NSInteger major = 0;
	NSInteger minor = 0;
	if (*value != nil && !NFKPluginManifestParseVersion(*value, &major, &minor)) {
		return [self failField:field reason:@"is not a \"major.minor\" version" error:error];
	}
	return YES;
}

@end

@interface NFKPluginManifest ()
@property (nonatomic, readwrite, copy) NSString *identifier;
@property (nonatomic, readwrite, copy) NSString *version;
@property (nonatomic, readwrite, copy) NSString *displayName;
@property (nonatomic, readwrite, copy, nullable) NSString *summary;
@property (nonatomic, readwrite, copy) NFKPluginEntryKind entryKind;
@property (nonatomic, readwrite, copy, nullable) NSString *executablePath;
@property (nonatomic, readwrite, copy, nullable) NSString *principalClassName;
@property (nonatomic, readwrite, copy, nullable) NSString *entrySymbol;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *arguments;
@property (nonatomic, readwrite, copy) NSString *minimumAPIVersion;
@property (nonatomic, readwrite, copy, nullable) NSString *pluginABIVersion;
@property (nonatomic, readwrite, copy) NSString *protocolVersion;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, NSString *> *minimumOSVersions;
@property (nonatomic, readwrite, copy) NSArray<NSString *> *architectures;
@property (nonatomic, readwrite, copy) NSArray<NFKModelDescriptor *> *models;
@property (nonatomic, readwrite, copy) NSArray<NSDictionary<NSString *, id> *> *tools;
@property (nonatomic, readwrite, copy) NSArray<NFKModelOption *> *options;
@property (nonatomic, readwrite) NSTimeInterval idleExitInterval;
@property (nonatomic, readwrite) BOOL allowsInProcess;
@property (nonatomic, readwrite) NSInteger maximumConcurrentRuns;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *extensions;
@property (nonatomic, readwrite, copy, nullable) NSURL *bundleURL;
@end

@implementation NFKPluginManifest

+ (NSInteger)schemaVersion
{
	return 1;
}

#pragma mark Reading

+ (nullable instancetype)manifestWithBundleURL:(NSURL *)bundleURL error:(NSError **)error
{
	NSURL *contents = [bundleURL URLByAppendingPathComponent:@"Contents" isDirectory:YES];
	NSData *data = [NSData dataWithContentsOfURL:[contents URLByAppendingPathComponent:NFKPluginManifestRelativePath] options:0 error:NULL];
	if (data == nil) {
		NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
							  [NSString stringWithFormat:@"%@ holds no Contents/%@", bundleURL.lastPathComponent, NFKPluginManifestRelativePath]);
		return nil;
	}
	NFKPluginManifest *manifest = [self manifestWithJSONData:data error:error];
	if (manifest == nil) {
		return nil;
	}
	NSDictionary *info = [NSDictionary dictionaryWithContentsOfURL:[contents URLByAppendingPathComponent:@"Info.plist"]];
	if (![manifest matchesInfoDictionary:info bundleURL:bundleURL error:error]) {
		return nil;
	}
	manifest.bundleURL = bundleURL;
	[manifest readBundleExecutableNameFromInfoDictionary:info];
	if (![manifest hasExecutableWithError:error]) {
		return nil;
	}
	return manifest;
}

+ (nullable instancetype)manifestWithJSONData:(NSData *)data error:(NSError **)error
{
	NSError *parseError = nil;
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&parseError];
	if (![object isKindOfClass:NSDictionary.class]) {
		NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
							  [NSString stringWithFormat:@"the manifest is not a JSON object: %@",
							   parseError.localizedDescription ?: @"another JSON value"]);
		return nil;
	}
	NFKPluginManifest *manifest = [[self alloc] initEmpty];
	return [manifest readObject:object error:error] ? manifest : nil;
}

- (instancetype)initEmpty
{
	return [super init];
}

- (BOOL)readObject:(NSDictionary<NSString *, id> *)object error:(NSError **)error
{
	id schema = object[@"schemaVersion"];
	if (!NFKPluginManifestIsInteger(schema) || [schema integerValue] < 1) {
		return NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest, @"the manifest's \"schemaVersion\" is not a positive integer");
	}
	if ([schema integerValue] > NFKPluginManifest.schemaVersion) {
		return NFKPluginManifestFail(error, NFKPluginErrorUnsupportedManifestVersion,
									 [NSString stringWithFormat:@"the manifest's schema version %@ is newer than %ld, the version this library reads",
									  schema, (long)NFKPluginManifest.schemaVersion]);
	}
	if (![object[@"kind"] isEqual:NFKPluginManifestKind]) {
		return NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
									 [NSString stringWithFormat:@"the manifest's \"kind\" is not \"%@\"", NFKPluginManifestKind]);
	}
	return [self readIdentityOf:object error:error]
		&& [self readCompatibilityOf:object error:error]
		&& [self readEntryOf:object error:error]
		&& [self readProcessOf:object error:error]
		&& [self readOfferingsOf:object error:error];
}

- (BOOL)readIdentityOf:(NSDictionary *)object error:(NSError **)error
{
	NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:object path:@""];
	NSString *identifier = nil;
	NSString *version = nil;
	NSString *displayName = nil;
	NSString *summary = nil;
	NSDictionary *extensions = nil;
	if (![reader readString:@"id" required:YES into:&identifier error:error]
		|| ![reader readString:@"version" required:YES into:&version error:error]
		|| ![reader readString:@"displayName" required:NO into:&displayName error:error]
		|| ![reader readString:@"summary" required:NO into:&summary error:error]
		|| ![reader readObject:@"extensions" required:NO into:&extensions error:error]) {
		return NO;
	}
	if (!NFKPluginManifestMatches(identifier, @"[A-Za-z0-9][A-Za-z0-9._-]*")) {
		return [reader failField:@"id" reason:@"is not a reverse-DNS name of letters, digits, \".\", \"-\", and \"_\"" error:error];
	}
	if (!NFKPluginManifestMatches(version, @"[0-9A-Za-z.+-]+")) {
		return [reader failField:@"version" reason:@"is not a semantic version" error:error];
	}
	self.identifier = identifier;
	self.version = version;
	self.displayName = displayName ?: identifier;
	self.summary = summary;
	self.extensions = extensions ?: @{};
	return YES;
}

- (BOOL)readCompatibilityOf:(NSDictionary *)object error:(NSError **)error
{
	NSDictionary *compatibility = nil;
	if (![[NFKPluginManifestReader readerWithObject:object path:@""] readObject:@"compatibility" required:YES into:&compatibility error:error]) {
		return NO;
	}
	NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:compatibility path:@"compatibility"];
	NSString *APIVersion = nil;
	NSString *ABIVersion = nil;
	NSString *protocolVersion = nil;
	NSDictionary *minimumOS = nil;
	NSArray *architectures = nil;
	if (![reader readVersion:@"inferkitAPI" required:YES into:&APIVersion error:error]
		|| ![reader readVersion:@"pluginABI" required:NO into:&ABIVersion error:error]
		|| ![reader readVersion:@"protocol" required:NO into:&protocolVersion error:error]
		|| ![reader readObject:@"minimumOS" required:NO into:&minimumOS error:error]
		|| ![reader readArray:@"architectures" of:NSString.class into:&architectures error:error]) {
		return NO;
	}
	for (id platform in minimumOS) {
		if (![minimumOS[platform] isKindOfClass:NSString.class]) {
			return [reader failField:@"minimumOS" reason:@"holds a version that is not a string" error:error];
		}
	}
	self.minimumAPIVersion = APIVersion;
	self.pluginABIVersion = ABIVersion;
	self.protocolVersion = protocolVersion ?: @"1.0";
	self.minimumOSVersions = minimumOS ?: @{};
	self.architectures = architectures;
	return YES;
}

- (BOOL)readEntryOf:(NSDictionary *)object error:(NSError **)error
{
	NSDictionary *entry = nil;
	if (![[NFKPluginManifestReader readerWithObject:object path:@""] readObject:@"entry" required:YES into:&entry error:error]) {
		return NO;
	}
	NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:entry path:@"entry"];
	NSString *kind = nil;
	NSString *executable = nil;
	NSString *symbol = nil;
	NSString *principalClass = nil;
	NSArray *arguments = nil;
	if (![reader readString:@"kind" required:YES into:&kind error:error]
		|| ![reader readString:@"executable" required:NO into:&executable error:error]
		|| ![reader readString:@"symbol" required:NO into:&symbol error:error]
		|| ![reader readString:@"principalClass" required:NO into:&principalClass error:error]
		|| ![reader readArray:@"arguments" of:NSString.class into:&arguments error:error]) {
		return NO;
	}
	NSArray<NSString *> *kinds = @[ NFKPluginEntryKindObjC, NFKPluginEntryKindCABI, NFKPluginEntryKindExecutable ];
	if (![kinds containsObject:kind]) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the plugin's entry kind \"%@\" is not one this library runs", kind]);
	}
	if (executable != nil && !NFKPluginManifestIsContainedPath(executable)) {
		return [reader failField:@"executable" reason:@"is not a relative path inside Contents/MacOS" error:error];
	}
	if ([kind isEqualToString:NFKPluginEntryKindCABI] && self.pluginABIVersion == nil) {
		return NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest, @"a \"c-abi\" plugin's manifest names no \"compatibility.pluginABI\"");
	}
	self.entryKind = kind;
	self.executablePath = executable;
	self.entrySymbol = symbol;
	self.principalClassName = principalClass;
	self.arguments = arguments;
	return YES;
}

- (BOOL)readProcessOf:(NSDictionary *)object error:(NSError **)error
{
	NSDictionary *process = nil;
	if (![[NFKPluginManifestReader readerWithObject:object path:@""] readObject:@"process" required:NO into:&process error:error]) {
		return NO;
	}
	NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:process ?: @{} path:@"process"];
	id idle = process[@"idleExitSeconds"];
	id inProcess = process[@"allowsInProcess"];
	id concurrency = process[@"maximumConcurrentRuns"];
	if (idle != nil && (![idle isKindOfClass:NSNumber.class] || NFKPluginManifestIsBoolean(idle) || !isfinite([idle doubleValue]) || [idle doubleValue] < 0)) {
		return [reader failField:@"idleExitSeconds" reason:@"is not a number of seconds" error:error];
	}
	if (inProcess != nil && !NFKPluginManifestIsBoolean(inProcess)) {
		return [reader failField:@"allowsInProcess" reason:@"is not a boolean" error:error];
	}
	if (concurrency != nil && (!NFKPluginManifestIsInteger(concurrency) || [concurrency integerValue] < 1)) {
		return [reader failField:@"maximumConcurrentRuns" reason:@"is not a positive integer" error:error];
	}
	self.idleExitInterval = idle != nil ? [idle doubleValue] : NFKPluginManifestDefaultIdleExitInterval;
	self.allowsInProcess = [inProcess boolValue];
	self.maximumConcurrentRuns = concurrency != nil ? [concurrency integerValue] : 1;
	return YES;
}

- (BOOL)readOfferingsOf:(NSDictionary *)object error:(NSError **)error
{
	NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:object path:@""];
	NSArray<NSDictionary *> *options = nil;
	NSArray<NSDictionary *> *models = nil;
	NSArray<NSDictionary *> *tools = nil;
	if (![reader readArray:@"options" of:NSDictionary.class into:&options error:error]
		|| ![reader readArray:@"models" of:NSDictionary.class into:&models error:error]
		|| ![reader readArray:@"tools" of:NSDictionary.class into:&tools error:error]) {
		return NO;
	}
	NSArray<NFKModelOption *> *readOptions = [self optionsFromDictionaries:options error:error];
	NSArray<NFKModelDescriptor *> *readModels = readOptions != nil ? [self modelsFromDictionaries:models error:error] : nil;
	NSArray<NSDictionary *> *readTools = readModels != nil ? [self toolsFromDictionaries:tools error:error] : nil;
	if (readTools == nil) {
		return NO;
	}
	self.options = readOptions;
	self.models = readModels;
	self.tools = readTools;
	return YES;
}

- (nullable NSArray<NFKModelOption *> *)optionsFromDictionaries:(NSArray<NSDictionary *> *)dictionaries error:(NSError **)error
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray arrayWithCapacity:dictionaries.count];
	NSMutableSet<NSString *> *keys = [NSMutableSet set];
	for (NSDictionary *dictionary in dictionaries) {
		NSError *optionError = nil;
		NFKModelOption *option = [NFKModelOption optionWithDictionary:dictionary error:&optionError];
		if (option == nil) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the manifest's \"options\" holds a malformed option: %@", optionError.localizedDescription]);
			return nil;
		}
		if ([keys containsObject:option.key]) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the manifest declares the option \"%@\" twice", option.key]);
			return nil;
		}
		[keys addObject:option.key];
		[options addObject:option];
	}
	return options;
}

- (nullable NSArray<NFKModelDescriptor *> *)modelsFromDictionaries:(NSArray<NSDictionary *> *)dictionaries error:(NSError **)error
{
	NSMutableArray<NFKModelDescriptor *> *models = [NSMutableArray arrayWithCapacity:dictionaries.count];
	NSMutableSet<NSString *> *identifiers = [NSMutableSet set];
	for (NSDictionary *dictionary in dictionaries) {
		if (dictionary[@"engine"] != nil && ![dictionary[@"engine"] isEqual:NFKModelEnginePlugin]) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the manifest's model \"%@\" names the engine \"%@\"; a plugin's models run on \"%@\"",
								   dictionary[@"id"], dictionary[@"engine"], NFKModelEnginePlugin]);
			return nil;
		}
		NSMutableDictionary *withEngine = [dictionary mutableCopy];
		withEngine[@"engine"] = NFKModelEnginePlugin;
		NSError *modelError = nil;
		NFKModelDescriptor *model = [NFKModelDescriptor descriptorWithDictionary:withEngine error:&modelError];
		if (model == nil) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the manifest's \"models\" holds a malformed model: %@", modelError.localizedDescription]);
			return nil;
		}
		if ([model.identifier containsString:@"/"]) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the model identifier \"%@\" holds a \"/\", which joins the plugin and model identifiers in the catalog",
								   model.identifier]);
			return nil;
		}
		if ([identifiers containsObject:model.identifier]) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest,
								  [NSString stringWithFormat:@"the manifest lists the model \"%@\" twice", model.identifier]);
			return nil;
		}
		[identifiers addObject:model.identifier];
		[models addObject:model];
	}
	return models;
}

- (nullable NSArray<NSDictionary *> *)toolsFromDictionaries:(NSArray<NSDictionary *> *)dictionaries error:(NSError **)error
{
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray arrayWithCapacity:dictionaries.count];
	NSMutableSet<NSString *> *names = [NSMutableSet set];
	for (NSDictionary *dictionary in dictionaries) {
		NFKPluginManifestReader *reader = [NFKPluginManifestReader readerWithObject:dictionary path:@"tools"];
		NSString *name = nil;
		NSString *description = nil;
		NSDictionary *parameters = nil;
		if (![reader readString:@"name" required:YES into:&name error:error]
			|| ![reader readString:@"description" required:NO into:&description error:error]
			|| ![reader readObject:@"parameters" required:NO into:&parameters error:error]) {
			return nil;
		}
		if (name == nil || !NFKPluginManifestMatches(name, @"[A-Za-z0-9_-]{1,64}")) {
			[reader failField:@"name" reason:[NSString stringWithFormat:@"\"%@\" is not 1 to 64 letters, digits, \"_\", and \"-\"", name] error:error];
			return nil;
		}
		if ([names containsObject:name]) {
			NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest, [NSString stringWithFormat:@"the manifest lists the tool \"%@\" twice", name]);
			return nil;
		}
		[names addObject:name];
		[tools addObject:@{ @"name": name,
							@"description": description ?: @"",
							@"parameters": parameters ?: @{ @"type": @"object", @"properties": @{} } }];
	}
	return tools;
}

#pragma mark Bundle

- (BOOL)matchesInfoDictionary:(nullable NSDictionary *)info bundleURL:(NSURL *)bundleURL error:(NSError **)error
{
	if (info == nil) {
		return NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
									 [NSString stringWithFormat:@"%@ holds no readable Contents/Info.plist", bundleURL.lastPathComponent]);
	}
	if (![info[@"CFBundleIdentifier"] isEqual:self.identifier]) {
		return NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
									 [NSString stringWithFormat:@"the bundle identifier \"%@\" differs from the manifest's \"%@\"",
									  info[@"CFBundleIdentifier"], self.identifier]);
	}
	if (![info[@"CFBundleShortVersionString"] isEqual:self.version]) {
		return NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
									 [NSString stringWithFormat:@"the bundle's short version \"%@\" differs from the manifest's \"%@\"",
									  info[@"CFBundleShortVersionString"], self.version]);
	}
	return YES;
}

- (void)readBundleExecutableNameFromInfoDictionary:(NSDictionary *)info
{
	_bundleExecutableName = [info[@"CFBundleExecutable"] isKindOfClass:NSString.class] ? [info[@"CFBundleExecutable"] copy] : nil;
}

- (BOOL)hasExecutableWithError:(NSError **)error
{
	return [self containedExecutableURLWithError:error] != nil;
}

- (nullable NSURL *)executableURL
{
	return [self containedExecutableURLWithError:NULL];
}

- (nullable NSURL *)containedExecutableURLWithError:(NSError **)error
{
	NSString *name = self.executablePath ?: self.bundleExecutableName;
	if (self.bundleURL == nil || name == nil) {
		NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
							  [NSString stringWithFormat:@"the plugin's bundle names no executable: %@",
							   self.bundleURL == nil ? @"the manifest was read without its bundle" : @"no entry.executable and no CFBundleExecutable"]);
		return nil;
	}
	if (!NFKPluginManifestIsContainedPath(name)) {
		NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
							  [NSString stringWithFormat:@"the plugin's executable \"%@\" is not a relative path inside Contents/MacOS", name]);
		return nil;
	}
	NSURL *candidate = [[self.bundleURL URLByAppendingPathComponent:@"Contents/MacOS" isDirectory:YES] URLByAppendingPathComponent:name isDirectory:NO];
	NSString *bundle = NFKPluginManifestResolvedPath(self.bundleURL);
	NSString *resolved = NFKPluginManifestResolvedPath(candidate);
	struct stat status;
	if (bundle == nil || resolved == nil || stat(resolved.fileSystemRepresentation, &status) != 0 || !S_ISREG(status.st_mode)) {
		NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
							  [NSString stringWithFormat:@"the plugin's executable \"%@\" is not a file in Contents/MacOS", name]);
		return nil;
	}
	NSString *prefix = [bundle stringByAppendingString:@"/Contents/MacOS/"];
	if (![resolved hasPrefix:prefix]) {
		NFKPluginManifestFail(error, NFKPluginErrorManifestMismatch,
							  [NSString stringWithFormat:@"the plugin's executable \"%@\" resolves to %@, outside the bundle's Contents/MacOS",
							   name, resolved]);
		return nil;
	}
	return [NSURL fileURLWithPath:resolved isDirectory:NO];
}

- (nullable NFKModelDescriptor *)modelWithIdentifier:(NSString *)identifier
{
	for (NFKModelDescriptor *model in self.models) {
		if ([model.identifier isEqualToString:identifier]) {
			return model;
		}
	}
	return nil;
}

#pragma mark Writing

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *compatibility = [NSMutableDictionary dictionary];
	compatibility[@"inferkitAPI"] = self.minimumAPIVersion;
	compatibility[@"pluginABI"] = self.pluginABIVersion;
	compatibility[@"protocol"] = self.protocolVersion;
	compatibility[@"minimumOS"] = self.minimumOSVersions.count > 0 ? self.minimumOSVersions : nil;
	compatibility[@"architectures"] = self.architectures.count > 0 ? self.architectures : nil;

	NSMutableDictionary<NSString *, id> *entry = [NSMutableDictionary dictionary];
	entry[@"kind"] = self.entryKind;
	entry[@"executable"] = self.executablePath;
	entry[@"symbol"] = self.entrySymbol;
	entry[@"principalClass"] = self.principalClassName;
	entry[@"arguments"] = self.arguments.count > 0 ? self.arguments : nil;

	NSMutableArray *options = [NSMutableArray arrayWithCapacity:self.options.count];
	for (NFKModelOption *option in self.options) {
		[options addObject:option.dictionaryRepresentation];
	}
	NSMutableArray *models = [NSMutableArray arrayWithCapacity:self.models.count];
	for (NFKModelDescriptor *model in self.models) {
		[models addObject:model.dictionaryRepresentation];
	}

	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"schemaVersion"] = @(NFKPluginManifest.schemaVersion);
	dictionary[@"kind"] = NFKPluginManifestKind;
	dictionary[@"id"] = self.identifier;
	dictionary[@"version"] = self.version;
	dictionary[@"displayName"] = self.displayName;
	dictionary[@"summary"] = self.summary;
	dictionary[@"compatibility"] = compatibility;
	dictionary[@"entry"] = entry;
	dictionary[@"process"] = @{ @"idleExitSeconds": @(self.idleExitInterval),
								@"allowsInProcess": @(self.allowsInProcess),
								@"maximumConcurrentRuns": @(self.maximumConcurrentRuns) };
	dictionary[@"options"] = options.count > 0 ? options : nil;
	dictionary[@"models"] = models.count > 0 ? models : nil;
	dictionary[@"tools"] = self.tools.count > 0 ? self.tools : nil;
	dictionary[@"extensions"] = self.extensions.count > 0 ? self.extensions : nil;
	return dictionary;
}

- (nullable NSData *)JSONDataWithError:(NSError **)error
{
	NSDictionary *dictionary = self.dictionaryRepresentation;
	if (![NSJSONSerialization isValidJSONObject:dictionary]) {
		NFKPluginManifestFail(error, NFKPluginErrorMalformedManifest, @"the manifest holds a value JSON cannot carry");
		return nil;
	}
	NSJSONWritingOptions options = NSJSONWritingSortedKeys | NSJSONWritingPrettyPrinted | NSJSONWritingWithoutEscapingSlashes;
	return [NSJSONSerialization dataWithJSONObject:dictionary options:options error:error];
}

#pragma mark Compatibility

- (BOOL)checkCompatibilityWithAPIVersion:(NSString *)APIVersion error:(NSError **)error
{
	NSInteger hostMajor = 0;
	NSInteger hostMinor = 0;
	NSInteger neededMajor = 0;
	NSInteger neededMinor = 0;
	if (!NFKPluginManifestParseVersion(APIVersion, &hostMajor, &hostMinor)
		|| !NFKPluginManifestParseVersion(self.minimumAPIVersion, &neededMajor, &neededMinor)) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the API version \"%@\" is not \"major.minor\"", APIVersion]);
	}
	if (hostMajor != neededMajor || hostMinor < neededMinor) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the plugin %@ needs InferKit API %@, and this library is API %@",
									  self.identifier, self.minimumAPIVersion, APIVersion]);
	}
	return YES;
}

#if TARGET_OS_OSX

- (BOOL)checkHostCompatibilityWithError:(NSError **)error
{
	if (![self checkCompatibilityWithAPIVersion:NFKInferKit.APIVersion error:error]) {
		return NO;
	}
	NSInteger major = 0;
	NSInteger minor = 0;
	NFKPluginManifestParseVersion(self.protocolVersion, &major, &minor);
	if (major != NFKPluginProtocolMajorVersion) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the plugin %@ speaks plugin protocol %@, and this library speaks %ld.%ld",
									  self.identifier, self.protocolVersion, (long)NFKPluginProtocolMajorVersion, (long)NFKPluginProtocolMinorVersion]);
	}
	if ([self.entryKind isEqualToString:NFKPluginEntryKindCABI]
		&& (!NFKPluginManifestParseVersion(self.pluginABIVersion, &major, &minor) || major != NFK_PLUGIN_ABI_VERSION_MAJOR)) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the plugin %@ was built against plugin ABI %@, and this library implements %d.%d",
									  self.identifier, self.pluginABIVersion, NFK_PLUGIN_ABI_VERSION_MAJOR, NFK_PLUGIN_ABI_VERSION_MINOR]);
	}
	return [self checkSystemWithError:error];
}

- (BOOL)checkSystemWithError:(NSError **)error
{
	NSString *minimumSystem = self.minimumOSVersions[@"macOS"];
	if (minimumSystem != nil) {
		NSArray<NSString *> *components = [minimumSystem componentsSeparatedByString:@"."];
		NSOperatingSystemVersion needed = { components[0].integerValue,
											components.count > 1 ? components[1].integerValue : 0,
											components.count > 2 ? components[2].integerValue : 0 };
		if (![NSProcessInfo.processInfo isOperatingSystemAtLeastVersion:needed]) {
			return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
										 [NSString stringWithFormat:@"the plugin %@ needs macOS %@", self.identifier, minimumSystem]);
		}
	}
#if defined(__arm64__)
	NSString *architecture = @"arm64";
#elif defined(__x86_64__)
	NSString *architecture = @"x86_64";
#else
	NSString *architecture = @"unknown";
#endif
	if (self.architectures.count > 0 && ![self.architectures containsObject:architecture]) {
		return NFKPluginManifestFail(error, NFKPluginErrorIncompatible,
									 [NSString stringWithFormat:@"the plugin %@ carries %@ code, and this process runs %@",
									  self.identifier, [self.architectures componentsJoinedByString:@" and "], architecture]);
	}
	return YES;
}

#endif

#pragma mark NSObject

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (object == self) {
		return YES;
	}
	if (![object isKindOfClass:NFKPluginManifest.class]) {
		return NO;
	}
	NFKPluginManifest *other = object;
	BOOL sameBundle = (self.bundleURL == nil && other.bundleURL == nil) || [self.bundleURL isEqual:other.bundleURL];
	return sameBundle && [self.dictionaryRepresentation isEqualToDictionary:other.dictionaryRepresentation];
}

- (NSUInteger)hash
{
	return self.identifier.hash ^ self.version.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ (%@)>", self.class, self.identifier, self.version, self.entryKind];
}

@end
