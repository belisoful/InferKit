//
//  NFKPluginSignaturePolicy.m
//  InferKit
//

#import "NFKPluginSignaturePolicy.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import <Security/Security.h>
#import <CommonCrypto/CommonDigest.h>
#import <sys/stat.h>

/*! The static checks every validation runs: every architecture, strict validation, and nested code. */
static const SecCSFlags NFKPluginStaticValidationFlags = kSecCSCheckAllArchitectures | kSecCSStrictValidate | kSecCSCheckNestedCode;

/*! The code's own words for why a check failed, with its OSStatus. */
static NSString *NFKPluginSecurityReason(OSStatus status, CFErrorRef _Nullable error)
{
	NSString *described = error != NULL ? CFBridgingRelease(CFErrorCopyDescription(error)) : nil;
	if (described == nil) {
		described = CFBridgingRelease(SecCopyErrorMessageString(status, NULL));
	}
	return [NSString stringWithFormat:@"%@ (OSStatus %d)", described ?: @"the check failed", (int)status];
}

static NSString *NFKPluginHexString(NSData *data)
{
	NSMutableString *hex = [NSMutableString stringWithCapacity:data.length * 2];
	const uint8_t *bytes = data.bytes;
	for (NSUInteger index = 0; index < data.length; index++) {
		[hex appendFormat:@"%02x", bytes[index]];
	}
	return hex;
}

/*! Checks one piece of code against a requirement, or against its signature alone when requirement is
	NULL. */
typedef OSStatus (^NFKPluginCodeCheck)(SecRequirementRef _Nullable requirement, CFErrorRef _Nullable * _Nonnull error);

@implementation NFKPluginSignaturePolicy

+ (instancetype)defaultPolicy
{
	return [[self alloc] init];
}

+ (instancetype)developmentPolicy
{
	NFKPluginSignaturePolicy *policy = [[self alloc] init];
	policy.allowsAdHocSignature = YES;
	return policy;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requiresValidSignature = YES;
	}
	return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	NFKPluginSignaturePolicy *copy = [[self.class alloc] init];
	copy.requiresValidSignature = self.requiresValidSignature;
	copy.teamIdentifier = self.teamIdentifier;
	copy.allowsAdHocSignature = self.allowsAdHocSignature;
	copy.requirementString = self.requirementString;
	return copy;
}

#pragma mark Cache

/*! The successful static validations of this process, each keyed by cdhash, policy, and the bundle's
	file fingerprint. */
+ (NSMutableSet<NSString *> *)validatedCodes
{
	static NSMutableSet<NSString *> *codes = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		codes = [NSMutableSet set];
	});
	return codes;
}

/*! What the policy demands, as one string, so a cached success applies only to the same demands. */
- (NSString *)policyKey
{
	return [NSString stringWithFormat:@"%d|%d|%@|%@", self.requiresValidSignature, self.allowsAdHocSignature,
			self.teamIdentifier ?: @"", self.requirementString ?: @""];
}

/*! A digest of every file's path, size, inode, mode, and status-change time under url. A change to any file
	changes the status-change time, which a process cannot set back, so an edit after a validation
	misses the cache and is validated again. */
+ (nullable NSString *)fingerprintOfCodeAtURL:(NSURL *)url
{
	CC_SHA256_CTX context;
	CC_SHA256_Init(&context);
	NSMutableArray<NSString *> *paths = [NSMutableArray arrayWithObject:url.path];
	NSDirectoryEnumerator<NSString *> *enumerator = [NSFileManager.defaultManager enumeratorAtPath:url.path];
	for (NSString *relative in enumerator) {
		[paths addObject:[url.path stringByAppendingPathComponent:relative]];
	}
	[paths sortUsingSelector:@selector(compare:)];
	for (NSString *path in paths) {
		struct stat status;
		if (lstat(path.fileSystemRepresentation, &status) != 0) {
			return nil;
		}
		const char *name = path.fileSystemRepresentation;
		CC_SHA256_Update(&context, name, (CC_LONG)strlen(name) + 1);
		int64_t fields[] = { (int64_t)status.st_size, (int64_t)status.st_ino, (int64_t)status.st_mode,
							 (int64_t)status.st_ctimespec.tv_sec, (int64_t)status.st_ctimespec.tv_nsec,
							 (int64_t)status.st_mtimespec.tv_sec, (int64_t)status.st_mtimespec.tv_nsec };
		CC_SHA256_Update(&context, fields, (CC_LONG)sizeof fields);
	}
	uint8_t digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256_Final(digest, &context);
	return NFKPluginHexString([NSData dataWithBytes:digest length:sizeof digest]);
}

#pragma mark Static code

- (BOOL)validateCodeAtURL:(NSURL *)url error:(NSError **)error
{
	SecStaticCodeRef code = NULL;
	OSStatus status = SecStaticCodeCreateWithPath((__bridge CFURLRef)url, kSecCSDefaultFlags, &code);
	if (status != errSecSuccess || code == NULL) {
		return NFKPluginSetError(error, NFKPluginErrorSignatureInvalid,
								 [NSString stringWithFormat:@"%@ could not be read as code: %@", url.lastPathComponent, NFKPluginSecurityReason(status, NULL)]);
	}
	CFDictionaryRef rawInformation = NULL;
	SecCodeCopySigningInformation(code, kSecCSSigningInformation, &rawInformation);
	NSDictionary *information = CFBridgingRelease(rawInformation);
	NSData *cdhash = [information[(__bridge NSString *)kSecCodeInfoUnique] isKindOfClass:NSData.class] ? information[(__bridge NSString *)kSecCodeInfoUnique] : nil;
	NSString *fingerprint = [self.class fingerprintOfCodeAtURL:url];
	NSString *cacheKey = cdhash != nil && fingerprint != nil
		? [NSString stringWithFormat:@"%@|%@|%@|%@", NFKPluginHexString(cdhash), url.URLByStandardizingPath.path, fingerprint, self.policyKey] : nil;
	NSMutableSet<NSString *> *validated = self.class.validatedCodes;
	if (cacheKey != nil) {
		@synchronized (validated) {
			if ([validated containsObject:cacheKey]) {
				CFRelease(code);
				return YES;
			}
		}
	}
	BOOL valid = [self validateCodeNamed:url.lastPathComponent information:information error:error
								   check:^OSStatus(SecRequirementRef requirement, CFErrorRef *checkError) {
		return SecStaticCodeCheckValidityWithErrors(code, NFKPluginStaticValidationFlags, requirement, checkError);
	}];
	CFRelease(code);
	if (valid && cacheKey != nil) {
		@synchronized (validated) {
			[validated addObject:cacheKey];
		}
	}
	return valid;
}

#pragma mark Running code

- (BOOL)validateProcessWithAuditToken:(audit_token_t)auditToken error:(NSError **)error
{
	NSData *token = [NSData dataWithBytes:&auditToken length:sizeof auditToken];
	NSDictionary *attributes = @{ (__bridge NSString *)kSecGuestAttributeAudit: token };
	SecCodeRef code = NULL;
	OSStatus status = SecCodeCopyGuestWithAttributes(NULL, (__bridge CFDictionaryRef)attributes, kSecCSDefaultFlags, &code);
	if (status != errSecSuccess || code == NULL) {
		return NFKPluginSetError(error, NFKPluginErrorSignatureInvalid,
								 [NSString stringWithFormat:@"the process's code could not be found: %@", NFKPluginSecurityReason(status, NULL)]);
	}
	CFDictionaryRef rawInformation = NULL;
	SecCodeCopySigningInformation((SecStaticCodeRef)code, kSecCSSigningInformation | kSecCSDynamicInformation, &rawInformation);
	NSDictionary *information = CFBridgingRelease(rawInformation);
	NSString *name = [information[(__bridge NSString *)kSecCodeInfoIdentifier] isKindOfClass:NSString.class]
		? information[(__bridge NSString *)kSecCodeInfoIdentifier] : @"the process";
	BOOL valid = [self validateCodeNamed:name information:information error:error
								   check:^OSStatus(SecRequirementRef requirement, CFErrorRef *checkError) {
		return SecCodeCheckValidityWithErrors(code, kSecCSDefaultFlags, requirement, checkError);
	}];
	CFRelease(code);
	return valid;
}

#pragma mark Policy

/*! Applies the policy in order: a signature at all, its validity, ad hoc, the team, then the
	requirement. */
- (BOOL)validateCodeNamed:(NSString *)name information:(NSDictionary *)information error:(NSError **)error check:(NFKPluginCodeCheck)check
{
	BOOL signerRestricted = self.teamIdentifier.length > 0 || self.requirementString.length > 0;
	if (information[(__bridge NSString *)kSecCodeInfoIdentifier] == nil) {
		if (!self.requiresValidSignature && !signerRestricted) {
			return YES;
		}
		return NFKPluginSetError(error, signerRestricted && !self.requiresValidSignature ? NFKPluginErrorTeamMismatch : NFKPluginErrorSignatureInvalid,
								 [NSString stringWithFormat:@"%@ is not signed", name]);
	}
	CFErrorRef checkError = NULL;
	OSStatus status = check(NULL, &checkError);
	if (status != errSecSuccess) {
		NSString *reason = NFKPluginSecurityReason(status, checkError);
		if (checkError != NULL) {
			CFRelease(checkError);
		}
		if (!self.requiresValidSignature && !signerRestricted) {
			return YES;
		}
		return NFKPluginSetError(error, NFKPluginErrorSignatureInvalid, [NSString stringWithFormat:@"%@'s signature is invalid: %@", name, reason]);
	}
	uint32_t flags = [information[(__bridge NSString *)kSecCodeInfoFlags] unsignedIntValue];
	BOOL adHoc = (flags & kSecCodeSignatureAdhoc) != 0;
	if (adHoc && !self.allowsAdHocSignature) {
		return NFKPluginSetError(error, NFKPluginErrorSignatureInvalid,
								 [NSString stringWithFormat:@"%@ carries an ad hoc signature, which the policy refuses", name]);
	}
	return [self checkSignerOfCodeNamed:name information:information error:error check:check];
}

- (BOOL)checkSignerOfCodeNamed:(NSString *)name information:(NSDictionary *)information error:(NSError **)error check:(NFKPluginCodeCheck)check
{
	NSString *team = [information[(__bridge NSString *)kSecCodeInfoTeamIdentifier] isKindOfClass:NSString.class]
		? information[(__bridge NSString *)kSecCodeInfoTeamIdentifier] : nil;
	if (self.teamIdentifier.length > 0 && ![team isEqualToString:self.teamIdentifier]) {
		return NFKPluginSetError(error, NFKPluginErrorTeamMismatch,
								 [NSString stringWithFormat:@"%@ is signed by %@, and the policy allows only team %@",
								  name, team != nil ? [NSString stringWithFormat:@"team %@", team] : @"no team", self.teamIdentifier]);
	}
	NSMutableArray<NSString *> *clauses = [NSMutableArray array];
	if (self.teamIdentifier.length > 0) {
		// The team identifier in signing information is the signer's claim; the certificate chain proves it.
		[clauses addObject:[NSString stringWithFormat:@"anchor apple generic and certificate leaf[subject.OU] = \"%@\"", self.teamIdentifier]];
	}
	if (self.requirementString.length > 0) {
		[clauses addObject:[NSString stringWithFormat:@"(%@)", self.requirementString]];
	}
	if (clauses.count == 0) {
		return YES;
	}
	SecRequirementRef requirement = NULL;
	CFErrorRef compileError = NULL;
	OSStatus status = SecRequirementCreateWithStringAndErrors((__bridge CFStringRef)[clauses componentsJoinedByString:@" and "],
															  kSecCSDefaultFlags, &compileError, &requirement);
	if (status != errSecSuccess || requirement == NULL) {
		NSString *reason = NFKPluginSecurityReason(status, compileError);
		if (compileError != NULL) {
			CFRelease(compileError);
		}
		return NFKPluginSetError(error, NFKPluginErrorSignatureInvalid, [NSString stringWithFormat:@"the policy's requirement does not compile: %@", reason]);
	}
	CFErrorRef checkError = NULL;
	status = check(requirement, &checkError);
	CFRelease(requirement);
	if (status != errSecSuccess) {
		NSString *reason = NFKPluginSecurityReason(status, checkError);
		if (checkError != NULL) {
			CFRelease(checkError);
		}
		return NFKPluginSetError(error, NFKPluginErrorTeamMismatch,
								 [NSString stringWithFormat:@"%@'s signer does not satisfy the policy: %@", name, reason]);
	}
	return YES;
}

@end

#endif
