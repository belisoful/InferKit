//
//  NFKHFHub.m
//  InferKit
//

#import "NFKHFHub.h"
#import "NFKErrors.h"
#import "NFK_ARC.h"
#import <CommonCrypto/CommonCrypto.h>

NSString * const NFKHFHubDefaultRevision = @"main";

@implementation NFKHFHub

@synthesize cacheDirectoryURL = _cacheDirectoryURL;
@synthesize endpointURL = _endpointURL;
@synthesize session = _session;
@synthesize accessToken = _accessToken;

+ (instancetype)hubWithCacheDirectoryURL:(nullable NSURL *)cacheDirectoryURL
{
	NFKHFHub *hub = [[self alloc] init];
	hub.cacheDirectoryURL = cacheDirectoryURL;
	return NARC_AUTORELEASE(hub);
}

- (void)dealloc
{
	NARC_RELEASE(_cacheDirectoryURL);
	NARC_RELEASE(_endpointURL);
	NARC_RELEASE(_session);
	NARC_RELEASE(_accessToken);
	SUPER_DEALLOC();
}

- (NSURL *)endpointURL
{
	if (_endpointURL == nil) {
		_endpointURL = NARC_RETAIN([NSURL URLWithString:@"https://huggingface.co"]);
	}
	return _endpointURL;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = NARC_RETAIN([NSURLSession sharedSession]);
	}
	return _session;
}

- (nullable NSString *)accessToken
{
	if (_accessToken == nil) {
		// The conventional place a Hugging Face credential lives, and where the tooling around the
		// hub reads it from.
		_accessToken = NARC_RETAIN(NSProcessInfo.processInfo.environment[@"HF_TOKEN"]);
	}
	return _accessToken;
}

#pragma mark URL construction

+ (NSURL *)defaultCacheDirectoryURL
{
	NSURL *base = [NSFileManager.defaultManager URLForDirectory:NSApplicationSupportDirectory
													  inDomain:NSUserDomainMask
											 appropriateForURL:nil
														create:YES
														 error:NULL];
	NSURL *directory = [[base URLByAppendingPathComponent:@"InferKit"] URLByAppendingPathComponent:@"models"];
	[NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];
	return directory;
}

- (NSString *)normalizedRevision:(nullable NSString *)revision
{
	return revision.length > 0 ? revision : NFKHFHubDefaultRevision;
}

- (NSURL *)URL:(NSURL *)base byAppendingPathString:(NSString *)pathString
{
	NSURL *result = base;
	for (NSString *component in [pathString componentsSeparatedByString:@"/"]) {
		if (component.length > 0) {
			result = [result URLByAppendingPathComponent:component];
		}
	}
	return result;
}

- (nullable NSURL *)remoteURLForRepo:(NSString *)repo
						  revision:(nullable NSString *)revision
							  path:(NSString *)path
{
	if (repo.length == 0 || path.length == 0) {
		return nil;
	}
	NSURL *url = [self URL:self.endpointURL byAppendingPathString:repo];
	url = [url URLByAppendingPathComponent:@"resolve"];
	url = [url URLByAppendingPathComponent:[self normalizedRevision:revision]];
	return [self URL:url byAppendingPathString:path];
}

- (nullable NSURL *)localURLForRepo:(NSString *)repo
						 revision:(nullable NSString *)revision
							 path:(NSString *)path
{
	if (self.cacheDirectoryURL == nil || repo.length == 0 || path.length == 0) {
		return nil;
	}
	NSURL *url = [self URL:self.cacheDirectoryURL byAppendingPathString:repo];
	url = [url URLByAppendingPathComponent:[self normalizedRevision:revision]];
	return [self URL:url byAppendingPathString:path];
}

- (BOOL)isCachedRepo:(NSString *)repo revision:(nullable NSString *)revision path:(NSString *)path
{
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	return localURL != nil && [NSFileManager.defaultManager fileExistsAtPath:localURL.path];
}

#pragma mark Download

- (nullable NSURL *)downloadRepo:(NSString *)repo
					  revision:(nullable NSString *)revision
						  path:(NSString *)path
						sha256:(nullable NSString *)expectedSHA256
						 error:(NSError * _Nullable *)outError
{
	if (self.cacheDirectoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no cache directory is set"];
		return nil;
	}
	NSURL *localURL = [self localURLForRepo:repo revision:revision path:path];
	NSURL *remoteURL = [self remoteURLForRepo:repo revision:revision path:path];
	if (localURL == nil || remoteURL == nil) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"invalid repo or path"];
		return nil;
	}

	NSFileManager *fileManager = NSFileManager.defaultManager;
	BOOL scoped = [self.cacheDirectoryURL startAccessingSecurityScopedResource];
	@try {
		if ([fileManager fileExistsAtPath:localURL.path]) {
			if (expectedSHA256 == nil || [self fileAtURL:localURL matchesSHA256:expectedSHA256]) {
				return localURL;
			}
			[fileManager removeItemAtURL:localURL error:NULL];
		}

		NSError *dirError = nil;
		if (![fileManager createDirectoryAtURL:localURL.URLByDeletingLastPathComponent
				   withIntermediateDirectories:YES
									attributes:nil
										 error:&dirError]) {
			[self propagateError:dirError to:outError];
			return nil;
		}

		NSURL *partialURL = [localURL URLByAppendingPathExtension:@"download"];
		[fileManager removeItemAtURL:partialURL error:NULL];
		if (![self fetchURL:remoteURL toFileURL:partialURL error:outError]) {
			return nil;
		}

		if (expectedSHA256 != nil && ![self fileAtURL:partialURL matchesSHA256:expectedSHA256]) {
			[fileManager removeItemAtURL:partialURL error:NULL];
			[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the download failed its checksum"];
			return nil;
		}

		[fileManager removeItemAtURL:localURL error:NULL];
		NSError *moveError = nil;
		if (![fileManager moveItemAtURL:partialURL toURL:localURL error:&moveError]) {
			[self propagateError:moveError to:outError];
			return nil;
		}
		return localURL;
	} @finally {
		if (scoped) {
			[self.cacheDirectoryURL stopAccessingSecurityScopedResource];
		}
	}
}

- (void)downloadRepo:(NSString *)repo
			revision:(nullable NSString *)revision
				path:(NSString *)path
			  sha256:(nullable NSString *)expectedSHA256
   completionHandler:(void (^)(NSURL * _Nullable, NSError * _Nullable))completionHandler
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
		NSError *error = nil;
		NSURL *localURL = [self downloadRepo:repo revision:revision path:path sha256:expectedSHA256 error:&error];
		completionHandler(localURL, error);
	});
}

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError * _Nullable *)outError
{
	__block NSError *resultError = nil;
	__block BOOL moved = NO;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:remoteURL];
	NSString *token = self.accessToken;
	if (token.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:token] forHTTPHeaderField:@"Authorization"];
	}

	NSURLSessionDownloadTask *task = [self.session downloadTaskWithRequest:request
													completionHandler:^(NSURL *location, NSURLResponse *response, NSError *error) {
		// The downloaded temp file is removed when this handler returns, so move it here.
		if (location != nil && error == nil) {
			NSHTTPURLResponse *http = [response isKindOfClass:NSHTTPURLResponse.class] ? (NSHTTPURLResponse *)response : nil;
			if (http != nil && (http.statusCode < 200 || http.statusCode >= 300)) {
				NSString *reason = [NSString stringWithFormat:@"the download returned HTTP %ld", (long)http.statusCode];
				NSError *httpError = [NSError errorWithDomain:NFKInferenceErrorDomain
														 code:kNFKError_InferenceBackendFailure
													 userInfo:@{ NSLocalizedDescriptionKey: reason }];
				resultError = NARC_RETAIN(httpError);
			} else {
				NSError *moveError = nil;
				[NSFileManager.defaultManager removeItemAtURL:destinationURL error:NULL];
				moved = [NSFileManager.defaultManager moveItemAtURL:location toURL:destinationURL error:&moveError];
				resultError = NARC_RETAIN(moveError);
			}
		} else {
			resultError = NARC_RETAIN(error);
		}
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (!moved) {
		[self propagateError:resultError to:outError];
	}
	return moved;
}

#pragma mark Checksum

- (BOOL)fileAtURL:(NSURL *)url matchesSHA256:(NSString *)expected
{
	NSString *actual = [self sha256OfFileAtURL:url];
	return actual != nil && [actual caseInsensitiveCompare:expected] == NSOrderedSame;
}

- (nullable NSString *)sha256OfFileAtURL:(NSURL *)url
{
	NSFileHandle *handle = [NSFileHandle fileHandleForReadingFromURL:url error:NULL];
	if (handle == nil) {
		return nil;
	}
	CC_SHA256_CTX context;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CC_SHA256_Init(&context);
#pragma clang diagnostic pop
	@try {
		while (YES) {
			@autoreleasepool {
				NSData *chunk = [handle readDataOfLength:(1 << 20)];
				if (chunk.length == 0) {
					break;
				}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				CC_SHA256_Update(&context, chunk.bytes, (CC_LONG)chunk.length);
#pragma clang diagnostic pop
			}
		}
	} @finally {
		[handle closeFile];
	}
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
	CC_SHA256_Final(digest, &context);
#pragma clang diagnostic pop
	NSMutableString *hex = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
	for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
		[hex appendFormat:@"%02x", digest[i]];
	}
	return hex;
}

#pragma mark Errors

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	*outError = error != nil ? error
							: [NSError errorWithDomain:NFKInferenceErrorDomain
												  code:kNFKError_InferenceBackendFailure
											  userInfo:@{ NSLocalizedDescriptionKey: @"the download failed" }];
	return NO;
}

@end
