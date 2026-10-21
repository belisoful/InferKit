//
//  NFKHFHubTransfer.m
//  InferKit
//

#import "NFKHFHubTransfer.h"
#import "NFKDownload.h"
#import "NFKDownloadInternal.h"
#import <sys/xattr.h>

/*! The extended attribute a partial file records its server's strong ETag under. */
static const char *NFKHFHubEntityTagAttribute = "com.inferkit.download.etag";

@implementation NFKHFHubTransferReport
@end

/*! How one request of a transfer ended. */
typedef NS_ENUM(NSInteger, NFKHFHubAttemptOutcome) {
	NFKHFHubAttemptSucceeded,
	NFKHFHubAttemptRetryable,
	NFKHFHubAttemptFailed,
};

@interface NFKHFHubTransfer () <NSURLSessionDataDelegate>
@property (nonatomic, strong) NSURLRequest *baseRequest;
@property (nonatomic, copy) NSURL *destinationURL;
@property (nonatomic, strong, nullable) NFKHFHubTransferReport *report;
@property (nonatomic, strong, nullable) NSFileHandle *fileHandle;
@property (nonatomic, assign) long long writtenByteCount;
@property (nonatomic, assign) long long expectedByteCount;
@property (nonatomic, assign) long long listedByteCount;
@property (nonatomic, assign) long long ceilingByteCount;
@property (nonatomic, copy, nullable) NSString *entityTag;
@property (nonatomic, strong, nullable) NSError *attemptError;
@property (nonatomic, assign) BOOL attemptRestarts;
@property (nonatomic, strong, nullable) NSURLSessionDataTask *task;
@property (nonatomic, strong) dispatch_semaphore_t attemptDone;
@end

@implementation NFKHFHubTransfer

+ (NSUInteger)maximumResumeCount
{
	return 3;
}

+ (BOOL)transferRequest:(NSURLRequest *)request
		  configuration:(NSURLSessionConfiguration *)configuration
			  toFileURL:(NSURL *)destinationURL
				 report:(nullable NFKHFHubTransferReport *)report
				  error:(NSError * _Nullable *)outError
{
	NFKHFHubTransfer *transfer = [[NFKHFHubTransfer alloc] init];
	transfer.baseRequest = request;
	transfer.destinationURL = destinationURL;
	transfer.report = report;
	transfer.expectedByteCount = report != nil ? report.expectedByteCount : NFKDownloadUnknownByteCount;
	transfer.listedByteCount = transfer.expectedByteCount;
	transfer.ceilingByteCount = report != nil ? report.ceilingByteCount : -1;
	transfer.attemptDone = dispatch_semaphore_create(0);
	NSError *error = nil;
	BOOL succeeded = [transfer runWithConfiguration:configuration error:&error];
	BOOL keeps = report.download.isPaused && !report.download.isCancelled;
	if (!succeeded && !keeps) {
		[NSFileManager.defaultManager removeItemAtURL:destinationURL error:NULL];
		if (outError != NULL) {
			*outError = error;
		}
	}
	return succeeded;
}

+ (nullable NSString *)entityTagOfPartialFileAtURL:(NSURL *)URL
{
	char buffer[1024];
	ssize_t length = getxattr(URL.fileSystemRepresentation, NFKHFHubEntityTagAttribute, buffer, sizeof(buffer), 0, 0);
	if (length <= 0) {
		return nil;
	}
	return [[NSString alloc] initWithBytes:buffer length:(NSUInteger)length encoding:NSUTF8StringEncoding];
}

+ (void)clearEntityTagOfFileAtURL:(NSURL *)URL
{
	removexattr(URL.fileSystemRepresentation, NFKHFHubEntityTagAttribute, 0);
}

/*! Records the strong ETag the destination's bytes are served under, or removes a stale one. */
- (void)recordEntityTag:(nullable NSString *)tag
{
	const char *path = self.destinationURL.fileSystemRepresentation;
	if (tag == nil) {
		removexattr(path, NFKHFHubEntityTagAttribute, 0);
		return;
	}
	NSData *bytes = [tag dataUsingEncoding:NSUTF8StringEncoding];
	setxattr(path, NFKHFHubEntityTagAttribute, bytes.bytes, bytes.length, 0, 0);
}

+ (NSError *)errorForHTTPStatus:(NSInteger)status subject:(NSString *)subject hadToken:(BOOL)hadToken
{
	NSString *reason = nil;
	NFKDownloadError code = NFKDownloadErrorHTTPStatus;
	if (status == 401 && !hadToken) {
		code = NFKDownloadErrorAuthenticationRequired;
		reason = [NSString stringWithFormat:@"%@ needs an access token (HTTP 401): the repository is gated, private, or "
				  "does not exist. Set NFKHFHub.accessToken or NFKHFHub.defaultAccessToken.", subject];
	} else if (status == 401) {
		code = NFKDownloadErrorAuthenticationRequired;
		reason = [NSString stringWithFormat:@"%@ refused the access token (HTTP 401), or the repository does not exist", subject];
	} else if (status == 403) {
		code = NFKDownloadErrorAuthenticationRequired;
		reason = [NSString stringWithFormat:@"%@ is gated (HTTP 403): the account behind the access token has not been "
				  "granted access. Accept the repository's terms on its Hugging Face page.", subject];
	} else if (status == 404) {
		code = NFKDownloadErrorNotFound;
		reason = [NSString stringWithFormat:@"%@ does not exist (HTTP 404)", subject];
	} else {
		reason = [NSString stringWithFormat:@"%@ returned HTTP %ld", subject, (long)status];
	}
	return [NSError errorWithDomain:NFKDownloadErrorDomain
							   code:code
						   userInfo:@{ NSLocalizedDescriptionKey: reason, NFKDownloadHTTPStatusCodeKey: @(status) }];
}

- (BOOL)runWithConfiguration:(NSURLSessionConfiguration *)configuration error:(NSError **)outError
{
	NFKDownload *download = self.report.download;
	if (download.isStopped) {
		*outError = download.stopError;
		return NO;
	}
	if (![self openDestinationError:outError]) {
		return NO;
	}

	NSOperationQueue *delegateQueue = [[NSOperationQueue alloc] init];
	delegateQueue.maxConcurrentOperationCount = 1;
	delegateQueue.qualityOfService = NSQualityOfServiceUserInitiated;
	NSURLSession *session = [NSURLSession sessionWithConfiguration:configuration delegate:self delegateQueue:delegateQueue];
	__weak NFKHFHubTransfer *weakSelf = self;
	id cancellationToken = [download addCancellationHandler:^{
		NFKHFHubTransfer *strongSelf = weakSelf;
		@synchronized (strongSelf) {
			[strongSelf.task cancel];
		}
	}];

	NFKHFHubAttemptOutcome outcome = NFKHFHubAttemptFailed;
	NSUInteger resumes = 0;
	@try {
		while (YES) {
			outcome = [self runAttemptInSession:session];
			if (outcome != NFKHFHubAttemptRetryable || resumes >= NFKHFHubTransfer.maximumResumeCount
				|| download.isStopped) {
				break;
			}
			resumes++;
		}
	} @finally {
		if (cancellationToken != nil) {
			[download removeCancellationHandler:cancellationToken];
		}
		[session invalidateAndCancel];
		[self.fileHandle closeAndReturnError:NULL];
		self.fileHandle = nil;
	}

	if (download.isStopped) {
		*outError = download.stopError;
		return NO;
	}
	if (outcome != NFKHFHubAttemptSucceeded) {
		*outError = self.attemptError ?: [NFKDownload errorWithCode:NFKDownloadErrorIncomplete reason:@"the transfer failed"];
		return NO;
	}
	if (self.expectedByteCount >= 0 && self.writtenByteCount != self.expectedByteCount) {
		NSString *reason = [NSString stringWithFormat:@"%@ arrived with %lld of %lld bytes",
							self.baseRequest.URL.lastPathComponent, self.writtenByteCount, self.expectedByteCount];
		*outError = [NFKDownload errorWithCode:NFKDownloadErrorIncomplete reason:reason];
		return NO;
	}
	return YES;
}

/*! Opens the destination for writing: at its end when the report names bytes a paused download kept
	there under a strong ETag, else emptied. */
- (BOOL)openDestinationError:(NSError **)outError
{
	NSFileManager *fileManager = NSFileManager.defaultManager;
	long long kept = self.report.keptByteCount;
	NSString *keptTag = self.report.keptEntityTag;
	long long size = [[fileManager attributesOfItemAtPath:self.destinationURL.path error:NULL][NSFileSize] longLongValue];
	if (kept > 0 && keptTag != nil && size == kept) {
		self.fileHandle = [NSFileHandle fileHandleForWritingToURL:self.destinationURL error:outError];
		if (self.fileHandle == nil || ![self.fileHandle seekToEndReturningOffset:NULL error:outError]) {
			return NO;
		}
		self.writtenByteCount = kept;
		self.entityTag = keptTag;
		return YES;
	}
	[fileManager removeItemAtURL:self.destinationURL error:NULL];
	if (![fileManager createFileAtPath:self.destinationURL.path contents:nil attributes:nil]) {
		if (outError != NULL) {
			*outError = [NFKDownload errorWithCode:NFKDownloadErrorHTTPStatus
											reason:[NSString stringWithFormat:@"cannot create %@", self.destinationURL.path]];
		}
		return NO;
	}
	self.fileHandle = [NSFileHandle fileHandleForWritingToURL:self.destinationURL error:outError];
	return self.fileHandle != nil;
}

- (NFKHFHubAttemptOutcome)runAttemptInSession:(NSURLSession *)session
{
	NSMutableURLRequest *request = [self.baseRequest mutableCopy];
	BOOL resuming = self.writtenByteCount > 0 && self.entityTag != nil;
	if (self.writtenByteCount > 0 && !resuming) {
		[self truncateFile];
	}
	if (resuming) {
		[request setValue:[NSString stringWithFormat:@"bytes=%lld-", self.writtenByteCount] forHTTPHeaderField:@"Range"];
		[request setValue:self.entityTag forHTTPHeaderField:@"If-Range"];
	}
	self.attemptError = nil;
	self.attemptRestarts = NO;
	NSURLSessionDataTask *task = [session dataTaskWithRequest:request];
	@synchronized (self) {
		self.task = task;
		// A cancel that ran before the task existed found nothing to cancel.
		if (self.report.download.isStopped) {
			[task cancel];
		}
	}
	[task resume];
	dispatch_semaphore_wait(self.attemptDone, DISPATCH_TIME_FOREVER);
	@synchronized (self) {
		self.task = nil;
	}
	if (self.attemptRestarts) {
		[self truncateFile];
		return NFKHFHubAttemptRetryable;
	}
	if (self.attemptError == nil) {
		return NFKHFHubAttemptSucceeded;
	}
	return [self isRetryableError:self.attemptError] ? NFKHFHubAttemptRetryable : NFKHFHubAttemptFailed;
}

- (BOOL)isRetryableError:(NSError *)error
{
	if (![error.domain isEqualToString:NSURLErrorDomain]) {
		return NO;
	}
	switch (error.code) {
		case NSURLErrorNetworkConnectionLost:
		case NSURLErrorTimedOut:
		case NSURLErrorCannotConnectToHost:
		case NSURLErrorResourceUnavailable:
			return YES;
		default:
			return NO;
	}
}

- (void)truncateFile
{
	[self.fileHandle truncateAtOffset:0 error:NULL];
	[self.fileHandle seekToOffset:0 error:NULL];
	self.writtenByteCount = 0;
}

#pragma mark Session delegate

- (void)URLSession:(NSURLSession *)session
		  dataTask:(NSURLSessionDataTask *)dataTask
didReceiveResponse:(NSURLResponse *)response
 completionHandler:(void (^)(NSURLSessionResponseDisposition))completionHandler
{
	NSHTTPURLResponse *http = [response isKindOfClass:NSHTTPURLResponse.class] ? (NSHTTPURLResponse *)response : nil;
	NSInteger status = http != nil ? http.statusCode : 200;
	if (status == 206) {
		long long start = -1;
		long long total = -1;
		NSString *range = [http valueForHTTPHeaderField:@"Content-Range"];
		if (range != nil) {
			sscanf(range.UTF8String, "bytes %lld-%*lld/%lld", &start, &total);
		}
		if (start != self.writtenByteCount) {
			self.attemptRestarts = YES;
			completionHandler(NSURLSessionResponseCancel);
			return;
		}
		if (total >= 0) {
			self.expectedByteCount = total;
		}
		completionHandler(NSURLSessionResponseAllow);
		return;
	}
	if (status >= 200 && status < 300) {
		if (self.writtenByteCount > 0) {
			[self truncateFile];
		}
		// A content-encoded body announces its encoded length, and the session hands over the
		// decoded bytes, so only an identity body's length is the file's size.
		NSString *encoding = [http valueForHTTPHeaderField:@"Content-Encoding"];
		BOOL identity = encoding.length == 0 || [encoding caseInsensitiveCompare:@"identity"] == NSOrderedSame;
		if (identity && response.expectedContentLength >= 0) {
			self.expectedByteCount = response.expectedContentLength;
		}
		NSString *tag = [http valueForHTTPHeaderField:@"ETag"];
		self.entityTag = tag.length > 0 && ![tag hasPrefix:@"W/"] ? tag : nil;
		[self recordEntityTag:self.entityTag];
		[self reportWrittenBytes];
		completionHandler(NSURLSessionResponseAllow);
		return;
	}
	if (status == 416 && self.writtenByteCount > 0) {
		self.attemptRestarts = YES;
		completionHandler(NSURLSessionResponseCancel);
		return;
	}
	BOOL hadToken = [dataTask.originalRequest valueForHTTPHeaderField:@"Authorization"] != nil;
	self.attemptError = [NFKHFHubTransfer errorForHTTPStatus:status
													 subject:dataTask.originalRequest.URL.absoluteString
													hadToken:hadToken];
	completionHandler(NSURLSessionResponseCancel);
}

/*! The most bytes the file may hold: its listed size, else the length the server announced, lowered
	to the ceiling; -1 for no limit. */
- (long long)streamLimit
{
	long long limit = self.listedByteCount >= 0 ? self.listedByteCount : self.expectedByteCount;
	long long ceiling = self.ceilingByteCount;
	if (ceiling >= 0 && (limit < 0 || ceiling < limit)) {
		limit = ceiling;
	}
	return limit;
}

- (void)URLSession:(NSURLSession *)session dataTask:(NSURLSessionDataTask *)dataTask didReceiveData:(NSData *)data
{
	long long limit = [self streamLimit];
	if (limit >= 0 && self.writtenByteCount + (long long)data.length > limit) {
		NSString *reason = [NSString stringWithFormat:@"%@ streamed past its limit of %lld bytes",
							self.baseRequest.URL.lastPathComponent, limit];
		self.attemptError = [NFKDownload errorWithCode:NFKDownloadErrorSizeMismatch reason:reason];
		[dataTask cancel];
		return;
	}
	NSError *writeError = nil;
	if (![self.fileHandle writeData:data error:&writeError]) {
		self.attemptError = writeError;
		[dataTask cancel];
		return;
	}
	self.writtenByteCount += (long long)data.length;
	[self reportWrittenBytes];
}

- (void)URLSession:(NSURLSession *)session task:(NSURLSessionTask *)task didCompleteWithError:(nullable NSError *)error
{
	if (self.attemptError == nil && !self.attemptRestarts && error != nil) {
		self.attemptError = error;
	}
	dispatch_semaphore_signal(self.attemptDone);
}

- (void)reportWrittenBytes
{
	NFKHFHubTransferReport *report = self.report;
	if (report == nil) {
		return;
	}
	[report.download fileAtLocalURL:report.localURL
				  didWriteByteCount:self.writtenByteCount
				  expectedByteCount:self.expectedByteCount];
}

@end
