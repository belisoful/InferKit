//
//  NFKAsyncGenerationBackend.m
//  InferKit
//

#import "NFKAsyncGenerationBackend.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceKeys.h"
#import "NFKVideoAsset.h"
#import "NFKErrors.h"

@implementation NFKAsyncGenerationBackend

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_pollInterval = 2.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.submitURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"async-generation";
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	if (self.submitURL == nil) {
		[job finishWithError:[self errorWithReason:@"no submit URL is set"]];
		return job;
	}

	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[self runJob:job forRequest:request];
	});
	return job;
}

- (void)runJob:(NFKInferenceJob *)job forRequest:(NFKInferenceRequest *)request
{
	NSError *error = nil;
	NSDictionary *submitResponse = [self sendJSONRequest:[self submitRequestForRequest:request] error:&error];
	if (submitResponse == nil) {
		[job finishWithError:error ?: [self errorWithReason:@"submit failed"]];
		return;
	}
	NSString *jobIdentifier = [self jobIdentifierFromResponse:submitResponse];
	if (jobIdentifier.length == 0) {
		[job finishWithError:[self errorWithReason:@"the submit response had no job id"]];
		return;
	}
	NSURL *statusURL = [self statusURLForJobIdentifier:jobIdentifier];
	if (statusURL == nil) {
		[job finishWithError:[self errorWithReason:@"could not form a status URL"]];
		return;
	}

	while (YES) {
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		NSError *pollError = nil;
		NSDictionary *statusResponse = [self sendJSONRequest:[self statusRequestForURL:statusURL] error:&pollError];
		if (statusResponse == nil) {
			[job finishWithError:pollError ?: [self errorWithReason:@"status poll failed"]];
			return;
		}
		if ([self isFailedStatusResponse:statusResponse]) {
			[job finishWithError:[self errorWithReason:@"the generation job failed"]];
			return;
		}
		if ([self isSucceededStatusResponse:statusResponse]) {
			NFKInferenceResult *result = [self resultFromStatusResponse:statusResponse
														 outputModality:request.outputModality];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:[self errorWithReason:@"the completed job had no usable output"]];
			}
			return;
		}
		[job reportProgress:[self progressFromStatusResponse:statusResponse]];
		if (self.pollInterval > 0.0) {
			[NSThread sleepForTimeInterval:self.pollInterval];
		}
	}
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NFKInferenceJob *job = [self submitInferenceJobForRequest:request];
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	job.completionHandler = ^(NFKInferenceJob *finished) {
		dispatch_semaphore_signal(semaphore);
	};
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (job.result != nil) {
		return job.result;
	}
	if (outError != NULL) {
		*outError = job.error ?: [self errorWithReason:@"the generation job did not produce a result"];
	}
	return nil;
}

#pragma mark Requests

- (NSURLRequest *)submitRequestForRequest:(NFKInferenceRequest *)request
{
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.submitURL];
	urlRequest.HTTPMethod = @"POST";
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[self applyAuthorization:urlRequest];
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:[self submitBodyForRequest:request] options:0 error:NULL];
	return urlRequest;
}

- (NSURLRequest *)statusRequestForURL:(NSURL *)url
{
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"GET";
	[self applyAuthorization:urlRequest];
	return urlRequest;
}

- (void)applyAuthorization:(NSMutableURLRequest *)request
{
	if (self.apiKey.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:self.apiKey] forHTTPHeaderField:@"Authorization"];
	}
}

#pragma mark Schema hooks (default provider-neutral shapes)

- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	for (NSString *key in request.inputs) {
		id value = request.inputs[key];
		// Only JSON-safe inputs (prompts, references by name) go in the body; media is provider-specific.
		if ([value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class]) {
			body[key] = value;
		}
	}
	for (NSString *key in request.parameters) {
		body[key] = request.parameters[key];
	}
	return body;
}

- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response
{
	id identifier = response[@"id"];
	return [identifier isKindOfClass:NSString.class] ? identifier : nil;
}

- (nullable NSURL *)statusURLForJobIdentifier:(NSString *)jobIdentifier
{
	return [self.submitURL URLByAppendingPathComponent:jobIdentifier];
}

- (double)progressFromStatusResponse:(NSDictionary *)response
{
	id progress = response[@"progress"];
	return [progress isKindOfClass:NSNumber.class] ? [progress doubleValue] : -1.0;
}

- (BOOL)isSucceededStatusResponse:(NSDictionary *)response
{
	NSString *status = [response[@"status"] isKindOfClass:NSString.class] ? response[@"status"] : nil;
	return [status isEqualToString:@"succeeded"] || [status isEqualToString:@"completed"];
}

- (BOOL)isFailedStatusResponse:(NSDictionary *)response
{
	NSString *status = [response[@"status"] isKindOfClass:NSString.class] ? response[@"status"] : nil;
	return [status isEqualToString:@"failed"] || [status isEqualToString:@"error"];
}

- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response
											 outputModality:(NFKModality)outputModality
{
	id outputString = response[@"output"];
	if (![outputString isKindOfClass:NSString.class]) {
		return nil;
	}
	NSURL *outputURL = [NSURL URLWithString:outputString];
	if (outputURL == nil) {
		return nil;
	}
	if (outputModality == NFKModalityVideo) {
		NFKVideoAsset *asset = [NFKVideoAsset videoAssetWithFileURL:outputURL];
		return [NFKInferenceResult resultWithOutputs:@{ NFKOutputVideo: asset }];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: outputURL }];
}

#pragma mark Transport seam

- (nullable NSDictionary *)sendJSONRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	__block NSData *resultData = nil;
	__block NSError *resultError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request
												completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		resultData = data;
		resultError = error;
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (resultData == nil) {
		if (outError != NULL) {
			*outError = resultError ?: [self errorWithReason:@"the request returned no data"];
		}
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:resultData options:0 error:outError];
	return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

#pragma mark Errors

- (NSError *)errorWithReason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain
							   code:kNFKError_InferenceBackendFailure
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

@end
