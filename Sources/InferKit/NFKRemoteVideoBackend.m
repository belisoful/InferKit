//
//  NFKRemoteVideoBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteVideoBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"

@implementation NFKRemoteVideoBackend

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (provider.apiStyle != NFKRemoteAPIStyleOpenAIChat) {
		return nil;
	}
	NFKRemoteVideoBackend *backend = [[self alloc] init];
	backend.submitURL = [provider URLForPath:@"videos"];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 300.0;
		self.pollInterval = 5.0;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return @"remote-video";
}

#pragma mark Submit

// The fields the contract names map onto the service's; everything else passes by name.
- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	body[@"prompt"] = request.prompt ?: @"";
	NSNumber *seconds = request.parameters[NFKParameterDurationSeconds];
	if ([seconds isKindOfClass:NSNumber.class]) {
		body[@"seconds"] = [NSString stringWithFormat:@"%ld", (long)seconds.integerValue];
	}
	NSNumber *width = request.parameters[NFKParameterWidth];
	NSNumber *height = request.parameters[NFKParameterHeight];
	if ([width isKindOfClass:NSNumber.class] && [height isKindOfClass:NSNumber.class]) {
		body[@"size"] = [NSString stringWithFormat:@"%ldx%ld", (long)width.integerValue, (long)height.integerValue];
	}
	NSSet<NSString *> *mapped = [NSSet setWithArray:@[ NFKParameterDurationSeconds, NFKParameterWidth, NFKParameterHeight ]];
	for (NSString *key in request.parameters) {
		if (![mapped containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	return body;
}

// A reference image makes the submit multipart, the image as a PNG file beside the fields.
- (NSURLRequest *)submitRequestForRequest:(NFKInferenceRequest *)request
{
	id image = [request inputForKey:NFKInputImage];
	NSData *png = image != nil ? [NFKImageCoding PNGDataForImage:image] : nil;
	if (png == nil) {
		NSMutableURLRequest *plain = [[super submitRequestForRequest:request] mutableCopy];
		plain.timeoutInterval = self.timeout;
		return plain;
	}
	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableData *body = [NSMutableData data];
	NSString *dashBoundary = [NSString stringWithFormat:@"--%@\r\n", boundary];
	NSDictionary<NSString *, id> *fields = [self submitBodyForRequest:request];
	for (NSString *key in fields) {
		[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
		NSString *part = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", key, [fields[key] description]];
		[body appendData:[part dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:[dashBoundary dataUsingEncoding:NSUTF8StringEncoding]];
	[body appendData:[@"Content-Disposition: form-data; name=\"input_reference\"; filename=\"reference.png\"\r\nContent-Type: image/png\r\n\r\n"
					  dataUsingEncoding:NSUTF8StringEncoding]];
	[body appendData:png];
	[body appendData:[[NSString stringWithFormat:@"\r\n--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.submitURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = body;
	[urlRequest setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

#pragma mark Status

// The service reports progress as a percentage.
- (double)progressFromStatusResponse:(NSDictionary *)response
{
	id progress = response[@"progress"];
	if (![progress isKindOfClass:NSNumber.class]) {
		return -1.0;
	}
	double value = [progress doubleValue];
	return value > 1.0 ? value / 100.0 : value;
}

// A completed job's clip is a second fetch, from the job's content path.
- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response
											 outputModality:(NFKModality)outputModality
{
	NSString *identifier = [self jobIdentifierFromResponse:response];
	if (identifier.length == 0) {
		return nil;
	}
	NSURL *contentURL = [[self.submitURL URLByAppendingPathComponent:identifier] URLByAppendingPathComponent:@"content"];
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:contentURL];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

	NSHTTPURLResponse *httpResponse = nil;
	NSData *bytes = [self sendRequest:request response:&httpResponse error:NULL];
	if (bytes == nil || [NFKRemoteTransport errorForResponse:httpResponse data:bytes] != nil) {
		return nil;
	}
	NSURL *fileURL = NFKRemoteWriteMediaFile(bytes, @"video", @"mp4", self.outputDirectoryURL, NULL);
	if (fileURL == nil) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputVideo: [NFKVideoAsset videoAssetWithFileURL:fileURL] }];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

@end
