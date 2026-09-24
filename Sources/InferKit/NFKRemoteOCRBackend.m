//
//  NFKRemoteOCRBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteOCRBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteFileStore.h>

@implementation NFKRemoteOCRBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteOCRBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (![provider.identifier isEqualToString:@"mistral"]) {
		return nil;
	}
	NFKRemoteOCRBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"ocr"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 300.0;
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
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote-ocr";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
	}
	NSDictionary *document = [self documentForRequest:request error:outError];
	if (document == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionaryWithObject:document forKey:@"document"];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	NSDictionary *schema = request.parameters[NFKParameterJSONSchema];
	if ([schema isKindOfClass:NSDictionary.class]) {
		body[@"document_annotation_format"] = @{ @"type": @"json_schema",
												 @"json_schema": @{ @"name": @"document", @"schema": schema } };
	}
	for (NSString *key in request.parameters) {
		if (![key isEqualToString:NFKParameterJSONSchema]) {
			body[key] = request.parameters[key];
		}
	}

	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the OCR call failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![reply isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object" error:outError];
	}
	return [self resultFromReply:reply];
}

// A PDF goes as a document_url (a hosted URL as it stands, local bytes as a data URI); an image as
// an image_url.
- (nullable NSDictionary *)documentForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	id source = [request inputForKey:NFKInputDocument];
	if ([source isKindOfClass:NFKRemoteFile.class]) {
		return @{ @"type": @"file", @"file_id": [(NFKRemoteFile *)source identifier] };
	}
	if ([source isKindOfClass:NSURL.class] && ![(NSURL *)source isFileURL]) {
		return @{ @"type": @"document_url", @"document_url": [(NSURL *)source absoluteString] };
	}
	NSData *pdf = [source isKindOfClass:NSData.class] ? source
				: [source isKindOfClass:NSURL.class] ? [NSData dataWithContentsOfURL:source] : nil;
	if (pdf != nil) {
		NSString *dataURL = [@"data:application/pdf;base64," stringByAppendingString:[pdf base64EncodedStringWithOptions:0]];
		return @{ @"type": @"document_url", @"document_url": dataURL };
	}
	id image = [request inputForKey:NFKInputImage];
	NSString *imageURL = image != nil ? [NFKImageCoding dataURLForImage:image] : nil;
	if (imageURL != nil) {
		return @{ @"type": @"image_url", @"image_url": imageURL };
	}
	[self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries neither a readable document nor an image" error:outError];
	return nil;
}

- (NFKInferenceResult *)resultFromReply:(NSDictionary *)reply
{
	NSMutableArray<NSString *> *markdown = [NSMutableArray array];
	NSMutableArray *images = [NSMutableArray array];
	NSArray *pages = [reply[@"pages"] isKindOfClass:NSArray.class] ? reply[@"pages"] : @[];
	for (NSDictionary *page in pages) {
		if (![page isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([page[@"markdown"] isKindOfClass:NSString.class]) {
			[markdown addObject:page[@"markdown"]];
		}
		for (NSDictionary *cut in [page[@"images"] isKindOfClass:NSArray.class] ? page[@"images"] : @[]) {
			NSString *encoded = [cut isKindOfClass:NSDictionary.class] ? cut[@"image_base64"] : nil;
			NSRange comma = [encoded isKindOfClass:NSString.class] ? [encoded rangeOfString:@","] : NSMakeRange(NSNotFound, 0);
			NSString *payload = comma.location != NSNotFound ? [encoded substringFromIndex:comma.location + 1] : encoded;
			NSData *bytes = [payload isKindOfClass:NSString.class]
				? [[NSData alloc] initWithBase64EncodedString:payload options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
			CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
			if (pixelBuffer != NULL) {
				[images addObject:(__bridge id)pixelBuffer];
				CVPixelBufferRelease(pixelBuffer);
			}
		}
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:reply forKey:NFKRemoteBackendRawKey];
	outputs[NFKOutputText] = [markdown componentsJoinedByString:@"\n\n"];
	if (images.count > 0) {
		outputs[NFKOutputImages] = images;
	}
	// The annotation arrives as a JSON string.
	NSString *annotation = [reply[@"document_annotation"] isKindOfClass:NSString.class] ? reply[@"document_annotation"] : nil;
	id structured = annotation != nil
		? [NSJSONSerialization JSONObjectWithData:[annotation dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
	if ([structured isKindOfClass:NSDictionary.class]) {
		outputs[NFKOutputStructured] = structured;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable NFKInferenceResult *)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
