//
//  NFKRemoteOCRBackendTests.m
//  InferKitTests
//
//  Mistral's OCR through a stubbed transport: the document's shape on the wire, the pages joined as
//  markdown, a schema as a document annotation, and the images cut from the pages.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteOCRBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubOCRBackend : NFKRemoteOCRBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKStubOCRBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}
@end

@interface NFKRemoteOCRBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubOCRBackend *backend;
@end

@implementation NFKRemoteOCRBackendTests

- (void)setUp
{
	[super setUp];
	NFKRemoteOCRBackend *made = [NFKRemoteOCRBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-ocr-latest"];
	self.backend = [NFKStubOCRBackend backendWithEndpointURL:made.endpointURL];
	self.backend.modelName = @"mistral-ocr-latest";
	self.backend.apiKey = @"k";
}

- (void)testOnlyMistralServesOCR
{
	XCTAssertEqualObjects(self.backend.endpointURL.absoluteString, @"https://api.mistral.ai/v1/ocr");
	XCTAssertNil([NFKRemoteOCRBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
}

- (void)testAPDFGoesAsADataURIAndThePagesJoinAsMarkdownWithTheAnnotationParsed
{
	self.backend.stagedBody = @"{\"pages\":[{\"index\":0,\"markdown\":\"# Invoice\",\"images\":[]},"
		"{\"index\":1,\"markdown\":\"Total: 42\",\"images\":[]}],\"model\":\"mistral-ocr\","
		"\"document_annotation\":\"{\\\"total\\\":42}\",\"usage_info\":{\"pages_processed\":2}}";
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"total": @{ @"type": @"number" } } };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputDocument: [@"%PDF-1.4" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:@{ NFKParameterJSONSchema: schema, @"table_format": @"html" }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"document"][@"type"], @"document_url");
	XCTAssertTrue([body[@"document"][@"document_url"] hasPrefix:@"data:application/pdf;base64,"]);
	XCTAssertEqualObjects(body[@"document_annotation_format"][@"json_schema"][@"schema"], schema);
	XCTAssertEqualObjects(body[@"table_format"], @"html");
	XCTAssertEqualObjects(result.text, @"# Invoice\n\nTotal: 42");
	XCTAssertEqualObjects(result.structured, (@{ @"total": @42 }));
}

- (void)testAHostedPDFGoesByURLAndAnImageAsAnImageURLAndCutImagesDecode
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSString *png = [[NFKImageCoding PNGDataForImage:(__bridge id)square] base64EncodedStringWithOptions:0];
	self.backend.stagedBody = [NSString stringWithFormat:@"{\"pages\":[{\"markdown\":\"![img-0](img-0)\","
							   "\"images\":[{\"id\":\"img-0\",\"image_base64\":\"data:image/png;base64,%@\"}]}]}", png];
	NSURL *hosted = [NSURL URLWithString:@"https://example.com/paper.pdf"];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputDocument: hosted }] error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"document"], (@{ @"type": @"document_url", @"document_url": @"https://example.com/paper.pdf" }));
	XCTAssertEqual([[result outputForKey:NFKOutputImages] count], 1);

	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)square }] error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"document"][@"type"], @"image_url");
	CGImageRelease(square);

	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

@end
