//
//  NFKValueAccessorsTests.m
//  NFKTests
//
//  The typed convenience accessors on the request and result: they return the value when it is the
//  expected type, and nil when it is absent or the wrong type (rather than crashing a cast).
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>

@interface NFKValueAccessorsTests : XCTestCase
@end

@implementation NFKValueAccessorsTests

#pragma mark Result

- (void)testResultTextReturnsTheString
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"hello" }];
	XCTAssertEqualObjects(result.text, @"hello");
}

- (void)testResultStructuredReturnsTheDictionary
{
	NSDictionary *fields = @{ @"name": @"Aria", @"age": @27 };
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputStructured: fields }];
	XCTAssertEqualObjects(result.structured, fields);
}

- (void)testResultEmbeddingReturnsTheVector
{
	NSArray<NSNumber *> *vector = @[ @0.1, @-0.2, @0.3 ];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: vector }];
	XCTAssertEqualObjects(result.embedding, vector);
}

- (void)testResultEmbeddingIsNilForAWrongType
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: @"not a vector" }];
	XCTAssertNil(result.embedding, @"a non-array value does not masquerade as an embedding");
}

- (void)testResultDetectionsReturnsTheArray
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"cat" classIndex:15 confidence:0.9
												  boundingBox:CGRectMake(0.1, 0.2, 0.3, 0.4)];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputDetections: @[ detection ] }];
	XCTAssertEqualObjects(result.detections, @[ detection ]);
}

- (void)testResultAccessorsAreNilWhenAbsent
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: @"opaque" }];
	XCTAssertNil(result.text);
	XCTAssertNil(result.structured);
	XCTAssertNil(result.embedding);
	XCTAssertNil(result.detections);
}

- (void)testResultTextIsNilForAWrongType
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @42 }];
	XCTAssertNil(result.text, @"a non-string value does not masquerade as text");
}

#pragma mark Request

- (void)testRequestPromptAndNegativePrompt
{
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a lighthouse",
												  NFKInputNegativePrompt: @"blurry" }];
	XCTAssertEqualObjects(request.prompt, @"a lighthouse");
	XCTAssertEqualObjects(request.negativePrompt, @"blurry");
	XCTAssertNil(request.messages);
}

- (void)testRequestMessages
{
	NSArray *messages = @[ @{ @"role": @"user", @"content": @"Hi" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }];
	XCTAssertEqualObjects(request.messages, messages);
	XCTAssertNil(request.prompt);
}

- (void)testRequestPromptIsNilForAWrongType
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @[@"not a string"] }];
	XCTAssertNil(request.prompt);
}

@end
