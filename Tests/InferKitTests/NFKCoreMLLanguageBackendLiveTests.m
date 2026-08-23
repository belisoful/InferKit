//
//  NFKCoreMLLanguageBackendLiveTests.m
//  NFKTests
//
//  An opt-in end-to-end run against a real converted model directory. It runs only when
//  INFERKIT_TEST_MODEL_DIR points at a directory the inferkit-convert tool produced (for example
//  `Local/tiny`), and is skipped otherwise, so CI stays green with no weights.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKCoreMLLanguageBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>

API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0))
@interface NFKCoreMLLanguageBackendLiveTests : XCTestCase
@end

@implementation NFKCoreMLLanguageBackendLiveTests

- (nullable NSURL *)modelDirectory
{
	NSString *path = NSProcessInfo.processInfo.environment[@"INFERKIT_TEST_MODEL_DIR"];
	return path.length > 0 ? [NSURL fileURLWithPath:path] : nil;
}

- (void)testLoadsAndGeneratesTextEndToEnd
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;		// opt-in: set INFERKIT_TEST_MODEL_DIR to run this
		}

		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		NSError *error = nil;
		XCTAssertTrue([backend prepareWithError:&error], @"prepare failed: %@", error);
		XCTAssertTrue(backend.isReady);

		NFKInferenceRequest *request =
			[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"abc" }
										parameters:@{ NFKParameterMaxTokens: @8, NFKParameterSeed: @1 }
									outputModality:NFKModalityText];
		NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
		XCTAssertNotNil(result, @"generation failed: %@", error);

		id text = [result outputForKey:NFKOutputText];
		XCTAssertTrue([text isKindOfClass:NSString.class]);
		NSLog(@"[live] generated %lu chars: %@", (unsigned long)[text length], text);
	}
}

- (void)testChatMessagesGenerateThroughTheTemplate
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;
		}
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		NSArray *messages = @[ @{ @"role": @"user", @"content": @"Name one color." } ];
		NFKInferenceRequest *request =
			[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
										parameters:@{ NFKParameterMaxTokens: @12, NFKParameterTemperature: @0 }
									outputModality:NFKModalityText];
		NSError *error = nil;
		NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
		XCTAssertNotNil(result, @"chat generation failed: %@", error);
		id text = [result outputForKey:NFKOutputText];
		XCTAssertTrue([text isKindOfClass:NSString.class]);
		NSLog(@"[live] chat reply: %@", text);
	}
}

- (void)testGreedyDecodingIsDeterministic
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;
		}
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		NFKInferenceRequest *request =
			[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }
										parameters:@{ NFKParameterMaxTokens: @8, NFKParameterTemperature: @0 }
									outputModality:NFKModalityText];
		NSError *error = nil;
		NSString *first = [[backend runInferenceForRequest:request error:&error] outputForKey:NFKOutputText];
		NSString *second = [[backend runInferenceForRequest:request error:&error] outputForKey:NFKOutputText];
		XCTAssertNotNil(first, @"%@", error);
		XCTAssertEqualObjects(first, second, @"greedy decoding should repeat");
	}
}

- (void)testStreamingEmitsGrowingPartialText
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;
		}
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		NSError *error = nil;
		XCTAssertTrue([backend prepareWithError:&error], @"%@", error);

		NFKInferenceRequest *request =
			[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"The" }
										parameters:@{ NFKParameterMaxTokens: @6, NFKParameterTemperature: @0 }
									outputModality:NFKModalityText];

		XCTestExpectation *finished = [self expectationWithDescription:@"job finished"];
		__block NSInteger updates = 0;
		__block NSUInteger lastLength = 0;
		__block BOOL monotonic = YES;

		NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
		job.progressHandler = ^(NFKInferenceJob *reporting) {
			NSString *partial = [reporting.partialResult outputForKey:NFKOutputText];
			if (partial != nil) {
				updates += 1;
				if (partial.length < lastLength) {
					monotonic = NO;
				}
				lastLength = partial.length;
			}
		};
		job.completionHandler = ^(NFKInferenceJob *done) { [finished fulfill]; };
		[self waitForExpectations:@[finished] timeout:60];

		XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
		XCTAssertGreaterThan(updates, 0, @"expected streamed partial results");
		XCTAssertTrue(monotonic, @"streamed text should only grow");
		XCTAssertTrue([[job.result outputForKey:NFKOutputText] isKindOfClass:NSString.class]);
		NSLog(@"[live] streamed %ld updates, final: %@", (long)updates, [job.result outputForKey:NFKOutputText]);
	}
}

- (void)testMeasurePromptPrefillTime
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;
		}
		NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
		NSError *error = nil;
		XCTAssertTrue([backend prepareWithError:&error], @"%@", error);

		// A long prompt with 1 output token isolates prompt processing.
		NSMutableString *longPrompt = [NSMutableString string];
		for (NSInteger i = 0; i < 40; i++) {
			[longPrompt appendString:@"The quick brown fox jumps over the lazy dog. "];
		}
		NFKInferenceRequest *request =
			[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: longPrompt }
										parameters:@{ NFKParameterMaxTokens: @1, NFKParameterTemperature: @0 }
									outputModality:NFKModalityText];
		[backend runInferenceForRequest:request error:&error];		// warm the compiled plan
		NSDate *start = [NSDate date];
		NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
		NSTimeInterval elapsed = -[start timeIntervalSinceNow];
		XCTAssertNotNil(result, @"%@", error);
		NSLog(@"[perf] prompt(~400 tokens) + 1 token: %.2fs", elapsed);
	}
}

- (void)testMeasureThroughputAcrossComputeUnits
{
	if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
		NSURL *directory = [self modelDirectory];
		if (directory == nil) {
			return;
		}
		NSArray<NSNumber *> *units = @[ @(MLComputeUnitsAll), @(MLComputeUnitsCPUAndNeuralEngine), @(MLComputeUnitsCPUOnly) ];
		NSArray<NSString *> *names = @[ @"all", @"cpu+ane", @"cpu" ];
		NSInteger count = 20;
		for (NSUInteger i = 0; i < units.count; i++) {
			NFKCoreMLLanguageBackend *backend = [NFKCoreMLLanguageBackend backendWithModelDirectoryURL:directory];
			backend.computeUnits = (MLComputeUnits)units[i].integerValue;
			NSError *error = nil;
			if (![backend prepareWithError:&error]) {
				NSLog(@"[perf] %@: load failed %@", names[i], error);
				continue;
			}
			NFKInferenceRequest *request =
				[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"The quick brown fox" }
											parameters:@{ NFKParameterMaxTokens: @(count), NFKParameterTemperature: @0 }
										outputModality:NFKModalityText];
			NSDate *start = [NSDate date];
			NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
			NSTimeInterval elapsed = -[start timeIntervalSinceNow];
			XCTAssertNotNil(result, @"%@", error);
			NSLog(@"[perf] %@: %ld tokens in %.2fs = %.1f tok/s", names[i], (long)count, elapsed, count / elapsed);
		}
	}
}

@end
