//
//  NFKSpeechRecognitionBackend.m
//  InferKit
//

#import "NFKSpeechRecognitionBackend.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceJob.h"
#import "NFKAudioAsset.h"
#import "NFKAudioSegment.h"
#import "NFKErrors.h"

// Apple ships no speech recognizer on tvOS, so the class is there and says so rather than the
// header changing shape per platform.
#if TARGET_OS_TV
	#define NFK_HAS_SPEECH_RECOGNITION 0
#else
	#define NFK_HAS_SPEECH_RECOGNITION 1
	#import <Speech/Speech.h>
#endif

@interface NFKSpeechRecognitionBackend ()
@property (nonatomic, copy, nullable) NSLocale *explicitLocale;
@end

@implementation NFKSpeechRecognitionBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithLocale:(NSLocale *)locale
{
	NFKSpeechRecognitionBackend *backend = [[self alloc] init];
	backend.locale = locale;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requiresOnDeviceRecognition = YES;
		_addsPunctuation = YES;
	}
	return self;
}

- (NSLocale *)locale
{
	return self.explicitLocale ?: NSLocale.currentLocale;
}

- (void)setLocale:(nullable NSLocale *)locale
{
	self.explicitLocale = locale;
}

- (NSString *)backendIdentifier
{
	return @"apple-speech";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputAudio];
}

+ (BOOL)isAuthorized
{
#if NFK_HAS_SPEECH_RECOGNITION
	return SFSpeechRecognizer.authorizationStatus == SFSpeechRecognizerAuthorizationStatusAuthorized;
#else
	return NO;
#endif
}

+ (void)requestAuthorizationWithCompletionHandler:(void (^)(BOOL authorized))handler
{
#if NFK_HAS_SPEECH_RECOGNITION
	[SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus status) {
		handler(status == SFSpeechRecognizerAuthorizationStatusAuthorized);
	}];
#else
	handler(NO);
#endif
}

- (BOOL)isReady
{
#if NFK_HAS_SPEECH_RECOGNITION
	if (!self.class.isAuthorized) {
		return NO;
	}
	SFSpeechRecognizer *recognizer = [self recognizer];
	return recognizer != nil && recognizer.isAvailable;
#else
	return NO;
#endif
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NFKInferenceJob *job = [self submitInferenceJobForRequest:request];
	dispatch_semaphore_t finished = dispatch_semaphore_create(0);
	job.completionHandler = ^(NFKInferenceJob *completed) {
		(void)completed;
		dispatch_semaphore_signal(finished);
	};
	dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
	if (job.result != nil) {
		return job.result;
	}
	if (outError != NULL) {
		*outError = job.error ?: [NSError errorWithDomain:NFKInferenceErrorDomain
													 code:kNFKError_InferenceBackendFailure
												 userInfo:@{ NSLocalizedDescriptionKey: @"the transcription produced nothing" }];
	}
	return nil;
}

// Recognition reports partial transcriptions as it goes, which is the job's progress, so the
// asynchronous path is the real one and the synchronous call waits on it.
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
#if NFK_HAS_SPEECH_RECOGNITION
	NSURL *audioURL = [self audioURLForRequest:request];
	if (audioURL == nil) {
		[job finishWithError:[self errorWithCode:kNFKError_InferenceMissingInput
										  reason:@"no audio file is set under NFKInputAudio"]];
		return job;
	}
	if (!self.class.isAuthorized) {
		[job finishWithError:[self errorWithCode:kNFKError_InferenceNotReady
										  reason:@"speech recognition is not authorized; ask with requestAuthorizationWithCompletionHandler:"]];
		return job;
	}
	SFSpeechRecognizer *recognizer = [self recognizer];
	if (recognizer == nil || !recognizer.isAvailable) {
		[job finishWithError:[self errorWithCode:kNFKError_InferenceNotReady
										  reason:@"no recognizer is available for this locale"]];
		return job;
	}

	SFSpeechURLRecognitionRequest *speechRequest = [[SFSpeechURLRecognitionRequest alloc] initWithURL:audioURL];
	speechRequest.requiresOnDeviceRecognition = self.requiresOnDeviceRecognition;
	speechRequest.shouldReportPartialResults = YES;
	if (@available(macOS 13.0, iOS 16.0, *)) {
		speechRequest.addsPunctuation = self.addsPunctuation;
	}

	NFKSpeechRecognitionBackend *backend = self;
	SFSpeechRecognitionTask *task =
		[recognizer recognitionTaskWithRequest:speechRequest
								 resultHandler:^(SFSpeechRecognitionResult * _Nullable result, NSError * _Nullable error) {
		if (error != nil) {
			[job finishWithError:error];
			return;
		}
		if (result == nil) {
			return;
		}
		NFKInferenceResult *mapped = [backend resultFromTranscription:result.bestTranscription];
		if (!result.isFinal) {
			[job reportProgress:-1.0 partialResult:mapped];
			return;
		}
		[job finishWithResult:mapped];
	}];
	job.cancellationHandler = ^{
		[task cancel];
	};
#else
	[job finishWithError:[self errorWithCode:kNFKError_InferenceUnsupported
									  reason:@"Apple ships no speech recognizer on this platform"]];
#endif
	return job;
}

#if NFK_HAS_SPEECH_RECOGNITION

- (nullable SFSpeechRecognizer *)recognizer
{
	return [[SFSpeechRecognizer alloc] initWithLocale:self.locale];
}

// Recognition reads a file, so audio held in memory is written to one the caller never sees.
- (nullable NSURL *)audioURLForRequest:(NFKInferenceRequest *)request
{
	id audio = [request inputForKey:NFKInputAudio];
	if ([audio isKindOfClass:NFKAudioAsset.class]) {
		return ((NFKAudioAsset *)audio).fileURL;
	}
	if ([audio isKindOfClass:NSURL.class]) {
		return audio;
	}
	if ([audio isKindOfClass:NSData.class]) {
		NSString *name = [NSString stringWithFormat:@"%@.wav", NSUUID.UUID.UUIDString];
		NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];
		return [(NSData *)audio writeToURL:url atomically:YES] ? url : nil;
	}
	return nil;
}

- (NFKInferenceResult *)resultFromTranscription:(SFTranscription *)transcription
{
	NSMutableArray<NFKAudioSegment *> *segments = [NSMutableArray arrayWithCapacity:transcription.segments.count];
	for (SFTranscriptionSegment *segment in transcription.segments) {
		[segments addObject:[NFKAudioSegment segmentWithStartSeconds:segment.timestamp
														 endSeconds:segment.timestamp + segment.duration
															  label:segment.substring
														 confidence:segment.confidence]];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: transcription.formattedString,
													NFKOutputSegments: segments }];
}

#endif

- (NSError *)errorWithCode:(NSInteger)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain
							   code:code
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

@end
