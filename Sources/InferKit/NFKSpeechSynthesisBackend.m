//
//  NFKSpeechSynthesisBackend.m
//  InferKit
//

#import "NFKSpeechSynthesisBackend.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKAudioAsset.h"
#import "NFKErrors.h"
#import <AVFoundation/AVFoundation.h>

NSString * const NFKSpeechSynthesisParameterRate	= @"rate";
NSString * const NFKSpeechSynthesisParameterPitch	= @"pitch";
NSString * const NFKSpeechSynthesisParameterVolume	= @"volume";

/*! What one request speaks with: the backend's properties, overridden by the request's parameters. */
@interface NFKSpeechSynthesisSettings : NSObject
@property (nonatomic, copy, nullable) NSString *voiceIdentifier;
@property (nonatomic, copy, nullable) NSString *language;
@property (nonatomic) double rate;
@property (nonatomic) double pitch;
@property (nonatomic) double volume;
@end

@implementation NFKSpeechSynthesisSettings
@end

@implementation NFKSpeechSynthesisBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_pitch = 1.0;
		_volume = 1.0;
	}
	return self;
}

+ (NSArray<NSString *> *)availableVoices
{
	NSMutableArray<NSString *> *identifiers = [NSMutableArray array];
	for (AVSpeechSynthesisVoice *voice in AVSpeechSynthesisVoice.speechVoices) {
		[identifiers addObject:voice.identifier];
	}
	return identifiers;
}

- (BOOL)isReady
{
	return AVSpeechSynthesisVoice.speechVoices.count > 0;
}

- (NSString *)backendIdentifier
{
	return @"apple-speech-synthesis";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputPrompt];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObject:NFKOutputAudio];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObjects:NFKParameterVoice, NFKParameterSourceLanguage, NFKSpeechSynthesisParameterRate,
			NFKSpeechSynthesisParameterPitch, NFKSpeechSynthesisParameterVolume, nil];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NSString *text = [request inputForKey:NFKInputPrompt];
	if (![text isKindOfClass:NSString.class] || text.length == 0) {
		[self failWithError:outError
					   code:kNFKError_InferenceMissingInput
					 reason:@"no text is set under NFKInputPrompt"];
		return nil;
	}

	NFKSpeechSynthesisSettings *settings = [self settingsForRequest:request error:outError];
	if (settings == nil) {
		return nil;
	}
	AVSpeechSynthesisVoice *voice = [self voiceForSettings:settings];
	if (voice == nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceUnsupported
					 reason:@"the machine has no speech voice for the chosen identifier or language"];
		return nil;
	}

	AVSpeechUtterance *utterance = [AVSpeechUtterance speechUtteranceWithString:text];
	utterance.voice = voice;
	if (settings.rate > 0.0) {
		utterance.rate = (float)settings.rate;
	}
	utterance.pitchMultiplier = (float)settings.pitch;
	utterance.volume = (float)settings.volume;

	NSURL *url = [self temporaryFileURL];
	if (![self writeUtterance:utterance toURL:url error:outError]) {
		return nil;
	}

	NFKAudioAsset *asset = [NFKAudioAsset audioAssetWithFileURL:url];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputAudio: asset }];
}

- (nullable NFKSpeechSynthesisSettings *)settingsForRequest:(NFKInferenceRequest *)request error:(NSError **)outError
{
	NFKSpeechSynthesisSettings *settings = [[NFKSpeechSynthesisSettings alloc] init];
	settings.voiceIdentifier = self.voiceIdentifier;
	settings.language = self.language;
	settings.rate = self.rate;
	settings.pitch = self.pitch;
	settings.volume = self.volume;
	NSDictionary<NSString *, id> *parameters = request.parameters;
	for (NSString *key in @[ NFKParameterVoice, NFKParameterSourceLanguage ]) {
		id value = parameters[key];
		if (value != nil && ![value isKindOfClass:NSString.class]) {
			[self failWithError:outError code:kNFKError_InferenceUnsupported
						 reason:[NSString stringWithFormat:@"\"%@\" is a string", key]];
			return nil;
		}
	}
	if (parameters[NFKParameterVoice] != nil) {
		settings.voiceIdentifier = parameters[NFKParameterVoice];
	}
	if (parameters[NFKParameterSourceLanguage] != nil) {
		settings.language = parameters[NFKParameterSourceLanguage];
	}
	NSDictionary<NSString *, NSArray<NSNumber *> *> *ranges = @{ NFKSpeechSynthesisParameterRate: @[ @0, @1 ],
																 NFKSpeechSynthesisParameterPitch: @[ @0.5, @2 ],
																 NFKSpeechSynthesisParameterVolume: @[ @0, @1 ] };
	for (NSString *key in ranges) {
		id value = parameters[key];
		if (value == nil) {
			continue;
		}
		double number = [value isKindOfClass:NSNumber.class] ? [value doubleValue] : NAN;
		if (!(number >= ranges[key][0].doubleValue && number <= ranges[key][1].doubleValue)) {
			[self failWithError:outError code:kNFKError_InferenceUnsupported
						 reason:[NSString stringWithFormat:@"\"%@\" is a number from %@ to %@", key, ranges[key][0], ranges[key][1]]];
			return nil;
		}
		// Each key names the settings property it overrides.
		[settings setValue:@(number) forKey:key];
	}
	return settings;
}

- (nullable AVSpeechSynthesisVoice *)voiceForSettings:(NFKSpeechSynthesisSettings *)settings
{
	if (settings.voiceIdentifier.length > 0) {
		return [AVSpeechSynthesisVoice voiceWithIdentifier:settings.voiceIdentifier];
	}
	if (settings.language.length > 0) {
		return [AVSpeechSynthesisVoice voiceWithLanguage:settings.language];
	}
	// Writing to buffers needs a named voice. An utterance that leaves the voice unset is spoken
	// aloud with the system default, and through writeUtterance: it produces no buffers at all,
	// reporting nothing (measured on macOS 26.6.2).
	AVSpeechSynthesisVoice *preferred =
		[AVSpeechSynthesisVoice voiceWithLanguage:AVSpeechSynthesisVoice.currentLanguageCode];
	return preferred ?: AVSpeechSynthesisVoice.speechVoices.firstObject;
}

// AVSpeechSynthesizer delivers its buffers on the MAIN run loop, and only there. A secondary thread
// running its own run loop receives nothing, with or without a port attached to it (measured: 83
// callbacks on the main run loop, 0 on a secondary one). So the utterance starts on the main thread
// either way, and how this call waits depends on where the caller is.
- (BOOL)writeUtterance:(AVSpeechUtterance *)utterance toURL:(NSURL *)url error:(NSError **)outError
{
	__block AVAudioFile *file = nil;
	__block NSError *writeError = nil;
	__block BOOL spoken = NO;
	dispatch_semaphore_t finished = dispatch_semaphore_create(0);

	// The synthesizer has to outlive the call that issues the write: released, it stops delivering
	// and the wait reaches its deadline having received nothing.
	AVSpeechSynthesizer *synthesizer = [[AVSpeechSynthesizer alloc] init];
	void (^speak)(void) = ^{
		[synthesizer writeUtterance:utterance toBufferCallback:^(AVAudioBuffer *buffer) {
			AVAudioPCMBuffer *pcm = (AVAudioPCMBuffer *)buffer;
			if (![pcm isKindOfClass:AVAudioPCMBuffer.class]) {
				return;
			}
			// A zero-length buffer is the synthesizer saying it has finished.
			if (pcm.frameLength == 0) {
				spoken = YES;
				dispatch_semaphore_signal(finished);
				return;
			}
			if (file == nil) {
				file = [[AVAudioFile alloc] initForWriting:url
												  settings:pcm.format.settings
											  commonFormat:pcm.format.commonFormat
											   interleaved:pcm.format.isInterleaved
													 error:&writeError];
			}
			if (file != nil && writeError == nil) {
				[file writeFromBuffer:pcm error:&writeError];
			}
		}];
	};

	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:120.0];
	if (NSThread.isMainThread) {
		// Nothing else is pumping the run loop the callback needs, so this call pumps it. A caller on
		// the main thread therefore lets its own timers and sources run while the audio is written.
		speak();
		while (!spoken && deadline.timeIntervalSinceNow > 0.0) {
			[NSRunLoop.currentRunLoop runMode:NSDefaultRunLoopMode
								   beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];
		}
	} else {
		// The contract's caller is here: off the main thread, where waiting leaves the main run loop
		// free to deliver the buffers. A program whose main thread is blocked instead reaches the
		// deadline and reports that no audio arrived.
		dispatch_async(dispatch_get_main_queue(), speak);
		dispatch_semaphore_wait(finished, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(120 * NSEC_PER_SEC)));
	}

	if (writeError != nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceBackendFailure
					 reason:writeError.localizedDescription ?: @"the spoken audio could not be written"];
		return NO;
	}
	if (file == nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceBackendFailure
					 reason:@"the synthesizer produced no audio"];
		return NO;
	}
	return YES;
}

- (NSURL *)temporaryFileURL
{
	NSString *name = [NSString stringWithFormat:@"%@.caf", NSUUID.UUID.UUIDString];
	return [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];
}

- (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
