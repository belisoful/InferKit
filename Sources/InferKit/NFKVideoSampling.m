//
//  NFKVideoSampling.m
//  InferKit
//

#import <InferKit/NFKVideoSampling.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>
#import <AVFoundation/AVFoundation.h>

@implementation NFKVideoSampling

+ (nullable NSArray *)framesOfVideoAtURL:(NSURL *)url
								   count:(NSUInteger)count
								   error:(NSError * _Nullable *)outError
{
	AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
	CMTime duration = [self durationOfAsset:asset error:outError];
	if (CMTIME_IS_INVALID(duration)) {
		return nil;
	}
	double seconds = CMTimeGetSeconds(duration);
	if (!(seconds > 0) || count == 0) {
		if (outError != NULL) {
			*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceMissingInput reason:@"the clip has no duration to sample"];
		}
		return nil;
	}

	// A sample time that lands exactly on a frame boundary is ambiguous under a zero tolerance and
	// can stall the generator; half a frame either way resolves it to the frame the time falls in.
	double frameRate = [self frameRateOfAsset:asset];
	CMTime halfFrame = CMTimeMakeWithSeconds(0.5 / (frameRate > 0 ? frameRate : 30.0), 600);
	AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
	generator.appliesPreferredTrackTransform = YES;
	generator.requestedTimeToleranceBefore = halfFrame;
	generator.requestedTimeToleranceAfter = halfFrame;

	NSMutableArray *frames = [NSMutableArray array];
	for (NSUInteger index = 0; index < count; index++) {
		// A millisecond past the midpoint keeps a sample inside the frame its midpoint falls in
		// rather than on the edge between two, where a clip whose frames divide the count evenly
		// would land every sample.
		double at = MIN(seconds * ((double)index + 0.5) / (double)count + 0.001, seconds);
		NSError *frameError = nil;
		CMTime time = CMTimeMakeWithSeconds(at, 600);
		CGImageRef image = [generator copyCGImageAtTime:time actualTime:NULL error:&frameError];
		// A hardware decode session can come up broken when the GPU has just been under another
		// load (measured: kVTVideoDecoderMalfunctionErr right after a large model ran on it), and
		// the next session works; one fresh generator is the retry.
		if (image == NULL && [frameError.domain isEqualToString:AVFoundationErrorDomain] && frameError.code == AVErrorDecodeFailed) {
			generator = [[AVAssetImageGenerator alloc] initWithAsset:asset];
			generator.appliesPreferredTrackTransform = YES;
			generator.requestedTimeToleranceBefore = halfFrame;
			generator.requestedTimeToleranceAfter = halfFrame;
			image = [generator copyCGImageAtTime:time actualTime:NULL error:&frameError];
		}
		if (image == NULL) {
			if (frames.count > 0) {
				break;		// a clip shorter than its declared duration yields what it has
			}
			if (outError != NULL) {
				*outError = frameError ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"no frame could be decoded"];
			}
			return nil;
		}
		[frames addObject:(__bridge id)image];
		CGImageRelease(image);
	}
	return frames;
}

// The first video track's nominal rate, or 0 when the clip has none or does not say.
+ (double)frameRateOfAsset:(AVURLAsset *)asset
{
	__block double frameRate = 0;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	[asset loadValuesAsynchronouslyForKeys:@[ @"tracks" ] completionHandler:^{
		AVAssetTrack *track = [asset statusOfValueForKey:@"tracks" error:NULL] == AVKeyValueStatusLoaded
			? [asset tracksWithMediaType:AVMediaTypeVideo].firstObject : nil;
		if (track == nil) {
			dispatch_semaphore_signal(semaphore);
			return;
		}
		[track loadValuesAsynchronouslyForKeys:@[ @"nominalFrameRate" ] completionHandler:^{
			if ([track statusOfValueForKey:@"nominalFrameRate" error:NULL] == AVKeyValueStatusLoaded) {
				frameRate = track.nominalFrameRate;
			}
			dispatch_semaphore_signal(semaphore);
		}];
	}];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
	return frameRate;
}

// The synchronous duration accessor is deprecated; the asynchronous load is waited on, since this
// call blocks by contract and is already off the render thread.
+ (CMTime)durationOfAsset:(AVURLAsset *)asset error:(NSError * _Nullable *)outError
{
	__block CMTime duration = kCMTimeInvalid;
	__block NSError *loadError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
	[asset loadValuesAsynchronouslyForKeys:@[ @"duration" ] completionHandler:^{
		NSError *error = nil;
		if ([asset statusOfValueForKey:@"duration" error:&error] == AVKeyValueStatusLoaded) {
			duration = asset.duration;
		} else {
			loadError = error;
		}
		dispatch_semaphore_signal(semaphore);
	}];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
	if (CMTIME_IS_INVALID(duration) && outError != NULL) {
		*outError = loadError ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceMissingInput reason:@"the clip could not be read"];
	}
	return duration;
}

@end
