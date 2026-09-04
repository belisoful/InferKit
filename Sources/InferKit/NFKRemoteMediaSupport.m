//
//  NFKRemoteMediaSupport.m
//  InferKit
//

#import "NFKRemoteMediaSupport.h"
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKVideoSampling.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

static const NSUInteger NFKRemoteDefaultVideoFrameCount = 8;

@interface NFKRemoteAttachments ()
@property (nonatomic, copy, readwrite) NSArray<NSData *> *imagePNGs;
@property (nonatomic, copy, readwrite, nullable) NSData *audioData;
@property (nonatomic, copy, readwrite, nullable) NSString *audioFormat;
@property (nonatomic, copy, readwrite) NSArray<NSDictionary<NSString *, id> *> *documents;
@end

@implementation NFKRemoteAttachments

+ (nullable instancetype)attachmentsForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NFKRemoteAttachments *attachments = [[self alloc] init];

	NSMutableArray<NSData *> *images = [NSMutableArray array];
	NSMutableArray *sources = [NSMutableArray array];
	id first = [request inputForKey:NFKInputImage];
	if (first != nil) {
		[sources addObject:first];
	}
	id more = [request inputForKey:NFKInputImages];
	if ([more isKindOfClass:NSArray.class]) {
		[sources addObjectsFromArray:more];
	}
	for (id image in sources) {
		NSData *png = [NFKImageCoding PNGDataForImage:image];
		if (png == nil) {
			return [self fail:outError reason:@"an image under NFKInputImage or NFKInputImages is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture"];
		}
		[images addObject:png];
	}

	id video = [request inputForKey:NFKInputVideo];
	if (video != nil) {
		NSURL *videoURL = [video isKindOfClass:NFKVideoAsset.class] ? [(NFKVideoAsset *)video fileURL]
			: [video isKindOfClass:NSURL.class] ? video : nil;
		if (videoURL == nil) {
			return [self fail:outError reason:@"the video under NFKInputVideo has no file to read"];
		}
		NSNumber *requested = request.parameters[NFKParameterVideoFrameCount];
		NSUInteger count = [requested isKindOfClass:NSNumber.class] && requested.integerValue > 0
			? (NSUInteger)requested.integerValue : NFKRemoteDefaultVideoFrameCount;
		NSError *sampleError = nil;
		NSArray *frames = [NFKVideoSampling framesOfVideoAtURL:videoURL count:count error:&sampleError];
		if (frames == nil) {
			return [self fail:outError reason:[NSString stringWithFormat:@"the video under NFKInputVideo could not be sampled: %@",
											   sampleError.localizedDescription ?: @"unknown"]];
		}
		for (id frame in frames) {
			NSData *png = [NFKImageCoding PNGDataForImage:frame];
			if (png != nil) {
				[images addObject:png];
			}
		}
	}
	attachments.imagePNGs = images;

	id audio = [request inputForKey:NFKInputAudio];
	if (audio != nil) {
		if ([audio isKindOfClass:NFKAudioAsset.class]) {
			NSURL *fileURL = [(NFKAudioAsset *)audio fileURL];
			NSData *data = fileURL != nil ? [NSData dataWithContentsOfURL:fileURL] : nil;
			if (data == nil) {
				return [self fail:outError reason:@"the audio under NFKInputAudio has no file to read"];
			}
			attachments.audioData = data;
			NSString *extension = fileURL.pathExtension.lowercaseString;
			attachments.audioFormat = extension.length > 0 ? extension : @"wav";
		} else if ([audio isKindOfClass:NSData.class]) {
			attachments.audioData = audio;
			attachments.audioFormat = @"wav";
		} else {
			return [self fail:outError reason:@"the audio under NFKInputAudio is not an NFKAudioAsset or NSData"];
		}
	}

	NSMutableArray<NSDictionary<NSString *, id> *> *documents = [NSMutableArray array];
	NSMutableArray *documentSources = [NSMutableArray array];
	id document = [request inputForKey:NFKInputDocument];
	if (document != nil) {
		[documentSources addObject:document];
	}
	id moreDocuments = [request inputForKey:NFKInputDocuments];
	if ([moreDocuments isKindOfClass:NSArray.class]) {
		[documentSources addObjectsFromArray:moreDocuments];
	}
	for (id source in documentSources) {
		if ([source isKindOfClass:NSURL.class]) {
			NSData *data = [NSData dataWithContentsOfURL:source];
			if (data == nil) {
				return [self fail:outError reason:[NSString stringWithFormat:@"the document at %@ could not be read", [source path]]];
			}
			NSString *name = [(NSURL *)source lastPathComponent];
			[documents addObject:@{ @"data": data, @"filename": name.length > 0 ? name : @"document.pdf" }];
		} else if ([source isKindOfClass:NSData.class]) {
			[documents addObject:@{ @"data": source, @"filename": @"document.pdf" }];
		} else {
			return [self fail:outError reason:@"a document under NFKInputDocument or NFKInputDocuments is not an NSURL or NSData"];
		}
	}
	attachments.documents = documents;
	return attachments;
}

- (BOOL)isEmpty
{
	return self.imagePNGs.count == 0 && self.audioData == nil && self.documents.count == 0;
}

+ (nullable instancetype)fail:(NSError * _Nullable *)outError reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:kNFKError_InferenceMissingInput reason:reason];
	}
	return nil;
}

@end

NSURL * _Nullable NFKRemoteWriteMediaFile(NSData *data, NSString *prefix, NSString *extension,
										  NSURL * _Nullable directory, NSError * _Nullable * _Nullable outError)
{
	NSURL *target = directory
		?: [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES];
	NSError *error = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:target withIntermediateDirectories:YES attributes:nil error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSURL *fileURL = [target URLByAppendingPathComponent:[NSString stringWithFormat:@"%@-%@.%@", prefix, NSUUID.UUID.UUIDString, extension]];
	if (![data writeToURL:fileURL options:NSDataWritingAtomic error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	return fileURL;
}
