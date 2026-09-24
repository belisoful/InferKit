//
//  NFKVisionRectangleBackend.m
//  InferKit
//

#import "NFKVisionRectangleBackend.h"
#import "NFKVisionSupport.h"
#import "NFKInferenceKeys.h"
#import "NFKDetection.h"
#import "NFKQuadrilateral.h"
#import "NFKErrors.h"
#import <Vision/Vision.h>

@implementation NFKVisionRectangleBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithKind:(NFKVisionRectangleKind)kind
{
	NFKVisionRectangleBackend *backend = [[self alloc] init];
	backend.kind = kind;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_maximumCount = 16;
		_minimumSize = 0.2;
		_quadratureTolerance = 30.0;
	}
	return self;
}

+ (NSArray<NSString *> *)supportedSymbologies
{
	VNDetectBarcodesRequest *request = [[VNDetectBarcodesRequest alloc] init];
	NSError *error = nil;
	return [request supportedSymbologiesAndReturnError:&error] ?: @[];
}

- (BOOL)isReady
{
	if (self.kind == NFKVisionRectangleKindDocument) {
		if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
			return YES;
		}
		return NO;
	}
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"vision-rectangle";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputImage];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	if (![self isReady]) {
		[NFKVisionSupport failWithError:outError
								   code:kNFKError_InferenceUnsupported
								 reason:@"document segmentation needs a newer OS than the one running"];
		return nil;
	}

	CGImageRef image = [NFKVisionSupport imageForRequest:request key:NFKInputImage error:outError];
	if (image == NULL) {
		return nil;
	}

	VNImageBasedRequest *visionRequest = [self makeRequest];
	BOOL ran = [NFKVisionSupport performRequests:@[ visionRequest ] onImage:image handler:NULL error:outError];
	CGImageRelease(image);
	if (!ran) {
		return nil;
	}

	NSMutableArray<NFKDetection *> *detections = [NSMutableArray array];
	for (VNRectangleObservation *observation in visionRequest.results) {
		[detections addObject:[NFKDetection detectionWithLabel:[self labelForObservation:observation]
													classIndex:detections.count
													confidence:observation.confidence
												 quadrilateral:[self quadrilateralForObservation:observation]]];
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputDetections: detections }];
}

- (VNImageBasedRequest *)makeRequest
{
	switch (self.kind) {
		case NFKVisionRectangleKindBarcode: {
			VNDetectBarcodesRequest *barcodes = [[VNDetectBarcodesRequest alloc] init];
			if (self.symbologies.count > 0) {
				barcodes.symbologies = self.symbologies;
			}
			return barcodes;
		}
		case NFKVisionRectangleKindDocument:
			if (@available(macOS 12.0, iOS 15.0, tvOS 15.0, *)) {
				return [[VNDetectDocumentSegmentationRequest alloc] init];
			}
			break;
		case NFKVisionRectangleKindRectangle: {
			VNDetectRectanglesRequest *rectangles = [[VNDetectRectanglesRequest alloc] init];
			rectangles.maximumObservations = (NSUInteger)MAX(self.maximumCount, 1);
			rectangles.minimumSize = (float)self.minimumSize;
			rectangles.quadratureTolerance = (float)self.quadratureTolerance;
			return rectangles;
		}
	}
	return [[VNDetectRectanglesRequest alloc] init];
}

// A barcode carries a payload, which is the useful label. The other kinds locate a shape and have
// nothing to name it with.
- (nullable NSString *)labelForObservation:(VNRectangleObservation *)observation
{
	if ([observation isKindOfClass:VNBarcodeObservation.class]) {
		return ((VNBarcodeObservation *)observation).payloadStringValue;
	}
	return nil;
}

- (NFKQuadrilateral *)quadrilateralForObservation:(VNRectangleObservation *)observation
{
	return [NFKQuadrilateral quadrilateralWithTopLeft:[NFKVisionSupport contractPoint:observation.topLeft]
											 topRight:[NFKVisionSupport contractPoint:observation.topRight]
										   bottomLeft:[NFKVisionSupport contractPoint:observation.bottomLeft]
										  bottomRight:[NFKVisionSupport contractPoint:observation.bottomRight]];
}

@end
