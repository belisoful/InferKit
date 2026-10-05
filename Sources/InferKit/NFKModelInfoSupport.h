//
//  NFKModelInfoSupport.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

/*! The name NFKModelInfoComputeUnits reports for a compute-units setting. */
NSString *NFKModelInfoComputeUnitsName(MLComputeUnits computeUnits);

/*! The bytes a file, or every file under a directory, occupies on disk; nil when url is nil or
	unreadable. */
NSNumber * _Nullable NFKModelInfoStorageBytesAtURL(NSURL * _Nullable url);

/*! A Core ML model's description: the compute units it loaded with, the version it declares, and the
	storage size of the compiled model at compiledURL. */
NSDictionary<NSString *, id> *NFKModelInfoForCoreMLModel(MLModel *model, NSURL * _Nullable compiledURL);

NS_ASSUME_NONNULL_END
