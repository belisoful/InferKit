//
//  NFKModelAdapterPrivate.h
//  InferKit
//
//  The adapter file reading the memory estimate and option validation share. Not a public header.
//

#ifndef NFKModelAdapterPrivate_h
#define NFKModelAdapterPrivate_h

#import "NFKModelAdapter.h"

NS_ASSUME_NONNULL_BEGIN

/*! YES when path's extension names a pickle file: bin, pt, pth, ckpt, or pkl. */
BOOL NFKModelAdapterIsPicklePath(NSString *path);

/*! The elements and the stored bytes of every tensor in the safetensors file at path. Returns NO and sets
	error, NFKModelAdapterErrorUnreadable, when the file cannot be read as safetensors. A file read once is
	not read again while its size and modification date stay the same. */
BOOL NFKModelAdapterTensorTotals(NSString *path, NSInteger *elementCount, NSInteger *byteCount, NSError **error);

NS_ASSUME_NONNULL_END

#endif /* NFKModelAdapterPrivate_h */
