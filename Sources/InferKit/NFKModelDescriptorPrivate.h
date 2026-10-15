//
//  NFKModelDescriptorPrivate.h
//  InferKit
//
//  The descriptor reading the catalog shares. Not a public header.
//

#ifndef NFKModelDescriptorPrivate_h
#define NFKModelDescriptorPrivate_h

#import "NFKModelDescriptor.h"
#import "NFKModelCatalog.h"

NS_ASSUME_NONNULL_BEGIN

/*! Sets error to an NFKModelCatalogErrorDomain error with the code and description; returns NO. */
BOOL NFKModelCatalogFail(NFKModelCatalogError code, NSString *reason, NSError **error);

@interface NFKModelDescriptor (NFKModelCatalogReading)

/*! Reads a descriptor, naming fields in an error from path, such as "models[2]". */
+ (nullable instancetype)descriptorFromObject:(id)object path:(NSString *)path error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelDescriptorPrivate_h */
