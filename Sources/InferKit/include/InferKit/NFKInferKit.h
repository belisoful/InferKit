//
//  NFKInferKit.h
//  InferKit
//

#ifndef NFKInferKit_h
#define NFKInferKit_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class		NFKInferKit
	@abstract	Library-wide information about the InferKit core.
	@discussion	Introduced in InferKit 0.2.0.
*/
@interface NFKInferKit : NSObject

/*!
	@abstract	The InferKit core version, as a semantic-version string (for example, @c "0.2.0").
	@discussion	This reports the CORE library's version. The optional MLX and Foundation Models
				companion packages are versioned separately. The value is kept in sync with
				@c InferKit.podspec and the release git tag by the release process.
*/
@property (class, nonatomic, readonly, copy) NSString *version;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferKit_h */
