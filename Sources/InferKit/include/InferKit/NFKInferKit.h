//
//  NFKInferKit.h
//  InferKit
//

#ifndef NFKInferKit_h
#define NFKInferKit_h

#import <Foundation/Foundation.h>

/*!
	@define		NFK_API_VERSION_MAJOR
	@abstract	The API major version of the headers a client compiles against.
	@discussion	A client records the interface it was built against with this macro and compares it with
				@c NFKInferKit.APIMajorVersion of the library it loads. Introduced in InferKit 0.4.0.
*/
#define NFK_API_VERSION_MAJOR 1

/*!
	@define		NFK_API_VERSION_MINOR
	@abstract	The API minor version of the headers a client compiles against.
	@discussion	A client built against minor N runs on a library whose minor version is N or higher.
				Introduced in InferKit 0.4.0.
*/
#define NFK_API_VERSION_MINOR 0

NS_ASSUME_NONNULL_BEGIN

/*!
	@class		NFKInferKit
	@abstract	Library-wide information about the InferKit core.
	@discussion	Introduced in InferKit 0.2.0.
*/
@interface NFKInferKit : NSObject

/*!
	@abstract	The InferKit release version, as a semantic-version string (for example, @c "0.4.0").
	@discussion	The value moves with @c InferKit.podspec and the release git tag. It is named
				@c releaseVersion because @c +[NSObject version] is the integer class version that
				archiving reads, and a class property named @c version would override it with a string.
				Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, readonly, copy) NSString *releaseVersion;

/*!
	@abstract	The API version of this library, as @c "major.minor" (for example, @c "1.0").
	@discussion	The API version counts changes to the binary interface a client binds to, and the
				release version counts releases. Several releases can carry the same API version.

				The promise covers the Objective-C interface and the C ABI: the classes, selectors,
				protocols, constants, enumeration values, functions, and structures the public headers
				declare, in the core and in the @c @objc interface the companions generate. It does not
				cover the Swift interface of InferKitMLX, whose public Swift API exposes mlx-swift types
				that change between mlx-swift releases.

				- An incompatible change raises the major version and resets the minor version to 0.
				  Removing or renaming a declaration, changing a type, a selector, a constant's value, or
				  a structure's layout, and adding a required method to a protocol a client adopts are
				  incompatible.
				- An addition raises the minor version. A new class, method, property, constant,
				  enumeration case, or optional protocol method is an addition.
				- A release that changes neither keeps the API version.

				A client built against @c NFK_API_VERSION_MAJOR and @c NFK_API_VERSION_MINOR runs on a
				library with the same major version and a minor version at least as high. Before the
				first tagged release the API version stays 1.0 while the interface changes, and the
				first tagged release fixes what 1.0 names. Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, readonly, copy) NSString *APIVersion;

/*!
	@abstract	The major component of @c APIVersion.
	@discussion	It changes only on an incompatible change to the Objective-C interface or the C ABI.
				Introduced in InferKit 0.4.0.
*/
@property (class, nonatomic, readonly) NSInteger APIMajorVersion;

/*!
	@abstract	The minor component of @c APIVersion.
	@discussion	It counts the additions since the major version last changed. Introduced in
				InferKit 0.4.0.
*/
@property (class, nonatomic, readonly) NSInteger APIMinorVersion;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferKit_h */
