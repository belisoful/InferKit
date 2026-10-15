/*
 *  NFKAPIVersion.h
 *  InferKit
 *
 *  The API version of the headers a client compiles against. Pure C, so the Objective-C headers, the
 *  C API, and a plugin all read the same numbers.
 */

#ifndef NFKAPIVersion_h
#define NFKAPIVersion_h

/*!
	@define		NFK_API_VERSION_MAJOR
	@abstract	The API major version of the headers a client compiles against.
	@discussion	A client records the interface it was built against with this macro and compares it with
				@c NFKInferKit.APIMajorVersion, or @c nfk_api_version(), of the library it loads.
				Introduced in InferKit 0.4.0.
*/
#define NFK_API_VERSION_MAJOR 1

/*!
	@define		NFK_API_VERSION_MINOR
	@abstract	The API minor version of the headers a client compiles against.
	@discussion	A client built against minor N runs on a library whose minor version is N or higher.
				Introduced in InferKit 0.4.0.
*/
#define NFK_API_VERSION_MINOR 0

/*!
	@define		NFK_API_VERSION
	@abstract	The API version as one number: the major version in the high 16 bits and the minor version
				in the low 16 bits.
	@discussion	Introduced in InferKit 0.4.0.
*/
#define NFK_API_VERSION ((NFK_API_VERSION_MAJOR << 16) | NFK_API_VERSION_MINOR)

#endif /* NFKAPIVersion_h */
