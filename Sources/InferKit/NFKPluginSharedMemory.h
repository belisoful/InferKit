//
//  NFKPluginSharedMemory.h
//  InferKit
//
//  A memory region two processes share through a descriptor, for the bytes of a plugin message's
//  attachments.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*!
	@class      NFKPluginSharedMemory
	@abstract   A mapped region and the descriptor that names it.
	@discussion A new region is a POSIX shared-memory object: shm_open with a random name and O_EXCL, sized
				once, and unlinked at once, so only descriptors reach it. The system sizes such an object
				once, so a peer cannot shrink it under a mapping. A sandboxed process names its app group
				as the prefix, and a name holds at most 31 characters. When no object can be made the
				region is an unlinked temporary file.

				A received region maps the descriptor's object as it is, writable and shared. A regular
				file, the fallback, is copied instead when the caller asks, because a peer that truncates
				it would fault the mapping. A descriptor that maps only for reading is copied too, so
				every region is writable. Anything else is refused. The region unmaps and closes its
				descriptor when it is released.
*/
@interface NFKPluginSharedMemory : NSObject

/*! A new region whose length is length bytes, filled with zeros, or nil with an error. Its object and
	its mapping are length rounded up to whole pages (mappedLength). prefix is a sandboxed process's app
	group prefix, such as "TEAMID.group.example/", or nil. */
+ (nullable instancetype)regionWithLength:(NSUInteger)length
							   namePrefix:(nullable NSString *)prefix
									error:(NSError * _Nullable *)outError;

/*! The region a received descriptor names, which the region takes ownership of, closing it on failure
	too. Its length is the whole object, an upper bound on what the sender wrote: the system rounds a
	shared-memory object to whole pages. */
+ (nullable instancetype)regionWithDescriptor:(int)descriptor
								  copiesFiles:(BOOL)copiesFiles
										error:(NSError * _Nullable *)outError;

/*! The region a received descriptor names, with length the sender stated, or NSNotFound for the whole
	object. A length beyond the object fails with NFKPluginErrorProtocolViolation. copiesFiles copies a
	regular file's length bytes into private memory and closes the descriptor. */
+ (nullable instancetype)regionWithDescriptor:(int)descriptor
									   length:(NSUInteger)length
								  copiesFiles:(BOOL)copiesFiles
										error:(NSError * _Nullable *)outError;

- (instancetype)init NS_UNAVAILABLE;

/*! The descriptor that names the region, or -1 for a copied file. The region still owns it. */
@property (nonatomic, readonly) int descriptor;

/*! The region's first byte, or NULL for an empty region. mappedLength bytes are addressable. */
@property (nonatomic, readonly, nullable) void *mutableBytes NS_RETURNS_INNER_POINTER;

/*! The region's logical length in bytes: what its creator asked for, or what its receiver was told. */
@property (nonatomic, readonly) NSUInteger length;

/*! The bytes mapped or copied, at least length: whole pages for a shared-memory object. */
@property (nonatomic, readonly) NSUInteger mappedLength;

/*! YES when the descriptor names a POSIX shared-memory object, NO when it names a file. */
@property (nonatomic, readonly, getter=isSharedMemoryObject) BOOL sharedMemoryObject;

/*! YES when the bytes are a private copy: a copied file, or a descriptor that maps only for reading. */
@property (nonatomic, readonly, getter=isCopied) BOOL copied;

/*! The region's length bytes without a copy. The data keeps the region mapped while it lives. */
- (NSData *)data;

/*! The region's first length bytes without a copy, or nil when length exceeds mappedLength. */
- (nullable NSData *)dataOfLength:(NSUInteger)length;

@end

#endif

NS_ASSUME_NONNULL_END
