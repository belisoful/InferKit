//
//  NFK_ARC.h
//  XPC Service
//
//  Created by ~ ~ on 3/19/24.
//
//		#import "NFK_ARC.h"

#ifndef NFK_ARC_h
#define NFK_ARC_h

//

#if __has_feature(objc_arc)

	#define NARC_RETAIN(obj)				(obj)
	#define NARC_AUTORELEASE(obj)			(obj)
	#define NARC_RETAIN_AUTORELEASE(obj)	(obj)
	// A true no-op, not `obj = nil`: ARC releases ivars in dealloc itself, and nil-ing one that backs a
	// `nonnull` property is what the static analyzer flags as NullPassedToNonnull. Every other use is a
	// local (ARC releases it at scope end) or is immediately reassigned, so nothing changes but the
	// warning. The `(void)(obj)` form keeps the argument referenced so no unused-variable warning
	// replaces it.
	#define NARC_RELEASE(obj)				((void)(obj))
	#define NARC_RELEASE_RAW(obj)
	#define SUPER_DEALLOC()
	#define BLOCK_COPY(block)   			[(block) copy]
	#define BLOCK_RELEASE(block)   			((block) = nil)

#else

	#define NARC_RETAIN(obj)				[(obj) retain]
	#define NARC_AUTORELEASE(obj)			[(obj) autorelease]
	#define NARC_RETAIN_AUTORELEASE(obj)	[[(obj) retain] autorelease]
	#define NARC_RELEASE(obj)				([(obj) release], obj = nil)
	#define NARC_RELEASE_RAW(obj)			[(obj) release]
	#define SUPER_DEALLOC() 				[super dealloc]
	#define BLOCK_COPY(block)   			Block_copy(block)
	#define BLOCK_RELEASE(block)   { if (block) { Block_release(block); block = nil; } }

#endif


#endif	//	BPS_ARC_h
