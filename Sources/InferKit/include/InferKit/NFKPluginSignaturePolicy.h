//
//  NFKPluginSignaturePolicy.h
//  InferKit
//

#ifndef NFKPluginSignaturePolicy_h
#define NFKPluginSignaturePolicy_h

#import <Foundation/Foundation.h>
#include <mach/message.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKPluginSignaturePolicy
	@abstract   Which code signatures a host accepts from a plugin.
	@discussion The default policy requires a valid signature and refuses an ad hoc one. It pins no team;
				an app that ships its own allowlist sets teamIdentifier or requirementString. The
				development policy accepts ad hoc signatures, for plugins built on the developer's own
				machine.

				validateCodeAtURL:error: checks a bundle or file on disk with every architecture and
				strict validation, and caches a success by the code's cdhash, so a second launch of the
				same code does not hash it again. validateProcessWithAuditToken:error: checks a running
				process, which is how a host confirms that the process on the other end of its channel
				runs the code it verified. The class is macOS only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginSignaturePolicy : NSObject <NSCopying>

/*! Requires a valid signature, refuses ad hoc signatures, and pins no team. */
+ (instancetype)defaultPolicy;

/*! Requires a valid signature and accepts ad hoc signatures. */
+ (instancetype)developmentPolicy;

/*! Whether code without a valid signature is refused. Defaults to YES. */
@property (nonatomic, assign) BOOL requiresValidSignature;

/*! The team whose signature is required, or nil for any team. */
@property (nonatomic, copy, nullable) NSString *teamIdentifier;

/*! Whether an ad hoc signature, which names no signer, is accepted. Defaults to NO. */
@property (nonatomic, assign) BOOL allowsAdHocSignature;

/*! A code requirement in the requirement language the code must also satisfy, or nil. */
@property (nonatomic, copy, nullable) NSString *requirementString;

/*! Checks the code at url. Returns NO with NFKPluginErrorSignatureInvalid or
	NFKPluginErrorTeamMismatch when the policy refuses it. */
- (BOOL)validateCodeAtURL:(NSURL *)url error:(NSError **)error;

/*! Checks the code a running process runs, named by its audit token. */
- (BOOL)validateProcessWithAuditToken:(audit_token_t)auditToken error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginSignaturePolicy_h */
