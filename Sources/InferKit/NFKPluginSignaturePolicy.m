//
//  NFKPluginSignaturePolicy.m
//  InferKit
//

#import "NFKPluginSignaturePolicy.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

@implementation NFKPluginSignaturePolicy

+ (instancetype)defaultPolicy
{
	return [[self alloc] init];
}

+ (instancetype)developmentPolicy
{
	NFKPluginSignaturePolicy *policy = [[self alloc] init];
	policy.allowsAdHocSignature = YES;
	return policy;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requiresValidSignature = YES;
	}
	return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	NFKPluginSignaturePolicy *copy = [[self.class alloc] init];
	copy.requiresValidSignature = self.requiresValidSignature;
	copy.teamIdentifier = self.teamIdentifier;
	copy.allowsAdHocSignature = self.allowsAdHocSignature;
	copy.requirementString = self.requirementString;
	return copy;
}

- (BOOL)validateCodeAtURL:(NSURL *)url error:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

- (BOOL)validateProcessWithAuditToken:(audit_token_t)auditToken error:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

@end

#endif
