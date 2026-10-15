//
//  NFKPluginService.m
//  InferKit
//

#import "NFKPluginService.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

@implementation NFKPluginService

- (instancetype)initWithPlugin:(id<NFKPlugin>)plugin manifest:(NFKPluginManifest *)manifest
{
	self = [super init];
	if (self != nil) {
		_plugin = plugin;
		_manifest = [manifest copy];
	}
	return self;
}

- (BOOL)runWithChannelDescriptor:(int)descriptor error:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

@end

int NFKPluginHostMain(int argc, char **argv)
{
	@autoreleasepool {
		NSError *error = nil;
		NFKPluginSetNotImplemented(&error, @selector(runWithChannelDescriptor:error:));
		fprintf(stderr, "inferkit-plugin-host: %s\n", error.localizedDescription.UTF8String);
		return 70;
	}
}

#endif
