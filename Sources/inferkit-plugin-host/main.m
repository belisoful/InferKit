//
//  main.m
//  inferkit-plugin-host
//
//  The process that serves one plugin bundle: inferkit-plugin-host --bundle <path> [--team-id <team>]
//  [--allow-ad-hoc]. NFKPluginHostMain is the whole program.
//

#import <TargetConditionals.h>

#if TARGET_OS_OSX

#import <InferKit/NFKPluginService.h>

int main(int argc, char *argv[])
{
	return NFKPluginHostMain(argc, argv);
}

#else

#include <stdio.h>

int main(int argc, char *argv[])
{
	fputs("inferkit-plugin-host runs on macOS only\n", stderr);
	return 1;
}

#endif
