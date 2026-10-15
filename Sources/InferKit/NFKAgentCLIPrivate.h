//
//  NFKAgentCLIPrivate.h
//  InferKit
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*! The directories where coding agents' command-line tools and their runtimes install, which a
	Finder-launched application's PATH lacks: ~/.local/bin, ~/.claude/local, the npm-global, Bun,
	Volta, and Cargo bins, Homebrew's two prefixes, and each nvm Node version's bin, newest first.
	NFKAgentCLI searches them after PATH, and NFKSubprocess searches them for a bare command that
	PATH does not hold. */
NSArray<NSString *> *NFKAgentCLIToolDirectories(void);

#endif

NS_ASSUME_NONNULL_END
