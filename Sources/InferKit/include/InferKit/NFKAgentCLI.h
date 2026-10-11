//
//  NFKAgentCLI.h
//  InferKit
//

#ifndef NFKAgentCLI_h
#define NFKAgentCLI_h

#import <Foundation/Foundation.h>

@class NFKInferenceServer;

NS_ASSUME_NONNULL_BEGIN

/*! The error domain for registering an MCP server with an agent's command-line tool. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKAgentCLIErrorDomain;

/*! The command line, joined with spaces, under an NFKAgentCLIErrorCommandFailed error's userInfo.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKAgentCLICommandErrorKey;

/*! The command's exit status, an NSNumber, under an NFKAgentCLIErrorCommandFailed error's userInfo.
	Introduced in InferKit 0.4.0. */
extern NSString * const NFKAgentCLIExitStatusErrorKey;

/*! What the command wrote to standard output and standard error, under an NFKAgentCLIError error's
	userInfo when it ran. Introduced in InferKit 0.4.0. */
extern NSString * const NFKAgentCLIOutputErrorKey;

/*!
	@enum       NFKAgentCLIError
	@abstract   Error codes in NFKAgentCLIErrorDomain.
	@constant   NFKAgentCLIErrorNotInstalled The tool's executable is not found.
	@constant   NFKAgentCLIErrorSandboxed The process runs in the App Sandbox, which cannot run
				another program's executable.
	@constant   NFKAgentCLIErrorScopeUnsupported The tool has no configuration of that scope.
	@constant   NFKAgentCLIErrorInvalidArgument A name, URL, or environment variable name is not one
				the tool accepts, or the server is not running.
	@constant   NFKAgentCLIErrorCommandFailed The command ran and exited with a failure status.
	@constant   NFKAgentCLIErrorTimedOut The command had not exited by timeout and was terminated.
	@constant   NFKAgentCLIErrorUnreadableOutput The command succeeded and its output names no
				server in a form the tool is known to print.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKAgentCLIErrorDomain, NFKAgentCLIError) {
	NFKAgentCLIErrorNotInstalled = 1,
	NFKAgentCLIErrorSandboxed = 2,
	NFKAgentCLIErrorScopeUnsupported = 3,
	NFKAgentCLIErrorInvalidArgument = 4,
	NFKAgentCLIErrorCommandFailed = 5,
	NFKAgentCLIErrorTimedOut = 6,
	NFKAgentCLIErrorUnreadableOutput = 7,
};

/*!
	@enum       NFKAgentCLIScope
	@abstract   Which of a tool's configurations holds a registration.
	@constant   NFKAgentCLIScopeUser Every project of the user.
	@constant   NFKAgentCLIScopeProject The project at projectDirectoryURL, in a file the project
				shares (Claude Code's .mcp.json, Gemini CLI's and Qwen Code's .gemini/ or .qwen/
				settings.json).
	@constant   NFKAgentCLIScopeLocal The project at projectDirectoryURL, private to the user
				(Claude Code only).
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKAgentCLIScope) {
	NFKAgentCLIScopeUser = 0,
	NFKAgentCLIScopeProject = 1,
	NFKAgentCLIScopeLocal = 2,
};

/*!
	@class      NFKAgentMCPLink
	@abstract   An MCP server registered with an agent's command-line tool, as the tool reports it.
	@discussion Each tool reports a different amount:
				- Claude Code reports the URL, the bearer token's environment variable, and whether
				  the server answered its check.
				- Codex reports the URL and the bearer token's environment variable. It does not
				  connect.
				- Gemini CLI and Qwen Code report the URL and whether the server answered. They do
				  not show headers, so reportsBearerToken is NO.
				Introduced in InferKit 0.4.0.
*/
@interface NFKAgentMCPLink : NSObject

/*! The name the server is registered under. */
@property (nonatomic, readonly, copy) NSString *name;

/*! The server's MCP endpoint. */
@property (nonatomic, readonly, copy) NSURL *URL;

/*! The environment variable whose value the tool sends as the Bearer token, or nil for none. */
@property (nonatomic, readonly, copy, nullable) NSString *bearerTokenEnvironmentVariable;

/*! Whether the tool reports the bearer token's environment variable. When NO,
	bearerTokenEnvironmentVariable is nil whatever the registration holds. */
@property (nonatomic, readonly) BOOL reportsBearerToken;

/*! Whether the server answered the tool's check. NO for a tool that does not check. */
@property (nonatomic, readonly, getter=isConnected) BOOL connected;

/*! Whether the registration reaches URL with that bearer token variable. The variable is compared
	only when reportsBearerToken is YES. */
- (BOOL)isCurrentForURL:(NSURL *)URL bearerTokenEnvironmentVariable:(nullable NSString *)variable
	NS_SWIFT_NAME(isCurrent(for:bearerTokenEnvironmentVariable:));

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKAgentCLI
	@abstract   Registers an MCP server with a coding agent's command-line tool by running the
				tool's own mcp commands.
	@discussion The presets drive the tools that add, remove, and report HTTP MCP servers from the
				command line:
				- claudeCode: claude mcp add / remove / get. Scopes: user, project, local.
				- codex: codex mcp add / remove / get --json. Scope: user.
				- geminiCLI: gemini mcp add / remove / list. Scopes: user, project.
				- qwenCode: qwen mcp add / remove / list. Scopes: user, project.

				linkServer:name:scope:bearerTokenEnvironmentVariable:error: registers a running
				NFKInferenceServer's localMCPURL, so the agent calls the server's tools. A
				registration that is already current is left alone; one that names another URL, as
				after a restart on another port, is replaced.

				The bearer token travels by environment variable name, never by value, so no key
				lands in a command line or a configuration file. The tool reads the variable when it
				connects, so the variable is set in the environment the agent runs in. A server that
				listens on loopback and leaves requiresAPIKeyOnLoopback NO takes no token.

				The executable is found under executableURL when it is set, otherwise on PATH and in
				the directories installers use (~/.local/bin, ~/.claude/local, /opt/homebrew/bin,
				/usr/local/bin, and the npm, Bun, Volta, Cargo, and nvm directories), since an
				application launched from the Finder has a short PATH. Those directories also go on the command's PATH, which a
				Node-based tool needs to find node. Commands block for up to timeout seconds; run
				them off the main thread.

				macOS only. A sandboxed process fails with NFKAgentCLIErrorSandboxed, since the App
				Sandbox does not run another program's executable. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKAgentCLI : NSObject

/*! Anthropic's Claude Code (claude). */
@property (class, nonatomic, readonly) NFKAgentCLI *claudeCode;

/*! OpenAI's Codex CLI (codex). */
@property (class, nonatomic, readonly) NFKAgentCLI *codex;

/*! Google's Gemini CLI (gemini). */
@property (class, nonatomic, readonly) NFKAgentCLI *geminiCLI;

/*! Qwen Code (qwen). */
@property (class, nonatomic, readonly) NFKAgentCLI *qwenCode;

/*! Every preset, in the order above. Each read returns new instances. */
@property (class, nonatomic, readonly, copy) NSArray<NFKAgentCLI *> *allAgents;

/*! The tool's product name, such as "Claude Code". */
@property (nonatomic, readonly, copy) NSString *name;

/*! The executable's file name, such as "claude". */
@property (nonatomic, readonly, copy) NSString *executableName;

/*! The executable to run instead of the one found by executableName. */
@property (nonatomic, copy, nullable) NSURL *executableURL;

/*! The executable a command runs, or nil when none is found. */
@property (nonatomic, readonly, copy, nullable) NSURL *resolvedExecutableURL;

/*! Whether resolvedExecutableURL names an executable. */
@property (nonatomic, readonly, getter=isInstalled) BOOL installed;

/*! The project a project or local registration belongs to, which is the command's working
	directory. Defaults to nil, which is the process's current directory. */
@property (nonatomic, copy, nullable) NSURL *projectDirectoryURL;

/*! How long a command runs before it is terminated, in seconds. Defaults to 60, since a status
	check connects to every registered server. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! Whether the tool has a configuration of that scope. */
- (BOOL)supportsScope:(NFKAgentCLIScope)scope;

/*!
	@abstract   Registers the MCP server at URL under name.
	@discussion A registration under name that already reaches URL with the same token variable is
				left as it is. Any other registration under name in that scope is replaced.
	@param      name The server's name in the tool: letters, digits, '-', and '_'.
	@param      URL The server's http or https MCP endpoint.
	@param      variable The environment variable holding the Bearer token, or nil for none.
	@param      scope The configuration that holds the registration.
	@param      error Set when the command fails.
	@return     YES when the tool holds the registration.
*/
- (BOOL)linkMCPServerNamed:(NSString *)name
					   URL:(NSURL *)URL
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
					 scope:(NFKAgentCLIScope)scope
					 error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(linkMCPServer(named:url:bearerTokenEnvironmentVariable:scope:));

/*!
	@abstract   Registers a running server's MCP endpoint (its localMCPURL) under name.
	@discussion Fails with NFKAgentCLIErrorInvalidArgument when the server is not running, and when
				the server requires a key on loopback (requiresAPIKeyOnLoopback) and variable is nil.
	@param      server The running server whose tools the agent calls.
	@param      name The server's name in the tool.
	@param      scope The configuration that holds the registration.
	@param      variable The environment variable holding the server's apiKey, or nil for none.
	@param      error Set when the server cannot be linked or the command fails.
	@return     YES when the tool holds the registration.
*/
- (BOOL)linkServer:(NFKInferenceServer *)server
			  name:(NSString *)name
			 scope:(NFKAgentCLIScope)scope
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
			 error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(link(_:name:scope:bearerTokenEnvironmentVariable:));

/*!
	@abstract   Removes the registration under name from that scope.
	@param      name The server's name in the tool.
	@param      scope The configuration that holds the registration.
	@param      error Set when the command fails.
	@return     YES when the scope holds no registration under name, including when it held none.
*/
- (BOOL)unlinkMCPServerNamed:(NSString *)name scope:(NFKAgentCLIScope)scope error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(unlinkMCPServer(named:scope:));

/*!
	@abstract   The registration under name as the tool reports it.
	@discussion Claude Code reports the scope of the registration it finds, and one in another scope
				counts as absent. Codex has one scope. Gemini CLI and Qwen Code report the
				registration their project settings resolve to, the project's over the user's.
	@param      name The server's name in the tool.
	@param      scope The configuration to read.
	@param      error Set when the command fails; untouched when there is no registration.
	@return     The registration, or nil with error untouched when the tool holds none under name.
				Swift returns an optional and throws only on a failed command.
*/
- (nullable NFKAgentMCPLink *)MCPLinkNamed:(NSString *)name scope:(NFKAgentCLIScope)scope error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(mcpLink(named:scope:)) __attribute__((swift_error(nonnull_error)));

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKAgentCLI_h */
