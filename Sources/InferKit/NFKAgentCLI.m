//
//  NFKAgentCLI.m
//  InferKit
//

#import "NFKAgentCLI.h"
#import "NFKInferenceServer.h"

#import <signal.h>

NSString * const NFKAgentCLIErrorDomain = @"NFKAgentCLIErrorDomain";
NSString * const NFKAgentCLICommandErrorKey = @"NFKAgentCLICommand";
NSString * const NFKAgentCLIExitStatusErrorKey = @"NFKAgentCLIExitStatus";
NSString * const NFKAgentCLIOutputErrorKey = @"NFKAgentCLIOutput";

static BOOL NFKAgentCLIFail(NSError **error, NFKAgentCLIError code, NSString *message, NSDictionary *details)
{
	if (error != NULL) {
		NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithDictionary:details ?: @{}];
		userInfo[NSLocalizedDescriptionKey] = message;
		*error = [NSError errorWithDomain:NFKAgentCLIErrorDomain code:code userInfo:userInfo];
	}
	return NO;
}

static BOOL NFKAgentCLIMatches(NSString *pattern, NSString *string)
{
	return [string rangeOfString:pattern options:NSRegularExpressionSearch].location != NSNotFound;
}

@interface NFKAgentMCPLink ()

/*! A Bearer token the tool sends that is no environment variable reference, which no request matches. */
@property (nonatomic, readonly) BOOL sendsOtherAuthorization;

- (instancetype)initWithName:(NSString *)name
						 URL:(NSURL *)URL
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
		  reportsBearerToken:(BOOL)reportsBearerToken
	 sendsOtherAuthorization:(BOOL)sendsOtherAuthorization
				   connected:(BOOL)connected NS_DESIGNATED_INITIALIZER;

@end

@implementation NFKAgentMCPLink

- (instancetype)initWithName:(NSString *)name
						 URL:(NSURL *)URL
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
		  reportsBearerToken:(BOOL)reportsBearerToken
	 sendsOtherAuthorization:(BOOL)sendsOtherAuthorization
				   connected:(BOOL)connected
{
	self = [super init];
	if (self != nil) {
		_name = [name copy];
		_URL = [URL copy];
		_bearerTokenEnvironmentVariable = reportsBearerToken ? [variable copy] : nil;
		_reportsBearerToken = reportsBearerToken;
		_sendsOtherAuthorization = sendsOtherAuthorization;
		_connected = connected;
	}
	return self;
}

- (BOOL)isCurrentForURL:(NSURL *)URL bearerTokenEnvironmentVariable:(nullable NSString *)variable
{
	if (![self.URL.absoluteString isEqualToString:URL.absoluteString]) {
		return NO;
	}
	if (!self.reportsBearerToken) {
		return YES;
	}
	if (self.sendsOtherAuthorization) {
		return NO;
	}
	return (self.bearerTokenEnvironmentVariable == nil && variable == nil) || [self.bearerTokenEnvironmentVariable isEqualToString:variable];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@%@%@>", self.class, self.name, self.URL.absoluteString,
			self.bearerTokenEnvironmentVariable != nil ? [@" $" stringByAppendingString:self.bearerTokenEnvironmentVariable] : @"",
			self.isConnected ? @" connected" : @""];
}

@end

#if TARGET_OS_OSX

typedef NS_ENUM(NSInteger, NFKAgentCLIKind) {
	NFKAgentCLIKindClaudeCode,
	NFKAgentCLIKindCodex,
	NFKAgentCLIKindGeminiCLI,
	NFKAgentCLIKindQwenCode,
};

/*! One finished command: its exit status and what it wrote. */
@interface NFKAgentCLIRun : NSObject
@property (nonatomic, assign) int status;
@property (nonatomic, copy) NSString *standardOutput;
@property (nonatomic, copy) NSString *standardError;
@property (nonatomic, readonly, copy) NSString *output;
@end

@implementation NFKAgentCLIRun

- (NSString *)output
{
	NSString *joined = [@[ self.standardOutput, self.standardError ] componentsJoinedByString:@"\n"];
	NSRegularExpression *escapes = [NSRegularExpression regularExpressionWithPattern:@"\\x1B\\[[0-9;?]*[A-Za-z]" options:0 error:NULL];
	return [escapes stringByReplacingMatchesInString:joined options:0 range:NSMakeRange(0, joined.length) withTemplate:@""];
}

@end

/*! A pipe's bytes, appended from its readability handler's thread. */
@interface NFKAgentCLIPipeReader : NSObject
@property (nonatomic, readonly) dispatch_group_t finished;
- (instancetype)initWithPipe:(NSPipe *)pipe;
- (NSString *)text;
- (void)stop;
@end

@implementation NFKAgentCLIPipeReader
{
	NSPipe *_pipe;
	NSMutableData *_data;
	NSLock *_lock;
	BOOL _ended;
}

- (instancetype)initWithPipe:(NSPipe *)pipe
{
	self = [super init];
	if (self != nil) {
		_pipe = pipe;
		_data = [NSMutableData data];
		_lock = [[NSLock alloc] init];
		_finished = dispatch_group_create();
		dispatch_group_enter(_finished);
		__weak NFKAgentCLIPipeReader *weakSelf = self;
		pipe.fileHandleForReading.readabilityHandler = ^(NSFileHandle *handle) {
			[weakSelf appendData:handle.availableData];
		};
	}
	return self;
}

- (void)appendData:(NSData *)data
{
	if (data.length == 0) {
		[self stop];
		return;
	}
	[_lock lock];
	[_data appendData:data];
	[_lock unlock];
}

- (void)stop
{
	[_lock lock];
	BOOL ended = _ended;
	_ended = YES;
	[_lock unlock];
	if (ended) {
		return;
	}
	_pipe.fileHandleForReading.readabilityHandler = nil;
	dispatch_group_leave(_finished);
}

- (NSString *)text
{
	[_lock lock];
	NSString *text = [[NSString alloc] initWithData:_data encoding:NSUTF8StringEncoding] ?: @"";
	[_lock unlock];
	return text;
}

@end

@implementation NFKAgentCLI
{
	NFKAgentCLIKind _kind;
}

#pragma mark Presets

- (instancetype)initWithKind:(NFKAgentCLIKind)kind name:(NSString *)name executableName:(NSString *)executableName
{
	self = [super init];
	if (self != nil) {
		_kind = kind;
		_name = [name copy];
		_executableName = [executableName copy];
		_timeout = 60;
	}
	return self;
}

+ (NFKAgentCLI *)claudeCode
{
	return [[self alloc] initWithKind:NFKAgentCLIKindClaudeCode name:@"Claude Code" executableName:@"claude"];
}

+ (NFKAgentCLI *)codex
{
	return [[self alloc] initWithKind:NFKAgentCLIKindCodex name:@"Codex" executableName:@"codex"];
}

+ (NFKAgentCLI *)geminiCLI
{
	return [[self alloc] initWithKind:NFKAgentCLIKindGeminiCLI name:@"Gemini CLI" executableName:@"gemini"];
}

+ (NFKAgentCLI *)qwenCode
{
	return [[self alloc] initWithKind:NFKAgentCLIKindQwenCode name:@"Qwen Code" executableName:@"qwen"];
}

+ (NSArray<NFKAgentCLI *> *)allAgents
{
	return @[ self.claudeCode, self.codex, self.geminiCLI, self.qwenCode ];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ (%@)>", self.class, self.name, self.resolvedExecutableURL.path ?: self.executableName];
}

- (BOOL)supportsScope:(NFKAgentCLIScope)scope
{
	switch (_kind) {
		case NFKAgentCLIKindClaudeCode:
			return scope == NFKAgentCLIScopeUser || scope == NFKAgentCLIScopeProject || scope == NFKAgentCLIScopeLocal;
		case NFKAgentCLIKindCodex:
			return scope == NFKAgentCLIScopeUser;
		case NFKAgentCLIKindGeminiCLI:
		case NFKAgentCLIKindQwenCode:
			return scope == NFKAgentCLIScopeUser || scope == NFKAgentCLIScopeProject;
	}
	return NO;
}

#pragma mark Finding the executable

// A Finder-launched application's PATH is /usr/bin:/bin:/usr/sbin:/sbin, which holds none of these.
- (NSArray<NSString *> *)searchDirectories
{
	NSString *home = NSHomeDirectory();
	NSMutableArray<NSString *> *directories = [NSMutableArray array];
	NSString *path = NSProcessInfo.processInfo.environment[@"PATH"];
	for (NSString *entry in [path componentsSeparatedByString:@":"]) {
		if (entry.length > 0) {
			[directories addObject:entry];
		}
	}
	for (NSString *relative in @[ @".local/bin", @".claude/local", @".npm-global/bin", @".bun/bin", @".volta/bin", @".cargo/bin" ]) {
		[directories addObject:[home stringByAppendingPathComponent:relative]];
	}
	[directories addObjectsFromArray:@[ @"/opt/homebrew/bin", @"/usr/local/bin" ]];
	NSString *nvm = [home stringByAppendingPathComponent:@".nvm/versions/node"];
	NSArray<NSString *> *versions = [[NSFileManager.defaultManager contentsOfDirectoryAtPath:nvm error:NULL]
									 sortedArrayUsingSelector:@selector(localizedStandardCompare:)];
	for (NSString *version in versions.reverseObjectEnumerator) {
		[directories addObject:[[nvm stringByAppendingPathComponent:version] stringByAppendingPathComponent:@"bin"]];
	}
	return [NSOrderedSet orderedSetWithArray:directories].array;
}

- (nullable NSURL *)resolvedExecutableURL
{
	NSFileManager *files = NSFileManager.defaultManager;
	if (self.executableURL != nil) {
		return [files isExecutableFileAtPath:self.executableURL.path] ? self.executableURL : nil;
	}
	for (NSString *directory in self.searchDirectories) {
		NSString *candidate = [directory stringByAppendingPathComponent:self.executableName];
		BOOL isDirectory = NO;
		if ([files fileExistsAtPath:candidate isDirectory:&isDirectory] && !isDirectory && [files isExecutableFileAtPath:candidate]) {
			return [NSURL fileURLWithPath:candidate];
		}
	}
	return nil;
}

- (BOOL)isInstalled
{
	return self.resolvedExecutableURL != nil;
}

#pragma mark Running a command

- (nullable NFKAgentCLIRun *)runArguments:(NSArray<NSString *> *)arguments error:(NSError **)error
{
	NSDictionary<NSString *, NSString *> *environment = NSProcessInfo.processInfo.environment;
	if (environment[@"APP_SANDBOX_CONTAINER_ID"] != nil) {
		NFKAgentCLIFail(error, NFKAgentCLIErrorSandboxed, @"the App Sandbox does not run another program's executable", nil);
		return nil;
	}
	NSURL *executable = self.resolvedExecutableURL;
	if (executable == nil) {
		NFKAgentCLIFail(error, NFKAgentCLIErrorNotInstalled,
						[NSString stringWithFormat:@"%@ (%@) is not installed", self.name, self.executableName], nil);
		return nil;
	}
	NSString *command = [[@[ executable.path ] arrayByAddingObjectsFromArray:arguments] componentsJoinedByString:@" "];

	NSMutableArray<NSString *> *path = [NSMutableArray arrayWithObject:executable.URLByDeletingLastPathComponent.path];
	[path addObjectsFromArray:self.searchDirectories];
	NSMutableDictionary<NSString *, NSString *> *childEnvironment = [environment mutableCopy];
	childEnvironment[@"PATH"] = [[NSOrderedSet orderedSetWithArray:path].array componentsJoinedByString:@":"];
	childEnvironment[@"NO_COLOR"] = @"1";

	NSTask *task = [[NSTask alloc] init];
	task.executableURL = executable;
	task.arguments = arguments;
	task.environment = childEnvironment;
	if (self.projectDirectoryURL != nil) {
		task.currentDirectoryURL = self.projectDirectoryURL;
	}
	NSPipe *outputPipe = [NSPipe pipe];
	NSPipe *errorPipe = [NSPipe pipe];
	task.standardOutput = outputPipe;
	task.standardError = errorPipe;
	task.standardInput = [NSFileHandle fileHandleWithNullDevice];
	NFKAgentCLIPipeReader *outputReader = [[NFKAgentCLIPipeReader alloc] initWithPipe:outputPipe];
	NFKAgentCLIPipeReader *errorReader = [[NFKAgentCLIPipeReader alloc] initWithPipe:errorPipe];
	dispatch_semaphore_t exited = dispatch_semaphore_create(0);
	task.terminationHandler = ^(NSTask *finished) {
		dispatch_semaphore_signal(exited);
	};

	NSError *launchError = nil;
	if (![task launchAndReturnError:&launchError]) {
		[outputReader stop];
		[errorReader stop];
		NFKAgentCLIFail(error, NFKAgentCLIErrorNotInstalled, [@"could not run " stringByAppendingString:command],
						launchError != nil ? @{ NSUnderlyingErrorKey: launchError, NFKAgentCLICommandErrorKey: command }
										   : @{ NFKAgentCLICommandErrorKey: command });
		return nil;
	}
	BOOL timedOut = dispatch_semaphore_wait(exited, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(self.timeout * NSEC_PER_SEC))) != 0;
	if (timedOut) {
		[task terminate];
		if (dispatch_semaphore_wait(exited, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC)) != 0) {
			kill(task.processIdentifier, SIGKILL);
			dispatch_semaphore_wait(exited, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
		}
	}
	// A child the command started can hold the pipes open after it exits; the read stops regardless.
	dispatch_group_wait(outputReader.finished, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
	dispatch_group_wait(errorReader.finished, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
	[outputReader stop];
	[errorReader stop];

	NFKAgentCLIRun *run = [[NFKAgentCLIRun alloc] init];
	run.standardOutput = outputReader.text;
	run.standardError = errorReader.text;
	run.status = timedOut ? -1 : task.terminationStatus;
	if (timedOut) {
		NFKAgentCLIFail(error, NFKAgentCLIErrorTimedOut,
						[NSString stringWithFormat:@"%@ did not exit within %.0f seconds", command, self.timeout],
						@{ NFKAgentCLICommandErrorKey: command, NFKAgentCLIOutputErrorKey: run.output });
		return nil;
	}
	return run;
}

- (BOOL)failRun:(NFKAgentCLIRun *)run arguments:(NSArray<NSString *> *)arguments error:(NSError **)error
{
	NSString *command = [[@[ self.executableName ] arrayByAddingObjectsFromArray:arguments] componentsJoinedByString:@" "];
	NSString *output = [run.output stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
	return NFKAgentCLIFail(error, NFKAgentCLIErrorCommandFailed,
						   [NSString stringWithFormat:@"%@ exited with status %d%@", command, run.status,
							output.length > 0 ? [@": " stringByAppendingString:output] : @""],
						   @{ NFKAgentCLICommandErrorKey: command, NFKAgentCLIExitStatusErrorKey: @(run.status),
							  NFKAgentCLIOutputErrorKey: run.output });
}

#pragma mark Commands

+ (NSString *)scopeArgument:(NFKAgentCLIScope)scope
{
	switch (scope) {
		case NFKAgentCLIScopeUser:
			return @"user";
		case NFKAgentCLIScopeProject:
			return @"project";
		case NFKAgentCLIScopeLocal:
			return @"local";
	}
	return @"user";
}

- (NSArray<NSString *> *)addArgumentsForName:(NSString *)name URL:(NSURL *)URL variable:(nullable NSString *)variable scope:(NFKAgentCLIScope)scope
{
	NSString *scopeName = [self.class scopeArgument:scope];
	NSString *header = variable != nil ? [NSString stringWithFormat:@"Authorization: Bearer ${%@}", variable] : nil;
	// The header goes last: Claude Code's and yargs' repeatable header options take the words after them.
	NSMutableArray<NSString *> *arguments = nil;
	switch (_kind) {
		case NFKAgentCLIKindClaudeCode:
			arguments = [NSMutableArray arrayWithArray:@[ @"mcp", @"add", @"--transport", @"http", @"--scope", scopeName, name, URL.absoluteString ]];
			break;
		case NFKAgentCLIKindCodex:
			arguments = [NSMutableArray arrayWithArray:@[ @"mcp", @"add", name, @"--url", URL.absoluteString ]];
			if (variable != nil) {
				[arguments addObjectsFromArray:@[ @"--bearer-token-env-var", variable ]];
			}
			return arguments;
		case NFKAgentCLIKindGeminiCLI:
		case NFKAgentCLIKindQwenCode:
			arguments = [NSMutableArray arrayWithArray:@[ @"mcp", @"add", @"--scope", scopeName, @"--transport", @"http", name, URL.absoluteString ]];
			break;
	}
	if (header != nil) {
		[arguments addObjectsFromArray:@[ @"--header", header ]];
	}
	return arguments;
}

- (NSArray<NSString *> *)removeArgumentsForName:(NSString *)name scope:(NFKAgentCLIScope)scope
{
	switch (_kind) {
		case NFKAgentCLIKindClaudeCode:
			return @[ @"mcp", @"remove", name, @"--scope", [self.class scopeArgument:scope] ];
		case NFKAgentCLIKindCodex:
			return @[ @"mcp", @"remove", name ];
		case NFKAgentCLIKindGeminiCLI:
		case NFKAgentCLIKindQwenCode:
			return @[ @"mcp", @"remove", @"--scope", [self.class scopeArgument:scope], name ];
	}
	return @[];
}

- (NSArray<NSString *> *)statusArgumentsForName:(NSString *)name
{
	switch (_kind) {
		case NFKAgentCLIKindClaudeCode:
			return @[ @"mcp", @"get", name ];
		case NFKAgentCLIKindCodex:
			return @[ @"mcp", @"get", name, @"--json" ];
		case NFKAgentCLIKindGeminiCLI:
		case NFKAgentCLIKindQwenCode:
			return @[ @"mcp", @"list" ];
	}
	return @[];
}

#pragma mark Checking arguments

- (BOOL)checkName:(NSString *)name scope:(NFKAgentCLIScope)scope error:(NSError **)error
{
	if (!NFKAgentCLIMatches(@"^[A-Za-z0-9_-]+$", name)) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorInvalidArgument,
							   [NSString stringWithFormat:@"\"%@\" is not a server name (letters, digits, '-', '_')", name], nil);
	}
	if (![self supportsScope:scope]) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorScopeUnsupported,
							   [NSString stringWithFormat:@"%@ has no %@ configuration", self.name, [self.class scopeArgument:scope]], nil);
	}
	return YES;
}

- (BOOL)checkURL:(NSURL *)URL variable:(nullable NSString *)variable error:(NSError **)error
{
	NSString *scheme = URL.scheme.lowercaseString;
	if (!([scheme isEqualToString:@"http"] || [scheme isEqualToString:@"https"]) || URL.host.length == 0) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorInvalidArgument,
							   [NSString stringWithFormat:@"%@ is not an http or https URL", URL.absoluteString], nil);
	}
	if (variable != nil && !NFKAgentCLIMatches(@"^[A-Za-z_][A-Za-z0-9_]*$", variable)) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorInvalidArgument,
							   [NSString stringWithFormat:@"\"%@\" is not an environment variable name", variable], nil);
	}
	return YES;
}

#pragma mark Linking

- (BOOL)linkMCPServerNamed:(NSString *)name
					   URL:(NSURL *)URL
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
					 scope:(NFKAgentCLIScope)scope
					 error:(NSError **)error
{
	if (![self checkName:name scope:scope error:error] || ![self checkURL:URL variable:variable error:error]) {
		return NO;
	}
	// Gemini CLI and Qwen Code overwrite on add and cannot show a header, so their check would only cost a connection to every server.
	BOOL reportsBearerToken = _kind == NFKAgentCLIKindClaudeCode || _kind == NFKAgentCLIKindCodex;
	if (reportsBearerToken) {
		NSError *statusError = nil;
		NFKAgentMCPLink *existing = [self MCPLinkNamed:name scope:scope error:&statusError];
		if (statusError != nil && statusError.code != NFKAgentCLIErrorUnreadableOutput) {
			if (error != NULL) {
				*error = statusError;
			}
			return NO;
		}
		if ([existing isCurrentForURL:URL bearerTokenEnvironmentVariable:variable]) {
			return YES;
		}
	}
	// Claude Code refuses to add a name its scope already holds.
	if (_kind == NFKAgentCLIKindClaudeCode && ![self unlinkMCPServerNamed:name scope:scope error:error]) {
		return NO;
	}
	NSArray<NSString *> *arguments = [self addArgumentsForName:name URL:URL variable:variable scope:scope];
	NFKAgentCLIRun *run = [self runArguments:arguments error:error];
	if (run == nil) {
		return NO;
	}
	if (run.status != 0) {
		return [self failRun:run arguments:arguments error:error];
	}
	return YES;
}

- (BOOL)linkServer:(NFKInferenceServer *)server
			  name:(NSString *)name
			 scope:(NFKAgentCLIScope)scope
	bearerTokenEnvironmentVariable:(nullable NSString *)variable
			 error:(NSError **)error
{
	NSURL *URL = server.localMCPURL;
	if (URL == nil) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorInvalidArgument, @"the server is not running", nil);
	}
	if (server.requiresAPIKeyOnLoopback && variable == nil) {
		return NFKAgentCLIFail(error, NFKAgentCLIErrorInvalidArgument,
							   @"the server requires a key on loopback, and no bearer token environment variable is named", nil);
	}
	return [self linkMCPServerNamed:name URL:URL bearerTokenEnvironmentVariable:variable scope:scope error:error];
}

- (BOOL)unlinkMCPServerNamed:(NSString *)name scope:(NFKAgentCLIScope)scope error:(NSError **)error
{
	if (![self checkName:name scope:scope error:error]) {
		return NO;
	}
	NSArray<NSString *> *arguments = [self removeArgumentsForName:name scope:scope];
	NFKAgentCLIRun *run = [self runArguments:arguments error:error];
	if (run == nil) {
		return NO;
	}
	if (run.status != 0 && !NFKAgentCLIMatches(@"(?i)no mcp server (found|named)|not found", run.output)) {
		return [self failRun:run arguments:arguments error:error];
	}
	return YES;
}

#pragma mark Reading a registration

- (nullable NFKAgentMCPLink *)MCPLinkNamed:(NSString *)name scope:(NFKAgentCLIScope)scope error:(NSError **)error
{
	if (![self checkName:name scope:scope error:error]) {
		return nil;
	}
	NSArray<NSString *> *arguments = [self statusArgumentsForName:name];
	NFKAgentCLIRun *run = [self runArguments:arguments error:error];
	if (run == nil) {
		return nil;
	}
	if (run.status != 0) {
		if (_kind != NFKAgentCLIKindGeminiCLI && _kind != NFKAgentCLIKindQwenCode
			&& NFKAgentCLIMatches(@"(?i)no mcp server (found|named)", run.output)) {
			return nil;
		}
		[self failRun:run arguments:arguments error:error];
		return nil;
	}
	switch (_kind) {
		case NFKAgentCLIKindClaudeCode:
			return [self claudeLinkNamed:name scope:scope run:run error:error];
		case NFKAgentCLIKindCodex:
			return [self codexLinkNamed:name run:run error:error];
		case NFKAgentCLIKindGeminiCLI:
		case NFKAgentCLIKindQwenCode:
			return [self listedLinkNamed:name run:run error:error];
	}
	return nil;
}

- (nullable NFKAgentMCPLink *)unreadable:(NFKAgentCLIRun *)run name:(NSString *)name error:(NSError **)error
{
	NFKAgentCLIFail(error, NFKAgentCLIErrorUnreadableOutput,
					[NSString stringWithFormat:@"%@ reports %@ in a form other than an HTTP server's", self.name, name],
					@{ NFKAgentCLIOutputErrorKey: run.output });
	return nil;
}

/*! Claude Code's get prints indented "Field: value" lines, the headers indented under "Headers:". */
- (nullable NFKAgentMCPLink *)claudeLinkNamed:(NSString *)name scope:(NFKAgentCLIScope)scope run:(NFKAgentCLIRun *)run error:(NSError **)error
{
	NSString *scopeLine = nil;
	NSString *URLText = nil;
	NSString *status = nil;
	NSString *authorization = nil;
	for (NSString *rawLine in [run.output componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]) {
		NSString *line = [rawLine stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		if ([line hasPrefix:@"Scope:"]) {
			scopeLine = [[line substringFromIndex:6] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet].lowercaseString;
		} else if ([line hasPrefix:@"URL:"]) {
			URLText = [[line substringFromIndex:4] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		} else if ([line hasPrefix:@"Status:"]) {
			status = line;
		} else if ([line.lowercaseString hasPrefix:@"authorization:"]) {
			authorization = [[line substringFromIndex:14] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		}
	}
	// "Scope: Local config (private to you in this project)"; a registration in another scope is not this one.
	if (scopeLine != nil && ![scopeLine hasPrefix:[self.class scopeArgument:scope]]) {
		return nil;
	}
	NSURL *URL = URLText.length > 0 ? [NSURL URLWithString:URLText] : nil;
	if (URL == nil) {
		return [self unreadable:run name:name error:error];
	}
	NSString *variable = nil;
	BOOL other = NO;
	if (authorization != nil) {
		NSRegularExpression *reference = [NSRegularExpression regularExpressionWithPattern:@"^Bearer\\s+\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}$"
																				   options:NSRegularExpressionCaseInsensitive error:NULL];
		NSTextCheckingResult *match = [reference firstMatchInString:authorization options:0 range:NSMakeRange(0, authorization.length)];
		variable = match != nil ? [authorization substringWithRange:[match rangeAtIndex:1]] : nil;
		other = match == nil;
	}
	BOOL connected = status != nil && [status rangeOfString:@"✓"].location != NSNotFound;
	return [[NFKAgentMCPLink alloc] initWithName:name URL:URL bearerTokenEnvironmentVariable:variable
							  reportsBearerToken:YES sendsOtherAuthorization:other connected:connected];
}

/*! Codex's get --json: {"name", "transport": {"type": "streamable_http", "url", "bearer_token_env_var", "http_headers"}}. */
- (nullable NFKAgentMCPLink *)codexLinkNamed:(NSString *)name run:(NFKAgentCLIRun *)run error:(NSError **)error
{
	NSData *data = [run.standardOutput dataUsingEncoding:NSUTF8StringEncoding];
	id object = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
	NSDictionary *transport = [object isKindOfClass:NSDictionary.class] ? object[@"transport"] : nil;
	if (![transport isKindOfClass:NSDictionary.class] || ![transport[@"type"] isEqual:@"streamable_http"]
		|| ![transport[@"url"] isKindOfClass:NSString.class] || [NSURL URLWithString:transport[@"url"]] == nil) {
		return [self unreadable:run name:name error:error];
	}
	NSString *variable = [transport[@"bearer_token_env_var"] isKindOfClass:NSString.class] ? transport[@"bearer_token_env_var"] : nil;
	BOOL other = NO;
	for (id headers in @[ transport[@"http_headers"] ?: NSNull.null, transport[@"env_http_headers"] ?: NSNull.null ]) {
		if ([headers isKindOfClass:NSDictionary.class]) {
			for (NSString *field in (NSDictionary *)headers) {
				other = other || [field caseInsensitiveCompare:@"Authorization"] == NSOrderedSame;
			}
		}
	}
	return [[NFKAgentMCPLink alloc] initWithName:name URL:[NSURL URLWithString:transport[@"url"]] bearerTokenEnvironmentVariable:variable
							  reportsBearerToken:YES sendsOtherAuthorization:other connected:NO];
}

/*! Gemini CLI's and Qwen Code's list: "<icon> <name>[ (from <extension>)]: <address> (<transport>) - <status>". */
- (nullable NFKAgentMCPLink *)listedLinkNamed:(NSString *)name run:(NFKAgentCLIRun *)run error:(NSError **)error
{
	NSRegularExpression *entry = [NSRegularExpression regularExpressionWithPattern:
								  @"^\\s*\\S+\\s+([A-Za-z0-9_-]+)(?: \\(from [^)]*\\))?: (.*) \\((http|sse|stdio)\\) - (.*)$"
																		   options:0 error:NULL];
	for (NSString *line in [run.output componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet]) {
		NSTextCheckingResult *match = [entry firstMatchInString:line options:0 range:NSMakeRange(0, line.length)];
		if (match == nil || ![[line substringWithRange:[match rangeAtIndex:1]] isEqualToString:name]) {
			continue;
		}
		NSURL *URL = [NSURL URLWithString:[line substringWithRange:[match rangeAtIndex:2]]];
		if (![[line substringWithRange:[match rangeAtIndex:3]] isEqualToString:@"http"] || URL == nil) {
			return [self unreadable:run name:name error:error];
		}
		BOOL connected = [[line substringWithRange:[match rangeAtIndex:4]] hasPrefix:@"Connected"];
		return [[NFKAgentMCPLink alloc] initWithName:name URL:URL bearerTokenEnvironmentVariable:nil
								  reportsBearerToken:NO sendsOtherAuthorization:NO connected:connected];
	}
	return nil;
}

@end

#endif
