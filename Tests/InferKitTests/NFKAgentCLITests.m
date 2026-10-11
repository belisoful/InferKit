//
//  NFKAgentCLITests.m
//  InferKitTests
//
//  Registering an MCP server with an agent's command-line tool. A stub executable stands in for each
//  tool: it records every command line and its working directory, and answers each subcommand with a
//  status and output the test sets in the tool's own format. The live test drives the installed
//  Claude Code against a running server and runs only when INFERKIT_LIVE_AGENT_CLI=1, since it
//  writes to Claude Code's configuration.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <stdlib.h>

@interface NFKAgentCLITests : XCTestCase
@property (nonatomic, copy) NSString *stubDirectory;
@property (nonatomic, copy) NSURL *projectURL;
@end

@implementation NFKAgentCLITests

- (void)setUp
{
	NSString *root = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"NFKAgentCLITests-" stringByAppendingString:NSUUID.UUID.UUIDString]];
	self.stubDirectory = [root stringByAppendingPathComponent:@"stub"];
	self.projectURL = [NSURL fileURLWithPath:[root stringByAppendingPathComponent:@"project"] isDirectory:YES];
	[NSFileManager.defaultManager createDirectoryAtPath:self.stubDirectory withIntermediateDirectories:YES attributes:nil error:NULL];
	[NSFileManager.defaultManager createDirectoryAtURL:self.projectURL withIntermediateDirectories:YES attributes:nil error:NULL];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtPath:self.stubDirectory.stringByDeletingLastPathComponent error:NULL];
}

#pragma mark The stub

/*! The agent with its executable replaced by a stub answering each subcommand (the word after
	"mcp") with @[status, output, optional seconds to sleep first]. */
- (NFKAgentCLI *)stub:(NFKAgentCLI *)agent answers:(NSDictionary<NSString *, NSArray *> *)answers
{
	NSString *script = [NSString stringWithFormat:
		@"#!/bin/sh\n"
		 "dir='%@'\n"
		 "{ pwd -P; for a in \"$@\"; do printf '%%s\\n' \"$a\"; done; printf '%%s\\n' '--end--'; } >> \"$dir/log\"\n"
		 "[ -f \"$dir/$2.sleep\" ] && sleep \"$(cat \"$dir/$2.sleep\")\"\n"
		 "[ -f \"$dir/$2.out\" ] && cat \"$dir/$2.out\"\n"
		 "exit \"$(cat \"$dir/$2.status\" 2>/dev/null || echo 0)\"\n", self.stubDirectory];
	NSString *path = [self.stubDirectory stringByAppendingPathComponent:agent.executableName];
	[script writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:NULL];
	[NSFileManager.defaultManager setAttributes:@{ NSFilePosixPermissions: @0755 } ofItemAtPath:path error:NULL];
	[answers enumerateKeysAndObjectsUsingBlock:^(NSString *subcommand, NSArray *answer, BOOL *stop) {
		NSString *base = [self.stubDirectory stringByAppendingPathComponent:subcommand];
		[[answer[0] stringValue] writeToFile:[base stringByAppendingString:@".status"] atomically:YES encoding:NSUTF8StringEncoding error:NULL];
		[answer[1] writeToFile:[base stringByAppendingString:@".out"] atomically:YES encoding:NSUTF8StringEncoding error:NULL];
		if (answer.count > 2) {
			[[answer[2] stringValue] writeToFile:[base stringByAppendingString:@".sleep"] atomically:YES encoding:NSUTF8StringEncoding error:NULL];
		}
	}];
	agent.executableURL = [NSURL fileURLWithPath:path];
	agent.projectDirectoryURL = self.projectURL;
	return agent;
}

/*! Each command the stub ran, as its arguments, after the working directory it ran in. */
- (NSArray<NSArray<NSString *> *> *)commands
{
	NSString *log = [NSString stringWithContentsOfFile:[self.stubDirectory stringByAppendingPathComponent:@"log"] encoding:NSUTF8StringEncoding error:NULL];
	NSMutableArray *commands = [NSMutableArray array];
	NSMutableArray *current = [NSMutableArray array];
	for (NSString *line in [log componentsSeparatedByString:@"\n"]) {
		if ([line isEqualToString:@"--end--"]) {
			[commands addObject:[current copy]];
			[current removeAllObjects];
		} else if (line.length > 0 || current.count > 0) {
			[current addObject:line];
		}
	}
	return commands;
}

- (NSString *)resolvedProjectPath
{
	char resolved[PATH_MAX];
	return realpath(self.projectURL.path.fileSystemRepresentation, resolved) != NULL ? @(resolved) : self.projectURL.path;
}

- (NSURL *)endpoint
{
	return [NSURL URLWithString:@"http://localhost:8765/mcp"];
}

/*! Claude Code's get output for an HTTP server. */
- (NSString *)claudeGetForURL:(NSString *)URL scope:(NSString *)scope header:(nullable NSString *)header
{
	return [NSString stringWithFormat:@"inferkit:\n  Scope: %@\n  Status: ✓ Connected\n  Type: http\n  URL: %@\n%@\n"
			 "To remove this server, run: claude mcp remove \"inferkit\" -s local\n",
			 scope, URL, header != nil ? [@"  Headers:\n    " stringByAppendingString:header] : @""];
}

#pragma mark The presets

- (void)testThePresetsNameTheirToolsAndScopes
{
	NSArray<NFKAgentCLI *> *agents = NFKAgentCLI.allAgents;
	XCTAssertEqualObjects([agents valueForKey:@"executableName"], (@[ @"claude", @"codex", @"gemini", @"qwen" ]));
	XCTAssertEqualObjects([agents valueForKey:@"name"], (@[ @"Claude Code", @"Codex", @"Gemini CLI", @"Qwen Code" ]));
	XCTAssertTrue([NFKAgentCLI.claudeCode supportsScope:NFKAgentCLIScopeLocal]);
	XCTAssertTrue([NFKAgentCLI.claudeCode supportsScope:NFKAgentCLIScopeProject]);
	XCTAssertTrue([NFKAgentCLI.codex supportsScope:NFKAgentCLIScopeUser]);
	XCTAssertFalse([NFKAgentCLI.codex supportsScope:NFKAgentCLIScopeProject]);
	XCTAssertTrue([NFKAgentCLI.geminiCLI supportsScope:NFKAgentCLIScopeProject]);
	XCTAssertFalse([NFKAgentCLI.qwenCode supportsScope:NFKAgentCLIScopeLocal]);
	XCTAssertEqual(NFKAgentCLI.claudeCode.timeout, 60);
	XCTAssertFalse(NFKAgentCLI.claudeCode == NFKAgentCLI.claudeCode, @"each read is a new instance a caller configures");
}

- (void)testAMissingExecutableIsNotInstalled
{
	NFKAgentCLI *agent = NFKAgentCLI.codex;
	agent.executableURL = [NSURL fileURLWithPath:[self.stubDirectory stringByAppendingPathComponent:@"absent"]];
	XCTAssertFalse(agent.isInstalled);
	XCTAssertNil(agent.resolvedExecutableURL);
	NSError *error = nil;
	XCTAssertFalse([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:nil scope:NFKAgentCLIScopeUser error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorNotInstalled);
}

- (void)testArgumentsAreCheckedBeforeAnyCommandRuns
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.codex answers:@{}];
	NSError *error = nil;
	XCTAssertFalse([agent linkMCPServerNamed:@"in ferkit" URL:self.endpoint bearerTokenEnvironmentVariable:nil scope:NFKAgentCLIScopeUser error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorInvalidArgument);
	XCTAssertFalse([agent linkMCPServerNamed:@"inferkit" URL:[NSURL URLWithString:@"ftp://localhost/mcp"]
			  bearerTokenEnvironmentVariable:nil scope:NFKAgentCLIScopeUser error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorInvalidArgument);
	XCTAssertFalse([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"KEY; rm" scope:NFKAgentCLIScopeUser error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorInvalidArgument);
	XCTAssertFalse([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:nil scope:NFKAgentCLIScopeProject error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorScopeUnsupported);
	XCTAssertFalse([agent unlinkMCPServerNamed:@"inferkit" scope:NFKAgentCLIScopeLocal error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorScopeUnsupported);
	XCTAssertEqual(self.commands.count, 0u);
}

#pragma mark Claude Code

- (void)testClaudeCodeReplacesAnOutdatedRegistration
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @0, [self claudeGetForURL:@"http://localhost:51234/mcp" scope:@"Local config (private to you in this project)" header:nil] ],
		@"remove": @[ @0, @"Removed MCP server inferkit from local config\n" ],
		@"add": @[ @0, @"Added HTTP MCP server inferkit\n" ] }];
	NSError *error = nil;
	XCTAssertTrue([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"
									  scope:NFKAgentCLIScopeLocal error:&error], @"%@", error);
	NSArray<NSArray<NSString *> *> *commands = self.commands;
	XCTAssertEqual(commands.count, 3u);
	XCTAssertEqualObjects(commands[0].firstObject, self.resolvedProjectPath, @"a command runs in the project");
	XCTAssertEqualObjects([commands[0] subarrayWithRange:NSMakeRange(1, 3)], (@[ @"mcp", @"get", @"inferkit" ]));
	XCTAssertEqualObjects([commands[1] subarrayWithRange:NSMakeRange(1, 5)], (@[ @"mcp", @"remove", @"inferkit", @"--scope", @"local" ]));
	XCTAssertEqualObjects([commands[2] subarrayWithRange:NSMakeRange(1, 10)],
						  (@[ @"mcp", @"add", @"--transport", @"http", @"--scope", @"local", @"inferkit", @"http://localhost:8765/mcp",
							  @"--header", @"Authorization: Bearer ${INFERKIT_KEY}" ]),
						  "the token goes by variable reference, which Claude Code expands when it connects");
}

- (void)testClaudeCodeLeavesACurrentRegistrationAlone
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @0, [self claudeGetForURL:self.endpoint.absoluteString scope:@"User config (available in all your projects)"
									  header:@"Authorization: Bearer ${INFERKIT_KEY}"] ] }];
	NSError *error = nil;
	XCTAssertTrue([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"
									  scope:NFKAgentCLIScopeUser error:&error], @"%@", error);
	XCTAssertEqual(self.commands.count, 1u, @"only the check ran");
}

- (void)testClaudeCodeReadsItsRegistration
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @0, [self claudeGetForURL:self.endpoint.absoluteString scope:@"Project config (shared via .mcp.json)"
									  header:@"Authorization: Bearer ${INFERKIT_KEY}"] ] }];
	NSError *error = nil;
	NFKAgentMCPLink *link = [agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeProject error:&error];
	XCTAssertNotNil(link, @"%@", error);
	XCTAssertEqualObjects(link.URL, self.endpoint);
	XCTAssertEqualObjects(link.bearerTokenEnvironmentVariable, @"INFERKIT_KEY");
	XCTAssertTrue(link.reportsBearerToken);
	XCTAssertTrue(link.isConnected);
	XCTAssertTrue([link isCurrentForURL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"]);
	XCTAssertFalse([link isCurrentForURL:self.endpoint bearerTokenEnvironmentVariable:nil]);
	XCTAssertFalse([link isCurrentForURL:[NSURL URLWithString:@"http://localhost:9/mcp"] bearerTokenEnvironmentVariable:@"INFERKIT_KEY"]);

	error = nil;
	XCTAssertNil([agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeUser error:&error], @"a registration in another scope is not this one");
	XCTAssertNil(error);
}

- (void)testClaudeCodeTreatsALiteralTokenAsOutdated
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @0, [self claudeGetForURL:self.endpoint.absoluteString scope:@"Local config" header:@"Authorization: Bearer sk-literal"] ] }];
	NFKAgentMCPLink *link = [agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeLocal error:NULL];
	XCTAssertNil(link.bearerTokenEnvironmentVariable);
	XCTAssertFalse([link isCurrentForURL:self.endpoint bearerTokenEnvironmentVariable:nil]);
}

- (void)testAnAbsentRegistrationReadsAsNilAndUnlinksCleanly
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @1, @"No MCP server found with name: inferkit\n" ],
		@"remove": @[ @1, @"No MCP server found with name: \"inferkit\"\n" ] }];
	NSError *error = nil;
	XCTAssertNil([agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeLocal error:&error]);
	XCTAssertNil(error, @"absence is no failure");
	XCTAssertTrue([agent unlinkMCPServerNamed:@"inferkit" scope:NFKAgentCLIScopeLocal error:&error], @"%@", error);
}

- (void)testAFailedCommandCarriesItsCommandLineAndOutput
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{
		@"get": @[ @1, @"No MCP server found with name: inferkit\n" ],
		@"add": @[ @2, @"Invalid transport\n" ] }];
	NSError *error = nil;
	XCTAssertFalse([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:nil scope:NFKAgentCLIScopeLocal error:&error]);
	XCTAssertEqualObjects(error.domain, NFKAgentCLIErrorDomain);
	XCTAssertEqual(error.code, NFKAgentCLIErrorCommandFailed);
	XCTAssertEqualObjects(error.userInfo[NFKAgentCLIExitStatusErrorKey], @2);
	XCTAssertTrue([error.userInfo[NFKAgentCLICommandErrorKey] hasPrefix:@"claude mcp add"], @"%@", error.userInfo);
	XCTAssertTrue([error.userInfo[NFKAgentCLIOutputErrorKey] containsString:@"Invalid transport"]);
	XCTAssertTrue([error.localizedDescription containsString:@"Invalid transport"], @"%@", error.localizedDescription);
}

- (void)testACommandThatOutlivesTheTimeoutIsTerminated
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.claudeCode answers:@{ @"get": @[ @0, @"", @30 ] }];
	agent.timeout = 0.5;
	NSDate *started = [NSDate date];
	NSError *error = nil;
	XCTAssertNil([agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeLocal error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorTimedOut);
	XCTAssertLessThan([NSDate.date timeIntervalSinceDate:started], 8, @"the wait ends even while the stub's sleep holds its pipes");
}

#pragma mark Codex

- (void)testCodexAddsWithItsTokenVariableFlag
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.codex answers:@{
		@"get": @[ @1, @"Error: No MCP server named 'inferkit' found.\n" ],
		@"add": @[ @0, @"Added global MCP server 'inferkit'.\n" ] }];
	NSError *error = nil;
	XCTAssertTrue([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"
									  scope:NFKAgentCLIScopeUser error:&error], @"%@", error);
	NSArray<NSArray<NSString *> *> *commands = self.commands;
	XCTAssertEqual(commands.count, 2u, @"Codex overwrites on add, so nothing is removed first");
	XCTAssertEqualObjects([commands[0] subarrayWithRange:NSMakeRange(1, 4)], (@[ @"mcp", @"get", @"inferkit", @"--json" ]));
	XCTAssertEqualObjects([commands[1] subarrayWithRange:NSMakeRange(1, 7)],
						  (@[ @"mcp", @"add", @"inferkit", @"--url", @"http://localhost:8765/mcp", @"--bearer-token-env-var", @"INFERKIT_KEY" ]));
}

- (void)testCodexReadsItsJSON
{
	NSString *json = @"{\"name\":\"inferkit\",\"enabled\":true,\"transport\":{\"type\":\"streamable_http\",\"url\":\"http://localhost:8765/mcp\","
					  "\"bearer_token_env_var\":\"INFERKIT_KEY\",\"http_headers\":null,\"env_http_headers\":null}}";
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.codex answers:@{ @"get": @[ @0, json ] }];
	NSError *error = nil;
	NFKAgentMCPLink *link = [agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeUser error:&error];
	XCTAssertEqualObjects(link.URL, self.endpoint, @"%@", error);
	XCTAssertEqualObjects(link.bearerTokenEnvironmentVariable, @"INFERKIT_KEY");
	XCTAssertFalse(link.isConnected, @"Codex does not connect to check");
	XCTAssertTrue([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"
									  scope:NFKAgentCLIScopeUser error:&error], @"%@", error);
	XCTAssertEqual(self.commands.count, 2u, @"a current registration is not added again");
}

#pragma mark Gemini CLI and Qwen Code

- (void)testGeminiCLIAndQwenCodeAddInTheirScope
{
	for (NFKAgentCLI *preset in @[ NFKAgentCLI.geminiCLI, NFKAgentCLI.qwenCode ]) {
		[NSFileManager.defaultManager removeItemAtPath:[self.stubDirectory stringByAppendingPathComponent:@"log"] error:NULL];
		NFKAgentCLI *agent = [self stub:preset answers:@{ @"add": @[ @0, @"MCP server \"inferkit\" added to project settings. (http)\n" ] }];
		NSError *error = nil;
		XCTAssertTrue([agent linkMCPServerNamed:@"inferkit" URL:self.endpoint bearerTokenEnvironmentVariable:@"INFERKIT_KEY"
										  scope:NFKAgentCLIScopeProject error:&error], @"%@ %@", preset.name, error);
		NSArray<NSArray<NSString *> *> *commands = self.commands;
		XCTAssertEqual(commands.count, 1u, @"%@ overwrites on add and shows no headers, so nothing is checked first", preset.name);
		XCTAssertEqualObjects([commands[0] subarrayWithRange:NSMakeRange(1, 10)],
							  (@[ @"mcp", @"add", @"--scope", @"project", @"--transport", @"http", @"inferkit", @"http://localhost:8765/mcp",
								  @"--header", @"Authorization: Bearer ${INFERKIT_KEY}" ]), @"%@", preset.name);

		XCTAssertTrue([agent unlinkMCPServerNamed:@"inferkit" scope:NFKAgentCLIScopeUser error:&error], @"%@", error);
		XCTAssertEqualObjects([self.commands[1] subarrayWithRange:NSMakeRange(1, 5)],
							  (@[ @"mcp", @"remove", @"--scope", @"user", @"inferkit" ]), @"%@", preset.name);
	}
}

- (void)testGeminiCLIReadsItsList
{
	NSString *list = @"Configured MCP servers:\n\n"
					  "\x1B[32m✓\x1B[0m local-files: npx files-server --root . (stdio) - Connected\n"
					  "\x1B[31m✗\x1B[0m inferkit: http://localhost:8765/mcp (http) - Disconnected\n"
					  "\x1B[32m✓\x1B[0m search (from web-tools): https://search.example.com/mcp (http) - Connected\n";
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.geminiCLI answers:@{ @"list": @[ @0, list ] }];
	NSError *error = nil;
	NFKAgentMCPLink *link = [agent MCPLinkNamed:@"inferkit" scope:NFKAgentCLIScopeUser error:&error];
	XCTAssertEqualObjects(link.URL, self.endpoint, @"%@", error);
	XCTAssertFalse(link.isConnected);
	XCTAssertFalse(link.reportsBearerToken);
	XCTAssertTrue([link isCurrentForURL:self.endpoint bearerTokenEnvironmentVariable:@"ANY"], @"a tool that shows no headers is compared by URL");

	NFKAgentMCPLink *extension = [agent MCPLinkNamed:@"search" scope:NFKAgentCLIScopeUser error:&error];
	XCTAssertEqualObjects(extension.URL.absoluteString, @"https://search.example.com/mcp");
	XCTAssertTrue(extension.isConnected);

	XCTAssertNil([agent MCPLinkNamed:@"absent" scope:NFKAgentCLIScopeUser error:&error]);
	error = nil;
	XCTAssertNil([agent MCPLinkNamed:@"local-files" scope:NFKAgentCLIScopeUser error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorUnreadableOutput, @"a stdio server under the name is no HTTP registration");
}

#pragma mark A running server

- (void)testLinkingAServerRegistersItsMCPEndpoint
{
	NFKAgentCLI *agent = [self stub:NFKAgentCLI.codex answers:@{ @"get": @[ @1, @"Error: No MCP server named 'inferkit' found.\n" ] }];
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	NSError *error = nil;
	XCTAssertFalse([agent linkServer:server name:@"inferkit" scope:NFKAgentCLIScopeUser bearerTokenEnvironmentVariable:nil error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorInvalidArgument, @"a stopped server has no endpoint");

	XCTAssertTrue([server startWithError:&error], @"%@", error);
	XCTAssertTrue([agent linkServer:server name:@"inferkit" scope:NFKAgentCLIScopeUser bearerTokenEnvironmentVariable:nil error:&error], @"%@", error);
	XCTAssertEqualObjects(self.commands.lastObject[5], server.localMCPURL.absoluteString);
	[server stop];

	NFKInferenceServer *keyed = [[NFKInferenceServer alloc] init];
	keyed.port = 0;
	keyed.loopbackOnly = YES;
	keyed.apiKey = @"secret";
	keyed.requiresAPIKeyOnLoopback = YES;
	XCTAssertTrue([keyed startWithError:&error], @"%@", error);
	XCTAssertFalse([agent linkServer:keyed name:@"inferkit" scope:NFKAgentCLIScopeUser bearerTokenEnvironmentVariable:nil error:&error]);
	XCTAssertEqual(error.code, NFKAgentCLIErrorInvalidArgument, @"a server that wants a key on loopback needs a token variable");
	[keyed stop];
}

#pragma mark The installed Claude Code

- (void)testTheInstalledClaudeCodeConnectsToARunningServer
{
	if (![NSProcessInfo.processInfo.environment[@"INFERKIT_LIVE_AGENT_CLI"] isEqualToString:@"1"]) {
		XCTSkip(@"set INFERKIT_LIVE_AGENT_CLI=1 to register a server with the installed Claude Code");
	}
	NFKAgentCLI *claude = NFKAgentCLI.claudeCode;
	if (!claude.isInstalled) {
		XCTSkip(@"Claude Code is not installed");
	}
	claude.projectDirectoryURL = self.projectURL;
	NSString *variable = @"INFERKIT_AGENT_CLI_TEST_KEY";
	NSString *name = [@"inferkit-test-" stringByAppendingString:[NSUUID.UUID.UUIDString substringToIndex:8].lowercaseString];

	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	server.apiKey = NSUUID.UUID.UUIDString;
	server.requiresAPIKeyOnLoopback = YES;
	[server addTool:[NFKInferenceTool toolWithName:@"echo" description:@"Echoes its text." parameters:@{ @"type": @"object" }
										  handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		return [NFKToolResult resultWithText:@"echo"];
	}]];
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	setenv(variable.UTF8String, server.apiKey.UTF8String, 1);
	[self addTeardownBlock:^{
		[claude unlinkMCPServerNamed:name scope:NFKAgentCLIScopeLocal error:NULL];
		unsetenv(variable.UTF8String);
		[server stop];
	}];

	XCTAssertTrue([claude linkServer:server name:name scope:NFKAgentCLIScopeLocal bearerTokenEnvironmentVariable:variable error:&error], @"%@", error);
	NFKAgentMCPLink *link = [claude MCPLinkNamed:name scope:NFKAgentCLIScopeLocal error:&error];
	XCTAssertTrue([link isCurrentForURL:server.localMCPURL bearerTokenEnvironmentVariable:variable], @"%@ %@", link, error);
	XCTAssertTrue(link.isConnected, @"Claude Code expanded the variable and the server accepted its key: %@", link);
	XCTAssertTrue([claude linkServer:server name:name scope:NFKAgentCLIScopeLocal bearerTokenEnvironmentVariable:variable error:&error], @"%@", error);

	setenv(variable.UTF8String, "the-wrong-key", 1);
	XCTAssertFalse([claude MCPLinkNamed:name scope:NFKAgentCLIScopeLocal error:&error].isConnected, @"the server refuses another key");

	XCTAssertTrue([claude unlinkMCPServerNamed:name scope:NFKAgentCLIScopeLocal error:&error], @"%@", error);
	error = nil;
	XCTAssertNil([claude MCPLinkNamed:name scope:NFKAgentCLIScopeLocal error:&error]);
	XCTAssertNil(error);
}

@end
