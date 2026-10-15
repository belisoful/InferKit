//
//  NFKMCPStdioTests.m
//  InferKitTests
//
//  The MCP client over stdio. The fake server is a POSIX shell
//  script the test writes to a temporary directory and runs as `/bin/sh <script> <mode>`: /bin/sh,
//  /usr/bin/sed, /bin/sleep, and /bin/pwd ship with every macOS, so the tests need no interpreter,
//  helper executable, or network. The script matches each request line by substring and answers with
//  printf, one JSON-RPC message per line. Its mode makes it ignore the end of its input, or SIGTERM
//  too, for the stop escalation.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import <signal.h>
#import <stdlib.h>

static NSString * const NFKFakeMCPServerScript =
	@"mode=\"$1\"\n"
	 "if [ \"$mode\" = ignore-sigterm ]; then trap '' TERM; fi\n"
	 "echo 'fake MCP server starting' >&2\n"
	 "while IFS= read -r line; do\n"
	 "  id=$(printf '%s\\n' \"$line\" | /usr/bin/sed -n 's/.*\"id\":\\([0-9][0-9]*\\).*/\\1/p')\n"
	 "  case \"$line\" in\n"
	 "  *'\"method\":\"initialize\"'*)\n"
	 "    printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{\"tools\":{}},"
	         "\"serverInfo\":{\"name\":\"fake-stdio\",\"version\":\"1.0\"},\"instructions\":\"A fake server for tests.\"}}\\n' \"$id\" ;;\n"
	 "  *'\"method\":\"notifications/'*) ;;\n"
	 "  *'\"method\":\"ping\"'*)\n"
	 "    printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{}}\\n' \"$id\" ;;\n"
	 "  *'\"method\":\"tools/list\"'*)\n"
	 "    case \"$line\" in\n"
	 "    *'\"cursor\":\"page-2\"'*)\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"tools\":[{\"name\":\"slow\",\"inputSchema\":{\"type\":\"object\"}},"
	           "{\"name\":\"ask_back\",\"inputSchema\":{\"type\":\"object\"}}]}}\\n' \"$id\" ;;\n"
	 "    *)\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"tools\":[{\"name\":\"where\",\"description\":\"Says where the server runs.\","
	           "\"inputSchema\":{\"type\":\"object\"}},{\"name\":\"log\",\"inputSchema\":{\"type\":\"object\"}},"
	           "{\"name\":\"crash\",\"inputSchema\":{\"type\":\"object\"}}],\"nextCursor\":\"page-2\"}}\\n' \"$id\" ;;\n"
	 "    esac ;;\n"
	 "  *'\"method\":\"tools/call\"'*)\n"
	 "    case \"$line\" in\n"
	 "    *'\"name\":\"where\"'*)\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"%s in %s, home %s, secret %s\"}]}}\\n' "
	           "\"$id\" \"${FAKE_MCP_GREETING:-unset}\" \"$(/bin/pwd -P)\" \"${HOME:-unset}\" \"${NFK_MCP_TEST_SECRET:-unset}\" ;;\n"
	 "    *'\"name\":\"log\"'*)\n"
	 "      echo 'log line one' >&2\n"
	 "      echo 'log line two' >&2\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"logged\"}]}}\\n' \"$id\" ;;\n"
	 "    *'\"name\":\"crash\"'*)\n"
	 "      echo 'about to crash' >&2\n"
	 "      exit 3 ;;\n"
	 "    *'\"name\":\"slow\"'*)\n"
	 "      /bin/sleep 2\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"late\"}]}}\\n' \"$id\" ;;\n"
	 "    *'\"name\":\"ask_back\"'*)\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":\"s1\",\"method\":\"ping\"}\\n'\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":\"s2\",\"method\":\"sampling/createMessage\",\"params\":{\"messages\":[],\"maxTokens\":1}}\\n'\n"
	 "      seen=''\n"
	 "      for n in 1 2; do\n"
	 "        IFS= read -r answer\n"
	 "        case \"$answer\" in\n"
	 "        *'\"id\":\"s1\"'*) case \"$answer\" in *'\"result\"'*) seen=\"$seen ping-answered\" ;; esac ;;\n"
	 "        *'\"id\":\"s2\"'*) case \"$answer\" in *'-32601'*) seen=\"$seen sampling-refused\" ;; esac ;;\n"
	 "        esac\n"
	 "      done\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"%s\"}]}}\\n' \"$id\" \"$seen\" ;;\n"
	 "    *)\n"
	 "      printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"error\":{\"code\":-32602,\"message\":\"Unknown tool\"}}\\n' \"$id\" ;;\n"
	 "    esac ;;\n"
	 "  *)\n"
	 "    if [ -n \"$id\" ]; then printf '{\"jsonrpc\":\"2.0\",\"id\":%s,\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}\\n' \"$id\"; fi ;;\n"
	 "  esac\n"
	 "done\n"
	 "echo 'fake MCP server input closed' >&2\n"
	 "case \"$mode\" in\n"
	 "ignore-input-close|ignore-sigterm) while :; do /bin/sleep 1; done ;;\n"
	 "esac\n"
	 "exit 0\n";

@interface NFKMCPStdioTests : XCTestCase
@property (nonatomic, copy) NSString *directory;
@property (nonatomic, copy) NSString *scriptPath;
@end

@implementation NFKMCPStdioTests

- (void)setUp
{
	[super setUp];
	self.directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[@"NFKMCPStdioTests-" stringByAppendingString:NSUUID.UUID.UUIDString]];
	[NSFileManager.defaultManager createDirectoryAtPath:self.directory withIntermediateDirectories:YES attributes:nil error:NULL];
	self.scriptPath = [self.directory stringByAppendingPathComponent:@"fake-mcp-server.sh"];
	NSError *error = nil;
	XCTAssertTrue([NFKFakeMCPServerScript writeToFile:self.scriptPath atomically:YES encoding:NSUTF8StringEncoding error:&error], @"%@", error);
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtPath:self.directory error:NULL];
	[super tearDown];
}

#pragma mark Helpers

- (NFKMCPClient *)clientInMode:(NSString *)mode environment:(nullable NSDictionary<NSString *, NSString *> *)environment
			  workingDirectory:(nullable NSURL *)workingDirectory
{
	NFKMCPClient *client = [[NFKMCPClient alloc] initWithCommand:@"/bin/sh" arguments:@[ self.scriptPath, mode ]
													 environment:environment workingDirectoryURL:workingDirectory];
	client.timeout = 10;
	client.serverProcess.stopInputTimeout = 0.3;
	client.serverProcess.stopTerminateTimeout = 0.3;
	[self addTeardownBlock:^{
		[client disconnect];
	}];
	return client;
}

- (NFKMCPClient *)client
{
	return [self clientInMode:@"normal" environment:nil workingDirectory:nil];
}

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow < 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.02];
	}
	return YES;
}

- (NSString *)resolvedPath:(NSString *)path
{
	char resolved[PATH_MAX];
	return realpath(path.fileSystemRepresentation, resolved) != NULL ? @(resolved) : path;
}

#pragma mark The client over stdio

- (void)testTheHandshakeListsEveryPageOfToolsAndCallsOne
{
	NFKMCPClient *client = self.client;
	XCTAssertEqual(client.transport, NFKMCPTransportStdio);
	XCTAssertNil(client.endpointURL);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateIdle, @"nothing launches before the first call");

	NSError *error = nil;
	NSArray<NFKInferenceTool *> *tools = [client toolsWithError:&error];
	NSArray *expected = @[ @"where", @"log", @"crash", @"slow", @"ask_back" ];
	XCTAssertEqualObjects([tools valueForKey:@"name"], expected, @"%@", error);
	XCTAssertEqualObjects(tools.firstObject.toolDescription, @"Says where the server runs.");
	XCTAssertEqualObjects(client.protocolVersion, @"2025-06-18");
	XCTAssertEqualObjects(client.serverInfo[@"name"], @"fake-stdio");
	XCTAssertEqualObjects(client.instructions, @"A fake server for tests.");
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateRunning);
	XCTAssertGreaterThan(client.serverProcess.processIdentifier, 0);
	XCTAssertTrue([client pingWithError:&error], @"%@", error);

	NFKToolResult *result = [tools.firstObject invokeWithArguments:@{}];
	XCTAssertFalse(result.isError, @"%@", result.text);
	XCTAssertTrue([result.text hasPrefix:@"unset in "], @"%@", result.text);

	XCTAssertNil([client resultOfToolNamed:@"missing" arguments:@{} error:&error]);
	XCTAssertEqualObjects(error.localizedDescription, @"tools/call: Unknown tool");
}

- (void)testTheEnvironmentAndWorkingDirectoryReachTheServer
{
	setenv("NFK_MCP_TEST_SECRET", "s3cret", 1);
	[self addTeardownBlock:^{
		unsetenv("NFK_MCP_TEST_SECRET");
	}];
	NSURL *workingDirectory = [NSURL fileURLWithPath:self.directory isDirectory:YES];
	NSString *directory = [self resolvedPath:self.directory];
	NSString *home = NSProcessInfo.processInfo.environment[@"HOME"] ?: @"unset";
	NFKMCPClient *safe = [self clientInMode:@"normal" environment:@{ @"FAKE_MCP_GREETING": @"hello" } workingDirectory:workingDirectory];
	XCTAssertFalse(safe.inheritsFullEnvironment);
	XCTAssertEqualObjects(safe.serverProcess.inheritedEnvironmentVariables, NFKMCPClient.defaultInheritedEnvironmentVariables);
	NSError *error = nil;
	NFKToolResult *result = [safe resultOfToolNamed:@"where" arguments:@{} error:&error];
	NSString *expected = [NSString stringWithFormat:@"hello in %@, home %@, secret unset", directory, home];
	XCTAssertEqualObjects(result.text, expected, @"the server inherits HOME and not the secret, %@", error);

	NFKMCPClient *full = [self clientInMode:@"normal" environment:@{ @"FAKE_MCP_GREETING": @"hi" } workingDirectory:workingDirectory];
	full.inheritsFullEnvironment = YES;
	result = [full resultOfToolNamed:@"where" arguments:@{} error:&error];
	expected = [NSString stringWithFormat:@"hi in %@, home %@, secret s3cret", directory, home];
	XCTAssertEqualObjects(result.text, expected, @"%@", error);

	NFKMCPClient *clean = [self clientInMode:@"normal" environment:@{ @"FAKE_MCP_GREETING": @"bare" } workingDirectory:workingDirectory];
	clean.serverProcess.inheritsEnvironment = NO;
	result = [clean resultOfToolNamed:@"where" arguments:@{} error:&error];
	expected = [NSString stringWithFormat:@"bare in %@, home unset, secret unset", directory];
	XCTAssertEqualObjects(result.text, expected, @"a clean environment holds only the variables given, %@", error);
}

- (void)testTheServersLogIsKeptAndForwardedLineByLine
{
	NFKMCPClient *client = self.client;
	NSMutableArray<NSString *> *forwarded = [NSMutableArray array];
	client.serverProcess.standardErrorLineHandler = ^(NSString *line) {
		@synchronized (forwarded) {
			[forwarded addObject:line];
		}
	};
	NSError *error = nil;
	XCTAssertEqualObjects([client resultOfToolNamed:@"log" arguments:@{} error:&error].text, @"logged", @"%@", error);
	XCTAssertTrue([self waitUntil:^BOOL {
		@synchronized (forwarded) {
			return [forwarded containsObject:@"log line two"];
		}
	} timeout:5]);
	NSArray *lines = nil;
	@synchronized (forwarded) {
		lines = [forwarded copy];
	}
	NSArray *expected = @[ @"fake MCP server starting", @"log line one", @"log line two" ];
	XCTAssertEqualObjects(lines, expected, @"standard error never reaches the JSON-RPC reader");
	XCTAssertTrue([client.serverProcess.recentStandardError hasSuffix:@"log line one\nlog line two\n"], @"%@", client.serverProcess.recentStandardError);
}

- (void)testACrashedServerStaysDownUntilConnectLaunchesIt
{
	NFKMCPClient *client = self.client;
	NSMutableArray<NSNumber *> *states = [NSMutableArray array];
	client.serverProcess.stateHandler = ^(NFKSubprocess *subprocess, NFKSubprocessState state) {
		@synchronized (states) {
			[states addObject:@(state)];
		}
	};
	NSError *error = nil;
	XCTAssertNil([client resultOfToolNamed:@"crash" arguments:@{} error:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	XCTAssertTrue([error.localizedDescription containsString:@"exited with status 3"], @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:@"about to crash"], @"the error carries the server's last log line: %@", error);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateExited);
	XCTAssertEqual(client.serverProcess.terminationReason, NFKSubprocessTerminationReasonExit);
	XCTAssertEqual(client.serverProcess.terminationStatus, 3);
	XCTAssertTrue([self waitUntil:^BOOL {
		@synchronized (states) {
			return [states.lastObject isEqual:@(NFKSubprocessStateExited)];
		}
	} timeout:5]);
	NSArray *observed = nil;
	@synchronized (states) {
		observed = [states copy];
	}
	NSArray *expected = @[ @(NFKSubprocessStateStarting), @(NFKSubprocessStateRunning), @(NFKSubprocessStateExited) ];
	XCTAssertEqualObjects(observed, expected);
	XCTAssertFalse(client.serverProcess.isRestartPending);

	NSError *down = nil;
	XCTAssertNil([client toolsWithError:&down], @"a call after the exit does not launch the server");
	XCTAssertEqual(down.code, kNFKError_RemoteUnreachable);
	XCTAssertTrue([down.localizedDescription containsString:@"exited with status 3"], @"%@", down);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateExited);
	@synchronized (states) {
		XCTAssertEqual(states.count, 3u, @"no launch began");
	}

	XCTAssertTrue([client connectWithError:&error], @"connect launches the server again, %@", error);
	XCTAssertEqual([client toolsWithError:&error].count, 5u, @"a new session on the new process, %@", error);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateRunning);
}

- (void)testTheServersOwnRequestsAreAnsweredOrRefused
{
	NSError *error = nil;
	NFKToolResult *result = [self.client resultOfToolNamed:@"ask_back" arguments:@{} error:&error];
	XCTAssertEqualObjects(result.text, @" ping-answered sampling-refused", @"%@", error);
}

- (void)testARequestPastItsTimeoutFailsAndTheClientCarriesOn
{
	NFKMCPClient *client = self.client;
	NSError *error = nil;
	XCTAssertTrue([client connectWithError:&error], @"%@", error);
	client.timeout = 0.5;
	XCTAssertNil([client resultOfToolNamed:@"slow" arguments:@{} error:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	XCTAssertTrue([error.localizedDescription containsString:@"did not answer within 0.5 seconds"], @"%@", error);

	client.timeout = 10;
	XCTAssertTrue([client pingWithError:&error], @"the late reply to the abandoned call is skipped, %@", error);
}

- (void)testAServerThatCannotLaunchReportsWhy
{
	NFKMCPClient *client = [[NFKMCPClient alloc] initWithCommand:@"inferkit-no-such-mcp-server" arguments:@[]
													 environment:nil workingDirectoryURL:nil];
	NSError *error = nil;
	XCTAssertFalse([client connectWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	NSError *underlying = error.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqualObjects(underlying.domain, NFKSubprocessErrorDomain);
	XCTAssertEqual(underlying.code, NFKSubprocessErrorLaunchFailed);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateFailedToLaunch);
	NSString *reason = client.serverProcess.launchError.localizedDescription;
	XCTAssertTrue([reason containsString:@"inferkit-no-such-mcp-server is not found in "], @"%@", reason);
	XCTAssertTrue([reason containsString:@"/usr/bin"], @"the server's PATH is searched, %@", reason);
	XCTAssertTrue([reason containsString:@"/opt/homebrew/bin"], @"then the agent tool directories, %@", reason);
	XCTAssertTrue([reason containsString:[NSHomeDirectory() stringByAppendingPathComponent:@".local/bin"]], @"%@", reason);
	NSError *again = nil;
	XCTAssertFalse([client pingWithError:&again]);
	XCTAssertTrue([again.localizedDescription containsString:@"could not be launched"], @"a later call reports the failed launch, %@", again);

	NFKMCPClient *badDirectory = [self clientInMode:@"normal" environment:nil
								   workingDirectory:[NSURL fileURLWithPath:[self.directory stringByAppendingPathComponent:@"absent"]]];
	XCTAssertFalse([badDirectory connectWithError:&error], @"a working directory that does not exist fails the launch");
	XCTAssertEqual(badDirectory.serverProcess.state, NFKSubprocessStateFailedToLaunch);
}

- (void)testTheRestartPolicyRelaunchesACrashedServer
{
	NFKMCPClient *client = self.client;
	client.serverProcess.restartPolicy = NFKSubprocessRestartPolicyOnCrash;
	client.serverProcess.restartDelay = 0.1;
	NSError *error = nil;
	XCTAssertNil([client resultOfToolNamed:@"crash" arguments:@{} error:&error]);
	XCTAssertTrue([self waitUntil:^BOOL {
		return client.serverProcess.restartCount == 1 && client.serverProcess.state == NFKSubprocessStateRunning;
	} timeout:5], @"the crashed server launches again on its own");
	XCTAssertEqualObjects([client resultOfToolNamed:@"log" arguments:@{} error:&error].text, @"logged",
						  @"the relaunched server takes a new session, %@", error);
	XCTAssertEqual(client.serverProcess.restartCount, 1u, @"the call reached the relaunched process");
}

#pragma mark Stopping

- (void)testDisconnectClosesTheServersInputAndItExits
{
	NFKMCPClient *client = self.client;
	client.serverProcess.stopInputTimeout = 5;
	NSError *error = nil;
	XCTAssertTrue([client connectWithError:&error], @"%@", error);
	NSDate *start = [NSDate date];
	[client disconnect];
	XCTAssertLessThan(-start.timeIntervalSinceNow, 4.0, @"the server exits on its input closing, before any signal");
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateExited);
	XCTAssertEqual(client.serverProcess.terminationReason, NFKSubprocessTerminationReasonExit);
	XCTAssertEqual(client.serverProcess.terminationStatus, 0);
	XCTAssertNil(client.protocolVersion);
	XCTAssertTrue([client.serverProcess.recentStandardError containsString:@"fake MCP server input closed"]);
	XCTAssertTrue([client pingWithError:&error], @"the next call after disconnect launches the server, %@", error);
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateRunning);
}

- (void)testAServerIgnoringItsInputClosingGetsSIGTERM
{
	NFKMCPClient *client = [self clientInMode:@"ignore-input-close" environment:nil workingDirectory:nil];
	client.serverProcess.stopTerminateTimeout = 5;
	NSError *error = nil;
	XCTAssertTrue([client connectWithError:&error], @"%@", error);
	[client disconnect];
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateExited);
	XCTAssertEqual(client.serverProcess.terminationReason, NFKSubprocessTerminationReasonSignal);
	XCTAssertEqual(client.serverProcess.terminationStatus, SIGTERM);
}

- (void)testAServerIgnoringSIGTERMGetsSIGKILL
{
	NFKMCPClient *client = [self clientInMode:@"ignore-sigterm" environment:nil workingDirectory:nil];
	NSError *error = nil;
	XCTAssertTrue([client connectWithError:&error], @"%@", error);
	NSDate *start = [NSDate date];
	[client disconnect];
	XCTAssertEqual(client.serverProcess.state, NFKSubprocessStateExited);
	XCTAssertEqual(client.serverProcess.terminationReason, NFKSubprocessTerminationReasonSignal);
	XCTAssertEqual(client.serverProcess.terminationStatus, SIGKILL);
	XCTAssertGreaterThanOrEqual(-start.timeIntervalSinceNow, 0.6, @"both timeouts pass before SIGKILL");
}

@end

@interface NFKMCPClientDisconnectTests : XCTestCase
@end

@implementation NFKMCPClientDisconnectTests

- (void)testDisconnectEndsAnHTTPSession
{
	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.port = 0;
	server.loopbackOnly = YES;
	NSError *error = nil;
	XCTAssertTrue([server startWithError:&error], @"%@", error);
	[self addTeardownBlock:^{
		[server stop];
	}];
	NFKMCPClient *client = [[NFKMCPClient alloc] initWithEndpointURL:server.localMCPURL apiKey:nil];
	XCTAssertEqual(client.transport, NFKMCPTransportStreamableHTTP);
	XCTAssertNil(client.serverProcess);
	XCTAssertTrue([client pingWithError:&error], @"%@", error);
	XCTAssertEqual(server.MCPSessions.count, 1u);
	[client disconnect];
	XCTAssertEqual(server.MCPSessions.count, 0u, @"disconnect deletes the session the server assigned");
	XCTAssertNil(client.protocolVersion);
	XCTAssertTrue([client pingWithError:&error], @"a later call connects again, %@", error);
}

@end
