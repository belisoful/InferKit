//
//  NFKToolLoop.m
//  InferKit
//

#import "NFKToolLoop.h"
#import "NFKToolMessages.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"

@implementation NFKToolInvocation

- (instancetype)initWithCallIdentifier:(NSString *)callIdentifier
							  toolName:(NSString *)toolName
							 arguments:(NSDictionary<NSString *, id> *)arguments
								result:(NFKToolResult *)result
					   durationSeconds:(NSTimeInterval)durationSeconds
{
	self = [super init];
	if (self != nil) {
		_callIdentifier = [callIdentifier copy];
		_toolName = [toolName copy];
		_arguments = [arguments copy];
		_result = result;
		_durationSeconds = durationSeconds;
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ (%@) → %@>", NSStringFromClass(self.class), self.toolName, self.callIdentifier, self.result];
}

@end

/*! What one run of the loop carries from turn to turn. */
@interface NFKToolLoopRun : NSObject
@property (nonatomic, strong, nullable) NFKInferenceJob *job;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *messages;
@property (nonatomic, strong) NSMutableArray<NFKToolInvocation *> *invocations;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSNumber *> *usage;
@property (nonatomic, strong) NSLock *lock;
@property (nonatomic, strong, nullable) NFKInferenceJob *currentTurnJob;
@end

@implementation NFKToolLoopRun

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_messages = [NSMutableArray array];
		_invocations = [NSMutableArray array];
		_usage = [NSMutableDictionary dictionary];
		_lock = [[NSLock alloc] init];
	}
	return self;
}

- (BOOL)isCancelled
{
	return self.job.status == NFKInferenceJobStatusCancelled;
}

@end

@implementation NFKToolLoop

- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend tools:(NSArray<NFKInferenceTool *> *)tools
{
	self = [super init];
	if (self != nil) {
		_backend = backend;
		_tools = [tools copy];
		_maximumTurns = 8;
	}
	return self;
}

- (nullable NFKInferenceTool *)toolNamed:(NSString *)name
{
	for (NFKInferenceTool *tool in self.tools) {
		if ([tool.name isEqualToString:name]) {
			return tool;
		}
	}
	return nil;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.backend.isReady;
}

- (NSString *)backendIdentifier
{
	return @"tool-loop";
}

- (BOOL)prepareWithError:(NSError **)error
{
	return NFKInferencePrepare(self.backend, error);
}

- (NSSet<NSString *> *)supportedInputKeys
{
	if ([self.backend respondsToSelector:@selector(supportedInputKeys)]) {
		return self.backend.supportedInputKeys;
	}
	return [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages ]];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	NSSet *keys = [self.backend respondsToSelector:@selector(supportedParameterKeys)] ? self.backend.supportedParameterKeys : [NSSet set];
	return [keys setByAddingObject:NFKParameterTools];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return [self.backend respondsToSelector:@selector(modelInfo)] ? self.backend.modelInfo : @{};
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [self resultForRequest:request run:[[NFKToolLoopRun alloc] init] error:error];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NFKToolLoopRun *run = [[NFKToolLoopRun alloc] init];
	run.job = job;
	__weak NFKToolLoopRun *weakRun = run;
	job.cancellationHandler = ^{
		NFKToolLoopRun *strongRun = weakRun;
		[strongRun.lock lock];
		NFKInferenceJob *turnJob = strongRun.currentTurnJob;
		[strongRun.lock unlock];
		[turnJob cancel];
	};
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKInferenceResult *result = [self resultForRequest:request run:run error:&error];
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:error ?: [self errorWithCode:kNFKError_InferenceBackendFailure reason:@"the tool loop ended without a result"]];
		}
	});
	return job;
}

#pragma mark The loop

- (nullable NFKInferenceResult *)resultForRequest:(NFKInferenceRequest *)request run:(NFKToolLoopRun *)run error:(NSError **)outError
{
	if (request.messages != nil) {
		[run.messages addObjectsFromArray:request.messages];
	} else if (request.prompt.length > 0) {
		[run.messages addObject:@{ @"role": @"user", @"content": request.prompt }];
	} else {
		[self errorWithCode:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages" into:outError];
		return nil;
	}
	NSArray<NSDictionary *> *declarations = [self declarationsForRequest:request];

	for (NSUInteger turn = 1; ; turn++) {
		if (run.isCancelled) {
			[self cancellationErrorInto:outError];
			return nil;
		}
		NFKInferenceResult *result = [self resultOfTurn:turn
												request:[self turnRequestFor:request messages:run.messages declarations:declarations]
													run:run
												  error:outError];
		if (result == nil) {
			return nil;
		}
		[self addUsageOf:result to:run.usage];
		NSDictionary *assistant = [NFKToolMessages assistantMessageForResult:result];
		[run.messages addObject:assistant];

		NSArray<NSDictionary *> *calls = [NFKToolMessages toolCallsOfAssistantMessage:assistant];
		if (calls.count == 0 || ![self holdsEveryToolIn:calls]) {
			return [self finalResultFrom:result run:run reachedLimit:NO];
		}
		if (turn >= self.maximumTurns) {
			return [self finalResultFrom:result run:run reachedLimit:YES];
		}
		for (NSDictionary *call in calls) {
			if (run.isCancelled) {
				[self cancellationErrorInto:outError];
				return nil;
			}
			NFKToolInvocation *invocation = [self invocationForCall:call];
			[run.invocations addObject:invocation];
			[run.messages addObject:[NFKToolMessages messageForToolCallIdentifier:invocation.callIdentifier
																			 name:invocation.toolName
																		   result:invocation.result]];
			void (^handler)(NFKToolInvocation *) = self.invocationHandler;
			if (handler != nil) {
				handler(invocation);
			}
		}
	}
}

// The loop's tools, then the request's own declarations of tools the caller runs.
- (NSArray<NSDictionary *> *)declarationsForRequest:(NFKInferenceRequest *)request
{
	NSMutableArray<NSDictionary *> *declarations = [NSMutableArray array];
	NSMutableSet<NSString *> *names = [NSMutableSet set];
	for (NFKInferenceTool *tool in self.tools) {
		[declarations addObject:tool.declaration];
		[names addObject:tool.name];
	}
	NSArray *declared = [request parameterForKey:NFKParameterTools];
	for (NSDictionary *entry in [declared isKindOfClass:NSArray.class] ? declared : @[]) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *name = [self nameOfDeclaration:entry];
		if (name == nil || ![names containsObject:name]) {
			[declarations addObject:entry];
		}
	}
	return declarations;
}

// A declaration is the contract's {name, …} or a provider's wire shape {type: function, function: {name, …}}.
- (nullable NSString *)nameOfDeclaration:(NSDictionary *)entry
{
	id name = entry[@"name"];
	if (name == nil && [entry[@"function"] isKindOfClass:NSDictionary.class]) {
		name = entry[@"function"][@"name"];
	}
	return [name isKindOfClass:NSString.class] ? name : nil;
}

- (NFKInferenceRequest *)turnRequestFor:(NFKInferenceRequest *)request
							   messages:(NSArray<NSDictionary *> *)messages
						   declarations:(NSArray<NSDictionary *> *)declarations
{
	NSMutableDictionary *inputs = [request.inputs mutableCopy];
	[inputs removeObjectForKey:NFKInputPrompt];
	inputs[NFKInputMessages] = [messages copy];
	NSMutableDictionary *parameters = [request.parameters mutableCopy];
	if (declarations.count > 0) {
		parameters[NFKParameterTools] = declarations;
	}
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:request.outputModality];
}

- (nullable NFKInferenceResult *)resultOfTurn:(NSUInteger)turn
									  request:(NFKInferenceRequest *)request
										  run:(NFKToolLoopRun *)run
										error:(NSError **)outError
{
	NFKInferenceJob *job = run.job;
	if (job == nil) {
		return [self.backend runInferenceForRequest:request error:outError];
	}
	NFKInferenceJob *turnJob = NFKInferenceSubmit(self.backend, request, NULL);
	[run.lock lock];
	run.currentTurnJob = turnJob;
	[run.lock unlock];
	if (job.status == NFKInferenceJobStatusCancelled) {
		[turnJob cancel];
	}

	double progress = (double)(turn - 1) / (double)MAX(self.maximumTurns, (NSUInteger)1);
	turnJob.progressHandler = ^(NFKInferenceJob *reporting) {
		[job reportProgress:progress partialResult:reporting.partialResult];
	};
	dispatch_semaphore_t finished = dispatch_semaphore_create(0);
	turnJob.completionHandler = ^(NFKInferenceJob *completed) {
		dispatch_semaphore_signal(finished);
	};
	dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);

	[run.lock lock];
	run.currentTurnJob = nil;
	[run.lock unlock];
	if (turnJob.status == NFKInferenceJobStatusCancelled) {
		[self cancellationErrorInto:outError];
		return nil;
	}
	if (turnJob.result == nil && outError != NULL) {
		*outError = turnJob.error;
	}
	return turnJob.result;
}

- (BOOL)holdsEveryToolIn:(NSArray<NSDictionary *> *)calls
{
	for (NSDictionary *call in calls) {
		if ([self toolNamed:call[@"name"]] == nil) {
			return NO;
		}
	}
	return YES;
}

- (NFKToolInvocation *)invocationForCall:(NSDictionary *)call
{
	NSString *name = call[@"name"];
	NSDictionary *arguments = call[@"arguments"];
	NFKToolCallApproval approval = self.approvalHandler;
	NSDate *start = [NSDate date];
	NFKToolResult *result = nil;
	if (approval != nil && !approval(name, arguments)) {
		result = [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"The user declined this call to %@.", name]];
	} else {
		result = [[self toolNamed:name] invokeWithArguments:arguments];
	}
	return [[NFKToolInvocation alloc] initWithCallIdentifier:call[@"id"]
													toolName:name
												   arguments:arguments
													  result:result
											 durationSeconds:-start.timeIntervalSinceNow];
}

- (void)addUsageOf:(NFKInferenceResult *)result to:(NSMutableDictionary<NSString *, NSNumber *> *)total
{
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	if (![usage isKindOfClass:NSDictionary.class]) {
		return;
	}
	for (NSString *key in usage) {
		if ([usage[key] isKindOfClass:NSNumber.class]) {
			total[key] = @(total[key].longLongValue + [usage[key] longLongValue]);
		}
	}
}

- (NFKInferenceResult *)finalResultFrom:(NFKInferenceResult *)result run:(NFKToolLoopRun *)run reachedLimit:(BOOL)reachedLimit
{
	NSMutableDictionary *outputs = [result.outputs mutableCopy];
	outputs[NFKOutputMessages] = [run.messages copy];
	if (run.invocations.count > 0) {
		outputs[NFKOutputToolInvocations] = [run.invocations copy];
	}
	if (run.usage.count > 0) {
		outputs[NFKOutputUsage] = [run.usage copy];
	}
	if (reachedLimit) {
		outputs[NFKOutputToolTurnLimitReached] = @YES;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Errors

- (NSError *)errorWithCode:(NFKInferenceError)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

- (BOOL)errorWithCode:(NFKInferenceError)code reason:(NSString *)reason into:(NSError **)outError
{
	if (outError != NULL) {
		*outError = [self errorWithCode:code reason:reason];
	}
	return NO;
}

- (BOOL)cancellationErrorInto:(NSError **)outError
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError
									userInfo:@{ NSLocalizedDescriptionKey: @"the tool loop was cancelled" }];
	}
	return NO;
}

@end
