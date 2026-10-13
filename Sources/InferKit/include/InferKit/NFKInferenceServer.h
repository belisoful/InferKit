//
//  NFKInferenceServer.h
//  InferKit
//

#ifndef NFKInferenceServer_h
#define NFKInferenceServer_h

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKServerStatus.h>

NS_ASSUME_NONNULL_BEGIN

/*! The port a server listens on unless a caller names another, and the port NFKRemoteProvider.inferKit
	points at. Introduced in InferKit 0.4.0. */
extern const uint16_t NFKInferenceServerDefaultPort;

/*! The Bonjour service type a server advertises under and NFKRemoteProvider discovers. An iOS app
	that browses or advertises lists it under NSBonjourServices in its Info.plist. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKInferenceServerServiceType;

/*! The error domain for serving failures that are not a run's own. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInferenceServerErrorDomain;

/*!
	@enum       NFKInferenceServerError
	@abstract   Error codes in NFKInferenceServerErrorDomain.
	@constant   NFKInferenceServerErrorListenFailed The server could not bind or listen; the POSIX
				error, when there is one, is under NSUnderlyingErrorKey.
	@constant   NFKInferenceServerErrorAPIKeyRequired The configuration requires a key from some
				client and no key is set.
	@constant   NFKInferenceServerErrorInvalidConfiguration The configuration cannot be served:
				maximumConcurrentRunsPerModel or maximumRequestBodyBytes is zero.
	@constant   NFKInferenceServerErrorModelNotFound A request named a model the server does not host.
	@constant   NFKInferenceServerErrorUnauthorized A request carried no key, or the wrong one.
	@constant   NFKInferenceServerErrorBusy A request found its model's queue full
				(maximumQueuedRunsPerModel). The reply is 503 with a Retry-After of the estimated wait,
				and NFKRemoteErrorRetryAfterKey carries the same moment.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKInferenceServerErrorDomain, NFKInferenceServerError) {
	NFKInferenceServerErrorListenFailed = 1,
	NFKInferenceServerErrorAPIKeyRequired = 2,
	NFKInferenceServerErrorInvalidConfiguration = 3,
	NFKInferenceServerErrorModelNotFound = 4,
	NFKInferenceServerErrorUnauthorized = 5,
	NFKInferenceServerErrorBusy = 6,
};

/*!
	@protocol   NFKServingRuntimeStatus
	@abstract   A runtime that reports its own state to a server's status route.
	@discussion A companion package's runtime class adopts it, and a server finds the class by name
				at runtime, so the core links nothing of the runtime. The report appears in the
				status route's host object under runtimes, keyed by runtimeStatusName, and is left
				out with every other host detail when reportsHostDetails is NO. A server looks for
				InferKitMLX's NFKMLXGPU by default; registerRuntimeStatusClassName: adds another
				class. Introduced in InferKit 0.4.0.
*/
@protocol NFKServingRuntimeStatus <NSObject>

/*! The name the report appears under, such as "mlx". */
@property (class, nonatomic, readonly, copy) NSString *runtimeStatusName;

/*! The runtime's state now, as strings and numbers. */
@property (class, nonatomic, readonly, copy) NSDictionary<NSString *, id> *runtimeStatus;

@optional

/*! Returns the memory the runtime keeps cached for reuse to the system. A server calls it after it
	unloads a model, when none of its models has a run in flight or a backend being built. Introduced in
	InferKit 0.4.0. */
+ (void)releaseCachedMemory;

@end

/*! Builds a hosted model's backend: returns it, or nil with an error. A server calls it on a background
	queue at user-initiated quality of service when a request needs the model and its backend is not in
	memory, one build at a time per model. Introduced in InferKit 0.4.0. */
typedef id<NFKInferenceBackend> _Nullable (^NFKServedBackendProvider)(NSError * _Nullable * _Nullable error);

/*!
	@class      NFKInferenceServer
	@abstract   Serves the models this process hosts to clients on this machine and on the network.
	@discussion A server hosts backends by model name and answers HTTP on two surfaces under /v1:

				- OpenAI-compatible routes: GET /models and /models/{name}, POST /chat/completions
				  (streamed as server-sent events when the body sets stream), /embeddings,
				  /audio/transcriptions, /audio/translations, /audio/speech, /images/generations, and
				  /images/edits. NFKRemoteBackend and the core's embedding, transcription, speech, and
				  image backends reach them unchanged, and so does any OpenAI client library.
				- The native route, POST /inferkit/run, which carries a whole NFKInferenceRequest and
				  NFKInferenceResult in InferKit's tagged JSON form, so every key reaches the hosted
				  backend: pixel buffers, masks, multi-arrays, audio and video files, and the core's
				  value types. NFKRemoteInferKitBackend is its client and streams the job's progress
				  and partial results.
				- The status route, GET /inferkit/status, which reports each model's load and the
				  machine's state for a load balancer or a dashboard (see Status).

				Beside them, POST /mcp serves the tools added with addTool: over the Model Context
				Protocol (Streamable HTTP), so an MCP client reaches them: Claude Code, Claude
				Desktop, an agent framework, or NFKMCPClient. Only the added tools are listed; a hosted
				model is a tool when an NFKBackendTool for it is added. A tool's text, the images it
				shows the model, and the audio it generates come back as MCP content. The route takes
				the same key as the others, and refuses a browser request whose Origin names another
				host.

				A request names its model in the body's model field. When the server hosts exactly
				one model, a request that names none runs on it.

				Access:
				- A client on this machine (a loopback peer) needs no key unless
				  requiresAPIKeyOnLoopback is set.
				- A client on another machine presents apiKey as a Bearer token, which is what the
				  remote backends' apiKey sends. requiresAPIKey is YES by default; setting it to NO
				  opens the server to anyone who can reach it.
				- startWithError: fails with NFKInferenceServerErrorAPIKeyRequired when the
				  configuration requires a key and none is set.
				A reverse proxy on this machine connects from loopback, so behind one set
				requiresAPIKeyOnLoopback or listen on loopback only.

				Setting TLSIdentity serves HTTPS with that identity. A client's URL session trusts
				the certificate only if the system does, so a self-signed identity needs the client to
				install it or to pin its SHA-256 fingerprint. NFKServerIdentity makes and keeps such an
				identity; a client pins it with pinnedCertificateSHA256.

				advertisesService publishes the server over Bonjour as NFKInferenceServerServiceType,
				carrying the base path, whether TLS is on, whether a key is required, and with TLS the
				certificate's SHA-256 fingerprint under cert_sha256. Bonjour is unauthenticated, so the
				fingerprint is advisory: a client confirms it with its user before pinning it, unless
				the client turns on automatic pinning (NFKRemoteProvider's discoveryPinnedServers),
				which trusts the first fingerprint seen for each service name and reports a later
				different one. NFKRemoteProvider's discoverInferKitServersWithTimeout: finds the server. On iOS, listening on
				or browsing the local network needs NSLocalNetworkUsageDescription in the Info.plist.

				Runs follow the backends' contract: each backend serves at most
				maximumConcurrentRunsPerModel runs at once and later requests queue in arrival order.
				A backend that implements submitInferenceJobForRequest: streams its partial results
				to a streamed request; a synchronous one answers in one piece. A client that
				disconnects cancels its run, or removes it from the queue. Files the server writes
				for inline media are removed when the run ends. maximumQueuedRunsPerModel bounds the
				queue: a request that finds it full is refused with 503 and a Retry-After, so a
				client or a load balancer can try another server.

				Status. Each model's backend describes its loaded model under "model" (its modelInfo:
				parameter count, weight and storage bytes, precision, quantization, context length,
				compute units) and its state now under "status" (its backendStatus, when it reports
				one) in the status route and in its GET /models entry, which also carry the model's
				load under "load":
				- limit, running, and queued (queue_limit when one is set).
				- completed, failed, cancelled, and refused run counts.
				- average_run_seconds and average_wait_seconds, moving averages that weight the
				  newest run by a fifth. Only runs that return a result feed the run average.
				- output_tokens_per_second, from the results' NFKOutputUsage, when a backend reports
				  tokens.
				- input_tokens, the input tokens the completed runs' usage reported since the server
				  started, and, over the runs whose usage also reported NFKUsageCachedTokens,
				  cached_input_tokens and cached_input_share, the share of their input a backend's
				  cache served. Two readings' difference gives the share over the time between them.
				- estimated_wait_seconds: when a request arriving now would start. A running job that
				  reports progress is estimated from that progress, any other run from the average
				  run. It is absent until one run has finished, unless a slot is free.
				- runs: each running run's elapsed_seconds, its progress when it reports one, and
				  its estimated_remaining_seconds.
				Every run reply carries the same reading in X-InferKit-Limit, X-InferKit-Running,
				X-InferKit-Queued, and X-InferKit-Estimated-Wait, taken when the reply's head is
				written.

				Residency. A model hosted with addBackendProvider:forModelName: is listed before its
				backend exists, and its first request builds it:
				- A request that finds the model not loaded starts the build and waits in the model's
				  queue. Requests that arrive during the build wait for the same build.
				- A build that fails fails every request waiting for it with the provider's error and
				  leaves the model not loaded, so the next request builds again.
				- unloadModelNamed: releases the backend and keeps the model hosted. It waits until the
				  model has no run in flight or queued, and never cuts a run.
				- A model hosted with addBackend:forModelName: is resident from the start and stays
				  resident, since the server has no way to build its backend again.
				- A model that is not resident reports ready, since a request builds it, and describes
				  the backend it last held: its identifier, inputs, parameters, and modelInfo.
				Each model's "residency" object carries state (not_loaded, loading, or resident),
				unloadable, last_used (seconds since 1970), idle_timeout_seconds (absent when the model
				is not unloaded for idleness), load_seconds, load_error, loads, and unloads.

				Unloading. Two independent options release a provider-hosted model that sits unused:
				- idleUnloadInterval (and setIdleUnloadInterval:forModelName:) unloads a model after a
				  fixed idle time.
				- unloadsUnderMemoryPressure shortens that time as memory pressure rises. Pressure is a
				  value in 0...1 from memoryPressureProvider. Below memoryPressureOnset it changes
				  nothing. From the onset to 1 the timeout falls linearly from the model's idle interval,
				  or from pressureUnloadInterval when idle unloading is off, to
				  shortestIdleUnloadInterval.
				The model's timeout is the shorter of the two, an option that is off contributing no
				limit. With both off, a model stays until unloadModelNamed:. Guards:
				- A model is not unloaded within unloadGracePeriod of its build finishing, or while a run
				  is in flight or queued.
				- A model past its idle interval is unloaded. Of the models past only their
				  pressure-shortened timeout, the least recently used one is unloaded per check, and the
				  next such unload waits pressureUnloadHysteresis.
				- A pressure reading that falls takes effect once it has held for
				  pressureUnloadHysteresis; a rising one takes effect at once.
				The server checks every residencyCheckInterval seconds. After an unload, when none of its
				models has a run in flight or a backend being built, it asks each registered runtime that
				implements releaseCachedMemory to return its cache.

				The status route's "host" object describes the machine: chip, model identifier, and
				core counts; thermal_state (nominal, fair, serious, critical) and low_power_mode;
				memory (physical, available, recommended working set, pressure as normal, warning,
				or critical, and this process's footprint); runtimes, each linked runtime's own
				report (NFKServingRuntimeStatus), such as MLX's active and cached memory; cpu (usage since the previous status
				reading, and the load averages); storage on the volume of storageDirectoryURL; and,
				on macOS, gpu.utilization from the IORegistry's undocumented "Device Utilization %",
				absent when it cannot be read. GPU utilization covers every process on the machine
				and stays near 1 while one run executes, so a load balancer routes on the queue
				figures. Run replies carry X-InferKit-Thermal-State, and the Bonjour
				advertisement adds the chip and the physical memory. Setting reportsHostDetails to
				NO leaves out the host object, the thermal header, and those advertisement entries.
				The status route needs a key under the same rules as every other route.

				apiKey, requiresAPIKey, requiresAPIKeyOnLoopback, reportsHostDetails,
				storageDirectoryURL, and the hosted models take effect on the next request. The other
				settings take effect on the next start; the advertisement's host entries follow
				reportsHostDetails as it was at start.
				Introduced in InferKit 0.4.0.
*/
@interface NFKInferenceServer : NSObject

/*! The port to listen on. Defaults to NFKInferenceServerDefaultPort; 0 lets the system choose. */
@property (nonatomic, assign) uint16_t port;

/*! Listens on the loopback interface only, so no other machine can connect. Defaults to NO. */
@property (nonatomic, assign) BOOL loopbackOnly;

/*! The key a client presents as a Bearer token. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! Whether a client on another machine must present apiKey. Defaults to YES. */
@property (nonatomic, assign) BOOL requiresAPIKey;

/*! Whether a client on this machine must present apiKey too. Defaults to NO. */
@property (nonatomic, assign) BOOL requiresAPIKeyOnLoopback;

/*! The identity to serve HTTPS with, or NULL for plain HTTP. The server retains it. */
@property (nonatomic, assign, nullable) SecIdentityRef TLSIdentity;

/*! Whether the server advertises itself over Bonjour. Defaults to YES; a loopback-only server is
	never advertised. */
@property (nonatomic, assign) BOOL advertisesService;

/*! The advertised name. nil uses the computer's name. */
@property (nonatomic, copy, nullable) NSString *serviceName;

/*! The largest request body accepted, in bytes. Defaults to 256 MiB. */
@property (nonatomic, assign) NSUInteger maximumRequestBodyBytes;

/*! How many runs one hosted backend serves at once. Defaults to 1, since a backend is not assumed to
	be safe to run concurrently; applies to models added after it is set. */
@property (nonatomic, assign) NSUInteger maximumConcurrentRunsPerModel;

/*! How many runs wait for one hosted backend before later requests are refused with
	NFKInferenceServerErrorBusy. Defaults to 0, which queues without limit; applies to models added
	after it is set. */
@property (nonatomic, assign) NSUInteger maximumQueuedRunsPerModel;

/*! Whether the status route, run replies, and the Bonjour advertisement describe this machine.
	Defaults to YES; NO leaves out every host detail and keeps the models' load. */
@property (nonatomic, assign) BOOL reportsHostDetails;

/*! A directory on the volume the status route's storage figures describe, typically where the models
	are stored. nil uses the app's caches directory. */
@property (nonatomic, copy, nullable) NSURL *storageDirectoryURL;

/*! The seconds a provider-hosted model may sit idle before it is unloaded, for every model without its
	own interval. Defaults to 0, which turns idle unloading off. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval idleUnloadInterval;

/*! Sets one model's idle interval, which overrides idleUnloadInterval; 0 turns idle unloading off for
	it, and nil returns it to the server's interval. Applies to the model hosted under the name now or
	later. Introduced in InferKit 0.4.0. */
- (void)setIdleUnloadInterval:(nullable NSNumber *)seconds forModelName:(NSString *)modelName;

/*! Whether memory pressure shortens a provider-hosted model's idle timeout. Defaults to NO. Introduced
	in InferKit 0.4.0. */
@property (nonatomic, assign) BOOL unloadsUnderMemoryPressure;

/*! The pressure, in 0...1, at which the timeout starts to shorten. Defaults to 0.6, the kernel's warning
	level. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) double memoryPressureOnset;

/*! The timeout at full pressure. Defaults to 10 seconds; a shorter idle interval is kept. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval shortestIdleUnloadInterval;

/*! The timeout at the onset for a model whose idle unloading is off, which then leaves only under
	pressure. Defaults to 600 seconds. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval pressureUnloadInterval;

/*! How long after its build finishes a model is kept from idle and pressure unloading. Defaults to 30
	seconds. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval unloadGracePeriod;

/*! How long a falling pressure reading must hold before it takes effect, and the least time between two
	pressure-driven unloads. Defaults to 10 seconds. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval pressureUnloadHysteresis;

/*! How often the server applies the unloading options, in seconds. Defaults to 1; 0 stops the periodic
	check, and the host calls unloadIdleModels itself. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval residencyCheckInterval;

/*! Reads memory pressure in 0...1 for unloadsUnderMemoryPressure. The default is the higher of the
	kernel's level (normal 0, warning 0.6, critical 1) and this process's footprint as a share of
	NFKHardwareProfile's recommendedWorkingSetSize. A host with its own memory budget sets its own;
	nil restores the default. Called on a background queue. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, null_resettable) double (^memoryPressureProvider)(void);

/*! The pressure the unloading options act on now: the provider's reading, held while it falls (see
	pressureUnloadHysteresis). 0 while unloadsUnderMemoryPressure is NO. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly) double memoryPressure;

/*! Whether the server is listening. */
@property (nonatomic, readonly, getter=isRunning) BOOL running;

/*! The port the server is listening on, which is the chosen one when port is 0; 0 while stopped. */
@property (nonatomic, readonly) uint16_t listeningPort;

/*! The base URL a client on this machine uses (http://localhost:<port>/v1, or https), or nil while
	stopped. A client elsewhere uses this machine's name or address, or discovery. */
@property (nonatomic, readonly, copy, nullable) NSURL *localBaseURL;

/*! Hosts a backend under a model name, replacing any backend already under it. The model is resident
	until it is removed. */
- (void)addBackend:(id<NFKInferenceBackend>)backend forModelName:(NSString *)modelName
	NS_SWIFT_NAME(addBackend(_:forModelName:));

/*!
	@method     addBackendProvider:forModelName:
	@abstract   Hosts a model whose backend the server builds when a request first needs it, replacing
				any model already under the name.
	@discussion The model is listed at once and is not loaded until a request or loadModelNamed: builds
				it. The provider is called again after an unload or a failed build. Introduced in
				InferKit 0.4.0.
*/
- (void)addBackendProvider:(NFKServedBackendProvider)provider forModelName:(NSString *)modelName
	NS_SWIFT_NAME(addBackendProvider(_:forModelName:));

/*!
	@method     loadModelNamed:completionHandler:
	@abstract   Builds a provider-hosted model's backend ahead of its first request.
	@discussion The handler receives nil once the model is resident, or the build's error. A model that
				is already resident, or hosted with a ready backend, completes at once; one being built
				completes with that build. Cancels an unload that is waiting for runs to end. The handler
				runs on a background queue. Introduced in InferKit 0.4.0.
*/
- (void)loadModelNamed:(NSString *)modelName
	 completionHandler:(nullable void (^)(NSError * _Nullable error))completionHandler;

/*!
	@method     unloadModelNamed:
	@abstract   Releases a provider-hosted model's backend and keeps the model hosted.
	@discussion Returns at once. A model with a run in flight or queued, or a build in progress, is
				unloaded when the last of them ends; its next request builds it again. Returns NO when
				no model is hosted under the name or it was hosted with a ready backend, which the server
				cannot build again. Introduced in InferKit 0.4.0.
*/
- (BOOL)unloadModelNamed:(NSString *)modelName;

/*! Whether a hosted model's backend is in memory, or NFKServerModelResidencyUnknown when no model is
	hosted under the name. Introduced in InferKit 0.4.0. */
- (NFKServerModelResidency)residencyForModelName:(NSString *)modelName;

/*! The seconds a provider-hosted model may sit idle before it is unloaded, at the current memory
	pressure, or INFINITY when it is not unloaded for idleness. Introduced in InferKit 0.4.0. */
- (NSTimeInterval)idleTimeoutForModelName:(NSString *)modelName;

/*! Applies the unloading options now, as the periodic check does, and returns the names of the models it
	unloaded. Introduced in InferKit 0.4.0. */
- (NSArray<NSString *> *)unloadIdleModels;

/*! Stops hosting the model. Its queued runs fail; its running ones finish. */
- (void)removeBackendForModelName:(NSString *)modelName;

/*! Adds a class, found by name at runtime, whose NFKServingRuntimeStatus report every server's
	status route carries. A name whose class is absent or does not adopt the protocol is skipped. */
+ (void)registerRuntimeStatusClassName:(NSString *)className;

/*! The backend hosted under a model name, or nil, which a provider-hosted model that is not resident
	also answers. */
- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName;

/*! The hosted model names, sorted. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *modelNames;

/*! Serves a tool over MCP, replacing any tool of the same name. Introduced in InferKit 0.4.0. */
- (void)addTool:(NFKInferenceTool *)tool;

/*! Stops serving the tool of that name over MCP. Introduced in InferKit 0.4.0. */
- (void)removeToolNamed:(NSString *)name;

/*! The tools served over MCP, in the order they were added. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

/*! What an MCP client's model is told about the server as a whole when it connects, such as which
	tools to prefer, or nil for nothing. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, nullable) NSString *MCPInstructions;

/*! The MCP endpoint a client on this machine uses (http://localhost:<port>/mcp, or https), or nil
	while stopped. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy, nullable) NSURL *localMCPURL;

/*!
	@method     startWithError:
	@abstract   Binds the port and starts serving; returns NO with an error when it cannot.
	@discussion Blocks until the listener is ready, which on a free port takes milliseconds. Fails
				with NFKInferenceServerErrorAPIKeyRequired or
				NFKInferenceServerErrorInvalidConfiguration before binding when the settings
				require it, and with NFKInferenceServerErrorListenFailed when the port is taken.
*/
- (BOOL)startWithError:(NSError * _Nullable *)outError;

/*! Stops listening, closes every connection, and cancels the runs they were waiting on. */
- (void)stop;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceServer_h */
