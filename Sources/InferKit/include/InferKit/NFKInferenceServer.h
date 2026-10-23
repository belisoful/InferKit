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
#import <InferKit/NFKModality.h>
#import <InferKit/NFKServerStatus.h>
#import <InferKit/NFKServerRoute.h>
#import <InferKit/NFKServerObservation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKModelCatalog;
@class NFKModelMemoryBudget;
@class NFKModelPreset;
@class NFKBackendTool;

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
	@constant   NFKInferenceServerErrorUnauthorized A request needed a key and carried none, or one the
				server does not hold. The reply is 401 with WWW-Authenticate: Bearer.
	@constant   NFKInferenceServerErrorBusy A request found its model's queue full
				(maximumQueuedRunsPerModel). The reply is 503 with a Retry-After of the estimated wait,
				and NFKRemoteErrorRetryAfterKey carries the same moment.
	@constant   NFKInferenceServerErrorInsufficientScope A request carried a key the server holds whose
				scopes do not include the route's. The reply is 403 with WWW-Authenticate: Bearer
				error="insufficient_scope".
	@constant   NFKInferenceServerErrorMisdirected A request from this machine named another host in its
				Host header, as a browser page does that reaches the server by DNS rebinding, and
				presented no key that opens the route. The reply is 421.
	@constant   NFKInferenceServerErrorNameConflict A name the call adds is taken: an alias or a preset
				names a model hosted under that name, or a model's MCP tool name is the name of another
				served tool.
	@constant   NFKInferenceServerErrorInvalidRequest A request's body has a field of the wrong type,
				such as a stream flag that is an array or a model that is a number. A null field is
				an absent one. The reply is 400, in the route's own error shape.
	@constant   NFKInferenceServerErrorOriginRefused A browser request's Origin names a site the server
				does not admit: not this machine, and neither the host the request was sent to nor, when
				allowedOrigins is set, one of those origins. A /v1 route replies 403 in its own error
				shape; /mcp replies 403 with a JSON-RPC error whose data carries this domain and code.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKInferenceServerErrorDomain, NFKInferenceServerError) {
	NFKInferenceServerErrorListenFailed = 1,
	NFKInferenceServerErrorAPIKeyRequired = 2,
	NFKInferenceServerErrorInvalidConfiguration = 3,
	NFKInferenceServerErrorModelNotFound = 4,
	NFKInferenceServerErrorUnauthorized = 5,
	NFKInferenceServerErrorBusy = 6,
	NFKInferenceServerErrorInsufficientScope = 7,
	NFKInferenceServerErrorMisdirected = 8,
	NFKInferenceServerErrorNameConflict = 9,
	NFKInferenceServerErrorInvalidRequest = 10,
	NFKInferenceServerErrorOriginRefused = 11,
};

/*! The name apiKey reads and writes among the server's keys. Introduced in InferKit 0.4.0. */
extern NSString * const NFKInferenceServerDefaultAPIKeyName;

/*! In the userInfo of a build admission refused: the hosted models that held memory, each a dictionary of
	"model" (its name), "bytes" (what it holds), and "reason", why admission could not release it:
	"pinned", "running", "queued", "grace", "permanent" (hosted with a ready backend), "loading", or
	"available" when it could go but releasing every such model frees too little. Introduced in InferKit
	0.4.0. */
extern NSString * const NFKInferenceServerAdmissionModelsKey;

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

/*! The bytes the runtime has allocated now, or a negative value when it cannot read them. A server reads
	it before and after it builds a hosted model's backend, and reports the difference as the model's
	resident bytes when the backend does not measure its own. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NSInteger allocatedByteCount;

/*! The most bytes the runtime has had allocated since resetPeakAllocatedByteCount, or a negative value
	when it cannot read them. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NSInteger peakAllocatedByteCount;

/*! Starts peakAllocatedByteCount again from the bytes allocated now. A server calls it before it builds
	a hosted model's backend. Introduced in InferKit 0.4.0. */
+ (void)resetPeakAllocatedByteCount;

/*! The bytes the runtime keeps cached for reuse now, which releaseCachedMemory returns, or a negative
	value when it cannot read them. Introduced in InferKit 0.4.0. */
@property (class, nonatomic, readonly) NSInteger cachedByteCount;

@end

/*! Builds a hosted model's backend: returns it, or nil with an error. A server calls it on a background
	queue at user-initiated quality of service when a request needs the model and its backend is not in
	memory, one build at a time per model. Swift imports it as a Sendable closure, so a closure written in
	main-actor code is nonisolated and runs on the server's queue. Introduced in InferKit 0.4.0. */
typedef id<NFKInferenceBackend> _Nullable (^ NS_SWIFT_SENDABLE NFKServedBackendProvider)(NSError * _Nullable * _Nullable error);

/*!
	@protocol   NFKServedBackendBuilder
	@abstract   An object that builds hosted models' backends, the throwing form of NFKServedBackendProvider.
	@discussion Swift imports the method as backend(forModelName:) throws, so a Swift builder throws its
				error. The protocol is Sendable in Swift: a server calls the method on a background queue
				at user-initiated quality of service, one build at a time per model, and a conformance
				declared in main-actor code runs nonisolated. One builder may serve several models.
				Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@protocol NFKServedBackendBuilder <NSObject>

/*! The backend for the model hosted under modelName, or nil with an error. */
- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName error:(NSError * _Nullable * _Nullable)error;

@end

/*!
	@class      NFKMemoryRelief
	@abstract   What -[NFKInferenceServer relieveMemoryByteCount:completionHandler:] freed, step by step.
	@discussion The steps run in order, and the cache steps stop once they have freed the requested bytes:
				- conversationCacheByteCount: the conversation caches of models with no run in flight;
				- expertCacheByteCount: the cached routed experts of models with no run in flight, down to
				  expertCacheReliefFloor each;
				- runtimeCacheByteCount: the runtimes' cached memory, released last because a runtime such
				  as MLX keeps the caches the steps above free until it returns its own, and only while no
				  hosted model has a run in flight or a backend being built. It counts the cache the
				  runtimes held before the steps above, so no byte is counted twice;
				- modelByteCount: the resident bytes of the idle models it released, least recently used
				  first, never a pinned model, one with a run in flight or queued, or one within
				  unloadGracePeriod of its build. Taken only when the cache steps fell short.
				Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKMemoryRelief : NSObject
@property (nonatomic, readonly) NSInteger requestedByteCount;
@property (nonatomic, readonly) NSInteger runtimeCacheByteCount;
@property (nonatomic, readonly) NSInteger conversationCacheByteCount;
@property (nonatomic, readonly) NSInteger expertCacheByteCount;
@property (nonatomic, readonly) NSInteger modelByteCount;
/*! The names of the models released, in the order they were released. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *releasedModelNames;
/*! The bytes every step freed together. */
@property (nonatomic, readonly) NSInteger byteCount;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@class      NFKInferenceServer
	@abstract   Serves the models this process hosts to clients on this machine and on the network.
	@discussion A server hosts backends by model name and answers HTTP under /v1, and Google's routes under
				/v1beta:

				- OpenAI-compatible routes: GET /models and /models/{name}, POST /chat/completions
				  (streamed as server-sent events when the body sets stream), /embeddings,
				  /audio/transcriptions, /audio/translations, /audio/speech, /images/generations, and
				  /images/edits. NFKRemoteBackend and the core's embedding, transcription, speech, and
				  image backends reach them unchanged, and so does any OpenAI client library.
				- OpenAI's Responses API: POST /responses (streamed as response.* events when the body
				  sets stream), and POST /responses/input_tokens. NFKRemoteResponsesBackend and Codex
				  reach it. Function tools and custom (free-form) tools are offered to the hosted
				  backend; built-in tools are not, since the server runs none. The server keeps no
				  responses: every reply has store false, previous_response_id is refused, and
				  GET and DELETE /responses/{id} answer 404, so each request carries its whole
				  conversation, as Codex sends it.
				- Anthropic's Messages API: POST /messages (streamed as message_start,
				  content_block_*, message_delta, and message_stop events when the body sets stream),
				  and POST /messages/count_tokens. NFKAnthropicBackend and Claude Code reach it. These
				  routes also take the key as x-api-key, accept any anthropic-version, and answer
				  errors in Anthropic's shape ({type: "error", error: {type, message}}). GET /models and
				  /models/{name} with an anthropic-version header answer in Anthropic's model-list
				  shape and take x-api-key too.
				- Google's generateContent API under /v1beta/models: POST {model}:generateContent,
				  {model}:streamGenerateContent (server-sent events with alt=sse), and
				  {model}:countTokens, and GET /v1beta/models and /v1beta/models/{model}.
				  NFKGeminiGenerateContentBackend and Gemini CLI reach it. These routes also take the key
				  as x-goog-api-key or the key query parameter, and answer errors in Google's shape
				  ({error: {code, message, status}}).
				- The native route, POST /inferkit/run, which carries a whole NFKInferenceRequest and
				  NFKInferenceResult in InferKit's tagged JSON form, so every key reaches the hosted
				  backend: pixel buffers, masks, multi-arrays, audio and video files, and the core's
				  value types. NFKRemoteInferKitBackend is its client and streams the job's progress
				  and partial results.
				- The status route, GET /inferkit/status, which reports each model's load and the
				  machine's state for a load balancer or a dashboard (see Status).
				- The catalog route, GET /inferkit/catalog, which answers the catalog the server builds
				  from as a manifest (NFKModelCatalog's JSON form): modelCatalog's descriptors and the
				  descriptors of the models hosted from another catalog.
				- The estimate route, POST /inferkit/estimate, which takes {model, options} and answers
				  the model's memory estimate under those load options against modelCatalog's
				  memoryBudget, this machine's by default, with the constraints the budget leaves of
				  each option (NFKModelMemory.h). model is a
				  catalog identifier, identifier@preset, or a model the server hosts from the catalog,
				  whose options the given ones override. With reportsHostDetails NO the reply carries the
				  estimate alone, computed without a budget, and leaves out the budget and the
				  constraints, whose reasons quote it.

				An app adds its own routes under a path prefix of its choosing with
				addRouteForPathPrefix:handler:error:; they share the listener and the key checks.

				Beside them, POST /mcp serves the tools added with addTool: over the Model Context
				Protocol (Streamable HTTP), so an MCP client reaches them: Claude Code, Claude
				Desktop, an agent framework, or NFKMCPClient. initialize issues an Mcp-Session-Id, and
				the session starts at the client's first request that carries it. A DELETE ends the
				session, and a GET opens its event stream, on which the server pings the client every
				MCPPingInterval. A session ends when its client misses MCPMissedPingLimit pings, or,
				with no stream open, sends nothing for MCPSessionIdleTimeout; its id is then answered
				404. A client that never sends the id back is served without a session, and the id it
				was issued lapses unreported. MCPSessions lists the sessions. Only the added tools are
				listed; a hosted model is a tool when an NFKBackendTool for it is added, or when
				addToolForModelName:outputModality:purpose:error: adds one named NFKModelToolName of the
				model's name. A tool's text,
				the images it shows the model, and the audio it generates come back as MCP content. The
				route takes a key with NFKServerKeyScopeMCP, and refuses a browser request whose Origin
				names another host.

				A request names its model in the body's model field, on every route. When the server
				hosts exactly one model, a request that names none runs on it. A coding agent pointed
				at the server therefore names a hosted model: Claude Code through ANTHROPIC_MODEL (and
				the models it uses for background work), Codex through its model setting.

				Names, presets, and aliases. A model field resolves in one place, in this order:
				1. The name of a hosted model or of an alias (addAlias:forModelName:requestDefaults:error:,
				   and the aliases addPreset:forBaseModelName:error: adds).
				2. base@preset, split at the last NFKModelPresetSeparator, where base is a hosted model
				   built from the catalog and preset is a suggested preset of its descriptor that sets
				   only request defaults. The run goes to the base's backend.
				3. Any other name is not served (NFKInferenceServerErrorModelNotFound), so a name such as
				   claude-3@20240620 stays whole.
				A preset that sets load options is served only once addPreset:forBaseModelName:error: adds
				it. An alias's and a preset's request defaults fill the parameters a request leaves out.
				An OpenAI-shaped reply's model field is the name the request sent, and every run reply
				carries X-InferKit-Base-Model, the base model the name reaches (a preset's base, an
				alias's model, or the model itself), and X-InferKit-Preset when a preset applies. The
				native route's reply carries model, base, preset, and loadOptions beside the result.

				A hosted backend that implements tokenCountForRequest:error: answers the two
				token-counting routes; any other answers 400. A count takes the model's run slot, so it
				waits behind the runs queued ahead of it: the backend contract promises that a count
				never overlaps a run of the same backend, and a model that is not resident is built
				before it counts.

				Requests are checked before they run:
				- A field of a JSON body that is present carries its documented type. A null field is
				  an absent one, as the providers' APIs read it, since clients and SDKs send null for
				  an optional parameter they leave unset; a required field that is null is missing. A
				  wrong type is refused with 400 (NFKInferenceServerErrorInvalidRequest) in the
				  route's error shape, before any run starts. So is an input or parameter that reaches
				  the backend with a value of another kind than its contract key carries
				  (NFKModelValueTypeForKey), such as a temperature that is a string.
				- A JSON route reads its body as JSON whatever its Content-Type, so curl -d, which
				  sends a form type, is served. A body that is not a JSON object is refused with 400.
				- An Objective-C exception answers 500 and leaves the server serving when it is raised
				  by a route, an app route's handler, the code that writes a reply, a model's build, or
				  a backend's runInferenceForRequest:error: or submitInferenceJobForRequest: call on the
				  server's own queue. These job backends of the core catch an exception their work
				  raises on their own queues and fail the job with kNFKError_InferenceBackendFailure,
				  which the route answers with 500: the remote and provider backends, through the
				  streams of NFKRemoteTransport and the work they queue; NFKCoreMLLanguageBackend;
				  NFKAsyncGenerationBackend; NFKBalancedBackend; NFKSpeechRecognitionBackend's
				  recognition callbacks; NFKPluginBackend, in reading a plugin's reply; and the run
				  NFKInferenceSubmit queues for a synchronous backend. A third-party
				  backend's work on its own queues is that backend's to guard: an exception raised
				  there ends the process.

				Access:
				- A client presents a key as a Bearer token, which is what the remote backends' apiKey
				  sends; on the Messages routes, as an x-api-key header, which is what
				  NFKAnthropicBackend and Claude Code send; or on Google's routes, as an x-goog-api-key
				  header or a key query parameter, which is what Gemini CLI sends. The server holds several keys, each named and opening a set of scopes
				  (NFKServerKeyScope): addAPIKey:named:scopes: adds one, removeAPIKeyNamed: revokes it
				  for the next request. apiKey is the key named NFKInferenceServerDefaultAPIKeyName
				  with every scope.
				- A client on another machine needs a key whose scopes include the route's unless
				  requiresAPIKey is NO, which opens the server to anyone who can reach it.
				- A client on this machine (a loopback peer) needs no key for the scopes in
				  loopbackExemptScopes, every scope by default. requiresAPIKeyOnLoopback set to YES
				  empties the set.
				- A loopback request whose Host header names another host needs a key that opens the
				  route, whatever the exemptions; without one it is refused with 421
				  (NFKInferenceServerErrorMisdirected). A browser page that reaches the server through
				  DNS rebinding sends its own host name, so it gets no key-free access. localhost,
				  127.0.0.1, and [::1] are loopback names. This Host check is what stops DNS
				  rebinding: such a page's Origin names the same host as its Host header.
				- A browser request to a /v1 route or /mcp whose Origin the server does not admit is
				  refused with 403 (NFKInferenceServerErrorOriginRefused), whatever key it presents.
				  An origin on localhost, 127.0.0.1, or [::1] is admitted. With allowedOrigins empty,
				  so is an origin whose host is the host the request was sent to; with it set, so is
				  each origin it lists, and no other. A client that is not a browser sends no Origin
				  and passes, so a local agent keeps its key-free access. A browser's cross-origin
				  request always carries an Origin, so this check keeps a page on another site from
				  starting a run.
				- A reverse proxy that rewrites Host, as nginx does by default (proxy_pass sends the
				  upstream's address), makes a same-origin page's Origin differ from the Host the server
				  sees, and the page is refused. Name the page's origin in allowedOrigins, such as
				  https://models.example.com, or have the proxy pass the original Host.
				- startWithError: fails with NFKInferenceServerErrorAPIKeyRequired when the
				  configuration requires a key and none is set.
				Keys are compared by their SHA-256 digests in constant time. Neither a key's value nor
				its name appears in a reply or the status route; apiKeyNames and
				scopesForAPIKeyNamed: list them in process. A reverse proxy on this machine connects
				from loopback, so behind one empty loopbackExemptScopes or listen on loopback only.

				Setting TLSIdentity serves HTTPS with that identity. A client's URL session trusts
				the certificate only if the system does, so a self-signed identity needs the client to
				install it or to pin its SHA-256 fingerprint. NFKServerIdentity makes and keeps such an
				identity; a client pins it with pinnedCertificateSHA256. servesPlainHTTPOnLoopback adds
				a second listener, plain HTTP on the loopback interface only, for local clients that
				trust no self-signed certificate (agent CLIs registered with localMCPURL). Both
				listeners serve the same models, routes, keys, MCP sessions, and observation.

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
				written. A model hosted from the catalog also carries catalogModel, its descriptor's
				identifier, and loadOptions, the load options it builds with (the backend's
				loadOptionValues once it reports them); a preset's model carries base and preset; and
				every model carries aliases, the names that reach it. The status route lists each alias
				under aliases with its model, preset, and request defaults.

				Residency. A model hosted with addBackendProvider:forModelName: (or
				addBackendBuilder:forModelName:) is listed before its
				backend exists, and its first request builds it:
				- A request that finds the model not loaded starts the build and waits in the model's
				  queue. Requests that arrive during the build wait for the same build.
				- A build that fails fails every request waiting for it with the provider's error and
				  leaves the model not loaded, so the next request builds again.
				- unloadModelNamed: releases the backend and keeps the model hosted. It waits until the
				  model has no run in flight or queued, and never cuts a run. A backend that implements
				  unload is unloaded before the server releases it, so its memory is freed even where
				  another object still holds the backend.
				- A model hosted with addBackend:forModelName: is resident from the start and stays
				  resident, since the server has no way to build its backend again.
				- A model that is not resident reports ready, since a request builds it, and describes
				  the backend it last held: its identifier, inputs, parameters, outputs, and modelInfo.
				Each model's "residency" object carries state (not_loaded, loading, or resident),
				unloadable, pinned, last_used (seconds since 1970), idle_timeout_seconds (absent when
				the model is not unloaded for idleness), load_seconds, load_error, loads, unloads, and
				resident_bytes and peak_bytes from the last load that measured them. A backend that measures its own load
				(residentByteCount above 0) gives both figures. Otherwise the server measures the build:
				it resets each registered runtime's peak (NFKServingRuntimeStatus), reads the bytes
				allocated before and after, and reports the rise. The two figures are absent when neither
				measured an allocation, and they stay after an unload. The runtime peak is one reading
				for the whole process, so overlapping loads and runs fall into each other's figures, and
				arrays a loader leaves for the first run to read are not counted.

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
				- A pinned model (setPinned:forModelName:) is never unloaded by either option.
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
				or critical, and this process's footprint); hardware, the machine's NFKHardwareProfile
				snapshot (dictionaryRepresentation), from which a client evaluates a model's
				constraints for this machine; runtimes, each linked runtime's own
				report (NFKServingRuntimeStatus), such as MLX's active and cached memory; cpu (usage since the previous status
				reading, and the load averages); storage on the volume of storageDirectoryURL; and,
				on macOS, gpu.utilization from the IORegistry's undocumented "Device Utilization %",
				absent when it cannot be read. GPU utilization covers every process on the machine
				and stays near 1 while one run executes, so a load balancer routes on the queue
				figures. Run replies carry X-InferKit-Thermal-State, and the Bonjour
				advertisement adds the chip and the physical memory. Setting reportsHostDetails to
				NO leaves out the host object, the thermal header, those advertisement entries, and
				the estimate route's budget and constraints.
				The status route needs a key with NFKServerKeyScopeStatus under the access rules above.
				currentStatus reads the same status in process.

				Observation. The delegate (NFKInferenceServerDelegate) learns, on delegateQueue and in
				order, when a run starts and ends (its model, route, client, key name, wait, duration,
				and outcome), when a model loads, fails to load, or unloads and why (a request, the
				host, idleness, pressure, relief, admission, or removal), when a client's connection
				passes its first key check, and when an MCP session starts or ends.

				The keys, requiresAPIKey, loopbackExemptScopes, allowedOrigins, reportsHostDetails,
				storageDirectoryURL, the app routes, and the hosted models take effect on the next
				request, and
				maximumConcurrentRunsPerModel and maximumQueuedRunsPerModel at once for every hosted
				model. The other settings take effect on the next start; the advertisement's host
				entries follow reportsHostDetails as it was at start. stop keeps the hosted models and
				their residency, so a listener setting changes with stop, change, and start on the same
				object.
				Introduced in InferKit 0.4.0.
*/
@interface NFKInferenceServer : NSObject

/*! The port to listen on. Defaults to NFKInferenceServerDefaultPort; 0 lets the system choose. */
@property (nonatomic, assign) uint16_t port;

/*! Listens on the loopback interface only, so no other machine can connect. Defaults to NO. */
@property (nonatomic, assign) BOOL loopbackOnly;

/*! The key named NFKInferenceServerDefaultAPIKeyName, which a client presents as a Bearer token.
	Setting it adds or replaces that key with every scope; nil removes it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! Whether a client on another machine must present a key with the route's scope. Defaults to YES. */
@property (nonatomic, assign) BOOL requiresAPIKey;

/*! Whether some scope requires a key from a client on this machine too, which is to say
	loopbackExemptScopes is not NFKServerKeyScopeAll. Setting YES empties loopbackExemptScopes and NO
	fills it. Defaults to NO. */
@property (nonatomic, assign) BOOL requiresAPIKeyOnLoopback;

/*! The scopes a client on this machine reaches without a key. Defaults to NFKServerKeyScopeAll.
	Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NFKServerKeyScope loopbackExemptScopes;

/*!
	@property   allowedOrigins
	@abstract   The browser origins the /v1 routes and /mcp admit, as scheme://host[:port] strings.
	@discussion Empty, the default, admits an origin whose host is the host the request was sent to.
				Set, it admits exactly the origins it lists, compared without case or a trailing slash,
				such as https://models.example.com for a page served behind a reverse proxy that
				rewrites Host. An origin on localhost, 127.0.0.1, or [::1] is admitted either way, and a
				request without an Origin header is not a browser's and passes. Takes effect on the next
				request. Introduced in InferKit 0.4.0.
*/
@property (atomic, copy) NSSet<NSString *> *allowedOrigins;

/*!
	@method     addAPIKey:named:scopes:
	@abstract   Adds a key that opens scopes, replacing the key under the same name.
	@discussion The next request sees it. A request that presents a key under two names takes the first
				name in sorted order. Adding NFKInferenceServerDefaultAPIKeyName replaces apiKey.
				Introduced in InferKit 0.4.0.
*/
- (void)addAPIKey:(NSString *)key named:(NSString *)name scopes:(NFKServerKeyScope)scopes
	NS_SWIFT_NAME(addAPIKey(_:named:scopes:));

/*! Revokes the key under name: the next request that presents it is treated as presenting no key, and
	the MCP sessions it started end. Where a route needs a key, that request is refused; a loopback
	client in loopbackExemptScopes, or any client while requiresAPIKey is NO, is still served. A reply
	already being written finishes. Introduced in InferKit 0.4.0. */
- (void)removeAPIKeyNamed:(NSString *)name;

/*! The names of the server's keys, sorted. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *apiKeyNames;

/*! The scopes of the key under name, or 0 when there is none. Introduced in InferKit 0.4.0. */
- (NFKServerKeyScope)scopesForAPIKeyNamed:(NSString *)name;

/*! The identity to serve HTTPS with, or NULL for plain HTTP. The server retains it. */
@property (nonatomic, assign, nullable) SecIdentityRef TLSIdentity;

/*!
	@property   servesPlainHTTPOnLoopback
	@abstract   Whether the server also listens for plain HTTP on the loopback interface. Defaults to NO.
	@discussion The second listener binds plainLoopbackPort on loopback only and never uses TLS, so a
				client on this machine reaches the server without trusting its certificate while
				clients on other machines use HTTPS. It is never advertised. A request on it is a
				loopback request, so loopbackExemptScopes applies. While it runs, localBaseURL and
				localMCPURL name it. startWithError: starts both listeners or neither. Introduced in
				InferKit 0.4.0.
*/
@property (nonatomic, assign) BOOL servesPlainHTTPOnLoopback;

/*! The port of the plain loopback listener. Defaults to 0, which lets the system choose. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) uint16_t plainLoopbackPort;

/*! Whether the server advertises itself over Bonjour. Defaults to YES; a loopback-only server is
	never advertised. */
@property (nonatomic, assign) BOOL advertisesService;

/*! The advertised name. nil uses the computer's name. */
@property (nonatomic, copy, nullable) NSString *serviceName;

/*! The largest request body accepted, in bytes. Defaults to 256 MiB. */
@property (nonatomic, assign) NSUInteger maximumRequestBodyBytes;

/*! How many runs one hosted backend serves at once. Defaults to 1, since a backend is not assumed to
	be safe to run concurrently. A change applies to every hosted model: a raised limit starts waiting
	runs at once, and a lowered one takes effect as running runs end, never cancelling one. */
@property (nonatomic, assign) NSUInteger maximumConcurrentRunsPerModel;

/*! How many runs wait for one hosted backend before later requests are refused with
	NFKInferenceServerErrorBusy. Defaults to 0, which queues without limit. A change applies to every
	hosted model; runs already queued beyond a lowered limit stay queued, and later requests are refused
	until the queue falls below it. */
@property (nonatomic, assign) NSUInteger maximumQueuedRunsPerModel;

/*! Whether the status route, run replies, the estimate route, and the Bonjour advertisement describe
	this machine. Defaults to YES; NO leaves out every host detail, the estimate route's budget and
	constraints included, and keeps the models' load. */
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
	nil restores the default. Called on the thread that reads the pressure: the server's residency queue,
	a request's queue, or the caller of memoryPressure. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, null_resettable) double (^ NS_SWIFT_SENDABLE memoryPressureProvider)(void);

/*! The pressure the unloading options act on now: the provider's reading, held while it falls (see
	pressureUnloadHysteresis). 0 while unloadsUnderMemoryPressure is NO. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly) double memoryPressure;

/*! Whether the server is listening. */
@property (nonatomic, readonly, getter=isRunning) BOOL running;

/*! The port the server is listening on, which is the chosen one when port is 0; 0 while stopped. */
@property (nonatomic, readonly) uint16_t listeningPort;

/*! The port the plain loopback listener is bound to, which is the chosen one when plainLoopbackPort is
	0; 0 while it is not running. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly) uint16_t plainLoopbackListeningPort;

/*! The base URL a client on this machine uses, or nil while stopped: http://localhost:<port>/v1 on the
	plain loopback listener while it runs, otherwise on listeningPort with https when TLSIdentity is
	set. */
@property (nonatomic, readonly, copy, nullable) NSURL *localBaseURL;

/*! The base URL a client on another machine uses: https (or http without TLSIdentity), this machine's
	host name, and listeningPort. nil while stopped or loopbackOnly. Discovery finds the same listener.
	Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy, nullable) NSURL *networkBaseURL;

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

/*! Hosts a model whose backend builder builds when a request first needs it, as
	addBackendProvider:forModelName: does. The server keeps the builder while it hosts the model.
	Introduced in InferKit 0.4.0. */
- (void)addBackendBuilder:(id<NFKServedBackendBuilder>)builder forModelName:(NSString *)modelName
	NS_SWIFT_NAME(addBackendBuilder(_:forModelName:));

/*!
	@method     addModelWithIdentifier:options:catalog:forModelName:error:
	@abstract   Hosts a catalog model under a model name, built through the catalog's build entry when a
				request first needs it.
	@discussion identifier is a catalog identifier, or identifier@preset for a preset its descriptor
				lists. The options are checked now: a model the catalog lacks fails with
				NFKModelCatalogErrorUnknownModel and an option that does not resolve with
				NFKModelCatalogErrorInvalidOptions, and nothing is hosted. The machine's requirements and
				the memory budget are checked when the backend is built, as
				-[NFKModelCatalog backendForModelIdentifier:options:error:] checks them, so a combination
				that does not fit fails the first request with NFKModelCatalogErrorDoesNotFit. The model
				is hosted as addBackendProvider:forModelName: hosts one, replacing any model under the
				name, and keeps its options for each build. A nil catalog is modelCatalog. Introduced in
				InferKit 0.4.0.
*/
- (BOOL)addModelWithIdentifier:(NSString *)identifier
					   options:(nullable NSDictionary<NSString *, id> *)options
					   catalog:(nullable NFKModelCatalog *)catalog
				  forModelName:(NSString *)modelName
						 error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(addModel(identifier:options:catalog:forModelName:));

/*!
	@method     addPreset:forBaseModelName:error:
	@abstract   Serves a hosted model under a preset, as base@preset.
	@discussion The preset applies to the model hosted under baseModelName; its baseModelIdentifier is not
				read.
				- A preset that sets only request defaults becomes an alias of the base: a request naming
				  base@preset runs on the base's backend with the preset's request defaults.
				- A preset that sets load options needs a base hosted with
				  addModelWithIdentifier:options:catalog:forModelName:error:. Its load options are merged
				  over the base's, and the result's canonicalLoadOptionsKey: is compared with the base's
				  and with those of the base's other preset models. Equal to one → an alias of that model,
				  which shares its backend. Equal to none → a separate model hosted under base@preset,
				  built on demand from the merged options, with its own load and residency.
				Adding a preset again replaces it, an alias or a model under base@preset alike. Fails with
				NFKInferenceServerErrorModelNotFound when no model is hosted under baseModelName,
				NFKInferenceServerErrorInvalidConfiguration when the preset sets load options and the base
				was not hosted from a catalog, NFKModelCatalogErrorInvalidOptions when the merged options
				or the request defaults do not resolve, and NFKInferenceServerErrorNameConflict when a
				model other than this preset's is hosted under base@preset. Introduced in InferKit 0.4.0.
*/
- (BOOL)addPreset:(NFKModelPreset *)preset forBaseModelName:(NSString *)baseModelName error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(addPreset(_:forBaseModelName:));

/*!
	@method     addAlias:forModelName:requestDefaults:error:
	@abstract   Serves a hosted model under a second name.
	@discussion A request naming alias runs on the model's backend, with requestDefaults filled where it
				leaves a parameter out. Adding an alias again replaces it. Hosting a model under the
				alias's name later removes the alias. The alias stays when its model is removed, and a
				request naming it fails until a model is hosted under modelName again. Fails with
				NFKInferenceServerErrorModelNotFound when no model is hosted under modelName, and with
				NFKInferenceServerErrorNameConflict when a model is hosted under alias. Introduced in
				InferKit 0.4.0.
*/
- (BOOL)addAlias:(NSString *)alias
	forModelName:(NSString *)modelName
 requestDefaults:(nullable NSDictionary<NSString *, id> *)requestDefaults
		   error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(addAlias(_:forModelName:requestDefaults:));

/*! Stops serving the alias, a preset's alias included. Introduced in InferKit 0.4.0. */
- (void)removeAlias:(NSString *)alias;

/*! The aliases, the presets' aliases included, sorted. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *aliasNames;

/*! The hosted model an alias names, or nil when there is no such alias. Introduced in InferKit 0.4.0. */
- (nullable NSString *)modelNameForAlias:(NSString *)alias;

/*! The catalog the catalog route serves and the estimate route and
	addModelWithIdentifier:options:catalog:forModelName:error: read. Defaults to
	NFKModelCatalog.sharedCatalog; nil restores it. Introduced in InferKit 0.4.0. */
@property (nonatomic, strong, null_resettable) NFKModelCatalog *modelCatalog;

/*!
	@method     loadModelNamed:completionHandler:
	@abstract   Builds a provider-hosted model's backend ahead of its first request.
	@discussion The handler receives nil once the model is resident, or the build's error. A model that
				is already resident, or hosted with a ready backend, completes at once; one being built
				completes with that build. Cancels an unload that is waiting for runs to end. The handler
				runs on a background queue. Introduced in InferKit 0.4.0.
*/
- (void)loadModelNamed:(NSString *)modelName
	 completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSError * _Nullable error))completionHandler;

/*!
	@method     unloadModelNamed:
	@abstract   Releases a provider-hosted model's backend and keeps the model hosted.
	@discussion Returns at once. A model with a run in flight or queued, or a build in progress, is
				unloaded when the last of them ends; its next request builds it again. The server calls the
				backend's unload, when it implements one, before it releases the backend. Returns NO when
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

/*!
	@method     setPinned:forModelName:
	@abstract   Pins or unpins a model, by name, hosted now or later.
	@discussion A pinned model is never unloaded by idle or memory-pressure unloading, and its idle
				timeout reads INFINITY. unloadModelNamed: still unloads it. Pinning builds nothing: a
				pinned model that is not loaded stays so until a request or loadModelNamed: builds it.
				The pin is kept when the model is removed and applies if the name is hosted again. The
				status route reports it as the residency object's pinned. Introduced in InferKit 0.4.0.
*/
- (void)setPinned:(BOOL)pinned forModelName:(NSString *)modelName;

/*! The names of the pinned models, sorted. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *pinnedModelNames;

/*! Applies the unloading options now, as the periodic check does, and returns the names of the models it
	unloaded. Introduced in InferKit 0.4.0. */
- (NSArray<NSString *> *)unloadIdleModels;

/*!
	@method     relieveMemoryByteCount:completionHandler:
	@abstract   Frees at least byteCount bytes where it can, releasing a model only after the caches.
	@discussion Climbs the steps NFKMemoryRelief lists until they have freed byteCount bytes, and calls
				completionHandler on a background queue with what each step freed. A released model is
				reported with NFKServerModelEventReasonRelief. Memory-pressure unloading climbs the
				caches' steps the same way before it releases a model. Introduced in InferKit 0.4.0.
*/
- (void)relieveMemoryByteCount:(NSInteger)byteCount
			 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NFKMemoryRelief *relief))completionHandler;

/*! The cached routed experts relief keeps in each model, in bytes. Defaults to 0. Introduced in InferKit
	0.4.0. */
@property (atomic, assign) NSInteger expertCacheReliefFloor;

/*!
	@property   admitsModelsByMemory
	@abstract   Whether a build first makes room for the model beside the models already resident.
	@discussion Defaults to YES. Before a hosted model builds, the server holds the build beside what the
				other hosted models hold (their measured resident bytes, and what admission reserved for
				builds under way) to admissionMemoryBudget, its reserve included:
				- a catalog model whose residency is automatic loads in the residency the planner chooses
				  for the room the budget leaves: resident where that fits, else staged, paged, or streamed;
				- any other model is admitted where its estimated peak fits (a catalog model's estimate, or
				  the peak its last load measured).
				When the build does not fit, idle models go, least recently used first, reported with
				NFKServerModelEventReasonAdmission, when releasing them frees enough, and the runtimes'
				caches are released after them. Admission counts measured resident bytes, which the
				conversation and expert caches do not change, so it leaves those caches in place.
				A build that still does not fit waits up to admissionWaitInterval for a model to become
				releasable, then fails with NFKModelCatalogErrorDoesNotFit in NFKModelCatalogErrorDomain,
				its userInfo naming the models considered under NFKInferenceServerAdmissionModelsKey. A
				build whose model is removed while it waits ends there. A build whose peak is unknown, or a
				budget the machine does not report, is admitted.
				Introduced in InferKit 0.4.0.
*/
@property (atomic, assign) BOOL admitsModelsByMemory;

/*! The budget admission holds the hosted models to, or nil for +[NFKModelMemoryBudget currentBudget]. A
	catalog model whose catalog sets a smaller memoryBudget is held to that one. Introduced in InferKit
	0.4.0. */
@property (atomic, copy, nullable) NFKModelMemoryBudget *admissionMemoryBudget;

/*! How long a build that does not fit waits for a model to become releasable, in seconds. Defaults to 0,
	which refuses at once. Introduced in InferKit 0.4.0. */
@property (atomic, assign) NSTimeInterval admissionWaitInterval;

/*! Stops hosting the model. Its queued runs fail; its running ones finish. The server calls the backend's
	unload, when it implements one, so its memory is freed once the running runs end. Replacing a model
	under the same name does the same for the backend it replaces. */
- (void)removeBackendForModelName:(NSString *)modelName;

/*! Adds a class, found by name at runtime, whose NFKServingRuntimeStatus report every server's
	status route carries. A name whose class is absent or does not adopt the protocol is skipped. */
+ (void)registerRuntimeStatusClassName:(NSString *)className;

/*! The backend hosted under a model name, or nil, which a provider-hosted model that is not resident
	also answers. A run made on the backend directly is not one of the server's runs: it takes no slot,
	and an idle or pressure unload may release the backend during it. */
- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName;

/*!
	@method     submitInferenceJobForRequest:modelName:
	@abstract   Runs a request on a hosted model in this process, as one of the server's runs.
	@discussion The run takes the path a request on the native route takes, without the listener:
				- The name resolves as a route's model field does: a hosted model, an alias, or
				  "model@preset", with their request defaults. nil names the only hosted model.
				- A model that is not resident builds, admitted against the memory budget with
				  admissionWaitInterval's wait, as a route's request builds it.
				- The run waits for a slot under maximumConcurrentRunsPerModel and counts against
				  maximumQueuedRunsPerModel. An idle, pressure, or admission release never takes the model
				  while the run holds or waits for a slot.
				- The delegate receives the run's events, with NFKServerRunRouteInProcess as the route, an
				  empty client address, and no key.
				The job reports Running when the run takes its slot, and the backend's progress and partial
				results when it reports them. Cancelling the job cancels the run, queued or running. A name
				that resolves to no model, a request whose input or parameter has another kind than its
				contract key, and a full queue fail the job before it runs, with the error a route answers.
				The server need not be started, and -stop leaves these runs to finish. The job's handlers run
				on the server's queues. Introduced in InferKit 0.4.0.
*/
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
										modelName:(nullable NSString *)modelName NS_SWIFT_NAME(submitInferenceJob(for:modelName:));

/*! The hosted model names, sorted. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *modelNames;

/*! Serves a tool over MCP, replacing any tool of the same name. Introduced in InferKit 0.4.0. */
- (void)addTool:(NFKInferenceTool *)tool;

/*! Stops serving the tool of that name over MCP. Introduced in InferKit 0.4.0. */
- (void)removeToolNamed:(NSString *)name;

/*!
	@method     addToolForModelName:outputModality:purpose:error:
	@abstract   Serves a model as an MCP tool named NFKModelToolName(modelName).
	@discussion modelName is any name a request can name: a hosted model, an alias, or base@preset. Each
				call resolves the name as a request does and waits in the model's queue, so the tool shares
				the model's backend, limits, and preset defaults. The tool has NFKBackendTool's default
				description and schema for outputModality, and purpose completes "Use it when …". The
				server keeps a table from each such tool's name to its model name (modelNameForToolName:).
				Two model names can map to one tool name, as "a@b" and "a__b" do; adding a tool whose name
				an added tool already has is refused with NFKInferenceServerErrorNameConflict, unless that
				tool is the same model's, which it replaces. Fails with
				NFKInferenceServerErrorModelNotFound when no request could name the model now. Returns the
				tool. Introduced in InferKit 0.4.0.
*/
- (nullable NFKBackendTool *)addToolForModelName:(NSString *)modelName
								  outputModality:(NFKModality)outputModality
										 purpose:(nullable NSString *)purpose
										   error:(NSError * _Nullable *)error
	NS_SWIFT_NAME(addTool(forModelName:outputModality:purpose:));

/*! The model name a tool added with addToolForModelName:outputModality:purpose:error: runs, or nil.
	Introduced in InferKit 0.4.0. */
- (nullable NSString *)modelNameForToolName:(NSString *)toolName;

/*! The tools served over MCP, in the order they were added. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

/*! What an MCP client's model is told about the server as a whole when it connects, such as which
	tools to prefer, or nil for nothing. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, nullable) NSString *MCPInstructions;

/*! The MCP endpoint a client on this machine uses, on the listener localBaseURL names, or nil while
	stopped. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy, nullable) NSURL *localMCPURL;

/*! The MCP endpoint a client on another machine uses, on the listener networkBaseURL names, or nil
	while stopped or loopbackOnly. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy, nullable) NSURL *networkMCPURL;

/*!
	@method     addRouteForPathPrefix:handler:error:
	@abstract   Serves an app's own requests under a path prefix, beside the server's routes.
	@discussion The route shares the server's listener, TLS identity, Bonjour advertisement, body limit
				(maximumRequestBodyBytes), and key checks. A request whose path is the prefix, or lies
				below it, reaches handler on a background queue after its key is checked; the longest
				matching prefix wins, and a path no route matches is answered 404. The prefix starts
				with "/" and names one or more non-empty segments; a trailing "/" is added. The root and
				every prefix under /v1 or /mcp, in any case, belong to the server and are refused with
				NFKInferenceServerErrorInvalidConfiguration. Adding a prefix again replaces its handler.
				The handler checks the request's method itself, and a browser-facing route checks the
				Origin header itself. Takes effect on the next request. Introduced in InferKit 0.4.0.
*/
- (BOOL)addRouteForPathPrefix:(NSString *)pathPrefix
					  handler:(NFKServerRouteHandler)handler
						error:(NSError * _Nullable * _Nullable)error
	NS_SWIFT_NAME(addRoute(pathPrefix:handler:));

/*! Stops serving the route under pathPrefix, given as it was added or as routePathPrefixes lists it.
	Introduced in InferKit 0.4.0. */
- (void)removeRouteForPathPrefix:(NSString *)pathPrefix;

/*! The app routes' prefixes, each with its trailing "/", sorted. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *routePathPrefixes;

/*! The seconds an MCP session with no stream open may go without a request before it ends. Defaults
	to 1800; 0 never ends a session for idleness. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval MCPSessionIdleTimeout;

/*! The seconds between the pings the server sends on each MCP session's open stream. Defaults to 30;
	0 sends none, and a session with its stream open then never ends on its own. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval MCPPingInterval;

/*! The unanswered pings after which an MCP session ends. Defaults to 3; at least 1. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NSUInteger MCPMissedPingLimit;

/*! The active MCP sessions, oldest first, as they stand now. A session starts at its client's first
	request that carries the Mcp-Session-Id initialize issued, so a client that never sends the id back
	is not listed. Introduced in InferKit 0.4.0. */
@property (nonatomic, readonly, copy) NSArray<NFKServerMCPSession *> *MCPSessions;

/*! Ends the MCP session with the identifier, closing its stream; the client's next request is answered
	404 and it initializes again. NO when no active session has the identifier. Introduced in InferKit
	0.4.0. */
- (BOOL)endMCPSessionWithIdentifier:(NSString *)identifier;

/*! Receives the server's runs, model loads and unloads, client connections, and MCP sessions in
	process. An MCP session starts, and the delegate learns of it, at its client's first request that
	carries the issued Mcp-Session-Id. Held weakly. Introduced in InferKit 0.4.0. */
@property (atomic, weak, nullable) id<NFKInferenceServerDelegate> delegate;

/*! The queue the delegate's methods run on. Defaults to a serial queue of the server's, so the events
	arrive in order; nil restores it. A concurrent queue gives up that order. Introduced in InferKit
	0.4.0. */
@property (atomic, strong, null_resettable) dispatch_queue_t delegateQueue;

/*! The status the status route would answer now, read in process: each model's load and residency, and
	the host details when reportsHostDetails is set. Introduced in InferKit 0.4.0. */
- (NFKServerStatus *)currentStatus;

/*!
	@method     startWithError:
	@abstract   Binds the port and starts serving; returns NO with an error when it cannot.
	@discussion Blocks until the listener is ready, which on a free port takes milliseconds. Fails
				with NFKInferenceServerErrorAPIKeyRequired or
				NFKInferenceServerErrorInvalidConfiguration before binding when the settings
				require it, and with NFKInferenceServerErrorListenFailed when the port is taken.
*/
- (BOOL)startWithError:(NSError * _Nullable *)outError;

/*! Stops listening, closes every connection, cancels the runs they were waiting on, and ends every MCP
	session. It returns once the ports are free, so a start on the same port can follow at once. The
	hosted models, their residency and counts, the tools, and every setting stay, so startWithError: on
	the same object listens again with the settings as they are then: a port, loopbackOnly,
	advertisesService, TLSIdentity, or plain loopback change is stop, change, start. */
- (void)stop;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceServer_h */
