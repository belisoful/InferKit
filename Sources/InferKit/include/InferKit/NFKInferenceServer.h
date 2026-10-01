//
//  NFKInferenceServer.h
//  InferKit
//

#ifndef NFKInferenceServer_h
#define NFKInferenceServer_h

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <InferKit/NFKInferenceBackend.h>

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
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKInferenceServerErrorDomain, NFKInferenceServerError) {
	NFKInferenceServerErrorListenFailed = 1,
	NFKInferenceServerErrorAPIKeyRequired = 2,
	NFKInferenceServerErrorInvalidConfiguration = 3,
	NFKInferenceServerErrorModelNotFound = 4,
	NFKInferenceServerErrorUnauthorized = 5,
};

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
				install it or to evaluate trust in its session delegate.

				advertisesService publishes the server over Bonjour as NFKInferenceServerServiceType,
				carrying the base path, whether TLS is on, and whether a key is required;
				NFKRemoteProvider's discoverInferKitServersWithTimeout: finds it. On iOS, listening on
				or browsing the local network needs NSLocalNetworkUsageDescription in the Info.plist.

				Runs follow the backends' contract: each backend serves at most
				maximumConcurrentRunsPerModel runs at once and later requests queue in arrival order.
				A backend that implements submitInferenceJobForRequest: streams its partial results
				to a streamed request; a synchronous one answers in one piece. A client that
				disconnects cancels its run, or removes it from the queue. Files the server writes
				for inline media are removed when the run ends.

				apiKey, requiresAPIKey, requiresAPIKeyOnLoopback, and the hosted models take effect
				on the next request. The other settings take effect on the next start.
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

/*! Whether the server is listening. */
@property (nonatomic, readonly, getter=isRunning) BOOL running;

/*! The port the server is listening on, which is the chosen one when port is 0; 0 while stopped. */
@property (nonatomic, readonly) uint16_t listeningPort;

/*! The base URL a client on this machine uses (http://localhost:<port>/v1, or https), or nil while
	stopped. A client elsewhere uses this machine's name or address, or discovery. */
@property (nonatomic, readonly, copy, nullable) NSURL *localBaseURL;

/*! Hosts a backend under a model name, replacing any backend already under it. */
- (void)addBackend:(id<NFKInferenceBackend>)backend forModelName:(NSString *)modelName
	NS_SWIFT_NAME(addBackend(_:forModelName:));

/*! Stops hosting the model. Its queued runs fail; its running ones finish. */
- (void)removeBackendForModelName:(NSString *)modelName;

/*! The backend hosted under a model name, or nil. */
- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName;

/*! The hosted model names, sorted. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *modelNames;

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
