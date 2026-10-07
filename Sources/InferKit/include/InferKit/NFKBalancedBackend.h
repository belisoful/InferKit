//
//  NFKBalancedBackend.h
//  InferKit
//

#ifndef NFKBalancedBackend_h
#define NFKBalancedBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKServerStatus.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKBalancingPolicy
	@abstract   How an NFKBalancedBackend picks the server for a request.
	@constant   NFKBalancingPolicyShortestExpectedWait The server whose next run is expected to start
				soonest: the server's estimated wait, plus an average run, shared over its slots, for
				each request this backend sent it since that estimate. While a candidate has finished
				no run, so has no average, the backend picks by fewest outstanding instead.
	@constant   NFKBalancingPolicyFewestOutstanding The server with the fewest running and queued runs
				per slot, counting the requests this backend sent it since its last reading.
	@constant   NFKBalancingPolicyRoundRobin Each candidate in turn.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKBalancingPolicy) {
	NFKBalancingPolicyShortestExpectedWait = 0,
	NFKBalancingPolicyFewestOutstanding,
	NFKBalancingPolicyRoundRobin,
};

/*!
	@class      NFKBalancedBackend
	@abstract   A backend that sends each request to one of several InferKit servers hosting the model.
	@discussion Each server is an NFKInferenceServer reached through its native route, so the whole
				request and result cross as they do for NFKRemoteInferKitBackend. Hosted in an
				NFKInferenceServer of its own, the backend makes that server a load balancer in front
				of the others.

				Choosing a server:
				- A candidate hosts the model, reports it ready, and is reachable. A server whose
				  model is itself a balanced backend is never a candidate, so a balancer that
				  discovers itself does not route to itself.
				- A candidate at the serious or critical thermal state, or under critical memory
				  pressure, is chosen only when no other candidate remains.
				- policy picks among the candidates; ties go to each in turn.
				- The backend reads a server's status route when its last reading is older than
				  statusInterval, and learns its load from every reply's X-InferKit-* headers in
				  between. GPU utilization takes no part.

				Failing over: a request moves to the next candidate only when it never started on
				the server it was sent to: the server could not be connected to, its queue was full
				(NFKInferenceServerErrorBusy), its model was not ready, or it does not host the
				model. A lost connection or a timeout may have reached a run in progress, so either
				fails the request. A server that cannot be connected to sits out for a backoff that
				doubles from 2 seconds to 60, and a successful reading returns it.

				Keeping conversations together: a request in a conversation goes to the server that
				answered the conversation's previous request, where a backend that keeps its prompt
				between requests (such as MLX's language backend asked to reuse its prompt cache)
				continues from it instead of reading the history again.
				- A conversation is named by the request's NFKParameterConversationKey, or else by the
				  messages up to and including its first user message, which every later turn of a
				  chat repeats. A request with neither belongs to no conversation. A conversation named
				  from the messages is sent to the server under NFKParameterConversationKey, so the
				  server's backend can keep the conversation's prompt as it would for a caller's key.
				- The conversation's server is used while it remains a candidate and no candidate's
				  expected wait is more than conversationWaitAllowance shorter than its own. Otherwise
				  the request is chosen as any other, and the conversation moves to the server that
				  answers it, as it does after a failover.
				- A conversation idle for conversationIdleInterval is forgotten.

				startDiscoveryWithInterval: adds the servers Bonjour finds and drops a discovered
				server once two browses in a row miss it. A server added by URL stays until it is
				removed. A server reached both ways counts as two.
				Introduced in InferKit 0.4.0.
*/
@interface NFKBalancedBackend : NSObject <NFKInferenceBackend>

/*! A backend for the model served under modelName, or each server's only model when nil. */
+ (instancetype)backendWithModelName:(nullable NSString *)modelName;

/*! The served model to run; nil runs each server's only model. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The key sent to a server added without one of its own, and to discovered servers. */
@property (atomic, copy, nullable) NSString *apiKey;

/*! Defaults to NFKBalancingPolicyShortestExpectedWait. */
@property (atomic, assign) NFKBalancingPolicy policy;

/*! The oldest a server's status reading may be before a request reads it again. Defaults to 2
	seconds. */
@property (atomic, assign) NSTimeInterval statusInterval;

/*! How long a status reading waits for a server. Defaults to 2 seconds. */
@property (atomic, assign) NSTimeInterval statusTimeout;

/*! The request timeout passed to each server's client. Defaults to 600 seconds. */
@property (atomic, assign) NSTimeInterval timeout;

/*! Whether a conversation's requests go to the server that answered its previous request. Defaults
	to YES. */
@property (atomic, assign) BOOL conversationAffinity;

/*! How long a conversation keeps its server after its last request. Defaults to 600 seconds. */
@property (atomic, assign) NSTimeInterval conversationIdleInterval;

/*! How much longer a conversation's server may be expected to wait than the shortest expected wait
	before the conversation moves. Defaults to 10 seconds. */
@property (atomic, assign) NSTimeInterval conversationWaitAllowance;

/*! The servers' base URLs, in the order they were added. */
@property (nonatomic, readonly, copy) NSArray<NSURL *> *serverBaseURLs;

/*! The server that answered the most recent request, or nil before one has. */
@property (atomic, readonly, copy, nullable) NSURL *lastServerBaseURL;

/*! Adds a server by its base URL (for example http://studio.local:11480/v1) with its own key, or nil
	for apiKey. Adding a URL already present replaces its key. */
- (void)addServerWithBaseURL:(NSURL *)baseURL apiKey:(nullable NSString *)apiKey;

/*! Stops sending to a server; a run already sent to it finishes there. */
- (void)removeServerWithBaseURL:(NSURL *)baseURL;

/*! Browses Bonjour now and then every interval seconds, adding the InferKit servers it finds. */
- (void)startDiscoveryWithInterval:(NSTimeInterval)interval;

/*! Stops browsing. Discovered servers stay until removed. */
- (void)stopDiscovery;

/*! Reads every server's status; YES when at least one can take the model now. */
- (BOOL)prepareWithError:(NSError * _Nullable *)outError;

/*! The streamed form: the job reports the chosen server's progress and partial results, fails over as
	runInferenceForRequest:error: does, and cancelling it cancels the run on the server. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The status seam: reads one server's status route. The default reads it with statusTimeout. */
- (nullable NFKServerStatus *)fetchStatusFromBaseURL:(NSURL *)baseURL
											 apiKey:(nullable NSString *)apiKey
											  error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKBalancedBackend_h */
