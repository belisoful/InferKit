//
//  NFKServerObservation.h
//  InferKit
//

#ifndef NFKServerObservation_h
#define NFKServerObservation_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKServerRoute.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceServer;

/*!
	@enum       NFKServerRunState
	@abstract   Where a served run stands when an event reports it.
	@constant   NFKServerRunStateStarted The run took a slot on its model and its work began.
	@constant   NFKServerRunStateCompleted The run returned a result.
	@constant   NFKServerRunStateFailed The run, or the build it waited for, failed, or its model stopped
				being served before it started.
	@constant   NFKServerRunStateCancelled The client went away, before or during the run.
	@constant   NFKServerRunStateRefused The model's queue was full (maximumQueuedRunsPerModel).
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerRunState) {
	NFKServerRunStateStarted = 0,
	NFKServerRunStateCompleted,
	NFKServerRunStateFailed,
	NFKServerRunStateCancelled,
	NFKServerRunStateRefused,
};

/*!
	@class      NFKServerRunEvent
	@abstract   One run's start or end, as NFKInferenceServerDelegate receives it.
	@discussion A run that starts is reported twice, started then ended, with the same identifier. A run
				that never starts (refused, cancelled while queued, failed with its build) is reported
				once, ended, with a duration of 0. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKServerRunEvent : NSObject

/*! The run's identifier, shared by its start and end events. */
@property (nonatomic, readonly, copy) NSString *identifier;

/*! The model the run was for. */
@property (nonatomic, readonly, copy) NSString *modelName;

/*! The route the request arrived on, such as "/v1/chat/completions". */
@property (nonatomic, readonly, copy) NSString *route;

/*! The client's IP address as text. */
@property (nonatomic, readonly, copy) NSString *clientAddress;

/*! The name of the key the request presented, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *keyName;

@property (nonatomic, readonly) NFKServerRunState state;

/*! Seconds from the request's arrival to its start, or to its end when it never started. */
@property (nonatomic, readonly) NSTimeInterval waitSeconds;

/*! Seconds the run held its slot; 0 in a start event and for a run that never started. */
@property (nonatomic, readonly) NSTimeInterval durationSeconds;

/*! Why the run failed, was cancelled, or was refused, or nil. */
@property (nonatomic, readonly, copy, nullable) NSError *error;

/*! When the event happened. */
@property (nonatomic, readonly, copy) NSDate *date;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@enum       NFKServerModelEventKind
	@abstract   What happened to a hosted model's backend.
	@constant   NFKServerModelEventKindLoaded A build finished and the backend is resident.
	@constant   NFKServerModelEventKindLoadFailed A build failed; the model stays not loaded.
	@constant   NFKServerModelEventKindUnloaded The backend was released.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerModelEventKind) {
	NFKServerModelEventKindLoaded = 0,
	NFKServerModelEventKindLoadFailed,
	NFKServerModelEventKindUnloaded,
};

/*!
	@enum       NFKServerModelEventReason
	@abstract   Why a model was built or released.
	@constant   NFKServerModelEventReasonRequest A request needed the model and started its build.
	@constant   NFKServerModelEventReasonExplicit The host called loadModelNamed: or unloadModelNamed:.
	@constant   NFKServerModelEventReasonIdle The model sat unused past its idle interval.
	@constant   NFKServerModelEventReasonPressure Memory pressure shortened its timeout and it was the least
				recently used model past it.
	@constant   NFKServerModelEventReasonRemoved The host removed or replaced the model.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerModelEventReason) {
	NFKServerModelEventReasonRequest = 0,
	NFKServerModelEventReasonExplicit,
	NFKServerModelEventReasonIdle,
	NFKServerModelEventReasonPressure,
	NFKServerModelEventReasonRemoved,
};

/*! One model's load or unload, as NFKInferenceServerDelegate receives it. Introduced in InferKit 0.4.0. */
NS_SWIFT_SENDABLE
@interface NFKServerModelEvent : NSObject

@property (nonatomic, readonly, copy) NSString *modelName;
@property (nonatomic, readonly) NFKServerModelEventKind kind;
@property (nonatomic, readonly) NFKServerModelEventReason reason;

/*! How long the build took, for a load or a failed load; 0 for an unload. */
@property (nonatomic, readonly) NSTimeInterval loadSeconds;

/*! The build's error for a failed load, or nil. */
@property (nonatomic, readonly, copy, nullable) NSError *error;

/*! When the event happened. */
@property (nonatomic, readonly, copy) NSDate *date;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKServerClientEvent
	@abstract   A client's new connection, reported when the connection's first request passes the key
				check.
	@discussion A client that keeps its connection alive is reported once per connection; a connection
				whose first request the key check refuses reports nothing. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKServerClientEvent : NSObject

/*! The client's IP address as text. */
@property (nonatomic, readonly, copy) NSString *clientAddress;

@property (nonatomic, readonly) BOOL fromLoopback;

/*! The name of the key the first request presented, or nil when it presented none. */
@property (nonatomic, readonly, copy, nullable) NSString *keyName;

/*! The scope of the first request's route. */
@property (nonatomic, readonly) NFKServerKeyScope scope;

/*! The first request's path. */
@property (nonatomic, readonly, copy) NSString *path;

/*! When the request arrived. */
@property (nonatomic, readonly, copy) NSDate *date;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@enum       NFKServerMCPSessionEndReason
	@abstract   Why an MCP session ended.
	@constant   NFKServerMCPSessionEndReasonNone The session is active.
	@constant   NFKServerMCPSessionEndReasonDeleted The client ended it with DELETE.
	@constant   NFKServerMCPSessionEndReasonPingTimeout The client left MCPMissedPingLimit pings on its
				stream unanswered.
	@constant   NFKServerMCPSessionEndReasonIdle The client had no stream open and sent nothing for
				MCPSessionIdleTimeout.
	@constant   NFKServerMCPSessionEndReasonRevoked The key the session started with was removed.
	@constant   NFKServerMCPSessionEndReasonEndedByHost The host called endMCPSessionWithIdentifier:.
	@constant   NFKServerMCPSessionEndReasonServerStopped The server stopped.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKServerMCPSessionEndReason) {
	NFKServerMCPSessionEndReasonNone = 0,
	NFKServerMCPSessionEndReasonDeleted,
	NFKServerMCPSessionEndReasonPingTimeout,
	NFKServerMCPSessionEndReasonIdle,
	NFKServerMCPSessionEndReasonRevoked,
	NFKServerMCPSessionEndReasonEndedByHost,
	NFKServerMCPSessionEndReasonServerStopped,
};

/*!
	@class      NFKServerMCPSession
	@abstract   An MCP client's session on the server, as it stood when the object was made.
	@discussion The server issues a session at initialize and reads the client's name, version, and
				protocol version from it. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@interface NFKServerMCPSession : NSObject

/*! The Mcp-Session-Id the server issued. */
@property (nonatomic, readonly, copy) NSString *identifier;

/*! clientInfo.name from the client's initialize, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *clientName;

/*! clientInfo.version from the client's initialize, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *clientVersion;

/*! The protocol version the session agreed on. */
@property (nonatomic, readonly, copy) NSString *protocolVersion;

/*! The IP address initialize came from. */
@property (nonatomic, readonly, copy) NSString *clientAddress;

/*! The name of the key initialize presented, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *keyName;

@property (nonatomic, readonly, copy) NSDate *startDate;

/*! When the client last sent a request, a notification, or a ping's answer, or closed its stream. */
@property (nonatomic, readonly, copy) NSDate *lastActivityDate;

/*! Whether the client holds its server-to-client stream (GET /mcp) open. */
@property (nonatomic, readonly, getter=isStreamOpen) BOOL streamOpen;

/*! Why the session ended, or NFKServerMCPSessionEndReasonNone while it is active. */
@property (nonatomic, readonly) NFKServerMCPSessionEndReason endReason;

/*! When the session ended, or nil while it is active. */
@property (nonatomic, readonly, copy, nullable) NSDate *endDate;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@protocol   NFKInferenceServerDelegate
	@abstract   Observes a server in process: its runs, its models' loads, its clients, and its MCP
				sessions.
	@discussion Every method is optional and runs on the server's delegateQueue, in the order the
				events happened. The protocol is Sendable in Swift, so a conformance declared in
				main-actor code runs nonisolated on that queue.

				Swift imports the methods as inferenceServer(_:didStartRun:),
				inferenceServer(_:didFinishRun:), inferenceServer(_:didChangeModel:),
				inferenceServer(_:didConnectClient:), inferenceServer(_:didStartMCPSession:), and
				inferenceServer(_:didEndMCPSession:). The methods are optional, so a Swift method
				whose name differs from these compiles and is never called. Introduced in InferKit 0.4.0.
*/
NS_SWIFT_SENDABLE
@protocol NFKInferenceServerDelegate <NSObject>

@optional

/*! A run took a slot and began. */
- (void)inferenceServer:(NFKInferenceServer *)server didStartRun:(NFKServerRunEvent *)event;

/*! A run ended, or a request ended without starting. */
- (void)inferenceServer:(NFKInferenceServer *)server didFinishRun:(NFKServerRunEvent *)event;

/*! A model was loaded, failed to load, or was unloaded. */
- (void)inferenceServer:(NFKInferenceServer *)server didChangeModel:(NFKServerModelEvent *)event;

/*! A client's connection passed its first key check. */
- (void)inferenceServer:(NFKInferenceServer *)server didConnectClient:(NFKServerClientEvent *)event;

/*! An MCP client initialized a session. */
- (void)inferenceServer:(NFKInferenceServer *)server didStartMCPSession:(NFKServerMCPSession *)session;

/*! An MCP session ended; endReason says why. */
- (void)inferenceServer:(NFKInferenceServer *)server didEndMCPSession:(NFKServerMCPSession *)session;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKServerObservation_h */
