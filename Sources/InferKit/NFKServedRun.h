//
//  NFKServedRun.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error a run ends with when its client went away. */
FOUNDATION_EXPORT NSError *NFKServedCancellation(void);

/*! Whether error is the one NFKServedCancellation returns. */
FOUNDATION_EXPORT BOOL NFKServedIsCancellation(NSError * _Nullable error);

/*! One request waiting for, or holding, a slot on its model. Finishes exactly once. */
@interface NFKServedRun : NSObject
@property (nonatomic, strong, nullable) NFKInferenceRequest *request;
/*! Runs in place of the backend's run when set; the run is then synchronous. */
@property (nonatomic, copy, nullable) NFKInferenceResult * _Nullable (^work)(id<NFKInferenceBackend> backend, NSError * _Nullable * _Nullable error);
@property (nonatomic, copy, nullable) void (^partialHandler)(NFKInferenceResult * _Nullable partial, double progress);
/*! Runs once, when the run takes its slot and its work begins. */
@property (nonatomic, copy, nullable) void (^startHandler)(void);
@property (nonatomic, copy, nullable) void (^completionHandler)(NFKInferenceResult * _Nullable result, NSError * _Nullable error);
@property (nonatomic, strong, nullable) NFKInferenceJob *job;
/*! Files written for the request's inline media, removed once the reply has been built, since a
	result may name one of them. */
@property (nonatomic, copy, nullable) NSArray<NSURL *> *temporaryFiles;
@property (atomic, assign) BOOL cancelled;
@property (atomic, assign) BOOL finished;
@property (nonatomic, assign) NSTimeInterval enqueuedAt;
@property (nonatomic, assign) NSTimeInterval startedAt;
/*! The job's last reported progress, 0 until it reports one. */
@property (atomic, assign) double reportedProgress;
/*! What the run's events report: an identifier shared by its start and end, and the request's route,
	client, and key. */
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, copy) NSString *route;
@property (nonatomic, copy) NSString *clientAddress;
@property (nonatomic, copy, nullable) NSString *keyName;
/*! Whether the run's work began, which its start event reported. */
@property (atomic, assign) BOOL began;
/*! Whether the run stays out of the model's averages, token totals, and outcome counts, as a token
	count does. */
@property (nonatomic, assign) BOOL unmeasured;

/*! An unmeasured run that counts request's input tokens with the backend's
	tokenCountForRequest:error:. Its result carries the count under NFKOutputUsage's
	NFKUsageInputTokens; a backend that does not count fails it with kNFKError_InferenceUnsupported. */
+ (instancetype)tokenCountRunForRequest:(NFKInferenceRequest *)request modelName:(NSString *)modelName;

- (void)cancel;
- (void)finishWithResult:(nullable NFKInferenceResult *)result error:(nullable NSError *)error;
- (void)reportPartial:(nullable NFKInferenceResult *)partial progress:(double)progress;
@end

NS_ASSUME_NONNULL_END
