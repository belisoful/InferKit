//
//  NFKRemoteUsageReporter.h
//  InferKit
//

#ifndef NFKRemoteUsageReporter_h
#define NFKRemoteUsageReporter_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKUsageBucket
	@abstract   The usage a service reported for one span of time. Introduced in InferKit 0.4.0.
*/
@interface NFKUsageBucket : NSObject
/*! The span's start. */
@property (nonatomic, copy, readonly) NSDate *startDate;
/*! The span's end, or nil where the service reports only a start. */
@property (nonatomic, copy, readonly, nullable) NSDate *endDate;
/*! One row per group the report was grouped by, as the service wrote it (token counts, requests,
	model, api_key_id, workspace_id, …). */
@property (nonatomic, copy, readonly) NSArray<NSDictionary<NSString *, id> *> *results;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@class      NFKCostEntry
	@abstract   One line of spend. Introduced in InferKit 0.4.0.
*/
@interface NFKCostEntry : NSObject
/*! The span's start. */
@property (nonatomic, copy, readonly) NSDate *startDate;
/*! The span's end, or nil. */
@property (nonatomic, copy, readonly, nullable) NSDate *endDate;
/*! The amount in the currency's main unit (dollars, not cents), whatever unit the service wrote. */
@property (nonatomic, copy, readonly) NSDecimalNumber *amount;
/*! The ISO currency code, upper case (USD). */
@property (nonatomic, copy, readonly) NSString *currency;
/*! What the spend was for: the line item, description, or model, or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *lineItem;
/*! The service's own record. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@class      NFKAccountBalance
	@abstract   What an account has left to spend. Introduced in InferKit 0.4.0.
*/
@interface NFKAccountBalance : NSObject
/*! The remaining balance in the currency's main unit. */
@property (nonatomic, copy, readonly) NSDecimalNumber *amount;
/*! The ISO currency code, upper case. */
@property (nonatomic, copy, readonly) NSString *currency;
/*! The service's own record. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@enum       NFKRemoteUsageAPIStyle
	@abstract   The wire shape of a usage and billing API.
	@constant   NFKRemoteUsageAPIStyleAnthropic Anthropic's Admin API usage_report and cost_report.
	@constant   NFKRemoteUsageAPIStyleOpenAI OpenAI's organization/usage/… and organization/costs.
	@constant   NFKRemoteUsageAPIStyleXAI xAI's management billing API.
	@constant   NFKRemoteUsageAPIStyleOpenRouter OpenRouter's credits, key, and activity.
	@constant   NFKRemoteUsageAPIStyleDeepSeek DeepSeek's user balance.
	@constant   NFKRemoteUsageAPIStyleMistral Mistral's admin usage.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteUsageAPIStyle) {
	NFKRemoteUsageAPIStyleAnthropic = 0,
	NFKRemoteUsageAPIStyleOpenAI,
	NFKRemoteUsageAPIStyleXAI,
	NFKRemoteUsageAPIStyleOpenRouter,
	NFKRemoteUsageAPIStyleDeepSeek,
	NFKRemoteUsageAPIStyleMistral,
};

/*!
	@class      NFKRemoteUsageReporter
	@abstract   Reads usage, spend, and balance from a hosted service's reporting API. Read only.
	@discussion Most of these calls take an administrative key (Anthropic sk-ant-admin…, an OpenAI admin
				key, an xAI management key, an OpenRouter management key, a Mistral admin key); DeepSeek's
				balance and OpenRouter's key usage take the ordinary one. An administrative key controls
				the whole organization, so it does not belong in a shipped binary: an app that offers
				usage reporting takes the key from its own user at run time and keeps it in the
				keychain. No member, workspace, or key management is offered.

				The services disagree on units, and the reporter normalizes them: money is dollars as
				an NSDecimalNumber (Anthropic writes cents as a string, OpenAI dollars as a number),
				time is NSDate (Anthropic writes RFC 3339, OpenAI Unix seconds), and pages are all read.
				Groq, Together, and Gemini report usage in their consoles only. Every call blocks; run
				it off the main thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteUsageReporter : NSObject

/*! The reporting root: …/v1 for Anthropic, OpenAI, and OpenRouter; https://management-api.x.ai/v1;
	https://api.deepseek.com; https://api.mistral.ai/v1/admin. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the reporter speaks. */
@property (nonatomic, assign) NFKRemoteUsageAPIStyle apiStyle;

/*! The key the reports need, usually an administrative one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! xAI's team id, which its billing paths name. */
@property (nonatomic, copy, nullable) NSString *teamIdentifier;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A reporter for the provider: anthropic, openai, xai, openrouter, deepseek, and mistral serve one;
	every other preset returns nil. */
+ (nullable instancetype)reporterForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*!
	@method     usageFromDate:toDate:bucketWidth:groupBy:report:error:
	@abstract   Usage over a span, bucketed.
	@param      startDate The span's start.
	@param      endDate The span's end, or nil to leave the span open at the service's default.
	@param      bucketWidth 1d, 1h, or 1m where the service takes a width (Anthropic, OpenAI), or nil
				for its default.
	@param      groupBy The dimensions to group by, in the service's spelling (model, api_key_id,
				workspace_id, project_id, description).
	@param      report Which report: Anthropic messages (default) or claude_code; OpenAI completions
				(default), embeddings, images, audio_speeches, audio_transcriptions, moderations,
				vector_stores, code_interpreter_sessions, web_search_calls, file_search_calls. Ignored
				elsewhere.
	@param      outError On failure, the reason.
	@discussion xAI reports usage in dollars per day; OpenRouter its last 30 days of activity, one
				bucket per day; Mistral one month, the month of startDate. DeepSeek reports no usage.
*/
- (nullable NSArray<NFKUsageBucket *> *)usageFromDate:(NSDate *)startDate
											   toDate:(nullable NSDate *)endDate
										  bucketWidth:(nullable NSString *)bucketWidth
											  groupBy:(nullable NSArray<NSString *> *)groupBy
											   report:(nullable NSString *)report
												error:(NSError * _Nullable *)outError;

/*! Spend over a span, one entry per day and group. Anthropic, OpenAI, xAI, and OpenRouter report it. */
- (nullable NSArray<NFKCostEntry *> *)costsFromDate:(NSDate *)startDate
											 toDate:(nullable NSDate *)endDate
											groupBy:(nullable NSArray<NSString *> *)groupBy
											  error:(NSError * _Nullable *)outError;

/*! What the account has left: OpenRouter's credits, DeepSeek's balance per currency, xAI's prepaid
	balance. */
- (nullable NSArray<NFKAccountBalance *> *)balancesWithError:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteUsageReporter_h */
