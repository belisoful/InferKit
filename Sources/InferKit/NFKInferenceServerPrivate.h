//
//  NFKInferenceServerPrivate.h
//  InferKit
//
//  What a served route family (NFKServedAnthropic, NFKServedResponses) reaches of the server: the
//  body, the model, the error reply, and the queue. Private to the core.
//

#import <InferKit/NFKInferenceServer.h>
#import "NFKHTTPServer.h"
#import "NFKServedRun.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKServedModel;

/*! The wire a route's replies, its errors included, are written in. */
typedef NS_ENUM(NSInteger, NFKServedWire) {
	NFKServedWireOpenAI,
	NFKServedWireAnthropic,
};

/*! The name a hosted model is served under. */
FOUNDATION_EXPORT NSString *NFKServedModelName(NFKServedModel *model);

@interface NFKInferenceServer ()

/*! The request's body as a JSON object, or nil after replying with the refusal in wire's shape. */
- (nullable NSDictionary *)JSONBodyOf:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response wire:(NFKServedWire)wire;

/*! The model a request names, or the only one hosted when it names none. Every route resolves its
	model field here. */
- (nullable NFKServedModel *)modelNamed:(nullable id)name error:(NSError * _Nullable *)outError;

/*! Replies with failure in wire's error shape, at the status statusForError: gives it. */
- (void)sendError:(nullable NSError *)failure response:(NFKHTTPResponse *)response wire:(NFKServedWire)wire;

/*! The HTTP status an error is served with. */
- (NSInteger)statusForError:(NSError *)error;

/*! Queues a run for a model, cancelling it when the client goes away. The reply's head carries the
	model's load as it stands when the head is written. */
- (void)enqueue:(NFKServedRun *)run on:(NFKServedModel *)model request:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response;

@end

NS_ASSUME_NONNULL_END
