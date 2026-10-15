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

/*! What a request's model field reaches: the hosted model, the name as the request sent it, and the
	preset and alias defaults that apply. */
@interface NFKServedTarget : NSObject
@property (nonatomic, strong) NFKServedModel *model;
/*! The name the request sent, or the hosted model's when it sent none; an OpenAI-shaped reply's model. */
@property (nonatomic, copy) NSString *name;
/*! X-InferKit-Base-Model: a preset's base, an alias's model, or the hosted model itself. */
@property (nonatomic, copy) NSString *baseModelName;
@property (nonatomic, copy, nullable) NSString *presetIdentifier;
/*! The alias's or preset's defaults over the hosted model's own. */
@property (nonatomic, copy) NSDictionary<NSString *, id> *requestDefaults;
@end

@interface NFKInferenceServer ()

/*! The request's body as a JSON object, or nil after replying with the refusal in wire's shape. */
- (nullable NSDictionary *)JSONBodyOf:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response wire:(NFKServedWire)wire;

/*! What a request's model field reaches, or the only hosted model when it names none. Every route
	resolves its model field here: a hosted model or an alias, else base@preset for a suggested
	request-defaults preset of a hosted catalog model. */
- (nullable NFKServedTarget *)modelNamed:(nullable id)name error:(NSError * _Nullable *)outError;

/*! Replies with failure in wire's error shape, at the status statusForError: gives it. */
- (void)sendError:(nullable NSError *)failure response:(NFKHTTPResponse *)response wire:(NFKServedWire)wire;

/*! The HTTP status an error is served with. */
- (NSInteger)statusForError:(NSError *)error;

/*! Queues a run for a target, with its request defaults filled, cancelling it when the client goes
	away. The reply's head carries the model's load as it stands when the head is written, and the
	target's base model and preset. */
- (void)enqueue:(NFKServedRun *)run on:(NFKServedTarget *)target request:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response;

@end

NS_ASSUME_NONNULL_END
