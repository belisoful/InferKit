//
//  NFKServedLive.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import "NFKHTTPServer.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKInferenceServer;

/*!
	@class      NFKServedLive
	@abstract   The live route, GET /v1/inferkit/live: a WebSocket that runs a hosted model on a live source.
	@discussion One connection is one live stream (NFKLiveStream.h) on one hosted model. The client's text
				messages are JSON objects:
				- {"type": "start", "model", "parameters", "valueObjects"} opens the stream and is answered
				  {"type": "started", "model", "base", "preset", "live"}, the model's live profile or null.
				  A second start closes the stream and opens another.
				- {"type": "unit", "seq", "inputs", "parameters"} pushes a unit whose inputs are the native
				  route's tagged JSON.
				- {"type": "reset"} forgets the stream's state after the units before it, answered
				  {"type": "reset"}.
				A binary message is a unit with raw buffers: a 4-byte little-endian header length, the header
				(a unit message with "buffers"), then the buffers' bytes. Each buffer names its input key,
				its offset and length in the bytes after the header, and its format:
				- "bgra" → a 32BGRA pixel buffer of width, height, and bytesPerRow;
				- "nv12" or "nv12-full" → a biplanar 4:2:0 pixel buffer, video or full range, the luma plane
				  of bytesPerRow then the chroma plane of chromaBytesPerRow (bytesPerRow when absent);
				- "pcm-f32" or "pcm-s16" → an interleaved PCM buffer of sampleRate and channels.
				Each unit is answered in order with {"type": "result", "seq", "outputs", "milliseconds",
				"dropped"}: a text message, or a binary message of the same layout when the result holds an
				image or PCM output, which travels as a buffer ("bgra", "gray8", "gray-f32", or "pcm-f32")
				instead of in outputs. A unit that fails is answered {"type": "error", "seq", "error"} and
				the stream goes on. A unit with a pixel buffer replaces the units queued behind the one in
				flight, so the newest frame runs next; other units queue in order, at most 256. dropped
				lists the sequence numbers left out since the last answer. Each unit runs in the hosted
				model's slot, as a request does.
*/
@interface NFKServedLive : NSObject

+ (void)serveRequest:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response server:(NFKInferenceServer *)server;

@end

NS_ASSUME_NONNULL_END
