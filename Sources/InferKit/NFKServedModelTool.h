//
//  NFKServedModelTool.h
//  InferKit
//
//  The MCP tool of a hosted model that has a descriptor: its argument schema, its description, and the
//  request its arguments build, all read from the descriptor.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelDescriptor.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKModality.h>

NS_ASSUME_NONNULL_BEGIN

/*! The tool's argument schema: one property per input port and per request-stage option, with each
	option's kind, choices, bounds, default, and help. requestDefaults (a preset's or an alias's) replace
	an option's default and keep it out of "required". A file-or-folder option is left out, since a path
	names a file on this machine. A model that takes a prompt and messages also takes "system". */
NSDictionary<NSString *, id> *NFKServedModelToolParameters(NFKModelDescriptor *descriptor,
														   NSDictionary<NSString *, id> * _Nullable requestDefaults);

/*! The tool's description: the model's task, display name, and summary, purpose completing "Use it when
	…", how a file argument is written, and what the call returns. */
NSString *NFKServedModelToolDescription(NFKModelDescriptor *descriptor, NSString *modelName, NSString * _Nullable purpose);

/*! The request the arguments build. Each input port's argument is decoded by its value type: a file
	(image, mask, audio, video, document, or a list of them) from a data: URL or InferKit's tagged form,
	never from a path. Each option's argument resolves as the descriptor resolves options. "system"
	with a prompt becomes a system message. Any other argument becomes the parameter of its name. The
	files decoding writes are added to temporaryFiles, which the caller removes after the run; on a
	failure they are already removed. */
NFKInferenceRequest * _Nullable NFKServedModelToolRequest(NSDictionary<NSString *, id> *arguments,
														  NFKModelDescriptor *descriptor,
														  NFKModality outputModality,
														  NSMutableArray<NSURL *> *temporaryFiles,
														  NSError * _Nullable * _Nullable error);

NS_ASSUME_NONNULL_END
