//
//  NFKBackendTool.h
//  InferKit
//

#ifndef NFKBackendTool_h
#define NFKBackendTool_h

#import <Foundation/Foundation.h>
#import "NFKInferenceTool.h"
#import "NFKInferenceBackend.h"
#import "NFKModality.h"

NS_ASSUME_NONNULL_BEGIN

/*! Builds the request a backend tool runs from the model's arguments, or returns nil with an error
	the model reads. */
typedef NFKInferenceRequest * _Nullable (^ NS_SWIFT_SENDABLE NFKBackendToolRequestBuilder)(NSDictionary<NSString *, id> *arguments,
																		NSError * _Nullable * _Nullable error);

/*! Turns the backend's result into what the model reads and is shown. */
typedef NFKToolResult * _Nonnull (^ NS_SWIFT_SENDABLE NFKBackendToolResultDescriber)(NFKInferenceResult *result);

/*!
	@class      NFKBackendTool
	@abstract   A tool that runs an inference backend: a second language model, a speech or music
				generator, an image or video generator.
	@discussion One model reaches another through it. A remote Claude or GPT model calls a local Qwen
				model, a local Qwen model calls a speech generator, and an MCP client calls either.

				toolWithBackend:name:outputModality:purpose: fills in what a model needs to use the tool
				well, chosen by what the backend produces:

				- NFKModalityText: a prompt (required) and an optional system instruction; the reply's
				  text comes back.
				- NFKModalityAudio: a prompt (the words to speak, or the sound or music to make) and an
				  optional length in seconds; the model reads the audio's length and format, and the
				  audio reaches the application.
				- NFKModalityImage: a prompt, an optional negative prompt, size, and seed; the model reads
				  the image's size and is shown the image, which also reaches the application.
				- NFKModalityVideo: a prompt, an optional length, and a seed; the model reads the clip's
				  length and frame size, and the clip reaches the application.

				The default description states what the tool does, when to use it, and what comes back,
				which is what a model reads to choose a tool. purpose completes "Use it when …", where a
				caller names what the model is good at; a model choosing between several tools decides
				from it. modelName names the model in the description; it defaults to the backend's
				modelName when the backend has one, and to its backendIdentifier otherwise.

				Each argument whose name is an NFKInput* key (prompt, negativePrompt, lyrics) becomes
				that input; system becomes a system message; every other argument becomes the
				parameter of that name, so a schema that names NFKParameter* keys (width, seed,
				durationSeconds) needs no request builder. fixedParameters joins every request, for the
				settings the caller fixes rather than the model (a voice, a token limit). A caller with
				another mapping sets requestBuilder, and one that describes results its own way sets
				resultDescriber. Introduced in InferKit 0.4.0.
*/
@interface NFKBackendTool : NFKInferenceTool

/*! The backend the tool runs. */
@property (nonatomic, readonly, strong) id<NFKInferenceBackend> backend;

/*! The modality each request asks for. */
@property (nonatomic, readonly) NFKModality outputModality;

/*! Parameters every request carries, under the model's own arguments. */
@property (nonatomic, copy) NSDictionary<NSString *, id> *fixedParameters;

/*! Builds each request in place of the default mapping, on the thread that invokes the tool. */
@property (nonatomic, copy, nullable) NFKBackendToolRequestBuilder requestBuilder;

/*! Describes each result in place of the default description, on the thread that invokes the tool. */
@property (nonatomic, copy, nullable) NFKBackendToolResultDescriber resultDescriber;

/*!
	@method     toolWithBackend:name:outputModality:purpose:
	@abstract   A tool with the default description and schema for what the backend produces.
	@param      backend The backend each call runs.
	@param      name The name the model calls the tool by.
	@param      outputModality What the backend produces, which picks the default description and schema.
	@param      purpose Completes "Use it when …", or nil for the modality's general case.
*/
+ (instancetype)toolWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
				 outputModality:(NFKModality)outputModality
						purpose:(nullable NSString *)purpose;

/*! A tool with the default description and schema, naming the model as modelName. */
+ (instancetype)toolWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
				 outputModality:(NFKModality)outputModality
					  modelName:(nullable NSString *)modelName
						purpose:(nullable NSString *)purpose;

/*! A tool with the caller's own description and schema. */
- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
						   name:(NSString *)name
					description:(NSString *)toolDescription
					 parameters:(NSDictionary<NSString *, id> *)parameters
				 outputModality:(NFKModality)outputModality NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters NS_UNAVAILABLE;

/*! The default description for a backend producing outputModality. */
+ (NSString *)defaultDescriptionForOutputModality:(NFKModality)outputModality
										modelName:(NSString *)modelName
										  purpose:(nullable NSString *)purpose;

/*! The default argument schema for a backend producing outputModality. */
+ (NSDictionary<NSString *, id> *)defaultParametersForOutputModality:(NFKModality)outputModality;

/*! The request the default mapping builds from arguments. */
- (NFKInferenceRequest *)defaultRequestForArguments:(NSDictionary<NSString *, id> *)arguments;

/*! What the default description reads and shows of a result. */
- (NFKToolResult *)defaultToolResultForResult:(NFKInferenceResult *)result;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKBackendTool_h */
