//
//  NFKModelPreset.h
//  InferKit
//

#ifndef NFKModelPreset_h
#define NFKModelPreset_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@header     NFKModelPreset
	@abstract   Named option sets a model is offered under, and the names they take.
	@discussion A preset is reached as model@preset: the model's identifier, NFKModelPresetSeparator, and
				the preset's identifier. A tool name, such as an MCP tool's, cannot hold "@", so
				NFKModelToolName writes the separator as NFKModelToolNameSeparator. Introduced in
				InferKit 0.4.0.
*/

/*! The separator between a model's identifier and a preset's: "@". Introduced in InferKit 0.4.0. */
extern NSString * const NFKModelPresetSeparator;

/*! The separator NFKModelToolName writes in place of NFKModelPresetSeparator: "__". Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKModelToolNameSeparator;

/*! The name of a model under a preset: "whisper" and "fast" give "whisper@fast". Introduced in
	InferKit 0.4.0. */
NSString *NFKModelPresetModelName(NSString *modelIdentifier, NSString *presetIdentifier);

/*!
	@function   NFKModelNameSplitAtPreset
	@abstract   Splits a name at its last NFKModelPresetSeparator into a model and a preset.
	@discussion "a@b@fast" splits into "a@b" and "fast". The split happens only where both sides are
				non-empty and isPreset, when given, answers YES for them, so "claude-3@20240620" stays
				whole where no such preset is defined. Without a split, model is the whole name and
				preset is nil. Returns YES when the name splits. Introduced in InferKit 0.4.0.
*/
BOOL NFKModelNameSplitAtPreset(NSString *name,
							   BOOL (^ _Nullable isPreset)(NSString *model, NSString *preset),
							   NSString * _Nullable * _Nullable model,
							   NSString * _Nullable * _Nullable preset);

/*!
	@function   NFKModelToolName
	@abstract   A model name as a tool name: letters, digits, "_", and "-", at most 64 characters.
	@discussion NFKModelPresetSeparator becomes NFKModelToolNameSeparator, so "whisper@fast" is
				"whisper__fast". Any other character outside the set becomes "_", so
				"mlx-community/Qwen2.5" is "mlx-community_Qwen2_5". A longer result is cut to 64
				characters. Two names can map to one tool name; a caller that lists tools checks for
				that. Introduced in InferKit 0.4.0.
*/
NSString *NFKModelToolName(NSString *modelName);

/*!
	@class      NFKModelPreset
	@abstract   A named set of load options and request defaults for one model.
	@discussion A preset that only sets request defaults runs on a backend already loaded for its base
				model. A preset that sets load options needs its own load. suggested marks a preset a
				server offers without being told to. Introduced in InferKit 0.4.0.
*/
@interface NFKModelPreset : NSObject <NSCopying>

/*! The preset's identifier, unique within its model, such as "fast". */
@property (nonatomic, readonly, copy) NSString *identifier;
/*! The preset's name, for a user interface. The identifier when absent. */
@property (nonatomic, readonly, copy) NSString *title;
/*! A sentence that explains the preset. */
@property (nonatomic, readonly, copy, nullable) NSString *help;
/*! The identifier of the model the preset applies to. nil for a preset listed under its descriptor,
	which applies to that descriptor. */
@property (nonatomic, readonly, copy, nullable) NSString *baseModelIdentifier;
/*! The load options the preset sets. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *loadOptions;
/*! The request parameters the preset fills where a request leaves them out. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *requestDefaults;
/*! YES for a preset a server offers without being told to. */
@property (nonatomic, readonly, getter=isSuggested) BOOL suggested;
/*! YES when the preset sets load options, so it needs a load of its own. */
@property (nonatomic, readonly) BOOL changesLoadOptions;
/*! The fields of the preset's dictionary this version does not read, written back as they came. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! A preset with every field. A nil title is the identifier. */
- (instancetype)initWithIdentifier:(NSString *)identifier
							 title:(nullable NSString *)title
							  help:(nullable NSString *)help
			   baseModelIdentifier:(nullable NSString *)baseModelIdentifier
					   loadOptions:(nullable NSDictionary<NSString *, id> *)loadOptions
				   requestDefaults:(nullable NSDictionary<NSString *, id> *)requestDefaults
						 suggested:(BOOL)suggested
						extensions:(nullable NSDictionary<NSString *, id> *)extensions NS_DESIGNATED_INITIALIZER;

/*! The preset's model name, model@preset, under its base model or the given one. */
- (NSString *)modelNameForBaseModelIdentifier:(nullable NSString *)baseModelIdentifier;

/*! Reads a preset from its dictionary: id, title, help, base, loadOptions, requestDefaults, and
	suggested. Returns nil and sets error when id is missing or a field has the wrong type. */
+ (nullable instancetype)presetWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error;
/*! The preset as its dictionary, with the extensions beside the fields. */
- (NSDictionary<NSString *, id> *)dictionaryRepresentation;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKModelPreset_h */
