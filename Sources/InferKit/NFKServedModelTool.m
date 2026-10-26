//
//  NFKServedModelTool.m
//  InferKit
//

#import "NFKServedModelTool.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKModelVocabulary.h>
#import <InferKit/NFKModelOptions.h>
#import <InferKit/NFKValueCoding.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import "NFKServedOpenAI.h"
#import "NFKInferenceWireCoding.h"
#import "NFKRemoteMediaSupport.h"

/*! The singular file type of a port's value type, such as "image" for "images", or nil for a value type
	that is not a file. */
static NSString * _Nullable NFKServedToolFileType(NSString * _Nullable valueType)
{
	NSDictionary<NSString *, NSString *> *files = @{
		NFKModelValueTypeImage: NFKModelValueTypeImage, NFKModelValueTypeImages: NFKModelValueTypeImage,
		NFKModelValueTypeMask: NFKModelValueTypeMask,
		NFKModelValueTypeAudio: NFKModelValueTypeAudio, NFKModelValueTypeAudios: NFKModelValueTypeAudio,
		NFKModelValueTypeVideo: NFKModelValueTypeVideo, NFKModelValueTypeVideos: NFKModelValueTypeVideo,
		NFKModelValueTypeDocument: NFKModelValueTypeDocument, NFKModelValueTypeDocuments: NFKModelValueTypeDocument };
	return valueType != nil ? files[valueType] : nil;
}

static BOOL NFKServedToolIsFileList(NSString * _Nullable valueType)
{
	return [@[ NFKModelValueTypeImages, NFKModelValueTypeAudios, NFKModelValueTypeVideos, NFKModelValueTypeDocuments ]
			containsObject:valueType ?: @""];
}

#pragma mark Schema

static NSString * _Nullable NFKServedToolText(NSString * _Nullable help, NSString * _Nullable title)
{
	return help.length > 0 ? help : (title.length > 0 ? title : nil);
}

static NSDictionary<NSString *, id> *NFKServedToolFileSchema(NSString *fileType)
{
	NSString *mediaType = [fileType isEqualToString:NFKModelValueTypeDocument] ? @"application/pdf"
		: [fileType isEqualToString:NFKModelValueTypeAudio] ? @"audio/wav"
		: [fileType isEqualToString:NFKModelValueTypeVideo] ? @"video/mp4" : @"image/png";
	return @{ @"type": @[ @"string", @"object" ],
			  @"description": [NSString stringWithFormat:@"A data: URL such as data:%@;base64,…, or InferKit's tagged form {\"$nfk\": …}.",
							   mediaType] };
}

static NSDictionary<NSString *, id> *NFKServedToolPortSchema(NFKModelPort *port)
{
	NSString *valueType = port.valueType;
	NSString *fileType = NFKServedToolFileType(valueType);
	NSMutableDictionary<NSString *, id> *schema = [NSMutableDictionary dictionary];
	if (fileType != nil && NFKServedToolIsFileList(valueType)) {
		schema[@"type"] = @"array";
		schema[@"items"] = NFKServedToolFileSchema(fileType);
	} else if (fileType != nil) {
		[schema addEntriesFromDictionary:NFKServedToolFileSchema(fileType)];
	} else if ([valueType isEqualToString:NFKModelValueTypeText]) {
		schema[@"type"] = @"string";
	} else if ([valueType isEqualToString:NFKModelValueTypeTextList]) {
		schema[@"type"] = @"array";
		schema[@"items"] = @{ @"type": @"string" };
	} else if ([valueType isEqualToString:NFKModelValueTypeNumber] || [valueType isEqualToString:NFKModelValueTypeInteger]
			   || [valueType isEqualToString:NFKModelValueTypeBoolean]) {
		schema[@"type"] = valueType;
	} else if ([valueType isEqualToString:NFKModelValueTypeMessages]) {
		schema[@"type"] = @"array";
		schema[@"items"] = @{ @"type": @"object", @"properties": @{ @"role": @{ @"type": @"string" }, @"content": @{} } };
	} else if ([valueType isEqualToString:NFKModelValueTypeEmbedding]) {
		schema[@"type"] = @"array";
		schema[@"items"] = @{ @"type": @"number" };
	}
	NSString *text = NFKServedToolText(port.help, port.title);
	if (text != nil && schema[@"description"] != nil) {
		schema[@"description"] = [NSString stringWithFormat:@"%@ %@", text, schema[@"description"]];
	} else if (text != nil) {
		schema[@"description"] = text;
	}
	return schema;
}

static NSDictionary<NSString *, id> *NFKServedToolOptionSchema(NFKModelOption *option, id _Nullable presetDefault)
{
	NSMutableDictionary<NSString *, id> *schema = [NSMutableDictionary dictionary];
	NSString *kind = option.kind;
	if ([kind isEqualToString:NFKModelOptionKindChoice] && option.choices.count > 0) {
		NSArray *values = [option.choices valueForKey:@"value"];
		schema[@"enum"] = values;
		BOOL strings = YES;
		for (id value in values) {
			strings = strings && [value isKindOfClass:NSString.class];
		}
		if (strings) {
			schema[@"type"] = @"string";
		}
	} else if ([kind isEqualToString:NFKModelOptionKindBoolean] || [kind isEqualToString:NFKModelOptionKindInteger]
			   || [kind isEqualToString:NFKModelOptionKindNumber]) {
		schema[@"type"] = kind;
	} else if ([kind isEqualToString:NFKModelOptionKindText] || [kind isEqualToString:NFKModelOptionKindModelReference]) {
		schema[@"type"] = @"string";
	} else if ([kind isEqualToString:NFKModelOptionKindTextList]) {
		schema[@"type"] = @"array";
		schema[@"items"] = @{ @"type": @"string" };
	}
	schema[@"minimum"] = option.minimum;
	schema[@"maximum"] = option.maximum;
	schema[@"default"] = presetDefault ?: option.defaultValue;
	schema[@"title"] = option.title;
	schema[@"description"] = option.help;
	return schema;
}

NSDictionary<NSString *, id> *NFKServedModelToolParameters(NFKModelDescriptor *descriptor,
														   NSDictionary<NSString *, id> * _Nullable requestDefaults)
{
	NSMutableDictionary<NSString *, id> *properties = [NSMutableDictionary dictionary];
	NSMutableArray<NSString *> *required = [NSMutableArray array];
	for (NFKModelPort *port in descriptor.inputs) {
		properties[port.key] = NFKServedToolPortSchema(port);
		if (port.isRequired) {
			[required addObject:port.key];
		}
	}
	if (properties[NFKInputPrompt] != nil && properties[NFKInputMessages] != nil) {
		properties[@"system"] = @{ @"type": @"string", @"description": @"Optional instructions that set the model's role, tone, or rules." };
	}
	for (NFKModelOption *option in descriptor.options) {
		if (![option.stage isEqualToString:NFKModelOptionStageRequest] || properties[option.key] != nil
			|| [option.kind isEqualToString:NFKModelOptionKindFileOrFolder]) {
			continue;
		}
		id presetDefault = requestDefaults[option.key];
		properties[option.key] = NFKServedToolOptionSchema(option, presetDefault);
		if (option.isRequired && presetDefault == nil && option.defaultValue == nil) {
			[required addObject:option.key];
		}
	}
	return @{ @"type": @"object", @"properties": properties, @"required": required };
}

#pragma mark Description

static NSString *NFKServedToolList(NSArray<NSString *> *items)
{
	if (items.count <= 1) {
		return items.firstObject ?: @"";
	}
	NSArray<NSString *> *head = [items subarrayWithRange:NSMakeRange(0, items.count - 1)];
	return [NSString stringWithFormat:@"%@ and %@", [head componentsJoinedByString:@", "], items.lastObject];
}

NSString *NFKServedModelToolDescription(NFKModelDescriptor *descriptor, NSString *modelName, NSString * _Nullable purpose)
{
	NSMutableArray<NSString *> *tasks = [NSMutableArray array];
	for (NSString *task in descriptor.tasks) {
		[tasks addObject:[task stringByReplacingOccurrencesOfString:@"-" withString:@" "]];
	}
	NSString *task = tasks.count > 0 ? NFKServedToolList(tasks) : @"inference";
	NSMutableString *text = [NSMutableString string];
	if ([descriptor.displayName isEqualToString:modelName] || descriptor.displayName.length == 0) {
		[text appendFormat:@"Runs \"%@\", a %@ model.", modelName, task];
	} else {
		[text appendFormat:@"Runs \"%@\" (%@), a %@ model.", modelName, descriptor.displayName, task];
	}
	if (descriptor.summary.length > 0) {
		[text appendFormat:@" %@", descriptor.summary];
	}
	[text appendFormat:@" Use it when %@.", purpose.length > 0 ? purpose : [NSString stringWithFormat:@"a task calls for %@", task]];
	BOOL files = NO;
	for (NFKModelPort *port in descriptor.inputs) {
		files = files || NFKServedToolFileType(port.valueType) != nil;
	}
	if (files) {
		[text appendString:@" A file argument is a data: URL or InferKit's tagged form; a path on the server's machine is refused."];
	}
	NSMutableArray<NSString *> *outputs = [NSMutableArray array];
	for (NFKModelPort *port in descriptor.outputs) {
		[outputs addObject:port.title.length > 0 ? port.title : port.key];
	}
	if (outputs.count > 0) {
		[text appendFormat:@" Returns %@.", NFKServedToolList(outputs)];
	}
	return text;
}

#pragma mark Request

static id _Nullable NFKServedToolFail(NSError **error, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return nil;
}

static NSString *NFKServedToolExtension(NSString * _Nullable mediaType, NSString *fileType)
{
	NSString *subtype = [mediaType componentsSeparatedByString:@"/"].lastObject.lowercaseString;
	NSDictionary<NSString *, NSString *> *aliases = @{ @"mpeg": @"mp3", @"x-wav": @"wav", @"wave": @"wav", @"quicktime": @"mov",
													   @"x-m4a": @"m4a", @"plain": @"txt" };
	NSDictionary<NSString *, NSString *> *fallbacks = @{ NFKModelValueTypeAudio: @"wav", NFKModelValueTypeVideo: @"mp4",
														 NFKModelValueTypeDocument: @"pdf" };
	NSString *extension = aliases[subtype ?: @""] ?: subtype;
	return NFKServedFileExtension(extension.length > 0 ? extension : fallbacks[fileType]);
}

/*! The file a decoded value reads, or nil for a value that is not a file. */
static NSURL * _Nullable NFKServedToolFileURL(id value)
{
	if ([value isKindOfClass:NFKAudioAsset.class]) {
		return [(NFKAudioAsset *)value fileURL];
	}
	if ([value isKindOfClass:NFKVideoAsset.class]) {
		return [(NFKVideoAsset *)value fileURL];
	}
	return [value isKindOfClass:NSURL.class] && [(NSURL *)value isFileURL] ? value : nil;
}

/*! One file argument: a tagged value that carries its bytes, or a data: URL. A path is refused, since
	the caller may be another machine, so a tagged file value reads only the file its decoding wrote. */
static id _Nullable NFKServedToolFile(id value, NSString *fileType, NSString *key, NSMutableArray<NSURL *> *files, NSError **error)
{
	if ([value isKindOfClass:NSDictionary.class] && [value[@"$nfk"] isKindOfClass:NSString.class]) {
		NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
		id decoded = [decoder valueFromJSONValue:value key:key error:error];
		[files addObjectsFromArray:decoder.temporaryFileURLs];
		NSURL *fileURL = decoded != nil ? NFKServedToolFileURL(decoded) : nil;
		BOOL image = [fileType isEqualToString:NFKModelValueTypeImage] || [fileType isEqualToString:NFKModelValueTypeMask];
		if (decoded != nil && (image ? fileURL != nil : fileURL == nil || ![decoder.temporaryFileURLs containsObject:fileURL])) {
			return NFKServedToolFail(error, [NSString stringWithFormat:@"the argument \"%@\" carries no bytes of its own; "
											 @"a tagged file value carries them in base64", key]);
		}
		return decoded;
	}
	NSString *mediaType = nil;
	NSData *bytes = NFKServedDataURLBytes(value, &mediaType);
	if (bytes == nil) {
		return NFKServedToolFail(error, [NSString stringWithFormat:@"the argument \"%@\" takes a data: URL or InferKit's tagged form, not %@",
										 key, [value isKindOfClass:NSString.class] ? @"a path or another string" : @"that value"]);
	}
	NFKValueCoding *coding = [[NFKValueCoding alloc] init];
	if ([fileType isEqualToString:NFKModelValueTypeImage] || [fileType isEqualToString:NFKModelValueTypeMask]) {
		return [coding valueOfType:fileType fromJSONValue:value error:error];
	}
	NSURL *url = NFKRemoteWriteMediaFile(bytes, @"tool", NFKServedToolExtension(mediaType, fileType), nil, error);
	if (url == nil) {
		return nil;
	}
	[files addObject:url];
	return [coding valueOfType:fileType fromFileURL:url error:error];
}

static id _Nullable NFKServedToolInput(id value, NFKModelPort *port, NSMutableArray<NSURL *> *files, NSError **error)
{
	NSString *fileType = NFKServedToolFileType(port.valueType);
	if (fileType == nil) {
		return port.valueType != nil ? [[[NFKValueCoding alloc] init] valueOfType:port.valueType fromJSONValue:value error:error] : value;
	}
	if (!NFKServedToolIsFileList(port.valueType)) {
		return NFKServedToolFile(value, fileType, port.key, files, error);
	}
	NSArray *elements = [value isKindOfClass:NSArray.class] ? value : @[ value ];
	NSMutableArray *decoded = [NSMutableArray array];
	for (id element in elements) {
		id file = NFKServedToolFile(element, fileType, port.key, files, error);
		if (file == nil) {
			return nil;
		}
		[decoded addObject:file];
	}
	return decoded;
}

NFKInferenceRequest * _Nullable NFKServedModelToolRequest(NSDictionary<NSString *, id> *arguments,
														  NFKModelDescriptor *descriptor,
														  NFKModality outputModality,
														  NSMutableArray<NSURL *> *temporaryFiles,
														  NSError * _Nullable * _Nullable error)
{
	NSMutableDictionary<NSString *, NFKModelPort *> *ports = [NSMutableDictionary dictionary];
	for (NFKModelPort *port in descriptor.inputs) {
		ports[port.key] = port;
	}
	NSMutableArray<NSURL *> *written = [NSMutableArray array];
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSString *, id> *options = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSString *system = nil;
	for (NSString *name in arguments) {
		id value = arguments[name];
		if (value == NSNull.null) {
			continue;
		}
		NFKModelOption *option = [descriptor optionWithKey:name];
		if (ports[name] != nil) {
			id input = NFKServedToolInput(value, ports[name], written, error);
			if (input == nil) {
				for (NSURL *file in written) {
					[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
				}
				return nil;
			}
			inputs[name] = input;
		} else if ([name isEqualToString:@"system"] && [value isKindOfClass:NSString.class]) {
			system = value;
		} else if ([option.stage isEqualToString:NFKModelOptionStageRequest]
				   && ![option.kind isEqualToString:NFKModelOptionKindFileOrFolder]) {
			options[name] = value;
		} else {
			parameters[name] = value;
		}
	}
	NSDictionary<NSString *, id> *resolved = [descriptor resolvedOptions:options stage:NFKModelOptionStageRequest error:error];
	if (resolved == nil) {
		for (NSURL *file in written) {
			[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
		}
		return nil;
	}
	// Only the options the call names: a default left out stays the model's or its preset's.
	for (NSString *key in options) {
		parameters[key] = resolved[key] ?: options[key];
	}
	NSString *prompt = [inputs[NFKInputPrompt] isKindOfClass:NSString.class] ? inputs[NFKInputPrompt] : nil;
	if (system.length > 0 && prompt != nil && inputs[NFKInputMessages] == nil) {
		inputs[NFKInputMessages] = @[ @{ @"role": @"system", @"content": system }, @{ @"role": @"user", @"content": prompt } ];
	}
	[temporaryFiles addObjectsFromArray:written];
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:outputModality];
}
