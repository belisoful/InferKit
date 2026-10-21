//
//  RemoteChatPlugin.m
//  InferKit plugin examples
//
//  An Objective-C plugin that serves a chat model from an OpenAI-compatible HTTP server. The backend is
//  InferKit's own NFKRemoteBackend, configured from the plugin's options: the server's base URL, the
//  model name, and an API key. An app shows the three options in its settings and sets them with
//  -[NFKPluginRegistry setOptionValues:forPluginIdentifier:error:].
//
//  The bundle binds to the InferKit already loaded in the process that loads it, so no class here
//  carries the NFK prefix.
//

#import <InferKit/InferKit.h>

static NSString * const RemoteChatModelIdentifier = @"chat";
static NSString * const RemoteChatBaseURLOption = @"baseURL";
static NSString * const RemoteChatModelOption = @"model";
static NSString * const RemoteChatAPIKeyOption = @"apiKey";

@interface RemoteChatPlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@property (nonatomic, strong, nullable) NFKPluginManifest *manifest;
@end

@implementation RemoteChatPlugin

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	self = [super init];
	if (self != nil) {
		_context = context;
		NSError *error = nil;
		_manifest = [NFKPluginManifest manifestWithBundleURL:[NSBundle bundleForClass:self.class].bundleURL error:&error];
		if (_manifest == nil) {
			[context logMessage:[NSString stringWithFormat:@"the manifest could not be read: %@", error] level:NFKPluginLogLevelError];
		}
	}
	return self;
}

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return self.manifest.models ?: @[];
}

- (NSArray<NFKModelOption *> *)pluginOptions
{
	return self.manifest.options ?: @[];
}

/*! A plugin option's value: the app's value when it set one, else the manifest's default. The context
	carries only the values the app set. */
- (nullable id)optionValueForKey:(NSString *)key
{
	id value = self.context.options[key];
	if (value != nil) {
		return value;
	}
	for (NFKModelOption *option in self.manifest.options) {
		if ([option.key isEqualToString:key]) {
			return option.defaultValue;
		}
	}
	return nil;
}

/*! A plugin option's text, or nil when it is empty. */
- (nullable NSString *)textOption:(NSString *)key
{
	id value = [self optionValueForKey:key];
	return [value isKindOfClass:NSString.class] && [value length] > 0 ? value : nil;
}

/*! The chat-completions endpoint under the baseURL option, or nil when the option is not an http or
	https URL. */
- (nullable NSURL *)endpointURL
{
	NSString *text = [self textOption:RemoteChatBaseURLOption];
	NSURL *baseURL = text != nil ? [NSURL URLWithString:text] : nil;
	NSString *scheme = baseURL.scheme.lowercaseString;
	if (![scheme isEqualToString:@"http"] && ![scheme isEqualToString:@"https"]) {
		return nil;
	}
	return [baseURL URLByAppendingPathComponent:@"chat/completions"];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if (![modelIdentifier isEqualToString:RemoteChatModelIdentifier]) {
		if (error != NULL) {
			NSString *message = [NSString stringWithFormat:@"the plugin has no model \"%@\"", modelIdentifier];
			*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel userInfo:@{ NSLocalizedDescriptionKey: message }];
		}
		return nil;
	}
	NSURL *endpoint = self.endpointURL;
	if (endpoint == nil) {
		if (error != NULL) {
			NSString *message = @"the baseURL option is not an http or https URL";
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceNotReady userInfo:@{ NSLocalizedDescriptionKey: message }];
		}
		return nil;
	}
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:endpoint];
	backend.modelName = [self textOption:RemoteChatModelOption];
	backend.apiKey = [self textOption:RemoteChatAPIKeyOption];
	return backend;
}

@end
