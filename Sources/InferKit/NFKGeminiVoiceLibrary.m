//
//  NFKGeminiVoiceLibrary.m
//  InferKit
//

#import <InferKit/NFKGeminiVoiceLibrary.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! The voice fields the library writes itself; an attribute of the same name does not overwrite them. */
static NSSet<NSString *> *NFKGeminiVoiceReservedFields(void)
{
	static NSSet<NSString *> *fields;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		fields = [NSSet setWithArray:@[ @"type", @"prompted", @"replicated", @"display_name" ]];
	});
	return fields;
}

@implementation NFKGeminiVoiceLibrary

@synthesize session = _session;

+ (nullable instancetype)voiceLibraryForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey
{
	if (![provider.identifier isEqualToString:@"gemini"]) {
		return nil;
	}
	NFKGeminiVoiceLibrary *library = [[self alloc] init];
	// The preset's base is the OpenAI layer under v1beta; the voices collection is its sibling.
	library.endpointURL = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"voices"];
	library.apiKey = apiKey;
	return library;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/voices"];
		_timeout = 60.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

#pragma mark Listing

- (nullable NSArray<NFKRemoteVoice *> *)voicesMatchingFilters:(nullable NSDictionary<NSString *, id> *)filters
														error:(NSError * _Nullable *)outError
{
	NSMutableArray<NFKRemoteVoice *> *voices = [NSMutableArray array];
	NSMutableSet<NSString *> *seenTokens = [NSMutableSet set];
	NSString *pageToken = nil;
	do {
		NSURL *url = [self listURLWithFilters:filters pageToken:pageToken];
		if (url == nil) {
			return [self failWithCode:kNFKError_InferenceMissingInput reason:@"no endpoint URL is set" error:outError];
		}
		NSDictionary *page = [self JSONForRequest:[self requestTo:url method:@"GET" body:nil] error:outError];
		if (page == nil) {
			return nil;
		}
		NSArray *entries = [page[@"voices"] isKindOfClass:NSArray.class] ? page[@"voices"] : @[];
		for (NSDictionary *entry in entries) {
			NFKRemoteVoice *voice = [self voiceFromEntry:entry];
			if (voice != nil) {
				[voices addObject:voice];
			}
		}
		pageToken = [page[@"next_page_token"] isKindOfClass:NSString.class] ? page[@"next_page_token"] : nil;
		// A service that hands back a token it already gave would otherwise loop forever.
		if (pageToken.length > 0 && [seenTokens containsObject:pageToken]) {
			break;
		}
		if (pageToken.length > 0) {
			[seenTokens addObject:pageToken];
		}
	} while (pageToken.length > 0);
	return voices;
}

- (nullable NSURL *)listURLWithFilters:(nullable NSDictionary<NSString *, id> *)filters pageToken:(nullable NSString *)pageToken
{
	if (self.endpointURL == nil) {
		return nil;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:self.endpointURL resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
	for (NSString *name in [filters.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		id value = filters[name];
		for (id element in [value isKindOfClass:NSArray.class] ? value : @[ value ]) {
			[items addObject:[NSURLQueryItem queryItemWithName:name value:[element description]]];
		}
	}
	if (pageToken.length > 0) {
		[items addObject:[NSURLQueryItem queryItemWithName:@"page_token" value:pageToken]];
	}
	components.queryItems = items.count > 0 ? items : nil;
	return components.URL;
}

- (nullable NFKRemoteVoice *)voiceWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSURL *url = [self URLForIdentifier:identifier];
	if (url == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"no voice identifier or endpoint URL is set" error:outError];
	}
	NSDictionary *entry = [self JSONForRequest:[self requestTo:url method:@"GET" body:nil] error:outError];
	return entry == nil ? nil : [self voiceInReply:entry error:outError];
}

#pragma mark Creating

- (nullable NFKRemoteVoice *)designVoiceWithDescription:(NSString *)description
											displayName:(nullable NSString *)displayName
											 attributes:(nullable NSDictionary<NSString *, id> *)attributes
												  error:(NSError * _Nullable *)outError
{
	if (description.length == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"a designed voice needs a description of how it sounds" error:outError];
	}
	NSMutableDictionary<NSString *, id> *voice = [self voiceWithAttributes:attributes];
	voice[@"type"] = @"prompted";
	voice[@"prompted"] = @{ @"input": description };
	if (displayName.length > 0) {
		voice[@"display_name"] = displayName;
	}
	return [self createVoice:voice stores:YES error:outError];
}

- (nullable NFKRemoteVoice *)replicateVoiceFromAudio:(id)sourceAudio
										consentAudio:(id)consentAudio
										 displayName:(nullable NSString *)displayName
											  stores:(BOOL)stores
											   error:(NSError * _Nullable *)outError
{
	NSDictionary *source = [self audioPayloadFor:sourceAudio role:@"source" error:outError];
	if (source == nil) {
		return nil;
	}
	NSDictionary *consent = [self audioPayloadFor:consentAudio role:@"consent" error:outError];
	if (consent == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *voice = [self voiceWithAttributes:nil];
	voice[@"type"] = @"replicated";
	voice[@"replicated"] = @{ @"source_audio": source, @"consent_audio": consent };
	if (stores && displayName.length > 0) {
		voice[@"display_name"] = displayName;
	}
	return [self createVoice:voice stores:stores error:outError];
}

- (NSMutableDictionary<NSString *, id> *)voiceWithAttributes:(nullable NSDictionary<NSString *, id> *)attributes
{
	NSMutableDictionary<NSString *, id> *voice = [NSMutableDictionary dictionary];
	for (NSString *key in attributes) {
		if (![NFKGeminiVoiceReservedFields() containsObject:key]) {
			voice[key] = attributes[key];
		}
	}
	if (self.modelName.length > 0 && voice[@"model"] == nil) {
		voice[@"model"] = self.modelName;
	}
	return voice;
}

- (nullable NFKRemoteVoice *)createVoice:(NSDictionary *)voice stores:(BOOL)stores error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"no endpoint URL is set" error:outError];
	}
	NSDictionary *body = @{ @"store": @(stores), @"voice": voice };
	if (![NSJSONSerialization isValidJSONObject:body]) {
		return [self failWithCode:kNFKError_InferenceMissingInput reason:@"an attribute is not JSON-serializable" error:outError];
	}
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:outError];
	if (payload == nil) {
		return nil;
	}
	NSDictionary *reply = [self JSONForRequest:[self requestTo:self.endpointURL method:@"POST" body:payload] error:outError];
	return reply == nil ? nil : [self voiceInReply:reply error:outError];
}

/*! An audio clip as the service's {mime_type, data} payload. The format comes from a file's
	extension or from the bytes' own signature, because the service needs the type named. */
- (nullable NSDictionary<NSString *, NSString *> *)audioPayloadFor:(id)audio role:(NSString *)role error:(NSError * _Nullable *)outError
{
	NSData *data = nil;
	NSString *mimeType = nil;
	if ([audio isKindOfClass:NSURL.class]) {
		NSDictionary<NSString *, NSString *> *types = @{ @"wav": @"audio/wav", @"wave": @"audio/wav", @"mp3": @"audio/mpeg", @"flac": @"audio/flac" };
		mimeType = types[[audio pathExtension].lowercaseString];
		data = mimeType == nil ? nil : [NSData dataWithContentsOfURL:audio options:0 error:outError];
		if (mimeType == nil) {
			[self failWithCode:kNFKError_InferenceUnsupported
						reason:[NSString stringWithFormat:@"the %@ audio is not a .wav, .mp3, or .flac file", role] error:outError];
		}
	} else if ([audio isKindOfClass:NSData.class]) {
		data = audio;
		mimeType = [self mimeTypeOfAudioBytes:audio];
		if (mimeType == nil) {
			[self failWithCode:kNFKError_InferenceUnsupported
						reason:[NSString stringWithFormat:@"the %@ audio is not WAV, MP3, or FLAC", role] error:outError];
		}
	} else {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:[NSString stringWithFormat:@"the %@ audio is not an NSData or a file NSURL", role] error:outError];
	}
	if (data == nil || mimeType == nil) {
		return nil;
	}
	return @{ @"mime_type": mimeType, @"data": [data base64EncodedStringWithOptions:0] };
}

- (nullable NSString *)mimeTypeOfAudioBytes:(NSData *)data
{
	if (data.length < 4) {
		return nil;
	}
	const unsigned char *bytes = data.bytes;
	if (memcmp(bytes, "RIFF", 4) == 0) {
		return @"audio/wav";
	}
	if (memcmp(bytes, "fLaC", 4) == 0) {
		return @"audio/flac";
	}
	if (memcmp(bytes, "ID3", 3) == 0 || (bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0)) {
		return @"audio/mpeg";
	}
	return nil;
}

#pragma mark Deleting

- (BOOL)deleteVoiceWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError
{
	NSURL *url = [self URLForIdentifier:identifier];
	if (url == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"no voice identifier or endpoint URL is set" error:outError];
		return NO;
	}
	return [self JSONForRequest:[self requestTo:url method:@"DELETE" body:nil] error:outError] != nil;
}

#pragma mark The wire

/*! The voice's resource URL. An identifier given as its resource name (voices/voice_…) loses the
	collection prefix. */
- (nullable NSURL *)URLForIdentifier:(NSString *)identifier
{
	NSString *name = [identifier hasPrefix:@"voices/"] ? [identifier substringFromIndex:7] : identifier;
	if (name.length == 0 || self.endpointURL == nil) {
		return nil;
	}
	return [self.endpointURL URLByAppendingPathComponent:name];
}

- (NSMutableURLRequest *)requestTo:(NSURL *)url method:(NSString *)method body:(nullable NSData *)body
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	if (body != nil) {
		request.HTTPBody = body;
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	}
	if (self.apiKey.length > 0) {
		[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}
	return request;
}

/*! The reply as a JSON object; an empty body (a deletion) reads as an empty object. */
- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	if (data == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	if (data.length == 0) {
		return @{};
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![reply isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the reply is not a JSON object" error:outError];
	}
	return reply;
}

- (nullable NFKRemoteVoice *)voiceInReply:(NSDictionary *)reply error:(NSError * _Nullable *)outError
{
	NFKRemoteVoice *voice = [self voiceFromEntry:reply];
	if (voice == nil) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the reply names no voice id or key" error:outError];
	}
	return voice;
}

/*! A stored or prebuilt voice is named by its id; a voice that is not stored, by its key. */
- (nullable NFKRemoteVoice *)voiceFromEntry:(NSDictionary *)entry
{
	if (![entry isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *identifier = [entry[@"id"] isKindOfClass:NSString.class] ? entry[@"id"] : nil;
	if (identifier.length == 0 && [entry[@"key"] isKindOfClass:NSString.class]) {
		identifier = entry[@"key"];
	}
	if (identifier.length == 0) {
		return nil;
	}
	NSString *name = [entry[@"display_name"] isKindOfClass:NSString.class] ? entry[@"display_name"] : nil;
	NSArray *languages = [entry[@"language_code"] isKindOfClass:NSString.class] ? @[ entry[@"language_code"] ] : @[];
	return [[NFKRemoteVoice alloc] initWithIdentifier:identifier name:name languages:languages raw:entry];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable id)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
