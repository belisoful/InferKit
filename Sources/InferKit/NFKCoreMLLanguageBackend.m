//
//  NFKCoreMLLanguageBackend.m
//  InferKit
//

#import "NFKCoreMLLanguageBackend.h"
#import "NFKTokenizer.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceKeys.h"
#import "NFKErrors.h"
#import <CoreML/CoreML.h>

#pragma mark Sampling primitives

// SplitMix64: a small, seedable generator, so a fixed NFKParameterSeed reproduces a run.
static uint64_t NFKSplitMix64(uint64_t *state)
{
	uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static double NFKNextUniform(uint64_t *state)
{
	return (NFKSplitMix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

static float NFKHalfToFloat(uint16_t half)
{
	uint32_t sign = (uint32_t)(half & 0x8000) << 16;
	uint32_t exponent = (half >> 10) & 0x1F;
	uint32_t mantissa = half & 0x3FF;
	uint32_t bits;
	if (exponent == 0) {
		if (mantissa == 0) {
			bits = sign;
		} else {
			exponent = 127 - 15 + 1;
			while ((mantissa & 0x400) == 0) {
				mantissa <<= 1;
				exponent -= 1;
			}
			mantissa &= 0x3FF;
			bits = sign | (exponent << 23) | (mantissa << 13);
		}
	} else if (exponent == 0x1F) {
		bits = sign | 0x7F800000 | (mantissa << 13);
	} else {
		bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
	}
	float result;
	memcpy(&result, &bits, sizeof(result));
	return result;
}

typedef struct {
	int32_t token;
	float score;
} NFKTokenScore;

static int NFKTokenScoreCompareDescending(const void *a, const void *b)
{
	float lhs = ((const NFKTokenScore *)a)->score;
	float rhs = ((const NFKTokenScore *)b)->score;
	if (lhs < rhs) {
		return 1;
	}
	if (lhs > rhs) {
		return -1;
	}
	return 0;
}

#pragma mark -

API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0))
@implementation NFKCoreMLLanguageBackend
{
	MLModel *_model;
	MLModel *_prefillModel;
	NSInteger _prefillLength;
	NFKTokenizer *_tokenizer;
	NSString *_inputFeatureName;
	NSString *_logitsFeatureName;
	NSString *_positionFeatureName;
	NSDictionary *_chatTemplate;
	NSInteger _contextLength;
}

@synthesize modelDirectoryURL = _modelDirectoryURL;
@synthesize computeUnits = _computeUnits;

+ (instancetype)backendWithModelDirectoryURL:(nullable NSURL *)modelDirectoryURL
{
	NFKCoreMLLanguageBackend *backend = [[self alloc] init];
	backend.modelDirectoryURL = modelDirectoryURL;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_computeUnits = MLComputeUnitsAll;
	}
	return self;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return _model != nil && _tokenizer != nil;
}

- (NSString *)backendIdentifier
{
	return @"coreml-llm";
}

- (BOOL)prepareWithError:(NSError * _Nullable *)outError
{
	if ([self isReady]) {
		return YES;
	}
	if (self.modelDirectoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no model directory is set"];
		return NO;
	}
	return [self loadModelFromDirectory:self.modelDirectoryURL error:outError];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError * _Nullable *)outError
{
	return [self generateForRequest:request progress:nil isCancelled:nil error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	__block BOOL cancelled = NO;
	job.cancellationHandler = ^{
		cancelled = YES;
	};
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKInferenceResult *result =
			[self generateForRequest:request
							progress:^(double progress, NSString *partialText) {
							[job reportProgress:progress
								partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: partialText }]];
						}
						 isCancelled:^BOOL { return cancelled; }
							   error:&error];
		if (result != nil) {
			[job finishWithResult:result];
		} else if (cancelled) {
			[job cancel];
		} else {
			[job finishWithError:error ?: [self errorWithCode:kNFKError_InferenceBackendFailure reason:@"generation failed"]];
		}
	});
	return job;
}

#pragma mark Loading

- (BOOL)loadModelFromDirectory:(NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	if (directoryURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no model directory to load"];
		return NO;
	}

	NSDictionary *manifest = [self manifestInDirectory:directoryURL error:outError];
	if (manifest == nil) {
		return NO;
	}

	NFKTokenizer *tokenizer = [NFKTokenizer tokenizerForManifest:manifest directory:directoryURL error:outError];
	if (tokenizer == nil) {
		return NO;
	}

	NSURL *modelURL = [self modelURLInDirectory:directoryURL manifest:manifest];
	if (modelURL == nil) {
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"the directory has no model.mlpackage or model.mlmodelc"];
		return NO;
	}
	NSURL *compiledURL = [self compileModelAtURL:modelURL error:outError];
	if (compiledURL == nil) {
		return NO;
	}
	MLModel *model = [self loadModelAtCompiledURL:compiledURL functionName:nil
									 computeUnits:self.computeUnits error:outError];
	if (model == nil) {
		return NO;
	}

	// A multifunction package declares a prefill function that runs a fixed chunk of prompt tokens
	// through the same weights and KV-cache state, so a prompt costs N/length calls instead of N.
	MLModel *prefillModel = nil;
	NSInteger prefillLength = 0;
	NSDictionary *prefillSection = [manifest[@"prefill"] isKindOfClass:NSDictionary.class] ? manifest[@"prefill"] : nil;
	if (prefillSection != nil) {
		NSString *functionName = [prefillSection[@"function"] isKindOfClass:NSString.class] ? prefillSection[@"function"] : @"prefill";
		prefillLength = [prefillSection[@"length"] isKindOfClass:NSNumber.class] ? [prefillSection[@"length"] integerValue] : 0;
		if (prefillLength > 1) {
			// The Neural Engine can fail to compile this graph, and with a functionName the load
			// surfaces that as an error instead of falling back. Retry without the ANE; when even
			// that fails, run without prefill rather than failing the load, since prefill only
			// accelerates prompt processing.
			prefillModel = [self loadModelAtCompiledURL:compiledURL functionName:functionName
										   computeUnits:self.computeUnits error:NULL];
			if (prefillModel == nil && self.computeUnits != MLComputeUnitsCPUAndGPU) {
				prefillModel = [self loadModelAtCompiledURL:compiledURL functionName:functionName
											   computeUnits:MLComputeUnitsCPUAndGPU error:NULL];
			}
		}
	}

	_model = model;
	_prefillModel = prefillModel;
	_prefillLength = prefillModel != nil ? prefillLength : 0;
	_tokenizer = tokenizer;
	_inputFeatureName = [manifest[@"inputFeature"] isKindOfClass:NSString.class] ? manifest[@"inputFeature"] : @"input_ids";
	_logitsFeatureName = [manifest[@"logitsFeature"] isKindOfClass:NSString.class] ? manifest[@"logitsFeature"] : @"logits";
	// Optional: a model that tracks its KV cache by position takes a cache_position input each step.
	_positionFeatureName = [manifest[@"positionFeature"] isKindOfClass:NSString.class] ? manifest[@"positionFeature"] : nil;
	NSDictionary *tokenizerSection = [manifest[@"tokenizer"] isKindOfClass:NSDictionary.class] ? manifest[@"tokenizer"] : nil;
	_chatTemplate = [tokenizerSection[@"chatTemplate"] isKindOfClass:NSDictionary.class] ? tokenizerSection[@"chatTemplate"] : nil;
	_contextLength = [manifest[@"contextLength"] isKindOfClass:NSNumber.class] ? [manifest[@"contextLength"] integerValue] : 0;
	self.modelDirectoryURL = directoryURL;
	return YES;
}

- (nullable NSDictionary *)manifestInDirectory:(NSURL *)directoryURL error:(NSError * _Nullable *)outError
{
	NSURL *manifestURL = [directoryURL URLByAppendingPathComponent:@"manifest.json"];
	NSData *data = [NSData dataWithContentsOfURL:manifestURL options:0 error:outError];
	if (data == nil) {
		return nil;
	}
	id manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:outError];
	if (![manifest isKindOfClass:NSDictionary.class]) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"manifest.json is not a JSON object"];
		return nil;
	}
	return manifest;
}

- (nullable NSURL *)modelURLInDirectory:(NSURL *)directoryURL manifest:(NSDictionary *)manifest
{
	NSString *named = [manifest[@"model"] isKindOfClass:NSString.class] ? manifest[@"model"] : nil;
	NSFileManager *fileManager = [NSFileManager defaultManager];
	NSArray<NSString *> *candidates = named != nil ? @[named] : @[@"model.mlmodelc", @"model.mlpackage", @"model.mlmodel"];
	for (NSString *candidate in candidates) {
		NSURL *url = [directoryURL URLByAppendingPathComponent:candidate];
		if ([fileManager fileExistsAtPath:url.path]) {
			return url;
		}
	}
	return nil;
}

- (nullable NSURL *)compileModelAtURL:(NSURL *)url error:(NSError * _Nullable *)outError
{
	NSString *extension = url.pathExtension.lowercaseString;
	if (![extension isEqualToString:@"mlpackage"] && ![extension isEqualToString:@"mlmodel"]) {
		return url;
	}
	NSError *error = nil;
	NSURL *compiledURL = [MLModel compileModelAtURL:url error:&error];
	if (compiledURL == nil) {
		[self propagateError:error to:outError];
		return nil;
	}
	return compiledURL;
}

- (nullable MLModel *)loadModelAtCompiledURL:(NSURL *)compiledURL
								functionName:(nullable NSString *)functionName
								computeUnits:(MLComputeUnits)computeUnits
									   error:(NSError * _Nullable *)outError
{
	MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
	configuration.computeUnits = computeUnits;
	if (functionName != nil) {
		configuration.functionName = functionName;
	}
	NSError *error = nil;
	MLModel *model = [MLModel modelWithContentsOfURL:compiledURL configuration:configuration error:&error];
	if (model == nil) {
		[self propagateError:error to:outError];
		return nil;
	}
	return model;
}

#pragma mark Generation

- (nullable NFKInferenceResult *)generateForRequest:(NFKInferenceRequest *)request
											progress:(nullable void (^)(double progress, NSString *partialText))progress
										 isCancelled:(nullable BOOL (^)(void))isCancelled
											   error:(NSError * _Nullable *)outError
{
	if (![self isReady] && ![self prepareWithError:outError]) {
		return nil;
	}

	BOOL templated = NO;
	NSString *prompt = [self promptForRequest:request templated:&templated];
	if (prompt == nil) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request has no prompt or messages"];
		return nil;
	}

	NSArray<NSNumber *> *promptTokens = [self promptTokensForText:prompt prependBos:!templated];
	if (promptTokens.count == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the prompt is empty after tokenization"];
		return nil;
	}

	double temperature = [self doubleParameter:NFKParameterTemperature default:1.0 request:request];
	NSInteger topK = [self integerParameter:NFKParameterTopK default:0 request:request];
	double topP = [self doubleParameter:NFKParameterTopP default:1.0 request:request];
	double repetitionPenalty = [self doubleParameter:NFKParameterRepetitionPenalty default:1.0 request:request];
	NSInteger maxTokens = [self integerParameter:NFKParameterMaxTokens default:256 request:request];
	NSArray<NSString *> *stops = [self stopSequencesForRequest:request];
	// A chat turn ends at the assistant suffix (for example <|im_end|>); stop there so the reply is
	// just the answer, not the start of the next turn.
	if (templated) {
		NSString *turnStop = [self chatTurnStop];
		if (turnStop.length > 0) {
			stops = [stops arrayByAddingObject:turnStop];
		}
	}
	uint64_t rng = (uint64_t)[self integerParameter:NFKParameterSeed default:0 request:request];

	MLState *state = [_model newState];
	NSInteger cachePosition = 0;

	// Prefill in fixed-size chunks through the prefill function when the package has one, then
	// finish the remainder one token at a time through decode. Both write the same KV-cache state.
	// The logits after the last prompt token seed generation.
	NSError *error = nil;
	MLMultiArray *logits = nil;
	NSUInteger consumed = 0;
	while (_prefillModel != nil && promptTokens.count - consumed >= (NSUInteger)_prefillLength) {
		NSArray<NSNumber *> *chunk = [promptTokens subarrayWithRange:NSMakeRange(consumed, _prefillLength)];
		logits = [self runTokens:chunk position:cachePosition model:_prefillModel intoState:state error:&error];
		if (logits == nil) {
			[self propagateError:error to:outError];
			return nil;
		}
		consumed += (NSUInteger)_prefillLength;
		cachePosition += _prefillLength;
	}
	for (NSUInteger i = consumed; i < promptTokens.count; i++) {
		logits = [self runTokens:@[promptTokens[i]] position:cachePosition model:_model intoState:state error:&error];
		if (logits == nil) {
			[self propagateError:error to:outError];
			return nil;
		}
		cachePosition += 1;
	}

	NSMutableArray<NSNumber *> *generated = [NSMutableArray array];
	NSMutableSet<NSNumber *> *penalized = [NSMutableSet setWithArray:promptTokens];
	NSMutableString *text = [NSMutableString string];

	for (NSInteger step = 0; step < maxTokens; step++) {
		if (isCancelled != nil && isCancelled()) {
			return nil;
		}

		NSInteger next = [self sampleFromLogits:logits
									temperature:temperature
										   topK:topK
										   topP:topP
							  repetitionPenalty:repetitionPenalty
									  penalized:penalized
											rng:&rng];
		if (next < 0) {
			break;
		}
		if (_tokenizer.eosTokenId >= 0 && next == _tokenizer.eosTokenId) {
			break;
		}

		[generated addObject:@(next)];
		[penalized addObject:@(next)];
		[text setString:[_tokenizer decode:generated]];

		NSString *trimmed = [self textTrimmedAtStop:text stops:stops];
		if (trimmed != nil) {
			return [self resultForText:trimmed];
		}

		if (progress != nil) {
			progress((double)(step + 1) / (double)maxTokens, text);
		}

		logits = [self runTokens:@[@(next)] position:cachePosition model:_model intoState:state error:&error];
		if (logits == nil) {
			[self propagateError:error to:outError];
			return nil;
		}
		cachePosition += 1;
	}

	return [self resultForText:text];
}

- (NFKInferenceResult *)resultForText:(NSString *)text
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: text }];
}

- (nullable NSString *)textTrimmedAtStop:(NSString *)text stops:(NSArray<NSString *> *)stops
{
	for (NSString *stop in stops) {
		if (stop.length > 0 && [text hasSuffix:stop]) {
			return [text substringToIndex:text.length - stop.length];
		}
	}
	return nil;
}

- (nullable MLMultiArray *)runTokens:(NSArray<NSNumber *> *)tokens
							position:(NSInteger)position
							   model:(MLModel *)model
						   intoState:(MLState *)state
							   error:(NSError * _Nullable *)outError
{
	MLMultiArray *input = [[MLMultiArray alloc] initWithShape:@[@1, @(tokens.count)]
													dataType:MLMultiArrayDataTypeInt32
													   error:outError];
	if (input == nil) {
		return nil;
	}
	int32_t *pointer = (int32_t *)input.dataPointer;
	for (NSUInteger i = 0; i < tokens.count; i++) {
		pointer[i] = (int32_t)tokens[i].integerValue;
	}

	NSMutableDictionary<NSString *, MLFeatureValue *> *features =
		[NSMutableDictionary dictionaryWithObject:[MLFeatureValue featureValueWithMultiArray:input] forKey:_inputFeatureName];
	if (_positionFeatureName != nil) {
		MLMultiArray *positions = [self positionArrayFrom:position count:tokens.count error:outError];
		if (positions == nil) {
			return nil;
		}
		features[_positionFeatureName] = [MLFeatureValue featureValueWithMultiArray:positions];
	}

	MLDictionaryFeatureProvider *provider =
		[[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:outError];
	if (provider == nil) {
		return nil;
	}
	id<MLFeatureProvider> prediction = [model predictionFromFeatures:provider usingState:state error:outError];
	if (prediction == nil) {
		return nil;
	}
	MLMultiArray *logits = [prediction featureValueForName:_logitsFeatureName].multiArrayValue;
	if (logits == nil) {
		[self setError:outError
				  code:kNFKError_InferenceBackendFailure
				reason:[NSString stringWithFormat:@"the model has no logits output named '%@'", _logitsFeatureName]];
		return nil;
	}
	return logits;
}

// cache_position is the absolute index of each new token in the KV cache: position, position+1, ...
- (nullable MLMultiArray *)positionArrayFrom:(NSInteger)position
									   count:(NSUInteger)count
									   error:(NSError * _Nullable *)outError
{
	MLMultiArray *positions = [[MLMultiArray alloc] initWithShape:@[@(count)]
														dataType:MLMultiArrayDataTypeInt32
														   error:outError];
	if (positions == nil) {
		return nil;
	}
	int32_t *pointer = (int32_t *)positions.dataPointer;
	for (NSUInteger i = 0; i < count; i++) {
		pointer[i] = (int32_t)(position + (NSInteger)i);
	}
	return positions;
}

- (NSInteger)sampleFromLogits:(MLMultiArray *)logits
				  temperature:(double)temperature
						 topK:(NSInteger)topK
						 topP:(double)topP
			repetitionPenalty:(double)repetitionPenalty
					penalized:(NSSet<NSNumber *> *)penalized
						  rng:(uint64_t *)rng
{
	NSUInteger vocab = 0;
	float *scores = [self lastStepLogitsFrom:logits count:&vocab];
	if (scores == NULL || vocab == 0) {
		free(scores);
		return -1;
	}

	if (repetitionPenalty != 1.0) {
		for (NSNumber *token in penalized) {
			NSInteger index = token.integerValue;
			if (index >= 0 && (NSUInteger)index < vocab) {
				scores[index] = scores[index] > 0 ? (float)(scores[index] / repetitionPenalty)
												  : (float)(scores[index] * repetitionPenalty);
			}
		}
	}

	NSInteger token;
	if (temperature <= 0.0) {
		token = [self argmaxOf:scores count:vocab];
	} else {
		token = [self sampleScores:scores count:vocab temperature:temperature topK:topK topP:topP rng:rng];
	}
	free(scores);
	return token;
}

- (NSInteger)argmaxOf:(const float *)scores count:(NSUInteger)count
{
	NSInteger best = 0;
	float bestScore = scores[0];
	for (NSUInteger i = 1; i < count; i++) {
		if (scores[i] > bestScore) {
			bestScore = scores[i];
			best = (NSInteger)i;
		}
	}
	return best;
}

- (NSInteger)sampleScores:(const float *)scores
					count:(NSUInteger)vocab
			  temperature:(double)temperature
					 topK:(NSInteger)topK
					 topP:(double)topP
					  rng:(uint64_t *)rng
{
	NFKTokenScore *ranked = malloc(sizeof(NFKTokenScore) * vocab);
	for (NSUInteger i = 0; i < vocab; i++) {
		ranked[i].token = (int32_t)i;
		ranked[i].score = (float)(scores[i] / temperature);
	}
	qsort(ranked, vocab, sizeof(NFKTokenScore), NFKTokenScoreCompareDescending);

	NSUInteger kept = vocab;
	if (topK > 0 && (NSUInteger)topK < kept) {
		kept = (NSUInteger)topK;
	}

	double maxScore = ranked[0].score;
	double *probabilities = malloc(sizeof(double) * kept);
	double sum = 0.0;
	for (NSUInteger i = 0; i < kept; i++) {
		double probability = exp(ranked[i].score - maxScore);
		probabilities[i] = probability;
		sum += probability;
	}

	NSUInteger nucleus = kept;
	if (topP > 0.0 && topP < 1.0) {
		double cumulative = 0.0;
		for (NSUInteger i = 0; i < kept; i++) {
			cumulative += probabilities[i] / sum;
			if (cumulative >= topP) {
				nucleus = i + 1;
				break;
			}
		}
	}

	double nucleusSum = 0.0;
	for (NSUInteger i = 0; i < nucleus; i++) {
		nucleusSum += probabilities[i];
	}

	double target = NFKNextUniform(rng) * nucleusSum;
	NSInteger token = ranked[0].token;
	double running = 0.0;
	for (NSUInteger i = 0; i < nucleus; i++) {
		running += probabilities[i];
		if (running >= target) {
			token = ranked[i].token;
			break;
		}
	}

	free(probabilities);
	free(ranked);
	return token;
}

// Reads the vocabulary-length logits for the final sequence position into a freshly allocated
// float buffer the caller frees. Handles the position axis via strides, so a [1, seq, vocab] or a
// [1, vocab] output both resolve to the last step.
- (nullable float *)lastStepLogitsFrom:(MLMultiArray *)logits count:(NSUInteger *)outCount
{
	NSArray<NSNumber *> *shape = logits.shape;
	NSArray<NSNumber *> *strides = logits.strides;
	NSUInteger rank = shape.count;
	if (rank < 2) {
		*outCount = 0;
		return NULL;
	}
	NSUInteger vocab = shape[rank - 1].unsignedIntegerValue;
	NSInteger vocabStride = strides[rank - 1].integerValue;
	NSInteger positions = shape[rank - 2].integerValue;
	NSInteger positionStride = strides[rank - 2].integerValue;
	NSInteger base = (positions - 1) * positionStride;

	float *out = malloc(sizeof(float) * vocab);
	const void *pointer = logits.dataPointer;
	switch (logits.dataType) {
		case MLMultiArrayDataTypeFloat32: {
			const float *values = (const float *)pointer;
			for (NSUInteger v = 0; v < vocab; v++) {
				out[v] = values[base + (NSInteger)v * vocabStride];
			}
			break;
		}
		case MLMultiArrayDataTypeDouble: {
			const double *values = (const double *)pointer;
			for (NSUInteger v = 0; v < vocab; v++) {
				out[v] = (float)values[base + (NSInteger)v * vocabStride];
			}
			break;
		}
		case MLMultiArrayDataTypeFloat16: {
			const uint16_t *values = (const uint16_t *)pointer;
			for (NSUInteger v = 0; v < vocab; v++) {
				out[v] = NFKHalfToFloat(values[base + (NSInteger)v * vocabStride]);
			}
			break;
		}
		default:
			free(out);
			*outCount = 0;
			return NULL;
	}
	*outCount = vocab;
	return out;
}

#pragma mark Request reading

// Returns the prompt string. *templated is YES when a messages array was rendered with the model's
// chat template, which already carries the model's special tokens (so BOS is not prepended again).
- (nullable NSString *)promptForRequest:(NFKInferenceRequest *)request templated:(BOOL *)templated
{
	*templated = NO;
	id messages = [request inputForKey:NFKInputMessages];
	if ([messages isKindOfClass:NSArray.class]) {
		if (_chatTemplate != nil) {
			*templated = YES;
			return [self renderChatTemplate:messages];
		}
		return [self promptFromMessages:messages];
	}
	id prompt = [request inputForKey:NFKInputPrompt];
	if ([prompt isKindOfClass:NSString.class]) {
		return prompt;
	}
	return nil;
}

// Renders messages with the manifest's chat template: per-role prefix/suffix markers, the default
// system message when the caller supplies none, and the generation prompt that invites the reply.
- (NSString *)renderChatTemplate:(NSArray *)messages
{
	NSMutableString *prompt = [NSMutableString string];

	NSString *defaultSystem = _chatTemplate[@"defaultSystem"];
	if ([defaultSystem isKindOfClass:NSString.class] && defaultSystem.length > 0 && ![self messages:messages containRole:@"system"]) {
		[self appendRole:@"system" content:defaultSystem to:prompt];
	}
	for (id entry in messages) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *role = [entry[@"role"] isKindOfClass:NSString.class] ? entry[@"role"] : @"user";
		NSString *content = [entry[@"content"] isKindOfClass:NSString.class] ? entry[@"content"] : @"";
		[self appendRole:role content:content to:prompt];
	}
	NSString *generationPrompt = _chatTemplate[@"generationPrompt"];
	if ([generationPrompt isKindOfClass:NSString.class]) {
		[prompt appendString:generationPrompt];
	}
	return prompt;
}

- (BOOL)messages:(NSArray *)messages containRole:(NSString *)role
{
	for (id entry in messages) {
		if ([entry isKindOfClass:NSDictionary.class] && [[entry objectForKey:@"role"] isEqual:role]) {
			return YES;
		}
	}
	return NO;
}

- (void)appendRole:(NSString *)role content:(NSString *)content to:(NSMutableString *)prompt
{
	NSArray *markers = [_chatTemplate[role] isKindOfClass:NSArray.class] ? _chatTemplate[role] : _chatTemplate[@"user"];
	if ([markers isKindOfClass:NSArray.class] && markers.count >= 2) {
		[prompt appendString:markers[0]];
		[prompt appendString:content];
		[prompt appendString:markers[1]];
	}
}

// The assistant suffix, trimmed of trailing whitespace, as the stop that ends a chat turn.
- (nullable NSString *)chatTurnStop
{
	NSArray *assistant = _chatTemplate[@"assistant"];
	if ([assistant isKindOfClass:NSArray.class] && assistant.count >= 2 && [assistant[1] isKindOfClass:NSString.class]) {
		return [assistant[1] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
	}
	return nil;
}

// A minimal role-prefixed rendering for a plain messages array when the model ships no chat template.
- (NSString *)promptFromMessages:(NSArray *)messages
{
	NSMutableString *prompt = [NSMutableString string];
	for (id entry in messages) {
		if (![entry isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *role = [entry[@"role"] isKindOfClass:NSString.class] ? entry[@"role"] : @"user";
		NSString *content = [entry[@"content"] isKindOfClass:NSString.class] ? entry[@"content"] : @"";
		[prompt appendFormat:@"%@: %@\n", role, content];
	}
	[prompt appendString:@"assistant:"];
	return prompt;
}

- (NSArray<NSNumber *> *)promptTokensForText:(NSString *)text prependBos:(BOOL)prependBos
{
	NSArray<NSNumber *> *tokens = [_tokenizer encode:text];
	if (prependBos && _tokenizer.bosTokenId >= 0) {
		NSMutableArray<NSNumber *> *withBos = [NSMutableArray arrayWithObject:@(_tokenizer.bosTokenId)];
		[withBos addObjectsFromArray:tokens];
		return withBos;
	}
	return tokens;
}

- (NSArray<NSString *> *)stopSequencesForRequest:(NFKInferenceRequest *)request
{
	id stops = [request parameterForKey:NFKParameterStopSequences];
	if ([stops isKindOfClass:NSArray.class]) {
		NSMutableArray<NSString *> *strings = [NSMutableArray array];
		for (id stop in stops) {
			if ([stop isKindOfClass:NSString.class]) {
				[strings addObject:stop];
			}
		}
		return strings;
	}
	if ([stops isKindOfClass:NSString.class]) {
		return @[stops];
	}
	return @[];
}

- (double)doubleParameter:(NSString *)key default:(double)fallback request:(NFKInferenceRequest *)request
{
	id value = [request parameterForKey:key];
	return [value isKindOfClass:NSNumber.class] ? [value doubleValue] : fallback;
}

- (NSInteger)integerParameter:(NSString *)key default:(NSInteger)fallback request:(NFKInferenceRequest *)request
{
	id value = [request parameterForKey:key];
	return [value isKindOfClass:NSNumber.class] ? [value integerValue] : fallback;
}

#pragma mark Errors

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [self errorWithCode:code reason:reason];
	}
	return NO;
}

- (NSError *)errorWithCode:(NSInteger)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	*outError = error ?: [self errorWithCode:kNFKError_InferenceBackendFailure reason:@"the Core ML run failed"];
	return NO;
}

@end
