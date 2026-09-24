//
//  NFKTextEmbeddingBackend.m
//  InferKit
//

#import "NFKTextEmbeddingBackend.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKErrors.h"
#import <NaturalLanguage/NaturalLanguage.h>

@implementation NFKTextEmbeddingBackend

+ (instancetype)backend
{
	return [[self alloc] init];
}

+ (instancetype)backendWithLanguage:(NSString *)language
{
	NFKTextEmbeddingBackend *backend = [[self alloc] init];
	backend.language = language;
	return backend;
}

// Apple ships a sentence model for few languages and names no list, so the answer is the set that
// actually loads.
+ (NSArray<NSString *> *)availableLanguages
{
	NSArray<NLLanguage> *candidates = @[ NLLanguageEnglish, NLLanguageFrench, NLLanguageGerman,
										 NLLanguageItalian, NLLanguagePortuguese, NLLanguageSpanish,
										 NLLanguageSimplifiedChinese, NLLanguageTraditionalChinese,
										 NLLanguageJapanese, NLLanguageKorean, NLLanguageRussian,
										 NLLanguageDutch, NLLanguageSwedish, NLLanguageTurkish,
										 NLLanguageArabic, NLLanguagePolish, NLLanguageThai,
										 NLLanguageVietnamese, NLLanguageIndonesian, NLLanguageHindi ];
	NSMutableArray<NSString *> *available = [NSMutableArray array];
	for (NLLanguage language in candidates) {
		if ([NLEmbedding sentenceEmbeddingForLanguage:language] != nil) {
			[available addObject:language];
		}
	}
	return available;
}

- (BOOL)isReady
{
	return [self embeddingForLanguage:self.language] != nil;
}

- (NSString *)backendIdentifier
{
	return @"natural-language-embedding";
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObject:NFKInputPrompt];
}

- (NSInteger)dimension
{
	return (NSInteger)[self embeddingForLanguage:self.language].dimension;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError **)outError
{
	NSString *text = [request inputForKey:NFKInputPrompt];
	if (![text isKindOfClass:NSString.class] || text.length == 0) {
		[self failWithError:outError
					   code:kNFKError_InferenceMissingInput
					 reason:@"no text is set under NFKInputPrompt"];
		return nil;
	}

	NSString *language = self.language ?: [self detectedLanguageIn:text];
	NLEmbedding *embedding = [self embeddingForLanguage:language];
	if (embedding == nil) {
		NSString *reason = [NSString stringWithFormat:@"Apple ships no sentence model for %@",
							language ?: @"this text's language"];
		[self failWithError:outError code:kNFKError_InferenceNotReady reason:reason];
		return nil;
	}

	NSArray<NSNumber *> *vector = [embedding vectorForString:text];
	if (vector == nil) {
		[self failWithError:outError
					   code:kNFKError_InferenceBackendFailure
					 reason:@"the model produced no vector for this text"];
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputEmbedding: vector }];
}

- (nullable NLEmbedding *)embeddingForLanguage:(nullable NSString *)language
{
	if (language.length == 0) {
		return [NLEmbedding sentenceEmbeddingForLanguage:NLLanguageEnglish];
	}
	return [NLEmbedding sentenceEmbeddingForLanguage:(NLLanguage)language];
}

- (nullable NSString *)detectedLanguageIn:(NSString *)text
{
	NLLanguageRecognizer *recognizer = [[NLLanguageRecognizer alloc] init];
	[recognizer processString:text];
	return recognizer.dominantLanguage;
}

- (BOOL)failWithError:(NSError **)error code:(NSInteger)code reason:(NSString *)reason
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
