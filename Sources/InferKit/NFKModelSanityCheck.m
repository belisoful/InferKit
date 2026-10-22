//
//  NFKModelSanityCheck.m
//  InferKit
//

#import "NFKModelSanityCheck.h"
#import "NFKErrors.h"
#import "NFKInferenceKeys.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"

/*! The longest stretch, in characters, whose repetition at the end of a reply counts as a loop. */
static const NSUInteger NFKModelSanityLongestStretch = 32;

@implementation NFKModelSanityResult

- (instancetype)initWithOutput:(NSString *)output findings:(NFKModelSanityFinding)findings
{
	self = [super init];
	if (self != nil) {
		_output = [output copy];
		_findings = findings;
	}
	return self;
}

- (BOOL)isPassed
{
	return self.findings == NFKModelSanityFindingNone;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ findings=%lu %@>", NSStringFromClass(self.class),
			(unsigned long)self.findings, self.output];
}

@end

@implementation NFKModelSanityCheck

+ (NSString *)defaultPrompt
{
	return @"The capital of France is";
}

+ (NSInteger)defaultTokenCount
{
	return 48;
}

+ (nullable NFKModelSanityResult *)checkBackend:(id<NFKInferenceBackend>)backend error:(NSError **)error
{
	return [self checkBackend:backend prompt:self.defaultPrompt tokenCount:self.defaultTokenCount error:error];
}

+ (nullable NFKModelSanityResult *)checkBackend:(id<NFKInferenceBackend>)backend
										 prompt:(NSString *)prompt
									 tokenCount:(NSInteger)tokenCount
										  error:(NSError **)error
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: prompt }
															   parameters:@{ NFKParameterTemperature: @0,
																			 NFKParameterMaxTokens: @(tokenCount) }];
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:error];
	if (result == nil) {
		return nil;
	}
	id output = [result outputForKey:NFKOutputText];
	if (![output isKindOfClass:NSString.class]) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
									 userInfo:@{ NSLocalizedDescriptionKey: @"The backend's result holds no text to check." }];
		}
		return nil;
	}
	return [self resultForOutput:output];
}

+ (NFKModelSanityResult *)resultForOutput:(NSString *)output
{
	NSString *text = [output stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
	NFKModelSanityFinding findings = NFKModelSanityFindingNone;
	if (text.length == 0) {
		findings |= NFKModelSanityFindingEmpty;
	}
	if ([self isGarbled:text]) {
		findings |= NFKModelSanityFindingGarbled;
	}
	if ([self isRepetitive:text]) {
		findings |= NFKModelSanityFindingRepetitive;
	}
	return [[NFKModelSanityResult alloc] initWithOutput:output findings:findings];
}

/*! YES for the replacement character or a control character other than a tab or a line break. */
+ (BOOL)isGarbled:(NSString *)text
{
	NSMutableCharacterSet *undecoded = [NSMutableCharacterSet controlCharacterSet];
	[undecoded removeCharactersInString:@"\t\n\r"];
	[undecoded addCharactersInString:@"�"];
	return [text rangeOfCharacterFromSet:undecoded].location != NSNotFound;
}

/*! YES when the text ends in a stretch of up to NFKModelSanityLongestStretch characters repeated (ten
	times for one or two characters, so a number such as 1000000 passes, six for three or four, and three
	for a longer one), or when a text of 32 characters or more uses three distinct characters or fewer. */
+ (BOOL)isRepetitive:(NSString *)text
{
	NSUInteger length = text.length;
	if (length >= 32 && [self distinctCharacterCountOf:text] <= 3) {
		return YES;
	}
	for (NSUInteger stretch = 1; stretch <= NFKModelSanityLongestStretch; stretch++) {
		NSUInteger repeats = stretch <= 2 ? 10 : stretch <= 4 ? 6 : 3;
		if (stretch * repeats > length) {
			break;
		}
		NSString *unit = [text substringFromIndex:length - stretch];
		BOOL repeated = YES;
		for (NSUInteger copy = 2; copy <= repeats && repeated; copy++) {
			NSRange range = NSMakeRange(length - stretch * copy, stretch);
			repeated = [[text substringWithRange:range] isEqualToString:unit];
		}
		if (repeated) {
			return YES;
		}
	}
	return NO;
}

+ (NSUInteger)distinctCharacterCountOf:(NSString *)text
{
	NSMutableSet<NSString *> *characters = [NSMutableSet set];
	[text enumerateSubstringsInRange:NSMakeRange(0, text.length) options:NSStringEnumerationByComposedCharacterSequences
						  usingBlock:^(NSString *character, NSRange range, NSRange enclosing, BOOL *stop) {
		[characters addObject:character];
	}];
	return characters.count;
}

@end
