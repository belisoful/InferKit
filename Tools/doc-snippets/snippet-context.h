//
//  snippet-context.h
//  Tools/doc-snippets
//
//  The variables the documented Objective-C snippets take from their surrounding prose, declared once
//  with the types the documents give them. check-objc.py places this text at the top of the function
//  body each snippet compiles in, so a snippet's own declaration of a name shadows the one here.
//  A name that means different types in different snippets stays out of this file; the snippet that
//  uses it declares it with an `objc-check: given` directive.
//

NSError *error = nil;
NSString *key = nil, *apiKey = nil, *userAdminKey = nil, *token = nil;
NSString *prompt = nil, *query = nil, *userText = nil;
NSArray<NSString *> *documents = nil, *shortlist = nil;

NSURL *url = nil, *endpoint = nil, *dir = nil, *folder = nil;
NSURL *releaseDirectory = nil, *rerankerDirectory = nil, *hugeRelease = nil, *qwen4B = nil, *qwen06B = nil;
NSURL *weightsURL = nil, *weights = nil, *localURL = nil, *tunedURL = nil, *checkpointURL = nil, *pthURL = nil;
NSURL *convertedURL = nil, *ggufURL = nil, *modelURL = nil, *compiledURL = nil, *unetURL = nil, *vaeURL = nil;
NSURL *detectorWeights = nil, *promptURL = nil;
NSURL *pdfURL = nil, *briefPDFURL = nil, *clipURL = nil, *recording = nil, *recordingURL = nil;
NSData *wavData = nil, *pdfData = nil, *appendixPDFData = nil, *microphonePCM = nil;
NSDate *monthStart = nil;

CGImageRef cgImage = NULL, image = NULL, mask = NULL, photo = NULL, photograph = NULL;
CGImageRef firstImage = NULL, secondImage = NULL, firstFrame = NULL, lastFrame = NULL;
id plate = nil, hint = nil, frame = nil, frameA = nil, frameB = nil, asset = nil;
NSArray *frames = nil;
id<MTLTexture> inputTexture = nil;
NFKAudioAsset *song = nil;
NSUInteger width = 0, height = 0;
float *interleavedRGBA = NULL;
MLMultiArray *outputArray = nil;

id<NFKInferenceBackend> backend = nil, remote = nil, llm = nil, claude = nil;
NFKInferenceRequest *request = nil;
NSArray<NSDictionary<NSString *, id> *> *messages = nil;
NSDictionary<NSString *, id> *record = nil;
NFKDecisionQuestion *intent = nil, *refund = nil;
NSArray<NFKDecisionQuestion *> *questions = nil;
NFKTokenVocabulary *vocabulary = nil;
NFKHFHub *hub = nil;
