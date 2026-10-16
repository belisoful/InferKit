//
//  NFKServedRequestChecks.h
//  InferKit
//
//  What every served route checks of a request before it reads it: the body's media type, the field
//  types, the contract types of what reaches a backend, and the browser origin. Private to the core.
//

#import <Foundation/Foundation.h>
#import "NFKHTTPServer.h"
#import <InferKit/NFKInferenceRequest.h>

NS_ASSUME_NONNULL_BEGIN

/*! An NFKInferenceServerErrorInvalidRequest error with reason, which a route answers with 400. */
FOUNDATION_EXPORT NSError *NFKServedInvalidRequest(NSString *reason);

/*! Whether value is a JSON boolean, as NSJSONSerialization reads true and false. */
FOUNDATION_EXPORT BOOL NFKServedIsBoolean(id _Nullable value);

/*! Whether value is of type. A type is boolean, string, number (a finite number that is not a
	boolean), integer (a number without a fraction), object, array, or null, or several of them
	joined by |. */
FOUNDATION_EXPORT BOOL NFKServedValueIsOfType(id _Nullable value, NSString *type);

/*! The refusal of value, named path in the reason, when it is present and not of type; nil when it
	is absent, null, or of type. A null value is an absent one: clients send null for an optional field
	they leave unset. */
FOUNDATION_EXPORT NSError * _Nullable NFKServedValueTypeError(id _Nullable value, NSString *type, NSString *path);

/*! The refusal of the first field of object whose value is not of the type types gives its name, or
	nil. A field types does not name passes, and so does an absent or null one. path, when set, prefixes
	each field's name in the reason. An absent or null object passes; any other that is not a dictionary
	is refused under path. */
FOUNDATION_EXPORT NSError * _Nullable NFKServedFieldTypeError(id _Nullable object,
															  NSDictionary<NSString *, NSString *> *types,
															  NSString * _Nullable path);

/*! object without its null-valued entries, so a route reads a null field as an absent one. */
FOUNDATION_EXPORT NSDictionary *NFKServedWithoutNulls(NSDictionary *object);

/*! request without the inputs and parameters whose value is null and whose key the contract types
	(NFKModelValueTypeForKey), so a backend meets an absent value. A key outside the contract keeps its
	null, which the native wire carries. Returns request itself when it holds none. */
FOUNDATION_EXPORT NFKInferenceRequest *NFKServedRequestWithoutNullContractValues(NFKInferenceRequest *request);

/*! The refusal of the first input or parameter whose value is not of the kind NFKModelValueTypeForKey
	gives its key: text, a number, an integer, a boolean, a list of text, or messages (dictionaries).
	A key outside the contract passes, and so does a media key, whose value a route decodes itself. */
FOUNDATION_EXPORT NSError * _Nullable NFKServedContractTypeError(NSDictionary<NSString *, id> *inputs,
																 NSDictionary<NSString *, id> *parameters);

/*! A form field's text as the kind NFKModelValueTypeForKey gives name: a number, an integer, true or
	false, or a one-item list of text. A name outside those kinds keeps its text. nil with a refusal
	when the text does not read as the kind. */
FOUNDATION_EXPORT id _Nullable NFKServedFormValue(NSString *name, NSString *text, NSError * _Nullable * _Nullable error);

/*! Whether a JSON route reads a body under a Content-Type header: none at all, application/json, or a
	type whose subtype ends in +json, with any parameters. A browser's cross-origin post always carries
	an Origin, which NFKServedOriginIsAllowed refuses, so a missing type admits no page. */
FOUNDATION_EXPORT BOOL NFKServedContentTypeAcceptsJSON(NSString * _Nullable contentType);

/*! Whether a request's Origin header lets it through. A request without one passes, as a client that
	is not a browser sends none. A browser's origin passes when it is a loopback name (localhost,
	127.0.0.1, ::1) or the host the request was sent to; null and any other host are refused, which
	keeps a page on another site from reaching the server through DNS rebinding. */
FOUNDATION_EXPORT BOOL NFKServedOriginIsAllowed(NFKHTTPRequest *request);

/*! The error that reports an exception raised while serving a request:
	kNFKError_InferenceBackendFailure, naming the exception and its reason. */
FOUNDATION_EXPORT NSError *NFKServedExceptionError(NSException *exception);

NS_ASSUME_NONNULL_END
