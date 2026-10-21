//
//  EverydayToolsPlugin.m
//  InferKit plugin examples
//
//  An Objective-C plugin that offers tools and no models. "convert_units" converts a measurement with
//  Foundation's units, and "calculate_date" adds days to a date or counts the days between two. The
//  bundle's manifest declares each tool's name, description, and JSON Schema; the plugin attaches a
//  handler to each declaration, so the two never disagree. The plugin option "decimalPlaces" rounds
//  converted values.
//
//  The bundle binds to the InferKit already loaded in the process that loads it, so no class here
//  carries the NFK prefix.
//

#import <InferKit/InferKit.h>

static NSString * const EverydayToolsErrorDomain = @"com.example.inferkit.everyday-tools";
static NSString * const EverydayToolsDecimalPlacesOption = @"decimalPlaces";
static const NSInteger EverydayToolsDefaultDecimalPlaces = 4;

static NSError *EverydayToolsError(NSString *message)
{
	return [NSError errorWithDomain:EverydayToolsErrorDomain code:1 userInfo:@{ NSLocalizedDescriptionKey: message }];
}

#pragma mark - Units

/*! The units convert_units reads, by the lowercase symbol a model writes. */
static NSDictionary<NSString *, NSUnit *> *EverydayToolsUnits(void)
{
	static NSDictionary<NSString *, NSUnit *> *units = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		units = @{
			@"mm": NSUnitLength.millimeters, @"cm": NSUnitLength.centimeters, @"m": NSUnitLength.meters,
			@"km": NSUnitLength.kilometers, @"in": NSUnitLength.inches, @"ft": NSUnitLength.feet,
			@"yd": NSUnitLength.yards, @"mi": NSUnitLength.miles,
			@"mg": NSUnitMass.milligrams, @"g": NSUnitMass.grams, @"kg": NSUnitMass.kilograms,
			@"oz": NSUnitMass.ounces, @"lb": NSUnitMass.poundsMass,
			@"ml": NSUnitVolume.milliliters, @"l": NSUnitVolume.liters, @"cup": NSUnitVolume.cups,
			@"qt": NSUnitVolume.quarts, @"gal": NSUnitVolume.gallons,
			@"c": NSUnitTemperature.celsius, @"f": NSUnitTemperature.fahrenheit, @"k": NSUnitTemperature.kelvin,
		};
	});
	return units;
}

static NSString *EverydayToolsFormatNumber(double value, NSInteger decimalPlaces)
{
	NSNumberFormatter *formatter = [[NSNumberFormatter alloc] init];
	formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
	formatter.numberStyle = NSNumberFormatterDecimalStyle;
	formatter.usesGroupingSeparator = NO;
	formatter.maximumFractionDigits = (NSUInteger)decimalPlaces;
	return [formatter stringFromNumber:@(value)] ?: [NSString stringWithFormat:@"%g", value];
}

static NFKToolResult *_Nullable EverydayToolsConvertUnits(NSDictionary<NSString *, id> *arguments, NSInteger decimalPlaces,
														  NSError **error)
{
	NSString *from = [arguments[@"from"] lowercaseString];
	NSString *to = [arguments[@"to"] lowercaseString];
	NSUnit *fromUnit = EverydayToolsUnits()[from];
	NSUnit *toUnit = EverydayToolsUnits()[to];
	if (fromUnit == nil || toUnit == nil) {
		if (error != NULL) {
			*error = EverydayToolsError([NSString stringWithFormat:@"\"%@\" is not a unit this tool knows", fromUnit == nil ? arguments[@"from"] : arguments[@"to"]]);
		}
		return nil;
	}
	if (![fromUnit isKindOfClass:toUnit.class]) {
		if (error != NULL) {
			*error = EverydayToolsError([NSString stringWithFormat:@"%@ and %@ measure different things", arguments[@"from"], arguments[@"to"]]);
		}
		return nil;
	}
	double value = [arguments[@"value"] doubleValue];
	NSMeasurement *measurement = [[NSMeasurement alloc] initWithDoubleValue:value unit:fromUnit];
	double converted = [measurement measurementByConvertingToUnit:toUnit].doubleValue;
	double scale = pow(10, (double)decimalPlaces);
	double rounded = round(converted * scale) / scale;
	NSString *text = [NSString stringWithFormat:@"%@ %@ is %@ %@", EverydayToolsFormatNumber(value, 12), arguments[@"from"],
					  EverydayToolsFormatNumber(rounded, decimalPlaces), arguments[@"to"]];
	NSDictionary *structured = @{ @"value": @(rounded), @"unit": arguments[@"to"] };
	return [NFKToolResult resultWithText:text outputs:@{ NFKOutputStructured: structured } imagePNGs:nil];
}

#pragma mark - Dates

static NSCalendar *EverydayToolsCalendar(void)
{
	NSCalendar *calendar = [NSCalendar calendarWithIdentifier:NSCalendarIdentifierGregorian];
	calendar.timeZone = [NSTimeZone timeZoneForSecondsFromGMT:0];
	return calendar;
}

static NSDateFormatter *EverydayToolsDateFormatter(NSString *format)
{
	NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
	formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
	formatter.calendar = EverydayToolsCalendar();
	formatter.timeZone = formatter.calendar.timeZone;
	formatter.dateFormat = format;
	return formatter;
}

static NSDate *_Nullable EverydayToolsParseDate(id text, NSError **error)
{
	NSDate *date = [text isKindOfClass:NSString.class] ? [EverydayToolsDateFormatter(@"yyyy-MM-dd") dateFromString:text] : nil;
	if (date == nil && error != NULL) {
		*error = EverydayToolsError([NSString stringWithFormat:@"\"%@\" is not a date written YYYY-MM-DD", text ?: @""]);
	}
	return date;
}

static NFKToolResult *_Nullable EverydayToolsAddDays(NSDate *date, id days, NSError **error)
{
	if (![days isKindOfClass:NSNumber.class]) {
		if (error != NULL) {
			*error = EverydayToolsError(@"add-days needs an integer \"days\"");
		}
		return nil;
	}
	NSDate *result = [EverydayToolsCalendar() dateByAddingUnit:NSCalendarUnitDay value:[days integerValue] toDate:date options:0];
	NSString *day = [EverydayToolsDateFormatter(@"yyyy-MM-dd") stringFromDate:result];
	NSString *weekday = [EverydayToolsDateFormatter(@"EEEE") stringFromDate:result];
	NSDictionary *structured = @{ @"date": day, @"weekday": weekday };
	return [NFKToolResult resultWithText:[NSString stringWithFormat:@"%@, a %@", day, weekday]
								 outputs:@{ NFKOutputStructured: structured } imagePNGs:nil];
}

static NFKToolResult *_Nullable EverydayToolsDaysBetween(NSDate *date, id otherText, NSError **error)
{
	NSDate *other = EverydayToolsParseDate(otherText, error);
	if (other == nil) {
		return nil;
	}
	NSInteger days = [EverydayToolsCalendar() components:NSCalendarUnitDay fromDate:date toDate:other options:0].day;
	return [NFKToolResult resultWithText:[NSString stringWithFormat:@"%ld days", (long)days]
								 outputs:@{ NFKOutputStructured: @{ @"days": @(days) } } imagePNGs:nil];
}

static NFKToolResult *_Nullable EverydayToolsCalculateDate(NSDictionary<NSString *, id> *arguments, NSError **error)
{
	NSDate *date = EverydayToolsParseDate(arguments[@"date"], error);
	if (date == nil) {
		return nil;
	}
	NSString *operation = arguments[@"operation"];
	if ([operation isEqualToString:@"add-days"]) {
		return EverydayToolsAddDays(date, arguments[@"days"], error);
	}
	if ([operation isEqualToString:@"days-between"]) {
		return EverydayToolsDaysBetween(date, arguments[@"otherDate"], error);
	}
	if (error != NULL) {
		*error = EverydayToolsError([NSString stringWithFormat:@"\"%@\" is not add-days or days-between", operation]);
	}
	return nil;
}

#pragma mark - Plugin

@interface EverydayToolsPlugin : NSObject <NFKPlugin>
@property (nonatomic, strong) id<NFKPluginContext> context;
@property (nonatomic, strong, nullable) NFKPluginManifest *manifest;
@end

@implementation EverydayToolsPlugin

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
	return @[];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if (error != NULL) {
		NSString *message = [NSString stringWithFormat:@"the plugin has no model \"%@\"; it offers tools only", modelIdentifier];
		*error = [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel userInfo:@{ NSLocalizedDescriptionKey: message }];
	}
	return nil;
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

- (NSInteger)decimalPlaces
{
	id value = [self optionValueForKey:EverydayToolsDecimalPlacesOption];
	return [value isKindOfClass:NSNumber.class] ? [value integerValue] : EverydayToolsDefaultDecimalPlaces;
}

/*! The code behind each tool the manifest declares, by tool name. */
- (NSDictionary<NSString *, NFKToolHandler> *)handlers
{
	NSInteger decimalPlaces = self.decimalPlaces;
	return @{
		@"convert_units": ^NFKToolResult *(NSDictionary<NSString *, id> *arguments, NSError **error) {
			return EverydayToolsConvertUnits(arguments, decimalPlaces, error);
		},
		@"calculate_date": ^NFKToolResult *(NSDictionary<NSString *, id> *arguments, NSError **error) {
			return EverydayToolsCalculateDate(arguments, error);
		},
	};
}

- (NSArray<NFKInferenceTool *> *)tools
{
	NSDictionary<NSString *, NFKToolHandler> *handlers = self.handlers;
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	for (NSDictionary<NSString *, id> *declaration in self.manifest.tools) {
		NFKToolHandler handler = handlers[declaration[@"name"]];
		if (handler == nil) {
			[self.context logMessage:[NSString stringWithFormat:@"the manifest declares a tool \"%@\" the plugin does not run", declaration[@"name"]]
							   level:NFKPluginLogLevelError];
			continue;
		}
		[tools addObject:[NFKInferenceTool toolWithName:declaration[@"name"] description:declaration[@"description"]
											 parameters:declaration[@"parameters"] handler:handler]];
	}
	return tools;
}

@end
