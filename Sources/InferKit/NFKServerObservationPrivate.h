//
//  NFKServerObservationPrivate.h
//  InferKit
//

#import <InferKit/NFKServerObservation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NFKServerRunEvent ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite) NSString *modelName;
@property (nonatomic, copy, readwrite) NSString *route;
@property (nonatomic, copy, readwrite) NSString *clientAddress;
@property (nonatomic, copy, readwrite, nullable) NSString *keyName;
@property (nonatomic, assign, readwrite) NFKServerRunState state;
@property (nonatomic, assign, readwrite) NSTimeInterval waitSeconds;
@property (nonatomic, assign, readwrite) NSTimeInterval durationSeconds;
@property (nonatomic, copy, readwrite, nullable) NSError *error;
@property (nonatomic, copy, readwrite) NSDate *date;
- (instancetype)initPrivately;
@end

@interface NFKServerModelEvent ()
@property (nonatomic, copy, readwrite) NSString *modelName;
@property (nonatomic, assign, readwrite) NFKServerModelEventKind kind;
@property (nonatomic, assign, readwrite) NFKServerModelEventReason reason;
@property (nonatomic, assign, readwrite) NSTimeInterval loadSeconds;
@property (nonatomic, copy, readwrite, nullable) NSError *error;
@property (nonatomic, copy, readwrite) NSDate *date;
- (instancetype)initPrivately;
@end

@interface NFKServerClientEvent ()
@property (nonatomic, copy, readwrite) NSString *clientAddress;
@property (nonatomic, assign, readwrite) BOOL fromLoopback;
@property (nonatomic, copy, readwrite, nullable) NSString *keyName;
@property (nonatomic, assign, readwrite) NFKServerKeyScope scope;
@property (nonatomic, copy, readwrite) NSString *path;
@property (nonatomic, copy, readwrite) NSDate *date;
- (instancetype)initPrivately;
@end

@interface NFKServerMCPSession ()
@property (nonatomic, copy, readwrite) NSString *identifier;
@property (nonatomic, copy, readwrite, nullable) NSString *clientName;
@property (nonatomic, copy, readwrite, nullable) NSString *clientVersion;
@property (nonatomic, copy, readwrite) NSString *protocolVersion;
@property (nonatomic, copy, readwrite) NSString *clientAddress;
@property (nonatomic, copy, readwrite, nullable) NSString *keyName;
@property (nonatomic, copy, readwrite) NSDate *startDate;
@property (nonatomic, copy, readwrite) NSDate *lastActivityDate;
@property (nonatomic, assign, readwrite, getter=isStreamOpen) BOOL streamOpen;
@property (nonatomic, assign, readwrite) NFKServerMCPSessionEndReason endReason;
@property (nonatomic, copy, readwrite, nullable) NSDate *endDate;
- (instancetype)initPrivately;
@end

NS_ASSUME_NONNULL_END
