// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "InferKit",
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
        .tvOS(.v14),
    ],
    products: [
        .library(name: "InferKit", targets: ["InferKit"]),
    ],
    targets: [
        .target(
            name: "InferKit",
            path: "Sources/InferKit",
            // Public API is include/InferKit/, so `#import <InferKit/Foo.h>` resolves the same
            // way it does against the built framework.
            publicHeadersPath: "include",
            cSettings: [
                // Quoted imports in the sources resolve through these, which SwiftPM does not
                // build header maps for.
                .headerSearchPath("include/InferKit"),
                .headerSearchPath("."),
            ],
            linkerSettings: [
                .linkedFramework("CoreML"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Metal"),
                .linkedFramework("IOSurface"),
            ]
        ),
        .testTarget(
            name: "InferKitTests",
            dependencies: ["InferKit"],
            path: "Tests/InferKitTests"
        ),
        // Compiled + run by CI so the code in Docs/examples.md cannot silently drift. The two targets
        // are the same examples in each language: Objective-C is the package's own idiom, and Swift
        // pins what the API looks like after the ObjC importer renames it.
        .testTarget(
            name: "InferKitExamples",
            dependencies: ["InferKit"],
            path: "Examples"
        ),
        .testTarget(
            name: "InferKitSwiftExamples",
            dependencies: ["InferKit"],
            path: "SwiftExamples"
        ),
    ],
    cLanguageStandard: .gnu17
)
