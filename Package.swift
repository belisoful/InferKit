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
        // The process that hosts one plugin bundle out of the app's process. macOS only; on other
        // platforms its main returns at once.
        .executable(name: "inferkit-plugin-host", targets: ["inferkit-plugin-host"]),
    ],
    targets: [
        // The pure-C headers: the API version, the plugin ABI a C, C++, or Rust plugin implements, and
        // the C API a program calls InferKit through. A second directory beside include/InferKit is
        // rejected by SwiftPM, so they are a target of their own. NFKPluginABILayout.c holds the
        // layout assertions and is the target's one source.
        .target(
            name: "InferKitC",
            path: "Sources/InferKitC",
            publicHeadersPath: "include"
        ),
        .target(
            name: "InferKit",
            dependencies: ["InferKitC"],
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
                .linkedFramework("Vision"),
                .linkedFramework("VideoToolbox"),
                .linkedFramework("Speech"),
                .linkedFramework("SoundAnalysis"),
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("ImageIO"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Metal"),
                .linkedFramework("IOSurface"),
                .linkedFramework("Network"),
                .linkedFramework("Security"),
                .linkedFramework("IOKit", .when(platforms: [.macOS])),
            ]
        ),
        // SwiftPM cannot sign with entitlements, so the entitlement files are excluded and an app signs
        // the host it embeds (Docs/agent-reference/plugins.md).
        .executableTarget(
            name: "inferkit-plugin-host",
            dependencies: ["InferKit"],
            path: "Sources/inferkit-plugin-host",
            exclude: ["Entitlements"]
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
