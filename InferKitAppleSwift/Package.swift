// swift-tools-version:5.9
import PackageDescription

// InferKitAppleSwift is a companion package for the Apple inference APIs that ship only in Swift.
// SpeechAnalyzer is an actor, and Vision's RecognizeDocumentsRequest and DetectLensSmudgeRequest live
// in Vision.swiftmodule with no VN* header, so the core, a pure Objective-C target, cannot reach any
// of them. This package wraps each as an @objc NFKInferenceBackend, which is what puts them back in
// front of an Objective-C app.
//
// The floor is macOS 26 / iOS 26, which is where all three APIs begin. The core's floor is unchanged.
let package = Package(
    name: "InferKitAppleSwift",
    platforms: [
        .macOS("26.0"),
        .iOS("26.0"),
    ],
    products: [
        .library(name: "InferKitAppleSwift", targets: ["InferKitAppleSwift"]),
    ],
    dependencies: [
        .package(path: ".."),
        // Dev-only build plugin for `swift package generate-documentation`; not linked into the library.
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "InferKitAppleSwift",
            dependencies: [.product(name: "InferKit", package: "InferKit")],
            linkerSettings: [
                .linkedFramework("Speech"),
                .linkedFramework("Vision"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Translation"),
            ]),
        .testTarget(
            name: "InferKitAppleSwiftTests",
            dependencies: ["InferKitAppleSwift"]),
        .testTarget(
            name: "InferKitAppleSwiftExamples",
            dependencies: ["InferKitAppleSwift"],
            path: "Examples"),
        .testTarget(
            name: "InferKitAppleSwiftObjCExamples",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
                "InferKitAppleSwift",
            ],
            path: "ObjCExamples"),
    ]
)
