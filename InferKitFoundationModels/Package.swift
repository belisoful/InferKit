// swift-tools-version:5.9
import PackageDescription

// InferKitFoundationModels is a companion package: it bridges InferKit and Apple's Foundation
// Models framework without raising the core's platform floor. The framework needs Apple
// Intelligence hardware and macOS 26 / iOS 26, so this package is opt-in.
//
// Direction one ships here: NFKFoundationModelsBackend wraps the on-device system language model
// (LanguageModelSession) as an NFKInferenceBackend, so an InferKit consumer swaps it in like any
// other engine. Direction two — adopting the Foundation Models provider protocols (LanguageModel /
// LanguageModelExecutor, WWDC26) so InferKit backends stand behind Apple's session API — needs the
// macOS 27 / iOS 27 SDK and follows when that SDK is the build baseline.
let package = Package(
    name: "InferKitFoundationModels",
    platforms: [
        .macOS("26.0"),
        .iOS("26.0"),
    ],
    products: [
        .library(name: "InferKitFoundationModels", targets: ["InferKitFoundationModels"]),
    ],
    dependencies: [
        .package(path: ".."),
        // Dev-only build plugin for `swift package generate-documentation`; not linked into the library.
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "InferKitFoundationModels",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
            ]
        ),
        .testTarget(
            name: "InferKitFoundationModelsTests",
            dependencies: ["InferKitFoundationModels"]
        ),
        // Compiled + run by CI so the Foundation Models snippets in Docs/examples.md cannot drift.
        .testTarget(
            name: "InferKitFoundationModelsExamples",
            dependencies: ["InferKitFoundationModels"],
            path: "Examples"
        ),
        // Objective-C: the backend, its tools, and its typed parameters are all `@objc`, so an ObjC
        // consumer drives Apple's on-device model without writing Swift.
        .testTarget(
            name: "InferKitFoundationModelsObjCExamples",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
                "InferKitFoundationModels",
            ],
            path: "ObjCExamples"
        ),
    ]
)
