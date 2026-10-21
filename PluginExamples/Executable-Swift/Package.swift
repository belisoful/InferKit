// swift-tools-version:5.9
import PackageDescription

// The program inside TextStatistics.inferkitplugin. It depends on the InferKit package two folders up by
// path; a plugin kept in its own repository names InferKit's URL and a version instead.
let package = Package(
    name: "TextStatistics",
    platforms: [
        .macOS(.v12),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "TextStatistics",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
            ],
            path: "Sources/TextStatistics"
        ),
    ]
)
