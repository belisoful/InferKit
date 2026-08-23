// swift-tools-version:5.9
import PackageDescription

// InferKitMLX is a companion package: it adds MLX-backed inference on top of the InferKit core
// without raising the core's platform floor or adding dependencies to it. MLX needs Apple Silicon
// and a newer OS, so this package is opt-in. Every model here is implemented in MLXNN against its
// reference, so the only dependency is mlx-swift itself.
let package = Package(
    name: "InferKitMLX",
    platforms: [
        .macOS(.v14),
        // mlx-swift's own floor.
        .iOS(.v17),
    ],
    products: [
        .library(name: "InferKitMLX", targets: ["InferKitMLX"]),
    ],
    dependencies: [
        .package(path: ".."),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        // Dev-only build plugin for `swift package generate-documentation`; not linked into the library.
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "InferKitMLX",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "InferKitMLXTests",
            dependencies: ["InferKitMLX"]
        ),
        // Compiled + run by CI so the MLX snippets in Docs/examples.md cannot silently drift.
        .testTarget(
            name: "InferKitMLXExamples",
            dependencies: ["InferKitMLX"],
            path: "Examples"
        ),
        // Objective-C: proves an ObjC consumer builds and drives an MLX model through the registry.
        .testTarget(
            name: "InferKitMLXObjCExamples",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
                "InferKitMLX",
            ],
            path: "ObjCExamples"
        ),
    ]
)
