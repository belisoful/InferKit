// swift-tools-version:5.9
import PackageDescription

// A maintainer tool: records whether Metal and MLX run in a launchd context (a LaunchAgent at the
// login window, a LaunchDaemon with no user logged in). README.md beside this file is the run sheet.
// It depends on the InferKit and InferKitMLX packages in this repository by path, so neither package
// builds it.
let package = Package(
    name: "InferKitBackgroundProbe",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    dependencies: [
        .package(path: "../.."),
        .package(path: "../../InferKitMLX"),
        // The revision InferKitMLX pins. A different requirement fails resolution.
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: "901941965d82e4a216d4d117231d847d194c563d"),
    ],
    targets: [
        .executableTarget(
            name: "inferkit-background-probe",
            dependencies: [
                .product(name: "InferKit", package: "InferKit"),
                .product(name: "InferKitMLX", package: "InferKitMLX"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/inferkit-background-probe"
        ),
    ]
)
