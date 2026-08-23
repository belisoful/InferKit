//
//  NFKMLXDevice.swift
//  InferKitMLX
//
//  Objective-C access to the device MLX computes on. mlx-swift models this as a `Device` struct and a
//  scoped `withDefaultDevice(_:_:)`, neither of which bridges to Objective-C, so an Objective-C consumer
//  had no way to select the CPU. A Swift caller keeps using `MLX.Device` directly.
//

import Foundation
import MLX

/// The processor MLX runs an operation on.
///
/// The GPU is the default and is what every shipped model is tuned for. The CPU is for the cases where
/// a graphics device is contended or unavailable to the process: a background task competing with a
/// render loop, or a host that denies GPU access.
@objc(NFKMLXDeviceType)
public enum NFKMLXDeviceType: Int {

    // Swift's generated Objective-C names would be `…Cpu` / `…Gpu`; both are initialisms.
    /// MLX evaluates on the CPU backend.
    @objc(NFKMLXDeviceTypeCPU) case cpu = 0

    /// MLX evaluates on the GPU through Metal.
    @objc(NFKMLXDeviceTypeGPU) case gpu = 1
}

/// Objective-C access to MLX's compute-device selection.
///
/// Selecting the CPU does not remove the Metal library requirement. MLX initializes its scheduler at the
/// first stream request, and on Apple platforms that constructs the Metal device, so a process without
/// `default.metallib` fails before the first operation whichever device it names. See
/// `Docs/installation.md`.
@objc(NFKMLXDevice)
public final class NFKMLXDevice: NSObject {

    /// The device MLX uses for work started on the calling thread now.
    @objc public static var currentType: NFKMLXDeviceType {
        MLX.Device.defaultDevice().deviceType == .cpu ? .cpu : .gpu
    }

    /// Runs `block` with MLX directed at `type`, restoring the previous device when it returns.
    ///
    /// @discussion The selection reaches the work `block` performs on the calling thread, which makes
    /// this the wrapper for a synchronous `runInferenceForRequest:`. It does NOT reach another thread:
    /// mlx-swift scopes the device to a task-local value, so work dispatched asynchronously — including
    /// the background queue `submitInferenceJobForRequest:` runs on — takes the global default instead.
    /// A caller who wants a whole inference on the CPU runs `runInferenceForRequest:` inside this block
    /// on their own background thread, which is where the contract puts a multi-second inference anyway.
    ///
    /// Objective-C: `[NFKMLXDevice performOnDeviceType:NFKMLXDeviceTypeCPU block:^{ ... }];`
    @objc(performOnDeviceType:block:)
    public static func perform(on type: NFKMLXDeviceType, block: () -> Void) {
        MLX.Device.withDefaultDevice(MLX.Device(type == .cpu ? .cpu : .gpu), block)
    }
}
