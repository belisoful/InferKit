//
//  main.swift
//  inferkit-background-probe
//
//  Records whether Metal and MLX run in the launchd context the probe is started in: a LaunchAgent in
//  a logged-in session, a LaunchAgent at the login window, and a LaunchDaemon with and without a user
//  logged in. Each run appends two JSON lines to the output file: the session and Metal as found, then
//  MLX's arithmetic and a decoder's forward pass. A run that dies inside MLX leaves only the first.
//  Tools/background-probe/README.md is the run sheet.
//

import Foundation
import Metal
import Security
import SystemConfiguration
import InferKit
import InferKitMLX
import MLX

let arguments = CommandLine.arguments
func value(after flag: String) -> String? {
    arguments.firstIndex(of: flag).flatMap { $0 + 1 < arguments.count ? arguments[$0 + 1] : nil }
}
let context = value(after: "--context") ?? "unlabeled"
let output = URL(fileURLWithPath: value(after: "--output") ?? "/Users/Shared/inferkit-probe/results.jsonl")
let modelDirectory = value(after: "--model").map { URL(fileURLWithPath: $0) }

func append(_ record: [String: Any]) {
    var line = record
    line["context"] = context
    line["date"] = ISO8601DateFormatter().string(from: Date())
    line["pid"] = Int(getpid())
    guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
    let folder = output.deletingLastPathComponent()
    try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
                                             attributes: [.posixPermissions: 0o1777])
    if !FileManager.default.fileExists(atPath: output.path) {
        FileManager.default.createFile(atPath: output.path, contents: nil, attributes: [.posixPermissions: 0o666])
    }
    guard let handle = try? FileHandle(forWritingTo: output) else { return }
    defer { try? handle.close() }
    _ = try? handle.seekToEnd()
    try? handle.write(contentsOf: data + Data("\n".utf8))
}

// The session the probe runs in, and the user at the console (nil at the login window).
var session: [String: Any] = ["phase": "session", "uid": Int(getuid()), "euid": Int(geteuid()), "user": NSUserName()]
var sessionID = SecuritySessionId()
var sessionAttributes = SessionAttributeBits()
if SessionGetInfo(callerSecuritySession, &sessionID, &sessionAttributes) == errSecSuccess {
    session["security_session"] = Int(sessionID)
    session["session_has_graphic_access"] = sessionAttributes.contains(.sessionHasGraphicAccess)
    session["session_is_root"] = sessionAttributes.contains(.sessionIsRoot)
    session["session_has_tty"] = sessionAttributes.contains(.sessionHasTTY)
}
var consoleUID: uid_t = 0
session["console_user"] = (SCDynamicStoreCopyConsoleUser(nil, &consoleUID, nil) as String?) ?? NSNull()
let device = MTLCreateSystemDefaultDevice()
session["metal_default_device"] = device?.name ?? NSNull()
session["metal_devices"] = MTLCopyAllDevices().map(\.name)
session["metal_recommended_working_set_bytes"] = device.map { Int($0.recommendedMaxWorkingSetSize) } ?? NSNull()
session["mlx_metal_library"] = NFKMLXGPU.metalLibraryURL?.path ?? NSNull()
append(session)

guard device != nil, NFKMLXGPU.metalLibraryURL != nil else {
    append(["phase": "mlx", "skipped": device == nil ? "no Metal device" : "no MLX Metal library beside the probe"])
    exit(1)
}

// MLX's arithmetic, then a decoder's forward pass on initial values, then a release when one is given.
var compute: [String: Any] = ["phase": "mlx", "device_architecture": NFKMLXGPU.deviceArchitecture,
                              "recommended_working_set_bytes": NFKMLXGPU.recommendedWorkingSetSize]
let started = Date()
MLXRandom.seed(7)
let matrix = MLXRandom.normal([1024, 1024])
let product = matmul(matrix, matrix.transposed())
eval(product)
compute["matmul_seconds"] = Date().timeIntervalSince(started)
compute["matmul_trace"] = product.diagonal().sum().item(Float.self)
let net = NFKMLXLanguage.makeNet(.tiny)
let logits = net(MLXArray([1, 2, 3].map { Int32($0) }).reshaped([1, 3]))
eval(logits)
compute["decoder_logits_shape"] = logits.shape
if let modelDirectory {
    do {
        let loadStarted = Date()
        let backend = try NFKMLXLanguage.backend(directoryURL: modelDirectory)
        let result = try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "The capital of France is"],
            parameters: [NFKParameterMaxTokens: 8, NFKParameterTemperature: 0]))
        compute["release_seconds"] = Date().timeIntervalSince(loadStarted)
        compute["release_text"] = result.outputs[NFKOutputText] as? String ?? NSNull()
    } catch {
        compute["release_error"] = "\(error)"
    }
}
compute["active_memory_bytes"] = NFKMLXGPU.activeMemory
append(compute)
