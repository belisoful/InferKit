//
//  main.swift
//  inferkit-background-probe
//
//  Records whether Metal and MLX run in the launchd context the probe is started in: a LaunchAgent in
//  a logged-in session, a LaunchAgent at the login window, and a LaunchDaemon with and without a user
//  logged in. Each run appends two JSON lines to results.jsonl in the job's output directory: the
//  session and Metal as found, then MLX's arithmetic and a decoder's forward pass. A run that dies
//  inside MLX leaves only the first. Its stderr goes to <context>.stderr.log in the same directory.
//  Once results.jsonl holds --max-runs runs of the job's context, a run exits without work.
//  Tools/background-probe/README.md is the run sheet.
//
//      inferkit-background-probe --context <name> --output-directory <directory>
//                                [--max-runs <count, default 30>] [--model <release directory>]
//

#if os(macOS)
import Darwin
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
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("inferkit-background-probe: \(message)\n".utf8))
    exit(2)
}

let context = value(after: "--context") ?? "unlabeled"
guard !context.isEmpty, context.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }) else {
    fail("--context takes letters, digits, and hyphens")
}
guard let outputPath = value(after: "--output-directory") else {
    fail("--output-directory is required")
}
guard let maximumRuns = Int(value(after: "--max-runs") ?? "30"), maximumRuns >= 0 else {
    fail("--max-runs takes a count")
}
let modelDirectory = value(after: "--model").map { URL(fileURLWithPath: $0) }

// Root runs two of the jobs. Every file opens relative to the output directory's descriptor with
// O_NOFOLLOW, and the directory must belong to the probe's own user with no group or other write
// permission, so no other user can plant a link that redirects a write.
func openOutputDirectory(_ path: String) -> Int32 {
    let descriptor = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard descriptor >= 0 else {
        fail("\(path): \(String(cString: strerror(errno))). It must be a directory, not a symbolic link.")
    }
    var status = stat()
    guard fstat(descriptor, &status) == 0, status.st_uid == geteuid(),
          status.st_mode & (S_IWGRP | S_IWOTH) == 0 else {
        fail("\(path) must belong to uid \(geteuid()) and be writable by its owner only")
    }
    return descriptor
}

func openFile(_ name: String, in directory: Int32, flags: Int32) -> Int32? {
    let descriptor = openat(directory, name, flags | O_NOFOLLOW | O_CLOEXEC, 0o644)
    guard descriptor >= 0 else {
        return nil
    }
    var status = stat()
    guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
        close(descriptor)
        return nil
    }
    return descriptor
}

func runsRecorded(in directory: Int32) -> Int {
    guard let descriptor = openFile("results.jsonl", in: directory, flags: O_RDONLY) else {
        return 0
    }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    let contents = (try? handle.readToEnd()) ?? Data()
    return contents.split(separator: UInt8(ascii: "\n")).filter { line in
        guard let record = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any] else {
            return false
        }
        return record["phase"] as? String == "session" && record["context"] as? String == context
    }.count
}

let outputDirectory = openOutputDirectory(outputPath)
guard runsRecorded(in: outputDirectory) < maximumRuns else {
    exit(0)
}
guard let resultsDescriptor = openFile("results.jsonl", in: outputDirectory, flags: O_WRONLY | O_APPEND | O_CREAT),
      let logDescriptor = openFile("\(context).stderr.log", in: outputDirectory,
                                   flags: O_WRONLY | O_APPEND | O_CREAT) else {
    fail("cannot open results.jsonl and \(context).stderr.log as regular files in \(outputPath)")
}
dup2(logDescriptor, STDERR_FILENO)
let results = FileHandle(fileDescriptor: resultsDescriptor, closeOnDealloc: false)

func append(_ record: [String: Any]) {
    var line = record
    line["context"] = context
    line["date"] = ISO8601DateFormatter().string(from: Date())
    line["pid"] = Int(getpid())
    guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else {
        return
    }
    try? results.write(contentsOf: data + Data("\n".utf8))
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
#else
import Foundation

FileHandle.standardError.write(Data("inferkit-background-probe runs on macOS only\n".utf8))
exit(1)
#endif
