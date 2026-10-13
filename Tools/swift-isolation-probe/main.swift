//
//  main.swift
//  swift-isolation-probe
//
//  Calls each InferKit block that runs off the caller's queue from closures written in main-actor
//  code, compiled with -swift-version 6 -default-isolation MainActor (an app service's flags). A
//  block that Swift infers main-actor isolated traps when InferKit calls it on its own queue, so a
//  run that reaches "all calls returned" shows every probed block imports as Sendable.
//

import Foundation
import InferKit

struct ProbeFailure: Error {}

nonisolated func makeBackend(_ name: String) throws -> any NFKInferenceBackend {
    guard name.hasPrefix("probe") else {
        throw ProbeFailure()
    }
    let backend = NFKPassthroughBackend()
    backend.outputMap = [NFKOutputText: NFKInputPrompt]
    return backend
}

final class Builder: NSObject, NFKServedBackendBuilder {
    func backend(forModelName modelName: String) throws -> any NFKInferenceBackend {
        try makeBackend(modelName)
    }
}

func waitFor(_ label: String, _ start: (@escaping @Sendable () -> Void) -> Void) {
    let done = DispatchSemaphore(value: 0)
    start { done.signal() }
    if done.wait(timeout: .now() + 10) == .timedOut {
        print("timed out: \(label)")
        exit(1)
    }
    print("returned: \(label)")
}

let server = NFKInferenceServer()
server.residencyCheckInterval = 0
server.addBackendProvider({ error in
    do {
        return try makeBackend("probe-provider")
    } catch let failure as NSError {
        error?.pointee = failure
        return nil
    }
}, forModelName: "probe-provider")
server.addBackendBuilder(Builder(), forModelName: "probe-builder")
server.unloadsUnderMemoryPressure = true
server.memoryPressureProvider = { 0.1 }

for name in ["probe-provider", "probe-builder"] {
    waitFor("loadModelNamed \(name)") { finish in
        server.loadModelNamed(name) { error in
            precondition(error == nil, "build failed: \(String(describing: error))")
            finish()
        }
    }
}
nonisolated(unsafe) let sharedServer = server
waitFor("memoryPressureProvider") { finish in
    DispatchQueue.global().async {
        _ = sharedServer.memoryPressure
        finish()
    }
}

let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: [:], outputModality: .text)
waitFor("NFKInferenceJob completionHandler") { finish in
    let job = NFKInferenceSubmit(NFKPassthroughBackend(), request, nil)
    job.progressHandler = { _ in }
    job.completionHandler = { _ in finish() }
}

let tool = NFKInferenceTool(name: "echo", description: "Echoes its text.",
                            parameters: ["type": "object", "properties": ["text": ["type": "string"]]]) { arguments, _ in
    NFKToolResult(text: arguments["text"] as? String ?? "")
}
nonisolated(unsafe) let sharedTool = tool
waitFor("NFKToolHandler") { finish in
    DispatchQueue.global().async {
        _ = sharedTool.invoke(withArguments: ["text": "hi"])
        finish()
    }
}

print("all calls returned")
