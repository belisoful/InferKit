//
//  NFKMLXToolCallingLiveTests.swift
//  InferKitMLXTests
//
//  Released models calling a tool through NFKToolLoop, one per call syntax the shipped templates
//  write: Qwen3 (Hermes JSON), Qwen3.5 (XML parameters), and gpt-oss (harmony). Each test is gated on
//  its release's key in ~/.inferkit-validation.json and skips without it.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXToolCallingLiveTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func release(_ key: String) throws -> URL {
        guard let path = config[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key) to the release directory")
        }
        return URL(fileURLWithPath: path)
    }

    /// Runs one weather question through a loop over `model`, then checks the call and the answer.
    private func assertTheModelCallsTheWeatherTool(_ model: any NFKInferenceBackend, name: String,
                                                   parameters: [String: Any]) throws {
        var calls: [[String: Any]] = []
        let weather = NFKInferenceTool(name: "get_weather",
                                       description: "Current weather in a city. Use it when the user asks about the weather now. Returns the temperature and the sky.",
                                       parameters: ["type": "object",
                                                    "properties": ["city": ["type": "string", "description": "The city name."]],
                                                    "required": ["city"]],
                                       handler: { arguments, _ in
            calls.append(arguments)
            return NFKToolResult(text: "18 C and clear skies")
        })
        let loop = NFKToolLoop(backend: model, tools: [weather])
        loop.maximumTurns = 3
        let request = NFKInferenceRequest(inputs: [NFKInputMessages: [["role": "user", "content": "What is the weather in Paris right now?"]]],
                                          parameters: parameters, outputModality: .text)
        let started = Date()
        let result = try loop.runInference(for: request)
        let invocations = result.output(forKey: NFKOutputToolInvocations) as? [NFKToolInvocation] ?? []
        print(String(format: "VALIDATION tool loop %@: %d calls %@ in %.1f s; answer: %@", name, invocations.count,
                     calls.description, Date().timeIntervalSince(started), result.text ?? "<none>"))
        XCTAssertEqual(invocations.first?.toolName, "get_weather", "\(name) called the tool")
        XCTAssertTrue((calls.first?["city"] as? String)?.lowercased().contains("paris") ?? false, "\(name) asked about Paris: \(calls)")
        XCTAssertFalse(invocations.first?.result.isError ?? true, "\(name)'s arguments met the schema")
        XCTAssertTrue(result.text?.contains("18") ?? false, "\(name) answered from the tool's result: \(result.text ?? "")")
        XCTAssertNil(result.output(forKey: NFKOutputToolTurnLimitReached))
    }

    func testQwen3CallsAToolInHermesJSON() throws {
        let model = try NFKMLXLanguage.backend(directoryURL: try release("IK_VAL_QWEN3_1_7B"))
        try assertTheModelCallsTheWeatherTool(model, name: "qwen3-1.7B",
                                              parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 256,
                                                           NFKParameterReasoningEffort: NFKReasoningEffortLight])
    }

    func testQwen35CallsAToolInXMLParameters() throws {
        let model = try NFKMLXHybridLanguage.backend(directoryURL: try release("IK_VAL_QWEN3_5_2B"))
        try assertTheModelCallsTheWeatherTool(model, name: "qwen3.5-2B",
                                              parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 160])
    }

    func testGptOssCallsAToolInHarmony() throws {
        let model = try NFKMLXLanguage.backend(directoryURL: try release("IK_VAL_GPT_OSS"))
        try assertTheModelCallsTheWeatherTool(model, name: "gpt-oss-20b",
                                              parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 384,
                                                           NFKParameterReasoningEffort: NFKReasoningEffortLight])
    }
}
