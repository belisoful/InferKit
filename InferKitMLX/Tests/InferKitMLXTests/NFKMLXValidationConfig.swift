//
//  NFKMLXValidationConfig.swift
//  InferKitMLX tests
//

import Foundation

/// The shared validation configuration a model's parity / released-weight test reads its
/// `IK_VAL_*` / `IK_PARITY_*` paths from: the process environment overlaid with
/// `~/.inferkit-validation.json` (the JSON wins over the environment).
///
/// Wiring a model's keys into that JSON is therefore enough to make the full check exercise the
/// model by default — no per-run environment is needed. A test that reads
/// `ProcessInfo.processInfo.environment` directly is invisible to the JSON, so a new model's test
/// reads `NFKMLXValidationConfig.environment` instead. See "Completing an InferKitMLX model to
/// parity" in `Docs/agent-reference/mlx-parity-checklist.md`.
enum NFKMLXValidationConfig {
	static var environment: [String: String] {
		var merged = ProcessInfo.processInfo.environment
		let url = FileManager.default.homeDirectoryForCurrentUser
			.appendingPathComponent(".inferkit-validation.json")
		if let data = try? Data(contentsOf: url),
		   let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
			json.forEach { merged[$0.key] = $0.value }
		}
		return merged
	}
}
