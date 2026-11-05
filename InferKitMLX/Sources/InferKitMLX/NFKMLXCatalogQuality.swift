//
//  NFKMLXCatalogQuality.swift
//  InferKitMLX
//
//  The published benchmark scores of the catalog's models, each read from the source it names.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The published benchmark scores of the catalog's models, on each task's reference benchmark
/// (`+[NFKModelCatalog referenceBenchmarkForTask:]`).
///
/// @discussion Each record was read from the page its `source` names, on the day `retrieved` states. A
/// record scores the release its source measured: the entry's default variant unless it names another. A
/// model whose reference score is not published has no record.
enum NFKMLXCatalogQuality {

    /// One published score of one catalog entry.
    struct Record {
        var model: String
        var task: String
        var benchmark: String
        var snapshot: String
        var metric: String
        var value: Double
        var higherIsBetter = true
        var variant: String? = nil
        var source: String
        var retrieved: String
        var note: String? = nil

        var score: NFKModelQualityScore {
            NFKModelQualityScore(task: task, benchmark: benchmark, snapshot: snapshot, metric: metric, value: value,
                                 higherIsBetter: higherIsBetter, variantIdentifier: variant,
                                 sourceURL: URL(string: source)!, retrievedDate: retrieved, note: note,
                                 extensions: nil)
        }
    }

    /// The entry with its scores. An entry with a code-generation score lists that task after its own.
    static func scored(_ entry: NFKMLXCatalogEntry) -> NFKMLXCatalogEntry {
        let records = byModel[entry.identifier] ?? []
        guard !records.isEmpty else {
            return entry
        }
        var scored = entry
        scored.qualityScores = records.map(\.score)
        if records.contains(where: { $0.task == NFKModelTaskCodeGeneration }),
           !scored.tasks.contains(NFKModelTaskCodeGeneration) {
            scored.tasks.append(NFKModelTaskCodeGeneration)
        }
        return scored
    }

    static let byModel: [String: [Record]] = Dictionary(grouping: records, by: \.model)

    static let records: [Record] = mmluPro + liveCodeBench + openASR + flores + genEval + ttsArena

    private static let retrieved = "2026-11-05"

    private static func card(_ repository: String) -> String {
        "https://huggingface.co/\(repository)"
    }

    private static func arXiv(_ identifier: String) -> String {
        "https://arxiv.org/abs/\(identifier)"
    }

    // MARK: Text generation: MMLU-Pro

    private static func mmluPro(_ model: String, _ value: Double, _ source: String, _ note: String) -> Record {
        Record(model: model, task: NFKModelTaskTextGeneration, benchmark: NFKModelBenchmarkMMLUPro, snapshot: "v1",
               metric: "accuracy", value: value, source: source, retrieved: retrieved, note: note)
    }

    private static let gemma3Report = arXiv("2503.19786")
    private static let tigerLab = "https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro"

    static let mmluPro: [Record] = [
        mmluPro("gemma-3-1b-it", 14.7, gemma3Report, "Instruction-tuned, zero-shot; Gemma 3 report, Table 6."),
        mmluPro("gemma-3-4b-it", 43.6, gemma3Report, "Instruction-tuned, zero-shot; Gemma 3 report, Table 6."),
        mmluPro("gemma-3-12b-it", 60.6, gemma3Report, "Instruction-tuned, zero-shot; Gemma 3 report, Table 6."),
        mmluPro("gemma-3-27b-it", 67.5, gemma3Report, "Instruction-tuned, zero-shot; Gemma 3 report, Table 6."),
        mmluPro("gemma-3n-e2b-it", 40.5, card("google/gemma-3n-E2B-it"), "Instruction-tuned, zero-shot."),
        mmluPro("gemma-3n-e4b-it", 50.6, card("google/gemma-3n-E4B-it"), "Instruction-tuned, zero-shot."),
        mmluPro("gemma-4-e2b-it", 60.0, card("google/gemma-4-E2B-it"), "Instruction-tuned; the card does not state the thinking mode."),
        mmluPro("gemma-4-e4b-it", 69.4, card("google/gemma-4-E4B-it"), "Instruction-tuned; the card does not state the thinking mode."),
        mmluPro("gemma-4-12b-it", 77.2, card("google/gemma-4-12B-it"), "Instruction-tuned; the card does not state the thinking mode."),
        mmluPro("gemma-4-26b-a4b-it", 82.6, card("google/gemma-4-26B-A4B-it"),
                "Instruction-tuned; the card does not state the thinking mode."),
        mmluPro("gemma-4-31b-it", 85.2, card("google/gemma-4-31B-it"), "Instruction-tuned; the card does not state the thinking mode."),
        mmluPro("gpt-oss-20b", 73.14, tigerLab, "Medium reasoning effort, the release's default; measured by TIGER-Lab."),
        mmluPro("granite-4.0-h", 32.86, card("ibm-granite/granite-4.0-h-1b"), "5-shot, chain of thought."),
        mmluPro("granite-4.0-h-tiny", 44.94, card("ibm-granite/granite-4.0-h-tiny"), "5-shot, chain of thought."),
        mmluPro("mistral-small-24b-instruct-2501", 66.3, card("mistralai/Mistral-Small-24B-Instruct-2501"),
                "5-shot, chain of thought; published as 0.663."),
        mmluPro("mixtral-8x7b-instruct", 43.27, tigerLab, "Measured by TIGER-Lab."),
        mmluPro("qwen3-30b-a3b", 78.5, tigerLab, "Thinking mode, the release's default; reported by Qwen to TIGER-Lab."),
        mmluPro("qwen3.5-2b", 55.3, card("Qwen/Qwen3.5-2B"), "Non-thinking mode, the release's default."),
        mmluPro("qwen3.5", 79.1, card("Qwen/Qwen3.5-4B"), "The card does not label the mode; the release thinks by default."),
        mmluPro("qwen3.5-9b", 82.5, card("Qwen/Qwen3.5-9B"), "The card does not label the mode; the release thinks by default."),
    ]

    // MARK: Code generation: LiveCodeBench

    private static func liveCodeBench(_ model: String, _ value: Double, _ snapshot: String, _ source: String,
                                      _ note: String) -> Record {
        Record(model: model, task: NFKModelTaskCodeGeneration, benchmark: NFKModelBenchmarkLiveCodeBench,
               snapshot: snapshot, metric: "pass@1", value: value, source: source, retrieved: retrieved, note: note)
    }

    private static let qwen3Report = arXiv("2505.09388")

    static let liveCodeBench: [Record] = [
        liveCodeBench("gemma-3n-e2b-it", 18.6, "v5", card("google/gemma-3n-E2B-it"), "Zero-shot; the card states no problem window."),
        liveCodeBench("gemma-3n-e4b-it", 25.7, "v5", card("google/gemma-3n-E4B-it"), "Zero-shot; the card states no problem window."),
        liveCodeBench("gemma-4-e2b-it", 44.0, "v6", card("google/gemma-4-E2B-it"), "The card does not state the thinking mode."),
        liveCodeBench("gemma-4-e4b-it", 52.0, "v6", card("google/gemma-4-E4B-it"), "The card does not state the thinking mode."),
        liveCodeBench("gemma-4-12b-it", 72.0, "v6", card("google/gemma-4-12B-it"), "The card does not state the thinking mode."),
        liveCodeBench("gemma-4-26b-a4b-it", 77.1, "v6", card("google/gemma-4-26B-A4B-it"),
                      "The card does not state the thinking mode."),
        liveCodeBench("gemma-4-31b-it", 80.0, "v6", card("google/gemma-4-31B-it"), "The card does not state the thinking mode."),
        liveCodeBench("nemotron-nano-2", 71.1, "v5-2407-2412", card("nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
                      "Reasoning on; the version and window are those of NVIDIA's reproduction guide the card links."),
        liveCodeBench("qwen3-0.6b", 12.3, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3-1.7b", 33.2, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3-4b", 54.2, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3-14b", 63.5, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3-30b-a3b", 62.6, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3-32b", 65.7, "v5-2410-2502", qwen3Report, "Thinking mode, the release's default; Qwen3 report."),
        liveCodeBench("qwen3.5", 55.8, "v6", card("Qwen/Qwen3.5-4B"), "The card does not label the mode; the release thinks by default."),
        liveCodeBench("qwen3.5-9b", 65.6, "v6", card("Qwen/Qwen3.5-9B"),
                      "The card does not label the mode; the release thinks by default."),
        liveCodeBench("qwen3.8-27b", 90.3, "v6", card("Qwen/Qwen3.8-27B"), "The card does not label the mode."),
        liveCodeBench("qwen4-exp", 91.9, "v6", card("Qwen/Qwen3.8-Flash-Next"), "The card does not label the mode."),
    ]

    // MARK: Transcription: the Open ASR Leaderboard

    private static func openASR(_ model: String, _ value: Double) -> Record {
        Record(model: model, task: NFKModelTaskTranscription, benchmark: NFKModelBenchmarkOpenASRLeaderboard,
               snapshot: "2026-11-02", metric: "average WER", value: value, higherIsBetter: false,
               source: "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard", retrieved: retrieved,
               note: "The mean of the leaderboard's nine default English test sets, which leave out LibriSpeech from 2026-11-02.")
    }

    static let openASR: [Record] = [
        openASR("canary-1b-v2", 8.35),
        openASR("parakeet-tdt", 6.91),
        openASR("voxtral-mini-3b", 8.29),
        openASR("whisper-large-v3", 8.18),
        openASR("whisper-large-v3-turbo", 8.68),
    ]

    // MARK: Translation: FLORES-200

    private static func flores(_ model: String, _ value: Double, _ row: String) -> Record {
        Record(model: model, task: NFKModelTaskTranslation, benchmark: NFKModelBenchmarkFLORES200, snapshot: "devtest-602",
               metric: "chrF++", value: value, source: arXiv("2207.04672"), retrieved: retrieved,
               note: "The average over English to 201 languages, 201 languages to English, and 200 sampled pairs "
                   + "without English; NLLB report, Table 41, row \"\(row)\".")
    }

    static let flores: [Record] = [
        flores("nllb-200", 44.6, "dense distilled 615M"),
        flores("nllb-200-distilled-1.3b", 46.9, "dense distilled 1.3B"),
        flores("nllb-200-1.3b", 46.4, "dense baseline 1.3B"),
    ]

    // MARK: Text to image: GenEval

    private static func genEval(_ model: String, _ value: Double, _ source: String, _ note: String) -> Record {
        Record(model: model, task: NFKModelTaskTextToImage, benchmark: NFKModelBenchmarkGenEval, snapshot: "v1",
               metric: "overall", value: value, source: source, retrieved: retrieved, note: note)
    }

    private static let genEvalPaper = arXiv("2310.11513")
    private static let sanaReport = arXiv("2410.10629")
    private static let scaleWiseDistillation = arXiv("2503.16397")

    static let genEval: [Record] = [
        genEval("flux.1-dev", 0.67, sanaReport, "1024 by 1024; SANA report, Tables 7 and 10."),
        genEval("flux.1-schnell", 0.71, sanaReport, "1024 by 1024; SANA report, Tables 7 and 10."),
        genEval("sana", 0.64, sanaReport, "1024 by 1024, 20 steps; SANA report, Tables 7 and 10."),
        genEval("sana-1600m", 0.66, sanaReport, "1024 by 1024, 20 steps; SANA report, Tables 7 and 10."),
        genEval("stable-diffusion-v1-5", 0.43, genEvalPaper, "Measured by GenEval's authors; GenEval paper, Table 2."),
        genEval("stable-diffusion-xl-base-1.0", 0.55, genEvalPaper,
                "The base model without the refiner, measured by GenEval's authors; GenEval paper, Table 2."),
        genEval("sdxl-turbo", 0.55, arXiv("2403.03206"), "Stable Diffusion 3 report, Table 5."),
        genEval("sd3", 0.69, scaleWiseDistillation, "40 steps, 1024 by 1024; scale-wise distillation paper, Table 3."),
        genEval("sd3.5-large", 0.70, scaleWiseDistillation, "28 steps, 1024 by 1024; scale-wise distillation paper, Table 3."),
        genEval("sd3.5-large-turbo", 0.70, scaleWiseDistillation,
                "4 steps, 1024 by 1024; scale-wise distillation paper, Table 3."),
        genEval("z-image", 0.82, arXiv("2511.22699"), "Z-Image report, Table 9."),
        genEval("hyper-sdxl-4steps-lora", 0.5398, arXiv("2506.00523"),
                "Hyper-SDXL's 4-step configuration at 1024 by 1024; SenseFlow paper, Table 4."),
    ]

    // MARK: Speech synthesis: TTS Arena

    static let ttsArena: [Record] = [
        Record(model: "kokoro-82m", task: NFKModelTaskSpeechSynthesis, benchmark: NFKModelBenchmarkTTSArena,
               snapshot: "v2-2026-11-05", metric: "Elo", value: 1465, source: "https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2",
               retrieved: retrieved, note: "Kokoro v1.0, 1033 votes; the arena refits every rating as votes arrive."),
    ]
}
