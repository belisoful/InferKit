//
//  NFKMLXReleaseTokenizerTests.swift
//  InferKitMLXTests
//
//  The release tokenizer reader: each shipped family's tokenizer.json read to the scheme it states and
//  held to the `tokenizers` library's ids on text that separates the schemes (digit runs, contractions,
//  whitespace runs, decomposed accents, case changes, code, added tokens written literally), and the
//  forms this package does not reproduce refused by name.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReleaseTokenizerTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    // The ids are `Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False).ids` from
    // the `tokenizers` library (0.23.1), one row per case.
    static let cases: [String] = [
        "In 2024, 12345 apples cost $1,299.99.",
        "I'll DON'T we've 's 'RE",
        "  leading spaces\n\nand newlines\t tabs\r\n",
        "naïve café — résumé 日本語テキスト 🚀",
        "e\u{0301} decomposed a\u{0308}",
        "CamelCaseWord ALLCAPS MiXeD",
        "def f(x):\n    return x**2  # square\n",
        "'hello' (world) /path/to/file.txt",
        "<s>Hello</s>",
    ]

    static let releases: [(key: String, scheme: String, ids: [[Int]])] = [
        ("IK_VAL_QWEN3", "qwen2", [
            [641, 220, 17, 15, 17, 19, 11, 220, 16, 17, 18, 19, 20, 40676, 2783, 400, 16, 11, 17, 24, 24, 13, 24, 24, 13],
            [40, 3278, 44273, 17323, 582, 3003, 364, 82, 364, 787],
            [220, 6388, 12621, 271, 437, 501, 7969, 197, 22398, 319],
            [3376, 37572, 586, 51950, 1959, 9333, 1242, 963, 75402, 21894, 102819, 56833, 61803, 70534, 11162, 248, 222],
            [963, 28502, 3865, 12709],
            [25406, 301, 4207, 10879, 13097, 31400, 50, 20740, 55, 68, 35],
            [750, 282, 2075, 982, 262, 470, 856, 334, 17, 220, 671, 9334, 198],
            [6, 14990, 6, 320, 14615, 8, 608, 2343, 32429, 23903, 3909],
            [44047, 79497, 522, 82, 29],
        ]),
        ("IK_VAL_QWEN3_5", "qwen35", [
            [623, 220, 17, 15, 17, 19, 11, 220, 16, 17, 18, 19, 20, 39332, 2695, 393, 16, 11, 17, 24, 24, 13, 24, 24, 13],
            [40, 3172, 42803, 16813, 567, 2908, 359, 82, 359, 762],
            [220, 6187, 12258, 271, 429, 491, 7718, 197, 21700, 317],
            [3267, 36328, 571, 50203, 1892, 238976, 220, 247359, 210342, 10838, 248, 222],
            [933, 27572, 3736, 12344],
            [24601, 300, 4068, 10561, 12719, 30368, 50, 20112, 246024, 35],
            [727, 281, 2007, 1590, 198, 262, 460, 830, 332, 17, 220, 653, 9057, 198],
            [6, 14556, 6, 318, 14194, 8, 593, 2267, 31358, 23156, 3778],
            [42589, 76759, 510, 82, 29],
        ]),
        ("IK_VAL_GPT_OSS", "o200k", [
            [637, 220, 1323, 19, 11, 220, 7633, 2548, 57814, 3097, 548, 16, 11, 25431, 13, 2058, 13],
            [67504, 153384, 24716, 461, 82, 461, 1099],
            [220, 8117, 18608, 279, 427, 620, 10105, 197, 38191, 370],
            [1503, 9954, 737, 30469, 2733, 140184, 17428, 40909, 16056, 18368, 38236, 169883, 222],
            [68, 13430, 57661, 25082, 261, 47565],
            [137910, 6187, 12929, 19465, 56928, 50, 13236, 148218, 35],
            [1314, 285, 4061, 1883, 271, 622, 1215, 410, 17, 220, 1069, 13749, 198],
            [6, 24912, 6, 350, 24169, 8, 820, 4189, 72231, 51766, 7186],
            [101950, 192249, 808, 82, 29],
        ]),
        ("IK_VAL_MISTRAL_SMALL_CUT4", "llama3", [
            [1785, 1032, 1050, 1048, 1050, 1052, 1044, 1032, 1049, 1050, 1051, 1052, 1053, 90767, 4362, 1659, 1049, 1044, 1050, 1057, 1057, 1046, 1057, 1057, 1046],
            [1073, 7534, 119801, 1039, 1084, 1729, 6483, 1576, 1115, 1576, 3609],
            [1032, 8924, 18971, 1267, 1421, 1875, 21067, 1009, 49324, 1013, 1010],
            [2302, 7884, 1672, 35858, 2251, 7677, 35088, 30367, 15199, 7282, 14742, 13854, 119685, 1154, 1128],
            [1101, 1204, 1129, 123909, 1261, 129636],
            [38487, 1299, 11139, 24469, 14646, 13515, 6292, 1083, 13064, 1088, 1101, 1068],
            [3149, 1284, 4790, 3640, 1293, 1850, 2460, 1438, 1050, 1032, 2569, 12333, 1010],
            [1039, 29706, 1039, 1319, 34049, 1041, 1987, 7368, 35466, 55998, 13477],
            [1, 22177, 2],
        ]),
        ("IK_VAL_PIXTRAL", "llama3", [
            [1785, 1032, 1050, 1048, 1050, 1052, 1044, 1032, 1049, 1050, 1051, 1052, 1053, 90767, 4362, 1659, 1049, 1044, 1050, 1057, 1057, 1046, 1057, 1057, 1046],
            [1073, 7534, 119801, 1039, 1084, 1729, 6483, 1576, 1115, 1576, 3609],
            [1032, 8924, 18971, 1267, 1421, 1875, 21067, 1009, 49324, 1013, 1010],
            [2302, 7884, 1672, 35858, 2251, 7677, 35088, 30367, 15199, 7282, 14742, 13854, 119685, 1154, 1128],
            [1101, 1204, 1129, 123909, 1261, 129636],
            [38487, 1299, 11139, 24469, 14646, 13515, 6292, 1083, 13064, 1088, 1101, 1068],
            [3149, 1284, 4790, 3640, 1293, 1850, 2460, 1438, 1050, 1032, 2569, 12333, 1010],
            [1039, 29706, 1039, 1319, 34049, 1041, 1987, 7368, 35466, 55998, 13477],
            [1, 22177, 2],
        ]),
        ("IK_VAL_GRANITE", "llama3", [
            [644, 220, 2366, 19, 11, 220, 4513, 1774, 41776, 2853, 400, 16, 11, 15531, 13, 1484, 13],
            [40, 3358, 45373, 17773, 584, 3077, 364, 82, 364, 793],
            [220, 6522, 12908, 271, 438, 502, 8128, 197, 23204, 319],
            [3458, 38672, 588, 53050, 2001, 9517, 1264, 978, 76502, 22656, 45918, 252, 57933, 62903, 71634, 11410, 248, 222],
            [68, 54939, 29602, 3950, 264, 136, 230],
            [26479, 301, 4301, 11116, 13398, 32500, 50, 21402, 55, 68, 35],
            [755, 282, 2120, 997, 262, 471, 865, 334, 17, 220, 674, 9518, 198],
            [6, 15339, 6, 320, 14957, 8, 611, 2398, 33529, 24849, 3996],
            [45147, 80597, 524, 82, 29],
        ]),
        ("IK_TOK_NEMOTRON_H", "nemotron", [
            [1785, 1032, 1050, 1048, 1050, 1052, 1044, 1032, 1049, 1050, 1051, 1052, 1053, 90767, 4362, 1659, 1049, 1044, 1050, 1057, 1057, 1046, 1057, 1057, 1046],
            [1073, 7534, 119801, 1039, 1084, 1729, 6483, 1576, 1115, 1576, 3609],
            [1032, 8924, 18971, 1267, 1421, 1875, 21067, 1009, 49324, 1013, 1010],
            [2302, 7884, 1672, 35858, 2251, 7677, 35088, 30367, 15199, 7282, 14742, 13854, 119685, 1154, 1128],
            [1101, 1204, 1129, 123909, 1261, 129636],
            [38487, 1299, 11139, 24469, 14646, 13515, 6292, 1083, 13064, 1088, 1101, 1068],
            [3149, 1284, 4790, 3640, 1293, 1850, 2460, 1438, 1050, 1032, 2569, 12333, 1010],
            [1039, 29706, 1039, 1319, 34049, 1041, 1987, 7368, 35466, 55998, 13477],
            [1, 22177, 2],
        ]),
        ("IK_VAL_GRANITE_SPEECH", "gpt2", [
            [383, 225, 36, 34, 36, 38, 30, 225, 35, 36, 37, 38, 39, 1123, 1882, 7847, 398, 35, 30, 36, 43, 43, 32, 43, 43, 32],
            [59, 4621, 475, 567, 25, 70, 996, 4763, 330, 101, 330, 613],
            [225, 15996, 13922, 203, 203, 363, 537, 4446, 202, 17002, 437],
            [3258, 43738, 587, 281, 1549, 1309, 12011, 12677, 1036, 1309, 225, 37404, 32236, 10371, 14216, 12875, 46896, 227],
            [87, 10044, 409, 44872, 312, 32632],
            [32804, 2732, 5095, 11865, 44080, 45266, 40397, 54],
            [589, 296, 26, 106, 711, 284, 442, 816, 326, 36, 225, 588, 14627, 203],
            [25, 7656, 25, 308, 5860, 27, 517, 983, 33, 452, 33, 761, 32, 2760],
            [46, 101, 48, 8279, 377, 101, 48],
        ]),
        ("IK_VAL_PHI4MM", "o200k", [
            [637, 220, 1323, 19, 11, 220, 7633, 2548, 57814, 3097, 548, 16, 11, 25431, 13, 2058, 13],
            [67504, 153384, 24716, 461, 82, 461, 1099],
            [220, 8117, 18608, 279, 427, 620, 10105, 197, 38191, 370],
            [1503, 9954, 737, 30469, 2733, 140184, 17428, 40909, 16056, 18368, 38236, 169883, 222],
            [68, 13430, 57661, 25082, 261, 47565],
            [137910, 6187, 12929, 19465, 56928, 50, 13236, 148218, 35],
            [1314, 285, 4061, 1883, 271, 622, 1215, 410, 17, 220, 1069, 13749, 198],
            [6, 24912, 6, 350, 24169, 8, 820, 4189, 72231, 51766, 7186],
            [101950, 192249, 808, 82, 29],
        ]),
        ("IK_VAL_MIXTRAL_CUT2", "metaspace", [
            [560, 28705, 28750, 28734, 28750, 28781, 28725, 28705, 28740, 28750, 28770, 28781, 28782, 979, 2815, 2434, 429, 28740, 28725, 28750, 28774, 28774, 28723, 28774, 28774, 28723],
            [315, 28742, 584, 384, 832, 28742, 28738, 478, 28742, 333, 464, 28713, 464, 896],
            [28705, 5374, 10599, 13, 13, 391, 633, 8063, 12, 261, 4737, 28801, 13],
            [1879, 28920, 333, 28345, 1040, 11310, 383, 28797, 28705, 29142, 29119, 30321, 29610, 29753, 29109, 29123, 28705, 30012],
            [317, 28949, 18626, 4481, 264, 30814],
            [5487, 301, 6160, 15170, 17565, 13706, 28735, 13609, 28814, 28706, 28757],
            [801, 285, 28732, 28744, 1329, 13, 2287, 604, 1318, 348, 28750, 28705, 422, 7930, 13],
            [464, 21558, 28742, 325, 9471, 28731, 732, 1740, 28748, 532, 28748, 1516, 28723, 7088],
            [1, 16230, 2],
        ]),
        ("IK_VAL_MLX_LLAMA32_1B_4BIT", "llama3", [
            [644, 220, 2366, 19, 11, 220, 4513, 1774, 41776, 2853, 400, 16, 11, 15531, 13, 1484, 13],
            [40, 3358, 45373, 17773, 584, 3077, 364, 82, 364, 793],
            [220, 6522, 12908, 271, 438, 502, 8128, 197, 23204, 319],
            [3458, 38672, 588, 53050, 2001, 9517, 1264, 978, 105180, 102158, 57933, 62903, 71634, 11410, 248, 222],
            [68, 54939, 29602, 3950, 264, 127104],
            [26479, 301, 4301, 11116, 13398, 32500, 50, 21402, 55, 68, 35],
            [755, 282, 2120, 997, 262, 471, 865, 334, 17, 220, 674, 9518, 198],
            [6, 15339, 6, 320, 14957, 8, 611, 2398, 33529, 24849, 3996],
            [45147, 80597, 524, 82, 29],
        ]),
    ]

    func testEveryShippedFamilyEncodesAsTokenizersDoes() throws {
        var checked = [String]()
        for release in Self.releases {
            guard let path = config[release.key] else { continue }
            let directory = URL(fileURLWithPath: path)
            let tokenizerJSON = try XCTUnwrap(JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [String: Any])
            let scheme = try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: tokenizerJSON, tokenizerConfiguration: nil)
            switch scheme {
            case .byteLevel(let pretokenization, _, _):
                XCTAssertEqual(pretokenization, release.scheme, release.key)
            case .metaspace:
                XCTAssertEqual("metaspace", release.scheme, release.key)
            }
            let tokenizer = try NFKMLXLanguage.readTokenizer(inDirectory: directory)
            for (text, expected) in zip(Self.cases, release.ids) {
                XCTAssertEqual(tokenizer.encode(text).map(\.intValue), expected,
                               "\(release.key): \(text.debugDescription)")
            }
            checked.append(release.key)
        }
        try XCTSkipIf(checked.isEmpty, "set any of " + Self.releases.map(\.key).joined(separator: ", "))
        print("VALIDATION structure release-tokenizers: \(checked.count) of \(Self.releases.count) families "
              + "encode \(Self.cases.count) cases as tokenizers does")
    }

    // MARK: Refusals

    private func byteLevel(split regex: String? = NFKMLXLanguage.pretokenizationPatterns.first { $0.value == "qwen2" }?.key,
                           normalizer: Any? = nil, model: [String: Any] = [:]) -> [String: Any] {
        var json: [String: Any] = [
            "model": ["type": "BPE", "vocab": [String: Int](), "merges": [String]()].merging(model) { $1 },
            "decoder": ["type": "ByteLevel"],
            "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Split", "pattern": ["Regex": regex ?? ""], "behavior": "Isolated", "invert": false],
                ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false],
            ]],
        ]
        if let normalizer { json["normalizer"] = normalizer }
        return json
    }

    func testTheSchemeIsReadStrictly() throws {
        XCTAssertEqual(try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: byteLevel(normalizer: ["type": "NFC"]),
                                                          tokenizerConfiguration: nil),
                       .byteLevel(pretokenization: "qwen2", composes: true, ignoresMerges: false))
        XCTAssertEqual(try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: byteLevel(model: ["ignore_merges": true]),
                                                          tokenizerConfiguration: nil),
                       .byteLevel(pretokenization: "qwen2", composes: false, ignoresMerges: true))
        let gpt2: [String: Any] = ["model": ["type": "BPE"], "decoder": ["type": "ByteLevel"],
                                   "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false, "use_regex": true]]
        XCTAssertEqual(try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: gpt2, tokenizerConfiguration: nil),
                       .byteLevel(pretokenization: "gpt2", composes: false, ignoresMerges: false))
        XCTAssertEqual(try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: nil,
                                                          tokenizerConfiguration: ["tokenizer_class": "Qwen2Tokenizer"]),
                       .byteLevel(pretokenization: "qwen2", composes: true, ignoresMerges: false),
                       "Qwen2's tokenizer class fixes its pattern and NFC")

        let refused: [(String, [String: Any]?, [String: Any]?)] = [
            ("an unknown Split pattern", byteLevel(split: #"\p{N}{1,2}|\p{L}+"#), nil),
            ("a normalizer other than NFC", byteLevel(normalizer: ["type": "NFKC"]), nil),
            ("byte fallback under a byte-level pre-tokenizer", byteLevel(model: ["byte_fallback": true]), nil),
            ("BPE dropout", byteLevel(model: ["dropout": 0.1]), nil),
            ("a subword affix", byteLevel(model: ["continuing_subword_prefix": "##"]), nil),
            ("a unigram model", ["model": ["type": "Unigram"]], nil),
            ("a ByteLevel that adds a prefix space",
             ["model": ["type": "BPE"], "decoder": ["type": "ByteLevel"],
              "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": true, "use_regex": true]], nil),
            ("Digits before the GPT-2 regex",
             ["model": ["type": "BPE"], "decoder": ["type": "ByteLevel"],
              "pre_tokenizer": ["type": "Sequence", "pretokenizers": [
                ["type": "Digits", "individual_digits": true],
                ["type": "ByteLevel", "add_prefix_space": false, "use_regex": true]]]], nil),
            ("a Metaspace that splits on spaces",
             ["model": ["type": "BPE", "byte_fallback": true], "pre_tokenizer": [
                "type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true]], nil),
            ("no tokenizer.json and no fixing class", nil, ["tokenizer_class": "LlamaTokenizer"]),
        ]
        for (reason, tokenizerJSON, configuration) in refused {
            XCTAssertThrowsError(try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: tokenizerJSON,
                                                                    tokenizerConfiguration: configuration), reason)
        }
    }
}
