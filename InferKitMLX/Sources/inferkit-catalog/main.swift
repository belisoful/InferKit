//
//  main.swift
//  inferkit-catalog
//
//  Writes the model catalog of the providers linked into this tool, the core's Apple engines and
//  InferKitMLX's models, as a manifest, and lists the variant sources Tools/catalog/measure.py measures.
//  The tool builds descriptors and evaluates no MLX array, so it runs without MLX's Metal library.
//
//      swift run inferkit-catalog export [--output FILE]
//      swift run inferkit-catalog sources [--output FILE]
//

import Foundation
import InferKit
import InferKitMLX

let usage = """
    usage: inferkit-catalog export [--output FILE]
           inferkit-catalog sources [--output FILE]

    export   the shared catalog's manifest, the bytes Docs/model-catalog.json holds
    sources  the Hugging Face and URL sources of every variant, with their files, as JSON
    """

/// A source's fields as JSON, with the file fields measure.py reads.
func sourceEntry(_ descriptor: NFKModelDescriptor, _ variant: NFKModelVariant, _ source: NFKModelSource) -> [String: Any] {
    var entry: [String: Any] = [
        "model": descriptor.identifier,
        "variant": variant.identifier,
        "kind": source.kind,
        "layout": source.layout,
        "requiresConversion": source.requiresConversion,
        "gated": descriptor.license(for: variant)?.isGated ?? false,
    ]
    entry["repository"] = source.repository
    entry["revision"] = source.revision
    entry["url"] = source.url?.absoluteString
    entry["publiclyServed"] = source.extensions["publiclyServed"] as? Bool ?? true
    entry["note"] = source.extensions["note"] as? String
    entry["files"] = source.files.map { file -> [String: Any] in
        var fields: [String: Any] = ["path": file.path, "role": file.role, "optional": file.isOptional]
        fields["repository"] = file.repository
        fields["revision"] = file.revision
        fields["url"] = file.url?.absoluteString
        return fields
    }
    return entry
}

/// Every variant source that downloads from Hugging Face or a URL, in the catalog's model order.
func sourcesData(_ catalog: NFKModelCatalog) throws -> Data {
    let measurable: Set<String> = [NFKModelSourceKindHuggingFace, NFKModelSourceKindURL]
    var sources = [[String: Any]]()
    for descriptor in catalog.descriptors {
        for variant in descriptor.variants {
            guard let source = variant.source, measurable.contains(source.kind) else { continue }
            sources.append(sourceEntry(descriptor, variant, source))
        }
    }
    let document: [String: Any] = ["kind": "inferkit.catalog-sources", "sources": sources]
    return try JSONSerialization.data(withJSONObject: document,
                                      options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
}

func write(_ data: Data, to path: String?) throws {
    let terminated = data + Data("\n".utf8)
    guard let path else {
        FileHandle.standardOutput.write(terminated)
        return
    }
    try terminated.write(to: URL(fileURLWithPath: path), options: .atomic)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

var arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else { fail(usage) }
arguments.removeFirst()
var output: String?
if arguments.first == "--output" {
    guard arguments.count == 2 else { fail(usage) }
    output = arguments[1]
} else if !arguments.isEmpty {
    fail(usage)
}

// The snapshot describes every machine, so it lists none of this machine's voices.
NFKAppleModelDescriptors.includesMachineChoices = false
NFKMLXReferenceModels.registerAll()
let catalog = NFKModelCatalog.shared
do {
    switch command {
    case "export":
        try write(try catalog.manifestData(), to: output)
    case "sources":
        try write(try sourcesData(catalog), to: output)
    default:
        fail(usage)
    }
} catch {
    fail("inferkit-catalog: \(error.localizedDescription)")
}
