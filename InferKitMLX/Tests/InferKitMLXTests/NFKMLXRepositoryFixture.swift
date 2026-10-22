//
//  NFKMLXRepositoryFixture.swift
//  InferKitMLXTests
//
//  Repositories a resolver judges, placed in a hub cache so no request leaves the process: files written
//  at one commit, a release directory from the validation store linked in, safetensors files of given
//  headers, and a hub served from memory for the resolver's own requests.
//

import CryptoKit
import XCTest
import InferKit
@testable import InferKitMLX

/// Answers a hub's requests from files served in memory, as the hub at `http://hub.test` would.
final class NFKMLXStubHub: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var responses = [String: (data: Data, headers: [String: String])]()

    static let endpointURL = URL(string: "http://hub.test")!

    static var session: URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [NFKMLXStubHub.self]
        return URLSession(configuration: configuration)
    }

    static func reset() {
        lock.withLock { responses = [:] }
    }

    /// Serves `files` at `commit`, its tree listing, and its revision information naming `commit`.
    static func serve(_ repository: String, commit: String, files: [String: Data], card: [String: Any]) {
        var listing = [[String: Any]]()
        lock.withLock {
            for (path, data) in files.sorted(by: { $0.key < $1.key }) {
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                listing.append(["type": "file", "path": path, "size": data.count, "oid": String(sha.prefix(40))])
                responses["/\(repository)/resolve/\(commit)/\(path)"] = (data, ["ETag": "\"\(sha)\""])
            }
            let tree = try! JSONSerialization.data(withJSONObject: listing)
            responses["/api/models/\(repository)/tree/\(commit)"] = (tree, [:])
            let information = try! JSONSerialization.data(withJSONObject: [
                "id": repository, "sha": commit, "gated": false, "cardData": card] as [String: Any])
            responses["/api/models/\(repository)/revision/main"] = (information, [:])
        }
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == endpointURL.host }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let served = Self.lock.withLock { Self.responses[url.path] }
        let response = HTTPURLResponse(url: url, statusCode: served == nil ? 404 : 200, httpVersion: "HTTP/1.1",
                                       headerFields: served?.headers ?? [:])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: served?.data ?? Data())
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// A hub cache in a temporary directory, holding repositories at ``commit`` as a resolver reads them.
struct NFKMLXRepositoryFixture {
    static let commit = "0123456789abcdef0123456789abcdef01234567"
    static let elementBytes = ["BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I8": 1, "I32": 4, "I64": 8,
                               "F8_E4M3": 1, "F8_E5M2": 1]

    let cache = FileManager.default.temporaryDirectory
        .appendingPathComponent("NFKMLXRepositoryFixture-\(UUID().uuidString)")

    func remove() {
        NFKMLXStubHub.reset()
        try? FileManager.default.removeItem(at: cache)
    }

    /// `files` written into the cache at ``commit``, and the repository a resolver reads.
    func repository(_ name: String, files: [String: Data], configuration: [String: Any],
                    license: NFKModelLicense?) throws -> NFKModelRepository {
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        var listed = [NFKDownloadPlanFile]()
        for (path, data) in files.sorted(by: { $0.key < $1.key }) {
            let local = try XCTUnwrap(hub.localURL(forRepo: name, revision: Self.commit, path: path))
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try data.write(to: local)
            let remote = try XCTUnwrap(hub.remoteURL(forRepo: name, revision: Self.commit, path: path))
            listed.append(NFKDownloadPlanFile(path: path, remoteURL: remote, localURL: local,
                                              byteCount: Int64(data.count), sha256: nil, cached: true, optional: false))
        }
        return NFKModelRepository(repository: name, revision: Self.commit, requestedRevision: nil,
                                  configuration: configuration, files: listed, license: license, hub: hub)
    }

    /// The release directory a validation key names, linked into the cache at ``commit``, and the repository
    /// a resolver reads. Skips where the key names no directory.
    func storedRelease(_ key: String, repository name: String) throws -> NFKModelRepository {
        var isDirectory: ObjCBool = false
        guard let path = NFKMLXValidationConfig.environment[key],
              FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("set \(key) to the release directory")
        }
        let directory = URL(fileURLWithPath: path)
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        var listed = [NFKDownloadPlanFile]()
        // Only the release's own files: `hf download --local-dir` leaves a `.cache` directory beside them.
        for file in try FileManager.default.contentsOfDirectory(atPath: path).sorted() where file != "REVISION" {
            let source = directory.appendingPathComponent(file)
            guard try source.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let local = try XCTUnwrap(hub.localURL(forRepo: name, revision: Self.commit, path: file))
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: local, withDestinationURL: source)
            let size = try XCTUnwrap(try source.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            let remote = try XCTUnwrap(hub.remoteURL(forRepo: name, revision: Self.commit, path: file))
            listed.append(NFKDownloadPlanFile(path: file, remoteURL: remote, localURL: local, byteCount: Int64(size),
                                              sha256: nil, cached: true, optional: false))
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return NFKModelRepository(repository: name, revision: Self.commit, requestedRevision: nil,
                                  configuration: configuration, files: listed, license: nil, hub: hub)
    }

    /// A safetensors file of `tensors`, each in its own dtype, zero-filled.
    static func safetensors(typed tensors: [(String, String, [Int])], metadata: [String: String]? = nil) -> Data {
        var header = [String: Any]()
        var offset = 0
        for (name, dtype, shape) in tensors {
            let bytes = shape.reduce(1, *) * (elementBytes[dtype] ?? 0)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        header["__metadata__"] = metadata
        let json = try! JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)
        data.append(Data(count: offset))
        return data
    }

    /// A safetensors file of `tensors` in `dtype`, zero-filled.
    static func safetensors(_ tensors: [(String, [Int])], dtype: String = "BF16", elementBytes: Int = 2,
                            metadata: [String: String]? = nil) -> Data {
        var header = [String: Any]()
        var offset = 0
        for (name, shape) in tensors {
            let bytes = shape.reduce(1, *) * elementBytes
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        header["__metadata__"] = metadata
        let json = try! JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)
        data.append(Data(count: offset))
        return data
    }

    /// Qwen2's pre-tokenization pattern, which Qwen3 releases state.
    static let qwen2Pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// A byte-level BPE `tokenizer.json` splitting by `pattern`, with an empty vocabulary.
    static func byteLevelTokenizer(pattern: String = qwen2Pattern) -> [String: Any] {
        let split: [String: Any] = ["type": "Split", "pattern": ["Regex": pattern], "behavior": "Isolated", "invert": false]
        let byteLevel: [String: Any] = ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
                                        "use_regex": false]
        return ["model": ["type": "BPE", "vocab": [String: Int](), "merges": [String]()],
                "pre_tokenizer": ["type": "Sequence", "pretokenizers": [split, byteLevel]], "decoder": ["type": "ByteLevel"]]
    }

    static func json(_ object: Any) -> Data {
        try! JSONSerialization.data(withJSONObject: object)
    }

    /// The shipped descriptor `identifier` names.
    static func template(_ identifier: String) throws -> NFKModelDescriptor {
        try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: identifier))
    }

    /// Asserts the judgment of `repository` on `identifier`'s loader refuses with `code`.
    static func assertRefusal(_ code: NFKModelResolutionError.Code, _ repository: NFKModelRepository,
                              template identifier: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let matching = try template(identifier)
        XCTAssertThrowsError(try NFKMLXModelDescriptors.descriptor(for: repository, matching: matching),
                             file: file, line: line) { error in
            let error = error as NSError
            XCTAssertEqual(error.domain, NFKModelResolutionErrorDomain, file: file, line: line)
            XCTAssertEqual(error.code, code.rawValue, "\(error.localizedDescription)", file: file, line: line)
        }
    }
}
