#!/usr/bin/env python3
"""Measure the model catalog's sources without downloading their weights.

Run from the repository root with the standard library only:

    python3 -I Tools/catalog/measure.py [--sources FILE] [--output FILE] [--jobs N]

The sources are the variants `swift run inferkit-catalog sources` lists (or the JSON file `--sources`
names). For a Hugging Face source the script reads the repository's file tree
(`/api/models/{repo}/tree/{revision}?recursive=1`), which gives each file's size, the SHA-256 Git LFS
records, and, through `/api/models/{repo}/revision/{revision}`, the commit the revision resolves to.
It reads each safetensors file's header by two HTTP range requests (`shapes.header`) to sum the
tensor bytes by dtype and count the elements, and from the tensor names and shapes it takes, per
top-level directory, the repeated layers and the largest of them, the routed-expert bytes, and the
elements a runtime weight quantization packs. It reads the repository's `config.json` for the
geometry the key-value cache follows. A URL source gets a HEAD request for its Content-Length. A Google Drive folder or a release page serves no size and is recorded as
unmeasurable. `HF_TOKEN` is sent when set; a repository that refuses the listing without it is
recorded as gated.

The download a variant makes is the one the package's download code makes: every required file,
each optional file the repository serves, the first weight candidate a directory serves, and the
shards a `*.index.json` names. Only small JSON files (shard indexes and `config.json`) are fetched
whole; no weight file is.

The result is InferKitMLX/Sources/InferKitMLX/NFKMLXCatalogMeasurements.swift, a generated table the
catalog builder reads. A summary goes to standard error.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "Tools" / "validation-assets"))
import shapes  # noqa: E402  (orders IPv4 first and reads safetensors headers)

OUTPUT = ROOT / "InferKitMLX" / "Sources" / "InferKitMLX" / "NFKMLXCatalogMeasurements.swift"
HUB = "https://huggingface.co"
UNMEASURABLE_HOSTS = ("drive.google.com", "docs.google.com")
INDEX_SUFFIX = ".index.json"


class Gated(Exception):
    """The repository refused the request without an accepted access token."""


class Missing(Exception):
    """The server answered 404."""


def _token_header(request):
    token = os.environ.get("HF_TOKEN")
    if token and urllib.parse.urlparse(request.full_url).hostname.endswith("huggingface.co"):
        request.add_header("Authorization", f"Bearer {token}")


def _with_backoff(call, what):
    """Runs `call`, waiting out a 429 or a 5xx, and maps 401/403 to Gated and 404 to Missing."""
    delay = 5
    for attempt in range(8):
        try:
            return call()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise Gated(f"{what}: HTTP {error.code}") from error
            if error.code == 404:
                raise Missing(f"{what}: HTTP 404") from error
            if error.code == 429 or error.code >= 500:
                wait = int(error.headers.get("Retry-After") or delay)
                time.sleep(min(wait, 120))
                delay = min(delay * 2, 120)
                continue
            raise
        except SystemExit as error:  # shapes._request gives up on a connection this way
            if attempt >= 2:
                raise RuntimeError(f"{what}: {error}") from None
            time.sleep(delay)
    raise RuntimeError(f"{what}: rate limited")


def _get(url, accept_json=True):
    request = urllib.request.Request(url)
    _token_header(request)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        return (json.loads(body) if accept_json else body), response.headers


class _HeadRedirects(urllib.request.HTTPRedirectHandler):
    """Keeps a redirected HEAD a HEAD, so following a release URL to its CDN transfers no body."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.method = req.get_method()
        return redirected


_head_opener = urllib.request.build_opener(_HeadRedirects)


def head_length(url):
    def call():
        request = urllib.request.Request(url, method="HEAD")
        _token_header(request)
        with _head_opener.open(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    return _with_backoff(call, url)


# MARK: - Hugging Face

_lock = threading.Lock()
_trees = {}
_commits = {}
_indexes = {}
_headers = {}


def _quoted(revision):
    return urllib.parse.quote(revision or "main", safe="")


def tree(repo, revision):
    """path -> (bytes, sha256 or None) for every file of the repository at `revision`."""
    key = (repo, revision)
    with _lock:
        if key in _trees:
            return _trees[key]
    files = {}
    url = f"{HUB}/api/models/{repo}/tree/{_quoted(revision)}?recursive=1"
    while url:
        entries, headers = _with_backoff(lambda u=url: _get(u), f"{repo} tree")
        for entry in entries:
            if entry.get("type") == "file":
                lfs = entry.get("lfs") or {}
                files[entry["path"]] = (int(entry.get("size") or lfs.get("size") or 0), lfs.get("oid"))
        url = None
        for part in (headers.get("Link") or "").split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    with _lock:
        _trees[key] = files
    return files


def commit(repo, revision):
    key = (repo, revision)
    with _lock:
        if key in _commits:
            return _commits[key]
    info, _ = _with_backoff(lambda: _get(f"{HUB}/api/models/{repo}/revision/{_quoted(revision)}"), f"{repo} revision")
    with _lock:
        _commits[key] = info.get("sha")
    return info.get("sha")


def resolve_url(repo, revision, path):
    return f"{HUB}/{repo}/resolve/{_quoted(revision)}/{urllib.parse.quote(path)}"


def shard_paths(repo, revision, index_path):
    """The shards a shard index names, relative to the repository root.

    A gated repository lists its files but refuses the index itself without an accepted token; its
    shards are then the files beside the index named `<stem>-NNNNN-of-NNNNN.safetensors`.
    """
    key = (repo, revision, index_path)
    with _lock:
        if key in _indexes:
            return _indexes[key]
    folder = os.path.dirname(index_path)
    try:
        body = _with_backoff(lambda: shapes._request(resolve_url(repo, revision, index_path)), index_path)
        weight_map = json.loads(body).get("weight_map") or {}
        shards = sorted({f"{folder}/{shard}" if folder else shard for shard in weight_map.values()})
    except Gated:
        stem = os.path.basename(index_path)[: -len(".safetensors" + INDEX_SUFFIX)]
        pattern = re.compile(re.escape(stem) + r"-\d{5}-of-\d{5}\.safetensors")
        shards = sorted(path for path in tree(repo, revision)
                        if os.path.dirname(path) == folder and pattern.fullmatch(os.path.basename(path)))
    with _lock:
        _indexes[key] = shards
    return shards


def tensor_stats(url):
    """dtype -> stored bytes, the element count, and (name, dtype, shape, bytes) per tensor, of the
    safetensors file at `url`."""
    with _lock:
        if url in _headers:
            return _headers[url]
    header = _with_backoff(lambda: shapes.header(url), url)
    by_dtype, elements, tensors = {}, 0, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + end - start
        count = 1
        for dimension in entry["shape"]:
            count *= dimension
        elements += count
        tensors.append((name, entry["dtype"], tuple(entry["shape"]), end - start))
    result = (by_dtype, elements, tensors)
    with _lock:
        _headers[url] = result
    return result


# MARK: - Tensor structure

# The first repeated-layer container in a tensor name: `model.layers.12.`, `transformer_blocks.3.`.
LAYER_PATTERN = re.compile(r"^((?:.*?\.)?(?:layers|h|blocks|transformer_blocks))\.(\d+)\.")
QUANTIZATION_GROUP = 64


def is_quantizable(name, shape):
    """A rank-2 `.weight` whose input width divides the quantization group, outside the embeddings and
    the norms."""
    lowered = name.lower()
    return (len(shape) == 2 and name.endswith(".weight") and shape[-1] % QUANTIZATION_GROUP == 0
            and "embed" not in lowered and "norm" not in lowered)


class Structure:
    """What one top-level directory's tensors hold beyond their bytes: the repeated layers of its
    largest layer container, its routed experts, and its quantizable elements."""

    def __init__(self):
        self.containers = {}  # container -> index -> dtype -> bytes
        self.expert_bytes = {}
        self.quantizable = 0

    def add(self, name, dtype, shape, nbytes):
        match = LAYER_PATTERN.match(name)
        if match:
            layer = self.containers.setdefault(match.group(1), {}).setdefault(int(match.group(2)), {})
            layer[dtype] = layer.get(dtype, 0) + nbytes
        if ".experts." in name:
            self.expert_bytes[dtype] = self.expert_bytes.get(dtype, 0) + nbytes
        if is_quantizable(name, shape):
            count = 1
            for dimension in shape:
                count *= dimension
            self.quantizable += count

    def layers(self):
        """(count, the largest layer's bytes by dtype) of the container holding the most bytes, or None."""
        if not self.containers:
            return None
        def total(layers):
            return sum(sum(dtypes.values()) for dtypes in layers.values())
        _, layers = max(self.containers.items(), key=lambda item: (total(item[1]), item[0]))
        largest = max(layers.values(), key=lambda dtypes: (sum(dtypes.values()), sorted(dtypes.items())))
        return len(layers), largest


# MARK: - Configuration geometry

# The fields the key-value cache follows, read from the root of config.json and, where a release nests
# its decoder, from text_config, which wins.
GEOMETRY_FIELDS = {
    "num_hidden_layers": "layerCount",
    "num_attention_heads": "attentionHeads",
    "num_key_value_heads": "keyValueHeads",
    "head_dim": "headDimension",
    "hidden_size": "hiddenSize",
    "max_position_embeddings": "maximumPositions",
    "sliding_window": "slidingWindow",
    "sliding_window_pattern": "slidingWindowPattern",
    "num_kv_shared_layers": "sharedKeyValueLayers",
    "global_head_dim": "globalHeadDimension",
    "num_global_key_value_heads": "globalKeyValueHeads",
    "vocab_size": "vocabularySize",
}
NESTED_CONFIGS = ("text_config", "llm_config", "language_config")
# Nemotron-H's hybrid_override_pattern names each layer by one character.
PATTERN_LAYER_TYPES = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}


def geometry_of(config):
    """The geometry fields a configuration states, or None when it states none."""
    scopes = [config] + [config[key] for key in NESTED_CONFIGS if isinstance(config.get(key), dict)]
    geometry = {}
    for scope in scopes:
        for field, name in GEOMETRY_FIELDS.items():
            value = scope.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                geometry[name] = value
        types = scope.get("layer_types") or scope.get("layers_block_type")
        if isinstance(types, list) and all(isinstance(kind, str) for kind in types):
            geometry["layerTypes"] = types
        pattern = scope.get("hybrid_override_pattern")
        if isinstance(pattern, str) and pattern:
            geometry["layerTypes"] = [PATTERN_LAYER_TYPES.get(character, character) for character in pattern]
    return geometry or None


def geometry(repo, revision, files):
    """The geometry of the repository's root config.json, or None where it has none, states none, or
    refuses it without an accepted token."""
    if "config.json" not in files:
        return None
    try:
        body = _with_backoff(lambda: shapes._request(resolve_url(repo, revision, "config.json")), f"{repo} config")
        config = json.loads(body)
    except (Gated, Missing, RuntimeError, ValueError):
        return None
    return geometry_of(config) if isinstance(config, dict) else None


# MARK: - One variant

def directory_of(path):
    return path.split("/", 1)[0] if "/" in path else ""


def is_unmeasurable_url(url):
    host = urllib.parse.urlparse(url or "").hostname or ""
    return not url or host in UNMEASURABLE_HOSTS


class Measurement:
    def __init__(self, source):
        self.model = source["model"]
        self.variant = source["variant"]
        self.status = "measured"
        self.note = None
        self.commit = None
        self.download_bytes = None
        self.parameter_count = None
        self.tensor_bytes = {}
        self.opaque_bytes = {}
        self.layers = {}
        self.expert_bytes = {}
        self.quantizable_elements = {}
        self.geometry = None


def file_location(source, file):
    """(location, the revision the catalog names, is a URL) of one of a source's files."""
    if file.get("repository"):
        return file["repository"], file.get("revision"), False
    if file.get("url"):
        return file["url"], None, True
    if source["kind"] == "url":
        return source.get("url"), None, True
    return source.get("repository"), file.get("revision") or source.get("revision"), False


def measure_url_variant(source, rows):
    measurement = Measurement(source)
    files = source.get("files") or []
    if not files or any(is_unmeasurable_url(file_location(source, f)[0]) for f in files):
        measurement.status = "unmeasurable"
        measurement.note = f"{source.get('url') or 'no URL'} serves no size"
        return measurement
    total, missing = 0, []
    for file in files:
        url = file_location(source, file)[0]
        try:
            length = head_length(url)
        except Missing:
            missing.append(file["path"])
            continue
        except Gated as error:
            measurement.status, measurement.note = "unmeasurable", str(error)
            return measurement
        if length is None:
            measurement.status, measurement.note = "unmeasurable", f"{url} reports no Content-Length"
            return measurement
        rows[(url, None, file["path"])] = (length, None)
        total += length
        if file["role"] == "weights":
            directory = directory_of(file["path"])
            measurement.opaque_bytes[directory] = measurement.opaque_bytes.get(directory, 0) + length
    if missing:
        measurement.status, measurement.note = "missing", "not served: " + ", ".join(missing)
        return measurement
    measurement.download_bytes = total
    return measurement


def fetched_hub_files(source, rows):
    """The (repo, revision, path, role) the download fetches from Hugging Face, and the required files
    not served."""
    fetched, missing, groups = [], [], {}
    for file in source.get("files") or []:
        repo, revision, is_url = file_location(source, file)
        if is_url:
            continue
        files = tree(repo, revision)
        served = files.get(file["path"])
        if served is not None:
            rows[(repo, revision, file["path"])] = served
        if file["role"] == "weights" and file["optional"]:
            groups.setdefault((repo, revision, os.path.dirname(file["path"])), []).append(file["path"])
        elif served is not None:
            fetched.append((repo, revision, file["path"], file["role"]))
        elif not file["optional"]:
            missing.append(file["path"])
    for (repo, revision, _), candidates in sorted(groups.items()):
        files = tree(repo, revision)
        chosen = next((path for path in candidates if path in files), None)
        if chosen is None:
            missing.append(" or ".join(candidates))
        else:
            fetched.append((repo, revision, chosen, "weights"))
    expanded = []
    for repo, revision, path, role in fetched:
        expanded.append((repo, revision, path, role))
        if path.endswith(INDEX_SUFFIX):
            for shard in shard_paths(repo, revision, path):
                if shard in tree(repo, revision):
                    expanded.append((repo, revision, shard, "weights"))
                else:
                    missing.append(shard)
    unique = list(dict.fromkeys(expanded))
    return unique, missing


def measure_hub_variant(source, rows):
    measurement = Measurement(source)
    try:
        measurement.commit = commit(source["repository"], source.get("revision"))
        fetched, missing = fetched_hub_files(source, rows)
    except Gated as error:
        measurement.status, measurement.note = "gated", f"gated, unmeasured ({error})"
        return measurement
    except Missing as error:
        measurement.status, measurement.note = "missing", str(error)
        return measurement
    sized = [(repo, revision, path, role, tree(repo, revision)[path][0]) for repo, revision, path, role in fetched]
    for file in source.get("files") or []:
        if file_location(source, file)[2]:
            try:
                length = head_length(file["url"])
            except Missing:
                missing.append(file["url"])
                continue
            rows[(file["url"], None, file["path"])] = (length, None)
            sized.append((file["url"], None, file["path"], file["role"], length or 0))
    if missing:
        measurement.status, measurement.note = "missing", "not served: " + ", ".join(missing)
        return measurement
    total, elements, headers_read = 0, 0, True
    unreadable = []
    structures = {}
    for repo, revision, path, role, size in sized:
        total += size
        directory = directory_of(path)
        if path.endswith(".safetensors") and not repo.startswith("http"):
            try:
                by_dtype, count, tensors = tensor_stats(resolve_url(repo, revision, path))
            except (Gated, Missing, RuntimeError) as error:
                headers_read = False
                reason = str(error).rsplit(": ", 1)[-1]
                unreadable.append(f"{path} ({reason})")
                measurement.opaque_bytes[directory] = measurement.opaque_bytes.get(directory, 0) + size
                continue
            bucket = measurement.tensor_bytes.setdefault(directory, {})
            for dtype, nbytes in by_dtype.items():
                bucket[dtype] = bucket.get(dtype, 0) + nbytes
            elements += count
            structure = structures.setdefault(directory, Structure())
            for tensor in tensors:
                structure.add(*tensor)
        elif role == "weights" and not path.endswith(INDEX_SUFFIX):
            measurement.opaque_bytes[directory] = measurement.opaque_bytes.get(directory, 0) + size
    for directory, structure in structures.items():
        layers = structure.layers()
        if layers is not None:
            measurement.layers[directory] = layers
        if structure.expert_bytes:
            measurement.expert_bytes[directory] = structure.expert_bytes
        if structure.quantizable:
            measurement.quantizable_elements[directory] = structure.quantizable
    repository, revision = source["repository"], source.get("revision")
    measurement.geometry = geometry(repository, revision, tree(repository, revision))
    measurement.download_bytes = total
    if headers_read and not measurement.opaque_bytes and measurement.tensor_bytes:
        measurement.parameter_count = elements
    if unreadable:
        more = f" and {len(unreadable) - 3} more" if len(unreadable) > 3 else ""
        measurement.note = "tensor headers unread: " + ", ".join(unreadable[:3]) + more
    return measurement


def measure(source, rows):
    if source.get("publiclyServed") is False:
        measurement = Measurement(source)
        measurement.status, measurement.note = "unmeasurable", source.get("note") or "not served publicly"
        return measurement
    try:
        if source["kind"] == "url":
            return measure_url_variant(source, rows)
        return measure_hub_variant(source, rows)
    except (RuntimeError, urllib.error.URLError, OSError, ValueError) as error:
        measurement = Measurement(source)
        measurement.status, measurement.note = "unmeasurable", f"{type(error).__name__}: {error}"
        return measurement


# MARK: - Output

def swift_string(value):
    if value is None:
        return "nil"
    out = []
    for character in value:
        if character in '"\\':
            out.append("\\" + character)
        elif 32 <= ord(character) < 127:
            out.append(character)
        else:
            out.append("\\u{%x}" % ord(character))
    return '"' + "".join(out) + '"'


def swift_int_dictionary(values):
    if not values:
        return "[:]"
    return "[" + ", ".join(f"{swift_string(k)}: {values[k]}" for k in sorted(values)) + "]"


def swift_nested_dictionary(values):
    if not values:
        return "[:]"
    return "[" + ", ".join(f"{swift_string(k)}: {swift_int_dictionary(values[k])}" for k in sorted(values)) + "]"


def swift_optional_int(value):
    return "nil" if value is None else str(value)


def swift_geometry(geometry):
    fields = []
    for name in GEOMETRY_FIELDS.values():
        if name in geometry:
            fields.append(f"{name}: {geometry[name]}")
    if "layerTypes" in geometry:
        fields.append("layerTypes: [" + ", ".join(swift_string(kind) for kind in geometry["layerTypes"]) + "]")
    return "NFKMLXMeasuredGeometry(" + ", ".join(fields) + ")"


def swift_structure(m):
    """The structure arguments a measurement writes, each left out where it is empty."""
    parts = []
    if m.layers:
        rows = ", ".join(f"{swift_string(directory)}: NFKMLXMeasuredLayers(count: {count}, "
                         f"largestBytes: {swift_int_dictionary(largest)})"
                         for directory, (count, largest) in sorted(m.layers.items()))
        parts.append(f"layers: [{rows}]")
    if m.expert_bytes:
        parts.append(f"expertBytes: {swift_nested_dictionary(m.expert_bytes)}")
    if m.quantizable_elements:
        parts.append(f"quantizableElements: {swift_int_dictionary(m.quantizable_elements)}")
    if m.geometry:
        parts.append(f"geometry: {swift_geometry(m.geometry)}")
    return "".join(", " + part for part in parts)


def write_swift(rows, measurements, path):
    lines = [
        "//",
        "//  NFKMLXCatalogMeasurements.swift",
        "//  InferKitMLX",
        "//",
        "//  GENERATED by Tools/catalog/measure.py. Do not edit.",
        "//  Regenerate from the repository root: python3 -I Tools/catalog/measure.py",
        "//",
        "",
        "extension NFKMLXCatalogMeasurements {",
        "    static let measuredFiles: [NFKMLXMeasuredFile] = [",
    ]
    for (location, revision, file_path) in sorted(rows, key=lambda key: (key[0], key[1] or "", key[2])):
        size, sha = rows[(location, revision, file_path)]
        lines.append(f"        NFKMLXMeasuredFile(location: {swift_string(location)}, revision: {swift_string(revision)}, "
                     f"path: {swift_string(file_path)}, bytes: {size}, sha256: {swift_string(sha)}),")
    lines += ["    ]", "", "    static let measuredVariants: [NFKMLXMeasuredVariant] = ["]
    for m in sorted(measurements, key=lambda m: (m.model, m.variant)):
        lines.append(f"        NFKMLXMeasuredVariant(model: {swift_string(m.model)}, variant: {swift_string(m.variant)}, "
                     f"status: .{m.status}, note: {swift_string(m.note)}, commit: {swift_string(m.commit)}, "
                     f"downloadBytes: {swift_optional_int(m.download_bytes)}, "
                     f"parameterCount: {swift_optional_int(m.parameter_count)}, "
                     f"tensorBytes: {swift_nested_dictionary(m.tensor_bytes)}, "
                     f"opaqueBytes: {swift_int_dictionary(m.opaque_bytes)}"
                     f"{swift_structure(m)}),")
    lines += ["    ]", "}", ""]
    path.write_text("\n".join(lines))


def load_sources(path):
    if path:
        return json.loads(pathlib.Path(path).read_text())["sources"]
    with tempfile.TemporaryDirectory(prefix="inferkit-catalog-") as directory:
        listing = pathlib.Path(directory) / "sources.json"
        subprocess.run(["swift", "run", "--package-path", str(ROOT / "InferKitMLX"), "inferkit-catalog", "sources",
                        "--output", str(listing)], check=True, stdout=sys.stderr)
        return json.loads(listing.read_text())["sources"]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", help="the JSON `inferkit-catalog sources` wrote; runs the tool when absent")
    parser.add_argument("--output", default=str(OUTPUT), help="the Swift file to write")
    parser.add_argument("--jobs", type=int, default=6, help="concurrent requests")
    arguments = parser.parse_args(argv[1:])

    sources = load_sources(arguments.sources)
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        futures = [pool.submit(measure, source, rows) for source in sources]
        measurements = []
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            measurement = future.result()
            measurements.append(measurement)
            if done % 25 == 0:
                print(f"{done}/{len(futures)} variants", file=sys.stderr)
    write_swift(rows, measurements, pathlib.Path(arguments.output))

    by_status = {}
    for m in measurements:
        by_status.setdefault(m.status, []).append(m)
    total = sum(m.download_bytes or 0 for m in measurements)
    print(f"{len(measurements)} variants: " + ", ".join(f"{len(v)} {k}" for k, v in sorted(by_status.items())),
          file=sys.stderr)
    print(f"download bytes, measured variants: {total} ({total / 1e12:.2f} TB)", file=sys.stderr)
    with_geometry = sum(1 for m in measurements if m.geometry)
    with_layers = sum(1 for m in measurements if m.layers)
    print(f"config geometry: {with_geometry} variants; repeated layers: {with_layers} variants", file=sys.stderr)
    for status in ("gated", "unmeasurable", "missing"):
        for m in sorted(by_status.get(status, []), key=lambda m: (m.model, m.variant)):
            print(f"  {status}: {m.model}/{m.variant}: {m.note}", file=sys.stderr)
    print(f"wrote {arguments.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
