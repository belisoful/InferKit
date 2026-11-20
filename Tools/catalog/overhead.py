#!/usr/bin/env python3
"""The catalog's load overheads: measure each variant's load and reference-request peaks, then table them.

    python3 -I Tools/catalog/overhead.py plan --output PLAN.json [--locations LOCATIONS.json]
    python3 -I Tools/catalog/overhead.py run --plan PLAN.json --results DIR [--only KEY,...] [--families F,...]
    python3 -I Tools/catalog/overhead.py table --plan PLAN.json --results DIR [--results DIR ...]

`plan` lists every MLX catalog variant with the stored release that holds its files, found by matching the
variant's source repository against `Tools/validation-assets/manifest.json`'s releases and reading each
release's keys in `~/.inferkit-validation.json`. `--locations` names releases the manifest does not, as a
JSON object of `model/variant` to path. A variant without a release is measured on initial values where its
float32 bytes are at most `--initial-limit`.

`run` measures the selected variants one process each through `NFKMLXLoadOverheadTests`, smallest first,
writing `DIR/<model>__<variant>.json`. A measurement whose MLX active memory passes `--memory-cap` ends with
a record that names the phase it reached. `run` skips a variant whose record exists without an error unless
`--retry` is given. It builds nothing: build the tests first, and run it under the Large Model Coordination
lock (`Tools/lmc/lmc.py`).

`table` turns the records into `NFKMLXCatalogLoadOverheads.swift`. A later `--results` directory's record
replaces an earlier one's. A measured variant's overhead is the larger of its load transient and its request
peak less the bytes the memory facts already count that the request allocates (the key-value cache at the
prompt's positions, an expert cache, and, where the residency holds less than the weights after the load,
the weights it brings in during the request). A variant without a measurement takes its family's largest
measured figure, scaled by the hidden size (else the vocabulary) for a decoder. Figures round up to a whole
MiB. Two kinds of variant take no figure and lend none to their family:

- one whose reference request exceeds what a Mac holds: on initial values it passes the memory cap, or it
  asks Metal for one buffer past the device's limit; its default needs tiling or a shorter input
- a stored release that passes the cap with its weights resident, which a run at a higher cap measures
"""

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(ROOT, "Docs", "model-catalog.json")
MANIFEST = os.path.join(ROOT, "Tools", "validation-assets", "manifest.json")
MEASUREMENTS = os.path.join(ROOT, "InferKitMLX", "Sources", "InferKitMLX", "NFKMLXCatalogMeasurements.swift")
TABLE = os.path.join(ROOT, "InferKitMLX", "Sources", "InferKitMLX", "NFKMLXCatalogLoadOverheads.swift")
VALIDATION = os.path.expanduser("~/.inferkit-validation.json")
TEST = "NFKMLXLoadOverheadTests/testTheNamedVariantsLoadAndReferenceRequestPeaks"
MIB = 1 << 20
BUFFER_LIMIT = "maximum allowed buffer size"


def mlx_variants():
    with open(CATALOG) as handle:
        catalog = json.load(handle)
    for entry in catalog["models"]:
        if entry.get("engine") != "mlx":
            continue
        for variant in entry["variants"]:
            yield entry, variant


def weight_files(variant):
    """The weight files a variant's own repository lists, optional alternatives included."""
    source = variant.get("source") or {}
    return [f["path"] for f in source.get("files", []) if f.get("role") == "weights" and not f.get("repository")]


def candidate_roots(path, files):
    roots = [path, os.path.dirname(path)]
    for name in files:
        folder = os.path.dirname(name)
        if folder and path.rstrip("/").endswith(folder):
            roots.append(path.rstrip("/")[: -len(folder)].rstrip("/"))
    return roots


def holds_candidates(root, files):
    """True where each folder of the listed weight files holds one of them: a listing names alternatives,
    such as a release's fp16 and full-precision files, of which the download takes the first served. A shard
    index counts only with its shards."""
    folders = {}
    for name in files:
        folders.setdefault(os.path.dirname(name), []).append(name)

    def present(name):
        path = os.path.join(root, name)
        if name.endswith(".index.json"):
            return os.path.exists(path) and holds_weights(os.path.dirname(path))
        return os.path.exists(path)
    return all(any(present(f) for f in group) for group in folders.values())


def holds_weights(folder):
    """True for a release folder that holds its weights: every shard its index names, else a weight file."""
    names = os.listdir(folder)
    for name in names:
        if name.endswith(".index.json"):
            with open(os.path.join(folder, name)) as handle:
                shards = set(json.load(handle).get("weight_map", {}).values())
            return bool(shards) and all(os.path.exists(os.path.join(folder, shard)) for shard in shards)
    return any(n.endswith((".safetensors", ".bin")) for n in names)


def stored_location(variant, releases, keys, given=None):
    source = variant.get("source") or {}
    files = weight_files(variant)
    layout = source.get("layout")
    paths = [given] if given else [keys.get(key) for release in releases.get(source.get("repository"), [])
                                   for key in release.get("config", {})]
    for path in paths:
        if not isinstance(path, str) or not os.path.exists(path):
            continue
        for root in candidate_roots(path, files):
            if layout == "file":
                if os.path.isfile(root) and any(os.path.basename(f) == os.path.basename(root) for f in files):
                    return root
            elif os.path.isdir(root) and holds_candidates(root, files) and (files or holds_weights(root)):
                return root
    return None


def float32_bytes(variant):
    facts = variant.get("memoryFacts") or {}
    total = 0
    for stage in facts.get("stages", []):
        sizes = stage.get("bytesByPrecision", {})
        total += sizes.get("float32") or sizes.get("checkpoint") or 0
    return total or variant.get("residentBytes") or 0


def plan(arguments):
    with open(MANIFEST) as handle:
        manifest = json.load(handle)
    with open(VALIDATION) as handle:
        keys = json.load(handle)
    given = {}
    if arguments.locations:
        with open(arguments.locations) as handle:
            given = json.load(handle)
    releases = {}
    for release in manifest["releases"]:
        if release.get("repo"):
            releases.setdefault(release["repo"], []).append(release)
    rows = []
    for entry, variant in mlx_variants():
        key = f"{entry['id']}/{variant['id']}"
        location = stored_location(variant, releases, keys, given.get(key))
        if key in given and location is None:
            print(f"{key}: {given[key]} does not hold the variant's weight files")
        size = float32_bytes(variant)
        rows.append({
            "key": key, "family": entry["family"], "location": location,
            "weights": "stored" if location else ("initial" if size <= arguments.initial_limit else "none"),
            "float32Bytes": size, "residentBytes": variant.get("residentBytes") or size,
        })
    with open(arguments.output, "w") as handle:
        json.dump(rows, handle, indent=1)
    counts = {}
    for row in rows:
        counts[row["weights"]] = counts.get(row["weights"], 0) + 1
    families = {row["family"] for row in rows if row["weights"] == "stored"}
    print(f"{len(rows)} variants: {counts}; stored releases cover {len(families)} families")


def record_path(results, key):
    return os.path.join(results, key.replace("/", "__") + ".json")


def group_resident_bytes(group):
    listing = subprocess.run(["ps", "-o", "rss=", "-g", str(group)], capture_output=True, text=True).stdout
    return sum(int(line) for line in listing.split() if line.isdigit()) * 1024


def watch(process, timeout, cap):
    """The process's exit status, or "timeout" or "over the memory cap" after killing its group."""
    deadline = time.time() + timeout
    while True:
        try:
            return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        reason = None
        if time.time() > deadline:
            reason = "timeout"
        elif group_resident_bytes(process.pid) > cap:
            reason = "over the memory cap"
        if reason:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return reason


def run(arguments):
    with open(arguments.plan) as handle:
        rows = json.load(handle)
    only = set(arguments.only.split(",")) if arguments.only else None
    families = set(arguments.families.split(",")) if arguments.families else None
    selected = [r for r in rows if r["weights"] != "none"
                and (only is None or r["key"] in only) and (families is None or r["family"] in families)
                and (arguments.max_bytes is None or r["residentBytes"] <= arguments.max_bytes)]
    selected.sort(key=lambda r: r["residentBytes"])
    os.makedirs(arguments.results, exist_ok=True)
    package = os.path.join(arguments.root or ROOT, "InferKitMLX")
    for index, row in enumerate(selected, 1):
        output = record_path(arguments.results, row["key"])
        if os.path.exists(output) and not arguments.retry:
            with open(output) as handle:
                if "error" not in json.load(handle):
                    continue
        environment = dict(os.environ, IK_LOAD_OVERHEAD_VARIANT=row["key"], IK_LOAD_OVERHEAD_OUTPUT=output,
                           IK_LOAD_OVERHEAD_LOCATION=row["location"] or "",
                           IK_LOAD_OVERHEAD_CAP=str(arguments.memory_cap))
        log = output[:-5] + ".log"
        started = time.time()
        print(f"[{index}/{len(selected)}] {row['key']} ({row['weights']}, {row['residentBytes'] / 2**30:.2f} GiB)",
              flush=True)
        with open(log, "w") as handle:
            process = subprocess.Popen(["swift", "test", "--skip-build", "--filter", TEST], cwd=package,
                                       env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status = watch(process, arguments.timeout, arguments.memory_cap)
        if not os.path.exists(output) or os.path.getmtime(output) < started:
            with open(output, "w") as handle:
                json.dump({"variantKey": row["key"], "weights": row["weights"],
                           "error": f"the process ended ({status}) without a record"}, handle, indent=1)
        print(f"    {status} in {time.time() - started:.0f} s", flush=True)


def hidden_sizes():
    import re
    sizes = {}
    pattern = re.compile(r'NFKMLXMeasuredVariant\(model: "([^"]+)", variant: "([^"]+)".*?'
                         r'geometry: NFKMLXMeasuredGeometry\(([^)]*)\)')
    with open(MEASUREMENTS) as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            fields = dict(re.findall(r"(\w+): (\d+)", match.group(3)))
            sizes[f"{match.group(1)}/{match.group(2)}"] = (int(fields.get("hiddenSize", 0)) or None,
                                                          int(fields.get("vocabularySize", 0)) or None)
    return sizes


def counted_during_request(record, facts):
    """The bytes the memory facts count in the resident figure that the request, not the load, allocates."""
    estimate = record.get("estimate") or {}
    components = estimate.get("components") or {}
    counted = components.get("expertCache", 0)
    cache = components.get("keyValueCache", 0)
    request = record.get("request") or {}
    tokens = request.get("promptTokens") or request.get("promptTokensTarget")
    if cache and tokens and facts.get("keyValueBytesPerToken"):
        cache = min(cache, facts["keyValueBytesPerToken"] * (tokens + 1) + facts.get("boundedKeyValueBytes", 0))
    counted += cache
    if estimate.get("residency") in ("staged", "streamed", "paged") and record.get("weights") == "stored":
        counted += max(0, components.get("weights", 0) - record.get("residentAfterLoad", 0))
    return counted


def figure(record, facts):
    """(overhead, load transient, request overhead) of a measured record."""
    load = record.get("loadTransient", 0) if record.get("weights") == "stored" else 0
    request = max(0, record.get("requestPeak", 0) - counted_during_request(record, facts))
    return max(load, request), load, request


def round_up(value):
    return int(math.ceil(value / MIB)) * MIB


def exceeding_buffer(record, path):
    """The Metal buffer-limit message a record's request or its process log ends on, or None."""
    if BUFFER_LIMIT in record.get("error", ""):
        return record["error"]
    log = path[:-5] + ".log"
    if "without a record" in record.get("error", "") and os.path.exists(log):
        with open(log, errors="replace") as handle:
            for line in handle:
                if BUFFER_LIMIT in line:
                    return line.strip()
    return None


def table(arguments):
    variants = {f"{e['id']}/{v['id']}": (e, v) for e, v in mlx_variants()}
    planned = {}
    if arguments.plan:
        with open(arguments.plan) as handle:
            planned = {row["key"]: row for row in json.load(handle)}
    records = {}
    for directory in arguments.results:
        for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            with open(path) as handle:
                record = json.load(handle)
            if record.get("variantKey") in variants:
                records[record["variantKey"]] = (record, path)
    geometry = hidden_sizes()
    measured, failures, exceeds, remeasure = {}, {}, {}, {}
    for key, (record, path) in records.items():
        weights = record.get("weights") or planned.get(key, {}).get("weights")
        limit = exceeding_buffer(record, path)
        if record.get("overCap") and weights == "initial":
            exceeds[key] = f"{record['activeBytes'] / 1e9:.1f} GB active in the {record.get('phase')}, past the cap"
            continue
        if limit:
            exceeds[key] = limit[limit.find("Attempting"):][:120] if "Attempting" in limit else limit[:120]
            continue
        if record.get("overCap"):
            remeasure[key] = f"{record['activeBytes'] / 1e9:.1f} GB active in the {record.get('phase')}"
            continue
        if "error" in record or "requestPeak" not in record:
            failures[key] = record.get("error", "no request peak")
            continue
        facts = variants[key][1].get("memoryFacts") or {}
        overhead, load, request = figure(record, facts)
        measured[key] = {"bytes": round_up(overhead), "load": round_up(load), "request": round_up(request),
                         "weights": record["weights"]}
    by_family = {}
    for key in measured:
        by_family.setdefault(variants[key][0]["family"], []).append(key)
    lines = []
    basis_counts = {"measured": 0, "family": 0, "none": 0}
    for key in sorted(variants):
        entry, variant = variants[key]
        if key in measured:
            row = measured[key]
            basis_counts["measured"] += 1
            lines.append(f'"{key}": NFKMLXLoadOverhead(bytes: {row["bytes"]}, basis: .measured, '
                         f'loadBytes: {row["load"]}, requestBytes: {row["request"]}, '
                         f'weights: .{row["weights"]})')
            continue
        sources = by_family.get(entry["family"], [])
        if not sources or key in exceeds or key in remeasure:
            basis_counts["none"] += 1
            continue
        source = max(sources, key=lambda k: measured[k]["bytes"])
        bytes_ = measured[source]["bytes"]
        facts = variant.get("memoryFacts") or {}
        if facts.get("keyValueBytesPerToken"):
            hidden, vocabulary = geometry.get(key, (None, None))
            source_hidden, source_vocabulary = geometry.get(source, (None, None))
            if hidden and source_hidden:
                bytes_ = round_up(bytes_ * hidden / source_hidden)
            elif vocabulary and source_vocabulary:
                bytes_ = round_up(bytes_ * vocabulary / source_vocabulary)
        basis_counts["family"] += 1
        lines.append(f'"{key}": NFKMLXLoadOverhead(bytes: {bytes_}, basis: .family, source: "{source}")')
    write_table(lines, arguments.output or TABLE)
    print(f"{len(variants)} variants: {basis_counts}; {len(failures)} records without a figure")
    for title, rows in (("default request exceeds a Mac (no figure)", exceeds),
                        ("stored release past the cap (remeasure at a higher cap)", remeasure),
                        ("no figure", failures)):
        print(f"{title}: {len(rows)}")
        for key, reason in sorted(rows.items()):
            print(f"  {key}: {reason}")
    if arguments.summary:
        with open(arguments.summary, "w") as handle:
            json.dump({"measured": measured, "exceeds": exceeds, "remeasure": remeasure, "failures": failures},
                      handle, indent=1, sort_keys=True)


HEADER = """//
//  NFKMLXCatalogLoadOverheads.swift
//  InferKitMLX
//
//  GENERATED by Tools/catalog/overhead.py. Do not edit.
//  Regenerate from the repository root: python3 -I Tools/catalog/overhead.py table --results <records>
//

extension NFKMLXCatalogMeasurements {
"""

# The rows a single dictionary literal holds. The type checker's time on one literal grows faster than its
# length: one literal of 344 rows took 0.4 s, past the 200 ms every expression is held to.
SWIFT_CHUNK_ROWS = 25


def write_table(rows, output):
    """Writes the rows, each a `"model/variant": NFKMLXLoadOverhead(...)` entry, as typed literals of
    `SWIFT_CHUNK_ROWS` rows that one dictionary joins."""
    declaration = "    static let loadOverheads: [String: NFKMLXLoadOverhead] = "
    if not rows:
        lines = [declaration + "[:]"]
    else:
        chunks = [rows[start:start + SWIFT_CHUNK_ROWS] for start in range(0, len(rows), SWIFT_CHUNK_ROWS)]
        names = [f"loadOverheads{index}" for index in range(len(chunks))]
        lines = [declaration + "Dictionary(uniqueKeysWithValues: ([",
                 *(f"        {name}," for name in names),
                 "    ] as [[String: NFKMLXLoadOverhead]]).joined().map { ($0.key, $0.value) })"]
        for name, chunk in zip(names, chunks):
            lines += ["", f"    private static let {name}: [String: NFKMLXLoadOverhead] = ["]
            lines += [f"        {row}," for row in chunk]
            lines.append("    ]")
    with open(output, "w") as handle:
        handle.write(HEADER + "\n".join(lines) + "\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--output", required=True)
    planning.add_argument("--locations", help="a JSON object of model/variant to the release that holds it, "
                          "for releases the manifest does not name")
    planning.add_argument("--initial-limit", type=int, default=6 << 30,
                          help="the largest float32 bytes measured on initial values (default 6 GiB)")
    running = commands.add_parser("run")
    running.add_argument("--plan", required=True)
    running.add_argument("--results", required=True)
    running.add_argument("--root", help="the repository whose InferKitMLX build runs (default this one)")
    running.add_argument("--only")
    running.add_argument("--families")
    running.add_argument("--max-bytes", type=int)
    running.add_argument("--timeout", type=int, default=1800)
    running.add_argument("--retry", action="store_true")
    running.add_argument("--memory-cap", type=int, default=18 << 30,
                         help="MLX's active bytes, or the group's resident bytes, at which a measurement ends "
                              "(default 18 GiB; one allocation can pass it by several GiB)")
    tabling = commands.add_parser("table")
    tabling.add_argument("--results", required=True, action="append",
                         help="a records directory; a later one's record replaces an earlier one's")
    tabling.add_argument("--plan", help="the plan the records came from, for records that do not name their weights")
    tabling.add_argument("--output", help="the table to write (default NFKMLXCatalogLoadOverheads.swift)")
    tabling.add_argument("--summary", help="a JSON file of the figures and of the variants without one")
    arguments = parser.parse_args()
    {"plan": plan, "run": run, "table": table}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
