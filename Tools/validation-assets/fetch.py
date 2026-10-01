#!/usr/bin/env python3
"""Fetch and convert every checkpoint the InferKitMLX parity and triage suites load.

The validation artifacts are large and were previously kept in a session scratchpad, which is deleted
between sessions: a green parity run then meant "nothing regressed among the records that still exist",
not "everything is verified". This puts them somewhere durable and makes rebuilding them one command.

    python3 fetch.py                 # fetch, convert, and point ~/.inferkit-validation.json at the result
    python3 fetch.py --root DIR      # somewhere other than the default validation root
    python3 fetch.py --only SAM CLIP # a subset, by manifest key (an asset or a release)
    python3 fetch.py --check         # report what is present and what is missing, download nothing
    python3 fetch.py --keep-in-backup # leave Time Machine's settings alone
    python3 fetch.py --config FILE   # read and write a key file other than ~/.inferkit-validation.json

On macOS the asset root, and the Hugging Face cache the reference oracles fill, are excluded from Time
Machine: everything in them can be fetched again.

An asset or release whose every key already names an existing path in the key file is present wherever
that path is, in this root, the other tree on the InferKit Models volume, or a local copy, and neither its
files nor its keys change. Downloads resume, and an asset whose file is already the expected size is
skipped, so re-running after an interruption costs only what is left. A release directory's file sizes come from the Hub's tree
listing, so a file left partial by an interrupted run is resumed rather than accepted.

Requires: curl, torch, safetensors (the converters' own requirements).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
CONFIG = os.path.expanduser("~/.inferkit-validation.json")


def read_config(path):
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def validation_root(config_path=CONFIG):
    """The validation store, in NFKMLXValidationConfig.root's order: IK_VALIDATION_ROOT from the
    environment overlaid with the key file (the key file wins), else the InferKit Models volume while it
    is mounted, else ~/.inferkit-validation."""
    merged = dict(os.environ)
    merged.update(read_config(config_path))
    if merged.get("IK_VALIDATION_ROOT"):
        return merged["IK_VALIDATION_ROOT"]
    external = "/Volumes/InferKit Models/inferkit-validation"
    if os.path.isdir(external):
        return external
    return os.path.expanduser("~/.inferkit-validation")


def resolved_by_config(keys, config):
    """Whether every key already names an existing path, wherever it is."""
    return bool(keys) and all(config.get(key) and os.path.exists(config[key]) for key in keys)


def resolved_directories(releases, config):
    """Each release directory the key file already locates: the existing path of a key whose value is
    the directory itself. Releases that share a directory (one per Cosmos Tokenizer variant) and carry
    no keys of their own are found through it."""
    located = {}
    for release in releases:
        for key, relative in release["config"].items():
            if not relative and config.get(key) and os.path.isdir(config[key]):
                located.setdefault(release.get("directory"), config[key])
    return located


def release_path(release, directory, root, relative):
    """Where a release's config value points. An empty value is the release directory; a value naming
    one of the release's files, or a bare file name, sits in the release directory; any other path
    (`records/…`, `raw/…`, another release's folder) is relative to the store root."""
    if not relative:
        return directory
    inside = any(name == relative or name.startswith(relative.rstrip("/") + "/")
                 for name in release.get("files", []))
    if inside or "/" not in relative:
        return os.path.join(directory, relative)
    return os.path.join(root, relative)


def huggingface_cache():
    """The folder huggingface_hub caches into, by the same environment variables it reads."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface")
    return os.path.join(home, "hub")


def exclude_from_backup(path):
    """Sets Time Machine's sticky exclusion, the attribute NFKHFHub sets on its own cache."""
    if sys.platform != "darwin" or not os.path.isdir(path):
        return
    subprocess.run(["tmutil", "addexclusion", path], check=False, capture_output=True)


def load_manifest():
    with open(os.path.join(HERE, "manifest.json")) as handle:
        return json.load(handle)


USER_AGENT = "InferKit-validation/0.1 (https://github.com/belisoful/InferKit)"
CURL_RANGE_ALREADY_SATISFIED = 33


def hub_headers():
    token = os.environ.get("HF_TOKEN")
    return ["-H", f"Authorization: Bearer {token}"] if token else []


def download(url, destination, expected):
    """Fetches `url` to `destination`, resuming a partial file; `expected` is its size, or 0 when unknown."""
    if expected and os.path.exists(destination) and os.path.getsize(destination) == expected:
        return True, "cached"
    # `--continue-at -` resumes a partial file; a source that does not support ranges restarts.
    # Wikimedia refuses a request with no User-Agent, so every fetch identifies itself.
    command = ["curl", "-sSL", "--fail", "--retry", "3", "--continue-at", "-",
               "-A", USER_AGENT, "-o", destination, url]
    before = os.path.getsize(destination) if os.path.exists(destination) else 0
    result = subprocess.run(command)
    size = os.path.getsize(destination) if os.path.exists(destination) else 0
    # A completed file answers a resume with 416: curl 8 exits 0 and leaves the file alone, older
    # curls exit 33. With a known size that is success only when the file has it; with no size on
    # record the range-satisfied answer itself is the completion signal.
    if result.returncode == CURL_RANGE_ALREADY_SATISFIED and (not expected or size == expected):
        return True, "cached"
    if result.returncode != 0:
        return False, f"curl exited {result.returncode}"
    if expected and size != expected:
        return False, f"size {size}, expected {expected}"
    return True, "cached" if before and size == before else "downloaded"


def hub_file_sizes(repo, revision):
    """The byte size of every file at `revision` of `repo`, by path; empty when the listing is unavailable."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    # `huggingface.co` advertises IPv6 addresses this host cannot route, so the listing pins IPv4.
    result = subprocess.run(["curl", "-4", "-sfL", "--retry", "3", *hub_headers(), url],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        entries = json.loads(result.stdout)
    except ValueError:
        return {}
    return {entry["path"]: entry["size"] for entry in entries
            if entry.get("type") == "file" and isinstance(entry.get("size"), int)}


def fetch_release(release, directory):
    """Downloads every file of a release directory, resuming partial ones; returns the relatives that failed."""
    if "repo" not in release:
        # A release fetched by hand (a `git` or `hf` entry) is only checked for presence.
        return [(relative, "no repo to fetch from") for relative in release["files"]
                if not os.path.exists(os.path.join(directory, relative))]
    revision = release.get("revision", "main")
    # A release's own sizes are not recorded in the manifest: these are directories of many files. The
    # Hub's tree listing supplies them, so a complete file is skipped without a request and a partial
    # one (a run interrupted mid-download) is resumed. A file the listing does not name is resumed
    # blind, and curl's range-satisfied exit marks it complete.
    sizes = hub_file_sizes(release["repo"], revision)
    failed = []
    for relative in release["files"]:
        destination = os.path.join(directory, relative)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f"https://huggingface.co/{release['repo']}/resolve/{revision}/{relative}"
        ok, detail = download(url, destination, sizes.get(relative, 0))
        if not ok:
            failed.append((relative, detail))
    return failed


def download_gdrive(file_id, destination, expected):
    """Fetch a Google Drive file by id, for a checkpoint served nowhere a plain URL reaches."""
    if os.path.exists(destination) and (not expected or os.path.getsize(destination) == expected):
        return True, "cached"
    result = subprocess.run([sys.executable, "-m", "gdown", "-q", file_id, "-O", destination])
    if result.returncode != 0:
        return False, f"gdown exited {result.returncode} (pip install gdown)"
    size = os.path.getsize(destination) if os.path.exists(destination) else 0
    if expected and size != expected:
        return False, f"size {size}, expected {expected}"
    return True, "downloaded (gdown)"


def download_zip_member(url, member, destination, raw_directory):
    """Download a zip archive and extract one member as the raw checkpoint, for a release
    distributed as an archive rather than a bare file."""
    archive = os.path.join(raw_directory, os.path.basename(url))
    ok, detail = download(url, archive, 0)                  # the archive's own size is not pinned
    if not ok:
        return False, detail
    try:
        with zipfile.ZipFile(archive) as zf, zf.open(member) as source, open(destination, "wb") as sink:
            shutil.copyfileobj(source, sink)
    except (KeyError, zipfile.BadZipFile) as error:
        return False, f"extract {member}: {error}"
    return True, f"extracted {member}"


def acquire_raw(asset, raw, raw_directory):
    """Fetch the asset's raw checkpoint to `raw`, by whichever route the manifest names: a Google
    Drive id (`gdrive`), a member of a downloaded zip (`extract`, the path inside the archive at
    `url`), or a plain `url`."""
    expected = asset.get("bytes", 0)
    if os.path.exists(raw) and (not expected or os.path.getsize(raw) == expected):
        return True, "cached"
    if asset.get("gdrive"):
        return download_gdrive(asset["gdrive"], raw, expected)
    if asset.get("extract"):
        return download_zip_member(asset["url"], asset["extract"], raw, raw_directory)
    return download(asset["url"], raw, expected)


def record_raw_path(asset, raw_directory, config):
    """Points IK_RAW_<KEY> at the kept raw checkpoint. The raw file is the Swift torch-checkpoint
    reader's test input and the converted file is its oracle, so the key exists for every asset
    whose raw download is still on disk. Derived rather than listed in the manifest, so a new asset
    gains one automatically."""
    raw_name = asset.get("raw")
    if not raw_name:
        return
    raw_path = os.path.join(raw_directory, raw_name)
    if os.path.exists(raw_path):
        config[f"IK_RAW_{asset['key']}"] = raw_path


def convert(asset, raw, converted):
    script = os.path.join(TOOLS, asset["converter"], "convert.py")
    if not os.path.exists(script):
        return False, f"no converter at {script}"
    result = subprocess.run([sys.executable, script, raw, converted],
                            capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(converted):
        tail = (result.stderr or result.stdout).strip().splitlines()
        return False, tail[-1] if tail else f"converter exited {result.returncode}"
    return True, "converted"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="where the assets live (default: the validation root)")
    parser.add_argument("--config", default=CONFIG, help="the key file to read and update")
    parser.add_argument("--only", nargs="*", help="manifest keys to act on")
    parser.add_argument("--check", action="store_true", help="report state, download nothing")
    parser.add_argument("--keep-in-backup", action="store_true",
                        help="do not exclude the asset root and the Hugging Face cache from Time Machine")
    args = parser.parse_args()
    root = args.root or validation_root(args.config)

    manifest = load_manifest()
    assets = manifest["assets"]
    wanted = set()
    if args.only:
        wanted = {key.upper() for key in args.only}
        releases = {release["key"] for release in manifest.get("releases", [])}
        assets = [asset for asset in assets if asset["key"] in wanted]
        missing = wanted - {asset["key"] for asset in assets} - releases
        if missing:
            raise SystemExit(f"no such manifest key: {', '.join(sorted(missing))}")

    raw_directory = os.path.join(root, "raw")
    inputs_directory = os.path.join(root, "inputs")
    converted_directory = os.path.join(root, "converted")
    if not args.check:
        for directory in (raw_directory, converted_directory, inputs_directory):
            os.makedirs(directory, exist_ok=True)
        if not args.keep_in_backup:
            exclude_from_backup(root)
            exclude_from_backup(huggingface_cache())

    config = read_config(args.config)
    loaded = dict(config)

    succeeded, failed = [], []
    for asset in assets:
        # An input asset lands in inputs/ as-is; everything else lands in converted/.
        home = inputs_directory if asset.get("kind") == "input" else converted_directory
        converted = os.path.join(home, asset["file"])
        if resolved_by_config(asset["config"], config):
            print(f"{asset['key']:<12} present  {config[asset['config'][0]]}")
            succeeded.append(asset["key"])
            continue
        if args.check:
            state = "present" if os.path.exists(converted) else "MISSING"
            print(f"{asset['key']:<12} {state:<8} {converted}")
            continue
        if os.path.exists(converted):
            print(f"{asset['key']:<12} present")
            for key in asset["config"]:
                config[key] = converted
            record_raw_path(asset, raw_directory, config)
            succeeded.append(asset["key"])
            continue

        # An input asset is a plate the tests read as-is; there is nothing to convert.
        if asset.get("kind") == "input":
            destination = os.path.join(inputs_directory, asset["file"])
            ok, detail = download(asset["url"], destination, asset.get("bytes", 0))
            if not ok:
                print(f"{asset['key']:<12} FETCH FAILED  {detail}")
                failed.append((asset["key"], detail))
                continue
            print(f"{asset['key']:<12} ready")
            for key in asset["config"]:
                config[key] = destination
            succeeded.append(asset["key"])
            continue

        raw = os.path.join(raw_directory, asset["raw"])
        ok, detail = acquire_raw(asset, raw, raw_directory)
        if not ok:
            print(f"{asset['key']:<12} FETCH FAILED  {detail}")
            failed.append((asset["key"], detail))
            continue
        ok, detail = convert(asset, raw, converted)
        if not ok:
            print(f"{asset['key']:<12} CONVERT FAILED  {detail}")
            failed.append((asset["key"], detail))
            continue
        print(f"{asset['key']:<12} ready")
        for key in asset["config"]:
            config[key] = converted
        record_raw_path(asset, raw_directory, config)
        succeeded.append(asset["key"])

    located = resolved_directories(manifest.get("releases", []), config)
    for release in manifest.get("releases", []):
        if args.only and release["key"] not in wanted:
            continue
        if resolved_by_config(list(release["config"]), config):
            print(f"{release['key']:<12} present  {config[next(iter(release['config']))]}")
            continue
        elsewhere = located.get(release.get("directory"))
        if (not release["config"] and elsewhere
                and all(os.path.exists(os.path.join(elsewhere, name)) for name in release["files"])):
            print(f"{release['key']:<12} present  {elsewhere}")
            continue
        # A release without a directory of its own (an `hf` entry) names its paths from the root.
        directory = os.path.join(root, release.get("directory", ""))
        paths = {key: release_path(release, directory, root, relative)
                 for key, relative in release["config"].items()}
        if args.check:
            state = "present" if all(os.path.exists(path) for path in paths.values()) else "MISSING"
            print(f"{release['key']:<12} {state:<8} {directory}")
            continue
        release_failed = fetch_release(release, directory)
        for relative, detail in release_failed:
            print(f"{release['key']:<12} FETCH FAILED  {relative}  {detail}")
            failed.append((release["key"], f"{relative}: {detail}"))
        config.update(paths)
        if not release_failed:
            print(f"{release['key']:<12} ready ({len(release['files'])} files)")

    if args.check:
        return 0

    sources = manifest.get("sources")
    if sources and not args.only:
        source_root = os.path.join(root, "sources")
        for relative, url in sources["files"]:
            destination = os.path.join(source_root, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if url is None:                                     # a package marker, not a download
                open(destination, "a").close()
                continue
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                continue
            result = subprocess.run(["curl", "-sSL", "--fail", "-o", destination, url])
            if result.returncode != 0:
                print(f"{'source':<12} FETCH FAILED  {relative}")
                failed.append((relative, "source download"))
        # The oracles read these paths from the environment; recording them in the config keeps the
        # whole recipe in one place.
        for key, relative in sources["env"].items():
            config[key] = os.path.join(source_root, relative)
        print(f"{'sources':<12} ready ({len(sources['files'])} files)")

    # A download runs for minutes to hours while other tools add their own keys, so only the keys this
    # run set are merged into the file as it stands now; writing the snapshot back would drop theirs.
    changed = {key: value for key, value in config.items() if loaded.get(key) != value}
    current = read_config(args.config)
    current.update(changed)
    partial = args.config + ".partial"
    with open(partial, "w") as handle:
        json.dump(current, handle, indent=2)
    os.replace(partial, args.config)
    print(f"\n{len(succeeded)} ready, {len(failed)} failed; {args.config} updated")
    for key, detail in failed:
        print(f"  {key}: {detail}")
    for entry in manifest.get("unresolved", []):
        print(f"  {entry['key']}: unresolved — {entry['note']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
