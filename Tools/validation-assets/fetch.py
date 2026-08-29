#!/usr/bin/env python3
"""Fetch and convert every checkpoint the InferKitMLX parity and triage suites load.

The validation artifacts are large and were previously kept in a session scratchpad, which is deleted
between sessions: a green parity run then meant "nothing regressed among the records that still exist",
not "everything is verified". This puts them somewhere durable and makes rebuilding them one command.

    python3 fetch.py                 # fetch, convert, and point ~/.inferkit-validation.json at the result
    python3 fetch.py --root DIR      # somewhere other than ~/.inferkit-validation
    python3 fetch.py --only SAM CLIP # a subset, by manifest key (an asset or a release)
    python3 fetch.py --check         # report what is present and what is missing, download nothing

Downloads resume, and an asset whose file is already the expected size is skipped, so re-running after
an interruption costs only what is left.

Requires: curl, torch, safetensors (the converters' own requirements).
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
CONFIG = os.path.expanduser("~/.inferkit-validation.json")
DEFAULT_ROOT = os.path.expanduser("~/.inferkit-validation")


def load_manifest():
    with open(os.path.join(HERE, "manifest.json")) as handle:
        return json.load(handle)


USER_AGENT = "InferKit-validation/0.1 (https://github.com/belisoful/InferKit)"


def download(url, destination, expected):
    if os.path.exists(destination) and os.path.getsize(destination) == expected:
        return True, "cached"
    # `--continue-at -` resumes a partial file; a source that does not support ranges restarts.
    # Wikimedia refuses a request with no User-Agent, so every fetch identifies itself.
    command = ["curl", "-sSL", "--fail", "--retry", "3", "--continue-at", "-",
               "-A", USER_AGENT, "-o", destination, url]
    result = subprocess.run(command)
    if result.returncode != 0:
        # A completed file makes curl exit 33 on a resume attempt; treat a correct size as success.
        if os.path.exists(destination) and os.path.getsize(destination) == expected:
            return True, "cached"
        return False, f"curl exited {result.returncode}"
    size = os.path.getsize(destination) if os.path.exists(destination) else 0
    if expected and size != expected:
        return False, f"size {size}, expected {expected}"
    return True, "downloaded"


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
    parser.add_argument("--root", default=DEFAULT_ROOT, help="where the assets live")
    parser.add_argument("--only", nargs="*", help="manifest keys to act on")
    parser.add_argument("--check", action="store_true", help="report state, download nothing")
    args = parser.parse_args()

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

    raw_directory = os.path.join(args.root, "raw")
    inputs_directory = os.path.join(args.root, "inputs")
    converted_directory = os.path.join(args.root, "converted")
    for directory in (raw_directory, converted_directory, inputs_directory):
        os.makedirs(directory, exist_ok=True)

    config = {}
    if os.path.exists(CONFIG):
        with open(CONFIG) as handle:
            config = json.load(handle)

    succeeded, failed = [], []
    for asset in assets:
        # An input asset lands in inputs/ as-is; everything else lands in converted/.
        home = inputs_directory if asset.get("kind") == "input" else converted_directory
        converted = os.path.join(home, asset["file"])
        if args.check:
            state = "present" if os.path.exists(converted) else "MISSING"
            print(f"{asset['key']:<12} {state:<8} {converted}")
            continue
        if os.path.exists(converted):
            print(f"{asset['key']:<12} present")
            for key in asset["config"]:
                config[key] = converted
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
        ok, detail = download(asset["url"], raw, asset.get("bytes", 0))
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
        succeeded.append(asset["key"])

    if args.check:
        return 0

    for release in manifest.get("releases", []):
        if args.only and release["key"] not in wanted:
            continue
        directory = os.path.join(args.root, release["directory"])
        for relative in release["files"]:
            destination = os.path.join(directory, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                continue
            url = f"https://huggingface.co/{release['repo']}/resolve/main/{relative}"
            # A release's own size is not recorded: these are directories of many files, and the
            # `.safetensors` header is what a wrong one shows up in, not a byte count.
            result = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3",
                                     "--continue-at", "-", "-o", destination, url])
            if result.returncode != 0:
                print(f"{release['key']:<12} FETCH FAILED  {relative}")
                failed.append((release["key"], relative))
        for key, relative in release["config"].items():
            config[key] = os.path.join(directory, relative) if relative else directory
        print(f"{release['key']:<12} ready ({len(release['files'])} files)")

    sources = manifest.get("sources")
    if sources and not args.only:
        source_root = os.path.join(args.root, "sources")
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

    with open(CONFIG, "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"\n{len(succeeded)} ready, {len(failed)} failed; {CONFIG} updated")
    for key, detail in failed:
        print(f"  {key}: {detail}")
    for entry in manifest.get("unresolved", []):
        print(f"  {entry['key']}: unresolved — {entry['note']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
