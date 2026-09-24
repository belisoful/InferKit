#!/usr/bin/env python3
"""Exercises fetch.py's resume logic against a fake curl and a fake Hub listing.

    python3 Tools/validation-assets/test_fetch.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch  # noqa: E402

PAYLOAD = b"0123456789" * 100


class FakeCurl:
    """Stands in for `subprocess.run`: serves `PAYLOAD` with range semantics and answers tree listings."""

    def __init__(self, listing=None, listing_exit=0):
        self.listing = listing
        self.listing_exit = listing_exit
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        url = command[-1]
        if "/api/models/" in url:
            stdout = json.dumps(self.listing) if self.listing is not None else ""
            return mock.Mock(returncode=self.listing_exit, stdout=stdout, stderr="")
        destination = command[command.index("-o") + 1]
        have = os.path.getsize(destination) if os.path.exists(destination) else 0
        if "--continue-at" in command and have >= len(PAYLOAD):
            return mock.Mock(returncode=fetch.CURL_RANGE_ALREADY_SATISFIED)
        with open(destination, "ab" if "--continue-at" in command else "wb") as handle:
            handle.write(PAYLOAD[have:] if "--continue-at" in command else PAYLOAD)
        return mock.Mock(returncode=0)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.destination = os.path.join(self.root.name, "model.safetensors")

    def tearDown(self):
        self.root.cleanup()

    def run_download(self, expected, curl=None):
        curl = curl or FakeCurl()
        with mock.patch.object(fetch.subprocess, "run", curl):
            return fetch.download("https://example.invalid/model.safetensors", self.destination, expected), curl

    def test_complete_file_with_known_size_is_skipped_without_a_request(self):
        write(self.destination, PAYLOAD)
        (ok, detail), curl = self.run_download(len(PAYLOAD))
        self.assertTrue(ok)
        self.assertEqual(detail, "cached")
        self.assertEqual(curl.calls, [])

    def test_partial_file_with_known_size_is_resumed(self):
        write(self.destination, PAYLOAD[:400])
        (ok, detail), curl = self.run_download(len(PAYLOAD))
        self.assertTrue(ok)
        self.assertEqual(detail, "downloaded")
        self.assertIn("--continue-at", curl.calls[0])
        with open(self.destination, "rb") as handle:
            self.assertEqual(handle.read(), PAYLOAD)

    def test_partial_file_with_unknown_size_is_resumed(self):
        write(self.destination, PAYLOAD[:400])
        (ok, _), _ = self.run_download(0)
        self.assertTrue(ok)
        self.assertEqual(os.path.getsize(self.destination), len(PAYLOAD))

    def test_complete_file_with_unknown_size_is_complete_on_range_satisfied(self):
        write(self.destination, PAYLOAD)
        (ok, detail), curl = self.run_download(0)
        self.assertTrue(ok)
        self.assertEqual(detail, "cached")
        self.assertEqual(len(curl.calls), 1)

    def test_range_satisfied_with_wrong_size_fails(self):
        write(self.destination, PAYLOAD + b"extra")
        (ok, detail), _ = self.run_download(len(PAYLOAD))
        self.assertFalse(ok)
        self.assertIn("33", detail)


class HubFileSizesTests(unittest.TestCase):
    LISTING = [
        {"type": "file", "path": "config.json", "size": 807},
        {"type": "file", "path": "model.safetensors", "size": 1502018592,
         "lfs": {"size": 1502018592, "pointerSize": 135}},
        {"type": "directory", "path": "assets"},
    ]

    def test_reads_sizes_by_path_over_ipv4(self):
        curl = FakeCurl(listing=self.LISTING)
        with mock.patch.object(fetch.subprocess, "run", curl):
            sizes = fetch.hub_file_sizes("facebook/vjepa2-vitl-fpc16-256-ssv2", "main")
        self.assertEqual(sizes, {"config.json": 807, "model.safetensors": 1502018592})
        self.assertIn("-4", curl.calls[0])
        self.assertTrue(curl.calls[0][-1].endswith("/tree/main?recursive=true"))

    def test_unavailable_listing_is_empty(self):
        with mock.patch.object(fetch.subprocess, "run", FakeCurl(listing_exit=22)):
            self.assertEqual(fetch.hub_file_sizes("some/repo", "main"), {})
        with mock.patch.object(fetch.subprocess, "run", FakeCurl(listing=None)):
            self.assertEqual(fetch.hub_file_sizes("some/repo", "main"), {})


class FetchReleaseTests(unittest.TestCase):
    RELEASE = {"key": "VJEPA2", "repo": "facebook/vjepa2-vitl-fpc16-256-ssv2",
               "directory": "vjepa2", "files": ["config.json", "model.safetensors"], "config": {}}

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.directory = os.path.join(self.root.name, "vjepa2")

    def tearDown(self):
        self.root.cleanup()

    def listing(self):
        return [{"type": "file", "path": name, "size": len(PAYLOAD)} for name in self.RELEASE["files"]]

    def test_partial_file_is_resumed_and_complete_file_is_skipped(self):
        write(os.path.join(self.directory, "config.json"), PAYLOAD)
        write(os.path.join(self.directory, "model.safetensors"), PAYLOAD[:835])
        curl = FakeCurl(listing=self.listing())
        with mock.patch.object(fetch.subprocess, "run", curl):
            failed = fetch.fetch_release(self.RELEASE, self.directory)
        self.assertEqual(failed, [])
        downloads = [call for call in curl.calls if "-o" in call]
        self.assertEqual([call[call.index("-o") + 1] for call in downloads],
                         [os.path.join(self.directory, "model.safetensors")])
        self.assertEqual(os.path.getsize(os.path.join(self.directory, "model.safetensors")), len(PAYLOAD))

    def test_partial_file_is_resumed_when_the_listing_is_unavailable(self):
        write(os.path.join(self.directory, "config.json"), PAYLOAD)
        write(os.path.join(self.directory, "model.safetensors"), PAYLOAD[:835])
        curl = FakeCurl(listing_exit=22)
        with mock.patch.object(fetch.subprocess, "run", curl):
            failed = fetch.fetch_release(self.RELEASE, self.directory)
        self.assertEqual(failed, [])
        for name in self.RELEASE["files"]:
            self.assertEqual(os.path.getsize(os.path.join(self.directory, name)), len(PAYLOAD))

    def test_revision_names_the_listing_and_the_files(self):
        release = dict(self.RELEASE, revision="5ca5edf5")
        curl = FakeCurl(listing=self.listing())
        with mock.patch.object(fetch.subprocess, "run", curl):
            fetch.fetch_release(release, self.directory)
        self.assertTrue(curl.calls[0][-1].endswith("/tree/5ca5edf5?recursive=true"))
        self.assertIn("/resolve/5ca5edf5/config.json", curl.calls[1][-1])

    def test_release_without_a_repo_reports_only_missing_files(self):
        release = {"key": "MPSENET", "git": "https://example.invalid/MP-SENet",
                   "directory": "mpsenet", "files": ["present", "absent"], "config": {}}
        write(os.path.join(self.directory, "present"), b"x")
        curl = FakeCurl()
        with mock.patch.object(fetch.subprocess, "run", curl):
            failed = fetch.fetch_release(release, self.directory)
        self.assertEqual(failed, [("absent", "no repo to fetch from")])
        self.assertEqual(curl.calls, [])


if __name__ == "__main__":
    unittest.main()
