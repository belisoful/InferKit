#!/usr/bin/env python3
"""Exercises the release version guard in inferkit-version.sh against throwaway git trees.

    python3 Tools/xcframework/test_release_version.py

Each test builds a minimal tree in a temporary directory (NFKInferKit.m, InferKit.podspec, one commit)
and runs the shell functions on it. Nothing touches the repository this file lives in.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "inferkit-version.sh"
TIMEOUT = 30

INFERKIT_M = """#import "NFKInferKit.h"

@implementation NFKInferKit

+ (NSString *)releaseVersion
{
\treturn @"%s";
}

@end
"""

PODSPEC = """Pod::Spec.new do |s|
  s.name             = 'InferKit'
  s.version          = '%s'
end
"""


class ReleaseVersionTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="inferkit-release-version-")
        self.tree = Path(self.directory.name)
        self.git("init", "--quiet")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Release Version Test")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")

    def tearDown(self):
        self.directory.cleanup()

    def git(self, *arguments):
        subprocess.run(["git", "-C", str(self.tree), *arguments], check=True, timeout=TIMEOUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write_versions(self, source, podspec=None):
        sources = self.tree / "Sources" / "InferKit"
        sources.mkdir(parents=True, exist_ok=True)
        (sources / "NFKInferKit.m").write_text(INFERKIT_M % source)
        (self.tree / "InferKit.podspec").write_text(PODSPEC % (podspec or source))

    def commit(self, message="commit"):
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", message)

    def run_function(self, function):
        command = 'set -euo pipefail; source "$1"; %s "$2"' % function
        return subprocess.run(["bash", "-c", command, "bash", str(SCRIPT), str(self.tree)],
                              capture_output=True, text=True, timeout=TIMEOUT)

    def check(self):
        return self.run_function("inferkit_check_release_version")

    # The version sources

    def test_a_podspec_that_disagrees_with_the_source_is_refused(self):
        self.write_versions("0.4.0", podspec="0.3.1")
        self.commit()
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("s.version in InferKit.podspec is 0.3.1", result.stdout)

    def test_a_version_that_is_not_three_numbers_is_refused(self):
        self.write_versions("0.4")
        self.commit()
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not X.Y.Z", result.stdout)

    def test_a_tree_without_a_commit_is_refused(self):
        self.write_versions("0.4.0")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a git checkout", result.stdout)

    # The tags

    def test_a_version_newer_than_every_tag_passes(self):
        self.write_versions("0.3.1")
        self.commit()
        self.git("tag", "v0.3.1")
        self.write_versions("0.4.0")
        self.commit()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_version_an_older_commit_was_tagged_with_is_refused(self):
        self.write_versions("0.3.1")
        self.commit()
        self.git("tag", "v0.3.1")
        self.commit("later work without a version move")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("not newer than the existing tag v0.3.1", result.stdout)

    def test_a_version_older_than_the_latest_tag_is_refused(self):
        self.write_versions("0.3.1")
        self.commit()
        self.git("tag", "v0.10.0")
        self.commit("later work without a version move")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("not newer than the existing tag v0.10.0", result.stdout)

    def test_the_tagged_commit_of_the_version_rebuilds(self):
        self.write_versions("0.3.1")
        self.commit()
        self.git("tag", "v0.3.1")
        self.write_versions("0.4.0")
        self.commit()
        self.git("tag", "v0.4.0")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_a_tag_on_head_that_names_another_version_is_refused(self):
        self.write_versions("0.4.0")
        self.commit()
        self.git("tag", "v0.5.0")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("HEAD carries the tag v0.5.0", result.stdout)

    def test_tags_that_are_not_releases_are_ignored(self):
        self.write_versions("0.4.0")
        self.commit()
        self.git("tag", "v0.9.0-rc1-elsewhere")
        self.commit("next")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)

    # Local changes

    def test_an_untracked_file_counts_as_a_local_change(self):
        self.write_versions("0.4.0")
        (self.tree / ".gitignore").write_text("ignored/\n")
        self.commit()
        self.assertNotEqual(self.run_function("inferkit_tree_has_local_changes").returncode, 0)
        (self.tree / "ignored").mkdir()
        (self.tree / "ignored" / "build.log").write_text("ignored output\n")
        self.assertNotEqual(self.run_function("inferkit_tree_has_local_changes").returncode, 0)
        (self.tree / "Sources" / "InferKit" / "NFKNew.m").write_text("// globbed by SwiftPM\n")
        self.assertEqual(self.run_function("inferkit_tree_has_local_changes").returncode, 0)

    def test_an_edited_tracked_file_counts_as_a_local_change(self):
        self.write_versions("0.4.0")
        self.commit()
        (self.tree / "InferKit.podspec").write_text(PODSPEC % "0.4.0" + "# edited\n")
        self.assertEqual(self.run_function("inferkit_tree_has_local_changes").returncode, 0)


if __name__ == "__main__":
    os.environ.pop("GIT_DIR", None)
    unittest.main()
