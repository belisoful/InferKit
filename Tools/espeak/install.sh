#!/usr/bin/env bash
#
# install.sh — install espeak-ng, the phonemizer InferKit's optional espeak path uses.
#
# InferKit does NOT bundle espeak-ng (it is GPLv3; the toolkit is MIT). This script externalizes the
# dependency: it installs espeak-ng onto your system so `NFKMLXEspeakPhonemizer` can use it when
# present. The in-toolkit neural G2P (`NFKMLXNeuralG2P`) needs none of this and ships with the toolkit.
#
# Usage:  Tools/espeak/install.sh
#
set -euo pipefail

if command -v espeak-ng >/dev/null 2>&1; then
    echo "espeak-ng already installed: $(espeak-ng --version 2>/dev/null | head -1)"
    exit 0
fi

case "$(uname -s)" in
    Darwin)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Homebrew not found. Install it from https://brew.sh, then re-run this script." >&2
            exit 1
        fi
        brew install espeak-ng
        ;;
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y espeak-ng
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y espeak-ng
        else
            echo "No supported package manager (apt-get/dnf). Install espeak-ng manually." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported OS: $(uname -s). Install espeak-ng manually." >&2
        exit 1
        ;;
esac

echo "Installed: $(espeak-ng --version 2>/dev/null | head -1)"
echo "Optional (Python tooling): pip install phonemizer"
