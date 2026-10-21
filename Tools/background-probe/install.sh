#!/bin/bash
#
# Installs the probe and its four launchd jobs. Run with sudo from the developer's account:
#
#   sudo Tools/background-probe/install.sh [--max-runs N] [release directory to load]
#
# - /Library/Application Support/InferKitProbe/ (root:wheel, 0755): the probe, mlx.metallib, the
#   optional release, and the output of the two jobs that run as root.
# - ~/Library/Application Support/InferKitProbe/ (the developer's, 0755): the output of the two jobs
#   that run as the developer.
# - com.inferkit.probe.agent: a LaunchAgent in the logged-in (Aqua) session, as the developer.
# - com.inferkit.probe.loginwindow: a LaunchAgent at the login window, which launchd runs as root.
# - com.inferkit.probe.daemon: a LaunchDaemon as root.
# - com.inferkit.probe.daemon-user: a LaunchDaemon as the developer, in the system domain.
# Each job runs at load and every two minutes until uninstalled, appending to results.jsonl in its
# output directory. A job does no work once that file holds N runs of its context (default 30).
#
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
MAX_RUNS=30
if [ "${1:-}" = "--max-runs" ]; then
    MAX_RUNS="${2:-}"
    shift 2 || true
fi
[[ "$MAX_RUNS" =~ ^[0-9]+$ ]] || { echo "--max-runs takes a count" >&2; exit 1; }
RELEASE="${1:-}"
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$PACKAGE/.build/stage"
USER_NAME="${SUDO_USER:?run with sudo from the developer account}"
USER_ID="$(id -u "$USER_NAME")"
HOME_DIR="$(dscl . -read "/Users/$USER_NAME" NFSHomeDirectory | awk '{print $2}')"
DEST="/Library/Application Support/InferKitProbe"
USER_DEST="$HOME_DIR/Library/Application Support/InferKitProbe"
USER_AGENTS="$HOME_DIR/Library/LaunchAgents"
[ -x "$STAGE/inferkit-background-probe" ] || { echo "run Tools/background-probe/build.sh first" >&2; exit 1; }

as_user() {
    sudo -u "$USER_NAME" -- "$@"
}

# Root writes only into a directory root owns. A symbolic link or a directory another user owns at
# that path stops the install.
[ ! -L "$DEST" ] || { echo "$DEST is a symbolic link; inspect and remove it, then run again" >&2; exit 1; }
[ -e "$DEST" ] || mkdir -m 0755 "$DEST"
if [ -L "$DEST" ] || [ ! -d "$DEST" ] || [ "$(stat -f %u "$DEST")" -ne 0 ]; then
    echo "$DEST must be a directory owned by root" >&2
    exit 1
fi
chown root:wheel "$DEST"
chmod 0755 "$DEST"
cp "$STAGE/inferkit-background-probe" "$STAGE/mlx.metallib" "$DEST/"
MODEL_ARGS=""
if [ -n "$RELEASE" ]; then
    rm -rf "$DEST/release"
    # -L copies the files a link names, so the root-owned copy holds no link back into the developer's
    # tree, and a Hugging Face snapshot's links into its blob store resolve.
    cp -RL "$RELEASE" "$DEST/release"
    MODEL_ARGS="<string>--model</string><string>$DEST/release</string>"
fi
chown -R root:wheel "$DEST"
chmod -R go-w "$DEST"

# The developer's directories are created and written as the developer.
[ ! -L "$USER_DEST" ] || { echo "$USER_DEST is a symbolic link; inspect and remove it, then run again" >&2; exit 1; }
as_user mkdir -p "$USER_AGENTS" "$USER_DEST"
as_user chmod 0755 "$USER_DEST"

plist() { # label context output-directory extra-keys
    cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$1</string>
<key>ProgramArguments</key><array><string>$DEST/inferkit-background-probe</string><string>--context</string><string>$2</string><string>--output-directory</string><string>$3</string><string>--max-runs</string><string>$MAX_RUNS</string>$MODEL_ARGS</array>
<key>RunAtLoad</key><true/>
<key>StartInterval</key><integer>120</integer>
$4
</dict></plist>
PLIST
}
plist com.inferkit.probe.agent agent-aqua "$USER_DEST" "<key>LimitLoadToSessionType</key><string>Aqua</string>" \
    | as_user tee "$USER_AGENTS/com.inferkit.probe.agent.plist" > /dev/null
plist com.inferkit.probe.loginwindow agent-loginwindow "$DEST" "<key>LimitLoadToSessionType</key><string>LoginWindow</string>" \
    > /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist
plist com.inferkit.probe.daemon daemon-root "$DEST" "" > /Library/LaunchDaemons/com.inferkit.probe.daemon.plist
plist com.inferkit.probe.daemon-user daemon-user "$USER_DEST" "<key>UserName</key><string>$USER_NAME</string>" \
    > /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist
chmod 644 /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist /Library/LaunchDaemons/com.inferkit.probe.daemon*.plist

launchctl bootstrap system /Library/LaunchDaemons/com.inferkit.probe.daemon.plist
launchctl bootstrap system /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist
launchctl bootstrap "gui/$USER_ID" "$USER_AGENTS/com.inferkit.probe.agent.plist" || true
echo "==> installed; the login-window agent loads at the next logout or restart"
echo "    each job runs every two minutes until uninstalled, and stops working after $MAX_RUNS runs"
echo "    results: $DEST/results.jsonl (agent-loginwindow, daemon-root)"
echo "             $USER_DEST/results.jsonl (agent-aqua, daemon-user)"
