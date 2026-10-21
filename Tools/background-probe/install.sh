#!/bin/bash
#
# Installs the probe and its four launchd jobs. Run with sudo from the developer's account:
#
#   sudo Tools/background-probe/install.sh [release directory to load, optional]
#
# - /Library/Application Support/InferKitProbe/: the probe, mlx.metallib, and the optional release.
# - com.inferkit.probe.agent: a LaunchAgent in the logged-in (Aqua) session.
# - com.inferkit.probe.loginwindow: a LaunchAgent at the login window, before anyone logs in.
# - com.inferkit.probe.daemon: a LaunchDaemon as root.
# - com.inferkit.probe.daemon-user: a LaunchDaemon as the developer's user, in the system domain.
# Each job runs at load and every two minutes, appending to /Users/Shared/inferkit-probe/results.jsonl.
#
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$ROOT/InferKitMLX/.build/background-probe"
USER_NAME="${SUDO_USER:?run with sudo from the developer account}"
USER_ID="$(id -u "$USER_NAME")"
HOME_DIR="$(dscl . -read "/Users/$USER_NAME" NFSHomeDirectory | awk '{print $2}')"
DEST="/Library/Application Support/InferKitProbe"
[ -x "$STAGE/inferkit-background-probe" ] || { echo "run Tools/background-probe/build.sh first" >&2; exit 1; }

mkdir -p "$DEST" /Users/Shared/inferkit-probe
chmod 1777 /Users/Shared/inferkit-probe
cp "$STAGE/inferkit-background-probe" "$STAGE/mlx.metallib" "$DEST/"
MODEL_ARGS=""
if [ -n "${1:-}" ]; then
    rm -rf "$DEST/release"
    cp -R "$1" "$DEST/release"
    MODEL_ARGS="<string>--model</string><string>$DEST/release</string>"
fi
chown -R root:wheel "$DEST"
chmod -R go-w "$DEST"

plist() { # label context extra-keys
    cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$1</string>
<key>ProgramArguments</key><array><string>$DEST/inferkit-background-probe</string><string>--context</string><string>$2</string>$MODEL_ARGS</array>
<key>RunAtLoad</key><true/>
<key>StartInterval</key><integer>120</integer>
<key>StandardErrorPath</key><string>/Users/Shared/inferkit-probe/$2.stderr.log</string>
$3
</dict></plist>
PLIST
}
plist com.inferkit.probe.agent agent-aqua "<key>LimitLoadToSessionType</key><string>Aqua</string>" \
    > "$HOME_DIR/Library/LaunchAgents/com.inferkit.probe.agent.plist"
chown "$USER_NAME" "$HOME_DIR/Library/LaunchAgents/com.inferkit.probe.agent.plist"
plist com.inferkit.probe.loginwindow agent-loginwindow "<key>LimitLoadToSessionType</key><string>LoginWindow</string>" \
    > /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist
plist com.inferkit.probe.daemon daemon-root "" > /Library/LaunchDaemons/com.inferkit.probe.daemon.plist
plist com.inferkit.probe.daemon-user daemon-user "<key>UserName</key><string>$USER_NAME</string>" \
    > /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist
chmod 644 /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist /Library/LaunchDaemons/com.inferkit.probe.daemon*.plist

launchctl bootstrap system /Library/LaunchDaemons/com.inferkit.probe.daemon.plist
launchctl bootstrap system /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist
launchctl bootstrap "gui/$USER_ID" "$HOME_DIR/Library/LaunchAgents/com.inferkit.probe.agent.plist" || true
echo "==> installed; the login-window agent loads at the next logout or restart"
echo "    results: /Users/Shared/inferkit-probe/results.jsonl"
