#!/bin/bash
#
# Removes the probe's launchd jobs and files. The results in /Users/Shared/inferkit-probe/ stay.
#
#   sudo Tools/background-probe/uninstall.sh
#
set -uo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
USER_NAME="${SUDO_USER:?run with sudo from the developer account}"
USER_ID="$(id -u "$USER_NAME")"
HOME_DIR="$(dscl . -read "/Users/$USER_NAME" NFSHomeDirectory | awk '{print $2}')"
launchctl bootout system/com.inferkit.probe.daemon 2>/dev/null
launchctl bootout system/com.inferkit.probe.daemon-user 2>/dev/null
launchctl bootout "gui/$USER_ID/com.inferkit.probe.agent" 2>/dev/null
launchctl bootout loginwindow/com.inferkit.probe.loginwindow 2>/dev/null
rm -f /Library/LaunchDaemons/com.inferkit.probe.daemon.plist /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist \
      /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist "$HOME_DIR/Library/LaunchAgents/com.inferkit.probe.agent.plist"
rm -rf "/Library/Application Support/InferKitProbe"
echo "==> removed; results stay in /Users/Shared/inferkit-probe/"
