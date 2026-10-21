#!/bin/bash
#
# Removes the probe's launchd jobs, the probe, mlx.metallib, and the release copy. The results and the
# stderr logs stay in /Library/Application Support/InferKitProbe/ and
# ~/Library/Application Support/InferKitProbe/.
#
#   sudo Tools/background-probe/uninstall.sh
#
set -uo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
USER_NAME="${SUDO_USER:?run with sudo from the developer account}"
USER_ID="$(id -u "$USER_NAME")"
HOME_DIR="$(dscl . -read "/Users/$USER_NAME" NFSHomeDirectory | awk '{print $2}')"
DEST="/Library/Application Support/InferKitProbe"
USER_DEST="$HOME_DIR/Library/Application Support/InferKitProbe"
launchctl bootout system/com.inferkit.probe.daemon 2>/dev/null
launchctl bootout system/com.inferkit.probe.daemon-user 2>/dev/null
launchctl bootout "gui/$USER_ID/com.inferkit.probe.agent" 2>/dev/null
# launchd runs a LoginWindow agent as root in the login window's GUI session, which gui/0 names, and
# unloads it when a user logs in. From a logged-in session this bootout finds nothing, and removing the
# plist below keeps the agent from loading at the next login window.
launchctl bootout gui/0/com.inferkit.probe.loginwindow 2>/dev/null
rm -f /Library/LaunchDaemons/com.inferkit.probe.daemon.plist /Library/LaunchDaemons/com.inferkit.probe.daemon-user.plist \
      /Library/LaunchAgents/com.inferkit.probe.loginwindow.plist
sudo -u "$USER_NAME" -- rm -f "$HOME_DIR/Library/LaunchAgents/com.inferkit.probe.agent.plist"
if [ -d "$DEST" ] && [ ! -L "$DEST" ] && [ "$(stat -f %u "$DEST")" -eq 0 ]; then
    rm -f "$DEST/inferkit-background-probe" "$DEST/mlx.metallib"
    rm -rf "$DEST/release"
elif [ -e "$DEST" ] || [ -L "$DEST" ]; then
    echo "$DEST is not a directory owned by root; its files stay" >&2
fi
echo "==> removed; results stay in $DEST/ and $USER_DEST/"
