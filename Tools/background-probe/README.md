# Background probe

Measures whether Metal and MLX run in the launchd contexts a background inference service uses. The
question it answers: can a service compute on the GPU from a LaunchAgent before anyone logs in, and
from a LaunchDaemon with no user session at all?

`inferkit-background-probe` (an executable target of InferKitMLX, shipped in no product) appends two
JSON lines per run to `/Users/Shared/inferkit-probe/results.jsonl`:

- `phase: session`: the uid, the security session's attributes (`session_has_graphic_access`,
  `session_is_root`), the console user (`null` at the login window), the default Metal device and every
  device `MTLCopyAllDevices` lists, and whether MLX found its Metal library.
- `phase: mlx`: a 1024 × 1024 matmul's time and trace, a decoder's forward pass on initial values, and,
  when the install named a release, the time and text of 8 greedy tokens from it.

A run that dies inside Metal or MLX leaves only its `session` line, and its stderr goes to
`/Users/Shared/inferkit-probe/<context>.stderr.log`.

## Run sheet

It needs an administrator at the keyboard, and one restart.

1. Build: `Tools/background-probe/build.sh`. This needs no administrator.
2. Install: `sudo Tools/background-probe/install.sh /path/to/a/release`. The release argument is
   optional. Name a small dense release directory the language loader reads, such as a copy of
   `Qwen/Qwen3-0.6B`, which is copied beside the probe. The script installs four jobs, each running at
   load and every two minutes:
   - `agent-aqua`: a LaunchAgent in the logged-in session. This is the control.
   - `agent-loginwindow`: a LaunchAgent at the login window.
   - `daemon-root`: a LaunchDaemon as root.
   - `daemon-user`: a LaunchDaemon as the developer's user, in the system domain.
3. Wait two minutes logged in. `agent-aqua`, `daemon-root`, and `daemon-user` each record a run.
4. Restart. At the login window, wait four minutes without logging in, so `agent-loginwindow`,
   `daemon-root`, and `daemon-user` each run with no user session.
5. Log in, then read the results:
   `cat /Users/Shared/inferkit-probe/results.jsonl`, plus any `*.stderr.log` beside it.
6. Remove the jobs: `sudo Tools/background-probe/uninstall.sh`. The results stay.

Each line's `console_user` tells the two daemon phases apart: `null` or `loginwindow` before login,
and the user's name after.
