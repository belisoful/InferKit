# Background probe

Measures whether Metal and MLX run in the launchd contexts a background inference service uses. The
question it answers: can a service compute on the GPU from a LaunchAgent before anyone logs in, and
from a LaunchDaemon with no user session at all?

`inferkit-background-probe` is the executable of the Swift package in this folder, which depends on
InferKit and InferKitMLX by path. Neither of those packages builds it, and it ships in no distribution.
Each run appends two JSON lines to `results.jsonl` in its job's output directory:

- `phase: session`: the uid, the security session's attributes (`session_has_graphic_access`,
  `session_is_root`), the console user (`null` at the login window), the default Metal device and every
  device `MTLCopyAllDevices` lists, and whether MLX found its Metal library.
- `phase: mlx`: a 1024 × 1024 matmul's time and trace, a decoder's forward pass on initial values, and,
  when the install named a release, the time and text of 8 greedy tokens from it.

A run that dies inside Metal or MLX leaves only its `session` line. Its stderr goes to
`<context>.stderr.log` in the same directory.

The output directories:

| Directory | Owner | Jobs |
| --- | --- | --- |
| `/Library/Application Support/InferKitProbe/` | `root:wheel`, 0755 | `agent-loginwindow`, `daemon-root` |
| `~/Library/Application Support/InferKitProbe/` | the developer, 0755 | `agent-aqua`, `daemon-user` |

The probe refuses an output directory that is a symbolic link, belongs to another user, or is writable
by group or others. It opens its files with `O_NOFOLLOW` and creates them with mode 0644.

## Run cap

Each job runs at load and every two minutes until uninstalled. A run that finds 30 runs of its context
in `results.jsonl` exits without work, so each job stops computing after about an hour of runs. The
install takes `--max-runs N` to change the count. Uninstall as soon as the results are collected: the
jobs stay loaded and start every two minutes until then, as root and as the developer.

## Run sheet

It needs an administrator at the keyboard, and one restart. Do steps 3 to 5 within an hour of the
install, or the daemons reach their cap before the login-window phase.

1. Build: `Tools/background-probe/build.sh`. This needs no administrator, and stages the probe and
   `mlx.metallib` in `Tools/background-probe/.build/stage/`.
2. Install: `sudo Tools/background-probe/install.sh [--max-runs N] /path/to/a/release`. Both
   arguments are optional. Name a small dense release directory the language loader reads, such as a
   copy of `Qwen/Qwen3-0.6B`, which is copied beside the probe. The script installs four jobs:
   - `agent-aqua`: a LaunchAgent in the logged-in session. This is the control.
   - `agent-loginwindow`: a LaunchAgent at the login window. launchd runs it as root.
   - `daemon-root`: a LaunchDaemon as root.
   - `daemon-user`: a LaunchDaemon as the developer's user, in the system domain.
3. Wait two minutes logged in. `agent-aqua`, `daemon-root`, and `daemon-user` each record a run.
4. Restart. At the login window, wait four minutes without logging in, so `agent-loginwindow`,
   `daemon-root`, and `daemon-user` each run with no user session.
5. Log in, then read the results:
   - `cat "/Library/Application Support/InferKitProbe/results.jsonl"`
   - `cat ~/Library/Application\ Support/InferKitProbe/results.jsonl`
   - any `*.stderr.log` beside either file.
6. Remove the jobs: `sudo Tools/background-probe/uninstall.sh`. The results and logs stay.

Each line's `console_user` tells the two daemon phases apart: `null` or `loginwindow` before login,
and the user's name after. A job that exits before the probe opens its log leaves its last exit status
in `launchctl print system/com.inferkit.probe.daemon` (or the job's own domain and label).
