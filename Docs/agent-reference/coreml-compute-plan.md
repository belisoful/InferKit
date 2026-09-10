<!-- Auxiliary reference for AGENTS.md / CLAUDE.md. This file holds the accumulated maintainer
notes that used to live inline in those files; the agent guidelines point here. Add new notes on
this subject to this file, not to AGENTS.md / CLAUDE.md. Keep the Documentation Style rules. -->

# Measuring where Core ML runs

`MLComputeUnits` is a request. Core ML places an operation the Neural Engine cannot run somewhere
else and reports nothing about having done so, so a model asked for the Neural Engine can run
entirely on the CPU and behave exactly as if it had not. `NFKComputePlan` answers that: it reads a
compiled model's plan per operation, without running it, and reports the counts per device plus
`operatorNamesOffNeuralEngine` — which is the list to work from when a conversion is being tuned,
since one unsupported operator in the middle of a network splits it and costs more than its own
share of the time.

It needs macOS 14.4 / iOS 17.4 / tvOS 17.4, which is where Core ML began publishing the information.
`isAvailable` reports whether the OS can answer, and an older system fails with
`kNFKError_InferenceUnsupported` rather than returning an empty plan, because zero operations on the
Neural Engine and "cannot tell" are different answers. `powermetrics --samplers ane_power` is the
runtime cross-check where the API is unavailable; it needs elevated privileges, which is why it is
documentation rather than code here.

What it says about this repository's own Core ML language model is bad news, and it is measured.
A GPT-2 converted by `Tools/inferkit-convert` places 0 of 448 operations on the Neural Engine —
all of them on the GPU, under `MLComputeUnits.ALL` — and the conversion emits `ANECCompile() FAILED`
into the middle of its ordinary output, which is the only warning there is. The Neural Engine is
reachable on the same machine: a plain attention block at sequence 64 places 100% of its operations
there. The transformer layout guidance did not help. The ordinary `(B, S, C)` + `nn.Linear` form
was already fully placed, and rewriting it into the 4-D `(B, C, 1, S)` 1×1-convolution form left
placement unchanged while taking the operation count from 40 to 313 — so the converter's ANE-friendly
rewrite is work with no measured benefit, and it is deliberately not done. What actually moves a
language model off the Neural Engine is still unidentified, and the single-token comparison in
`Tools/ane-placement/` does not settle it: both of those models landed on the CPU with four and eight
placed operations, which is Core ML declining to dispatch a trivial graph rather than evidence about
Neural Engine eligibility. That experiment is inconclusive, not negative.
The cause is now isolated, by adding one property at a time to a model that is fully placed
(`Tools/ane-placement/add_one_property.py`). Two facts, over the same twelve-layer 768-wide model:
a single-token forward is not placed on the Neural Engine — sequence 64 scores 100% and sequence 1
scores 0%, with the stateful cache innocent and the embedding gather costing four CPU operations — and
a multifunction package takes one placement decision, so the seq-64 prefill function that scores
100% alone drops to 0% when packaged with the seq-1 decode function. That is exactly what the
converter emits, and exactly why the whole thing runs on the GPU.
`ANECCompile() FAILED` was a red herring: it appeared once during conversion and does not reproduce.

Whether to act on it is a smaller question than it looks. Timed on the same models, prefill takes
3.77 ms on the Neural Engine against 4.98 ms on the GPU, and decode is unchanged — so splitting the
package into two models buys about 1.3× on time-to-first-token and nothing per token. Note also that a
compute plan reports the preferred device, not an execution trace: the seq-1 model is planned entirely
onto the GPU and still runs fastest under `ALL`. Placement is where Core ML intends to run something;
timing is what decides.

`coremltools` cannot be used for this. Its own `MLComputePlan` binding returns `None` for every
operation in 9.0 on macOS 26, including for models the Objective-C API reports on in the same session,
so a Python-side check reads as "nothing is on the Neural Engine" whatever the truth is.

`NFKCoreMLBackend` gained `computeUnits` to go with it (`NFKCoreMLLanguageBackend` already had one).
**`MLComputeUnitsCPUOnly` is zero**, so the property is initialized explicitly — an unset one would
quietly move every model to the CPU.
