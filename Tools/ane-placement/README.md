# ane-placement

Builds paired Core ML models that differ in exactly one property, so where Core ML places their
operations can be compared rather than assumed.

`MLComputeUnits` is a request. Core ML moves an operation the Neural Engine cannot run and reports
nothing about having done so, so the only way to know is to read the compute plan — and the only way
to know whether a change *helps* is to convert the same computation both ways.

## Running it

```bash
python3 build_fixture.py out/ --report            # layout: (B,S,C)+Linear against (B,C,1,S)+1x1 conv
python3 build_fixture.py out/ --stateful --report # a single step, with and without a Core ML state
```

Both write `.mlpackage` bundles. `--report` prints the plan through coremltools — **which currently
reports nothing**: `MLComputePlan.get_compute_device_usage_for_mlprogram_operation` returns `None` for
every operation in coremltools 9.0 on macOS 26. Take the numbers from the Objective-C side instead,
which works:

```bash
python3 -c "
import shutil; from coremltools.models.utils import compile_model
shutil.move(compile_model('out/naive.mlpackage'), 'out/naive.mlmodelc')"

INFERKIT_TEST_MLMODELC=out/naive.mlmodelc swift test --filter NFKComputePlanTests
```

That prints `NFKComputePlan.describedPlacement` and the operators that fell off the Neural Engine.

## What it found

Measured on an M1 Max, macOS 26, coremltools 9.0, `minimum_deployment_target=iOS18`, fp16:

| Model | Neural Engine share |
|---|---|
| Attention block, sequence 64, `(B, S, C)` with `nn.Linear` | 18/18 placed (100%) |
| The same block as `(B, C, 1, S)` with 1×1 convolutions | 137/137 placed (100%), 40 operations → 313 |
| A single-token step, stateless | 0% — CPU |
| A single-token step carrying a Core ML state | 0% — CPU |
| GPT-2 through `Tools/inferkit-convert` | 0/448 — GPU |

The layout rewrite that transformer-on-ANE guidance recommends **changed nothing** here: the ordinary
layout was already fully placed, and the rewrite cost eight times the operations for the same result.

A real language model is a different story — none of it lands on the Neural Engine.
`add_one_property.py` isolates why, by adding one candidate at a time to a GPT-2-shaped model that IS
fully placed:

| Variant | Neural Engine |
|---|---|
| 12 layers, 768 wide, sequence 64 | 265/265 (100%) |
| the same model at sequence 1 | 0/265 — all GPU |
| sequence 64 + embedding gather | 265 ANE, 4 CPU (98.5%) |
| sequence 1 + Core ML state | 0% |
| sequence 1 + embedding + state | 0% |
| sequence 64 and sequence 1 in one multifunction package | 0% |

**A single-token forward is not placed on the Neural Engine**, and **a multifunction package takes one
placement decision** — so the seq-64 prefill function that scores 100% alone loses the Neural Engine by
shipping with the seq-1 decode function. That is what `Tools/inferkit-convert` emits.

The stateful cache is innocent, which was not the expectation. `ANECCompile() FAILED` is a red herring:
it appeared once during conversion and does not reproduce on a fresh compile and load.

## Does it matter

Milliseconds per call, same models:

| Shape | `ALL` | `cpuAndGPU` | `cpuAndNeuralEngine` | `cpuOnly` |
|---|---|---|---|---|
| prefill, sequence 64 | 3.82 | 4.98 | **3.77** | 7.92 |
| decode, sequence 1 | 3.18 | 3.76 | 3.47 | 4.88 |

About **1.3x on prefill, nothing on decode**. Splitting the converter's package into two models would
buy that much time-to-first-token and no more.

Placement and timing measure different things: a compute plan reports the *preferred* device, not an
execution trace. The seq-1 model is planned entirely onto the GPU and still runs fastest under `ALL`.
Timing decides.

## The equivalence check

`build_fixture.py`'s `main` compares the two layouts' outputs before converting anything and exits if
they disagree. A placement comparison between two models that compute different functions measures
nothing, and the check has already earned itself: `nn.GroupNorm(1, C)` on a 4-D tensor looks like the
natural way to express `nn.LayerNorm(C)` and is a different function — one group normalizes over the
channels and the positions together, where a layer norm normalizes the channels separately at each
position.

Measured on an M1 Max, macOS 26, coremltools 9.0, fp16, `minimum_deployment_target=iOS18`.
