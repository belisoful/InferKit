#!/usr/bin/env python3
"""Every seam of a dense mlx_lm decoder's forward, recorded at the stored precision.

The release loader's run at the stored precision is held to mlx_lm (`mlx_lm_logits.py`). Where the two
part, this record localizes it: the forward is rebuilt from the model's own modules and mlx_lm's own
attention and activation functions, and each intermediate is saved in the type mlx_lm computes it in.
The rebuilt logits must equal `model(tokens)` bit for bit, or the script refuses to write the record.

Run under the `mlxlm` interpreter of Tools/validation-assets/manifest.json:

    mlx_lm_seams.py <output.safetensors> --checkpoint <release directory> --record <mlx_quantized_release record>

Keys, each `[1, length, …]` unless noted:

- `tokens`, `embedding`, `norm` (the final norm's output), and `output` (the logits, float32).
- `layers.<i>.<seam>`: `input`, `input_norm`, `q_proj`, `k_proj`, `v_proj`, `q_norm` and `k_norm` (where the
  family normalizes each head; `[1, length, heads, head_dim]`), `q_rope` and `k_rope` (`[1, heads, length,
  head_dim]`), `attention` (the attention output before `o_proj`, `[1, length, heads · head_dim]`), `o_proj`,
  `attended` (the residual after attention), `post_norm`, `gate`, `up`, `swiglu`, `down`, and `output`.
"""

import argparse
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention


def attention(module, x, mask, seams):
    batch, length, _ = x.shape
    queries, keys, values = module.q_proj(x), module.k_proj(x), module.v_proj(x)
    seams.update(q_proj=queries, k_proj=keys, v_proj=values)
    queries = queries.reshape(batch, length, module.n_heads, -1)
    keys = keys.reshape(batch, length, module.n_kv_heads, -1)
    if hasattr(module, "q_norm"):
        queries, keys = module.q_norm(queries), module.k_norm(keys)
        seams.update(q_norm=queries, k_norm=keys)
    queries = module.rope(queries.transpose(0, 2, 1, 3))
    keys = module.rope(keys.transpose(0, 2, 1, 3))
    values = values.reshape(batch, length, module.n_kv_heads, -1).transpose(0, 2, 1, 3)
    seams.update(q_rope=queries, k_rope=keys)
    output = scaled_dot_product_attention(queries, keys, values, cache=None, scale=module.scale, mask=mask)
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    seams["attention"] = output
    projected = module.o_proj(output)
    seams["o_proj"] = projected
    return projected


def block(layer, x, mask, seams):
    seams["input"] = x
    normed = layer.input_layernorm(x)
    seams["input_norm"] = normed
    attended = x + attention(layer.self_attn, normed, mask, seams)
    seams["attended"] = attended
    normed = layer.post_attention_layernorm(attended)
    seams["post_norm"] = normed
    mlp = layer.mlp
    gate, up = mlp.gate_proj(normed), mlp.up_proj(normed)
    activated = swiglu(gate, up)
    down = mlp.down_proj(activated)
    seams.update(gate=gate, up=up, swiglu=activated, down=down)
    output = attended + down
    seams["output"] = output
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="path to write the .safetensors record")
    parser.add_argument("--checkpoint", required=True, help="the release directory")
    parser.add_argument("--record", required=True, help="the mlx_quantized_release record whose tokens to read")
    args = parser.parse_args()

    tokens = mx.load(args.record)["tokens"].astype(mx.int32)[None]
    model, _ = load(args.checkpoint)
    expected = model(tokens).astype(mx.float32)

    inner = model.model
    hidden = inner.embed_tokens(tokens)
    record = {"tokens": tokens[0], "embedding": hidden}
    mask = create_attention_mask(hidden, None)
    for index, layer in enumerate(inner.layers):
        seams = {}
        hidden = block(layer, hidden, mask, seams)
        record.update({f"layers.{index}.{name}": value for name, value in seams.items()})
    hidden = inner.norm(hidden)
    record["norm"] = hidden
    if getattr(model.args, "tie_word_embeddings", False) or not hasattr(model, "lm_head"):
        logits = inner.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)
    logits = logits.astype(mx.float32)
    record["output"] = logits[0]
    mx.eval(expected, record)

    if not mx.array_equal(logits, expected).item():
        difference = mx.abs(logits - expected).max().item()
        print(f"the rebuilt forward differs from model(tokens) by up to {difference}; no record written")
        return 1
    mx.save_safetensors(args.output, record)
    print(f"wrote {args.output}: {len(inner.layers)} layers, {len(record)} arrays, rebuilt logits equal model(tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
