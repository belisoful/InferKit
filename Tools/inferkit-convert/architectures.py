"""Per-architecture stateful KV-cache adapters for inferkit-convert.

A central registry maps a Hugging Face ``model_type`` to an ``Architecture`` that knows how to wrap a
causal LM with a Core ML stateful KV cache. Support a new family by writing an ``Architecture``
subclass and registering its ``model_types``.

Each architecture derives its cache shape from the model config, declares the Core ML state and the
input/logits features, and builds the tracing wrapper. The runtime feeds ``input_ids`` and, when the
architecture sets ``position_feature``, a 1-D ``cache_position`` (the absolute index of each new
token). The causal mask is derived inside the model from ``cache_position``, so the runtime supplies
no mask.

torch, numpy, and coremltools are imported lazily inside the methods that need them, so the registry
and its shape math can be inspected without those packages.

Verification note: the graph-level cache wiring in ``build_wrapper`` targets the coremltools
stateful-models path (a fixed cache updated by ``cache_position`` and exposed as Core ML state). It
must be validated against coremltools with real weights; the shape and feature contracts here are
what the runtime relies on.
"""

REGISTRY = {}


def register(architecture):
    """Registers an Architecture instance under each of its model_types."""
    for model_type in architecture.model_types:
        REGISTRY[model_type] = architecture
    return architecture


def resolve(model_type):
    """Returns the Architecture for a model_type, or raises KeyError with the supported list."""
    architecture = REGISTRY.get(model_type)
    if architecture is None:
        raise KeyError(f"no architecture for model_type '{model_type}'; supported: {supported_types()}")
    return architecture


def supported_types():
    return sorted(REGISTRY.keys())


class Architecture:
    """Base adapter. Subclasses set model_types and, when needed, override the cache wiring."""

    model_types = ()
    input_feature = "input_ids"
    logits_feature = "logits"
    position_feature = "cache_position"
    state_names = ("k_cache", "v_cache")

    def describe(self, config):
        """The layer count, KV-head count, and head dimension the cache shape needs."""
        kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        return {"layers": config.num_hidden_layers, "kv_heads": kv_heads, "head_dim": head_dim}

    def cache_shape(self, config, context_length):
        dimensions = self.describe(config)
        return (dimensions["layers"], 1, dimensions["kv_heads"], context_length, dimensions["head_dim"])

    def state_types(self, config, context_length):
        import numpy as np
        import coremltools as ct
        shape = self.cache_shape(config, context_length)
        return [ct.StateType(wrapped_type=ct.TensorType(shape=shape, dtype=np.float16), name=name)
                for name in self.state_buffer_names()]

    def state_buffer_names(self):
        """The buffer names coremltools matches StateType against; the qualified named_buffers() path."""
        return list(self.state_names)

    def input_types(self, context_length):
        import numpy as np
        import coremltools as ct
        sequence = ct.RangeDim(lower_bound=1, upper_bound=context_length, default=1)
        inputs = [ct.TensorType(name=self.input_feature, shape=(1, sequence), dtype=np.int32)]
        if self.position_feature is not None:
            inputs.append(ct.TensorType(name=self.position_feature, shape=(sequence,), dtype=np.int32))
        return inputs

    def output_types(self):
        import numpy as np
        import coremltools as ct
        return [ct.TensorType(name=self.logits_feature, dtype=np.float16)]

    def manifest_section(self):
        section = {"inputFeature": self.input_feature,
                   "logitsFeature": self.logits_feature,
                   "stateNames": list(self.state_names)}
        if self.position_feature is not None:
            section["positionFeature"] = self.position_feature
        return section

    def build_wrapper(self, model, config, context_length):
        """Returns an nn.Module accepting (input_ids, cache_position) and returning logits."""
        raise NotImplementedError


class _StaticCacheArchitecture(Architecture):
    """Shared wiring for decoders that accept a transformers Cache and a cache_position.

    Covers the decoder-only families whose attention transformers drives through the Cache interface
    (including GPT-2 in recent transformers); the subclass only names the model_types and, where the
    config field names differ, overrides describe().

    Version note: the cache plugs into the transformers Cache interface, which changed across the
    ~4.54 redesign. The wrapper supports both: it owns the cache buffers (the 4.54+ Cache is not an
    nn.Module, so its own tensors would not reach coremltools state), overrides Cache.update (which
    keeps its layer_idx across the redesign), and handles either constructor signature. Validated
    numerically faithful on transformers 4.46 and 4.57.
    """

    # Buffers live on the wrapper (an nn.Module), so coremltools sees k_cache / v_cache directly.
    def state_buffer_names(self):
        return list(self.state_names)

    # A fixed single-token graph: the static cache + scatter update need static shapes to convert, so
    # the runtime feeds one token at a time (prefill included). cache_position places it in the cache.
    def input_types(self, context_length):
        import numpy as np
        import coremltools as ct
        return [ct.TensorType(name=self.input_feature, shape=(1, 1), dtype=np.int32),
                ct.TensorType(name=self.position_feature, shape=(1,), dtype=np.int32)]

    def build_wrapper(self, model, config, context_length):
        import torch
        from transformers.cache_utils import Cache

        cache_shape = self.cache_shape(config, context_length)
        cache_length = cache_shape[3]

        # The Cache holds no buffers of its own; the wrapper owns them. update() is overridden, so the
        # per-layer machinery is unused. Cache.update keeps layer_idx across the ~4.54 redesign, so one
        # override serves both APIs; only the constructor signature differs (old: no args; new: needs
        # layer_class_to_replicate). The wrapper reference is set outside nn.Module tracking to avoid a
        # cycle when the old Cache (an nn.Module) is held by the wrapper.
        class SliceUpdateCache(Cache):
            def __init__(self, owner):
                try:
                    super().__init__()					# pre-4.54 Cache API
                except (TypeError, ValueError):
                    from transformers.cache_utils import StaticLayer
                    super().__init__(layer_class_to_replicate=StaticLayer)	# 4.54+ redesign
                object.__setattr__(self, "_owner", owner)

            def update(self, key, value, layer_idx, cache_kwargs=None):
                # Scatter the new keys/values into the fixed-length cache at cache_position and return
                # the whole cache. Attention masks the unwritten positions (from cache_position), so the
                # graph is fixed-shape and uses coremltools-supported ops (scatter, not index_copy).
                position = cache_kwargs["cache_position"]
                index = position.view(1, 1, -1, 1).expand_as(key).to(torch.long)
                updated_k = self._owner.k_cache[layer_idx].to(torch.float32).scatter(2, index, key.to(torch.float32))
                updated_v = self._owner.v_cache[layer_idx].to(torch.float32).scatter(2, index, value.to(torch.float32))
                self._owner.k_cache[layer_idx] = updated_k.to(torch.float16)
                self._owner.v_cache[layer_idx] = updated_v.to(torch.float16)
                return (updated_k.to(key.dtype), updated_v.to(value.dtype))

            def get_seq_length(self, layer_idx=0):
                return int(self._owner.k_cache.shape[3])	# full cache length; the mask handles validity

            def get_mask_sizes(self, cache_position, layer_idx=0):
                return int(self._owner.k_cache.shape[3]), 0

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=torch.float16))
                self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=torch.float16))
                self.cache = SliceUpdateCache(self)

            def forward(self, input_ids, cache_position):
                # The static cache holds unwritten (future) positions; build the causal mask from
                # cache_position so attention ignores them. Per query row i, a key position is
                # visible when it is at or before cache_position[i], which is causal for both a
                # single decode token and a multi-token prefill chunk.
                key_positions = torch.arange(cache_length, device=cache_position.device)
                visible = key_positions.view(1, -1) <= cache_position.view(-1, 1)
                mask = torch.where(visible, 0.0, float("-inf")).view(1, 1, -1, cache_length)
                outputs = self.inner(
                    input_ids=input_ids.long(),
                    cache_position=cache_position.long(),
                    attention_mask=mask,
                    past_key_values=self.cache,
                    use_cache=True,
                )
                return outputs.logits

        return Wrapper(model).eval()


class RoPEDecoderArchitecture(_StaticCacheArchitecture):
    """Rotary-embedding dense decoder-only transformers (Cache + cache_position).

    Mixture-of-experts variants (e.g. qwen2_moe) are excluded: their sparse expert routing
    (dynamic per-expert dispatch, unbind over a symbolic count) does not lower to a static Core ML
    graph. A dense-expert rewrite would be needed, which is out of scope here.
    """

    model_types = ("llama", "qwen2", "mistral", "gemma", "gemma2",
                   "phi3", "stablelm", "starcoder2")


class GPT2Architecture(_StaticCacheArchitecture):
    """GPT-2 family (byte-level BPE). Recent transformers expose GPT-2 through the Cache interface,
    so it shares the static-cache wrapper; only the config field names differ."""

    model_types = ("gpt2",)

    def describe(self, config):
        heads = config.n_head
        return {"layers": config.n_layer, "kv_heads": heads, "head_dim": config.n_embd // heads}


register(RoPEDecoderArchitecture())
register(GPT2Architecture())
