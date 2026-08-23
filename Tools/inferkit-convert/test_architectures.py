"""Registry tests that need no torch / coremltools (shape math and dispatch only)."""

import unittest

import architectures


class _FakeConfig:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class ArchitectureRegistryTests(unittest.TestCase):

    def test_known_types_resolve_to_an_architecture(self):
        for model_type in ("llama", "qwen2", "mistral", "gpt2"):
            self.assertIn(model_type, architectures.supported_types())
            self.assertIsNotNone(architectures.resolve(model_type))

    def test_an_unknown_type_raises_with_the_supported_list(self):
        with self.assertRaises(KeyError) as raised:
            architectures.resolve("no-such-arch")
        self.assertIn("supported", str(raised.exception))

    def test_rope_cache_shape_follows_the_config(self):
        architecture = architectures.resolve("qwen2")
        config = _FakeConfig(num_hidden_layers=4, num_attention_heads=8,
                             num_key_value_heads=2, hidden_size=64)
        # (layers, batch, kv_heads, context, head_dim=hidden/heads=8)
        self.assertEqual(architecture.cache_shape(config, 128), (4, 1, 2, 128, 8))

    def test_rope_declares_a_cache_position_input(self):
        architecture = architectures.resolve("llama")
        section = architecture.manifest_section()
        self.assertEqual(section["positionFeature"], "cache_position")
        self.assertEqual(section["inputFeature"], "input_ids")
        self.assertEqual(section["stateNames"], ["k_cache", "v_cache"])

    def test_gpt2_shape_uses_its_config_field_names(self):
        architecture = architectures.resolve("gpt2")
        config = _FakeConfig(n_layer=3, n_embd=48, n_head=6)
        self.assertEqual(architecture.cache_shape(config, 64), (3, 1, 6, 64, 8))


if __name__ == "__main__":
    unittest.main()
