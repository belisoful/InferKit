"""Tests the chat-template extraction parsing against a fake ChatML tokenizer (no transformers)."""

import unittest

import convert


class _ChatMLTokenizer:
    """Mimics a ChatML instruct tokenizer's apply_chat_template, including a default system message."""

    chat_template = "chatml"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = ""
        if not any(m["role"] == "system" for m in messages):
            rendered += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class _PlainTokenizer:
    chat_template = None


class ChatTemplateExtractionTests(unittest.TestCase):

    def test_extracts_chatml_markers(self):
        descriptor = convert._extract_chat_template(_ChatMLTokenizer())
        self.assertEqual(descriptor["user"], ["<|im_start|>user\n", "<|im_end|>\n"])
        self.assertEqual(descriptor["assistant"], ["<|im_start|>assistant\n", "<|im_end|>\n"])
        self.assertEqual(descriptor["system"], ["<|im_start|>system\n", "<|im_end|>\n"])
        self.assertEqual(descriptor["generationPrompt"], "<|im_start|>assistant\n")
        self.assertEqual(descriptor["defaultSystem"], "You are a helpful assistant.")

    def test_no_template_returns_none(self):
        self.assertIsNone(convert._extract_chat_template(_PlainTokenizer()))


if __name__ == "__main__":
    unittest.main()
