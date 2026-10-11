#!/usr/bin/env python3
"""Generate the release chat-template reference, tool conversations included.

The chat-template renderer (`NFKMLXChatTemplateRenderer`) is held to transformers' own renderer over
the templates the releases ship. This reads every release directory the validation config names
(`IK_VAL_*` keys), takes each distinct `chat_template`, and records what transformers renders for:

- `chat`: a system turn, a user turn, an assistant turn, and a user turn.
- `tools_declared`: two tools declared, a system turn, and a user turn (templates that read `tools`).
- `tool_conversation`: the tools, a user turn, an assistant turn calling them, and their answers, in
  the shape `NFKMLXToolCalling.templateMessages` prepares: each call's arguments an object in the
  order the model wrote them, each answer carrying its tool's name, one call per message for
  harmony, whose template renders only the first.

A template that refuses a case (a system turn it does not take, content it reads as parts) is retried
with the content as text parts; a case it still refuses is recorded with its error, which the Swift
test expects as a throw. `strftime_now` reports 2026-01-15 09:30, which the Swift test passes as the
date. The special tokens of each release's `tokenizer_config.json` are bound as transformers binds
them. Point `IK_CHAT_TEMPLATE_TOOLS_REF` at the output.

Run under the LLM oracle interpreter (transformers installed):
    ~/.inferkit-validation/llmvenv/bin/python3 -I Tools/reference-parity/generate_tool_chat_templates.py \
        [output.json]
"""
import copy
import datetime as datetime_module
import json
import os
import sys

from transformers.utils import chat_template_utils

FIXED_NOW = datetime_module.datetime(2026, 1, 15, 9, 30)


class FixedDatetime(datetime_module.datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


chat_template_utils.datetime = FixedDatetime

# Each key in the order NFKJinjaKeyOrder writes a declaration, so a Foundation dictionary renders it
# the same way.
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather in a city. Use it when the user asks about the weather now. Returns the temperature and the sky.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "The city name."},
                                      "unit": {"type": "string", "description": "celsius or fahrenheit.",
                                               "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "speak",
        "description": "Generates speech from text. Returns the audio's length.",
        "parameters": {"type": "object",
                       "properties": {"durationSeconds": {"type": "number", "description": "Optional length, in seconds."},
                                      "prompt": {"type": "string", "description": "The words to speak."}},
                       "required": ["prompt"]}}},
]

CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello there."},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Name a color."},
]

DECLARED = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris, and say it aloud?"},
]


def tool_conversation(one_call_per_message):
    weather = {"id": "call_1", "type": "function",
               "function": {"name": "get_weather", "arguments": {"unit": "celsius", "city": "Paris"}}}
    speak = {"id": "call_2", "type": "function",
             "function": {"name": "speak", "arguments": {"prompt": "It is 18 degrees.", "durationSeconds": 2}}}
    calls = [{"role": "assistant", "content": "", "tool_calls": [weather]},
             {"role": "assistant", "content": "", "tool_calls": [speak]}] if one_call_per_message \
        else [{"role": "assistant", "content": "", "tool_calls": [weather, speak]}]
    return ([{"role": "user", "content": "What is the weather in Paris, and say it aloud?"}] + calls[:1]
            + ([{"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18 C and clear"}] if one_call_per_message else [])
            + calls[1:]
            + ([] if one_call_per_message else [{"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18 C and clear"}])
            + [{"role": "tool", "tool_call_id": "call_2", "name": "speak", "content": "Error: the voice is not installed"}])


def as_parts(messages):
    parts = copy.deepcopy(messages)
    for message in parts:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
    return parts


def special_tokens(directory):
    path = os.path.join(directory, "tokenizer_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        config = json.load(handle)
    tokens = {}
    for key, value in config.items():
        if not key.endswith("_token"):
            continue
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            tokens[key] = value
    return tokens


def templates(directory):
    found = []
    path = os.path.join(directory, "chat_template.jinja")
    if os.path.exists(path):
        with open(path) as handle:
            found.append(handle.read())
    path = os.path.join(directory, "tokenizer_config.json")
    if os.path.exists(path):
        with open(path) as handle:
            value = json.load(handle).get("chat_template")
        if isinstance(value, list):
            found += [entry.get("template", "") for entry in value]
        elif isinstance(value, str):
            found.append(value)
    return [template for template in found if template]


def render(template, messages, tools, variables):
    rendered, _ = chat_template_utils.render_jinja_template(
        conversations=[messages], tools=tools, chat_template=template, add_generation_prompt=True, **variables)
    return rendered[0]


def record_case(template, messages, tools, variables, foundation_order):
    entry = dict(template=template, tools=tools, variables=variables, foundation_order=foundation_order)
    for attempt in (messages, as_parts(messages)):
        try:
            entry.update(messages=attempt, expected=render(template, attempt, tools, variables))
            return entry
        except Exception as error:  # the template refused the case
            entry.update(messages=attempt, error=f"{type(error).__name__}: {error}")
    return entry


def main():
    with open(os.path.expanduser("~/.inferkit-validation.json")) as handle:
        config = json.load(handle)
    default = os.path.join(os.path.dirname(config.get("IK_CHAT_TEMPLATE_REF", os.path.expanduser("~/.inferkit-validation/reference/x"))),
                           "chat-template-tools-reference.json")
    out_path = sys.argv[1] if len(sys.argv) > 1 else default
    seen = {}
    for key in sorted(config):
        directory = config[key]
        if not key.startswith("IK_VAL_") or not isinstance(directory, str) or not os.path.isdir(directory):
            continue
        for index, template in enumerate(templates(directory)):
            if template in seen:
                continue
            seen[template] = f"{key}#{index}"
    reference = {}
    for template, name in seen.items():
        directory = config[name.split("#")[0]]
        variables = special_tokens(directory)
        reference[f"{name}/chat"] = record_case(template, CHAT, None, variables, True)
        if "tools" in template:
            harmony = "to=functions" in template or "functions." in template and "<|channel|>" in template
            reference[f"{name}/tools_declared"] = record_case(template, DECLARED, TOOLS, variables, True)
            reference[f"{name}/tool_conversation"] = record_case(template, tool_conversation(harmony), TOOLS, variables, False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(reference, handle, indent=1, ensure_ascii=False)
    refused = sum(1 for entry in reference.values() if "error" in entry)
    print(f"wrote {out_path}: {len(reference)} cases from {len(seen)} templates, {refused} refused")


if __name__ == "__main__":
    main()
