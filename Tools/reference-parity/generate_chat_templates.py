#!/usr/bin/env python3
"""Generate the chat-template parity reference.

The chat-template renderer (`NFKMLXChatTemplateRenderer`) is held to transformers' own
`apply_chat_template`. This writes the reference record `NFKMLXChatTemplateTests` reads: for each
case, the Jinja template, the messages, and the exact rendered string transformers produces. Point
`IK_CHAT_TEMPLATE_REF` at the output.

The templates are inlined rather than downloaded so the reference is reproducible offline: the Qwen3
template (the hardest — namespaces, reversed slicing, `is` tests, string methods), Llama-3 (bos +
`| trim` precedence), and Gemma (`%`, `!=` on booleans, `set role`, `raise_exception` guards).

The environment mirrors transformers' `_compile_jinja_template`: an immutable sandbox with
`trim_blocks` and `lstrip_blocks`, a `tojson` filter, and a `raise_exception` global.

Run under the LLM oracle interpreter (transformers installed):
    ~/.inferkit-validation/llmvenv/bin/python3 Tools/reference-parity/generate_chat_templates.py \
        [output.json]
"""
import json
import os
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment


def render(template, messages, add_generation_prompt, bos_token="", eos_token="", tools=None):
    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    compiled = env.from_string(template)
    kwargs = dict(messages=messages, add_generation_prompt=add_generation_prompt,
                  bos_token=bos_token, eos_token=eos_token)
    if tools is not None:
        kwargs["tools"] = tools
    return compiled.render(**kwargs)


QWEN = (
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n"
    "        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may "
    "call one or more functions to assist with the user query.\\n\\nYou are provided with function "
    "signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n"
    "        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n"
    "\\nFor each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments"
    "\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if "
    "messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + "
    "'<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, "
    "last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index "
    "= (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == "
    "\"user\" and message.content is string and not(message.content.startswith('<tool_response>') and "
    "message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n"
    "        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message "
    "in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content "
    "%}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == "
    "\"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + "
    "message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == "
    "\"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if "
    "message.reasoning_content is string %}\n            {%- set reasoning_content = "
    "message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n')."
    "split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = "
    "content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n"
    "        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not "
    "loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + "
    "'\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
    "            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + "
    "'\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for "
    "tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not "
    "loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- "
    "if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n"
    "                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n"
    "                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n"
    "                {%- if tool_call.arguments is string %}\n                    {{- "
    "tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | "
    "tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- "
    "endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == "
    "\"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
    "            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or "
    "(messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- "
    "endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- "
    "'<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false "
    "%}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
)

LLAMA = (
    "{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}{%- if "
    "add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %}"
)

GEMMA = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

CASES = {
    "qwen_user": dict(template=QWEN, messages=[{"role": "user", "content": "What is 2+2?"}],
                      add_generation_prompt=True, bos_token="<|begin_of_text|>"),
    "qwen_multi": dict(template=QWEN, messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ], add_generation_prompt=True, bos_token="<|begin_of_text|>"),
    "qwen_nogp": dict(template=QWEN, messages=[{"role": "user", "content": "Ping"}],
                      add_generation_prompt=False, bos_token="<|begin_of_text|>"),
    "qwen_toolcall": dict(template=QWEN, messages=[
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "sunny"},
    ], add_generation_prompt=True, bos_token="<|begin_of_text|>"),
    "llama_user": dict(template=LLAMA, messages=[
        {"role": "user", "content": "  Hello  "},
        {"role": "assistant", "content": "Hi there"},
    ], add_generation_prompt=True, bos_token="<|begin_of_text|>"),
    "gemma_multi": dict(template=GEMMA, messages=[
        {"role": "user", "content": "  Hi  "},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ], add_generation_prompt=True, bos_token="<bos>"),
}


def main():
    default = os.path.expanduser("~/.inferkit-validation/reference/chat-template-reference.json")
    out_path = sys.argv[1] if len(sys.argv) > 1 else default
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ref = {}
    for name, c in CASES.items():
        rendered = render(c["template"], c["messages"], c["add_generation_prompt"],
                          c.get("bos_token", ""))
        ref[name] = dict(template=c["template"], messages=c["messages"],
                         add_generation_prompt=c["add_generation_prompt"],
                         bos_token=c.get("bos_token", ""), expected=rendered)
        print(f"generated {name}: {rendered!r}"[:120])
    with open(out_path, "w") as f:
        json.dump(ref, f, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
