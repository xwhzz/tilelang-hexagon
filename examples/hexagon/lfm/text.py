"""The host's side of the text: the checkpoint's tokenizer and chat template.

It needs `tokenizers` and `jinja2`, not transformers: the template renders as
``apply_chat_template`` renders it, and the ids match transformers' tokenizer.
"""

import json
import os

import jinja2
import jinja2.ext
from tokenizers import Tokenizer


class Generation(jinja2.ext.Extension):
    """transformers' ``{% generation %}`` block (LFM3's template has one): it
    marks the assistant's text for training masks and renders its body as is."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


class Text:
    """The tokenizer and the chat template of a checkpoint directory
    (tokenizer.json, chat_template.jinja, special_tokens_map.json)."""

    def __init__(self, checkpoint: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))
        special = json.load(open(os.path.join(checkpoint, "special_tokens_map.json")))

        def token(name):  # LFM2 spells it {"content": ...}, LFM3 a string
            t = special[name]
            return t["content"] if isinstance(t, dict) else t

        self.bos_token = token("bos_token")
        self.eos = self.tokenizer.token_to_id(token("eos_token"))

        def raise_exception(message):
            raise jinja2.TemplateError(message)

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", Generation])
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(open(os.path.join(checkpoint, "chat_template.jinja")).read())

    def encode(self, text: str) -> list:
        """The ids of `text`, no special tokens added (those it spells stay)."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def chat(self, messages: list, **options) -> list:
        """The ids of a conversation ({"role", "content"} messages) and the
        start of the assistant's turn; `options` are the template's own
        (LFM3's enable_thinking, say)."""
        return self.encode(self.template.render(messages=messages, add_generation_prompt=True, bos_token=self.bos_token, **options))

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class Detokenizer:
    """Text as its ids arrive: each call returns what the new ids add. A
    character whose bytes are split between ids waits for the rest."""

    def __init__(self, text: Text):
        self.text, self.ids, self.shown = text, [], 0

    def add(self, ids: list) -> str:
        self.ids += ids
        s = self.text.decode(self.ids)
        end = len(s) - 1 if s.endswith("�") else len(s)
        new, self.shown = s[self.shown : end], max(self.shown, end)
        return new
