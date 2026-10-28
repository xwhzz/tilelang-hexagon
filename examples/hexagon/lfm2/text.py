"""The host's side of the text: the checkpoint's tokenizer and chat template.

It needs `tokenizers` and `jinja2`, not transformers: the template renders as
``apply_chat_template`` renders it, and the ids match transformers' tokenizer.
"""

import json
import os

import jinja2
from tokenizers import Tokenizer


class Text:
    """The tokenizer and the chat template of a checkpoint directory
    (tokenizer.json, chat_template.jinja, special_tokens_map.json)."""

    def __init__(self, checkpoint: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))
        special = json.load(open(os.path.join(checkpoint, "special_tokens_map.json")))
        self.bos_token = special["bos_token"]["content"]
        self.eos = self.tokenizer.token_to_id(special["eos_token"]["content"])

        def raise_exception(message):
            raise jinja2.TemplateError(message)

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(open(os.path.join(checkpoint, "chat_template.jinja")).read())

    def encode(self, text: str) -> list:
        """The ids of `text`, no special tokens added (those it spells stay)."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def chat(self, messages: list) -> list:
        """The ids of a conversation ({"role", "content"} messages) and the
        start of the assistant's turn."""
        return self.encode(self.template.render(messages=messages, add_generation_prompt=True, bos_token=self.bos_token))

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
