"""
Vocab loader matching the ICDAR-paper convention:
    - blank token implicitly at index 0
    - real characters indexed 1..(V-1)
    - special tokens (<pad>, <sos>, <eos>, <blank>) are SKIPPED from the
      working vocabulary; the model only outputs blank + real chars.

`load_vocab(json_path)` returns char_to_idx with chars starting at 1.
Vocab size with blank = len(char_to_idx) + 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    specials = {"<pad>", "<sos>", "<eos>", "<blank>"}
    char_to_idx: Dict[str, int] = {}
    sorted_items = sorted(raw.items(), key=lambda kv: int(kv[1]))
    next_idx = 1
    for ch, _ in sorted_items:
        if ch in specials or (ch.startswith("<") and ch.endswith(">")):
            continue
        if ch not in char_to_idx:
            char_to_idx[ch] = next_idx
            next_idx += 1
    return char_to_idx


# Built-in normalization map (subset of char_tokenizer.py's mapping; covers
# the most common Urdu text variants found in our datasets).
NORMALIZATION_MAP = {
    "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
    "ؤ": "و", "ئ": "ی", "ة": "ہ", "ۃ": "ہ", "ۂ": "ہ", "ۓ": "ے",
    "ي": "ی",
    "ٴ": "", "ٔ": "", "ٰ": "", "ٓ": "",
    "َ": "", "ِ": "", "ُ": "", "ّ": "", "ْ": "",
    "ً": "", "ٌ": "", "ٍ": "",
    "ؒ": "", "ؓ": "", "ؔ": "", "٪": "",
    "|": "", "\"": "", "'": "", "‘": "", "’": "", "“": "", "”": "",
    "ﷲ": "", "ﷺ": "",
}


def normalize_text(text: str, valid_chars: set) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in valid_chars:
            out.append(ch)
            continue
        if ch in NORMALIZATION_MAP:
            mapped = NORMALIZATION_MAP[ch]
            if mapped:
                out.append(mapped)
            continue
        if unicodedata.category(ch) == "Mn":
            continue
        if ch.isspace():
            out.append(" ")
            continue
        # unsupported char -> skip
    return "".join(out)


def encode_text(text: str, char_to_idx: Dict[str, int]) -> List[int]:
    valid = set(char_to_idx.keys())
    text = normalize_text(text, valid)
    return [char_to_idx[c] for c in text if c in char_to_idx]


def decode_ids(ids: List[int], char_to_idx: Dict[str, int]) -> str:
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return "".join(idx_to_char.get(i, "") for i in ids if i in idx_to_char)
