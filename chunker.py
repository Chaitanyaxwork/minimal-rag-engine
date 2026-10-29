"""chunker.py - Phase 1 of the minimal RAG engine.

Splits a document into overlapping, sentence-aware chunks using only the
Python standard library.
"""

import re
from typing import List

# A sentence ends at . ! or ? (optionally followed by a closing quote or
# bracket) and then whitespace. Two fixed-width lookbehinds cover both cases.
_SENTENCE_END = re.compile(r"(?<=[.!?][\"')\]])\s+|(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Whitespace inside a sentence is collapsed."""
    sentences: List[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        for raw in _SENTENCE_END.split(paragraph):
            sentence = " ".join(raw.split())  # collapse tabs/newlines/runs of spaces
            if sentence:
                sentences.append(sentence)
    return sentences


def split_long_sentence(sentence: str, max_len: int) -> List[str]:
    """Break one oversized sentence into pieces of at most max_len characters.

    Prefers cutting at the last space inside the window; if the window has
    no space at all (e.g. a long URL) it hard-cuts at max_len.
    """
    pieces: List[str] = []
    rest = sentence
    while len(rest) > max_len:
        window = rest[: max_len + 1]       # +1 lets us see if a space sits right at the limit
        cut = window.rfind(" ")
        if cut <= 0:                       # no usable space -> hard cut
            cut = max_len
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def _overlap_seed(parts: List[str], overlap: int) -> List[str]:
    """Choose the trailing text of a finished chunk to start the next one.

    First choice: whole trailing sentences that fit within `overlap` chars.
    Fallback: the last `overlap` characters, trimmed forward to a word start.
    """
    if overlap == 0 or not parts:
        return []
    seed: List[str] = []
    total = 0
    for part in reversed(parts):
        added = len(part) + (1 if seed else 0)  # +1 for the joining space
        if total + added > overlap:
            break
        seed.insert(0, part)
        total += added
    if seed:
        return seed
    last = parts[-1]                            # longer than overlap, so slicing is safe
    tail = last[-overlap:]
    if last[-overlap - 1] != " ":               # we cut in the middle of a word
        space = tail.find(" ")
        tail = tail[space + 1:] if space != -1 else ""
    return [tail] if tail else []


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split `text` into chunks of at most `chunk_size` characters.

    Chunks end on sentence boundaries where possible, and each chunk starts
    with up to `overlap` characters repeated from the end of the previous one.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Oversized sentences are cut smaller than chunk_size so that an overlap
    # seed (<= overlap chars) plus a joining space plus the piece still fits.
    max_piece = max(1, chunk_size - overlap - 1)
    units: List[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            units.extend(split_long_sentence(sentence, max_piece))
        else:
            units.append(sentence)

    chunks: List[str] = []
    parts: List[str] = []                       # pieces of the chunk being built
    for unit in units:
        if parts and len(" ".join(parts + [unit])) > chunk_size:
            chunks.append(" ".join(parts))      # chunk is full: emit it
            parts = _overlap_seed(parts, overlap)
            if parts and len(" ".join(parts + [unit])) > chunk_size:
                parts = []                      # seed doesn't fit with this unit: drop it
        parts.append(unit)                      # every new chunk gets >= 1 new unit
    if parts:
        chunks.append(" ".join(parts))
    return chunks
