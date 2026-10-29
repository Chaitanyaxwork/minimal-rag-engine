# Minimal RAG Engine (from scratch)

A learning project: building a Retrieval-Augmented Generation engine using
only the Python standard library and NumPy. No LangChain, LlamaIndex,
Haystack, FAISS, Chroma, or vector databases.

## Status

| Phase | Module | Status |
|---|---|---|
| 1 | `chunker.py` - hybrid sentence-aware chunking with overlap | Implemented, tests included |
| 2 | NumPy in-memory vector store (cosine similarity, top-k) | Not started |
| 3 | Provider-independent embedding/LLM HTTP client (`urllib`) | Not started |
| 4 | End-to-end RAG orchestration | Not started |

## Usage

```python
from chunker import chunk_text

chunks = chunk_text("Cats purr. Dogs bark. Birds sing. Fish swim.",
                    chunk_size=25, overlap=11)
# ['Cats purr. Dogs bark.', 'Dogs bark. Birds sing.', 'Birds sing. Fish swim.']
```

## Run the tests

```bash
python -m unittest -v
```

Requires Python 3.8+ and no third-party packages.

## Known limitations (Phase 1)

- Sentence detection is regex-based (e.g. "Dr. Smith" is split after "Dr.").
- Whitespace is normalized, so newlines and layout are not preserved.
- Overlap is approximate: between 0 and `overlap` characters.
- Sizes are measured in characters, not model tokens.
