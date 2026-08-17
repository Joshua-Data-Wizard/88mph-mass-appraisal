"""
rag.py — letting the agent read the office's filing cabinet.

Section 8. RAG stands for Retrieval-Augmented Generation, which is a
complicated name for a simple sequence:

    1. Chop the documents into pieces.
    2. Turn every piece into a list of numbers that captures its meaning.
    3. Turn the question into numbers the same way.
    4. Hand the model the pieces whose numbers are closest to the question's.

Step 4 is the whole trick. The model is not searching. It is being handed the
right page and asked to read it — which is exactly what you would do with a
new hire and a policy binder.

Three decisions in here that are easy to get wrong
--------------------------------------------------
**The embedding model is named explicitly.** Left to itself, Chroma downloads
a 79 MB ONNX model the first time you index anything. That is a surprise
network call in the middle of a live demo. We name our own, so the download
happens when we choose and the "text becomes numbers" step is visible in the
code — which is the part Section 8 needs the audience to see.

**It runs on the CPU on purpose.** `bge-small` is 33M parameters and embeds our
whole policy corpus in about a second without a graphics card. That matters:
the hosted twin of Section 8 assumes there is no GPU, and RAG still needs
embeddings computed locally even when the *answering* happens in the cloud.
Keeping it off the GPU also leaves the VRAM free for the language model.

**Embeddings are normalized.** BGE is trained for cosine similarity; Chroma
defaults to L2 distance. Rather than configure the metric, we normalize — and
for unit vectors the two rank identically, so the setting stops mattering.
Measured: 6 of 6 retrieval either way.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import data

#: MIT licensed, ungated, 33M parameters, 384 dimensions, ~128 MB on disk.
#: Swap this one line to try a different embedding model.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: CPU by default. See the module docstring — this is a deliberate choice, not
#: an oversight. Pass device="cuda" if you want the 14x and have the headroom.
EMBEDDING_DEVICE = "cpu"

COLLECTION_NAME = "hill_valley_policy"

# Built once per process and kept. Rebuilding costs about a second, but the
# embedding model takes ~7s to load the first time and there is no reason to
# pay that twice.
_CACHE: dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────
#  Turning text into numbers
# ─────────────────────────────────────────────────────────────────────────

def get_embedder(device: str | None = None):
    """Load the sentence-transformer that turns text into vectors."""
    from sentence_transformers import SentenceTransformer

    device = device or EMBEDDING_DEVICE
    key = f"embedder:{EMBEDDING_MODEL}:{device}"
    if key not in _CACHE:
        _CACHE[key] = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _CACHE[key]


def embed(texts: list[str], device: str | None = None) -> list[list[float]]:
    """Turn a list of strings into a list of vectors.

    This is the entire "text becomes numbers" step. Every string comes back as
    384 floating point numbers that encode what it means — near-identical
    numbers for "may I split my lot" and "minimum lot area for a subdivision",
    even though they share almost no words.
    """
    model = get_embedder(device)
    vectors = model.encode(list(texts), normalize_embeddings=True,
                           show_progress_bar=False)
    return [v.tolist() for v in vectors]


class LocalEmbedder:
    """The bridge between our embedding model and Chroma.

    Chroma will happily supply its own embedding function — and quietly
    download 79 MB to do it. Handing it this class instead keeps the choice,
    the model, and the network access under our control.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or EMBEDDING_DEVICE

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        # Chroma requires this parameter to be named `input`.
        return embed(list(input), self.device)

    def embed_query(self, input: list[str] | str) -> list[list[float]]:  # noqa: A002
        """Embed a QUESTION rather than a passage.

        Chroma keeps this separate from `__call__` because some embedding
        models are *asymmetric* — they want a question phrased differently from
        the documents being searched. BGE is one of them: its authors suggest
        prefixing queries with "Represent this sentence for searching relevant
        passages: ".

        We tested that on this corpus and it changed nothing — 6 of 6 hero
        questions retrieved the right policy section either way, and v1.5 is
        documented as needing it less than earlier versions. So we skip it,
        rather than carry a magic string nobody reading this could explain.

        Worth revisiting on a larger or mushier corpus, where it may earn its
        keep.
        """
        if isinstance(input, str):
            input = [input]
        return embed(list(input), self.device)

    # Chroma asks a custom embedding function to identify and describe itself
    # so a persisted collection can be reopened later.
    @staticmethod
    def name() -> str:
        return "delorean-bge-small"

    def get_config(self) -> dict:
        return {"model": EMBEDDING_MODEL, "device": self.device}

    @staticmethod
    def build_from_config(config: dict) -> "LocalEmbedder":
        return LocalEmbedder(device=config.get("device"))


# ─────────────────────────────────────────────────────────────────────────
#  Chopping the documents up
# ─────────────────────────────────────────────────────────────────────────

def chunk_markdown(path: Path) -> list[dict]:
    """Split one policy document into chunks, one per numbered section.

    We split on `###` headings rather than on a fixed number of characters,
    because these documents are already written in numbered sections and a
    retrieved chunk that begins "4.3 Minimum lot size" is one a human can
    check. Fixed-size chunking would cut tables in half and produce citations
    nobody can look up.
    """
    text = path.read_text(encoding="utf-8")
    chunks = []

    for part in re.split(r"\n(?=### )", text):
        body = part.strip()

        # Keep only real numbered sections. The first piece of every document
        # is its title block and effective date, which retrieves badly and --
        # worse -- cites as "§" with no section number. An assessment office
        # cannot act on a quotation it cannot look up, so a chunk that cannot
        # name its own source does not belong in the index.
        if not body.startswith("### ") or len(body) < 60:
            continue
        heading = body.splitlines()[0].lstrip("# ").strip()
        number = re.match(r"^([\d.]+)", heading)
        chunks.append({
            "text": body,
            "document": path.stem,
            "heading": heading,
            "section": number.group(1).rstrip(".") if number else "",
        })
    return chunks


def load_chunks() -> list[dict]:
    """Every chunk of every policy document in data/policy/."""
    chunks: list[dict] = []
    for path in sorted(data.POLICY_DIR.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks


# ─────────────────────────────────────────────────────────────────────────
#  The index
# ─────────────────────────────────────────────────────────────────────────

def build_index(device: str | None = None, verbose: bool = False):
    """Embed every policy chunk and put it in a searchable collection.

    In memory, not on disk. The corpus is small enough that rebuilding takes
    about a second, and an index that cannot go stale is one less thing to
    debug in front of an audience.
    """
    import chromadb

    chunks = load_chunks()
    if not chunks:
        raise FileNotFoundError(
            f"No policy documents found in {data.POLICY_DIR}. "
            "They ship with the repository — check the folder exists."
        )

    client = chromadb.EphemeralClient()

    # `EphemeralClient()` hands back a SHARED in-memory database, not a fresh
    # one -- call it twice in a process and you get the same store, so a second
    # build fails with "collection already exists". Dropping it first makes
    # rebuilding always safe, which is what you want when someone re-runs a
    # notebook cell.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - not existing yet is the normal case
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=LocalEmbedder(device),
        # Cosine, to match what BGE was trained for. Because we normalize the
        # vectors this is belt and braces -- L2 would rank identically -- but
        # saying it out loud beats relying on a coincidence of geometry.
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"{c['document']}#{c['section'] or n}" for n, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        metadatas=[{"document": c["document"], "heading": c["heading"],
                    "section": c["section"]} for c in chunks],
    )

    if verbose:
        print(f"Indexed {len(chunks)} chunks from "
              f"{len(set(c['document'] for c in chunks))} policy documents.")

    # Cache it here, so building it directly and asking for it later agree.
    _CACHE["collection"] = collection
    return collection


def get_collection(device: str | None = None):
    """The policy index, built on first use and kept for the process."""
    if "collection" not in _CACHE:
        _CACHE["collection"] = build_index(device)
    return _CACHE["collection"]


def search(question: str, top_k: int = 3) -> list[dict]:
    """Find the policy sections that best answer a question.

    Returns dicts with the text and — critically — **where it came from**.
    An assessment office cannot act on an answer it cannot trace back to a
    document and a section number.
    """
    collection = get_collection()
    top_k = max(1, min(int(top_k), 10))

    result = collection.query(query_texts=[question], n_results=top_k)

    hits = []
    for n in range(len(result["ids"][0])):
        meta = result["metadatas"][0][n]
        distance = result["distances"][0][n]
        hits.append({
            "document": meta["document"],
            "section": meta["section"],
            "heading": meta["heading"],
            "text": result["documents"][0][n],
            # Chroma returns cosine DISTANCE; 1 - d is the similarity people
            # expect to see, where 1.0 would be an identical passage.
            "relevance": round(1.0 - float(distance), 3),
        })
    return hits
