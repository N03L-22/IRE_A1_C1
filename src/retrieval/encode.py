"""Article embeddings -- one forward pass, no training (Q3.1).

> [!important] Computing embeddings is not training anything
> Q3 says "compute or load". Computing means **one forward pass of an
> already-trained encoder** over ~86K articles: no gradients, no labels,
> minutes on the RTX 4060. That is categorically different from fine-tuning
> an encoder, which is out of scope.

**Which encoder -- settled by measurement, not by argument.** The brief names
BERT and XLM-RoBERTa, so `xlm-roberta-base` was the intended primary. The
concern raised before writing any code was *anisotropy*: XLM-R and BERT are
trained for masked-language modelling, not to make cosine similarity
meaningful, and their representations are known to collapse into a narrow cone
where everything looks similar.

``danish_probe()`` tested that concern rather than assuming it, and the result
was unambiguous (finding F37):

===================  =========  ===========  =========  ==========
encoder              related    unrelated    margin     verdict
===================  =========  ===========  =========  ==========
xlm-roberta-base       0.9972       0.9954    +0.0018   OVERLAPS
MiniLM (384-d)         0.6523       0.0253    +0.6271   SEPARATES
===================  =========  ===========  =========  ==========

XLM-R rates "Brøndby beat FCK" and "a new apple cake recipe" as 0.995 similar
-- indistinguishable from two reports of the same match. **A retriever built on
those vectors returns arbitrary articles while producing perfectly plausible
metrics**, which is the failure this probe exists to catch. MiniLM achieves a
348x larger margin from *half* the dimensions, because it is trained so that
cosine similarity means something.

So `minilm` is the working default and `xlmr-base` stays as the brief-named
ablation row: reporting a measured failure is a stronger result than quietly
not running it.

Three mitigations are applied regardless of encoder:

1. **Mean-pool the final hidden states, masked by attention -- never `[CLS]`.**
   `[CLS]` is the worst offender for MLM-trained models.
2. **L2-normalise**, so inner product is cosine and no article can win on
   magnitude alone. An un-normalised index with inner-product search silently
   becomes a popularity ranker.
3. **Run the probe before trusting any number** the encoder produces.

See plan/3-Semantic-Embeddings.md D2 and decisions.md D-ENC.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: The ablation ladder. Dimensions are a consequence of the model, not a
#: free parameter -- xlm-roberta-large is 1024-d because it is larger, not
#: because a dimension was chosen.
MODELS = {
    "xlmr-base": ("xlm-roberta-base", 768),
    "xlmr-large": ("xlm-roberta-large", 1024),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
}

#: MEASURED, not assumed. The Danish probe (F37) settles this:
#:
#:   xlm-roberta-base  related 0.9972 vs unrelated 0.9954  margin +0.0018  OVERLAPS
#:   MiniLM (384-d)    related 0.6523 vs unrelated 0.0253  margin +0.6271  SEPARATES
#:
#: XLM-R collapses every Danish pair to ~0.996 cosine regardless of content --
#: it carries no usable retrieval geometry, and a retriever built on it would
#: return arbitrary articles while producing plausible metrics. MiniLM has a
#: 348x larger margin from HALF the dimensions, because it is trained so that
#: cosine similarity means something.
#:
#: CORRECTED BY F71 (2026-08-27): "no usable retrieval geometry" is too strong.
#: The collapse is anisotropy and it is fixable -- subtracting the mean
#: direction takes the margin +0.0018 -> +0.3875 (215x), and with 128-d
#: truncation XLM-R separates cleanly at +0.5070. More tellingly, even at the
#: collapsed baseline 4/5 related pairs still RANK in the top 5, and retrieval
#: consumes ranking rather than absolute similarity. MiniLM still wins on both
#: axes (+0.6270, 5/5) so the shipped choice is unchanged -- but the probe
#: tested magnitude while the system depends on order. Centering is a missing
#: fourth mitigation alongside mean-pool/L2/probe; one line, flagged for C-2.
#:
#: xlmr-base stays in MODELS as the brief-named ablation row: reporting it as
#: a measured failure is a stronger result than quietly not running it.
DEFAULT_MODEL = "minilm"

#: News titles and abstracts are short; 128 tokens covers title + abstract
#: with room to spare and keeps the batch small enough for 8 GB of VRAM.
MAX_TOKENS = 128

#: Measured on the RTX 4060 (8 GB) over 3,000 real EB-NeRD articles, MiniLM:
#:
#:   batch  64:  783 art/s   VRAM peak 0.66 GB
#:   batch 128: 1241 art/s   VRAM peak 0.67 GB   <- chosen
#:   batch 256: 1238 art/s   VRAM peak 0.67 GB
#:   batch 512:  999 art/s   VRAM peak 0.86 GB
#:
#: 128 and 256 tie; 128 is chosen as the smaller of the two, leaving more
#: headroom for a larger model. Throughput falls again at 512 -- padding
#: waste grows with batch width once batches straddle very different lengths.
#: Note VRAM is NOT the constraint here: 0.67 GB of 8 GB. The GPU is
#: underutilised and a bigger encoder would fit comfortably.
DEFAULT_BATCH_SIZE = 128


@dataclass
class EncodeStats:
    model: str
    dim: int
    n_articles: int
    seconds: float
    device: str
    batch_size: int

    def __str__(self) -> str:
        rate = self.n_articles / self.seconds if self.seconds else 0
        return (
            f"{self.model} ({self.dim}-d): {self.n_articles:,} articles in "
            f"{self.seconds:.0f}s = {rate:.0f}/s on {self.device} "
            f"(batch {self.batch_size})"
        )


def mean_pool(hidden, attention_mask):
    """Attention-masked mean over tokens.

    Padding tokens carry hidden states too, and averaging them in dilutes
    every short document toward the same vector -- which would look exactly
    like the anisotropy this function exists to avoid.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Unit-length rows, so inner product is cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def encode_texts(
    texts: list[str],
    model_key: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    max_tokens: int = MAX_TOKENS,
) -> tuple[np.ndarray, EncodeStats]:
    """Encode texts to L2-normalised float32 vectors.

    Runs in fp16 on CUDA (halves VRAM, no measurable quality cost for a
    forward pass) and returns float32 so downstream index code has one dtype
    to reason about.
    """
    import time

    import torch
    from transformers import AutoModel, AutoTokenizer

    name, dim = MODELS[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval().to(device)
    if device == "cuda":
        model.half()

    out = np.empty((len(texts), dim), dtype=np.float32)
    started = time.perf_counter()

    # Sort by length before batching, then scatter results back to the
    # caller's order. Every batch is padded to its longest member, so mixing
    # a 5-token headline with a 120-token abstract wastes most of the compute
    # on padding. Grouping similar lengths together removes that waste.
    #
    # Measured on 6,000 real EB-NeRD articles, MiniLM, batch 128:
    #   natural order  3,064 art/s
    #   length-sorted  5,522 art/s   -> 1.80x
    #
    # Results are identical: each text is still encoded independently, and
    # padding is masked out of the mean pool either way. Only the grouping
    # changes.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]

    def tokenise(rows: list[int]):
        return tokenizer(
            [texts[i] for i in rows],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )

    # Overlap CPU tokenisation with GPU compute. Profiling the serial loop
    # showed the split as 80% forward pass, 19% tokenisation, 1% transfer --
    # so tokenising batch n+1 on a worker thread while the GPU runs batch n
    # hides almost all of that 19%. The tokeniser is a fast (Rust) one that
    # releases the GIL, which is what makes a thread rather than a process
    # sufficient.
    #
    # Measured on 8,000 EB-NeRD articles, MiniLM, batch 128, length-sorted:
    #   serial     4,966 art/s
    #   pipelined  6,286 art/s   -> 1.27x
    done = 0
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(tokenise, batches[0]) if batches else None
        for k, rows in enumerate(batches):
            enc = pending.result()
            if k + 1 < len(batches):
                pending = pool.submit(tokenise, batches[k + 1])
            enc = {key: val.to(device, non_blocking=True) for key, val in enc.items()}

            hidden = model(**enc).last_hidden_state
            pooled = mean_pool(hidden, enc["attention_mask"]).float().cpu().numpy()
            out[rows] = pooled  # scatter back to the caller's order

            done += len(rows)
            if k and k % 200 == 0:
                rate = done / (time.perf_counter() - started)
                log.info("  encoded %s/%s (%.0f/s)", f"{done:,}", f"{len(texts):,}", rate)

    out = l2_normalise(out)
    stats = EncodeStats(
        model=name,
        dim=dim,
        n_articles=len(texts),
        seconds=time.perf_counter() - started,
        device=device,
        batch_size=batch_size,
    )
    log.info("%s", stats)
    return out, stats


def danish_probe(model_key: str = DEFAULT_MODEL, batch_size: int = 16) -> dict:
    """Does this encoder separate related Danish text from unrelated?

    **This gates every Q3 number.** An English-centric encoder on Danish does
    not degrade gracefully -- it tokenises into near-meaningless subwords and
    produces vectors with no useful geometry, while still producing *numbers*.
    Running metrics on top of that yields a plausible, wrong result.

    The probe embeds pairs that are obviously related and pairs that are
    obviously not, then checks the related pairs score higher on average. It
    is a smoke test, not a benchmark: passing does not prove the encoder is
    good, but failing proves it is unusable.
    """
    related = [
        ("Brøndby vandt kampen mod FCK", "FC København tabte til Brøndby i går"),
        ("Regeringen hæver skatten", "Ny skattestigning vedtaget i Folketinget"),
        ("Storm rammer Jylland i nat", "Kraftig blæst ventes over Vestjylland"),
        ("Prisen på benzin stiger igen", "Benzinpriserne er steget markant"),
        ("Ny corona-variant opdaget", "Sundhedsstyrelsen advarer om ny variant"),
    ]
    unrelated = [
        ("Brøndby vandt kampen mod FCK", "Ny opskrift på æblekage med kanel"),
        ("Regeringen hæver skatten", "Håndboldlandsholdet træner i Herning"),
        ("Storm rammer Jylland i nat", "Aktiekursen på Novo Nordisk falder"),
        ("Prisen på benzin stiger igen", "Kongehuset offentliggør nye billeder"),
        ("Ny corona-variant opdaget", "Fodboldklubben skifter cheftræner"),
    ]

    texts = [t for pair in related + unrelated for t in pair]
    vecs, stats = encode_texts(texts, model_key=model_key, batch_size=batch_size)

    sims = [float(vecs[2 * i] @ vecs[2 * i + 1]) for i in range(len(texts) // 2)]
    rel = sims[: len(related)]
    unrel = sims[len(related) :]

    result = {
        "model": stats.model,
        "dim": stats.dim,
        "related_mean": sum(rel) / len(rel),
        "unrelated_mean": sum(unrel) / len(unrel),
        "margin": sum(rel) / len(rel) - sum(unrel) / len(unrel),
        "min_related": min(rel),
        "max_unrelated": max(unrel),
        "separates": min(rel) > max(unrel),
    }
    log.info(
        "danish probe %s: related %.4f vs unrelated %.4f (margin %+.4f) -- %s",
        result["model"],
        result["related_mean"],
        result["unrelated_mean"],
        result["margin"],
        "SEPARATES" if result["separates"] else "OVERLAPS",
    )
    return result


def encode_cached(
    texts: list[str],
    ids: list[str],
    model_key: str = DEFAULT_MODEL,
    cache_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs,
) -> tuple[np.ndarray, EncodeStats | None]:
    """Encode once, reuse thereafter.

    The Q3.5 comparison builds several retrievers over the same corpus --
    semantic alone, semantic inside a recency window, semantic inside a
    fusion. Encoding is deterministic, so doing it once per (corpus, model)
    and reusing the vectors is free correctness *and* the single largest
    saving available: three retrievers means three identical forward passes
    otherwise.

    The cache key covers the model and the exact article-id list, so a
    different corpus or a different encoder never silently reuses vectors.
    A stale cache is worse than no cache: it would compare two retrievers on
    embeddings from different runs.
    """
    import hashlib
    import time

    cache_dir = Path(cache_dir or "data/store/embeddings")
    digest = hashlib.sha256(
        (model_key + "|" + "|".join(ids)).encode("utf-8")
    ).hexdigest()[:16]
    path = cache_dir / f"{model_key}_{len(ids)}_{digest}.npy"

    if path.exists():
        vecs = np.load(path)
        if vecs.shape[0] == len(ids):
            log.info("loaded cached embeddings: %s (%s)", path.name, vecs.shape)
            return vecs, None
        log.warning("cache %s has wrong shape %s -- re-encoding", path.name, vecs.shape)

    vecs, stats = encode_texts(texts, model_key=model_key, batch_size=batch_size, **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # np.save appends ".npy" unless the name already ends in it, so write to
    # a real temp file handle rather than guessing the final name.
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, vecs)
    tmp.replace(path)  # atomic: a killed run never leaves a half-written cache
    log.info("cached embeddings -> %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return vecs, stats
