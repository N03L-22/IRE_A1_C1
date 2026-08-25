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
from dataclasses import dataclass

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
#: xlmr-base stays in MODELS as the brief-named ablation row: reporting it as
#: a measured failure is a stronger result than quietly not running it.
DEFAULT_MODEL = "minilm"

#: News titles and abstracts are short; 128 tokens covers title + abstract
#: with room to spare and keeps the batch small enough for 8 GB of VRAM.
MAX_TOKENS = 128


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
    batch_size: int = 64,
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

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state
            pooled = mean_pool(hidden, enc["attention_mask"])
            out[start : start + len(batch)] = pooled.float().cpu().numpy()

            if start and start % (batch_size * 200) == 0:
                done = start + len(batch)
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
