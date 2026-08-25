"""Tokenisation -- one pipeline, both languages, no stemming.

plan/2-Lexical-BM25.md D3 decided this: regex word tokens, lowercased, NFC
normalised, no stemming and no stopword removal, applied identically to MIND
(English) and EB-NeRD (Danish).

The reasoning, briefly, because it is the kind of choice that looks arbitrary
later:

no stemming
    An English stemmer on Danish text is actively harmful, and using a
    *different* stemmer per dataset would confound the cross-dataset
    comparison Q3.5 asks for. Per-language Snowball stemming is the obvious
    ablation -- run it, report the delta, let the number decide.

no stopword removal
    BM25's IDF already down-weights terms that appear everywhere. Removing
    them is largely redundant and risks dropping words that carry meaning in
    one language but not the other.

NFC, non-negotiable
    Danish ae/oe/aa can be encoded as one code point or as base+combining
    pair. Two encodings of one word are two index terms, silently halving
    recall on affected queries. Applied to corpus *and* query or it achieves
    nothing -- which is what ``test_tokenise.py`` asserts.
"""

from __future__ import annotations

import re
import unicodedata

#: Unicode-aware word characters. ``\w`` with re.UNICODE keeps Danish
#: letters and digits, and splits on punctuation without gluing it to tokens.
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenise(text: str) -> list[str]:
    """Lowercased, NFC-normalised word tokens.

    >>> tokenise("Trump's aid freeze -- the cost")
    ['trump', 's', 'aid', 'freeze', 'the', 'cost']
    """
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def build_query(
    history_text: list[str],
    last_n: int = 15,
    dedup: bool = True,
) -> list[str]:
    """Turn a click history into query terms.

    The manufactured-query move: there is no query in a recommendation
    dataset, so we concatenate the titles of recent clicks and treat that as
    search text. Two knobs, both from D4:

    ``last_n``
        BM25 was tuned for 2-5 word queries. EB-NeRD users average ~160
        clicks; concatenating all of them yields thousands of tokens, at which
        point hundreds of terms each contribute a little, common words
        dominate the sum, and every document looks moderately relevant.
        Default 15 titles (~120 tokens) is already outside BM25's design
        regime -- which is the single most important caveat on any lexical
        number this project reports.

    ``dedup``
        ``k1`` saturates repetition *within a document*; repetition within the
        *query* is a different axis the formula was not designed around. Dedup
        is the default, no-dedup is an ablation row.

    ``history_text`` is assumed newest-last, so the tail is the recent slice.
    """
    if not history_text:
        return []
    recent = history_text[-last_n:] if last_n > 0 else history_text
    terms: list[str] = []
    for text in recent:
        terms.extend(tokenise(text))
    if not dedup:
        return terms
    # dict.fromkeys preserves first-seen order -- stable across runs, which
    # matters when a score tie is broken by position.
    return list(dict.fromkeys(terms))
