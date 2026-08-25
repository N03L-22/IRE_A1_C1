"""Tests for the Phase 1 readers.

Two kinds here. Pure unit tests on parsing logic run anywhere. Tests that need
extracted data are skipped when it is absent, so a fresh clone still gets a
green run before `make data`.

The properties asserted are the ones that would silently corrupt every
downstream number if they broke: label parsing, NFC normalisation, and the
history/timestamp asymmetry between the two datasets.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data.readers import EbnerdReader, MindReader, normalise
from src.data.schema import Article, History

WORK = Path("data/work")
HAS_MIND = (WORK / "mind" / "train" / "behaviors.tsv").exists()
HAS_EBNERD = (WORK / "ebnerd" / "demo" / "articles.parquet").exists()


# --------------------------------------------------------------- parsing ---


class TestSlateParsing:
    """MIND packs the slate and its labels into one string."""

    def test_labelled_slate(self):
        cands, clicked = MindReader._parse_slate("N1-0 N2-1 N3-0 N4-1")
        assert cands == ["N1", "N2", "N3", "N4"]
        assert clicked == ["N2", "N4"]

    def test_unlabelled_test_row(self):
        """MINDlarge_test has bare ids -- must not be read as clicks (F14)."""
        cands, clicked = MindReader._parse_slate("N1 N2 N3")
        assert cands == ["N1", "N2", "N3"]
        assert clicked == []

    def test_hyphen_in_article_id(self):
        """rpartition, not split: an id containing '-' must survive."""
        cands, clicked = MindReader._parse_slate("N-odd-1 N2-0")
        assert cands == ["N-odd", "N2"]
        assert clicked == ["N-odd"]

    def test_empty_slate(self):
        assert MindReader._parse_slate("") == ([], [])


class TestEntityParsing:
    def test_extracts_labels(self):
        raw = '[{"Label": "Denmark", "Type": "G"}, {"Label": "EU", "Type": "O"}]'
        assert MindReader._entities(raw) == ["Denmark", "EU"]

    def test_malformed_json_is_not_fatal(self):
        """A bad row must not kill a 2M-row parse."""
        assert MindReader._entities("{not json") == []

    def test_empty(self):
        assert MindReader._entities("[]") == []
        assert MindReader._entities("") == []


class TestNormalise:
    def test_collapses_whitespace(self):
        assert normalise("  a\t\tb\n c ") == "a b c"

    def test_nfc_unifies_danish_encodings(self):
        """The silent-recall-killer: two encodings of one Danish word.

        Of the three Danish letters only 'å' (U+00E5) has a canonical
        decomposition -- 'a' + U+030A COMBINING RING ABOVE. 'ø' (U+00F8) and
        'æ' (U+00E6) are atomic and NFD leaves them alone. So 'å' is the
        character that can actually arrive in two encodings, look identical,
        and become two different index terms.
        """
        composed = "århus"
        decomposed = unicodedata.normalize("NFD", composed)
        assert composed != decomposed  # genuinely different byte sequences
        assert normalise(composed) == normalise(decomposed)

    def test_atomic_danish_letters_pass_through(self):
        """'ø' and 'æ' have no decomposition -- normalising must not alter them."""
        for word in ("rødgrød", "æble"):
            assert normalise(word) == word


# ---------------------------------------------------------- leakage edge ---


class TestHistoryBoundary:
    """History.before() is the leakage boundary. Everything rests on it."""

    def test_filters_strictly_before(self):
        t = datetime(2023, 5, 20, 12, 0)
        h = History(
            user_id="u1",
            clicked_ids=["a", "b", "c"],
            times=[t - timedelta(hours=2), t - timedelta(minutes=1), t + timedelta(minutes=1)],
        )
        assert h.before(t) == ["a", "b"]  # 'c' is in the future

    def test_boundary_is_exclusive(self):
        """A click at exactly t is not 'before' t."""
        t = datetime(2023, 5, 20, 12, 0)
        h = History(user_id="u1", clicked_ids=["a"], times=[t])
        assert h.before(t) == []

    def test_without_timestamps_returns_all(self):
        """MIND: no timestamps, so no filtering is possible (F1).

        This is the documented assumption, asserted so it stays visible rather
        than becoming a silent behaviour.
        """
        h = History(user_id="u1", clicked_ids=["a", "b"], times=None)
        assert h.before(datetime(2020, 1, 1)) == ["a", "b"]
        assert h.is_verifiable is False

    def test_with_timestamps_is_verifiable(self):
        h = History(user_id="u1", clicked_ids=["a"], times=[datetime(2023, 5, 20)])
        assert h.is_verifiable is True


class TestArticle:
    def test_retrieval_text_is_title_plus_abstract(self):
        """Q2.1, and the only field pair available on both datasets (F10)."""
        a = Article(article_id="1", title="Title", abstract="Abstract", body="Body")
        assert a.retrieval_text == "Title Abstract"
        assert "Body" not in a.retrieval_text

    def test_handles_missing_abstract(self):
        a = Article(article_id="1", title="Only title")
        assert a.retrieval_text == "Only title"


# ------------------------------------------------------- integration ---


@pytest.mark.skipif(not HAS_EBNERD, reason="run `make data` first")
class TestEbnerdIntegration:
    @pytest.fixture(scope="class")
    def reader(self):
        return EbnerdReader(WORK / "ebnerd" / "demo")

    def test_articles_have_ids_and_titles(self, reader):
        arts = [a for _, a in zip(range(100), reader.articles())]
        assert len(arts) == 100
        assert all(a.article_id for a in arts)
        assert sum(1 for a in arts if a.title) > 90

    def test_history_carries_timestamps(self, reader):
        """EB-NeRD's distinguishing property -- the boundary is provable."""
        h = next(iter(reader.histories("train")))
        assert h.is_verifiable
        assert len(h.times) == len(h.clicked_ids)

    def test_clicked_is_subset_of_candidates(self, reader):
        for _, imp in zip(range(500), reader.impressions("train")):
            assert set(imp.clicked) <= set(imp.candidates), imp.impression_id

    def test_post_click_fields_are_not_exposed(self, reader):
        """F6: read_time/scroll/next_* must never reach the schema."""
        imp = next(iter(reader.impressions("train")))
        for leak in ("read_time", "scroll_percentage", "next_read_time"):
            assert not hasattr(imp, leak)


@pytest.mark.skipif(not HAS_MIND, reason="run `make data` first")
class TestMindIntegration:
    @pytest.fixture(scope="class")
    def reader(self):
        return MindReader(WORK / "mind")

    def test_article_union_is_deduplicated(self, reader):
        """F2: train and dev ship different news.tsv files."""
        ids = [a.article_id for _, a in zip(range(5000), reader.articles())]
        assert len(ids) == len(set(ids))

    def test_history_has_no_timestamps(self, reader):
        h = next(iter(reader.histories("train")))
        assert h.times is None
        assert not h.is_verifiable

    def test_impressions_parse(self, reader):
        imps = [i for _, i in zip(range(200), reader.impressions("train"))]
        assert len(imps) == 200
        assert all(i.candidates for i in imps)
        assert all(set(i.clicked) <= set(i.candidates) for i in imps)


HAS_EBNERD_TEST = (Path("data/work/ebnerd/testset/test/behaviors.parquet")).exists()


@pytest.mark.skipif(not HAS_EBNERD_TEST, reason="run `make data-testset` first")
def test_ebnerd_reader_handles_the_unlabelled_test_schema() -> None:
    """The test split has 14 columns; train has 17 (F14).

    `article_ids_clicked` and `article_id` (the labels) and `next_read_time` /
    `next_scroll_percentage` (future information) are removed by the
    organisers. A hardcoded column list raises KeyError on the parquet read --
    which is exactly how the first EB-NeRD submission run died, 25 seconds in
    and after a 5-second article load, so the failure was easy to miss.

    An unlabelled impression is a legitimate object here, not an error: it
    reports is_labelled == False, the harness skips it, and the submission
    path does not need a label.
    """
    reader = EbnerdReader(WORK / "ebnerd" / "testset")

    seen = 0
    for imp in reader.impressions("test"):
        assert imp.candidates, "test impression with no slate to rank"
        assert imp.clicked == [], "test split must not carry labels"
        assert imp.is_labelled is False
        seen += 1
        if seen >= 50:
            break
    assert seen == 50, "reader yielded nothing from the test split"


@pytest.mark.skipif(not HAS_EBNERD_TEST, reason="run `make data-testset` first")
def test_ebnerd_reader_still_reads_labels_where_they_exist() -> None:
    """The schema-tolerant path must not silently drop labels on train/val."""
    if not Path("data/work/ebnerd/small/train/behaviors.parquet").exists():
        pytest.skip("small tier not extracted")
    reader = EbnerdReader(WORK / "ebnerd" / "small")
    labelled = sum(1 for i, imp in enumerate(reader.impressions("train"))
                   if imp.is_labelled or i > 200)
    assert labelled > 0, "labels lost on a split that has them"
