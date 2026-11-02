from assistant.ingest import chunk_pages, clean

DOC = {"id": "doc", "title": "Doc", "url": "https://example.com/doc.pdf"}


def test_clean_rejoins_hyphenated_words_and_spaces():
    assert clean("regu-\nlation  applies\n\nhere") == "regulation applies here"


def test_short_page_is_one_chunk_and_empty_pages_are_skipped():
    chunks = chunk_pages(["one two three", "", "four five"], DOC, size=10, overlap=2)
    assert [c["page"] for c in chunks] == [1, 3]
    assert chunks[0]["text"] == "one two three"


def test_long_page_chunks_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_pages([" ".join(words)], DOC, size=10, overlap=3)
    texts = [c["text"].split() for c in chunks]
    assert texts[0][-3:] == texts[1][:3]
    assert set(words) == {w for t in texts for w in t}
    assert all(len(t) <= 10 for t in texts)
    assert len({c["id"] for c in chunks}) == len(chunks)
