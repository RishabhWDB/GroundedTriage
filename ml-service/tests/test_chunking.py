import pytest

from rag.chunking import (
    Section,
    approx_tokens,
    chunk_sections,
    html_sections,
    markdown_sections,
)

HTML = """<div data-component-type="text"><div>Intro paragraph about events.</div></div>
<div><img src="x.gif"/></div><script>var x = 1;</script>
<div data-component-type="heading"><h2 id="a">Event Details Page</h2></div>
<div><div>Visitors see the <strong>Event Details</strong> page first.</div></div>
<div class="info"><ul><li>Tip one.</li><li>Tip two.</li></ul></div>
<h3>Registration Form</h3><p>Fill in the form.</p><h3>Empty heading</h3>"""


def test_html_sections_split_at_headings_and_drop_media():
    secs = html_sections(HTML)
    assert [s.heading for s in secs] == [
        "",
        "Event Details Page",
        "Registration Form",
        "Empty heading",
    ]
    assert secs[0].paragraphs == ["Intro paragraph about events."]
    assert secs[1].paragraphs == [
        "Visitors see the Event Details page first.", "- Tip one.", "- Tip two."
    ]  # fmt: skip
    assert "var x" not in " ".join(p for s in secs for p in s.paragraphs)


def test_markdown_sections():
    md = "# Title\nIntro line one\nline two\n\n## Refunds\nAsk within 30 days.\n\n### Notes\n"
    secs = markdown_sections(md)
    assert [(s.heading, s.paragraphs) for s in secs] == [
        ("Title", ["Intro line one line two"]),
        ("Refunds", ["Ask within 30 days."]),
        ("Notes", []),
    ]


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_sections_are_packed_with_their_headings():
    chunks = chunk_sections(html_sections(HTML))
    assert len(chunks) == 1 and chunks[0].heading == ""
    text = chunks[0].text
    assert text.startswith("Intro paragraph about events.")
    assert "\n\nEvent Details Page\nVisitors see" in text
    assert "\n\nRegistration Form\nFill in the form." in text
    assert "Empty heading" not in text  # a heading with no text is dropped


def test_packing_respects_the_budget_and_section_boundaries():
    secs = [Section(f"H{i}", [words(120, f"s{i}_")]) for i in range(5)]
    chunks = chunk_sections(secs, max_words=300, overlap_words=45)
    assert [c.heading for c in chunks] == ["H0", "H2", "H4"]  # 2 sections (242 words) per chunk
    for c in chunks:
        assert len(c.text.split()) <= 300
    # no section is split across chunks
    for i in range(5):
        assert sum(f"s{i}_0" in c.text.split() for c in chunks) == 1


def test_long_section_windows_with_overlap_inside_section_only():
    secs = [Section("Long", [words(250, "a"), words(250, "b")]), Section("Next", [words(10, "c")])]
    chunks = chunk_sections(secs, max_words=300, overlap_words=45)
    first = chunks[0].text.split("\n", 1)[1].split()
    second = chunks[1].text.split("\n\n")[0].split("\n", 1)[1].split()
    assert chunks[0].heading == "Long" and len(first) == 299  # budget minus the heading word
    assert second[:45] == first[-45:]  # overlap carried over inside the section
    assert second[-1] == "b249"
    assert chunks[-1].text.endswith("Next\n" + words(10, "c"))  # packed after the last window
    assert all(len(c.text.split()) <= 300 for c in chunks)


def test_single_huge_paragraph_is_windowed():
    chunks = chunk_sections([Section("", [words(1000)])], max_words=300, overlap_words=45)
    assert len(chunks) == 4
    assert all(len(c.text.split()) <= 300 for c in chunks)
    joined = set(w for c in chunks for w in c.text.split())
    assert joined == set(words(1000).split())  # nothing lost
    assert chunks[1].text.split()[:45] == chunks[0].text.split()[-45:]


def test_token_estimate_and_bad_overlap():
    assert approx_tokens("one two three") == 4
    with pytest.raises(ValueError):
        chunk_sections([], max_words=10, overlap_words=10)


def test_repeated_words_are_not_mistaken_for_overlap():
    chunks = chunk_sections([Section("", [" ".join(["w"] * 700)])], max_words=300, overlap_words=45)
    assert len(chunks) == 3 and sum(len(c.text.split()) for c in chunks) == 700 + 2 * 45
