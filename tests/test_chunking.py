from pinpoint.chunking import chunk_text, line_windows


def test_fits_in_one_window():
    assert line_windows([10, 10, 10], size=500) == [(1, 3)]


def test_windows_overlap_by_trailing_lines():
    # Lines of 100 tokens, size 300, overlap 100: each window repeats the previous last line.
    assert line_windows([100] * 6, size=300, overlap=100) == [(1, 3), (3, 5), (5, 6)]


def test_no_overlap():
    assert line_windows([100] * 6, size=300, overlap=0) == [(1, 3), (4, 6)]


def test_overlap_never_steps_back_to_window_start():
    # A huge overlap must still move forward, or the loop would never end.
    assert line_windows([100] * 4, size=200, overlap=10_000) == [(1, 2), (2, 3), (3, 4)]


def test_long_line_gets_its_own_window():
    assert line_windows([50, 900, 50], size=500, overlap=0) == [(1, 1), (2, 2), (3, 3)]


def test_empty_file():
    assert line_windows([]) == []


def test_chunk_text_returns_line_ranges_and_text():
    text = "a b\nc d\ne f\n"
    count = lambda lines: [len(line.split()) for line in lines]
    assert chunk_text(text, count, size=4, overlap=0) == [(1, 2, "a b\nc d"), (3, 3, "e f")]
