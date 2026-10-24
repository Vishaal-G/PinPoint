"""Fixed-size chunking: cut a file into ~N-token windows that start and end on line boundaries.

Line-aligned windows map back to line ranges, which is what lets a chunk score be
handed to the functions it overlaps.
"""
from collections.abc import Callable, Sequence


def line_windows(line_tokens: Sequence[int], size: int = 500, overlap: int = 50) -> list[tuple[int, int]]:
    """Group lines into windows of at most `size` tokens; each window repeats up to `overlap`
    tokens of trailing lines from the one before. Returns 1-based inclusive (start, end) lines.

    A single line longer than `size` gets a window to itself (the model truncates it).
    """
    windows = []
    n, start = len(line_tokens), 0
    while start < n:
        end, total = start, 0
        while end < n and (end == start or total + line_tokens[end] <= size):
            total += line_tokens[end]
            end += 1
        windows.append((start + 1, end))
        if end == n:
            break
        # Step back over trailing lines worth <= overlap tokens, but always move forward.
        nxt, back = end, 0
        while nxt - 1 > start and back + line_tokens[nxt - 1] <= overlap:
            nxt -= 1
            back += line_tokens[nxt]
        start = nxt
    return windows


def chunk_text(text: str, count_tokens: Callable[[list[str]], list[int]],
               size: int = 500, overlap: int = 50) -> list[tuple[int, int, str]]:
    """(start_line, end_line, chunk text) for every window of `text`.
    `count_tokens` maps a list of lines to their token counts (the model's own tokenizer)."""
    lines = text.splitlines()
    return [(s, e, "\n".join(lines[s - 1:e]))
            for s, e in line_windows(count_tokens(lines), size, overlap)]
