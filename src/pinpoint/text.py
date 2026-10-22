import re

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def split_identifier(name: str) -> list[str]:
    """Split a snake_case or camelCase identifier into lowercase words."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    # Acronym followed by a word: "HTTPError" -> "HTTP Error", "parseXMLFile" -> "parse XML File".
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)
    return [w.lower() for w in words.replace("_", " ").split()]


def tokenize(text: str, mode: str = "both") -> list[str]:
    """Turn code or prose into lowercase search tokens.

    mode="whole": identifiers as-is   is_rewrite_disabled -> [is_rewrite_disabled]
    mode="split": only their parts    is_rewrite_disabled -> [is, rewrite, disabled]
    mode="both":  whole + parts, so an exact name and its words can both match
    Punctuation and numbers are dropped, and so are 1-letter tokens (x, i, _).
    """
    if mode not in ("whole", "split", "both"):
        raise ValueError(f"unknown mode {mode!r}")
    tokens = []
    for ident in IDENTIFIER.findall(text):
        parts = split_identifier(ident)
        whole = ident.lower().strip("_")
        if mode in ("whole", "both"):
            tokens.append(whole)
        if mode == "split" or (mode == "both" and parts != [whole]):
            tokens.extend(parts)
    return [t for t in tokens if len(t) > 1]
