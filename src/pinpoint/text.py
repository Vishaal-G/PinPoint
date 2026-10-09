import re


def split_identifier(name: str) -> list[str]:
    """Split a snake_case or camelCase identifier into lowercase words."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()
    return [w.lower() for w in words]
