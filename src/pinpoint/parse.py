import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PARSER = Parser(Language(tspython.language()))


@dataclass
class Function:
    path: str        # relative to the snapshot root, forward slashes (matches patch paths)
    qualname: str    # e.g. "AssertionRewriter.run", "outer.inner"
    start_line: int  # 1-based, inclusive; includes decorators
    end_line: int    # 1-based, inclusive
    source: str
    hash: str        # sha256 of source: same code anywhere -> same hash


def parse_file(source: bytes, path: str) -> list[Function]:
    """Every function and method in one file, nested ones included, in file order."""
    tree = PARSER.parse(source)
    found = []
    _walk(tree.root_node, [], path, found)
    return found


def _walk(node: Node, scope: list[str], path: str, found: list[Function]) -> None:
    for child in node.named_children:
        # `@decorator\ndef f` is a decorated_definition wrapping the def. Use the wrapper's
        # range so a change to a decorator line still counts as a change to the function.
        outer = child
        if child.type == "decorated_definition":
            child = child.child_by_field_name("definition")

        if child.type == "function_definition":
            name = child.child_by_field_name("name").text.decode()
            text = outer.text.decode(errors="replace")
            found.append(Function(
                path=path,
                qualname=".".join(scope + [name]),
                start_line=outer.start_point[0] + 1,
                end_line=outer.end_point[0] + 1,
                source=text,
                hash=hashlib.sha256(text.encode()).hexdigest(),
            ))
            _walk(child.child_by_field_name("body"), scope + [name], path, found)
        elif child.type == "class_definition":
            name = child.child_by_field_name("name").text.decode()
            _walk(child.child_by_field_name("body"), scope + [name], path, found)
        else:
            # if/try/with blocks etc. can hold defs too (e.g. `if sys.version_info...: def f`).
            _walk(child, scope, path, found)


def parse_snapshot(root: Path) -> pd.DataFrame:
    """Parse every .py file under a snapshot folder into one table, one row per function."""
    rows = []
    for file in sorted(root.rglob("*.py")):
        rel = file.relative_to(root).as_posix()
        rows += [asdict(f) for f in parse_file(file.read_bytes(), rel)]
    return pd.DataFrame(rows, columns=[f for f in Function.__dataclass_fields__])
