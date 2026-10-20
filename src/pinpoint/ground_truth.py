import pandas as pd


def function_id(path: str, qualname: str) -> str:
    """One string that names a function across the repo, e.g. "src/a.py::Cls.meth"."""
    return f"{path}::{qualname}"


def gold_functions(changed: dict[str, set[int]], functions: pd.DataFrame) -> list[str]:
    """Function ids the fix touched: for each changed line, the innermost function containing it.

    `changed` is changed_source_lines(patch); `functions` is parse_snapshot(...) for the
    issue's base_commit. Lines outside every function (imports, module constants, blank lines
    between defs) map to nothing. Sorted, no duplicates.
    """
    found = set()
    for path, lines in changed.items():
        in_file = functions[functions["path"] == path]
        for line in lines:
            containing = in_file[(in_file["start_line"] <= line) & (in_file["end_line"] >= line)]
            if containing.empty:
                continue
            # Nested functions sit inside their parent, so the innermost starts last.
            innermost = containing.loc[containing["start_line"].idxmax()]
            found.add(function_id(path, innermost["qualname"]))
    return sorted(found)
