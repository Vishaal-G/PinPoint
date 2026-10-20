from pinpoint.parse import parse_file, parse_snapshot

SAMPLE = b'''\
import functools


def top(x):
    return x


class Greeter:
    def greet(self, name):
        return name

    @functools.cache
    @staticmethod
    def cached(n):
        return n

    async def later(self):
        def helper():
            return 1
        return helper()

    class Inner:
        def deep(self):
            pass


if True:
    def conditional():
        pass
'''


def by_name(functions):
    return {f.qualname: f for f in functions}


def test_finds_all_functions_with_qualified_names():
    names = [f.qualname for f in parse_file(SAMPLE, "pkg/mod.py")]
    assert names == [
        "top",
        "Greeter.greet",
        "Greeter.cached",
        "Greeter.later",
        "Greeter.later.helper",
        "Greeter.Inner.deep",
        "conditional",
    ]


def test_line_numbers_are_1_based_and_inclusive():
    f = by_name(parse_file(SAMPLE, "pkg/mod.py"))
    assert (f["top"].start_line, f["top"].end_line) == (4, 5)
    assert (f["Greeter.greet"].start_line, f["Greeter.greet"].end_line) == (9, 10)
    assert (f["Greeter.Inner.deep"].start_line, f["Greeter.Inner.deep"].end_line) == (23, 24)


def test_decorators_are_part_of_the_function():
    cached = by_name(parse_file(SAMPLE, "pkg/mod.py"))["Greeter.cached"]
    assert (cached.start_line, cached.end_line) == (12, 15)
    assert cached.source.startswith("@functools.cache")


def test_nested_function_is_inside_its_parent():
    f = by_name(parse_file(SAMPLE, "pkg/mod.py"))
    outer, inner = f["Greeter.later"], f["Greeter.later.helper"]
    assert outer.start_line < inner.start_line <= inner.end_line <= outer.end_line


def test_source_is_the_function_text():
    top = by_name(parse_file(SAMPLE, "pkg/mod.py"))["top"]
    assert top.source == "def top(x):\n    return x"


def test_same_code_same_hash_in_different_files():
    a = by_name(parse_file(SAMPLE, "a.py"))["top"]
    b = by_name(parse_file(b"# moved\n\ndef top(x):\n    return x\n", "b.py"))["top"]
    assert a.hash == b.hash
    assert a.start_line != b.start_line


def test_different_code_different_hash():
    f = by_name(parse_file(SAMPLE, "pkg/mod.py"))
    assert f["top"].hash != f["Greeter.greet"].hash


def test_survives_syntax_errors():
    broken = b"def ok():\n    pass\n\nprint 'python 2'\n\ndef also_ok():\n    pass\n"
    names = [f.qualname for f in parse_file(broken, "old.py")]
    assert names == ["ok", "also_ok"]


def test_parse_snapshot_uses_relative_forward_slash_paths(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.py").write_bytes(SAMPLE)
    (tmp_path / "README.md").write_text("not python")
    df = parse_snapshot(tmp_path)
    assert set(df["path"]) == {"src/pkg/mod.py"}
    assert len(df) == 7
    assert list(df.columns) == [
        "path", "qualname", "start_line", "end_line", "source", "hash", "calls"
    ]


CALLS = b'''\
import os

@register("x")
def outer(p):
    path = os.path.join(p, "a")
    helper(path).strip()

    def inner():
        return deep_only()

    class Local:
        def m(self):
            return in_method()

    handlers[0]()
    return inner()

class C:
    def a(self):
        return self.b()
'''


def test_calls_by_name():
    f = by_name(parse_file(CALLS, "m.py"))
    # attribute calls keep the last part; decorator call counts; nested def's calls don't
    assert f["outer"].calls == ["helper", "inner", "join", "register", "strip"]
    assert f["outer.inner"].calls == ["deep_only"]
    assert f["outer.Local.m"].calls == ["in_method"]
    assert f["C.a"].calls == ["b"]


def test_function_with_no_calls():
    assert by_name(parse_file(SAMPLE, "pkg/mod.py"))["top"].calls == []
