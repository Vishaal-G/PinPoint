from pinpoint.patches import changed_source_lines

# Real fix from pytest-dev__pytest-11143: one line added, nothing removed.
PURE_ADDITION = """\
diff --git a/src/_pytest/assertion/rewrite.py b/src/_pytest/assertion/rewrite.py
--- a/src/_pytest/assertion/rewrite.py
+++ b/src/_pytest/assertion/rewrite.py
@@ -676,6 +676,7 @@ def run(self, mod: ast.Module) -> None:
                 expect_docstring
                 and isinstance(item, ast.Expr)
                 and isinstance(item.value, ast.Constant)
+                and isinstance(item.value.value, str)
             ):
                 doc = item.value.value
                 if self.is_rewrite_disabled(doc):
"""

MODIFICATION = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,4 +10,4 @@ def f():
     a = 1
-    b = 2
+    b = 3
     c = 4
     return a + b + c
"""

DELETION = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -20,4 +20,2 @@ def g():
     x = 1
-    y = 2
-    z = 3
     return x
"""

ADDITION_AT_TOP = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,2 +1,3 @@
+import logging
 import threading
 import warnings
"""

TWO_HUNKS = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -5,3 +5,3 @@
 a = 1
-b = 2
+b = 20
 c = 3
@@ -50,2 +50,3 @@
 x = 1
+y = 2
 z = 3
"""

NEW_FILE = """\
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+def h():
+    return 1
"""

TWO_FILES = MODIFICATION + NEW_FILE


def test_pure_addition_points_at_line_before():
    assert changed_source_lines(PURE_ADDITION) == {"src/_pytest/assertion/rewrite.py": {678}}


def test_modification_points_at_removed_line():
    assert changed_source_lines(MODIFICATION) == {"pkg/mod.py": {11}}


def test_deletion_records_every_removed_line():
    assert changed_source_lines(DELETION) == {"pkg/mod.py": {21, 22}}


def test_addition_at_top_of_file_is_line_zero():
    assert changed_source_lines(ADDITION_AT_TOP) == {"pkg/mod.py": {0}}


def test_multiple_hunks_in_one_file():
    assert changed_source_lines(TWO_HUNKS) == {"pkg/mod.py": {6, 50}}


def test_new_file_is_skipped():
    assert changed_source_lines(NEW_FILE) == {}


def test_new_file_skipped_but_other_files_kept():
    assert changed_source_lines(TWO_FILES) == {"pkg/mod.py": {11}}
