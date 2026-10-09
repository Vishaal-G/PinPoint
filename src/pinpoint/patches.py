from unidiff import PatchSet


def changed_source_lines(patch_text: str) -> dict[str, set[int]]:
    """Map each file the patch touches to the line numbers it changed in the OLD (buggy) file.

    Removed lines count as themselves. Added lines have no old line number, so they count as
    the old line they were inserted after. Brand-new files are skipped: there's no old version.
    """
    patch_set = PatchSet(patch_text)
    changed_lines = {}
    for patched_file in patch_set:
        if patched_file.is_added_file:
            continue
        for hunk in patched_file:
            # Most recent old-file line seen; an addition is anchored to it.
            last_source = hunk.source_start - 1
            for line in hunk:
                if line.source_line_no is not None:
                    last_source = line.source_line_no

                if line.is_removed:
                    changed_lines.setdefault(patched_file.path, set()).add(line.source_line_no)
                elif line.is_added:
                    changed_lines.setdefault(patched_file.path, set()).add(last_source)
    return changed_lines
