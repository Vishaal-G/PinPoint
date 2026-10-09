from pinpoint.text import split_identifier


def test_snake_case():
    assert split_identifier("parse_patch_file") == ["parse", "patch", "file"]


def test_camel_case():
    assert split_identifier("getUserName") == ["get", "user", "name"]
