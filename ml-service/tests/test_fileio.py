import json

import pytest

from app.fileio import append_line, atomic_path, atomic_write_text, open_append, read_jsonl


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("old")
    atomic_write_text(p, '{"a": 1}')
    assert json.loads(p.read_text()) == {"a": 1}
    assert [x.name for x in tmp_path.iterdir()] == ["c.json"]


def test_atomic_path_keeps_old_file_on_error(tmp_path):
    p = tmp_path / "split.parquet"
    p.write_text("old")
    with pytest.raises(RuntimeError), atomic_path(p) as tmp:
        assert tmp.suffix == ".parquet" and tmp.parent == tmp_path
        tmp.write_text("half")
        raise RuntimeError("crash")
    assert p.read_text() == "old"
    assert [x.name for x in tmp_path.iterdir()] == ["split.parquet"]


def test_read_jsonl_drops_only_a_truncated_last_line(tmp_path):
    p = tmp_path / "progress.jsonl"
    p.write_text('{"id": 1}\n{"id": 2}\n{"id": 3, "te')
    assert read_jsonl(p) == [{"id": 1}, {"id": 2}]
    p.write_text('{"id": 1}\n{"id": 2}')  # complete, newline missing: kept
    assert read_jsonl(p) == [{"id": 1}, {"id": 2}]
    p.write_text('{"id": 1}\n{bad\n{"id": 3}\n')
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(p)
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_open_append_repairs_the_tail(tmp_path):
    p = tmp_path / "progress.jsonl"
    p.write_text('{"id": 1}\n{"id": 2, "te')
    with open_append(p) as f:
        append_line(f, {"id": 2})
    assert read_jsonl(p) == [{"id": 1}, {"id": 2}]
    p.write_text('{"id": 1}')
    with open_append(p) as f:
        append_line(f, {"id": 2})
    assert p.read_text() == '{"id": 1}\n{"id": 2}\n'
