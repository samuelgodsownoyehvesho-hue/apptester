"""Artifact store behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.store import ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


class TestSaving:
    def test_saves_and_reads_back_bytes(self, store: ArtifactStore) -> None:
        ref = store.save_bytes("run_1", "screenshot", b"PNGDATA", suffix=".png")
        assert store.read_bytes(ref.key) == b"PNGDATA"
        assert ref.size_bytes == len(b"PNGDATA")
        assert ref.key.endswith(".png")
        assert ref.key.startswith("run_1/screenshot/")

    def test_json_roundtrip(self, store: ArtifactStore) -> None:
        payload = {"lines": [{"qty": 2}], "total": 39.98}
        ref = store.save_json("run_1", "cart-state", payload)
        assert store.read_json(ref.key) == payload

    def test_sha256_is_recorded(self, store: ArtifactStore) -> None:
        ref = store.save_text("run_1", "console", "hello")
        # sha256("hello")
        assert ref.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_kind_is_sanitised(self, store: ArtifactStore) -> None:
        # A kind containing separators must not be able to create directories
        # elsewhere in the tree.
        ref = store.save_text("run_1", "../../escape", "x")
        assert ".." not in ref.key
        assert store.path_for(ref.key).is_relative_to(store.root)


class TestDeduplication:
    def test_identical_content_yields_the_same_key(self, store: ArtifactStore) -> None:
        first = store.save_bytes("run_1", "screenshot", b"SAME")
        second = store.save_bytes("run_1", "screenshot", b"SAME")
        assert first.key == second.key

    def test_identical_content_is_written_once(self, store: ArtifactStore) -> None:
        store.save_bytes("run_1", "screenshot", b"SAME")
        store.save_bytes("run_1", "screenshot", b"SAME")
        assert len(store.list_run("run_1")) == 1

    def test_different_content_yields_different_keys(self, store: ArtifactStore) -> None:
        first = store.save_bytes("run_1", "screenshot", b"ONE")
        second = store.save_bytes("run_1", "screenshot", b"TWO")
        assert first.key != second.key
        assert len(store.list_run("run_1")) == 2

    def test_runs_are_isolated(self, store: ArtifactStore) -> None:
        store.save_bytes("run_1", "screenshot", b"X")
        store.save_bytes("run_2", "screenshot", b"X")
        assert len(store.list_run("run_1")) == 1
        assert len(store.list_run("run_2")) == 1


class TestPathSafety:
    @pytest.mark.parametrize(
        "key",
        ["../outside.txt", "run_1/../../outside.txt", "..\\outside.txt"],
    )
    def test_rejects_traversal(self, store: ArtifactStore, key: str) -> None:
        # Keys are derived from target-controlled input in places, so this must
        # be enforced here rather than trusted upstream.
        with pytest.raises(ValueError, match="escapes the store root"):
            store.path_for(key)

    def test_accepts_a_key_inside_the_root(self, store: ArtifactStore) -> None:
        ref = store.save_text("run_1", "console", "hello")
        assert store.path_for(ref.key).is_file()


class TestHousekeeping:
    def test_list_run_returns_refs(self, store: ArtifactStore) -> None:
        store.save_text("run_1", "console", "a")
        store.save_text("run_1", "trace", "b")
        refs = store.list_run("run_1")
        assert {ref.kind for ref in refs} == {"console", "trace"}

    def test_list_run_on_unknown_run_is_empty(self, store: ArtifactStore) -> None:
        assert store.list_run("nope") == []

    def test_total_bytes_sums_artifacts(self, store: ArtifactStore) -> None:
        store.save_bytes("run_1", "screenshot", b"12345")
        store.save_bytes("run_1", "screenshot", b"1234567890")
        assert store.total_bytes("run_1") == 15

    def test_delete_run_removes_everything(self, store: ArtifactStore) -> None:
        store.save_text("run_1", "console", "a")
        store.save_text("run_1", "trace", "b")
        removed = store.delete_run("run_1")
        assert removed == 2
        assert store.list_run("run_1") == []

    def test_delete_unknown_run_is_a_noop(self, store: ArtifactStore) -> None:
        assert store.delete_run("nope") == 0

    def test_partial_writes_are_not_listed(self, store: ArtifactStore) -> None:
        # A crash mid-write must not leave something that looks like evidence.
        stray = store.root / "run_1" / "screenshot"
        stray.mkdir(parents=True)
        (stray / "abc123.png.part").write_bytes(b"truncated")
        assert store.list_run("run_1") == []
