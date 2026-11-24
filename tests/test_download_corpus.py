import json
from pathlib import Path

import pytest
import download_corpus as dc


def test_download_preserves_existing(tmp_path, monkeypatch):
    data = b'libertas seruitus\n'
    sha = dc.git_blob_sha(data)
    relative = 'new_lemmatized_texts/sample.txt'
    path = tmp_path / relative
    calls = []
    monkeypatch.setattr(dc, 'fetch', lambda url: calls.append(url) or data)
    assert dc.download_one(tmp_path, relative, sha)
    assert path.read_bytes() == data
    assert not dc.download_one(tmp_path, relative, sha)
    assert len(calls) == 1
    path.write_bytes(data.replace(b'\n', b'\r\n'))
    assert dc.matches_file(path, sha)
    assert not dc.download_one(tmp_path, relative, sha)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='отличается'):
        dc.download_one(tmp_path, relative, sha)
    assert path.read_bytes() == b'changed'


def test_bad_download_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, 'fetch', lambda _: b'wrong')
    relative = 'new_lemmatized_texts/sample.txt'
    with pytest.raises(ValueError, match='Git blob'):
        dc.download_one(tmp_path, relative, dc.git_blob_sha(b'expected'))
    assert not (tmp_path / relative).exists()


def test_check_only_and_missing_metadata(tmp_path, monkeypatch):
    (tmp_path / 'metadata.csv').write_text('file\nsample.txt\n')
    relative = 'new_lemmatized_texts/sample.txt'
    data = b'libertas\n'
    files = {relative: dc.git_blob_sha(data)}
    monkeypatch.setattr(dc, 'fetch', lambda _: b'{}')
    monkeypatch.setattr(dc, 'source_files', lambda *args: files)
    with pytest.raises(FileNotFoundError):
        dc.prepare_corpus(tmp_path, check_only=True)
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(data)
    assert dc.prepare_corpus(tmp_path, check_only=True)['downloaded'] == 0
    (tmp_path / 'metadata.csv').write_text('file\nmissing.txt\n')
    with pytest.raises(ValueError, match='metadata.csv'):
        dc.prepare_corpus(tmp_path, check_only=True)


def test_source_tree_rejects_truncation_and_wrong_count():
    with pytest.raises(ValueError, match='неполное'):
        dc.source_files({'truncated': True, 'tree': []})
    with pytest.raises(ValueError, match='1153'):
        dc.source_files({'truncated': False, 'tree': []})
