"""Tests for minifsd.train.pack: datasets zipped for Google Drive (T2)."""

import pathlib
import zipfile

import pytest

from minifsd.train import pack


def _dataset(root: pathlib.Path) -> pathlib.Path:
    folder = root / "v9"
    for split in ("train", "val"):
        (folder / split).mkdir(parents=True)
        (folder / split / "meta.jsonl").write_text(f'{{"split": "{split}"}}\n')
        (folder / split / "a_rgb.png").write_bytes(bytes(range(256)) * 3)
    (folder / "README").write_text("v9")
    return folder


def _files(folder: pathlib.Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_round_trip_keeps_every_file_byte_for_byte(
    tmp_path: pathlib.Path,
) -> None:
    folder = _dataset(tmp_path / "src")
    zip_path = tmp_path / "v9.zip"
    assert pack.pack(folder, zip_path) == 5
    out = pack.unpack(zip_path, tmp_path / "dest")
    assert out == tmp_path / "dest" / "v9"  # the folder keeps its name
    assert _files(out) == _files(folder)


def test_files_are_stored_not_compressed(tmp_path: pathlib.Path) -> None:
    # PNG and JPEG are compressed already; deflating them again costs time
    # on both ends and saves almost nothing.
    zip_path = tmp_path / "v9.zip"
    pack.pack(_dataset(tmp_path / "src"), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


def test_an_existing_zip_is_not_overwritten(tmp_path: pathlib.Path) -> None:
    zip_path = tmp_path / "v9.zip"
    zip_path.write_bytes(b"older pack")
    with pytest.raises(FileExistsError):
        pack.pack(_dataset(tmp_path / "src"), zip_path)
    assert zip_path.read_bytes() == b"older pack"


def test_a_failed_pack_leaves_no_zip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _dataset(tmp_path / "src")

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", broken)
    with pytest.raises(OSError):
        pack.pack(folder, tmp_path / "v9.zip")
    assert list(tmp_path.glob("v9.zip*")) == []


def test_unpack_refuses_paths_outside_the_destination(
    tmp_path: pathlib.Path,
) -> None:
    # A zip can name "../x"; unpacking must never write outside dest.
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../outside.txt", "no")
    with pytest.raises(ValueError):
        pack.unpack(zip_path, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()
