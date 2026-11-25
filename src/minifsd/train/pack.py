"""Datasets zipped for Google Drive, and unzipped again on Colab (T2).

    pixi run python -m minifsd.train.pack --out data/packs
        data/sim/v9 data/coco-people

(one command, wrapped here to fit)

Drive is slow with thousands of small files, so each dataset goes up as one
zip and each Colab session copies it to the session disk and unpacks it
there (ADR 0007). Files are stored, not compressed: PNG and JPEG are
compressed already.
"""

import argparse
import pathlib
import zipfile
from collections.abc import Sequence


def pack(folder: pathlib.Path, zip_path: pathlib.Path) -> int:
    """Zips a folder; inside the zip, paths start with the folder's name.

    Written under a temporary name and renamed when complete, so a failed
    pack never leaves a zip that looks whole.

    Args:
        folder: The dataset folder, e.g. data/sim/v9.
        zip_path: The zip to write.

    Returns:
        How many files went in.

    Raises:
        FileExistsError: If zip_path exists; packs are never overwritten.
    """
    if zip_path.exists():
        raise FileExistsError(zip_path)
    part = zip_path.with_name(zip_path.name + ".part")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
            for p in files:
                z.write(p, p.relative_to(folder.parent))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.rename(zip_path)
    return len(files)


def unpack(zip_path: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Unzips a pack into dest.

    Args:
        zip_path: A zip written by pack().
        dest: Where to unpack; created if missing.

    Returns:
        The dataset folder inside dest.

    Raises:
        ValueError: If the zip names a path outside dest, or holds more than
            one top-level folder.
    """
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for name in names:
            target = (dest / name).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"{name!r} would land outside {dest}")
        tops = {pathlib.PurePosixPath(n).parts[0] for n in names}
        if len(tops) != 1:
            raise ValueError(f"expected one top folder, got {sorted(tops)}")
        z.extractall(dest)
    return dest / tops.pop()


def main(argv: Sequence[str] | None = None) -> None:
    """Packs each dataset folder into out/<name>.zip."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for folder in args.folders:
        zip_path = args.out / f"{folder.name}.zip"
        n = pack(folder, zip_path)
        size = zip_path.stat().st_size / 2**20
        print(f"{folder} -> {zip_path}: {n} files, {size:.0f} MB")


if __name__ == "__main__":
    main()
