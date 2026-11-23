"""Loader for the NYU Depth V2 labeled set.

1,449 indoor photos from a Kinect, each with a depth map in metres and a
hand-drawn label for every pixel. Official split: 795 train, 654 test. We
report on test only and never tune on it.

Files (download into data/nyuv2/, which is gitignored):
    nyu_depth_v2_labeled.mat  MATLAB v7.3, which is HDF5 inside (~3 GB).
    splits.mat                The official split, older MATLAB format.

Two traps in the .mat file, both handled here:
    1. MATLAB stores arrays column-major, so h5py returns every image with
       its axes reversed: (3, 640, 480) for channels, width, height. We want
       (480, 640, 3). Getting this wrong gives sideways images and no error.
    2. Label 0 means nobody labeled that pixel. Those pixels must be left out
       of the metrics (the `valid` argument of metrics.floor_iou).
"""

import dataclasses
import pathlib
import types

import h5py
import numpy as np
import scipy.io

FLOOR_CLASS_NAME = "floor"
UNLABELED = 0


@dataclasses.dataclass(frozen=True, eq=False)
class Sample:
    """One NYUv2 image with its ground truth.

    Attributes:
        index: 0-based position in the .mat file.
        rgb: (H, W, 3) uint8 colour image. H = 480, W = 640 in the real set.
        depth: (H, W) float32 depth in metres.
        floor: (H, W) bool, True where the label is floor.
        labeled: (H, W) bool, False where nobody labeled the pixel.
    """

    index: int
    rgb: np.ndarray
    depth: np.ndarray
    floor: np.ndarray
    labeled: np.ndarray


def matlab_to_hwc(image: np.ndarray) -> np.ndarray:
    """Converts one MATLAB image from (C, W, H) to (H, W, C).

    Args:
        image: One image as h5py returns it, axes (channels, width, height).

    Returns:
        The same pixels with axes (height, width, channels).
    """
    # Input axes: 0 = C, 1 = W, 2 = H. The output wants H first (input axis
    # 2), then W (axis 1), then C (axis 0). For 3 axes that is a full
    # reversal, the same as image.transpose() with no arguments; the explicit
    # order says what we mean.
    return np.transpose(image, (2, 1, 0))


def matlab_to_hw(array: np.ndarray) -> np.ndarray:
    """Converts one MATLAB 2-D map (depth or labels) from (W, H) to (H, W)."""
    # With two axes, reversing them is a plain transpose.
    return array.T


def load_split(splits_path: str | pathlib.Path) -> dict[str, list[int]]:
    """Reads the official train/test split.

    Args:
        splits_path: Path to splits.mat.

    Returns:
        {"train": [...], "test": [...]}, each a sorted list of 0-based
        indices into the .mat file.
    """
    mat = scipy.io.loadmat(str(splits_path))
    # Each comes back as a column of MATLAB indices, shape (n, 1).
    train = mat["trainNdxs"].ravel()
    test = mat["testNdxs"].ravel()
    # MATLAB counts from 1, Python from 0, so subtract 1. int() turns each
    # NumPy float into a plain Python int.
    return {
        "train": sorted(int(i) - 1 for i in train),
        "test": sorted(int(i) - 1 for i in test),
    }


class NyuV2:
    """Reads samples from nyu_depth_v2_labeled.mat on demand.

    The file stays open and only the requested image is read, so the 3 GB
    file never has to fit in memory. Use it in a `with` block, or call
    close() when done.

    Example:
        with nyuv2.NyuV2("data/nyuv2/nyu_depth_v2_labeled.mat") as data:
            sample = data[0]
    """

    def __init__(self, mat_path: str | pathlib.Path) -> None:
        """Opens the file and finds the floor class.

        Args:
            mat_path: Path to nyu_depth_v2_labeled.mat.

        Raises:
            ValueError: If the file has no class named "floor".
        """
        self._file = h5py.File(mat_path, "r")
        self._floor_id = self._class_id(FLOOR_CLASS_NAME)

    def __len__(self) -> int:
        """Number of images in the file."""
        return int(self._file["images"].shape[0])

    def __getitem__(self, index: int) -> Sample:
        """Reads one sample.

        Raises:
            IndexError: If index is outside 0..len-1.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range 0..{len(self) - 1}")
        rgb = matlab_to_hwc(self._file["images"][index])
        depth = matlab_to_hw(self._file["depths"][index]).astype(np.float32)
        labels = matlab_to_hw(self._file["labels"][index])
        return Sample(
            index=index,
            rgb=rgb,
            depth=depth,
            floor=labels == self._floor_id,
            labeled=labels != UNLABELED,
        )

    def close(self) -> None:
        """Closes the file."""
        self._file.close()

    def __enter__(self) -> "NyuV2":
        """Returns self, for use in a `with` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Closes the file when the `with` block ends."""
        self.close()

    def _class_id(self, name: str) -> int:
        """Returns the label value used for a class name.

        The file's `names` dataset holds one reference per class. Each points
        to the name stored as MATLAB character codes. Label value k means
        names[k - 1], because label 0 is reserved for unlabeled pixels.
        """
        names = [
            "".join(chr(int(c)) for c in self._file[ref][()].ravel())
            for ref in self._file["names"][0]
        ]
        if name not in names:
            raise ValueError(f"no class named {name!r} in the file")
        return names.index(name) + 1
