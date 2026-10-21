"""Read Lasagna channels and the CT volume for one scroll from the open-data bucket.

The published Lasagna product is four separate OME-Zarr pyramids (cos, grad_mag,
nx, ny), all uint8. Encodings, from villa's lasagna/omezarr_pyramid.py and
preprocess_cos_omezarr.py:

    cos       = clip(cos * 255, 0, 255)      -> c  = u8 / 255,   one-sided [0, 1]
    grad_mag  = clip(grad_mag * 1000, 0, 255) -> g  = u8 / 1000
    nx, ny    = n * 127 + 128                 -> n  = (u8 - 128) / 127

Everything here is a plain read; nothing is written to the bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import zarr
from scipy.ndimage import zoom

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


@dataclass(frozen=True)
class ScrollStores:
    """URLs of one scroll's Lasagna product and CT volume at one pyramid level."""

    scroll: str
    volume_id: str
    lasagna_model: str
    ct_volume_name: str
    level: int

    @property
    def lasagna_root(self) -> str:
        return (
            f"{BUCKET}/{self.scroll}/representations/predictions/lasagna/"
            f"{self.volume_id}-lasagna-{self.lasagna_model}"
        )

    def channel_url(self, channel: str) -> str:
        return f"{self.lasagna_root}/{self.scroll}_{channel}.ome.zarr/{self.level}"

    @property
    def ct_url(self) -> str:
        return f"{BUCKET}/{self.scroll}/volumes/{self.ct_volume_name}/{self.level}"


# Verified 2026-10-21 against the bucket listing and .zarray files.
PHERC0826_L2 = ScrollStores(
    scroll="PHerc0826",
    volume_id="20250821151701",
    lasagna_model="20260419180421",
    ct_volume_name="20250821151701-9.362um-1.2m-113keV-masked.zarr",
    level=2,
)


@dataclass
class SliceWindow:
    """One z slice of the decoded Lasagna channels and the CT, over a y/x window."""

    z: int
    y0: int
    x0: int
    cos: np.ndarray  # float32 in [0, 1]
    grad_mag: np.ndarray  # float32
    nx: np.ndarray  # float32 in [-1, 1]
    ny: np.ndarray  # float32 in [-1, 1]
    ct: np.ndarray  # uint8


def open_array(url: str) -> zarr.Array:
    return zarr.open_array(store=url, mode="r")


def decode_cos(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def decode_grad_mag(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 1000.0


def decode_normal(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) - 128.0) / 127.0


def read_slice_window(
    stores: ScrollStores, z: int, y0: int, y1: int, x0: int, x1: int
) -> SliceWindow:
    """Read one z slice over [y0:y1, x0:x1] at the stores' pyramid level."""
    sel = (z, slice(y0, y1), slice(x0, x1))
    cos_u8 = open_array(stores.channel_url("cos"))[sel]
    grad_u8 = open_array(stores.channel_url("grad_mag"))[sel]
    nx_u8 = open_array(stores.channel_url("nx"))[sel]
    ny_u8 = open_array(stores.channel_url("ny"))[sel]
    ct = open_array(stores.ct_url)[sel]
    return SliceWindow(
        z=z,
        y0=y0,
        x0=x0,
        cos=decode_cos(cos_u8),
        grad_mag=decode_grad_mag(grad_u8),
        nx=decode_normal(nx_u8),
        ny=decode_normal(ny_u8),
        ct=np.asarray(ct),
    )


def read_slice_window_cos_level1(
    stores: ScrollStores, z: int, y0: int, y1: int, x0: int, x1: int
) -> SliceWindow:
    """Like read_slice_window, but cos at pyramid level 1 (2x finer) with the
    level-2 normals, grad_mag and CT upsampled 2x (linear) to match.

    z, y0, y1, x0, x1 are level-2 coordinates; the returned window is at
    level-1 resolution with origin (2*y0, 2*x0) and z index 2*z.
    """
    l2 = read_slice_window(stores, z, y0, y1, x0, x1)
    cos_l1_url = f"{stores.lasagna_root}/{stores.scroll}_cos.ome.zarr/1"
    cos_u8 = open_array(cos_l1_url)[2 * z, 2 * y0 : 2 * y1, 2 * x0 : 2 * x1]
    up = lambda a: zoom(a, 2, order=1)  # noqa: E731
    return SliceWindow(
        z=2 * z,
        y0=2 * y0,
        x0=2 * x0,
        cos=decode_cos(cos_u8),
        grad_mag=up(l2.grad_mag),
        nx=up(l2.nx),
        ny=up(l2.ny),
        ct=up(l2.ct.astype(np.float32)).astype(np.uint8),
    )
