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

# Every eligible scroll carries a Lasagna run from the same model with the same
# pyramid levels (inventory 2026-10-23); add a scroll here and in data/.
SCROLLS = {
    "PHerc0826": PHERC0826_L2,
    "PHerc0211": ScrollStores(
        scroll="PHerc0211",
        volume_id="20250821151803",
        lasagna_model="20260419180421",
        ct_volume_name="20250821151803-9.362um-1.2m-113keV-masked.zarr",
        level=2,
    ),
}


def level_shape(stores: "ScrollStores") -> tuple[int, int, int]:
    """(z, y, x) shape of the scroll's cos store at the stores' level."""
    return tuple(int(v) for v in open_array(stores.channel_url("cos")).shape)


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
    # Level 1 is ceil(level0 / 2) wide, so a doubled level-2 window can overrun
    # it by one pixel at the slice edge; resample to the cos array's exact shape.
    target = cos_u8.shape

    def up(a: np.ndarray) -> np.ndarray:
        factors = (target[0] / a.shape[0], target[1] / a.shape[1])
        return zoom(a, factors, order=1)

    # Normals are an unoriented axis (n and -n are the same sample), so they
    # are interpolated as a structure tensor, never as vectors (villa lasagna
    # docs, confirmed by the author 2026-10-21).
    nx_up, ny_up = upsample_normals_tensor(l2.nx, l2.ny, up)

    return SliceWindow(
        z=2 * z,
        y0=2 * y0,
        x0=2 * x0,
        cos=decode_cos(cos_u8),
        grad_mag=up(l2.grad_mag),
        nx=nx_up,
        ny=ny_up,
        ct=up(l2.ct.astype(np.float32)).astype(np.uint8),
    )


def upsample_normals_tensor(nx: np.ndarray, ny: np.ndarray, up) -> tuple[np.ndarray, np.ndarray]:
    """Resample an unoriented in-plane normal field through its structure tensor.

    The tensor components nx^2, ny^2, nx*ny are interpolated with `up`, and the
    principal eigenvector is taken back; its in-plane length is kept so that
    pixels whose normal points mostly along z stay short.
    """
    txx, tyy, txy = up(nx * nx), up(ny * ny), up(nx * ny)
    theta = 0.5 * np.arctan2(2.0 * txy, txx - tyy)
    length = np.sqrt(np.clip(txx + tyy, 0.0, None))
    return (length * np.cos(theta)).astype(np.float32), (length * np.sin(theta)).astype(np.float32)
