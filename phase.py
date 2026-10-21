"""Wrapped phase from Lasagna's cos channel, and its residues.

Lasagna predicts a cosine of the winding coordinate w, with the peak on the
papyrus sheet. Measured on PHerc0826 level 2 (2026-10-21): inside the scroll
the stored channel is never zero; it oscillates between about 0.65 and 0.98
from sheet to sheet, so it is a contrast-compressed cosine (the published
pyramid is mean-pooled, and the model's output is soft). Taking arccos of it
directly gives a phase that never reaches +-pi, so nothing ever wraps and the
residue map is blind by construction.

So the phase is built in three steps per slice:

1. Local contrast normalisation: c' = 2 (c - lo) / (hi - lo) - 1, with lo and
   hi the running minimum and maximum over a square window of about one to
   two fringe periods. Then c' = +1 on the sheet and -1 between sheets.
2. Magnitude: |phi| = arccos(c') in [0, pi].
3. Quadrature by the Riesz transform (monogenic signal): R = F^-1(-i k/|k| F(c')).
   Projected on the predicted in-plane normal n, q = R . n is the sine
   component along the direction in which the winding coordinate grows, so
   phi = atan2(q, c'). This needs no pointwise sign decision, which was the
   noise source in the first attempt (balanced +1/-1 residue pairs everywhere).
   The overall sign convention is checked empirically: phi must increase
   along +n (mean of grad phi . n over the valid area is reported).

Between sheets phi reaches +-pi and wraps. Residues are the standard 2x2 loop
sums of wrapped differences; a non-zero residue is a fringe dislocation, which
is where sheets merge, tear, or the normal field flips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, maximum_filter, minimum_filter

TWO_PI = 2.0 * np.pi


@dataclass
class PhaseField:
    phi: np.ndarray  # float32, wrapped to (-pi, pi]; 0 where invalid
    valid: np.ndarray  # bool: inside the scroll and with fringe contrast
    quadrature: np.ndarray  # float32: Riesz component along the normal
    contrast: np.ndarray  # float32: local hi - lo of the stored cos
    normal_agreement: float  # mean of grad phi . n over valid pixels; should be > 0


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % TWO_PI - np.pi


def normalise_contrast(cos: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Map the stored cos onto [-1, 1] per local window; return (c_norm, contrast)."""
    lo = minimum_filter(cos, size=window, mode="nearest")
    hi = maximum_filter(cos, size=window, mode="nearest")
    contrast = hi - lo
    c_norm = np.where(contrast > 0, 2.0 * (cos - lo) / np.maximum(contrast, 1e-6) - 1.0, 0.0)
    return np.clip(c_norm, -1.0, 1.0).astype(np.float32), contrast.astype(np.float32)


def riesz_transform(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Riesz (2D Hilbert) transform of a real image: returns (R_y, R_x)."""
    fa = np.fft.fft2(a)
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.fftfreq(a.shape[1])[None, :]
    k = np.hypot(ky, kx)
    k[0, 0] = 1.0
    ry = np.real(np.fft.ifft2(-1j * ky / k * fa))
    rx = np.real(np.fft.ifft2(-1j * kx / k * fa))
    return ry.astype(np.float32), rx.astype(np.float32)


def wrapped_phase(
    cos: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    window: int = 15,
    min_cos: float = 1e-3,
    min_contrast: float = 0.05,
    smooth_sigma: float = 0.7,
    grad_mag: np.ndarray | None = None,
    min_grad_mag: float = 0.0,
) -> PhaseField:
    """Reconstruct the wrapped phase of one slice.

    cos, nx, ny are (Y, X) float arrays from lasagna_store, indexed [y, x];
    nx pairs with the x axis and ny with the y axis. `window` is the contrast
    window in pixels at the stores' level (about one to two fringe periods).
    """
    cos = cos.astype(np.float32)
    inside = cos > min_cos
    # The contrast window and the Riesz transform both smear the zero border of
    # unprocessed chunks into the scroll; keep only pixels a full window away.
    inside = binary_erosion(inside, np.ones((window, window), bool))
    c_norm, contrast = normalise_contrast(gaussian_filter(cos, smooth_sigma), window)
    valid = inside & (contrast > min_contrast)
    if grad_mag is not None:
        valid &= grad_mag > min_grad_mag
    c_norm = np.where(valid, c_norm, 0.0).astype(np.float32)

    ry, rx = riesz_transform(c_norm)
    q = rx * nx + ry * ny
    phi = np.arctan2(q, c_norm).astype(np.float32)

    # Sign check: the winding coordinate must grow along +n.
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gx[:, :-1] = wrap(phi[:, 1:] - phi[:, :-1])
    gy[:-1, :] = wrap(phi[1:, :] - phi[:-1, :])
    dphi_n = gx * nx + gy * ny
    agreement = float(np.mean(dphi_n[valid])) if valid.any() else 0.0
    if agreement < 0:
        phi = -phi
        agreement = -agreement
    phi = np.where(valid, phi, 0.0).astype(np.float32)
    return PhaseField(phi=phi, valid=valid, quadrature=q.astype(np.float32), contrast=contrast, normal_agreement=agreement)


def residues(field: PhaseField) -> np.ndarray:
    """Integer residue per 2x2 cell, shape (Y-1, X-1); 0 where any corner is invalid."""
    phi, valid = field.phi, field.valid
    d_right = wrap(phi[:-1, 1:] - phi[:-1, :-1])
    d_down = wrap(phi[1:, 1:] - phi[:-1, 1:])
    d_left = wrap(phi[1:, :-1] - phi[1:, 1:])
    d_up = wrap(phi[:-1, :-1] - phi[1:, :-1])
    loop = (d_right + d_down + d_left + d_up) / TWO_PI
    all_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    return np.where(all_valid, np.rint(loop), 0).astype(np.int8)
