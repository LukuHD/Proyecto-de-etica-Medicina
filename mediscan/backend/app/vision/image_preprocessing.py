from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class FFTFilterConfig:
    notch_radius: int = 6
    max_notches: int = 4


class ImagePreprocessor:
    """Image preprocessing utilities optimized for edge hardware execution."""

    @staticmethod
    def extract_zero_copy_frame_buffer(frame: np.ndarray) -> memoryview:
        """
        Return a zero-copy view over a contiguous frame buffer.

        Complexity: O(1)
        Formula: no transformation, only memory view creation.
        """
        if not frame.flags.c_contiguous:
            raise ValueError("Frame must be C-contiguous for zero-copy extraction.")
        return memoryview(frame)

    @staticmethod
    def laplacian_variance(gray: UInt8Array) -> float:
        """
        Compute blur score using Laplacian variance.

        Complexity: O(H*W)
        Mathematical steps:
        1) Discrete Laplacian: L(x,y) = I(x+1,y)+I(x-1,y)+I(x,y+1)+I(x,y-1)-4I(x,y)
        2) Variance: sigma^2 = (1/N) * sum_i (L_i - mean(L))^2
        """
        image = gray.astype(np.float64, copy=False)

        center = image[1:-1, 1:-1]
        laplacian = (
            image[1:-1, 2:]
            + image[1:-1, :-2]
            + image[2:, 1:-1]
            + image[:-2, 1:-1]
            - 4.0 * center
        )
        return float(np.var(laplacian, dtype=np.float64))

    @staticmethod
    def fft_periodic_noise_filter(
        gray: UInt8Array,
        config: FFTFilterConfig = FFTFilterConfig(),
        notch_centers: Sequence[tuple[int, int]] | None = None,
    ) -> UInt8Array:
        """
        Remove periodic sensor noise in the frequency domain using notch rejection.

        Complexity: O(H*W*log(H*W)) due to FFT.
        Mathematical steps:
        1) F(u,v) = FFT2(I(x,y))
        2) Shift zero-frequency to center.
        3) Apply notch mask M(u,v) where M=0 inside radius r at ±(u_k,v_k), else 1.
        4) G(u,v) = F(u,v) * M(u,v)
        5) I'(x,y) = Re(IFFT2(G(u,v)))
        """
        image = gray.astype(np.float64, copy=False)
        spectrum = np.fft.fftshift(np.fft.fft2(image))

        rows, cols = image.shape
        center_r, center_c = rows // 2, cols // 2

        if notch_centers is None:
            notch_centers = tuple(
                ImagePreprocessor._detect_periodic_peaks(
                    np.abs(spectrum), config.max_notches, (center_r, center_c)
                )
            )

        mask = np.ones((rows, cols), dtype=np.float64)
        rr, cc = np.ogrid[:rows, :cols]

        for dr, dc in notch_centers:
            r1, c1 = center_r + dr, center_c + dc
            r2, c2 = center_r - dr, center_c - dc

            if 0 <= r1 < rows and 0 <= c1 < cols:
                mask[(rr - r1) ** 2 + (cc - c1) ** 2 <= config.notch_radius**2] = 0.0
            if 0 <= r2 < rows and 0 <= c2 < cols:
                mask[(rr - r2) ** 2 + (cc - c2) ** 2 <= config.notch_radius**2] = 0.0

        filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
        return np.clip(filtered, 0, 255).astype(np.uint8)

    @staticmethod
    def _detect_periodic_peaks(
        magnitude: FloatArray,
        max_notches: int,
        center: tuple[int, int],
    ) -> Iterable[tuple[int, int]]:
        """Find strongest off-center peaks in FFT magnitude to build notch filters."""
        rows, cols = magnitude.shape
        center_r, center_c = center

        mag = magnitude.copy()

        exclusion = min(rows, cols) // 16
        rr, cc = np.ogrid[:rows, :cols]
        mag[(rr - center_r) ** 2 + (cc - center_c) ** 2 <= exclusion**2] = 0

        flat_indices = np.argpartition(mag.ravel(), -max_notches)[-max_notches:]

        selected: list[tuple[int, int]] = []
        for idx in flat_indices:
            r, c = divmod(int(idx), cols)
            dr, dc = r - center_r, c - center_c
            if dr == 0 and dc == 0:
                continue
            selected.append((dr, dc))

        selected.sort(key=lambda offset: mag[center_r + offset[0], center_c + offset[1]], reverse=True)
        return selected[:max_notches]
