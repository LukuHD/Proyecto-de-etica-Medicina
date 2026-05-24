import unittest

import numpy as np

from mediscan.backend.app.vision.image_preprocessing import FFTFilterConfig, ImagePreprocessor


class TestImagePreprocessing(unittest.TestCase):
    def test_zero_copy_frame_buffer(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        buffer = ImagePreprocessor.extract_zero_copy_frame_buffer(frame)
        frame[0, 0, 0] = 123
        self.assertEqual(buffer.cast("B")[0], 123)

    def test_laplacian_variance_constant_image(self) -> None:
        image = np.full((32, 32), 128, dtype=np.uint8)
        score = ImagePreprocessor.laplacian_variance(image)
        self.assertEqual(score, 0.0)

    def test_fft_filter_reduces_periodic_component(self) -> None:
        size = 128
        x = np.arange(size)
        periodic = 30.0 * np.sin(2.0 * np.pi * 10.0 * x / size)
        image = np.tile(120.0 + periodic, (size, 1)).astype(np.float64)
        noisy = np.clip(image, 0, 255).astype(np.uint8)

        filtered = ImagePreprocessor.fft_periodic_noise_filter(
            noisy,
            config=FFTFilterConfig(notch_radius=4, max_notches=2),
            notch_centers=[(0, 10)],
        )

        before_std = float(np.std(noisy[0].astype(np.float64)))
        after_std = float(np.std(filtered[0].astype(np.float64)))

        self.assertLess(after_std, before_std)


if __name__ == "__main__":
    unittest.main()
