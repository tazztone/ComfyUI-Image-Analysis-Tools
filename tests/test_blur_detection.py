import pytest
import torch
from blur_detection import BlurDetection
from tests.utils import create_sharp_image, create_blurry_image

class TestBlurDetection:
    def setup_method(self):
        self.blur_detector = BlurDetection()

    def test_interpret_blur(self):
        assert "Very blurry" in self.blur_detector.interpret_blur(49)
        assert "Slightly blurry" in self.blur_detector.interpret_blur(149)
        assert "Acceptably sharp" in self.blur_detector.interpret_blur(299)
        assert "Very sharp" in self.blur_detector.interpret_blur(300)

    def test_analyze_sharp_image(self):
        sharp_image = create_sharp_image(128, 128)
        score, _, interpretation = self.blur_detector.analyze(sharp_image, 32, False)
        assert score > 150  # Expect a high score for a sharp image
        assert "sharp" in interpretation

    def test_analyze_blurry_image(self):
        sharp_image = create_sharp_image(128, 128)
        blurry_image = create_blurry_image(sharp_image, kernel_size=25)
        score, _, interpretation = self.blur_detector.analyze(blurry_image, 32, False)
        assert score < 150  # Expect a low score for a blurry image
        assert "blurry" in interpretation

    def test_visualize_blur_map_toggle(self):
        sharp_image = create_sharp_image(128, 128)
        # Test with visualization enabled
        _, blur_map_vis, _ = self.blur_detector.analyze(sharp_image, 32, True)
        assert torch.any(blur_map_vis > 0)

        # Test with visualization disabled
        _, blur_map_no_vis, _ = self.blur_detector.analyze(sharp_image, 32, False)
        assert torch.all(blur_map_no_vis == 0)

    def test_block_size_variation(self):
        sharp_image = create_sharp_image(128, 128)
        score_32, _, _ = self.blur_detector.analyze(sharp_image, 32, False)
        score_64, _, _ = self.blur_detector.analyze(sharp_image, 64, False)
        assert score_32 != score_64
        assert score_32 > 150
        assert score_64 > 150
