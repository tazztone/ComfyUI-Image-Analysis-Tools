import pytest
import torch
from contrast_analysis import ContrastAnalysis
from tests.utils import create_sharp_image, create_low_contrast_image

class TestContrastAnalysis:
    def setup_method(self):
        self.contrast_analyzer = ContrastAnalysis()
        self.high_contrast_image = create_sharp_image(128, 128)
        self.low_contrast_image = create_low_contrast_image(128, 128)

    @pytest.mark.parametrize("method", ["Global", "Local", "Hybrid"])
    @pytest.mark.parametrize("comparison_method", ["Michelson", "RMS", "Weber"])
    def test_contrast_scores(self, method, comparison_method):
        high_contrast_score, _ = self.contrast_analyzer.analyze(
            self.high_contrast_image, method, comparison_method, 32, False
        )
        low_contrast_score, _ = self.contrast_analyzer.analyze(
            self.low_contrast_image, method, comparison_method, 32, False
        )
        assert high_contrast_score > low_contrast_score

    def test_visualize_contrast_map_toggle(self):
        # Test with visualization enabled
        _, contrast_map_vis = self.contrast_analyzer.analyze(self.high_contrast_image, "Local", "RMS", 32, True)
        assert torch.any(contrast_map_vis > 0)

        # Test with visualization disabled
        _, contrast_map_no_vis = self.contrast_analyzer.analyze(self.high_contrast_image, "Local", "RMS", 32, False)
        assert torch.all(contrast_map_no_vis == 0)

    def test_global_method_no_visualization(self):
        # Visualization should be disabled for Global method
        _, contrast_map = self.contrast_analyzer.analyze(self.high_contrast_image, "Global", "RMS", 32, True)
        assert torch.all(contrast_map == 0)
