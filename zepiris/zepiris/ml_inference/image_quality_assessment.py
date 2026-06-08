"""Image quality assessment combining NSFW, spoof, and blur detection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService
from zepiris.schemas.ml_inference import ImageQualityAssessmentResult


class ImageQualityAssessmentService:
    """Image quality assessment service combining three quality checks.

    Runs NSFW detection, spoof detection, and blur detection on an image
    in parallel using a thread pool. Accepts pre-loaded service instances
    to avoid redundant model loading.
    """

    def __init__(
        self,
        nsfw_service: NSFWDetectionService,
        spoof_service: SpoofDetectionService,
        blur_service: BlurDetectionService,
    ) -> None:
        self.nsfw_service = nsfw_service
        self.spoof_service = spoof_service
        self.blur_service = blur_service
        self._executor = ThreadPoolExecutor(max_workers=3)

    def assess(self, image_rgb: np.ndarray) -> ImageQualityAssessmentResult:
        """Run image quality assessment on all three checks in parallel.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            ImageQualityAssessmentResult: Aggregated quality assessment result.
                passed=True if all checks pass (is_safe AND is_live AND is_sharp),
                otherwise passed=False
        """
        nsfw_future = self._executor.submit(self.nsfw_service.forward, image_rgb)
        spoof_future = self._executor.submit(self.spoof_service.forward, image_rgb)
        blur_future = self._executor.submit(self.blur_service.forward, image_rgb)

        nsfw_result = nsfw_future.result()
        spoof_result = spoof_future.result()
        blur_result = blur_future.result()

        passed = nsfw_result.is_safe and spoof_result.is_live and blur_result.is_sharp

        return ImageQualityAssessmentResult(
            passed=passed,
            nsfw=nsfw_result,
            spoof=spoof_result,
            blur=blur_result,
        )
