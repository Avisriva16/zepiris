"""ML inference services for ZepIris.

Includes face embedding, spoof detection, NSFW detection, blur detection,
and combined image quality assessment.
All models are downloaded from Hugging Face Hub.
"""

from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.face_embedding import FaceEmbeddingService
from zepiris.ml_inference.image_quality_assessment import ImageQualityAssessmentService
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService

__all__ = [
    "FaceEmbeddingService",
    "SpoofDetectionService",
    "NSFWDetectionService",
    "BlurDetectionService",
    "ImageQualityAssessmentService",
]
