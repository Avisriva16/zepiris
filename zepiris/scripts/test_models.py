#!/usr/bin/env python3
"""Test script for ZepIris ML inference models.

Loads configuration from .env (via Settings) and allows CLI overrides.
Supports both local model files and Hugging Face Hub models.

Usage:
    # Use .env defaults (set ZEPIRIS_*_LOCAL_MODEL_PATH in .env):
    poetry install --extras ml
    poetry run python scripts/test_models.py --test-image path/to/test/image.jpg

    # Override model paths from CLI:
    poetry run python scripts/test_models.py \\
        --test-image path/to/test/image.jpg \\
        --nsfw-model ./models/nsfw_detection.pt \\
        --spoof-model ./models/spoof_detection.pt \\
        --blur-model ./models/blur_detection.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zepiris.config import get_settings
from zepiris.ml_inference.blur_detection import BlurDetectionService
from zepiris.ml_inference.face_embedding import FaceEmbeddingService
from zepiris.ml_inference.image_quality_assessment import ImageQualityAssessmentService
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
from zepiris.ml_inference.spoof_detection import SpoofDetectionService


def load_image(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def test_nsfw_detection(image_rgb: np.ndarray, **kwargs) -> None:
    print("\n" + "=" * 70)
    print("Testing NSFW Detection")
    print("=" * 70)

    service = NSFWDetectionService(**kwargs)

    start = time.time()
    result = service.forward(image_rgb)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Is safe: {result.is_safe}")
    print(f"Probability (safe): {result.probability:.4f}")


def test_spoof_detection(image_rgb: np.ndarray, **kwargs) -> None:
    print("\n" + "=" * 70)
    print("Testing Spoof Detection")
    print("=" * 70)

    service = SpoofDetectionService(**kwargs)

    start = time.time()
    result = service.forward(image_rgb)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Is live: {result.is_live}")
    print(f"Probability (live): {result.probability:.4f}")


def test_blur_detection(image_rgb: np.ndarray, **kwargs) -> None:
    print("\n" + "=" * 70)
    print("Testing Blur Detection")
    print("=" * 70)

    service = BlurDetectionService(**kwargs)

    start = time.time()
    result = service.forward(image_rgb)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Is sharp: {result.is_sharp}")
    print(f"Probability (sharp): {result.probability:.4f}")


def test_face_embedding(image_rgb: np.ndarray, **kwargs) -> None:
    print("\n" + "=" * 70)
    print("Testing Face Embedding")
    print("=" * 70)

    service = FaceEmbeddingService(**kwargs)

    start = time.time()
    result = service.embed(image_rgb)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Face detected: {result.face_detected}")
    print(f"Embedding dimension: {result.embedding_dim}")
    print(f"Embedding (first 5 values): {result.embedding[:5]}")


def test_image_quality_assessment(image_rgb: np.ndarray, **kwargs) -> None:
    print("\n" + "=" * 70)
    print("Testing Image Quality Assessment (Parallel Inference)")
    print("=" * 70)

    service = ImageQualityAssessmentService(**kwargs)

    start = time.time()
    result = service.assess(image_rgb)
    elapsed = time.time() - start

    print(f"\nTime (all 3 models parallel): {elapsed:.3f}s")
    print(f"\nOverall Quality Assessment: {'PASSED' if result.passed else 'FAILED'}")
    print("\nIndividual Results:")
    print("  NSFW Detection:")
    print(f"    - Is safe: {result.nsfw.is_safe}")
    print(f"    - Probability (safe): {result.nsfw.probability:.4f}")
    print("  Spoof Detection:")
    print(f"    - Is live: {result.spoof.is_live}")
    print(f"    - Probability (live): {result.spoof.probability:.4f}")
    print("  Blur Detection:")
    print(f"    - Is sharp: {result.blur.is_sharp}")
    print(f"    - Probability (sharp): {result.blur.probability:.4f}")


def create_test_image(height: int = 224, width: int = 224) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j] = [
                int(255 * i / height),
                int(255 * j / width),
                int(128 + 127 * np.sin((i + j) / 50)),
            ]
    return img


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Test ZepIris ML inference models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-image",
        type=str,
        help="Path to test image (if omitted, a synthetic image is created)",
    )
    parser.add_argument(
        "--nsfw-model",
        type=str,
        help="Local path to NSFW model (overrides ZEPIRIS_NSFW_LOCAL_MODEL_PATH)",
    )
    parser.add_argument(
        "--spoof-model",
        type=str,
        help="Local path to spoof model (overrides ZEPIRIS_SPOOF_LOCAL_MODEL_PATH)",
    )
    parser.add_argument(
        "--blur-model",
        type=str,
        help="Local path to blur model (overrides ZEPIRIS_BLUR_LOCAL_MODEL_PATH)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Inference device (overrides ZEPIRIS_ML_DEVICE)",
    )
    parser.add_argument(
        "--skip-face-embedding",
        action="store_true",
        help="Skip face embedding test (requires AuraFace OR Buffalo_l)",
    )

    args = parser.parse_args()

    device = args.device or settings.ml_device
    nsfw_local = args.nsfw_model or settings.nsfw_local_model_path or None
    spoof_local = args.spoof_model or settings.spoof_local_model_path or None
    blur_local = args.blur_model or settings.blur_local_model_path or None

    if args.test_image:
        print(f"Loading test image: {args.test_image}")
        image_rgb = load_image(args.test_image)
    else:
        print("Creating synthetic test image...")
        image_rgb = create_test_image()

    print(f"Image shape: {image_rgb.shape}, dtype: {image_rgb.dtype}")
    print(f"Device: {device}")

    nsfw_kwargs = dict(
        huggingface_repo_id=settings.nsfw_hf_repo_id,
        huggingface_model_file=settings.nsfw_hf_model_file,
        local_model_path=nsfw_local,
        nsfw_threshold=settings.nsfw_threshold,
        device=device,
    )
    spoof_kwargs = dict(
        huggingface_repo_id=settings.spoof_hf_repo_id,
        huggingface_model_file=settings.spoof_hf_model_file,
        local_model_path=spoof_local,
        spoof_threshold=settings.spoof_threshold,
        device=device,
    )
    blur_kwargs = dict(
        huggingface_repo_id=settings.blur_hf_repo_id,
        huggingface_model_file=settings.blur_hf_model_file,
        local_model_path=blur_local,
        blur_threshold=settings.blur_threshold,
        device=device,
    )

    # Individual model tests
    try:
        test_nsfw_detection(image_rgb, **nsfw_kwargs)
    except Exception as e:
        print(f"Error in NSFW detection test: {e}")

    try:
        test_spoof_detection(image_rgb, **spoof_kwargs)
    except Exception as e:
        print(f"Error in spoof detection test: {e}")

    try:
        test_blur_detection(image_rgb, **blur_kwargs)
    except Exception as e:
        print(f"Error in blur detection test: {e}")

    if not args.skip_face_embedding:
        try:
            test_face_embedding(
                image_rgb,
                embedding_dim=settings.face_embedding_dim,
                detection_size=(settings.face_detection_width, settings.face_detection_height),
                facial_area_threshold=settings.face_area_threshold,
                device=device,
            )
        except Exception as e:
            print(f"Error in face embedding test: {e}")

    # Combined parallel inference test
    try:
        test_image_quality_assessment(
            image_rgb,
            nsfw_repo_id=settings.nsfw_hf_repo_id,
            nsfw_model_file=settings.nsfw_hf_model_file,
            nsfw_local_path=nsfw_local,
            nsfw_threshold=settings.nsfw_threshold,
            spoof_repo_id=settings.spoof_hf_repo_id,
            spoof_model_file=settings.spoof_hf_model_file,
            spoof_local_path=spoof_local,
            spoof_threshold=settings.spoof_threshold,
            blur_repo_id=settings.blur_hf_repo_id,
            blur_model_file=settings.blur_hf_model_file,
            blur_local_path=blur_local,
            blur_threshold=settings.blur_threshold,
            device=device,
        )
    except Exception as e:
        print(f"Error in image quality assessment test: {e}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
