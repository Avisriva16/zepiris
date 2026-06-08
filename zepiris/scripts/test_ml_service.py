#!/usr/bin/env python3
"""End-to-end test for the ML inference HTTP service.

Starts the FastAPI ML service in a subprocess and sends requests via
MLInferenceClient to verify all endpoints work correctly.

Prerequisites:
    poetry install --extras ml

Usage:
    # Provide model paths and a test image:
    python scripts/test_ml_service.py \
        --test-image path/to/image.jpg \
        --nsfw-model ./models/nsfw_model.pth \
        --spoof-model ./models/spoof_model.pth \
        --blur-model ./models/blur_model.pth

    # Skip face embedding (if AuraFace OR Buffalo_l not installed):
    python scripts/test_ml_service.py \
        --test-image path/to/image.jpg \
        --nsfw-model ./models/nsfw_model.pth \
        --spoof-model ./models/spoof_model.pth \
        --blur-model ./models/blur_model.pth \
        --skip-face-embedding
"""

import argparse
import base64
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zepiris.services.ml_client import MLInferenceClient

SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = 8099
BASE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"
STARTUP_TIMEOUT = 60


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded bytes."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def create_synthetic_image_b64() -> str:
    """Create a simple gradient image and return as base64."""
    import numpy as np

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)
    img[:, :, 1] = np.linspace(255, 0, 224, dtype=np.uint8).reshape(224, 1)
    img[:, :, 2] = 128
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def wait_for_service(client: MLInferenceClient, timeout: int = STARTUP_TIMEOUT) -> None:
    """Poll the health endpoint until the service is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = client.healthz()
            if resp.get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"ML service did not become healthy within {timeout}s")


def run_tests(client: MLInferenceClient, image_b64: str, skip_face: bool) -> None:
    print("\n" + "=" * 70)
    print("  ML Inference Service — End-to-End Tests")
    print("=" * 70)

    # Health
    print("\n[1/6] GET /healthz")
    resp = client.healthz()
    print(f"  Response: {resp}")
    assert resp["status"] == "ok", "Health check failed"
    print("  PASSED")

    # NSFW
    print("\n[2/6] POST /v1/iqa/nsfw_check")
    t0 = time.time()
    nsfw = client.detect_nsfw(image_b64)
    print(f"  Time: {time.time() - t0:.3f}s")
    print(f"  is_safe={nsfw.is_safe}  probability={nsfw.probability:.4f}")
    print("  PASSED")

    # Spoof
    print("\n[3/6] POST /v1/iqa/spoof_check")
    t0 = time.time()
    spoof = client.detect_spoof(image_b64)
    print(f"  Time: {time.time() - t0:.3f}s")
    print(f"  is_live={spoof.is_live}  probability={spoof.probability:.4f}")
    print("  PASSED")

    # Blur
    print("\n[4/6] POST /v1/iqa/blur_check")
    t0 = time.time()
    blur = client.detect_blur(image_b64)
    print(f"  Time: {time.time() - t0:.3f}s")
    print(f"  is_sharp={blur.is_sharp}  probability={blur.probability:.4f}")
    print("  PASSED")

    # Face embedding
    if skip_face:
        print("\n[5/6] POST /v1/face/embed — SKIPPED (--skip-face-embedding)")
    else:
        print("\n[5/6] POST /v1/face/embed")
        t0 = time.time()
        embed = client.embed_face(image_b64)
        print(f"  Time: {time.time() - t0:.3f}s")
        print(
            f"  face_detected={embed.face_detected}  embedding_dim={embed.embedding_dim}  first_5={embed.embedding[:5]}"
        )
        print("  PASSED")

    # Combined IQA
    print("\n[6/6] POST /v1/iqa/assess")
    t0 = time.time()
    assess = client.assess_image_quality(image_b64)
    print(f"  Time: {time.time() - t0:.3f}s")
    print(f"  passed={assess.passed}")
    print(f"    nsfw:   is_safe={assess.nsfw.is_safe}  prob={assess.nsfw.probability:.4f}")
    print(f"    spoof:  is_live={assess.spoof.is_live}  prob={assess.spoof.probability:.4f}")
    print(f"    blur:   is_sharp={assess.blur.is_sharp}  prob={assess.blur.probability:.4f}")
    print("  PASSED")

    print("\n" + "=" * 70)
    print("  All tests passed!")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end test for the ML inference HTTP service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-image",
        type=str,
        help="Path to test image (if omitted, a synthetic image is used)",
    )
    parser.add_argument("--nsfw-model", type=str, help="Local path to NSFW model file")
    parser.add_argument("--spoof-model", type=str, help="Local path to spoof model file")
    parser.add_argument("--blur-model", type=str, help="Local path to blur model file")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-face-embedding", action="store_true")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start the server; assume it's already running at the base URL",
    )
    parser.add_argument("--port", type=int, default=SERVICE_PORT, help="Port for the ML service")
    args = parser.parse_args()

    port = args.port
    base_url = f"http://{SERVICE_HOST}:{port}"

    # Prepare image
    if args.test_image:
        print(f"Loading test image: {args.test_image}")
        image_b64 = encode_image_to_base64(args.test_image)
    else:
        print("Creating synthetic test image...")
        image_b64 = create_synthetic_image_b64()

    print(f"Base64 payload size: {len(image_b64)} chars")

    proc = None
    if not args.no_server:
        env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parent.parent / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["ML_SERVICE_HOST"] = SERVICE_HOST
        env["ML_SERVICE_PORT"] = str(port)
        env["ML_SERVICE_ML_DEVICE"] = args.device

        if args.nsfw_model:
            env["ML_SERVICE_NSFW_LOCAL_MODEL_PATH"] = str(Path(args.nsfw_model).resolve())
        if args.spoof_model:
            env["ML_SERVICE_SPOOF_LOCAL_MODEL_PATH"] = str(Path(args.spoof_model).resolve())
        if args.blur_model:
            env["ML_SERVICE_BLUR_LOCAL_MODEL_PATH"] = str(Path(args.blur_model).resolve())

        print(f"\nStarting ML service on {base_url} ...")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "zepiris.ml_inference.app:app",
                "--host",
                SERVICE_HOST,
                "--port",
                str(port),
            ],
            env=env,
            cwd=str(Path(__file__).parent.parent),
        )

    try:
        client = MLInferenceClient(base_url)
        print("Waiting for service to be ready...")
        wait_for_service(client)
        print("Service is healthy!")

        run_tests(client, image_b64, skip_face=args.skip_face_embedding)
        client.close()
    finally:
        if proc is not None:
            print("\nShutting down ML service...")
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=10)
            print("Service stopped.")


if __name__ == "__main__":
    main()
