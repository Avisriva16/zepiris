# ML Inference Microservice

The ML inference service runs in a separate Docker container and exposes an HTTP API for model inference (NSFW detection, spoof detection, blur detection, and face embedding).

## Architecture

```
Main Service (port 8000)
    ↓ HTTP JSON (base64-encoded images)
ML Inference Service (port 8001)
    ├─ NSFW Detection (MobileNetV2)
    ├─ Spoof Detection (MobileNetV3)
    ├─ Blur Detection (ResNet18)
    └─ Face Embedding (AuraFace OR Buffalo_l based on the flag ML_SERVICE_FACE_MODEL set in .env)
```

## Files

**API Service Code** (all in `zepiris/ml_inference/`):

- `app.py` — FastAPI app, `MLServiceSettings`, lifespan (model loading), app factory
- `model_loading.py` — Model weight loading adapter (`auto` / `local` / `huggingface`)
- `routes.py` — All 6 endpoints (health, NSFW, spoof, blur, embed, assess)
- `deps.py` — FastAPI dependency injection for model services

**Docker**:

- `ml_inference.Dockerfile` — Builds the ML inference container (includes model files)

**Client**:

- `zepiris/services/ml_client.py` — HTTP client for the main app to call the ML service

**Test Scripts**:

- `scripts/test_ml_service.py` — End-to-end test: starts the service and hits all endpoints
- `scripts/test_models.py` — Direct in-process model testing (no HTTP)

**Configuration**:

- `.env.example` — `ML_SERVICE_`* prefixed settings for the inference container

## Running the ML Inference Service

### Standalone (Development)

```bash
# Install dependencies including optional ML packages (virtualenv: ./.venv)
poetry install --extras ml

# Run the service
poetry run zepiris-ml-inference-api
# or
poetry run uvicorn zepiris.ml_inference.app:app --host 0.0.0.0 --port 8001
```

### Docker (Production)

Model files are baked into the image from the `models/` directory at the project root.

```bash
# Place model files in the models/ directory:
#   models/nsfw_model.pth
#   models/spoof_model.pth
#   models/blur_model.pth

# Build the container
docker build -f ml_inference.Dockerfile -t zepiris-ml-service .

# Run the container (models already at /app/models/ inside the image)
# Default env vars already point to /app/models/, so just run:
docker run -p 8001:8001 zepiris-ml-service

# Or override device (e.g., for GPU):
docker run -p 8001:8001 \
  -e ML_SERVICE_ML_DEVICE=cuda \
  zepiris-ml-service
```

## API Endpoints

All POST endpoints accept a JSON body:

```json
{
  "image_b64": "<base64-encoded image bytes>"
}
```

The service decodes the base64 string, parses it as an image (JPEG/PNG) using OpenCV, and converts to RGB before passing to models.

### Health

- `GET /healthz` → `{"status": "ok"}`

### Individual Models

- `POST /v1/iqa/nsfw_check` → `NSFWDetectionResult`
- `POST /v1/iqa/spoof_check` → `SpoofDetectionResult`
- `POST /v1/iqa/blur_check` → `BlurDetectionResult`
- `POST /v1/face/embed` → `FaceEmbeddingResult`

### Combined Assessment

- `POST /v1/iqa/assess` → `ImageQualityAssessmentResult` (runs NSFW + spoof + blur in parallel)

## Configuration

Set environment variables with prefix `ML_SERVICE_`:

```bash
# Service
ML_SERVICE_HOST=0.0.0.0
ML_SERVICE_PORT=8001

# Device
ML_SERVICE_ML_DEVICE=cpu

# NSFW model
ML_SERVICE_NSFW_MODEL_SOURCE=auto          # auto | local | huggingface
ML_SERVICE_NSFW_HF_REPO_ID=
ML_SERVICE_NSFW_HF_MODEL_FILE=nsfw_model.pth
ML_SERVICE_NSFW_LOCAL_MODEL_PATH=/app/models/nsfw_model.pth
ML_SERVICE_NSFW_THRESHOLD=0.5

# Spoof model
ML_SERVICE_SPOOF_MODEL_SOURCE=auto           # auto | local | huggingface
ML_SERVICE_SPOOF_HF_REPO_ID=
ML_SERVICE_SPOOF_HF_MODEL_FILE=spoof_model.pth
ML_SERVICE_SPOOF_LOCAL_MODEL_PATH=/app/models/spoof_model.pth
ML_SERVICE_SPOOF_THRESHOLD=0.5

# Blur model
ML_SERVICE_BLUR_MODEL_SOURCE=auto            # auto | local | huggingface
ML_SERVICE_BLUR_HF_REPO_ID=
ML_SERVICE_BLUR_HF_MODEL_FILE=blur_model.pth
ML_SERVICE_BLUR_LOCAL_MODEL_PATH=/app/models/blur_model.pth
ML_SERVICE_BLUR_THRESHOLD=0.5

# Face embedding (AuraFace OR Buffalo_l — downloads its own model on first startup)
ML_SERVICE_FACE_MODEL=auraface
# Setting this to "auraface" uses fal/AuraFace-v1 model for face detection and embedding generation. Other acceptable value: "buffalo_l" sets the model to InsightFace's buffalo_l.
# NOTE: The use of InsightFace's buffalo_l model weights are licensed for non-commercial research purposes only. Commercial use requires a separate license. Contact: recognition-oss-pack@insightface.ai

ML_SERVICE_FACE_EMBEDDING_DIM=512
ML_SERVICE_FACE_DETECTION_WIDTH=640
ML_SERVICE_FACE_DETECTION_HEIGHT=640
ML_SERVICE_FACE_AREA_THRESHOLD=0.01
```

## Client Usage (from Main Service)

```python
import base64
import cv2
from zepiris.services.ml_client import MLInferenceClient

client = MLInferenceClient("http://localhost:8001")

# Load image and encode to base64
image_bgr = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
_, image_bytes = cv2.imencode(".jpg", image_rgb)
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Run individual checks
nsfw_result = client.detect_nsfw(image_b64)
spoof_result = client.detect_spoof(image_b64)
blur_result = client.detect_blur(image_b64)

# Generate face embedding
embed_result = client.embed_face(image_b64)

# Or run all three quality checks in parallel
quality_result = client.assess_image_quality(image_b64)
print(f"Passed: {quality_result.passed}")

client.close()
```

## Model Loading & Startup

**All models are eagerly loaded on service startup** (in the lifespan context manager) to avoid request timeouts.

Each model has a `*_MODEL_SOURCE` env var (`auto` | `local` | `huggingface`) that controls where weights are loaded from:

| Value | Behaviour |
|---|---|
| `auto` (default) | Use the local file if it exists on disk, otherwise download from Hugging Face Hub. |
| `local` | **Only** load from `*_LOCAL_MODEL_PATH`. Fails immediately if the file is missing. |
| `huggingface` | **Only** download from `*_HF_REPO_ID`. Ignores any local file. |

The loading logic lives in `model_loading.py` and is called by the `ModelService._download_model()` base method.

**Service initialization flow:**
1. **NSFW model** loaded (per `ML_SERVICE_NSFW_MODEL_SOURCE`)
2. **Spoof model** loaded (per `ML_SERVICE_SPOOF_MODEL_SOURCE`)
3. **Blur model** loaded (per `ML_SERVICE_BLUR_MODEL_SOURCE`)
4. **Face embedding model** loaded (AuraFace OR InsightFace's Buffalo_l built-in mechanism based on the flag ML_SERVICE_FACE_MODEL set in .env)
5. **IQA service** created — reuses the already-loaded NSFW/spoof/blur service instances (no redundant model loading)

When a request arrives, all models are already in memory and inference begins immediately.

## Docker Compose (Optional)

For local multi-container development:

```yaml
version: '3.8'

services:
  main:
    image: zepiris-main
    ports:
      - "8000:8000"
    environment:
      - ML_SERVICE_URL=http://ml:8001
    depends_on:
      - ml

  ml:
    image: zepiris-ml-service
    ports:
      - "8001:8001"
    environment:
      - ML_SERVICE_ML_DEVICE=cpu
      - ML_SERVICE_NSFW_MODEL_SOURCE=auto
      - ML_SERVICE_NSFW_LOCAL_MODEL_PATH=/app/models/nsfw_model.pth
      - ML_SERVICE_SPOOF_MODEL_SOURCE=auto
      - ML_SERVICE_SPOOF_LOCAL_MODEL_PATH=/app/models/spoof_model.pth
      - ML_SERVICE_BLUR_MODEL_SOURCE=auto
      - ML_SERVICE_BLUR_LOCAL_MODEL_PATH=/app/models/blur_model.pth
```

## Testing

### End-to-end (HTTP service)

```bash
# Starts the service, sends requests to all endpoints, then shuts down
poetry install --extras ml
poetry run python scripts/test_ml_service.py \
    --test-image path/to/image.jpg \
    --nsfw-model ./models/nsfw_model.pth \
    --spoof-model ./models/spoof_model.pth \
    --blur-model ./models/blur_model.pth

# Against an already-running service
poetry run python scripts/test_ml_service.py \
    --test-image path/to/image.jpg \
    --no-server --port 8001
```

### Direct model testing (in-process)

```bash
poetry run python scripts/test_models.py \
    --test-image path/to/image.jpg \
    --nsfw-model ./models/nsfw_model.pth \
    --spoof-model ./models/spoof_model.pth \
    --blur-model ./models/blur_model.pth
```

See `ML_SERVICE_TESTING.md` for full testing documentation.

## Notes

- The ML service is **CPU-only** by default (configured in Dockerfile). For GPU support, modify the Dockerfile to use a CUDA-enabled base image and set `ML_SERVICE_ML_DEVICE=cuda`.
- **All models are eagerly loaded on startup** (in the lifespan context manager) to ensure requests don't timeout waiting for downloads. This adds 30-60s to startup time but guarantees instant responses after that.
- The **IQA service reuses the same loaded model instances** (NSFW, spoof, blur) rather than creating its own copies, avoiding redundant model loading and memory duplication.
- The service uses a **ThreadPoolExecutor** for the `/v1/iqa/assess` endpoint to run all three quality checks in parallel.
- Images are transmitted as **base64-encoded JSON** for simpler HTTP communication and better compatibility with web clients.
- The Dockerfile installs `g++` temporarily to compile InsightFace's Cython extension, then removes it to reduce image size.
- **Default model paths** are `/app/models/{nsfw,spoof,blur}_model.pth` — models baked into the image at build time. Set `*_MODEL_SOURCE=huggingface` to always pull from Hugging Face Hub, or `local` to require the file on disk.
