# ML Inference Module — Final Structure

## Overview

Production-ready ML inference service with FastAPI microservice, Hugging Face model support, and eager model loading on startup.

## Module Architecture

```
zepiris/ml_inference/
├── __init__.py                    # Package exports
├── base.py                        # ModelService + ModelServiceConfig (base classes)
├── nsfw_detection.py              # NSFWDetectionService
├── spoof_detection.py             # SpoofDetectionService
├── blur_detection.py              # BlurDetectionService
├── face_embedding.py              # FaceEmbeddingService (standalone, no inheritance)
├── image_quality_assessment.py    # ImageQualityAssessmentService (combines above 3)
├── app.py                         # FastAPI app, MLServiceSettings, lifespan (model loading)
├── routes.py                      # All 6 API endpoints
└── deps.py                        # FastAPI dependency injection

zepiris/services/
└── ml_client.py                   # MLInferenceClient (HTTP client for main app)

zepiris/schemas/
└── ml_inference.py                # Result schemas (Pydantic)

scripts/
├── test_models.py                 # Direct in-process model testing
└── test_ml_service.py             # End-to-end HTTP service testing

ml_inference.Dockerfile            # Container image (includes models/)
```

## Core Classes

### Base: `ModelService` + `ModelServiceConfig`

**`ModelServiceConfig`** — Pydantic configuration:
```python
class ModelServiceConfig(BaseModel):
    model_name: str                     # e.g., "nsfw_detection"
    huggingface_repo_id: str           # HF repo ID (blank = skip HF)
    huggingface_model_file: str        # Model filename in repo
    local_model_path: str | None       # Local filesystem path (takes precedence)
    device: str = "cpu"                # "cpu" or "cuda"
```

**`ModelService`** — Base class for PyTorch models:
```python
class ModelService:
    def __init__(self, config: ModelServiceConfig) -> None: ...
    def _download_model() -> bytes: ...        # From local path or HF Hub
    def load_model() -> torch.nn.Module: ...   # Downloads + caches + loads
    def preprocess(image_rgb: np.ndarray) -> dict: ...
    def predict(preprocessed_data: dict) -> np.ndarray: ...
    def postprocess(output: np.ndarray) -> BaseModel: ...
```

### Detection Services (inherit from `ModelService`)

- **`NSFWDetectionService`** → `NSFWDetectionResult`
- **`SpoofDetectionService`** → `SpoofDetectionResult`
- **`BlurDetectionService`** → `BlurDetectionResult`

Each accepts:
- `huggingface_repo_id`, `huggingface_model_file`, `local_model_path` (model loading)
- `nsfw_threshold` / `*_threshold` (detection threshold)
- `device` ("cpu" or "cuda")

### Special Services

**`FaceEmbeddingService`** — Standalone (no inheritance from `ModelService`):
- Uses AuraFace OR InsightFace's Buffalo_l built-in model downloading based on the flag ML_SERVICE_FACE_MODEL set in .env
- No Hugging Face configuration needed
- Takes: `embedding_dim`, `detection_size`, `facial_area_threshold`, `device`

**`ImageQualityAssessmentService`** — Combines three detections:
- **Constructor change:** Now accepts pre-loaded service instances instead of creating its own
- Reuses NSFW, spoof, blur services (no model duplication)
- Runs all three in parallel via `ThreadPoolExecutor`

## Result Schemas (Pydantic)

All in `zepiris/schemas/ml_inference.py`:

```python
class NSFWDetectionResult(BaseModel):
    is_safe: bool
    probability: float  # [0, 1] validated

class SpoofDetectionResult(BaseModel):
    is_live: bool
    probability: float  # [0, 1] validated

class BlurDetectionResult(BaseModel):
    is_sharp: bool
    probability: float  # [0, 1] validated

class FaceEmbeddingResult(BaseModel):
    embedding: list[float]
    embedding_dim: int

class ImageQualityAssessmentResult(BaseModel):
    passed: bool
    nsfw: NSFWDetectionResult
    spoof: SpoofDetectionResult
    blur: BlurDetectionResult
```

## FastAPI Microservice

**`app.py`** — The service itself:

```python
class MLServiceSettings(BaseSettings):
    # Service config
    host: str = "0.0.0.0"
    port: int = 8001
    ml_device: str = "cpu"

    # Model paths + thresholds for NSFW, spoof, blur
    nsfw_hf_repo_id: str = ""
    nsfw_local_model_path: str = "/app/models/nsfw_model.pth"
    nsfw_threshold: float = 0.5

    # ... similar for spoof, blur ...

    # Face embedding params
    face_embedding_dim: int = 512
    face_detection_width: int = 640
    face_detection_height: int = 640
    face_area_threshold: float = 0.01

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager model loading on startup
    app.state.nsfw_service = NSFWDetectionService(...)
    app.state.nsfw_service.load_model()

    app.state.spoof_service = SpoofDetectionService(...)
    app.state.spoof_service.load_model()

    app.state.blur_service = BlurDetectionService(...)
    app.state.blur_service.load_model()

    app.state.face_embedding_service = FaceEmbeddingService(...)
    app.state.face_embedding_service.load_model()

    # IQA reuses existing services (no duplication)
    app.state.iqa_service = ImageQualityAssessmentService(
        nsfw_service=app.state.nsfw_service,
        spoof_service=app.state.spoof_service,
        blur_service=app.state.blur_service,
    )
    yield
```

**`routes.py`** — API endpoints:
- `GET /healthz` → health check
- `POST /v1/iqa/nsfw_check` → NSFW detection
- `POST /v1/iqa/spoof_check` → spoof detection
- `POST /v1/iqa/blur_check` → blur detection
- `POST /v1/face/embed` → face embedding
- `POST /v1/iqa/assess` → combined quality assessment

All endpoints:
- Accept JSON: `{"image_b64": "<base64-encoded image>"}`
- Decode base64 → parse image (JPEG/PNG) → convert to RGB
- Return Pydantic result schema as JSON

**`deps.py`** — FastAPI dependency injection:
```python
def _nsfw(request: Request) -> NSFWDetectionService:
    return request.app.state.nsfw_service

NSFWDep = Annotated[NSFWDetectionService, Depends(_nsfw)]
# ... similar for other services ...
```

Routes use: `async def detect_nsfw(service: NSFWDep, payload: ImagePayload) -> ...`

## HTTP Client

**`zepiris/services/ml_client.py`** — For main app to call ML service:

```python
class MLInferenceClient:
    def __init__(self, base_url: str) -> None: ...
    def detect_nsfw(self, image_b64: str) -> NSFWDetectionResult: ...
    def detect_spoof(self, image_b64: str) -> SpoofDetectionResult: ...
    def detect_blur(self, image_b64: str) -> BlurDetectionResult: ...
    def embed_face(self, image_b64: str) -> FaceEmbeddingResult: ...
    def assess_image_quality(self, image_b64: str) -> ImageQualityAssessmentResult: ...

    def close(self) -> None: ...
```

## Model Loading Strategy

**Priority (checked in order):**
1. **Local filesystem** — if `local_model_path` is set, load from there
2. **Hugging Face Hub** — if `huggingface_repo_id` is set, download from HF
3. **Docker image** — default paths point to `/app/models/` (baked in during build)

**Timing:**
- **Eager loading** — all models loaded on service startup (lifespan hook)
- **No lazy loading** — first request doesn't wait for downloads
- **Shared instances** — IQA service reuses NSFW/spoof/blur services (no duplication)

## Docker Containerization

**`ml_inference.Dockerfile`**:
- Base: `python:3.13-slim`
- System deps: `libgl1 libglib2.0-0 g++` (for OpenCV + InsightFace)
- Installs: `torch`, `torchvision`, `insightface`, `onnxruntime`
- Copies: `models/` → `/app/models/` (model files baked in)
- Removes: `g++` after build (to reduce image size)
- Exposes: port 8001
- Runs: `uvicorn zepiris.ml_inference.app:app`

## Usage Examples

### Direct Model Usage (in-process)

```python
from zepiris.ml_inference.nsfw_detection import NSFWDetectionService
import numpy as np

service = NSFWDetectionService(
    huggingface_repo_id="",
    huggingface_model_file="nsfw_model.pth",
    local_model_path="/path/to/nsfw_model.pth",
    nsfw_threshold=0.5,
    device="cpu",
)

image_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
result = service.forward(image_rgb)
print(result.is_safe, result.probability)
```

### HTTP Client Usage (to ML microservice)

```python
import base64
import cv2
from zepiris.services.ml_client import MLInferenceClient

client = MLInferenceClient("http://localhost:8001")

# Encode image to base64
image_bgr = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
_, buf = cv2.imencode(".jpg", image_rgb)
image_b64 = base64.b64encode(buf).decode("utf-8")

# Call service
result = client.detect_nsfw(image_b64)
print(result.is_safe, result.probability)

client.close()
```

### Docker Usage

```bash
# Place models in ./models/
mkdir models
cp nsfw_model.pth spoof_model.pth blur_model.pth models/

# Build
docker build -f ml_inference.Dockerfile -t zepiris-ml-service .

# Run
docker run -p 8001:8001 zepiris-ml-service
```

## Key Design Decisions
✓ **Eager model loading** — all models loaded at startup to avoid first-request timeouts
✓ **Shared service instances** — IQA reuses NSFW/spoof/blur to avoid loading models 3x
✓ **HF or local models** — flexibility between cloud-hosted and local model files
✓ **Base64 JSON transport** — simpler HTTP than multipart form data
✓ **FastAPI + Pydantic** — type-safe endpoints with automatic validation + OpenAPI docs
✓ **Microservice architecture** — ML inference isolated in separate container + HTTP boundary
✓ **CPU-only by default** — easy GPU support via `ML_SERVICE_ML_DEVICE=cuda` + Dockerfile base image change
