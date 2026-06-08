# ZepIris — Configuration Guide

Complete configuration reference for ZepIris services.

## Environment Variables

All configuration uses the `ZEPIRIS_` prefix. Values can be set via:
1. Environment variables: `export ZEPIRIS_MILVUS_HOST=localhost`
2. `.env` file: Create `.env` in project root
3. Python code: Edit `zepiris/config.py`

## API Configuration

### ZEPIRIS_API_TITLE
**Type**: `str`
**Default**: `"ZepIris"`
**Description**: API service name (displayed in OpenAPI docs)

```env
ZEPIRIS_API_TITLE=ZepIris Face Auth
```

### ZEPIRIS_API_VERSION
**Type**: `str`
**Default**: `"1.0.0"`
**Description**: API version (follows semantic versioning)

```env
ZEPIRIS_API_VERSION=1.0.0
```

## MinIO / S3 Configuration

### ZEPIRIS_MINIO_ENDPOINT
**Type**: `str`
**Default**: `"localhost:9002"`
**Description**: MinIO/S3 server address with port

**Examples**:
```env
# Local MinIO (Docker Compose)
ZEPIRIS_MINIO_ENDPOINT=localhost:9002

# AWS S3
ZEPIRIS_MINIO_ENDPOINT=s3.amazonaws.com

# Custom S3-compatible (DigitalOcean Spaces, etc.)
ZEPIRIS_MINIO_ENDPOINT=nyc3.digitaloceanspaces.com
```

### ZEPIRIS_MINIO_ACCESS_KEY
**Type**: `str`
**Default**: `"minioadmin"`
**Description**: Access key for S3 authentication

### ZEPIRIS_MINIO_SECRET_KEY
**Type**: `str`
**Default**: `"minioadmin"`
**Description**: Secret key for S3 authentication

⚠️ **Security Warning**: Use strong passwords in production!

```env
ZEPIRIS_MINIO_ACCESS_KEY=your_strong_access_key_here
ZEPIRIS_MINIO_SECRET_KEY=your_strong_secret_key_here
```

### ZEPIRIS_MINIO_BUCKET
**Type**: `str`
**Default**: `"zepiris"`
**Description**: Bucket name for storing images

**Notes**:
- Must follow S3 bucket naming rules (lowercase, no underscores)
- Created automatically if doesn't exist

```env
ZEPIRIS_MINIO_BUCKET=face-auth-prod
```

### ZEPIRIS_MINIO_SECURE
**Type**: `bool`
**Default**: `false`
**Description**: Use HTTPS/TLS for MinIO connection

```env
# Local development (HTTP)
ZEPIRIS_MINIO_SECURE=false

# Production (HTTPS)
ZEPIRIS_MINIO_SECURE=true
```

## Milvus Configuration

### ZEPIRIS_MILVUS_HOST
**Type**: `str`
**Default**: `"localhost"`
**Description**: Milvus server hostname or IP

```env
# Docker Compose
ZEPIRIS_MILVUS_HOST=milvus

# Local
ZEPIRIS_MILVUS_HOST=localhost

# Remote
ZEPIRIS_MILVUS_HOST=milvus.example.com
```

### ZEPIRIS_MILVUS_PORT
**Type**: `int`
**Default**: `19530`
**Description**: Milvus gRPC API port

```env
ZEPIRIS_MILVUS_PORT=19530
```

### ZEPIRIS_MILVUS_COLLECTION
**Type**: `str`
**Default**: `"zepiris_faces"`
**Description**: Milvus collection name for face embeddings

**Notes**:
- Collection is created automatically on startup
- Schema: `face_id (PK), tenant, object_key, embedding (FLOAT_VECTOR)`
- Supports multi-tenancy: each tenant has isolated face namespace

```env
ZEPIRIS_MILVUS_COLLECTION=face_embeddings_prod
```

### ZEPIRIS_MILVUS_EMBEDDING_DIM
**Type**: `int`
**Default**: `512`
**Description**: Dimensionality of face embedding vectors

**Common Values**:
- `128` - Fast but less accurate
- `256` - Good balance
- `512` - Default, good accuracy
- `768` - High accuracy (slower)

⚠️ **Important**: This must match your embedding model output!

```env
ZEPIRIS_MILVUS_EMBEDDING_DIM=512
```

### ZEPIRIS_ML_INFERENCE_SERVICE_URL
**Type**: `str` (**required**)
**Description**: Base URL of the ML inference microservice. The main API **always** calls it for **`POST /v1/iqa/assess`** (NSFW, spoof, blur) and **`POST /v1/face/embed`** (face embedding) after encoding the upload as base64 JPEG.

If `ZEPIRIS_ML_INFERENCE_SERVICE_URL` is empty, **`ML_INFERENCE_SERVICE_URL`** (no `ZEPIRIS_` prefix) is used instead. One of these must be set or **startup fails**.

```env
# Docker Compose (service name)
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference:8001

# API on host, ML published on 8001
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

## Image quality (IQA)

There are **no** `ZEPIRIS_IQA_*` knobs on the main API. Image quality is determined entirely by the **ml-inference** service (`POST /v1/iqa/assess`). Tune blur, NSFW, and spoof behavior with **`ML_SERVICE_*`** environment variables on that container (see [ML Inference Service Configuration](#ml-inference-service-configuration) below).

## ML Inference Service Configuration

The ML inference microservice runs in a **separate container** with its own settings class (`MLServiceSettings` in `zepiris/ml_inference/app.py`). These variables use the **`ML_SERVICE_`** prefix (not `ZEPIRIS_`) and must be set on the ml-inference container.

### Service Binding

| Variable               | Default   | Description                                  |
| ---------------------- | --------- | -------------------------------------------- |
| `ML_SERVICE_HOST`      | `0.0.0.0` | Bind host                                    |
| `ML_SERVICE_PORT`      | `8001`    | Bind port                                    |
| `ML_SERVICE_ML_DEVICE` | `cpu`     | Inference device: `cpu`, `cuda:0`, `mps`     |

### Face Embedding Model

#### ML_SERVICE_FACE_MODEL
**Type**: `str`
**Default**: `"auraface"`
**Accepted values**: `"auraface"`, `"buffalo_l"`
**Description**: Selects which model is used for face detection and embedding generation. Both produce 512-dimensional embeddings.

- `auraface` — uses [`fal/AuraFace-v1`](https://huggingface.co/fal/AuraFace-v1). Weights are downloaded from the HuggingFace Hub into `models/auraface` on first startup.
- `buffalo_l` — uses InsightFace's `buffalo_l`. Downloaded via InsightFace's built-in mechanism on first startup.

⚠️ **Licensing**: InsightFace's `buffalo_l` model weights are licensed for **non-commercial research purposes only**. Commercial use requires a separate license — contact `recognition-oss-pack@insightface.ai`. AuraFace-v1 is the default to avoid this restriction.

```env
# Default
ML_SERVICE_FACE_MODEL=auraface

# InsightFace buffalo_l (non-commercial research only)
ML_SERVICE_FACE_MODEL=buffalo_l
```

> **Note**: An unsupported value raises a `ValueError` at startup. Currently only `cpu` inference is supported for face embedding regardless of `ML_SERVICE_ML_DEVICE`.

| Variable                        | Default | Description                                              |
| ------------------------------- | ------- | -------------------------------------------------------- |
| `ML_SERVICE_FACE_EMBEDDING_DIM` | `512`   | Expected embedding dimension (must match Milvus dim)     |
| `ML_SERVICE_FACE_DETECTION_WIDTH`  | `640`   | Face detection input width                            |
| `ML_SERVICE_FACE_DETECTION_HEIGHT` | `640`   | Face detection input height                           |
| `ML_SERVICE_FACE_AREA_THRESHOLD`   | `0.01`  | Minimum face area as a fraction of total image area   |

### Content Safety Models

Each safety model (NSFW, spoof, blur) follows the same configuration pattern. `*_MODEL_SOURCE` selects where weights load from (`local`, `hf`, or `auto`); `*_THRESHOLD` tunes the classification cutoff (0–1).

| Variable                             | Default                        | Description                                  |
| ------------------------------------ | ------------------------------ | -------------------------------------------- |
| `ML_SERVICE_NSFW_MODEL_SOURCE`       | `auto`                         | Where to load NSFW weights (`local`/`hf`/`auto`) |
| `ML_SERVICE_NSFW_LOCAL_MODEL_PATH`   | `/app/models/nsfw_model.pth`   | NSFW model file (local source)               |
| `ML_SERVICE_NSFW_HF_REPO_ID`         | ``                             | HuggingFace repo for NSFW model (hf source)  |
| `ML_SERVICE_NSFW_HF_MODEL_FILE`      | `nsfw_model.pth`               | NSFW model filename within the HF repo       |
| `ML_SERVICE_NSFW_THRESHOLD`          | `0.5`                          | NSFW classification threshold (0–1)          |
| `ML_SERVICE_SPOOF_MODEL_SOURCE`      | `auto`                         | Where to load spoof weights                  |
| `ML_SERVICE_SPOOF_LOCAL_MODEL_PATH`  | `/app/models/spoof_model.pth`  | Spoof model file (local source)              |
| `ML_SERVICE_SPOOF_HF_REPO_ID`        | ``                             | HuggingFace repo for spoof model (hf source) |
| `ML_SERVICE_SPOOF_HF_MODEL_FILE`     | `spoof_model.pth`              | Spoof model filename within the HF repo      |
| `ML_SERVICE_SPOOF_THRESHOLD`         | `0.5`                          | Spoof classification threshold (0–1)         |
| `ML_SERVICE_BLUR_MODEL_SOURCE`       | `auto`                         | Where to load blur weights                   |
| `ML_SERVICE_BLUR_LOCAL_MODEL_PATH`   | `/app/models/blur_model.pth`   | Blur model file (local source)               |
| `ML_SERVICE_BLUR_HF_REPO_ID`         | ``                             | HuggingFace repo for blur model (hf source)  |
| `ML_SERVICE_BLUR_HF_MODEL_FILE`      | `blur_model.pth`               | Blur model filename within the HF repo       |
| `ML_SERVICE_BLUR_THRESHOLD`          | `0.5`                          | Blur classification threshold (0–1)          |

`POST /v1/iqa/assess` returns `passed: true` when `nsfw.is_safe AND (NOT spoof.is_spoof) AND blur.is_sharp`.

> **Note:** The default `*_THRESHOLD` values (`0.5`) are starting points, not universal settings. You should **tune the IQA models' thresholds against your own dataset and use case** — image characteristics (resolution, lighting, capture device), acceptable false-positive/false-negative trade-offs, and risk tolerance all vary by deployment. Run the models over a representative sample of your images, inspect the score distributions, and adjust each threshold to balance rejecting bad inputs against passing legitimate ones.

## Advanced Configuration

### Custom Milvus Settings

Edit `/milvus/configs/milvus.yaml` (in Docker) or environment:

```env
# Search parameters
MILVUS_SEARCH_NLIST=128
MILVUS_SEARCH_NPROBE=8

# Memory settings
MILVUS_MEMORY_HIGH_WATERMARK=0.95
MILVUS_MEMORY_LOW_WATERMARK=0.85
```

### Custom MinIO Settings

For production S3/MinIO deployments:

```env
# Performance
MINIO_ACCESS_CONCURRENCY=20
MINIO_UPLOAD_TIMEOUT=24h

# Security
MINIO_ENABLE_HTTPS=true
MINIO_CERT_FILE=/path/to/cert.pem
MINIO_KEY_FILE=/path/to/key.pem
```

## Configuration Presets

### Development Preset
```env
# .env for local development
ZEPIRIS_MINIO_ENDPOINT=localhost:9002
ZEPIRIS_MINIO_SECURE=false
ZEPIRIS_MILVUS_HOST=localhost
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
```

### Production Preset (AWS)
```env
# .env for production with AWS
ZEPIRIS_MINIO_ENDPOINT=s3.amazonaws.com
ZEPIRIS_MINIO_BUCKET=your-prod-bucket
ZEPIRIS_MINIO_SECURE=true
ZEPIRIS_MILVUS_HOST=milvus-prod.internal
ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference.internal:8001
ZEPIRIS_MILVUS_EMBEDDING_DIM=768
```

### Kubernetes Preset
```env
# .env for Kubernetes deployment
ZEPIRIS_MINIO_ENDPOINT=minio.default.svc.cluster.local:9000
ZEPIRIS_MILVUS_HOST=milvus.default.svc.cluster.local
ZEPIRIS_MINIO_BUCKET=zepiris-prod
ZEPIRIS_MILVUS_COLLECTION=zepiris_faces_prod
```

## Configuration Validation

On startup, ZepIris validates:
- ✓ `ZEPIRIS_ML_INFERENCE_SERVICE_URL` (or `ML_INFERENCE_SERVICE_URL`) is set
- ✓ Milvus connectivity
- ✓ MinIO bucket accessibility
- ✓ Embedding dimension is positive

**Validation Errors** appear in logs:
```
[ERROR] Could not connect to Milvus at localhost:19530
[ERROR] MinIO bucket 'zepiris' does not exist and could not be created
```

## Performance Tuning

### For High Throughput
```env
ZEPIRIS_MILVUS_EMBEDDING_DIM=256  # Smaller vectors = faster search
# Stricter or looser IQA: adjust ML_SERVICE_* on the ml-inference container
```

### For High Accuracy
```env
ZEPIRIS_MILVUS_EMBEDDING_DIM=768  # Larger vectors = more accurate
# Stricter blur/NSFW/spoof: adjust ML_SERVICE_* on the ml-inference container
```

### For Large Datasets
```env
ZEPIRIS_MILVUS_COLLECTION=zepiris_faces_large
# Use distributed Milvus setup (see Kubernetes guide)
```

## Troubleshooting Configuration

### Issue: "Connection refused" for Milvus
```bash
# Check Milvus is running
docker-compose ps milvus

# Check host/port
echo $ZEPIRIS_MILVUS_HOST
echo $ZEPIRIS_MILVUS_PORT
```

### Issue: "MinIO bucket not found"
```bash
# Create bucket
docker exec zepiris-minio mc mb minio/zepiris
```

### Issue: Images failing IQA
IQA is enforced by **ml-inference** (`/v1/iqa/assess`). Loosen thresholds there, e.g.:

```env
ML_SERVICE_BLUR_THRESHOLD=0.85
```
(Set on the `ml-inference` service / its `.env`, not on the main API.)

## Environment Variable Priority

Settings are loaded in order (highest priority first):
1. Environment variables (`export ZEPIRIS_...`)
2. `.env` file
3. Docker secrets (Kubernetes)
4. Hardcoded defaults in `config.py`

Example:
```bash
# Override with environment variable
export ZEPIRIS_MILVUS_HOST=remote-milvus.com
docker-compose up
```

## Configuration File Format

### .env File Format
```env
# Comments start with #
ZEPIRIS_API_TITLE=ZepIris

# Multiline values (not typical for config)
# Quote if value contains spaces
ZEPIRIS_API_TITLE="My ZepIris Service"

# Boolean values
ZEPIRIS_MINIO_SECURE=true  # or false

# Numbers
ZEPIRIS_MILVUS_PORT=19530
ZEPIRIS_IQA_MIN_LAPLACIAN_VARIANCE=50.0
```

## Loading Custom Configuration

### Programmatically
```python
from zepiris.config import Settings

settings = Settings(
    minio_endpoint="custom:9000",
    milvus_host="custom-milvus"
)
```

### From File
```bash
# Create custom.env
source custom.env
poetry run uvicorn zepiris.main:app
```

## Security Best Practices

1. **Never commit `.env` with real credentials**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Use strong passwords**
   ```bash
   # Generate secure key
   openssl rand -base64 32
   ```

3. **Rotate credentials regularly**
   - Update MinIO keys monthly
   - Audit Milvus access logs

4. **Use TLS/HTTPS in production**
   ```env
   ZEPIRIS_MINIO_SECURE=true
   ```

5. **Restrict network access**
   - Milvus: only from API servers
   - MinIO: only from API servers

## See Also

- [README.md](../README.md) - Project overview
- [SETUP_GUIDE.md](../SETUP_GUIDE.md) - Setup instructions
- [API_REFERENCE.md](API_REFERENCE.md) - API endpoints
