# Local Setup & Testing Guide

Complete step-by-step guide to test ZepIris locally with all components running in Docker.

**Estimated time: 20-30 minutes**

---

## Prerequisites

- **Docker**: v20.10+ ([Install Docker](https://docs.docker.com/install/))
- **Docker Compose**: v1.29+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **4GB+ RAM** for Milvus
- **10GB+ free disk space** for volumes
- **curl** or **Postman** for API testing (optional but recommended)
- **ML weights** in `models/` at the repo root: `nsfw_model.pth`, `spoof_model.pth`, `blur_model.pth` (used by the `ml-inference` service; Compose mounts `./models` → `/app/models`)

Verify installation:
```bash
docker --version
docker-compose --version
```

---

## Step 1: Clone & Navigate to Project

```bash
cd zepiris
```

---

## Step 2: Start All Services

```bash
docker-compose up -d
```

Builds include copying `models/` into the ML image; Compose also **bind-mounts** `./models` to `/app/models` so local weights are always used.

**Expected output:**
```
Creating network "zepiris-network" with default driver
Creating zepiris-minio ... done
Creating zepiris-etcd ... done
Creating zepiris-milvus ... done
Creating zepiris-ml-inference ... done
Creating zepiris-api ... done
```

**Wait for all services to be healthy** (takes ~30-60 seconds):

```bash
docker-compose ps
```

**Expected status:**
```
NAME              STATUS              PORTS
zepiris-minio     Up (healthy)        0.0.0.0:9002->9000/tcp, 0.0.0.0:9001->9001/tcp
zepiris-etcd      Up (healthy)        0.0.0.0:2379->2379/tcp
zepiris-milvus    Up (healthy)        0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
zepiris-api       Up (healthy)        0.0.0.0:8000->8000/tcp
```

If any service shows `Unhealthy` or `Not running`:
```bash
# Check logs
docker-compose logs <service-name>

# Restart the service
docker-compose restart <service-name>
```

---

## Step 3: Verify Services Are Healthy

### Health Endpoint
```bash
curl http://localhost:8000/healthz
```

**Expected response:**
```json
{"status":"ok"}
```

### Readiness Endpoint
```bash
curl http://localhost:8000/readyz
```

**Expected response:**
```json
{"status":"ready"}
```

### MinIO Console
Open browser: http://localhost:9001
- **Login**: `minioadmin` / `minioadmin`
- You should see an empty bucket named `zepiris`
- **Note**: MinIO API is on port 9002 (external), 9000 (internal container)

### API Documentation
Open browser: http://localhost:8000/docs
- Swagger UI with all endpoints visible
- You can test endpoints directly from here

---

## Step 4: Create test images

The API accepts **raw image files** (`multipart/form-data`). The server stores the bytes in MinIO, encodes JPEG to **base64**, and calls the ML inference service (`/v1/iqa/assess`, `/v1/face/embed`).

Generate sample JPEGs:

```bash
python3 << 'EOF'
import numpy as np
from PIL import Image
import os

os.makedirs('/tmp/test_images', exist_ok=True)

for i in range(1, 4):
    img_array = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'/tmp/test_images/test_image_{i}.jpg')
    print(f'Created: /tmp/test_images/test_image_{i}.jpg')
EOF
```

You can also use any JPEG/PNG on disk (e.g. `Test_Images/Archive/...jpeg`).

---

## Step 5: Test API Endpoints

### A. INSERT a face

```bash
curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=person-001" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/test_image_1.jpg"
```

**Expected response (success):**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": true, "probability": 1.0 },
    "blur": { "is_sharp": true, "probability": 0.92 }
  },
  "userOperationResult": {
    "operation": "INSERT",
    "status": "success"
  }
}
```

**Checks:** upload size ≤ 5MB; decodable image. IQA is **always** ML `POST /v1/iqa/assess`, then `POST /v1/face/embed` (requires `ZEPIRIS_ML_INFERENCE_SERVICE_URL`, e.g. `http://ml-inference:8001` in Docker Compose).

### B. SEARCH (1-to-N)

```bash
curl -X POST "http://localhost:8000/v1/faces/search?top_k=5" \
  -F "id=query-001" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/test_image_1.jpg"
```

### C. GET face metadata

```bash
curl -X GET "http://localhost:8000/v1/faces/get/person-001"
```

### D. UPSERT

```bash
curl -X POST "http://localhost:8000/v1/faces/upsert" \
  -F "id=person-001" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/test_image_2.jpg"
```

### E. DELETE

```bash
curl -X DELETE "http://localhost:8000/v1/faces/delete?id=person-001"
```

### Legacy: `/enroll` and `/query`

```bash
curl -X POST "http://localhost:8000/v1/faces/enroll" \
  -F "file=@/tmp/test_images/test_image_1.jpg"

curl -X POST "http://localhost:8000/v1/faces/query?top_k=5" \
  -F "file=@/tmp/test_images/test_image_1.jpg"
```

---

## Step 6: Test Error Cases

### Missing file field
```bash
curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=test" \
  -F "tenant=default"
```

**Expected**: `422 Unprocessable Entity` (validation: missing `file`)

### Empty upload
```bash
touch /tmp/empty.jpg
curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=test" \
  -F "tenant=default" \
  -F "file=@/tmp/empty.jpg"
```

**Expected**: `400 Bad Request` — detail `empty_upload`

### Not a valid image (bytes do not decode)
```bash
echo 'not-an-image' > /tmp/bad.bin
curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=test" \
  -F "tenant=default" \
  -F "file=@/tmp/bad.bin"
```

**Expected**: `422 Unprocessable Entity` — `image_quality_check_failed` (body includes `imageQualityAssessment` with `passed: false`)

### Image too large (> 5MB)
```bash
python3 << 'EOF'
import numpy as np
from PIL import Image
import os

os.makedirs('/tmp/test_images', exist_ok=True)
img_array = np.random.randint(0, 256, (10000, 10000, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
path = '/tmp/test_images/huge.jpg'
img.save(path, format='JPEG', quality=95)
print(path, os.path.getsize(path) // (1024*1024), 'MB approx')
EOF

curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=test" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/huge.jpg"
```

**Expected**: `422 Unprocessable Entity` — detail like `image_too_large_…_max_5MB`

### Non-existent face
```bash
curl -X GET "http://localhost:8000/v1/faces/get/non-existent-id"
```

**Expected**: `404 Not Found` — `face_id_non-existent-id_not_found`

### Low resolution / poor quality (ML IQA)
Create a 32×32 JPEG and insert (ML `/v1/iqa/assess` may reject):
```bash
python3 << 'EOF'
from PIL import Image
import numpy as np
import os
os.makedirs('/tmp/test_images', exist_ok=True)
img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save('/tmp/test_images/tiny.jpg')
EOF

curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=small" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/tiny.jpg"
```

**Expected**: `422 Unprocessable Entity` — `image_quality_check_failed`

---

## Step 7: Full Integration Test

Run a complete workflow with multipart uploads:

```bash
#!/bin/bash
set -e

echo "1. Inserting 3 faces..."
for i in 1 2 3; do
  curl -s -X POST "http://localhost:8000/v1/faces/insert" \
    -F "id=person-$i" \
    -F "tenant=default" \
    -F "file=@/tmp/test_images/test_image_${i}.jpg" | jq '.userOperationResult'
done

echo -e "\n2. Searching for person-1..."
curl -s -X POST "http://localhost:8000/v1/faces/search?top_k=3" \
  -F "id=query-1" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/test_image_1.jpg" | jq '.searchResult.matches'

echo -e "\n3. Getting metadata for person-1..."
curl -s -X GET "http://localhost:8000/v1/faces/get/person-1" | jq '.'

echo -e "\n4. Updating person-1..."
curl -s -X POST "http://localhost:8000/v1/faces/upsert" \
  -F "id=person-1" \
  -F "tenant=default" \
  -F "file=@/tmp/test_images/test_image_3.jpg" | jq '.userOperationResult'

echo -e "\n5. Deleting person-2..."
curl -s -X DELETE "http://localhost:8000/v1/faces/delete?id=person-2" | jq '.userOperationResult'

echo -e "\nDone!"
```

---

## Step 8: Monitor Logs

### View all logs
```bash
docker-compose logs -f
```

### View specific service logs
```bash
docker-compose logs -f api
docker-compose logs -f milvus
docker-compose logs -f minio
```

### Stop log streaming
Press `Ctrl+C`

---

## Step 9: Cleanup

### Stop services (keep data)
```bash
docker-compose stop
```

### Stop and remove containers
```bash
docker-compose down
```

### Remove volumes (DELETE ALL DATA!)
```bash
docker-compose down -v
```

### Start again
```bash
docker-compose up -d
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find what's using port 8000
lsof -i :8000

# Free up port manually or change port in docker-compose.yml
```

### Milvus Connection Refused
```bash
# Milvus might still be initializing, wait 30 seconds
docker-compose logs milvus

# Restart if needed
docker-compose restart milvus
```

### MinIO Bucket Not Found
```bash
# Create bucket manually
docker exec zepiris-minio mc mb minio/zepiris
```

### No Disk Space
```bash
# Check usage
docker system df

# Clean up
docker system prune -a --volumes
```

---

## Performance Expectations

| Operation | Latency | Notes |
|-----------|---------|-------|
| Enroll (INSERT) | 100-200ms | Image upload + embed + insert |
| Search (1-to-N, top_k=5) | 50-100ms | Embedding + Milvus search |
| Get metadata | 10-20ms | Direct database lookup |
| Update (UPSERT) | 100-200ms | Delete + insert |
| Delete | 30-50ms | Index update |

---

## Docker Compose Breakdown

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **minio** | minio/minio:latest | 9002 (ext), 9001 | S3-compatible storage |
| **etcd** | quay.io/coreos/etcd:v3.5.5 | 2379 | Metadata for Milvus |
| **milvus** | milvusdb/milvus:v2.6.12 | 19530 | Vector database |
| **api** | (local build) | 8000 | FastAPI application |
| **ml-inference** | (local build) | 8001 | Assess + embed (HTTP JSON base64) |

---

## Database Schema

```
Collection: zepiris_faces

Fields:
  - face_id     (VARCHAR, 128 chars, PRIMARY KEY)
  - tenant      (VARCHAR, 256 chars)
  - object_key  (VARCHAR, 512 chars)  [path in MinIO]
  - embedding   (FLOAT_VECTOR, 512 dims)

Index: FLAT on embedding with COSINE metric
```

---

## Endpoints Summary

### New Doc-Aligned API
- `POST /v1/faces/search` - Search similar faces (1-to-N)
- `POST /v1/faces/insert` - Register a face
- `POST /v1/faces/upsert` - Register or update a face
- `DELETE /v1/faces/delete` - Remove a face

### Legacy (Backward Compatible)
- `POST /v1/faces/query` - Same as /search
- `POST /v1/faces/enroll` - Same as /insert with auto-ID
- `GET /v1/faces/get/{face_id}` - Get metadata

### Health
- `GET /healthz` - Liveness probe
- `GET /readyz` - Readiness probe

---

## Configuration

All settings use `ZEPIRIS_` prefix. Configured in `docker-compose.yml`:

- `ZEPIRIS_MINIO_ENDPOINT=minio:9000`
- `ZEPIRIS_MILVUS_HOST=milvus`
- `ZEPIRIS_MILVUS_PORT=19530`
- `ZEPIRIS_MILVUS_EMBEDDING_DIM=512`
- `ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://ml-inference:8001` (**required**; API calls `/v1/iqa/assess` and `/v1/face/embed`; legacy `ML_INFERENCE_SERVICE_URL` is accepted if unset)

Edit these in `.env` file or directly in `docker-compose.yml`.

---

## Next Steps

1. ✅ All services running locally
2. ✅ API endpoints tested
3. **Push to GitHub** - Create a feature branch and PR
4. **Add CI/CD** - Setup automated testing
5. **Production deployment** - Kubernetes or cloud hosting
6. **Tune ML models** — adjust `ML_SERVICE_*` on the `ml-inference` container as needed

---

## FAQ

**Q: Can I test without Docker?**
A: Yes, run services separately:
```bash
# Terminal 1: Minio (if port 9002 is also in use, change -p 9002:9000 to another port)
docker run -p 9002:9000 minio/minio server /data

# Terminal 2: Etcd
docker run -p 2379:2379 quay.io/coreos/etcd:v3.5.5

# Terminal 3: Milvus
docker run -p 19530:19530 milvusdb/milvus:v2.6.12 run standalone

# Terminal 4: ML inference on port 8001 (easiest: from repo root: docker compose up -d ml-inference)

# Terminal 5: API (requires ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001)
poetry install
export ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://localhost:8001
poetry run zepiris-api
```

**Q: How do I reset the database?**
A: Stop, remove volumes, and restart:
```bash
docker-compose down -v && docker-compose up -d
```

**Q: How do I see MinIO files?**
A: Access MinIO console at http://localhost:9001 or use MC CLI:
```bash
docker exec zepiris-minio mc ls minio/zepiris
```

**Q: How do I change image quality thresholds?**
A: IQA is enforced by **ml-inference** (`/v1/iqa/assess`). Set **`ML_SERVICE_*`** on that service (e.g. `ML_SERVICE_BLUR_THRESHOLD`); see `zepiris/ml_inference/app.py` and `.env.example`.

---

## Success Criteria

- [ ] All containers running and healthy (minio, etcd, milvus, api, ml-inference)
- [ ] Health endpoint returns `{"status":"ok"}`
- [ ] Multipart upload: missing/empty file rejected appropriately
- [ ] Upload size validation works (>5MB rejected with 422)
- [ ] Can insert a face via `/v1/faces/insert` with `-F file=@...`
- [ ] Can search via `/v1/faces/search` with multipart file
- [ ] Can retrieve face metadata via `/v1/faces/get/{id}`
- [ ] Can upsert via `/v1/faces/upsert` with multipart file
- [ ] Can delete via `DELETE /v1/faces/delete?id=...`
- [ ] API documentation available at `/docs` (multipart `file`, `id`, `tenant`)
- [ ] MinIO console accessible at `localhost:9001`

**When all criteria pass, you're ready for production!**

---

## Support

- 📖 See `README.md` for project overview
- 📖 See `SETUP_GUIDE.md` for detailed setup
- 📖 See `docs/API_REFERENCE.md` for API details
- 📖 See `docs/CONFIGURATION.md` for configuration options
