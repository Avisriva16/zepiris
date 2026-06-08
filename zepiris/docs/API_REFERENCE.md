# ZepIris — API Reference

Complete API specification for ZepIris face recognition service.

**Base URL**: `http://localhost:8000`

---

## Overview

The ZepIris API provides face embedding, similarity search, and CRUD operations. All responses include:
- `requestId` (UUID) for traceability
- `imageQualityAssessment` (ML_STRUCT) for image quality metrics
- Standardized error messages

**Image pipeline:** Clients send **raw image bytes** as `multipart/form-data`. The API stores the file in MinIO, encodes it as base64 JPEG, and **always** calls the **ML inference service** (requires **`ZEPIRIS_ML_INFERENCE_SERVICE_URL`**) for **`POST /v1/iqa/assess`** (NSFW, spoof, blur) and **`POST /v1/face/embed`** (face embedding), then Milvus. Design-doc **IMAGE_STRUCT** (base64 + validations) is enforced **inside the API** (upload size ≤ 5MB; valid decodable image).

---

## Health & Readiness

### GET /healthz

Liveness probe — check if service is running.

**Response (200 OK)**
```json
{
  "status": "ok"
}
```

---

### GET /readyz

Readiness probe — check if service is ready to handle requests.

**Response (200 OK)**
```json
{
  "status": "ready"
}
```

---

## Face Operations (Doc-Aligned)

### POST /v1/faces/search

1-to-N similarity search — find matching enrolled faces by image.

**Request**
```
Content-Type: multipart/form-data

Query parameter:
- top_k (optional): Number of results (default: 5)

Form fields:
- id (required): Query / trace id (string)
- tenant (required): Tenant identifier (string)
- file (required): Image file (JPEG/PNG, etc.)
```

**Validation:** upload size ≤ 5MB; image must decode with OpenCV. ML checks run on the server via base64 to the inference service.

**Response (200 OK)**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": true, "probability": 1.0 },
    "blur": { "is_sharp": true, "probability": 0.85 }
  },
  "searchResult": {
    "matches": [
      {
        "id": "person-001",
        "score": 0.02
      },
      {
        "id": "person-002",
        "score": 0.35
      }
    ]
  }
}
```

**Response (200 OK - No Matches)**
```json
{
  "requestId": "...",
  "imageQualityAssessment": { ... },
  "searchResult": {
    "matches": []
  }
}
```

**Response (422 Unprocessable - IQA Failed)**
```json
{
  "detail": {
    "message": "image_quality_check_failed",
    "imageQualityAssessment": {
      "passed": false,
      "nsfw": { ... },
      "spoof": { ... },
      "blur": { "is_sharp": false, "probability": 0.15 }
    },
    "object_key": "faces/temp_id"
  }
}
```

---

### POST /v1/faces/insert

Register a new face.

**Request**
```
Content-Type: multipart/form-data

Form fields:
- id (required): Face ID (string)
- tenant (required): Tenant identifier (string)
- file (required): Image file
```

**Validation:** upload size ≤ 5MB; server-side ML assess + embed when `ZEPIRIS_ML_INFERENCE_SERVICE_URL` is set.

**Response (200 OK)**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "imageQualityAssessment": {
    "passed": true,
    "nsfw": { "is_safe": true, "probability": 1.0 },
    "spoof": { "is_live": true, "probability": 1.0 },
    "blur": { "is_sharp": true, "probability": 0.90 }
  },
  "userOperationResult": {
    "operation": "INSERT",
    "status": "success"
  }
}
```

**Response (422 Unprocessable - IQA Failed)**
```json
{
  "detail": {
    "message": "image_quality_check_failed",
    "imageQualityAssessment": {
      "passed": false,
      "nsfw": { ... },
      "spoof": { ... },
      "blur": { "is_sharp": false, "probability": 0.05 }
    },
    "object_key": "faces/temp_id"
  }
}
```

---

### POST /v1/faces/upsert

Insert or update a face (delete if exists, then insert).

**Request**
```
Content-Type: multipart/form-data

Form fields:
- id (required): Face ID
- tenant (required): Tenant identifier
- file (required): Image file
```

**Response (200 OK)**
```json
{
  "requestId": "...",
  "imageQualityAssessment": { ... },
  "userOperationResult": {
    "operation": "UPSERT",
    "status": "success"
  }
}
```

---

### DELETE /v1/faces/delete

Remove a face record by ID.

**Request**
```
Query parameter:
- id (required): Face ID to delete
```

**Response (200 OK)**
```json
{
  "requestId": "...",
  "userOperationResult": {
    "operation": "DELETE",
    "status": "success"
  }
}
```

**Response (200 OK - Not Found)**
```json
{
  "requestId": "...",
  "userOperationResult": {
    "operation": "DELETE",
    "status": "not_found"
  }
}
```

---

### GET /v1/faces/get/{face_id}

Retrieve face metadata by ID.

**Request**
```
Path Parameter:
- face_id (required): Face ID (string)
```

**Response (200 OK)**
```json
{
  "face_id": "person-001",
  "tenant": "default",
  "object_key": "faces/abc123def456xyz..."
}
```

**Response (404 Not Found)**
```json
{
  "detail": "face_id_person-001_not_found"
}
```

---

## Legacy Endpoints (Backward Compatibility)

### POST /v1/faces/query

Legacy 1-to-N search with implicit `tenant="default"`.

**Request**
```
Content-Type: multipart/form-data

Form fields:
- file (required): Image file

Query parameter:
- top_k (optional): Number of results (default: 5)
```

**Response** (same as `/search`)

---

### POST /v1/faces/enroll

Legacy enroll with auto-generated face id and `tenant="default"`.

**Request**
```
Content-Type: multipart/form-data

Form fields:
- file (required): Image file
```

**Response** (same as `/insert`)

---

## Data Models

### IMAGE_STRUCT (design doc)

The public API accepts **raw image uploads** (`multipart/form-data`). Internally the API validates **size ≤ 5MB** and builds a base64 JPEG for the ML microservice, matching the design-doc intent (base64 + validation at the gateway).

---

### ImageQualityAssessmentResult

Image quality assessment (ML_STRUCT from design doc).

```json
{
  "passed": true,
  "nsfw": {
    "is_safe": true,
    "probability": 1.0
  },
  "spoof": {
    "is_live": true,
    "probability": 1.0
  },
  "blur": {
    "is_sharp": true,
    "probability": 0.85
  }
}
```

**Fields**:
- `passed` (bool): True if all quality checks pass
- `nsfw` (NSFWDetectionResult): NSFW check
- `spoof` (SpoofDetectionResult): Liveness/spoof check
- `blur` (BlurDetectionResult): Sharpness/blur check

### NSFWDetectionResult

```json
{
  "is_safe": true,
  "probability": 1.0
}
```

- `is_safe` (bool): True if content is safe (no NSFW detected)
- `probability` (float): Confidence [0.0, 1.0]

### SpoofDetectionResult

```json
{
  "is_live": true,
  "probability": 1.0
}
```

- `is_live` (bool): True if image is genuine/live
- `probability` (float): Liveness confidence [0.0, 1.0]

### BlurDetectionResult

```json
{
  "is_sharp": true,
  "probability": 0.85
}
```

- `is_sharp` (bool): True if image is sharp (not blurry)
- `probability` (float): Sharpness confidence [0.0, 1.0]

### SearchMatch

Single search result.

```json
{
  "id": "person-001",
  "score": 0.02
}
```

- `id` (str): Face ID of matching face
- `score` (float): Cosine distance (0=identical, 1=opposite)

### SearchStruct

Vector search results (SEARCH_STRUCT from design doc).

```json
{
  "matches": [
    { "id": "person-001", "score": 0.02 },
    { "id": "person-002", "score": 0.35 }
  ]
}
```

### CRUDResult

CRUD operation outcome (CRUD_RESULT from design doc).

```json
{
  "operation": "INSERT",
  "status": "success"
}
```

- `operation` (string): One of `INSERT`, `UPSERT`, `DELETE`
- `status` (string): One of `success`, `failed`, `not_found`

---

## Error Handling

### Common Error Codes

| Code | Condition | Example |
|------|-----------|---------|
| 400 | Bad request (empty upload) | `{"detail": "empty_upload"}` |
| 404 | Face not found | `{"detail": "face_id_xyz_not_found"}` |
| 422 | IQA failed | See IQA failed response above |
| 500 | Server error (Milvus, embedding dim mismatch) | `{"detail": "embedding_dim_mismatch_..."}` |

### IQA Failure Detail

When image quality check fails (422):

```json
{
  "detail": {
    "message": "image_quality_check_failed",
    "imageQualityAssessment": {
      "passed": false,
      "nsfw": { ... },
      "spoof": { ... },
      "blur": { "is_sharp": false, "probability": 0.15 }
    },
    "object_key": "faces/temp_id"
  }
}
```

---

## Search Similarity Score (Distance Metric)

**Metric**: COSINE distance

- `0.0` = identical faces (100% match)
- `0.3` = high confidence match
- `0.5` = moderate match
- `1.0` = completely different faces

**Threshold Recommendations**:
- `< 0.3`: Accept (high confidence)
- `0.3 - 0.5`: Review (medium confidence)
- `> 0.5`: Reject (low confidence / different person)

---

## Examples

### cURL - Insert Face

```bash
curl -X POST "http://localhost:8000/v1/faces/insert" \
  -F "id=person-001" \
  -F "tenant=default" \
  -F "file=@person.jpg"
```

### cURL - Search Faces

```bash
curl -X POST "http://localhost:8000/v1/faces/search" \
  -F "id=query-001" \
  -F "tenant=default" \
  -F "file=@unknown.jpg" \
  -F "top_k=5"
```

### cURL - Get Metadata

```bash
curl -X GET "http://localhost:8000/v1/faces/get/person-001"
```

### cURL - Upsert Face

```bash
curl -X POST "http://localhost:8000/v1/faces/upsert" \
  -F "id=person-001" \
  -F "tenant=default" \
  -F "file=@new_photo.jpg"
```

### cURL - Delete Face

```bash
curl -X DELETE "http://localhost:8000/v1/faces/delete?id=person-001"
```

### Python - Insert & Search

```python
import requests

BASE_URL = "http://localhost:8000"

# Insert
with open("person.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/v1/faces/insert",
        files={"file": f},
        data={"id": "person-001", "tenant": "default"}
    )
    print(response.json())

# Search
with open("unknown.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/v1/faces/search",
        files={"file": f},
        data={"id": "query-001", "tenant": "default", "top_k": 5}
    )
    matches = response.json()["searchResult"]["matches"]
    for match in matches:
        print(f"Match: {match['id']} (score: {match['score']})")
```

---

## Rate Limiting

Currently not implemented. For production, planned:
- 100 requests/minute per IP
- 1000 requests/hour per API key

---

## Authentication

Currently no authentication. For production:
- JWT tokens recommended
- API key management
- See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## API Versioning

Current: `v1`

Future versions will use: `/v2/faces/...`

---

## See Also

- [README.md](../README.md) - Project overview
- [SETUP_GUIDE.md](../SETUP_GUIDE.md) - Setup instructions
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [LOCAL_SETUP_AND_TEST.md](../LOCAL_SETUP_AND_TEST.md) - Testing guide
