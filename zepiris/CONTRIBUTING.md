# Contributing to ZepIris

Thank you for your interest in contributing! We welcome all contributions, from bug reports to feature implementations.

## Code of Conduct

We are committed to providing a welcoming and inspiring community. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/zepto-labs/zepiris.git
cd zepiris
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Install Development Dependencies
```bash
poetry install --with dev
poetry run pre-commit install
```

**`poetry install`** reads `pyproject.toml` / `poetry.lock`, creates **`./.venv`** (see `poetry.toml`, `in-project = true`), and installs **main runtime dependencies** plus the `zepiris` package in editable mode. With **`--with dev`**, dev tools (Ruff, pre-commit, mypy) are installed into the same environment.

Poetry does **not** auto-activate your shell. Use **`poetry shell`** to start a subshell with the virtualenv activated, **`poetry run <command>`** for a single command, or activate the venv manually (e.g. **`source .venv/bin/activate`** on Linux/macOS).

The `pre-commit` hook runs Ruff (lint + format) and basic file checks on every commit.

### 4. Make Your Changes
Follow the style guide and write tests for your changes.

## Development Workflow

### Code Style

We use:
- **Ruff** for linting and formatting (line length: 100; configured in `pyproject.toml`)
- **MyPy** for type checking (run manually; not part of the default pre-commit hook)

Format and lint (or rely on `poetry run pre-commit run --all-files`):
```bash
poetry run ruff check --fix zepiris/ scripts/ tests/
poetry run ruff format zepiris/ scripts/ tests/
poetry run mypy zepiris/
```

### Testing

Write tests for all new features:
```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run pytest tests/test_milvus_store.py

# With coverage
poetry run pytest --cov=src tests/
```

### Type Hints

All functions should have type hints:

```python
from typing import Optional
from numpy import ndarray
from zepiris.schemas.ml_inference import FaceEmbeddingResult

def embed(self, image_bgr: ndarray) -> FaceEmbeddingResult:
    """Generate face embedding from image (with face detection status)."""
    ...

def get_by_id(self, face_id: str) -> Optional[dict]:
    """Retrieve face record or None if not found."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def search(self, embedding: list[float], top_k: int) -> list[VectorMatch]:
    """Search for similar face embeddings.

    Args:
        embedding: Face embedding vector (512-dim).
        top_k: Number of results to return (default: 5).

    Returns:
        List of VectorMatch objects with face_id, object_key, and distance.

    Raises:
        ValueError: If embedding dimension doesn't match collection schema.
    """
```

## Commit Guidelines

- Use clear, descriptive commit messages
- Reference issues in commits: `Fixes #123`
- One feature per commit
- Keep commits atomic and logical

Good commit message:
```
feat: Add face embedding delete operation

- Implement MilvusFaceStore.delete(face_id) method
- Add DELETE /v1/faces/delete/{face_id} endpoint
- Update Milvus schema documentation

Fixes #45
```

## Pull Request Process

1. **Rebase on main**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open PR on GitHub**
   - Clear title and description
   - Link related issues
   - Describe changes and testing

4. **Address Review Comments**
   - Request changes
   - Don't commit new changes, squash into existing commits

5. **Wait for Approval**
   - At least 2 approvals required
   - All CI checks must pass

## Areas for Contribution

### High Priority
- [ ] Real ML models (ArcFace, AdaFace for embeddings)
- [ ] Kubernetes deployment (Helm charts)
- [ ] Authentication & authorization
- [ ] Comprehensive logging
- [ ] API rate limiting

### Medium Priority
- [ ] Test coverage improvements
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Error handling enhancements
- [ ] Monitoring & metrics

### Low Priority
- [ ] Code refactoring
- [ ] Dependency updates
- [ ] Type hint improvements
- [ ] Minor bug fixes

### Submitting Issues

Use GitHub Issues with clear titles and descriptions:

```markdown
## Title
Brief description

## Expected Behavior
What should happen

## Actual Behavior
What currently happens

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Environment
- OS: macOS 14.0
- Python: 3.10–3.14 (see `requires-python` in `pyproject.toml`)
- Docker: 24.0.0
```

## Testing ML Models

When integrating real ML models:

1. **Add Model Card**: Document performance metrics
2. **Benchmark**: Compare against stub
3. **Test Edge Cases**: Small faces, low quality images
4. **Validate Accuracy**: ROC curves, confusion matrices

Example:
```python
def test_face_embedding_dimension():
    """Verify embedding dimension matches configuration."""
    service = RealFaceEmbeddingService(model_path)

    image = load_test_image()
    embedding = service.embed(image)

    assert len(embedding) == 512
    assert np.linalg.norm(embedding) <= 1.01  # L2-normalized

def test_face_embedding_consistency():
    """Verify same image produces same embedding."""
    service = RealFaceEmbeddingService(model_path)
    image = load_test_image()

    emb1 = service.embed(image)
    emb2 = service.embed(image)

    distance = spatial.distance.cosine(emb1, emb2)
    assert distance < 1e-6  # Nearly identical
```

## Documentation

Update docs when making changes:

- **API Changes**: Update `/docs` endpoints in code
- **Configuration**: Update `README.md` & `SETUP_GUIDE.md`
- **Architecture**: Add ADRs in `docs/adr/`

## Deployment Testing

Before merging, test on multiple platforms:

```bash
# Docker Compose (API image installs from poetry.lock; use python -m pytest inside the container)
docker-compose up -d
docker-compose exec api python -m pytest tests/

# Local with Poetry
poetry run pytest tests/
```

## Performance Considerations

When optimizing code:

1. **Measure First**: Use `pytest-benchmark`
2. **Document Impact**: Include benchmarks in PR
3. **Consider Trade-offs**: Speed vs. memory vs. complexity

Example benchmark PR:
```
## Performance Impact
- Query latency: 45ms → 32ms (28% improvement)
- Memory usage: Unchanged
- Trade-off: More CPU cores needed for indexing
```

## Release Process

Maintainers use semantic versioning:
- **MAJOR**: Breaking changes (0→1, 1→2)
- **MINOR**: New features (1.0→1.1)
- **PATCH**: Bug fixes (1.0.0→1.0.1)

## Questions?

- 📧 Email: opensource@zepto.com
- 💬 GitHub Discussions: [Join us!](https://github.com/zepto-labs/zepiris/discussions)
- 📖 Read [README.md](README.md) & [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

**Thank you for contributing to ZepIris!**
