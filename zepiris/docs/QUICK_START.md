# ZepIris — Quick Start

**Pick the right guide:**

## 🚀 I Want to Test Locally RIGHT NOW
→ **[docs/LOCAL_SETUP_AND_TEST.md](docs/LOCAL_SETUP_AND_TEST.md)** (20-30 min)
- Complete step-by-step guide
- Start Docker services
- Test all 4 endpoints
- Troubleshoot issues
- **USE THIS ONE** ✅

## 📖 I Want Project Overview
→ **[README.md](README.md)**
- What is ZepIris
- Architecture diagram
- Key features
- Tech stack

## 🔧 I Want to Set Up For Development
→ **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**
- System requirements
- Local Python setup (without Docker)
- Configuration options
- Performance tuning

## 📚 I Want Complete API Documentation
→ **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)**
- All 7 endpoints documented
- Request/response examples
- Error codes
- cURL & Python examples

## ⚙️ I Want to Configure Everything
→ **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**
- All environment variables
- Milvus settings
- MinIO settings
- IQA thresholds
- Performance tuning

## 👨‍💻 I Want to Contribute Code
→ **[CONTRIBUTING.md](CONTRIBUTING.md)**
- Development workflow
- Code style & testing
- Type hints & docstrings
- Pull request process

## 📋 I Want Status Report
→ **[docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)**
- What was built
- Requirements checklist
- Files modified
- Next steps

## The Fastest Path to Testing

```bash
# 1. Start services
cd zepiris
docker-compose up -d

# 2. Check health
curl http://localhost:8000/healthz

# 3. Open interactive API docs
# http://localhost:8000/docs

# 4. For detailed tests, follow LOCAL_SETUP_AND_TEST.md Step 5-7
```

---

## File Cleanup Summary

**Removed (redundant):**
- ❌ LOCAL_TESTING.md (old version)
- ❌ TASKS_COMPLETED.md (status tracked in IMPLEMENTATION_COMPLETE.md)
- ❌ QUICK_REFERENCE.txt (text version, markdown better)
- ❌ IMPLEMENTATION_SUMMARY.md (replaced by IMPLEMENTATION_COMPLETE.md)
- ❌ CRISP_SUMMARY.md (content merged)
- ❌ CONTAINER_SETUP.md (details in SETUP_GUIDE.md)
- ❌ DOCUMENTATION_INDEX.md (this file replaces it)

**Kept (essential):**
- ✅ README.md - Project overview
- ✅ LOCAL_SETUP_AND_TEST.md - **PRIMARY TESTING GUIDE** ⭐
- ✅ SETUP_GUIDE.md - Development setup
- ✅ CONTRIBUTING.md - Code guidelines
- ✅ IMPLEMENTATION_COMPLETE.md - Status report
- ✅ docs/API_REFERENCE.md - API documentation
- ✅ docs/CONFIGURATION.md - Configuration reference

---

## Next Steps

1. **Read**: [README.md](README.md) (5 min) — Project overview & architecture
2. **Test**: [docs/LOCAL_SETUP_AND_TEST.md](docs/LOCAL_SETUP_AND_TEST.md) (20-30 min) — Docker Compose setup & testing
3. **Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — API endpoints & examples
4. **Contribute**: [CONTRIBUTING.md](CONTRIBUTING.md) — Development workflow
5. **Deploy**: Push to GitHub when done ✅
