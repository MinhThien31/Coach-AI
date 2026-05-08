# Batch 1 Partial Results — Phase A (Meta) + Phase G (CORS)

**Date:** 2026-05-08  
**Tester:** automated curl batch  
**Server:** `http://localhost:8001` (uvicorn, 1 worker)  
**Evidence dir:** `evidence/batch1/`

---

## Pass/Fail Table

| TC | Description | Status | Note |
|---|---|---|---|
| A1 | Health check | ✅ PASS | 200, body `{"status":"ok"}` — exact match. Evidence: `evidence/batch1/A1.txt` |
| A2 | Exercises list | ✅ PASS | 200, all 5 exercises present in correct order (`bench_press, bicep_curl, deadlift, push_up, squat`). Evidence: `evidence/batch1/A2.txt` |
| A3 | OpenAPI schema | ✅ PASS | 200, JSON valid; title = "Sport Companion AI API", version = "0.1.0", paths = `['/analyze', '/exercises', '/health']`, analyze methods = `['post']`. Evidence: `evidence/batch1/A3.txt` |
| A4 | Swagger UI render | ✅ PASS | 200, `content-type: text/html; charset=utf-8`, body contains `swagger-ui` (4 occurrences). Evidence: `evidence/batch1/A4.txt` |
| A5 | ReDoc render | ✅ PASS | 200, `content-type: text/html; charset=utf-8`. Evidence: `evidence/batch1/A5.txt` |
| A6 | 404 unknown path | ✅ PASS | 404, body `{"detail":"Not Found"}` — FastAPI default, exact match. Evidence: `evidence/batch1/A6.txt` |
| G1 | CORS preflight | ✅ PASS | 200, `access-control-allow-origin: *`, `access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT` (includes POST). Evidence: `evidence/batch1/G1.txt` |

**Summary: 7/7 PASS**

---

## Findings

No deviations from expected behavior. All test cases pass cleanly.

---

## Notes

- **G1 (CORS):** The `access-control-allow-methods` header returns the full method list (`DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT`) rather than just `POST`. This is standard FastAPI/Starlette CORS middleware behavior and is correct — the plan only required POST to be included, which it is. No finding raised.
- **G1 (CORS):** Response body is `OK` (plain text, 2 bytes) rather than empty. This is standard Starlette behavior for preflight responses. No functional impact.
- **A2:** Exercises are returned as a flat list in alphabetical order. Schema is `{"exercises": [...]}` — consistent with the plan's `{"exercises":[...]}` shape.
- **A3:** `analyze methods` returns `['post']` (lowercase), confirming OpenAPI spec correctly registers only POST on `/analyze`.
- **A4:** `swagger-ui` string appears 4 times in the Swagger UI HTML — all within script/link tags referencing the swagger-ui CDN bundle. Passes the `grep -c "swagger-ui" > 0` assertion.
- All responses returned within milliseconds — no latency concerns for meta endpoints.
