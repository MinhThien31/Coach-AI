# Batch 2 Partial Results — Phase B (Happy Paths) + Phase E (Skeleton Modes)

**Date:** 2026-05-08  
**Tester:** Batch-2 sub-agent  
**Server:** `uvicorn api.main:app --port 8001 --workers 1`  
**Evidence dir:** `docs/superpowers/test-runs/2026-05-08-phase-2-api/evidence/batch2/`

---

## Phase B — Happy Paths Summary Table

| TC | Fixture | Exercise | HTTP | total_reps | passed_reps | avg_score | len(frames) | Manifest OK? | Status |
|---|---|---|---|---|---|---|---|---|---|
| B1 | squat_emptybar.mp4 | squat | 200 | 3 | 0 | 75.0 | 9 | ✅ reps 2–5, score 50–100 | ✅ PASS |
| B2 | deadlift_man.mp4 | deadlift | 200 | 3 | 0 | 65.0 | 7 | ✅ reps 1–5, score 40–100 | ✅ PASS |
| B3 | bench_woman.mp4 | bench_press | 200 | 1 | 1 | 95.0 | 3 | ✅ reps 1–3, passed_reps 1–3, score 80–100 | ✅ PASS |
| B4 | pushup_incline.mp4 | push_up | 200 | 16 | 0 | 75.0 | 37 | ✅ reps 3–25 | ✅ PASS |
| B5 | curl_dumbbell.mp4 | bicep_curl | 200 | 7 | 7 | 83.6 | 19 | ✅ reps 4–12, passed_reps 4–12, score 70–100 | ✅ PASS |

### Required Fields Check (all B1–B5)

All five responses include:

| Field | Present |
|---|---|
| `exercise` (string, matches request) | ✅ all 5 |
| `version` | ✅ all 5 (`"0.1.0"`) |
| `pose_model` | ✅ all 5 (`"mediapipe-blazepose-full"`) |
| `enriched` (bool, `false`) | ✅ all 5 |
| `video.fps` (int) | ✅ all 5 |
| `video.duration_ms` (int) | ✅ all 5 |
| `skeleton_schema.keypoint_names` (len=33) | ✅ all 5 |
| `skeleton_schema.edges` (len=12) | ✅ all 5 |
| `frames` (array, default keyframes mode) | ✅ all 5 |
| `total_reps` (int >= 1) | ✅ all 5 |
| `passed_reps` (int) | ✅ all 5 |
| `avg_score` (float) | ✅ all 5 |
| `reps` (array, len == total_reps) | ✅ all 5 |
| `reps[].score` | ✅ all 5 |
| `reps[].passed` / `reps[].inconclusive` | ✅ all 5 |
| `reps[].issues` | ✅ all 5 |
| `reps[].metrics` | ✅ all 5 |
| `reps[].keyframes` | ✅ all 5 (len=3 per rep in keyframes mode) |
| `warnings` (array, may be empty) | ✅ all 5 |
| `session_summary` (null when not enriched) | ✅ all 5 (`null`) |

### Manifest Range Verification

| TC | Check | Manifest Range | Observed | In Range |
|---|---|---|---|---|
| B1 | total_reps | 2–5 | 3 | ✅ |
| B1 | avg_score | 50–100 | 75.0 | ✅ |
| B2 | total_reps | 1–5 | 3 | ✅ |
| B2 | avg_score | 40–100 | 65.0 | ✅ |
| B3 | total_reps | 1–3 | 1 | ✅ |
| B3 | passed_reps | 1–3 | 1 | ✅ |
| B3 | avg_score | 80–100 | 95.0 | ✅ |
| B3 | required_issues_absent: BENCH_PARTIAL_ROM | — | absent | ✅ |
| B3 | required_issues_absent: BENCH_ELBOW_FLARE | — | absent | ✅ |
| B4 | total_reps | 3–25 | 16 | ✅ |
| B5 | total_reps | 4–12 | 7 | ✅ |
| B5 | passed_reps | 4–12 | 7 | ✅ |
| B5 | avg_score | 70–100 | 83.6 | ✅ |

---

## Phase E — Skeleton Output Modes Summary Table

All E1–E4 use `squat_emptybar.mp4` (fps=25, duration_ms=7600 → expected_full=190 frames), squat exercise.

| TC | skeleton_output | HTTP | len(frames) | total_reps | Expected | Status |
|---|---|---|---|---|---|---|
| E1 | `full` | 200 | 190 | 3 | ~190 (fps×dur/1000=190) | ✅ PASS |
| E2 | `sampled` | 200 | 38 | 3 | >0 and <190 | ✅ PASS |
| E3 | `keyframes` | 200 | 9 | 3 | 3×3=9 | ✅ PASS |
| E4 | `none` | 200 | 0 | 3 | 0 | ✅ PASS |

### E3 Per-Rep Keyframe Detail

| Rep | keyframes len |
|---|---|
| rep[0] | 3 (start/peak/end) |
| rep[1] | 3 (start/peak/end) |
| rep[2] | 3 (start/peak/end) |

`len(frames) = 9 == 3 × total_reps` exactly. ✅

---

## Findings & Deviations

### LOW — B1, B2, B4: passed_reps = 0

**TC:** B1 (squat_emptybar), B2 (deadlift_man), B4 (pushup_incline)  
**Severity:** LOW  
**Observation:** All reps marked `passed=False` due to known camera-angle false positives:
- B1: `SQUAT_BACK_TOO_VERTICAL` + `SQUAT_KNEE_VALGUS` — front-facing camera inflates valgus ratio. Documented in manifest.  
- B2: `DEADLIFT_BACK_ROUND` — may be real or false positive from bar occlusion. Documented in manifest.  
- B4: `PUSHUP_HIP_SAG` — incline angle shifts midline, fires on all 16 reps. Documented in manifest.  
**Disposition:** No fix needed; all documented known behavior. Manifest does not require `passed_reps >= 1` for these fixtures.

### LOW — B2: LOW_POSE_CONFIDENCE warning

**TC:** B2 (deadlift_man.mp4)  
**Severity:** LOW  
**Observation:** Warning present: `{"code": "LOW_POSE_CONFIDENCE", "message_vi": "Khoảng 49% frames có pose detection yếu", "affected_frame_count": 85}`  
**Disposition:** Expected — deadlift from side with bar partially occluding the skeleton. Correctly surfaced as a `warnings[]` entry with Vietnamese message.

### INFO — E2 sampled: 38 frames from 190 full (~20%)

**Severity:** INFO (not a deviation — no exact ratio specified)  
**Observation:** `sampled` mode returns 38/190 frames. This implies approximately 5 fps effective sampling rate (every 5th frame). The plan only requires `> 0 and < full`. ✅ satisfied.

---

## Notes

- All 9 `/analyze` calls returned **200 OK**. No 4xx or 5xx observed.
- MediaPipe first-call warmup was not measurably slow — server was already warm from Batch 1 / prior runs.
- B3 (bench_woman) is the cleanest result: `avg_score=95`, `passed_reps=1/1`, only `BENCH_ASYMMETRY` (LOW, non-blocking). Confirms the pipeline correctly produces high-confidence "good rep" when camera angle + form are both clean.
- B5 (curl_dumbbell) is the strongest multi-rep result: 7/7 reps pass, `avg_score=83.6`. CURL_ELBOW_DRIFT and CURL_TOO_FAST are LOW/MEDIUM severity and don't block pass verdict — correct behavior.
- `skeleton_schema.keypoint_names` consistently returns 33 keypoints (MediaPipe BlazePose full body skeleton) across all fixtures.
- Response `Content-Type: application/json` confirmed in all header files.

---

## Evidence Files

| TC | Body JSON | Headers |
|---|---|---|
| B1 | `evidence/batch2/B1-body.json` | `evidence/batch2/B1-headers.txt` |
| B2 | `evidence/batch2/B2-body.json` | `evidence/batch2/B2-headers.txt` |
| B3 | `evidence/batch2/B3-body.json` | `evidence/batch2/B3-headers.txt` |
| B4 | `evidence/batch2/B4-body.json` | `evidence/batch2/B4-headers.txt` |
| B5 | `evidence/batch2/B5-body.json` | `evidence/batch2/B5-headers.txt` |
| E1 | `evidence/batch2/E1-body.json` | `evidence/batch2/E1-headers.txt` |
| E2 | `evidence/batch2/E2-body.json` | `evidence/batch2/E2-headers.txt` |
| E3 | `evidence/batch2/E3-body.json` | `evidence/batch2/E3-headers.txt` |
| E4 | `evidence/batch2/E4-body.json` | `evidence/batch2/E4-headers.txt` |
