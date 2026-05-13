# Codex Work Log

Ngày cập nhật: 2026-05-13

## Tổng quan

Repo: `d:\AI_Sport\sport-companion-ai`

Mục tiêu đã làm:

- Mở rộng backend Python rule engine cho nhiều bài tập.
- Mở API metadata để FE `test-ai.html` tự gọi `/exercises`.
- Thêm category Cầu lông và Yoga.
- Sửa lỗi 2D skeleton playback không có frames khi `keyframes` rỗng.
- Test API thật với video fixture và test AI/NVIDIA enrichment.

API local hiện chạy ở:

```text
http://127.0.0.1:8000
```

## Files đã thay đổi chính

- `sport_companion_ai/exercises/badminton.py`
- `sport_companion_ai/exercises/yoga.py`
- `sport_companion_ai/exercises/__init__.py`
- `sport_companion_ai/sampling.py`
- `test-ai.html`
- `tests/exercises/test_new_rules.py`
- `tests/test_api.py`
- `tests/test_analyzer.py`
- `tests/test_sampling.py`

## API Metadata

Endpoint:

```http
GET /exercises
```

Trả về list object trong key `exercises`, gồm:

- `name`
- `display_name_vi`
- `category`
- `equipment`
- `movement_type`
- `primary_joints`
- `issue_codes`

Endpoint phân tích:

```http
POST /analyze
```

Giữ contract cũ:

- `video`
- `exercise`
- `skeleton_output`: `full`, `sampled`, `keyframes`, `none`
- `enrich`: `true` hoặc `false`

## Danh sách bài tập hiện có

Tổng cộng: 47 bài.

### Gym / Thể hình: 10 bài

- `squat`
- `deadlift`
- `bench_press`
- `push_up`
- `bicep_curl`
- `overhead_press`
- `pull_up`
- `lunge`
- `plank`
- `lateral_raise`

### Cầu lông: 27 bài

Nhóm kỹ thuật đánh cầu:

- `badminton_clear` - Cầu lông - Phông cầu
- `badminton_backhand_clear` - Cầu lông - Phông trái tay
- `badminton_smash` - Cầu lông - Đập cầu
- `badminton_drop_shot` - Cầu lông - Bỏ nhỏ
- `badminton_drive` - Cầu lông - Tạt cầu
- `badminton_push_shot` - Cầu lông - Đẩy cầu / Tạt cầu
- `badminton_serve` - Cầu lông - Giao cầu
- `badminton_low_serve` - Cầu lông - Giao cầu thấp ngắn
- `badminton_high_serve` - Cầu lông - Giao cầu cao sâu
- `badminton_net_shot` - Cầu lông - Đánh lưới
- `badminton_lift_shot` - Cầu lông - Búng cầu / Lob lưới
- `badminton_net_kill` - Cầu lông - Bỏ nhỏ / Chụp lưới
- `badminton_juggle` - Cầu lông - Tâng cầu
- `badminton_defensive_block` - Cầu lông - Chặn cầu
- `badminton_jump_smash` - Cầu lông - Bước nhảy đánh cầu

Nhóm bộ pháp / di chuyển:

- `badminton_lunge` - Cầu lông - Bước lunge đỡ cầu
- `badminton_split_step` - Cầu lông - Split-step
- `badminton_front_corners_footwork` - Cầu lông - Di chuyển 2 góc lưới
- `badminton_rear_corners_footwork` - Cầu lông - Di chuyển 2 góc cuối sân
- `badminton_mid_corners_footwork` - Cầu lông - Di chuyển 2 góc giữa sân
- `badminton_forward_backward_footwork` - Cầu lông - Di chuyển tiến lùi thẳng đứng
- `badminton_multi_point_footwork` - Cầu lông - Di chuyển đa điểm

Nhóm thể lực / phản xạ:

- `badminton_multi_shuttle` - Cầu lông - Tập đa cầu
- `badminton_wall_rally` - Cầu lông - Đánh cầu vào tường
- `badminton_jump_rope` - Cầu lông - Nhảy dây
- `badminton_interval_run` - Cầu lông - Chạy biến tốc
- `badminton_heavy_racket` - Cầu lông - Tập với vợt nặng

### Yoga: 10 bài

- `yoga_warrior_ii` - Yoga - Chiến binh II
- `yoga_tree_pose` - Yoga - Tư thế cây
- `yoga_downward_dog` - Yoga - Chó úp mặt
- `yoga_chair_pose` - Yoga - Tư thế ghế
- `yoga_cobra_pose` - Yoga - Rắn hổ mang
- `yoga_triangle_pose` - Yoga - Tư thế tam giác
- `yoga_bridge_pose` - Yoga - Tư thế cây cầu
- `yoga_child_pose` - Yoga - Tư thế em bé
- `yoga_boat_pose` - Yoga - Tư thế con thuyền
- `yoga_low_lunge` - Yoga - Low lunge

## FE test-ai.html

Đã cập nhật:

- Dropdown bài tập load từ `/exercises`.
- Group theo category:
  - Thể hình
  - Cầu lông
  - Yoga
- Breadcrumb màu cam là category, màu tím là bài tập.
- 2D skeleton playback dùng frames từ API.
- Progress khi bật AI/NVIDIA có timer để tránh tưởng bị treo.

Lưu ý khi test:

- Nếu muốn ra nhanh: tắt checkbox `AI gợi ý`.
- Nếu bật `AI gợi ý`: NVIDIA/NIM có thể mất khoảng 60-90 giây.
- Nếu không có skeleton 2D: video có thể không detect được pose rõ. Nên quay đủ sáng, thấy toàn thân, không cắt mất người.

## Fix skeleton playback

Lỗi cũ:

```text
Không có skeleton frames để phát 2D. Chọn skeleton là sampled hoặc full rồi phân tích lại.
```

Nguyên nhân:

- Khi `skeleton_output=keyframes` và không detect được reps/keyframes, backend trả `frames: []`.
- FE không có frame để vẽ 2D.

Fix:

- `sport_companion_ai/sampling.py`:
  - `keyframes` fallback sang sampled frames nếu không có keyframes.
  - `sampled` ưu tiên frame có skeleton nếu sample đều rơi vào frame không có skeleton.

## NVIDIA/NIM

Config:

- `.env` có `NVIDIA_API_KEY`.
- Nếu không có `NVIDIA_NIM_MODEL`, code dùng default:

```text
qwen/qwen3-next-80b-a3b-instruct
```

Luồng đúng:

- MediaPipe + deterministic rules tạo JSON trước.
- NVIDIA/NIM chỉ enrich report sau.
- Nếu NIM lỗi, `/analyze` không nên fail toàn bộ; trả warning enrichment.

## Test đã chạy

Backend/API/analyzer/sampling:

```bash
pytest tests/test_api.py tests/test_analyzer.py tests/test_sampling.py -q
# 77 passed
```

Full suite gần nhất:

```bash
pytest -q
# 315 passed, 17 deselected
```

Test API thật với video:

```text
Video: tests/fixtures/videos/squat_emptybar.mp4
exercise=squat
skeleton_output=sampled
enrich=false
HTTP=200
total_reps=3
avg_score=75.0
frames=38
```

Test AI/NVIDIA thật:

```text
Video: tests/fixtures/videos/squat_emptybar.mp4
exercise=squat
skeleton_output=sampled
enrich=true
HTTP=200
enriched=true
ai_feedback=true
thời gian khoảng 72 giây
```

## Trạng thái Git

Các thay đổi hiện vẫn local, chưa push GitHub.

Các file đang modified/untracked:

```text
M sport_companion_ai/exercises/__init__.py
M sport_companion_ai/exercises/badminton.py
M sport_companion_ai/sampling.py
M test-ai.html
M tests/exercises/test_new_rules.py
M tests/test_analyzer.py
M tests/test_api.py
M tests/test_sampling.py
?? sport_companion_ai/exercises/yoga.py
?? codex.md
```

Remote đã từng dùng:

```text
coach-ai -> https://github.com/MinhThien31/Coach-AI.git
```

Branch:

```text
main tracks coach-ai/main
```

## Cách chạy lại

Start API:

```bash
uvicorn api.main:app --reload
```

Hoặc:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Mở test UI:

```text
test-ai.html
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

List bài tập:

```bash
curl http://127.0.0.1:8000/exercises
```
