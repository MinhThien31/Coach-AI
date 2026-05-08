from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import VideoMeta
from sport_companion_ai.warnings import detect_warnings


def make_frame(i: int, vis: float = 1.0):
    skel = Skeleton(keypoints={"nose": Keypoint(x=0.5, y=0.5, visibility=vis)})
    return Frame(index=i, timestamp_ms=i * 33, skeleton=skel)


def test_no_warnings_clean_video():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=3)
    assert warns == []


def test_low_pose_confidence_when_many_frames_low_vis():
    frames = [make_frame(i, vis=0.3) for i in range(50)] + [make_frame(i, vis=0.9) for i in range(50, 100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=3)
    codes = {w.code for w in warns}
    assert "LOW_POSE_CONFIDENCE" in codes


def test_no_reps_detected():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=0)
    codes = {w.code for w in warns}
    assert "NO_REPS_DETECTED" in codes


def test_video_too_short():
    frames = [make_frame(i) for i in range(60)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=2000)
    warns = detect_warnings(frames, meta, n_reps=1)
    codes = {w.code for w in warns}
    assert "VIDEO_TOO_SHORT" in codes


def test_low_fps_warning():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=10, duration_ms=10000)
    warns = detect_warnings(frames, meta, n_reps=3)
    codes = {w.code for w in warns}
    assert "LOW_FPS" in codes
