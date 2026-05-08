from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import Issue, RepEvaluation
from sport_companion_ai.sampling import select_frames_for_output, SkeletonOutputMode


def make_frame(i: int) -> Frame:
    return Frame(index=i, timestamp_ms=i * 33, skeleton=Skeleton(
        keypoints={"nose": Keypoint(x=0.5, y=0.5, visibility=1.0)}))


def make_eval(rep_index: int, start: int, peak: int, end: int,
              issue_frames: list[int] | None = None) -> RepEvaluation:
    return RepEvaluation(
        rep_index=rep_index, score=80, passed=True,
        issues=[Issue(code="X", severity="LOW", message_vi="",
                      frame_indices=issue_frames or [])],
        keyframes={"start": start, "peak": peak, "end": end},
    )


def test_full_returns_all():
    frames = [make_frame(i) for i in range(100)]
    out = select_frames_for_output(frames, [], SkeletonOutputMode.FULL, fps=30)
    assert len(out) == 100


def test_none_returns_empty():
    frames = [make_frame(i) for i in range(100)]
    out = select_frames_for_output(frames, [], SkeletonOutputMode.NONE, fps=30)
    assert out == []


def test_sampled_at_5fps():
    frames = [make_frame(i) for i in range(60)]
    out = select_frames_for_output(frames, [], SkeletonOutputMode.SAMPLED, fps=30)
    assert 8 <= len(out) <= 12
    assert out[0].index == 0


def test_keyframes_only_returns_rep_keyframes_and_issue_indices():
    frames = [make_frame(i) for i in range(100)]
    reps = [
        make_eval(0, start=10, peak=20, end=30, issue_frames=[25]),
        make_eval(1, start=50, peak=60, end=70, issue_frames=[55, 58]),
    ]
    out = select_frames_for_output(frames, reps, SkeletonOutputMode.KEYFRAMES, fps=30)
    out_indices = {f.index for f in out}
    assert out_indices == {10, 20, 25, 30, 50, 55, 58, 60, 70}
