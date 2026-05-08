"""End-to-end demo: analyze a video and print the JSON report.

Usage:
  python examples/analyze_squat.py path/to/squat.mp4 [--exercise squat] [--skeleton keyframes]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from sport_companion_ai import VideoAnalyzer, SkeletonOutputMode
from sport_companion_ai.feedback.nim import NvidiaNimEnricher


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a gym video for form correctness")
    parser.add_argument("video", help="Path to MP4 video")
    parser.add_argument("--exercise", default="squat",
                        choices=["squat", "deadlift", "bench_press", "push_up", "bicep_curl"])
    parser.add_argument("--skeleton", default="keyframes",
                        choices=[m.value for m in SkeletonOutputMode])
    parser.add_argument("--enrich-with-nim", action="store_true",
                        help="Use NVIDIA NIM for natural-language session summary")
    args = parser.parse_args()

    enricher = None
    if args.enrich_with_nim:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            print("ERROR: --enrich-with-nim requires NVIDIA_API_KEY", file=sys.stderr)
            return 2
        enricher = NvidiaNimEnricher(api_key=api_key)

    analyzer = VideoAnalyzer(enricher=enricher)
    report = analyzer.analyze(args.video, exercise=args.exercise, skeleton_output=args.skeleton)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
