"""End-to-end demo: analyze a video and print the JSON report.

Usage:
  python examples/analyze_squat.py path/to/squat.mp4 [--exercise squat] [--skeleton keyframes]

Loads `.env` from the project root if present (NVIDIA_API_KEY for --enrich-with-nim).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no external dep). Reads `KEY=VALUE` lines."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

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
