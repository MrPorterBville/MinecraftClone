from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class _FrameState:
    kind: str
    start_time: float
    context: dict[str, Any] = field(default_factory=dict)
    section_totals_ms: dict[str, float] = field(default_factory=dict)


class RuntimeProfiler:
    def __init__(self, enabled: bool = True, slow_frame_ms: float = 25.0, max_slow_frames: int = 400) -> None:
        self.enabled = enabled
        self.slow_frame_ms = slow_frame_ms
        self.max_slow_frames = max_slow_frames
        self.section_samples_ms: dict[str, list[float]] = defaultdict(list)
        self.frame_samples_ms: dict[str, list[float]] = defaultdict(list)
        self.slow_frames: list[dict[str, Any]] = []
        self._active_frame: _FrameState | None = None

    def begin_frame(self, kind: str, context: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        if self._active_frame is not None:
            self.end_frame({"warning": "frame_auto_closed"})
        self._active_frame = _FrameState(kind=kind, start_time=time.perf_counter(), context=dict(context or {}))

    def end_frame(self, extra_context: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        frame = self._active_frame
        if frame is None:
            return
        self._active_frame = None

        total_ms = (time.perf_counter() - frame.start_time) * 1000.0
        frame_key = f"frame.{frame.kind}"
        self.frame_samples_ms[frame_key].append(total_ms)

        if total_ms >= self.slow_frame_ms:
            context = dict(frame.context)
            if extra_context:
                context.update(extra_context)
            self.slow_frames.append(
                {
                    "kind": frame.kind,
                    "total_ms": total_ms,
                    "context": context,
                    "sections_ms": dict(sorted(frame.section_totals_ms.items(), key=lambda item: item[1], reverse=True)),
                }
            )
            if len(self.slow_frames) > self.max_slow_frames:
                self.slow_frames.pop(0)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_section_ms(name, (time.perf_counter() - start) * 1000.0)

    def record_section_ms(self, name: str, duration_ms: float) -> None:
        if not self.enabled:
            return
        self.section_samples_ms[name].append(duration_ms)
        frame = self._active_frame
        if frame is None:
            return
        frame.section_totals_ms[name] = frame.section_totals_ms.get(name, 0.0) + duration_ms

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        rank = max(0, min(len(sorted_values) - 1, int(math.ceil(len(sorted_values) * p)) - 1))
        return sorted_values[rank]

    def _stats(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0.0, "avg_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        return {
            "count": float(len(values)),
            "avg_ms": sum(values) / len(values),
            "p95_ms": self._percentile(values, 0.95),
            "p99_ms": self._percentile(values, 0.99),
            "max_ms": max(values),
        }

    @staticmethod
    def clear_previous_reports(output_dir: str | Path = "profiling") -> None:
        out_dir = Path(output_dir)
        if not out_dir.exists():
            return
        patterns = (
            "lag_report_*.txt",
            "lag_report_*.json",
            "lag_report_latest.txt",
            "lag_report_latest.json",
        )
        for pattern in patterns:
            for path in out_dir.glob(pattern):
                try:
                    path.unlink()
                except OSError:
                    continue

    def write_report(self, output_dir: str | Path = "profiling") -> tuple[Path, Path] | None:
        if not self.enabled:
            return None

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        txt_path = out_dir / f"lag_report_{stamp}.txt"
        json_path = out_dir / f"lag_report_{stamp}.json"
        latest_txt = out_dir / "lag_report_latest.txt"
        latest_json = out_dir / "lag_report_latest.json"

        section_stats = {name: self._stats(samples) for name, samples in self.section_samples_ms.items()}
        frame_stats = {name: self._stats(samples) for name, samples in self.frame_samples_ms.items()}
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slow_frame_threshold_ms": self.slow_frame_ms,
            "frame_stats_ms": frame_stats,
            "section_stats_ms": section_stats,
            "slow_frames": self.slow_frames,
        }

        json_text = json.dumps(report, indent=2)
        json_path.write_text(json_text, encoding="utf-8")
        latest_json.write_text(json_text, encoding="utf-8")

        lines: list[str] = []
        lines.append("Minecraft Clone Lag Report")
        lines.append(f"Generated: {report['generated_at']}")
        lines.append(f"Slow frame threshold: {self.slow_frame_ms:.2f} ms")
        lines.append("")
        lines.append("Frame Stats")
        for name, stats in sorted(frame_stats.items(), key=lambda item: item[1]["p99_ms"], reverse=True):
            lines.append(
                f"- {name}: count={int(stats['count'])} avg={stats['avg_ms']:.2f}ms "
                f"p95={stats['p95_ms']:.2f}ms p99={stats['p99_ms']:.2f}ms max={stats['max_ms']:.2f}ms"
            )

        lines.append("")
        lines.append("Section Stats")
        for name, stats in sorted(section_stats.items(), key=lambda item: item[1]["p99_ms"], reverse=True):
            lines.append(
                f"- {name}: count={int(stats['count'])} avg={stats['avg_ms']:.3f}ms "
                f"p95={stats['p95_ms']:.3f}ms p99={stats['p99_ms']:.3f}ms max={stats['max_ms']:.3f}ms"
            )

        lines.append("")
        lines.append(f"Slow Frames ({len(self.slow_frames)})")
        for index, frame in enumerate(sorted(self.slow_frames, key=lambda f: f["total_ms"], reverse=True)[:50], start=1):
            lines.append(f"{index}. {frame['kind']} total={frame['total_ms']:.2f}ms context={frame['context']}")
            top_sections = list(frame["sections_ms"].items())[:5]
            for sec_name, sec_ms in top_sections:
                lines.append(f"   - {sec_name}: {sec_ms:.2f}ms")

        text = "\n".join(lines) + "\n"
        txt_path.write_text(text, encoding="utf-8")
        latest_txt.write_text(text, encoding="utf-8")
        return txt_path, json_path
