from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.render import (
    _build_reaction_filter_complex,
    _build_reaction_timeline_filter_complex,
    _percent_crop_to_pixels,
    build_manual_chapter_segments,
    create_caption_overlay_png,
)
from src.pipeline import _normalize_imported_clip_plan
from src.web.app import ProcessRequest, _normalize_ai_clip_plan, repair_ai_json


def test_percent_crop_to_pixels_clamps_and_even_sizes() -> None:
    crop = {
        "x_percent": 90,
        "y_percent": 90,
        "width_percent": 30,
        "height_percent": 30,
    }

    x, y, width, height = _percent_crop_to_pixels(crop, 1920, 1080)

    assert x + width <= 1920
    assert y + height <= 1080
    assert width % 2 == 0
    assert height % 2 == 0
    assert width >= 2
    assert height >= 2


def test_caption_overlay_png_generation(tmp_path: Path) -> None:
    out = create_caption_overlay_png(
        "This is a bold reaction caption that should wrap",
        tmp_path / "caption.png",
        output_width=1080,
        font_size=44,
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_process_request_reaction_layout_validation() -> None:
    payload = {
        "url": "input.mp4",
        "title": "Title",
        "channel": "anime recaps",
        "y_scale_mode": "reaction_layout",
        "reaction_layout_enabled": True,
        "main_crop": {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
        "facecam_crop": {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
        "facecam_shape": "circle",
        "caption_duration_mode": "custom",
        "caption_duration_seconds": 4.5,
    }

    request = ProcessRequest.model_validate(payload)

    assert request.reaction_layout_enabled is True
    assert request.main_crop is not None
    assert request.facecam_crop is not None

    with pytest.raises(ValidationError):
        ProcessRequest.model_validate({**payload, "caption_duration_seconds": 0})


def test_reaction_filter_complex_contains_expected_graph() -> None:
    filter_complex, video_map, debug = _build_reaction_filter_complex(
        source_width=1920,
        source_height=1080,
        output_width=1080,
        output_height=1920,
        main_crop={"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
        facecam_crop={"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
        reaction_layout_preset="content_top_facecam_bottom",
        caption_text="Hello",
        caption_input_index=1,
        caption_position="between",
        caption_duration_mode="first_5_seconds",
        caption_duration_seconds=None,
        overlay_y_percent=50,
        playback_speed=1.0,
        subtitle_ass_path=None,
    )

    assert "vstack=inputs=2" in filter_complex
    assert "overlay=x=(W-w)/2" in filter_complex
    assert "enable='between(t,0,5)'" in filter_complex
    assert video_map == "vout"
    assert debug["main_crop_pixels"] == (0, 0, 1920, 702)


def test_process_request_accepts_reaction_timeline() -> None:
    payload = {
        "url": "input.mp4",
        "title": "Title",
        "channel": "anime recaps",
        "y_scale_mode": "reaction_layout",
        "reaction_layout_enabled": True,
        "reaction_layout_mode": "timeline",
        "reaction_timeline": [
            {
                "start": "00:00:00",
                "end": "00:00:10",
                "caption": "First layout",
                "layout_preset": "facecam_right",
                "facecam_shape": "rounded",
                "caption_duration": "first_3_seconds",
                "main_crop": {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
                "facecam_crop": {"x_percent": 65, "y_percent": 0, "width_percent": 35, "height_percent": 35},
                "divider_split": {"direction": "horizontal", "ratio": 0.65},
            }
        ],
    }

    request = ProcessRequest.model_validate(payload)

    assert request.reaction_layout_mode == "timeline"
    assert request.reaction_timeline[0].facecam_shape == "rounded_rectangle"
    assert request.reaction_timeline[0].caption_duration == "first_3_seconds"


def test_reaction_timeline_filter_complex_trims_and_concats() -> None:
    intervals = [
        {
            "start": 0.0,
            "end": 4.0,
            "row": {
                "id": "a",
                "caption": "A",
                "caption_enabled": True,
                "caption_duration": "first_3_seconds",
                "layout_preset": "main_top_reaction_bottom",
                "keep_aspect_ratio": True,
                "main_crop": {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 65},
                "facecam_crop": {"x_percent": 60, "y_percent": 55, "width_percent": 35, "height_percent": 40},
                "divider_split": {"direction": "horizontal", "ratio": 0.65},
                "facecam_shape": "rectangle",
            },
        },
        {
            "start": 4.0,
            "end": 8.0,
            "row": {
                "id": "b",
                "caption": "",
                "caption_enabled": False,
                "layout_preset": "facecam_right",
                "keep_aspect_ratio": False,
                "main_crop": {"x_percent": 0, "y_percent": 0, "width_percent": 100, "height_percent": 100},
                "facecam_crop": {"x_percent": 65, "y_percent": 0, "width_percent": 35, "height_percent": 35},
                "facecam_shape": "rectangle",
            },
        },
    ]

    filter_complex, video_map, debug = _build_reaction_timeline_filter_complex(
        source_width=1920,
        source_height=1080,
        output_width=1080,
        output_height=1920,
        intervals=intervals,
        caption_input_indexes={0: 1},
        overlay_y_percent=50,
        playback_speed=1.0,
        subtitle_ass_path=None,
    )

    assert "trim=start=0.000:end=4.000" in filter_complex
    assert "trim=start=4.000:end=8.000" in filter_complex
    assert "concat=n=2:v=1:a=0" in filter_complex
    assert "enable='between(t,0,3)'" in filter_complex
    assert video_map == "vout"
    assert len(debug["timeline_intervals"]) == 2


def test_process_request_accepts_imported_clip_plan() -> None:
    request = ProcessRequest.model_validate(
        {
            "url": "input.mp4",
            "title": "Title",
            "channel": "anime recaps",
            "split_mode": "manual",
            "imported_clip_plan": {
                "video_title": "Source",
                "clips": [
                    {
                        "id": 1,
                        "enabled": True,
                        "start": "00:00:03",
                        "end": "00:00:45",
                        "title": "The First Twist",
                        "caption_text": "Nobody saw this coming",
                        "suggested_layout": "standard_vertical",
                    }
                ],
            },
        }
    )

    assert request.imported_clip_plan is not None
    assert request.imported_clip_plan["clips"][0]["caption_text"] == "Nobody saw this coming"


def test_imported_clip_plan_builds_exact_segments_and_metadata() -> None:
    segments, titles, metadata, normalized = _normalize_imported_clip_plan(
        {
            "video_title": "Source",
            "clips": [
                {
                    "id": 7,
                    "start": "00:00:10",
                    "end": "00:01:10",
                    "title": "Day 1 Went Wrong",
                    "caption_text": "Day 1 Went Wrong Fast",
                    "clip_type": "failure",
                    "mood": "chaotic",
                    "enabled": True,
                }
            ],
        }
    )

    assert segments[0].start == 10
    assert segments[0].end == 70
    assert titles == ["Day 1 Went Wrong Fast"]
    assert metadata[0]["clip_type"] == "failure"
    assert normalized is not None
    assert normalized["clips"][0]["id"] == 7


def test_manual_chapters_support_explicit_end_times() -> None:
    segments, titles = build_manual_chapter_segments(
        [
            {"start": "00:00:05", "end": "00:00:20", "title": "First Clip"},
            {"start": "00:00:30", "end": "00:00:50", "title": "Second Clip"},
        ],
        total_duration=120,
    )

    assert [(segment.start, segment.end) for segment in segments] == [(5, 20), (30, 50)]
    assert titles == ["First Clip", "Second Clip"]


def test_manual_chapter_end_at_duration_is_allowed() -> None:
    segments, _titles = build_manual_chapter_segments(
        [{"start": "00:38:30", "end": "00:39:09", "title": "Final Clip"}],
        total_duration=2349.0,
    )

    assert segments[0].end == 2349.0


def test_manual_chapter_final_end_near_duration_is_clamped() -> None:
    logs: list[str] = []
    segments, _titles = build_manual_chapter_segments(
        [{"start": "00:38:30", "end": "00:39:09", "title": "Final Clip"}],
        total_duration=2348.25,
        log=logs.append,
    )

    assert segments[0].end == 2348.25
    assert logs == ["Clamped final clip end to video duration."]


def test_manual_chapter_end_beyond_tolerance_still_fails() -> None:
    with pytest.raises(ValueError, match="ends after the video duration"):
        build_manual_chapter_segments(
            [{"start": "00:38:30", "end": "00:39:09", "title": "Final Clip"}],
            total_duration=2347.8,
        )


def test_ai_json_repair_merges_adjacent_metadata_object() -> None:
    payload = repair_ai_json(
        """
        ```json
        {"video_title":"Source","clips":[{"start":"00:00:01","end":"00:00:20","title":"First"}]}
        {"import_ready_for_quickclips": true}
        ```
        """
    )

    assert payload["import_ready_for_quickclips"] is True
    assert payload["clips"][0]["title"] == "First"


def test_ai_clip_plan_normalizes_array_to_quickclips_shape() -> None:
    plan = _normalize_ai_clip_plan(
        [
            {
                "start": "00:00:03",
                "end": "00:00:45",
                "title": "The Hook",
                "caption_text": "The Real Hook",
                "summary": "A short moment.",
            }
        ]
    )

    assert plan["import_ready_for_quickclips"] is True
    assert plan["clips"][0]["caption_text"] == "The Real Hook"
