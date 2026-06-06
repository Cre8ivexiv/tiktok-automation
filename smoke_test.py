
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.render import (
    Segment,
    build_manual_chapter_segments,
    build_video_filter,
    detect_scene_segments,
    render_parts,
)

def test_filter_string():
    print("Testing build_video_filter logic...")
    vf = build_video_filter(
        part_number=1,
        crop_top_px=0,
        output_width=1080,
        output_height=1920,
        video_y_scale=2.08,
        render_preset="legacy",
        title_mask_px=0, # disable mask for clarity
        part_overlay_enabled=False # disable overlay for clarity
    )
    print(f"Generated Filter Chain:\n{vf}")
    
    expected_parts = [
        "scale=1080:-2",
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black",
        "drawbox=x=0:y=0:w=iw:h=45:color=black@1.0:t=fill",
        "setsar=1",
    ]
    
    for part in expected_parts:
        if part in vf:
            print(f"[OK] Found: {part}")
        else:
            print(f"[FAIL] Missing: {part}")

def run_render_test():
    input_video = Path("temp_test.webm")
    if not input_video.exists():
        print(f"Warning: {input_video} not found, skipping actual render.")
        return

    out_dir = Path("smoke_test_offset")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    print(f"\nRendering short clip from {input_video}...")
    try:
        render_parts(
            input_video=input_video,
            out_dir=out_dir,
            segments=[Segment(0, 3.0)], # 3 seconds
            crop_top_px=0,
            video_y_scale=2.08,
            part_overlay_enabled=False,
            title_mask_px=0
        )
        print("Render complete.")
        print(f"Output saved to {out_dir}/part_1.mp4")
    except Exception as e:
        print(f"Render failed: {e}")

def test_scene_detection() -> None:
    input_video = Path("temp_test.webm")
    if not input_video.exists():
        print(f"Skipping test_scene_detection: {input_video} not found.")
        return

    print(f"\nRunning scene detection on {input_video}...")
    try:
        segments = detect_scene_segments(input_video)
        print(f"Scenes found: {len(segments)}")
        for i, seg in enumerate(segments, start=1):
            print(f"  Scene {i}: start={seg.start:.3f}s  end={seg.end:.3f}s  duration={seg.duration:.3f}s")
    except Exception as e:
        print(f"Scene detection failed: {e}")


def test_manual_chapter_segments() -> None:
    print("\nTesting manual chapter segment builder...")
    segments, titles = build_manual_chapter_segments(
        [
            {"start": "00:00", "title": "Intro"},
            {"start": "00:25", "title": "Part 1"},
            {"start": "00:50", "title": "Part 2"},
        ],
        total_duration=75,
    )

    expected = [
        (0.0, 25.0, "Intro"),
        (25.0, 50.0, "Part 1"),
        (50.0, 75.0, "Part 2"),
    ]
    actual = [(segment.start, segment.end, title) for segment, title in zip(segments, titles)]

    if actual == expected:
        print("[OK] Manual chapter segments resolved correctly.")
    else:
        print(f"[FAIL] Expected {expected} but got {actual}")


if __name__ == "__main__":
    test_filter_string()
    run_render_test()
    test_scene_detection()
    test_manual_chapter_segments()
