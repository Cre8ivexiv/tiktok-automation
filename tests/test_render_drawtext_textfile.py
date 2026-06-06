from __future__ import annotations

from pathlib import Path

import pytest

from src.render import (
    build_drawtext_filter_from_file,
    build_video_filter,
    write_drawtext_textfile,
)


SMOKE_TEXTS = [
    "Europe's GPS?",
    "Flash's RNG",
    "$300,000 Split Or Steal?",
    'He said: "No way!"',
    "Emoji 😳🔥",
]


@pytest.mark.parametrize("text", SMOKE_TEXTS)
def test_drawtext_textfile_filter_does_not_inline_dynamic_text(tmp_path: Path, text: str) -> None:
    textfile = write_drawtext_textfile(text, tmp_path, "caption.txt")
    drawtext = build_drawtext_filter_from_file(
        Path(textfile.name),
        fontsize=64,
        fontcolor="black",
        x_expr="(w-text_w)/2",
        y_expr="(h-text_h)*12/100",
    )

    assert textfile.read_text(encoding="utf-8") == text
    assert "drawtext=textfile='caption.txt'" in drawtext
    assert "text='" not in drawtext
    assert text not in drawtext


def test_build_video_filter_uses_textfiles_for_caption_and_part_label(tmp_path: Path) -> None:
    vf = build_video_filter(
        part_number=1,
        crop_top_px=0,
        output_width=1080,
        output_height=1920,
        manual_caption_text="What is jamming Europe's GPS?",
        part_overlay_enabled=True,
        part_label_position="custom-drag",
        part_label_x_percent=50,
        part_label_y_percent=4,
        subtitle_ass_path=Path("part_1.ass"),
        textfile_dir=tmp_path,
        show_youtube_credit=True,
        youtube_credit_text="The Infographics Show",
        youtube_credit_position="below_frame",
    )

    assert (tmp_path / "caption_part_1.txt").read_text(encoding="utf-8") == "What is jamming Europe's GPS?"
    assert (tmp_path / "part_label_1.txt").read_text(encoding="utf-8") == "Part 1"
    assert (tmp_path / "yt_credit_part_1.txt").read_text(encoding="utf-8") == "YT: The Infographics Show"
    assert "drawtext=textfile='caption_part_1.txt'" in vf
    assert "drawtext=textfile='part_label_1.txt'" in vf
    assert "drawtext=textfile='yt_credit_part_1.txt'" in vf
    assert "x=(w-text_w)/2:y=(h-text_h)*4/100" in vf
    assert "drawtext=text='" not in vf
    assert "Europe's GPS" not in vf
    assert "Part 1" not in vf
    assert vf.endswith("subtitles=filename='part_1.ass':force_style='Encoding=UTF-8'")
