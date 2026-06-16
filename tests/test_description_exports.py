import json
from pathlib import Path

from src.render import RenderedPart, _write_description_exports, rendered_parts_to_dict


def test_write_description_exports_creates_text_and_json(tmp_path: Path) -> None:
    part_rows = [
        {
            "part_number": 1,
            "output_path": str(tmp_path / "part_1.mp4"),
            "title": "What is jamming Europe's GPS?",
            "hashtags": "#fyp #geopolitics #news",
            "upload_description": "What is jamming Europe's GPS? (Part 1) #fyp #geopolitics #news",
        },
        {
            "part_number": 2,
            "output_path": str(tmp_path / "part_2.mp4"),
            "title": "The GPS mystery gets worse",
            "hashtags": "#fyp #geopolitics #news",
            "upload_description": "The GPS mystery gets worse (Part 2) #fyp #geopolitics #news",
        },
    ]

    exports = _write_description_exports(tmp_path, part_rows)

    txt_path = Path(exports["txt"])
    json_path = Path(exports["json"])
    assert txt_path == tmp_path / "descriptions.txt"
    assert json_path == tmp_path / "descriptions.json"
    assert txt_path.read_text(encoding="utf-8") == (
        "Part 1:\n"
        "What is jamming Europe's GPS? (Part 1) #fyp #geopolitics #news\n\n"
        "Part 2:\n"
        "The GPS mystery gets worse (Part 2) #fyp #geopolitics #news\n"
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {
            "part": 1,
            "filename": "part_1.mp4",
            "title": "What is jamming Europe's GPS?",
            "description": "What is jamming Europe's GPS? (Part 1) #fyp #geopolitics #news",
        },
        {
            "part": 2,
            "filename": "part_2.mp4",
            "title": "The GPS mystery gets worse",
            "description": "The GPS mystery gets worse (Part 2) #fyp #geopolitics #news",
        },
    ]


def test_rendered_parts_to_dict_includes_description_fields(tmp_path: Path) -> None:
    description_file = tmp_path / "descriptions.txt"
    part = RenderedPart(
        part_number=1,
        start=0,
        end=10,
        path=tmp_path / "part_1.mp4",
        title="Part title",
        hashtags="#fyp",
        upload_description="Part title (Part 1) #fyp",
        description_file=description_file,
    )

    rows = rendered_parts_to_dict([part])

    assert rows[0]["title"] == "Part title"
    assert rows[0]["hashtags"] == "#fyp"
    assert rows[0]["upload_description"] == "Part title (Part 1) #fyp"
    assert rows[0]["description_file"] == str(description_file)
