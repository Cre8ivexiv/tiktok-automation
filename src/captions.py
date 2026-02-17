from __future__ import annotations


FIXED_HASHTAGS = [
    "#fyp",
    "#animevideos",
    "#anime",
    "#movie",
    "#manhwa",
    "#animerecap",
    "#animeedits",
    "#isekaianime",
    "#animeedit",
    "#recap",
    "#animerecommendations",
]


def build_caption(title: str, part_number: int) -> str:
    clean_title = title.strip() or "Untitled"
    tags = [f"#part{part_number}"] + FIXED_HASHTAGS
    return f"{clean_title} (Part {part_number}) " + " ".join(tags)

