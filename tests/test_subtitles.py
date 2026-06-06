from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src import subtitles


class SubtitlePipelineTests(unittest.TestCase):
    def test_slice_words_for_segment_converts_to_local_time(self) -> None:
        words = [{"word": "hello", "start": 592.3, "end": 592.7}]

        sliced = subtitles.slice_words_for_segment(words, segment_start=592.0, segment_end=623.0)

        self.assertEqual(len(sliced), 1)
        self.assertEqual(sliced[0]["word"], "hello")
        self.assertAlmostEqual(sliced[0]["start"], 0.3, places=6)
        self.assertAlmostEqual(sliced[0]["end"], 0.7, places=6)

    def test_slice_words_for_segment_applies_negative_offset(self) -> None:
        words = [{"word": "hello", "start": 592.3, "end": 592.7}]

        sliced = subtitles.slice_words_for_segment(
            words,
            segment_start=592.0,
            segment_end=623.0,
            subtitle_offset_seconds=-0.2,
        )

        self.assertEqual(len(sliced), 1)
        self.assertEqual(sliced[0]["word"], "hello")
        self.assertAlmostEqual(sliced[0]["start"], 0.1, places=6)
        self.assertAlmostEqual(sliced[0]["end"], 0.5, places=6)

    def test_extract_audio_for_transcription_generates_and_reuses_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_video = root / "source clip.mp4"
            input_video.write_bytes(b"video")
            expected_audio = root / "source clip_audio_16k.wav"

            def fake_run(cmd: list[str], **_kwargs: object) -> types.SimpleNamespace:
                Path(cmd[-1]).write_bytes(b"wav")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.object(subtitles, "_resolve_binary_if_available", return_value="ffmpeg.exe"), patch(
                "src.subtitles.subprocess.run",
                side_effect=fake_run,
            ) as mocked_run:
                audio_path = subtitles.extract_audio_for_transcription(input_video, log=lambda _msg: None)

            self.assertEqual(audio_path, expected_audio)
            self.assertTrue(expected_audio.exists())
            self.assertEqual(mocked_run.call_count, 1)

            expected_audio.touch()
            with patch("src.subtitles.subprocess.run", side_effect=AssertionError("ffmpeg should not run")):
                cached_path = subtitles.extract_audio_for_transcription(input_video, log=lambda _msg: None)

            self.assertEqual(cached_path, expected_audio)

    def test_transcribe_video_writes_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_video = root / "source.mp4"
            input_video.write_bytes(b"video")
            audio_path = root / "source_audio_16k.wav"
            audio_path.write_bytes(b"wav")
            logs: list[str] = []
            expected_words = [{"word": "hello", "start": 0.12, "end": 0.45}]

            fake_whisperx = types.SimpleNamespace(
                load_model=lambda *args, **kwargs: types.SimpleNamespace(
                    transcribe=lambda _audio, language=None: {"language": language or "en", "segments": [{"text": "hello"}]}
                ),
                load_audio=lambda _path: "audio-buffer",
                load_align_model=lambda language_code, device: ("align-model", {"language": language_code, "device": device}),
                align=lambda *args, **kwargs: {"segments": [{"words": expected_words}]},
            )
            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False)
            )

            with patch.object(subtitles, "extract_audio_for_transcription", return_value=audio_path), patch.dict(
                sys.modules,
                {"whisperx": fake_whisperx, "torch": fake_torch},
                clear=False,
            ):
                words = subtitles.transcribe_video(
                    input_video=input_video,
                    model_size="medium",
                    use_cache=True,
                    log=logs.append,
                )

            self.assertEqual(words, expected_words)
            cache_path = input_video.parent / "source_transcript.json"
            self.assertTrue(cache_path.exists())
            self.assertEqual(json.loads(cache_path.read_text(encoding="utf-8")), expected_words)
            self.assertTrue(any("Transcript cached to" in line for line in logs))

            with patch.object(
                subtitles,
                "extract_audio_for_transcription",
                side_effect=AssertionError("cache should skip audio extraction"),
            ):
                cached_words = subtitles.transcribe_video(input_video=input_video, use_cache=True, log=logs.append)

            self.assertEqual(cached_words, expected_words)
            self.assertTrue(any("Transcript cache found" in line for line in logs))

    def test_build_ass_subtitles_uses_clip_local_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "part_1.ass"
            subtitles.build_ass_subtitles(
                [{"word": "hello", "start": 0.3, "end": 0.7}],
                output_path,
                style="standard",
                clip_duration=3.0,
            )

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("0:00:00.30", content)
            self.assertNotIn("0:09:52.30", content)
if __name__ == "__main__":
    unittest.main()
