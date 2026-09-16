"""Script parsing, tag renumbering, job resolution and LoRA stack parsing.

Pure Python -- runs without ComfyUI or torch:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_h3.lora_stack import (  # noqa: E402
    LORA_SLOTS,
    entries_from_slots,
    modality,
    parse_lora_stack,
    slot_widget_names,
)
from hawk_h3.script import (  # noqa: E402
    ScriptError,
    build_jobs,
    frames_for_seconds,
    job_key,
    parse_script,
)

REFS = {"Picture": 3, "Video": 1, "Audio": 2}


def jobs_for(text, available=REFS, video_has_audio=(False,), **overrides):
    options = dict(default_seconds=8.0, continuity="tail_22", base_seed=100)
    options.update(overrides)
    return build_jobs(
        parse_script(text), available=available, video_has_audio=list(video_has_audio), **options
    )


import json  # noqa: E402

from hawk_h3.script import (  # noqa: E402
    ScriptError,
    build_jobs,
    drop_unavailable_references,
    parse_script,
    reference_counts_line,
)


class FrameGrid(unittest.TestCase):
    def test_snaps_to_17k_plus_5(self):
        self.assertEqual(frames_for_seconds(5), 124)
        self.assertEqual(frames_for_seconds(15), 362)
        self.assertEqual(frames_for_seconds(1), 39)
        self.assertEqual(frames_for_seconds(0.01), 5)
        for seconds in (0.5, 2, 7.3, 10, 12.5):
            self.assertEqual(frames_for_seconds(seconds) % 17, 5)


class TextScripts(unittest.TestCase):
    def test_blocks_headers_and_style(self):
        script = parse_script(
            "style: Warm film look.\nNo subtitles.\n"
            "---\n"
            "title: One\nduration: 6s\npictures: 1, 3\nFirst prompt\nsecond line\n"
            "---\n"
            "continuity: hard cut\nseed: 7\nSecond prompt"
        )
        self.assertEqual(script.style, "Warm film look.\nNo subtitles.")
        self.assertEqual(len(script.segments), 2)
        first, second = script.segments
        self.assertEqual((first.title, first.duration, first.pictures), ("One", 6.0, [1, 3]))
        self.assertEqual(first.prompt, "First prompt\nsecond line")
        self.assertIsNone(first.videos)
        self.assertEqual((second.continuity, second.seed), ("off", 7))

    def test_single_plain_prompt(self):
        script = parse_script("A cat walks across a sunny kitchen.")
        self.assertEqual(len(script.segments), 1)
        self.assertEqual(script.style, "")

    def test_empty_is_an_error(self):
        with self.assertRaises(ScriptError):
            parse_script("   ")

    def test_none_and_all(self):
        script = parse_script("pictures: none\naudios: all\nPrompt")
        self.assertEqual(script.segments[0].pictures, [])
        self.assertIsNone(script.segments[0].audios)

    def test_bad_values(self):
        for text in ("duration: soon\nx", "pictures: 12\nx", "continuity: wobble\nx", "seed: -1\nx"):
            with self.subTest(text=text), self.assertRaises(ScriptError):
                parse_script(text)


class JsonScripts(unittest.TestCase):
    def test_planner_shape_with_fences_and_chatter(self):
        reply = (
            'Here is your plan:\n{"title": "T", "style": "S", "segments": ['
            '{"title": "A", "duration": 9, "pictures": [2], "videos": [], "audios": [1], '
            '"continuity": "inherit", "prompt": "<Picture 2> walks."}]}\nEnjoy!'
        )
        script = parse_script(reply)
        segment = script.segments[0]
        self.assertEqual(script.style, "S")
        self.assertEqual((segment.pictures, segment.videos, segment.audios), ([2], [], [1]))
        self.assertIsNone(segment.continuity)

        fenced = parse_script('```json\n{"segments": ["just a prompt"]}\n```')
        self.assertEqual(fenced.segments[0].prompt, "just a prompt")

    def test_round_trip(self):
        script = parse_script("style: S\n---\npictures: 1\nduration: 5\nHello <Picture 1>")
        again = parse_script(script.to_json())
        self.assertEqual(again, script)

    def test_braces_in_plain_text_stay_text(self):
        script = parse_script("She writes {curly} notes on the board.")
        self.assertEqual(script.segments[0].prompt, "She writes {curly} notes on the board.")

    def test_broken_json_is_an_error(self):
        with self.assertRaises(ScriptError):
            parse_script('{"segments": [}')


class Tags(unittest.TestCase):
    def test_normalises_variants(self):
        (job,) = jobs_for("Image 1 and @image2 wear <image_3>; video 1 shows the walk; audio 2 is her voice.")
        self.assertEqual(
            job.prompt,
            "<Picture 1> and <Picture 2> wear <Picture 3>; <Video 1> shows the walk; <Audio 2> is her voice.",
        )

    def test_renumbers_subsets(self):
        (job,) = jobs_for("pictures: 1,3\naudios: 2\n<Picture 3> beside <Picture 1>, voice <Audio 2>")
        self.assertEqual(job.pictures, [1, 3])
        self.assertEqual(job.prompt, "<Picture 2> beside <Picture 1>, voice <Audio 1>")

    def test_video_soundtrack_shifts_audio_numbers(self):
        (job,) = jobs_for("<Video 1> moves; <Audio 1> speaks", video_has_audio=(True,))
        self.assertEqual(job.prompt, "<Video 1> moves; <Audio 2> speaks")
        (job,) = jobs_for("videos: none\n<Audio 1> speaks", video_has_audio=(True,))
        self.assertEqual(job.prompt, "<Audio 1> speaks")

    def test_style_is_prepended_and_remapped(self):
        (job,) = jobs_for("style: Keep <Picture 3> lighting\n---\npictures: 3\nGo")
        self.assertEqual(job.prompt, "Keep <Picture 1> lighting\n\nGo")

    def test_mentioning_an_unselected_reference_fails(self):
        with self.assertRaisesRegex(ScriptError, "leaves it out"):
            jobs_for("pictures: 1\n<Picture 2> smiles")

    def test_mentioning_a_missing_reference_fails(self):
        with self.assertRaisesRegex(ScriptError, "only 3"):
            jobs_for("<Picture 4> smiles")
        with self.assertRaisesRegex(ScriptError, "only 3"):
            jobs_for("pictures: 5\nsmiles")


class Poses(unittest.TestCase):
    REFS = {"Picture": 2, "Pose": 3, "Video": 0, "Audio": 0}

    def jobs(self, text, **overrides):
        return jobs_for(text, available=self.REFS, video_has_audio=(), **overrides)

    def test_mentioned_poses_become_pictures_after_the_pictures(self):
        (job,) = self.jobs("pictures: 1\n<Picture 1> ends in <Pose 3>, then pose 2")
        self.assertEqual((job.pictures, job.poses), ([1], [2, 3]))
        body, instruction = job.prompt.split("\n\n")
        self.assertEqual(body, "<Picture 1> ends in <Picture 3>, then <Picture 2>")
        self.assertIn("Pose reference <Picture 2> and <Picture 3>", instruction)

    def test_unmentioned_poses_are_not_sent(self):
        (job,) = self.jobs("A scene with no pose")
        self.assertEqual(job.poses, [])
        self.assertNotIn("Pose reference", job.prompt)

    def test_explicit_list_and_custom_or_empty_instruction(self):
        (job,) = self.jobs("poses: 1\nShe dances", pose_instruction="POSE ONLY {tags}")
        self.assertEqual(job.poses, [1])
        self.assertTrue(job.prompt.endswith("POSE ONLY <Picture 3>"))
        (job,) = self.jobs("poses: 1\nShe dances", pose_instruction="")
        self.assertEqual(job.prompt, "She dances")

    def test_errors(self):
        with self.assertRaisesRegex(ScriptError, "only 3 pose"):
            self.jobs("Ends in <Pose 4>")
        with self.assertRaisesRegex(ScriptError, "leaves it out"):
            self.jobs("poses: 1\nEnds in <Pose 2>")
        with self.assertRaisesRegex(ScriptError, "at most 9 images"):
            jobs_for("poses: 1,2\nGo", available={"Picture": 8, "Pose": 2, "Video": 0, "Audio": 0}, video_has_audio=())

    def test_json_poses_round_trip(self):
        script = parse_script('{"segments": [{"prompt": "<Pose 1>", "poses": [1]}]}')
        self.assertEqual(script.segments[0].poses, [1])
        self.assertEqual(parse_script(script.to_json()), script)


class Jobs(unittest.TestCase):
    def test_durations_seeds_and_continuity(self):
        jobs = jobs_for("duration: 5\nA\n---\nB\n---\nseed: 9\ncontinuity: last frame\nC")
        self.assertEqual([j.frames for j in jobs], [124, 192, 192])
        self.assertEqual([j.seed for j in jobs], [100, 101, 9])
        self.assertEqual([j.tail_frames for j in jobs], [0, 22, 1])

    def test_same_seed_mode(self):
        jobs = jobs_for("A\n---\nB", seed_mode="same")
        self.assertEqual([j.seed for j in jobs], [100, 100])

    def test_tail_is_downgraded_for_short_segments(self):
        jobs = jobs_for("A\n---\nduration: 0.2\nB", continuity="tail_39")
        self.assertEqual(jobs[1].frames, 5)
        self.assertEqual(jobs[1].tail_frames, 1)
        self.assertTrue(jobs[1].warnings)

    def test_long_segments_are_clamped(self):
        (job,) = jobs_for("duration: 30\nA")
        self.assertEqual(job.frames, 362)
        self.assertTrue(job.warnings)

    def test_no_references(self):
        (job,) = jobs_for("A plain scene", available={"Picture": 0, "Video": 0, "Audio": 0}, video_has_audio=())
        self.assertEqual((job.pictures, job.videos, job.audios), ([], [], []))

    def test_keys_chain(self):
        a1, b1 = jobs_for("A\n---\nB")
        a2, b2 = jobs_for("A changed\n---\nB")
        key_a1 = job_key(a1, {"x": 1})
        key_a2 = job_key(a2, {"x": 1})
        self.assertNotEqual(key_a1, key_a2)
        # Segment B is identical, but its predecessor changed, so it must re-render.
        self.assertNotEqual(job_key(b1, {"x": 1}, key_a1), job_key(b2, {"x": 1}, key_a2))
        self.assertEqual(job_key(b1, {"x": 1}, key_a1), job_key(b1, {"x": 1}, key_a1))


class PlannerTolerance(unittest.TestCase):
    AVAILABLE = {"Picture": 4, "Pose": 0, "Video": 0, "Audio": 0}

    def build(self, script):
        return build_jobs(script, available=self.AVAILABLE, video_has_audio=[], default_seconds=5,
                          continuity="tail_22", base_seed=0)

    def test_drops_numbers_the_prompt_never_mentions(self):
        # The live failure: the LLM copied picture 4 into poses with no poses connected.
        script = parse_script(json.dumps({"segments": [
            {"title": "Pose", "prompt": "<Picture 1> ends in the pose from <Picture 4>.", "pictures": [1, 4], "poses": [4]},
            {"prompt": "She waves.", "pictures": [1], "poses": []},
        ]}))
        fixed, warnings = drop_unavailable_references(script, self.AVAILABLE)
        self.assertEqual((fixed.segments[0].pictures, fixed.segments[0].poses), ([1, 4], []))
        self.assertIs(fixed.segments[1], script.segments[1])
        self.assertEqual(len(warnings), 1)
        self.assertIn("Segment 1 (Pose): removed pose [4] from its poses list; only 0 pose(s)", warnings[0])
        self.assertEqual(self.build(fixed)[0].pictures, [1, 4])

    def test_mentioned_numbers_are_kept_and_still_fail(self):
        script = parse_script('{"segments": [{"prompt": "She ends in <Pose 2>.", "poses": [2]}]}')
        fixed, warnings = drop_unavailable_references(script, self.AVAILABLE)
        self.assertEqual((fixed.segments[0].poses, warnings), ([2], []))
        with self.assertRaisesRegex(ScriptError, "only 0 pose"):
            self.build(fixed)

    def test_warnings_round_trip_and_counts_line(self):
        script = parse_script('{"segments": [{"prompt": "x"}]}')
        text = script.to_json(warnings=["fixed something"])
        self.assertEqual(json.loads(text)["warnings"], ["fixed something"])
        self.assertEqual(parse_script(text).segments[0].prompt, "x")
        self.assertNotIn("warnings", json.loads(script.to_json()))
        self.assertIn("3 picture(s), 0 pose(s), 0 video(s), 0 audio(s)", reference_counts_line({"Picture": 3}))


class StructuredPrompts(unittest.TestCase):
    """H3's native section format: style and the pose note must not break the layout."""

    R2V = (
        "subject_definitions:\n<Subject 1> is the woman whose face comes from <Picture 1>.\n\n"
        "summary:\nreference generation <Subject 1> walks to the window.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - face and hair.\n\n"
        "detailed_description:\n[Shot 1] She walks to the window and ends in the pose from <Pose 1>.\n\n"
        "overall_soundscape: Wordless footsteps on wood. No speech, no voices.\n\n"
        "non_diegetic_music: N/A"
    )

    def build(self, script, available):
        return build_jobs(script, available=available, video_has_audio=[], default_seconds=5,
                          continuity="tail_22", base_seed=0)

    def test_style_goes_inside_the_description_and_pose_note_before_sound(self):
        script = parse_script(json.dumps({"style": "Cinematic live-action, 35mm.", "segments": [{"prompt": self.R2V}]}))
        prompt = self.build(script, {"Picture": 1, "Pose": 1})[0].prompt
        self.assertTrue(prompt.startswith("subject_definitions:\n<Subject 1>"))
        self.assertIn("detailed_description:\nCinematic live-action, 35mm. [Shot 1] She walks", prompt)
        body, sound = prompt.split("overall_soundscape:")
        self.assertIn("ends in the pose from <Picture 2>. Pose reference <Picture 2>: take only the body pose", body)
        self.assertTrue(sound.strip().startswith("Wordless footsteps"))
        self.assertTrue(prompt.rstrip().endswith("non_diegetic_music: N/A"))

    def test_t2va_single_line_field(self):
        text = "integrated_multimodal_description: [Shot 1] Rain on a window.\n\noverall_soundscape: Rain only.\n\nnon_diegetic_music: N/A"
        script = parse_script(json.dumps({"style": "Moody night.", "segments": [{"prompt": text}]}))
        prompt = self.build(script, {})[0].prompt
        self.assertTrue(prompt.startswith("integrated_multimodal_description: Moody night. [Shot 1] Rain"))

    def test_plain_prompts_unchanged(self):
        script = parse_script("style: Warm light.\n---\n<Picture 1> smiles at <Pose 1>.")
        prompt = self.build(script, {"Picture": 1, "Pose": 1})[0].prompt
        self.assertTrue(prompt.startswith("Warm light.\n\n<Picture 1> smiles at <Picture 2>."))
        self.assertIn("\n\nPose reference <Picture 2>:", prompt)

    def test_planner_prompt_teaches_the_h3_format(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hawk_h3", "prompts", "planner_system.md")
        text = open(path, encoding="utf-8").read()
        for needle in ("subject_definitions:", "overall_soundscape:", "non_diegetic_music:", "<d>[English]", "Anti-filler checklist"):
            self.assertIn(needle, text)


class LoraStack(unittest.TestCase):
    def test_lines(self):
        entries = parse_lora_stack(
            "turbo.safetensors : 1\n# off.safetensors : 1\nstyle/look.safetensors : 0.7 : v=1 a=0.5 t=0\nplain.safetensors"
        )
        self.assertEqual([e.name for e in entries], ["turbo.safetensors", "style/look.safetensors", "plain.safetensors"])
        self.assertEqual((entries[1].strength, entries[1].video, entries[1].audio, entries[1].text), (0.7, 1.0, 0.5, 0.0))
        self.assertEqual(entries[2].strength, 1.0)

    def test_plaguekind_json(self):
        entries = parse_lora_stack(
            '[{"on":true,"lora":"a.safetensors","str":1,"v":1,"a":1,"t":0},'
            '{"on":false,"lora":"b.safetensors","str":0.7,"v":1,"a":1,"t":0}]'
        )
        self.assertEqual([(e.name, e.text) for e in entries], [("a.safetensors", 0.0)])

    def test_errors(self):
        for text in ("not a lora", "a.safetensors : loud", "a.safetensors : 1 : q=2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_lora_stack(text)

    def test_slot_widgets(self):
        required, optional = slot_widget_names()
        self.assertEqual(required[:4], ["lora_1", "strength_1", "lora_2", "strength_2"])
        self.assertEqual(optional[:3], ["video_1", "audio_1", "text_1"])
        self.assertEqual((len(required), len(optional)), (2 * LORA_SLOTS, 3 * LORA_SLOTS))

    def test_entries_from_slots(self):
        entries = entries_from_slots(
            {
                "lora_1": "turbo.safetensors", "strength_1": 1.0,
                "lora_2": "None", "strength_2": 1.0,
                "lora_3": "off.safetensors", "strength_3": 0.0,
                "lora_4": "style.safetensors", "strength_4": 0.6, "audio_4": 0.0,
            }
        )
        self.assertEqual([(e.name, e.strength, e.audio) for e in entries],
                         [("turbo.safetensors", 1.0, 1.0), ("style.safetensors", 0.6, 0.0)])
        self.assertEqual(entries_from_slots({}), [])

    def test_modality(self):
        self.assertEqual(modality("diffusion_model.video_patch_proj.lora_up.weight"), "video")
        self.assertEqual(modality("diffusion_model.final_layer.audio_out.lora_down.weight"), "audio")
        self.assertEqual(modality("diffusion_model.token_refiner.0.lora_up.weight"), "text")
        self.assertEqual(modality("diffusion_model.blocks.3.attn.qkv_proj.lora_up.weight"), "joint")


if __name__ == "__main__":
    unittest.main()
