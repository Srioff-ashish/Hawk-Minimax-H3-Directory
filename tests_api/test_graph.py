"""API-format graphs match the node schemas. Pure Python."""

from __future__ import annotations

import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_workflows  # noqa: E402
from hawk_api.config import ModelSettings  # noqa: E402
from hawk_api.graph import GraphError, Ref, RenderParams, plan_graph, render_graph  # noqa: E402
from hawk_api.loras import ResolvedLora  # noqa: E402

SOCKETS = {
    "HawkH3ModelLoader": set(),
    "HawkH3LoraStack": {"pipe"},
    "HawkH3References": {"refs_in"},
    "HawkH3StoryPlanner": {"refs"},
    "HawkH3Director": {"pipe", "refs"},
}
AUTOGROW = re.compile(r"^(pictures\.picture_[0-8]|poses\.pose_[0-8]|videos\.video_[0-2]|video_soundtracks\.video_soundtrack_[0-2]|audios\.audio_[0-2])$")
CORE_INPUTS = {"LoadImage": {"image"}, "LoadAudio": {"audio"}, "LoadVideo": {"file"}, "GetVideoComponents": {"video"}, "PreviewAny": {"source"}}
OUTPUT_COUNTS = {
    "LoadImage": 2, "LoadAudio": 1, "LoadVideo": 1, "GetVideoComponents": 5, "PreviewAny": 1,
    "HawkH3ModelLoader": 5, "HawkH3LoraStack": 3, "HawkH3References": 2, "HawkH3StoryPlanner": 2, "HawkH3Director": 5,
}

REFS = [
    Ref("face", "image", "hawk_api/face/face.png", "picture", "her face"),
    Ref("coat", "image", "hawk_api/coat/coat.png", "picture"),
    Ref("wave", "image", "hawk_api/wave/wave.png", "pose", "waving"),
    Ref("clip", "video", "hawk_api/clip/clip.mp4", "video", "dance moves"),
    Ref("clip", "video", "hawk_api/clip/clip.mp4", "audio", "her voice"),
    Ref("music", "audio", "hawk_api/music/music.mp3", "video_soundtrack", for_video=1),
]
PLANNER = dict(story="A story", segment_count=3, segment_seconds=8.0, aspect_ratio="16:9", model="xai/grok-4.3", seed=5)
LORAS = [ResolvedLora(f"l{i}", f"lora_{i}.safetensors", 0.5 + i / 10, i == 0, "request") for i in range(6)]


def params():
    return RenderParams(run_name="api_test", seed=123, steps=8)


def by_class(prompt, class_type):
    return [(node_id, node) for node_id, node in prompt.items() if node["class_type"] == class_type]


class Schema(unittest.TestCase):
    def assert_valid(self, prompt):
        for node_id, node in prompt.items():
            cls, inputs = node["class_type"], node["inputs"]
            with self.subTest(node=node_id, cls=cls):
                if cls in SOCKETS:
                    widgets = [w for w in build_workflows.widget_order(cls) if w != "control_after_generate"]
                    non_links = {k for k in inputs if k not in SOCKETS[cls] and not AUTOGROW.match(k)}
                    self.assertEqual(non_links, set(widgets), f"{cls} widget keys differ from its schema")
                else:
                    self.assertTrue(set(inputs) <= CORE_INPUTS[cls] and set(inputs), f"{cls} inputs {set(inputs)}")
                for key, value in inputs.items():
                    if isinstance(value, list):
                        self.assertEqual(len(value), 2)
                        source, index = value
                        self.assertIn(source, prompt, f"{key} links to missing node {source}")
                        self.assertLess(index, OUTPUT_COUNTS[prompt[source]["class_type"]])

    def test_plan_graph(self):
        built, wiring = plan_graph(REFS, PLANNER)
        prompt = built.prompt
        self.assert_valid(prompt)
        self.assertEqual(wiring.available, {"Picture": 2, "Pose": 1, "Video": 1, "Audio": 1})
        self.assertEqual(wiring.video_has_audio, [True])
        ((refs_id, refs),) = by_class(prompt, "HawkH3References")
        inputs = refs["inputs"]
        self.assertEqual(sorted(k for k in inputs if AUTOGROW.match(k)),
                         ["audios.audio_0", "pictures.picture_0", "pictures.picture_1", "poses.pose_0",
                          "video_soundtracks.video_soundtrack_0", "videos.video_0"])
        (parts_id, _), = by_class(prompt, "GetVideoComponents")
        self.assertEqual(len(by_class(prompt, "LoadVideo")), 1, "one video asset loads once")
        self.assertEqual(inputs["videos.video_0"], [parts_id, 0])
        self.assertEqual(inputs["audios.audio_0"], [parts_id, 1])
        self.assertEqual(inputs["video_fps"], [parts_id, 2])
        self.assertEqual(inputs["labels"], "Picture 1: her face\nPose 1: waving\nVideo 1: dance moves\nAudio 1: her voice")
        planner = prompt[built.nodes["planner"]]["inputs"]
        self.assertEqual((planner["refs"], planner["api_key"]), ([refs_id, 0], ""))
        self.assertEqual(prompt[built.nodes["plan_preview"]]["inputs"]["source"], [built.nodes["planner"], 0])

    def test_render_with_script_chains_loras(self):
        built, _ = render_graph(REFS[:2], ModelSettings(), LORAS, params(), script="A scene")
        prompt = built.prompt
        self.assert_valid(prompt)
        stacks = built.nodes["lora_stacks"]
        self.assertEqual(len(stacks), 2)
        first, second = (prompt[s]["inputs"] for s in stacks)
        self.assertEqual([first[f"lora_{i}"] for i in range(1, 5)], [f"lora_{i}.safetensors" for i in range(4)])
        self.assertEqual([second["lora_1"], second["lora_2"], second["lora_3"]], ["lora_4.safetensors", "lora_5.safetensors", "None"])
        self.assertEqual(second["pipe"], [stacks[0], 0])
        director = prompt[built.nodes["director"]]["inputs"]
        self.assertEqual((director["pipe"], director["script"], director["seed"], director["steps"]), ([stacks[1], 0], "A scene", 123, 8))
        self.assertEqual(prompt[by_class(prompt, "HawkH3ModelLoader")[0][0]]["inputs"]["lora_stack"], "")

    def test_one_call_links_planner_into_director(self):
        built, _ = render_graph(REFS[:1], ModelSettings(), [], params(), planner=PLANNER)
        prompt = built.prompt
        self.assert_valid(prompt)
        director = prompt[built.nodes["director"]]["inputs"]
        self.assertEqual(director["script"], [built.nodes["planner"], 0])
        self.assertEqual(built.nodes["lora_stacks"], [])
        loader_id = by_class(prompt, "HawkH3ModelLoader")[0][0]
        self.assertEqual(director["pipe"], [loader_id, 0])

    def test_no_references(self):
        built, wiring = render_graph([], ModelSettings(), [], params(), script="x")
        self.assertNotIn("refs", built.prompt[built.nodes["director"]]["inputs"])
        self.assertEqual(wiring.available["Picture"], 0)

    def test_errors(self):
        with self.assertRaisesRegex(GraphError, "needs image"):
            plan_graph([Ref("a", "audio", "a.mp3", "picture")], PLANNER)
        with self.assertRaisesRegex(GraphError, "for_video"):
            plan_graph([Ref("a", "audio", "a.mp3", "video_soundtrack")], PLANNER)
        with self.assertRaisesRegex(GraphError, "At most 9"):
            plan_graph([Ref(str(i), "image", f"{i}.png", "picture") for i in range(10)], PLANNER)
        with self.assertRaisesRegex(GraphError, "exactly one"):
            render_graph([], ModelSettings(), [], params())


if __name__ == "__main__":
    unittest.main()
