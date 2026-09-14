"""The example workflows are loadable: consistent links, and every Hawk node's
widgets_values in the order its current schema declares.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import build_workflows  # noqa: E402
from hawk_h3.script import build_jobs, parse_script  # noqa: E402

WORKFLOW_DIR = os.path.join(ROOT, "example_workflows")


def load_all():
    for name in sorted(os.listdir(WORKFLOW_DIR)):
        if name.endswith(".json"):
            with open(os.path.join(WORKFLOW_DIR, name), encoding="utf-8") as handle:
                yield name, json.load(handle)


class ExampleWorkflows(unittest.TestCase):
    def test_present(self):
        self.assertEqual(len(list(load_all())), len(build_workflows.WORKFLOWS))

    def test_up_to_date_with_generator(self):
        for build in build_workflows.WORKFLOWS:
            graph = build()
            with open(os.path.join(WORKFLOW_DIR, f"{graph.name}.json"), encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), json.loads(json.dumps(graph.to_json())),
                                 f"{graph.name}.json is stale; run tools/build_workflows.py")

    def test_links_and_widgets(self):
        for name, workflow in load_all():
            with self.subTest(workflow=name):
                build_workflows.validate(workflow, name)
                for node in workflow["nodes"]:
                    if node["type"] in build_workflows.NODE_FILES:
                        order = build_workflows.widget_order(node["type"])
                        self.assertEqual(list(node["widgets_values_named"]), order)
                        self.assertEqual(len(node["widgets_values"]), len(order))

    def test_scripts_are_valid_for_their_references(self):
        for name, workflow in load_all():
            nodes = {node["id"]: node for node in workflow["nodes"]}
            for node in workflow["nodes"]:
                if node["type"] != "HawkH3Director" or not node["widgets_values_named"]["script"]:
                    continue
                refs = next(
                    (nodes[link[1]] for link in workflow["links"] if link[3] == node["id"] and link[5] == "HAWK_H3_REFS"),
                    None,
                )
                counts = {"Picture": 0, "Video": 0, "Audio": 0}
                if refs is not None:
                    for entry in refs["inputs"]:
                        if entry["link"] is None:
                            continue
                        for prefix, kind in (("pictures.", "Picture"), ("videos.", "Video"), ("audios.", "Audio")):
                            if entry["name"].startswith(prefix):
                                counts[kind] += 1
                with self.subTest(workflow=name):
                    build_jobs(
                        parse_script(node["widgets_values_named"]["script"]),
                        available=counts,
                        video_has_audio=[False] * counts["Video"],
                        default_seconds=10,
                        continuity="tail_22",
                        base_seed=0,
                    )


if __name__ == "__main__":
    unittest.main()
