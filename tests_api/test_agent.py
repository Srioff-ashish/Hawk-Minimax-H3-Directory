"""Planner model options and the autonomous agent, end to end against a fake ComfyUI
and a fake Atlas with scripted model replies.

    python -m unittest tests_api.test_agent
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests_api"))

try:
    import httpx
    import uvicorn
    from aiohttp import web

    from hawk_api import agent as agent_module
    from hawk_api.agent import AgentService, AgentStore, parse_reply
    from hawk_api.app import create_app
    from hawk_api.config import Settings
    from test_gateway import PNG, TOKEN, FakeComfy, free_port
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"agent test dependencies missing: {exc}")

MODELS = [
    {"id": "xai/grok-4.3", "name": "Grok 4.3", "input_modalities": ["text", "image"], "output_modalities": ["text"],
     "context_length": 1000000, "pricing": {"prompt": "0.00000125", "completion": "0.0000025"}},
    {"id": "openai/gpt-image-2", "name": "GPT Image 2", "input_modalities": ["text"], "output_modalities": ["image"], "pricing": {}},
    {"id": "anthropic/claude-opus-4.8-coding", "name": "Opus coding", "input_modalities": ["text"], "output_modalities": ["text"], "pricing": {}},
    {"id": "xai/grok-4.6", "name": "Grok 4.6", "input_modalities": ["text", "image"], "output_modalities": ["text"],
     "context_length": 500000, "pricing": {"prompt": "0.000002", "completion": "0.000006"}},
]


class FakeAtlas:
    def __init__(self):
        self.requests: list[dict] = []
        self.reply = lambda body: json.dumps({"say": "hi", "actions": [], "done": True})
        self.delay = 0.0
        self.app = web.Application()
        self.image_requests: list[dict] = []
        self.polls = 0
        self.app.add_routes([web.get("/v1/models", self.models), web.post("/v1/chat/completions", self.chat),
                             web.post("/api/v1/model/generateImage", self.generate), web.get("/api/v1/model/prediction/{pid}", self.prediction)])

    async def models(self, _request):
        return web.json_response({"data": MODELS})

    async def generate(self, request):
        self.image_requests.append(await request.json())
        return web.json_response({"code": 200, "data": {"id": f"pred{len(self.image_requests)}"}})

    async def prediction(self, request):
        self.polls += 1
        payload = self.image_requests[-1]
        if self.polls % 2:  # one "processing" poll before each result
            return web.json_response({"data": {"status": "processing"}})
        outputs = ["data:image/png;base64," + base64.b64encode(tiny_png()).decode()] * payload.get("n", 1)
        return web.json_response({"data": {"status": "completed", "outputs": outputs}})

    async def chat(self, request):
        body = await request.json()
        self.requests.append(body)
        if self.delay:
            await asyncio.sleep(self.delay)
        text = self.reply(body)
        return web.json_response({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})


def tiny_png(color=(200, 80, 40), size=(96, 64)) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return PNG
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def tool_result(body: dict, tool: str) -> dict:
    for message in reversed(body["messages"]):
        match = re.search(rf"TOOL RESULT {tool}: (.*?)(?:\n\nTOOL RESULT|\Z)", message["content"], re.S)
        if message["role"] == "user" and match:
            return json.loads(match.group(1))
    raise AssertionError(f"no {tool} result in the conversation")


def assistant_turns(body: dict) -> int:
    return sum(1 for message in body["messages"] if message["role"] == "assistant")


class AgentApi(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = {name: getattr(agent_module, name) for name in ("WAIT_POLL_SECONDS", "SUMMARY_TOKEN_LIMIT", "RECENT_MESSAGES", "MAX_STEPS")}
        agent_module.WAIT_POLL_SECONDS = 0.05
        self.fake, self.atlas = FakeComfy(), FakeAtlas()
        self.runners = []
        comfy_port, atlas_port, api_port = free_port(), free_port(), free_port()
        for app, port in ((self.fake.app, comfy_port), (self.atlas.app, atlas_port)):
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            self.runners.append(runner)
        self.base = f"http://127.0.0.1:{api_port}"
        settings = Settings(
            token=TOKEN, comfy_url=f"http://127.0.0.1:{comfy_port}", public_base_url=self.base,
            data_dir=tempfile.mkdtemp(prefix="hawk_agent_test_"), lora_cache_seconds=0, reconcile_seconds=0.3,
            atlas_url=f"http://127.0.0.1:{atlas_port}/v1", atlas_api_key="test-atlas-key",
        )
        self.server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=api_port, log_level="warning"))
        self.server_task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        self.http = httpx.AsyncClient(base_url=self.base, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        for name, value in self.saved.items():
            setattr(agent_module, name, value)
        await self.http.aclose()
        self.server.should_exit = True
        await self.server_task
        for runner in self.runners:
            await runner.cleanup()

    async def new_chat(self, **body) -> str:
        response = await self.http.post("/v1/agent/sessions", json=body)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    async def settle(self, session_id: str, timeout: float = 15.0) -> dict:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            view = (await self.http.get(f"/v1/agent/sessions/{session_id}")).json()
            if view["session"]["status"] not in ("running", "stopping"):
                return view
            if asyncio.get_event_loop().time() > deadline:
                self.fail(f"agent still running: {view}")
            await asyncio.sleep(0.05)

    async def test_planner_models_and_model_choice(self):
        options = (await self.http.get("/v1/options")).json()
        self.assertEqual([m["id"] for m in options["planner_models"]], ["xai/grok-4.6", "xai/grok-4.3"])
        self.assertTrue(options["planner_models"][0]["vision"])
        self.assertEqual((options["default_planner_model"], options["default_agent_model"]), ("xai/grok-4.3", "xai/grok-4.6"))
        plan = (await self.http.post("/v1/plans", json={"story": "A walk", "model": "xai/grok-4.6"})).json()
        planner = next(n for n in self.fake.prompts[plan["id"]].values() if n["class_type"] == "HawkH3StoryPlanner")
        self.assertEqual(planner["inputs"]["model"], "xai/grok-4.6")
        listed = (await self.http.get("/v1/agent/models")).json()
        self.assertTrue(listed["configured"])

    async def test_autonomous_render_turn(self):
        uploaded = await self.http.post("/v1/assets", files={"files": ("face.png", PNG, "image/png")})
        asset = uploaded.json()["assets"][0]["id"]

        def reply(body):
            turn = assistant_turns(body)
            if turn == 0:
                return json.dumps({"say": "Checking the server.", "actions": [
                    {"tool": "rename_chat", "args": {"title": "Chai ad"}}, {"tool": "list_options", "args": {}}]})
            if turn == 1:
                return json.dumps({"say": "Rendering a preview.", "actions": [{"tool": "render_film", "args": {
                    "references": [{"asset_id": asset, "role": "picture"}],
                    "script": "duration: 5\n<Picture 1> pours chai.", "settings": {"megapixels": 0.4}}}]})
            if turn == 2:
                return json.dumps({"say": "Waiting.", "actions": [{"tool": "wait_for_job", "args": {"job_id": tool_result(body, "render_film")["id"]}}]})
            return json.dumps({"say": f"Done: {tool_result(body, 'wait_for_job')['video_url']}", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat(persona="Bollywood ad-film director", model="xai/grok-4.6")
        sent = await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Make a chai ad preview", "attachments": [asset]})
        self.assertEqual(sent.status_code, 202, sent.text)
        view = await self.settle(chat)

        session, messages = view["session"], view["messages"]
        self.assertEqual((session["status"], session["title"]), ("idle", "Chai ad"))
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "tool", "assistant", "tool", "assistant", "tool", "assistant"])
        render = next(m for m in messages if m["role"] == "tool" and m["content"]["tool"] == "render_film")
        self.assertTrue(render["content"]["ok"] and render["content"]["job_id"])
        self.assertEqual(self.fake.prompts[render["content"]["job_id"]] is not None, True)
        final = messages[-1]["content"]
        self.assertTrue(final["done"] and "/v1/jobs/" in final["say"] and "sig=" in final["say"])
        self.assertEqual(session["usage"]["steps"], 4)
        self.assertGreater(session["usage"]["cost_usd"], 0)

        first = self.atlas.requests[0]
        self.assertEqual(first["model"], "xai/grok-4.6")
        system = first["messages"][0]["content"]
        self.assertIn("Bollywood ad-film director", system)
        for tool in ("render_film", "plan_film", "list_options", "wait_for_job"):
            self.assertIn(f"- {tool}:", system)
        self.assertIn(f"Attached assets: {asset}", first["messages"][-1]["content"])

        later = (await self.http.get(f"/v1/agent/sessions/{chat}", params={"after": messages[-2]["id"]})).json()
        self.assertEqual(len(later["messages"]), 1)

    async def test_generate_and_edit_images(self):
        agent_module.WAIT_POLL_SECONDS = 0.05
        created = (await self.http.post("/v1/images", json={"prompt": "A fit model in her forties, studio portrait", "n": 2, "size": "1536x2048"})).json()
        self.assertEqual(created["model"], "bytedance/seedream-v5.0-pro/text-to-image")
        self.assertEqual(len(created["assets"]), 2)
        first = created["assets"][0]
        self.assertEqual((first["kind"], first["filename"][:4]), ("image", "gen_"))
        self.assertEqual(self.atlas.image_requests[-1], {"model": "bytedance/seedream-v5.0-pro/text-to-image",
                                                          "prompt": "A fit model in her forties, studio portrait", "size": "1536x2048", "n": 2})
        async with httpx.AsyncClient() as browser:  # signed links need no token
            thumb = await browser.get(first["thumb_url"])
            full = await browser.get(first["file_url"])
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(full.content[:4], b"\x89PNG")
        try:
            import PIL  # noqa: F401
            self.assertEqual((thumb.headers["content-type"], thumb.content[:3]), ("image/jpeg", b"\xff\xd8\xff"))
        except ImportError:
            pass
        listed = (await self.http.get("/v1/assets")).json()["assets"]
        self.assertTrue(all("file_url" in a for a in listed) and "thumb_url" in listed[0])

        def reply(body):
            if assistant_turns(body) == 0:
                return json.dumps({"say": "Making her outfit variation.", "actions": [{"tool": "generate_image", "args": {
                    "prompt": "Same woman, red satin dress, white studio", "reference_asset_ids": [first["asset_id"] if "asset_id" in first else first["id"]]}}]})
            result = tool_result(body, "generate_image")
            return json.dumps({"say": f"Here it is: {result['assets'][0]['thumb_url']}", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Put her in a red dress", "attachments": [first["id"]]})
        view = await self.settle(chat)
        tool = next(m for m in view["messages"] if m["role"] == "tool")["content"]
        self.assertTrue(tool["ok"], tool)
        self.assertEqual(tool["result"]["model"], "bytedance/seedream-v5.0-pro/edit")
        edit_payload = self.atlas.image_requests[-1]
        self.assertTrue(edit_payload["images"][0].startswith("data:image/png;base64,"))
        self.assertIn("thumb_url", view["messages"][0]["content"]["attachments"][0])
        self.assertIn("&w=320", view["messages"][-1]["content"]["say"])

        audio = (await self.http.post("/v1/assets", files={"files": ("beat.mp3", b"ID3" + b"0" * 64, "audio/mpeg")})).json()["assets"][0]
        bad = await self.http.post("/v1/images", json={"prompt": "x", "reference_asset_ids": [audio["id"]]})
        self.assertEqual(bad.status_code, 422)
        wrong = await self.http.post("/v1/images", json={"prompt": "x", "model": "bytedance/seedream-v5.0-pro/edit"})
        self.assertEqual(wrong.status_code, 422)

    async def test_invalid_json_is_repaired_then_fails(self):
        replies = iter(["Sure! I will do it.", json.dumps({"say": "Fixed.", "actions": [], "done": True})])
        self.atlas.reply = lambda body: next(replies)
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "hi"})
        view = await self.settle(chat)
        self.assertEqual(view["session"]["status"], "idle")
        self.assertTrue(any(m["role"] == "note" and m["content"].get("kind") == "repair" for m in view["messages"]))
        self.assertEqual(view["messages"][-1]["content"]["say"], "Fixed.")

        self.atlas.reply = lambda body: "still not json"
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "again"})
        view = await self.settle(chat)
        self.assertEqual(view["session"]["status"], "error")

    async def test_stop_and_conflict(self):
        self.atlas.delay = 0.3
        self.atlas.reply = lambda body: json.dumps({"say": "Looking.", "actions": [{"tool": "list_references", "args": {}}]})
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "loop"})
        busy = await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "another"})
        self.assertEqual(busy.status_code, 409)
        await asyncio.sleep(0.5)
        stopped = (await self.http.post(f"/v1/agent/sessions/{chat}/stop")).json()
        self.assertEqual(stopped["status"], "stopping")
        view = await self.settle(chat)
        self.assertEqual(view["session"]["status"], "idle")
        self.assertEqual(view["messages"][-1]["content"]["text"], "Stopped by you.")
        self.assertEqual((await self.http.delete(f"/v1/agent/sessions/{chat}")).status_code, 200)
        self.assertEqual((await self.http.get(f"/v1/agent/sessions/{chat}")).status_code, 404)

    async def test_old_messages_are_summarised(self):
        agent_module.SUMMARY_TOKEN_LIMIT = 1
        agent_module.RECENT_MESSAGES = 2

        def reply(body):
            if "response_format" not in body:
                return "SUMMARY: user wants a chai ad"
            return json.dumps({"say": "ok", "actions": [] if assistant_turns(body) else [{"tool": "list_references", "args": {}}], "done": False})

        self.atlas.reply = reply
        chat = await self.new_chat()
        for text in ("first", "second"):
            await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text})
            view = await self.settle(chat)
        self.assertEqual(view["session"]["summary"], "SUMMARY: user wants a chai ad")
        self.assertGreater(view["session"]["summary_upto"], 0)
        last = [r for r in self.atlas.requests if "response_format" in r][-1]
        self.assertTrue(any("SUMMARY OF THE EARLIER CONVERSATION" in m["content"] for m in last["messages"] if m["role"] == "system"))


class Pieces(unittest.IsolatedAsyncioTestCase):
    def test_parse_reply(self):
        self.assertEqual(parse_reply('```json\n{"say": "a", "actions": [{"tool": "x"}]}\n```'),
                         {"say": "a", "actions": [{"tool": "x", "args": {}}], "done": False})
        self.assertEqual(parse_reply('Here: {"say": "b", "actions": [], "done": true} ok')["done"], True)
        self.assertIsNone(parse_reply("no json here"))
        self.assertIsNone(parse_reply('{"actions": [{"name": "x"}]}'))

    async def test_restart_marks_running_chats_idle(self):
        path = os.path.join(tempfile.mkdtemp(), "jobs.sqlite3")
        service = SimpleNamespace(settings=SimpleNamespace(db_path=path, agent_model="xai/grok-4.6"), atlas=None)
        agents = AgentService(service, mcp=None, store=AgentStore(path))
        session = agents.create_session(persona="x")
        session["status"] = "running"
        agents.store.save_session(session)
        await AgentService(service, mcp=None, store=AgentStore(path)).start()
        view = agents.view(session["id"])
        self.assertEqual(view["session"]["status"], "idle")
        self.assertIn("server restart", view["messages"][-1]["content"]["text"])


if __name__ == "__main__":
    unittest.main()
