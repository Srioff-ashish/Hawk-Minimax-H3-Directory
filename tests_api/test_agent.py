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
     "context_length": 500000, "pricing": {"prompt": "0.000002", "completion": "0.000006", "input_cache_read": "0.0000005"}},
]
DEEPSEEK = {"id": "deepseek-ai/deepseek-v4.1-flash", "name": "DeepSeek V4.1 Flash", "input_modalities": ["text", "image"],
            "output_modalities": ["text"], "context_length": 1048576, "pricing": {"prompt": "0.0000003", "completion": "0.0000012"}}


class FakeAtlas:
    def __init__(self):
        self.requests: list[dict] = []
        self.reply = lambda body: json.dumps({"say": "hi", "actions": [], "done": True})
        self.delay = 0.0
        self.app = web.Application()
        self.image_requests: list[dict] = []
        self.polls = 0
        self.model_list = MODELS
        self.fail_vision: set[str] = set()  # these models answer 400 to calls with images
        self.usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        self.app.add_routes([web.get("/v1/models", self.models), web.post("/v1/chat/completions", self.chat),
                             web.post("/api/v1/model/generateImage", self.generate), web.get("/api/v1/model/prediction/{pid}", self.prediction)])

    async def models(self, _request):
        return web.json_response({"data": self.model_list})

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
        if body["model"] in self.fail_vision and isinstance(body["messages"][0]["content"], list):
            return web.json_response({"error": {"message": "content policy: image rejected"}}, status=400)
        text = self.reply(body)
        return web.json_response({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                  "usage": self.usage})


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


def tool_result_text(content: str, tool: str) -> str:
    match = re.search(rf"TOOL RESULT {tool}: (.*?)(?:\n\nTOOL RESULT|\Z)", content, re.S)
    assert match, f"no {tool} result"
    return match.group(1)


def assistant_turns(body: dict) -> int:
    return sum(1 for message in body["messages"] if message["role"] == "assistant")


class AgentApi(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = {name: getattr(agent_module, name) for name in ("WAIT_POLL_SECONDS", "MAX_STEPS")}
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
        self.app = create_app(settings)
        self.agent = self.app.state.agent
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=api_port, log_level="warning"))
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

    async def test_agent_sees_loras_with_a_large_model_catalogue(self):
        # Atlas lists 100+ models with pricing; the LoRA list must still reach the agent.
        big = [{**MODELS[0], "id": f"vendor/chat-model-{n}", "name": f"Vendor: Chat model {n} with a long descriptive name"} for n in range(150)]
        self.atlas.model_list = MODELS + big
        self.fake.model_files["loras"].append("extra/style-alpha-minimax-h3.safetensors")

        def reply(body):
            turn = assistant_turns(body)
            if turn == 0:
                return json.dumps({"say": "Checking LoRAs.", "actions": [{"tool": "list_options", "args": {}}, {"tool": "list_loras", "args": {}}]})
            return json.dumps({"say": "ok", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Which LoRAs can you use?"})
        await self.settle(chat)
        last = self.atlas.requests[-1]["messages"][-1]["content"]
        for tool in ("list_options", "list_loras"):
            self.assertIn("extra/style-alpha-minimax-h3.safetensors", tool_result_text(last, tool), tool)
        self.assertIn("- list_loras:", self.atlas.requests[0]["messages"][0]["content"])

    async def test_persona_picture_becomes_the_chat_avatar(self):
        agent_module.WAIT_POLL_SECONDS = 0.05

        def reply(body):
            turn = assistant_turns(body)
            if turn == 0:
                return json.dumps({"say": "Ek second, apni photo bana rahi hoon.", "actions": [{"tool": "generate_image", "args": {
                    "prompt": "Portrait of Maya, a warm 45-year-old Indian woman, soft window light"}}]})
            if turn == 1:
                asset = tool_result(body, "generate_image")["assets"][0]["id"]
                return json.dumps({"say": "", "actions": [{"tool": "set_avatar", "args": {"asset_id": asset}}]})
            return json.dumps({"say": "Yeh main hoon.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat(persona="You are Maya, a warm, witty Delhi fashion stylist.")
        other = await self.new_chat(persona="A Bollywood ad-film director")
        listed = {s["id"]: s for s in (await self.http.get("/v1/agent/sessions")).json()["sessions"]}
        self.assertEqual((listed[chat]["persona_name"], listed[chat]["avatar_url"]), ("Maya", None))
        self.assertEqual(listed[other]["persona_name"], "")

        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Apni photo dikhao"})
        view = await self.settle(chat)
        session = view["session"]
        avatar = next(m for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "set_avatar")["content"]
        self.assertTrue(avatar["ok"], avatar)
        generated = next(m for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "generate_image")["content"]
        self.assertEqual(session["avatar_asset_id"], generated["result"]["assets"][0]["id"])
        self.assertEqual(session["persona_name"], "Maya")
        async with httpx.AsyncClient() as browser:
            self.assertEqual((await browser.get(session["avatar_url"])).status_code, 200)

        first_system = self.atlas.requests[0]["messages"][0]["content"]
        self.assertIn("Your name in this chat: Maya.", first_system)
        self.assertIn("You have no avatar yet", first_system)
        self.assertIn("- set_avatar:", first_system)
        last_system = self.atlas.requests[-1]["messages"][0]["content"]
        self.assertIn(f"Your avatar (a picture of you): asset {session['avatar_asset_id']}", last_system)
        self.assertEqual((await self.http.get(f"/v1/agent/sessions/{other}")).json()["session"]["avatar_url"], None, "per chat")

        renamed = (await self.http.patch(f"/v1/agent/sessions/{chat}", json={"name": "Maya Ji"})).json()
        self.assertEqual(renamed["persona_name"], "Maya Ji")
        audio = (await self.http.post("/v1/assets", files={"files": ("beat.mp3", b"ID3" + b"1" * 64, "audio/mpeg")})).json()["assets"][0]["id"]
        self.assertEqual((await self.http.patch(f"/v1/agent/sessions/{chat}", json={"avatar_asset_id": audio})).status_code, 422)
        cleared = (await self.http.patch(f"/v1/agent/sessions/{chat}", json={"avatar_asset_id": ""})).json()
        self.assertEqual((cleared["avatar_url"], cleared["persona_name"]), (None, "Maya Ji"))

    async def test_group_chat_cast_lines_and_avatars(self):
        agent_module.WAIT_POLL_SECONDS = 0.05
        cast = [{"persona": "You are Maya, a warm Delhi stylist."}, {"name": "Riya", "persona": "Maya's sarcastic Mumbai friend."}]
        bad = [
            [{"persona": "A stylist"}, {"name": "Riya", "persona": "x"}],  # first has no name
            [{"name": "Riya", "persona": "a"}, {"name": "riya", "persona": "b"}],  # same name
            [{"name": f"P{n}", "persona": "x"} for n in range(5)],  # too many
        ]
        for members in bad:
            self.assertEqual((await self.http.post("/v1/agent/sessions", json={"cast": members})).status_code, 422, members)

        def reply(body):
            turn = assistant_turns(body)
            if turn == 0:
                chatter = [{"speaker": "Maya" if n % 2 == 0 else "Riya", "say": f"line {n}"} for n in range(10)]
                return json.dumps({"lines": chatter, "actions": [
                    {"tool": "generate_image", "by": "Riya", "args": {"prompt": "Portrait of Riya"}}]})
            if turn == 1:
                asset = tool_result(body, "generate_image")["assets"][0]["id"]
                return json.dumps({"lines": [], "actions": [{"tool": "set_avatar", "by": "Riya", "args": {"asset_id": asset, "speaker": "Riya"}},
                                                           {"tool": "set_persona", "args": {"speaker": "Zoya", "persona": "A shy poet."}}]})
            return json.dumps({"lines": [{"speaker": "Riya", "say": "Yeh main hoon."}], "actions": [], "done": True})

        self.atlas.reply = reply
        response = await self.http.post("/v1/agent/sessions", json={"cast": cast})
        self.assertEqual(response.status_code, 201, response.text)
        chat = response.json()
        self.assertEqual([m["display_name"] for m in chat["cast"]], ["Maya", "Riya"])
        self.assertEqual((chat["persona_name"], chat["persona"]), ("Maya", cast[0]["persona"]))

        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "@Riya apni photo dikhao"})
        view = await self.settle(chat["id"])
        first = next(m for m in view["messages"] if m["role"] == "assistant")["content"]
        self.assertEqual(len(first["lines"]), 8, "capped at 8 lines")
        self.assertEqual(first["actions"][0]["by"], "Riya")
        self.assertTrue(first["say"].startswith("Maya: line 0"))
        members = {m["display_name"]: m for m in view["session"]["cast"]}
        self.assertEqual(list(members), ["Maya", "Riya", "Zoya"], "set_persona with a new speaker adds a character")
        self.assertTrue(members["Riya"]["avatar_url"] and members["Riya"]["avatar_asset_id"])
        self.assertIsNone(members["Maya"]["avatar_url"])

        system = self.atlas.requests[0]["messages"][0]["content"]
        self.assertIn("You voice a cast of 2 characters: Maya, Riya", system)
        self.assertIn("At most 8 lines per reply", system)
        self.assertIn(f"Avatar: asset {members['Riya']['avatar_asset_id']}", self.atlas.requests[-1]["messages"][0]["content"])

        self.atlas.reply = lambda body: json.dumps({"lines": [], "actions": [{"tool": "remove_character", "args": {"speaker": "zoya"}}], "done": False}) \
            if assistant_turns(body) == 3 else json.dumps({"lines": [{"speaker": "Maya", "say": "Bye Zoya"}], "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "Zoya ko hatao"})
        view = await self.settle(chat["id"])
        self.assertEqual([m["display_name"] for m in view["session"]["cast"]], ["Maya", "Riya"])

        edited = (await self.http.patch(f"/v1/agent/sessions/{chat['id']}", json={"persona": "You are Maya, now a film director."})).json()
        self.assertEqual((edited["cast"][0]["persona"], edited["cast"][1]["display_name"]), ("You are Maya, now a film director.", "Riya"))

    async def test_adaptive_personas_grow(self):
        cast = [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}]
        grow = {"lines": [{"speaker": "Maya", "say": "Noted!"}], "actions": [], "done": True,
                "grow": [{"speaker": "Maya", "note": "Calls the user 'boss' now."}, {"speaker": "Riya", "note": "Warming up to Maya."},
                         {"speaker": "Nobody", "note": "ignored"}]}
        self.atlas.reply = lambda body: json.dumps(grow)
        chat = (await self.http.post("/v1/agent/sessions", json={"cast": cast})).json()
        self.assertFalse(chat["adaptive"])
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "hi"})
        view = await self.settle(chat["id"])
        self.assertEqual([m["growth"] for m in view["session"]["cast"]], [[], []], "no growth while adaptive is off")
        self.assertNotIn("ADAPTIVE PERSONA", self.atlas.requests[-1]["messages"][0]["content"])

        chat = (await self.http.patch(f"/v1/agent/sessions/{chat['id']}", json={"adaptive": True})).json()
        self.assertTrue(chat["adaptive"])
        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "call me boss"})
        view = await self.settle(chat["id"])
        self.assertIn("ADAPTIVE PERSONA", self.atlas.requests[-1]["messages"][0]["content"])
        members = {m["display_name"]: m for m in view["session"]["cast"]}
        self.assertEqual(members["Maya"]["growth"], ["Calls the user 'boss' now."])
        self.assertEqual(members["Riya"]["growth"], ["Warming up to Maya."])
        notes = [m["content"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "grow"]
        self.assertEqual([(n["speaker"], n["text"]) for n in notes], [("Maya", "Calls the user 'boss' now."), ("Riya", "Warming up to Maya.")])

        await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "again"})
        view = await self.settle(chat["id"])
        self.assertEqual(view["session"]["cast"][0]["growth"], ["Calls the user 'boss' now."], "repeats are not stored twice")
        self.assertIn("Calls the user 'boss' now.", self.atlas.requests[-1]["messages"][0]["content"])

        # editing a persona keeps growth; growth [] resets it
        edited = view["session"]["cast"]
        kept = (await self.http.patch(f"/v1/agent/sessions/{chat['id']}", json={"cast": [
            {"id": edited[0]["id"], "name": "Maya", "persona": "stylist, now a director"},
            {"id": edited[1]["id"], "name": "Riya", "persona": "director", "growth": []}]})).json()
        self.assertEqual([m["growth"] for m in kept["cast"]], [["Calls the user 'boss' now."], []])

        # at 8 notes the model folds them into a few; a failed or empty merge keeps them
        counter = iter(range(100))

        def merging(body):
            if body["messages"][0]["content"].startswith("You keep a character's memory"):
                self.assertIn("- Riya note 0", body["messages"][1]["content"])
                return json.dumps({"notes": ["Feels sidelined by the user.", "Softening towards Maya.", "", "x", "y", "z"]})
            return json.dumps({"lines": [{"speaker": "Riya", "say": "hm"}], "actions": [], "done": True,
                               "grow": [{"speaker": "Riya", "note": f"Riya note {next(counter)}"}]})
        self.atlas.reply = merging
        for _ in range(8):
            await self.http.post(f"/v1/agent/sessions/{chat['id']}/messages", json={"text": "next"})
            view = await self.settle(chat["id"])
        riya = view["session"]["cast"][1]["growth"]
        self.assertEqual(riya, ["Feels sidelined by the user.", "Softening towards Maya.", "x", "y"], "merged at 8 into at most 4")
        merges = [m for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "grow-merge"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(view["session"]["cast"][0]["growth"], ["Calls the user 'boss' now."], "other characters untouched")
        self.assertIn("not what happened", self.atlas.requests[-2]["messages"][0]["content"])

        solo = await self.new_chat(persona="You are Maya.")
        await self.http.patch(f"/v1/agent/sessions/{solo}", json={"adaptive": True})
        self.atlas.reply = lambda body: json.dumps({"say": "ok", "actions": [], "done": True, "grow": [{"note": "Prefers Hinglish."}]})
        await self.http.post(f"/v1/agent/sessions/{solo}/messages", json={"text": "hinglish please"})
        view = await self.settle(solo)
        self.assertEqual(view["session"]["cast"][0]["growth"], ["Prefers Hinglish."])
        self.assertEqual(view["session"]["persona"], "You are Maya.")
        await self.http.post(f"/v1/agent/sessions/{solo}/messages", json={"text": "next"})
        await self.settle(solo)
        self.assertIn("you have grown in this chat so far", self.atlas.requests[-1]["messages"][0]["content"])

    async def test_let_them_talk(self):
        cast = [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}, {"name": "Zoya", "persona": "poet"}]
        chat = (await self.http.post("/v1/agent/sessions", json={"cast": cast})).json()["id"]
        solo = await self.new_chat(persona="You are Maya.")
        self.assertEqual((await self.http.post(f"/v1/agent/sessions/{solo}/talk", json={"rounds": 3})).status_code, 422)
        self.assertEqual((await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 11})).status_code, 422)
        script = iter([("Maya", "Riya, yeh lehenga dekho!"), ("Riya", "Maroon phir se? Zoya, tum batao."),
                       ("Zoya", "Mujhe toh pasand hai."), ("Maya", "Dekha?"), ("Riya", "Theek hai, jeet gayi tum.")])
        speakers = []

        def reply(body):
            system = body["messages"][0]["content"]
            who = re.match(r"You are (\w+), one of the characters", system).group(1)
            speakers.append(who)
            expected, say = next(script)
            self.assertEqual(who, expected, "the named character speaks next, else whoever waited longest")
            return json.dumps({"say": say, "to": "all", "pause": len(speakers) == 5})

        self.atlas.reply = reply
        started = await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 6})
        self.assertEqual(started.status_code, 202, started.text)
        view = await self.settle(chat)
        self.assertEqual(speakers, ["Maya", "Riya", "Zoya", "Maya", "Riya"], "stops when someone pauses")
        talks = [m["content"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "talk"]
        self.assertEqual([(n["round"], n["of"]) for n in talks], [(1, 6), (2, 6)], "a round is one turn each")
        lines = [m["content"]["lines"][0] for m in view["messages"] if m["role"] == "assistant"]
        self.assertEqual(lines[1], {"speaker": "Riya", "say": "Maroon phir se? Zoya, tum batao.", "to": "all"})
        last = self.atlas.requests[-1]["messages"]
        self.assertNotIn("You are Hawk", last[0]["content"], "a character turn has no director prompt")
        self.assertNotIn("render_film", last[0]["content"], "and no tool catalogue")
        self.assertIn("Maya: Dekha?", last[1]["content"])
        self.assertIn("It's your turn, Riya", last[1]["content"])
        self.assertEqual(view["session"]["status"], "idle")

    async def test_setting_up_a_cast_in_a_fresh_chat(self):
        # What models do when asked to "set them as personas": one set_persona per character with speaker.
        def reply(body):
            if assistant_turns(body) == 0:
                return json.dumps({"say": "", "actions": [
                    {"tool": "set_persona", "args": {"speaker": "Nisha", "name": "Nisha", "persona": "You are Nisha, 45."}},
                    {"tool": "set_persona", "args": {"speaker": "Sonia", "name": "Sonia Mausi", "persona": "You are Sonia, 35."}},
                    {"tool": "set_persona", "args": {"speaker": "Ananya", "name": "Ananya", "persona": "You are Ananya, 19."}},
                    {"tool": "set_avatar", "args": {"speaker": "Sonia", "asset_id": ""}},
                    {"tool": "set_persona", "args": {"name": "Ananya", "persona": "You are Ananya, 19, a college student."}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "set them as personas"})
        view = await self.settle(chat)
        tools = [m["content"] for m in view["messages"] if m["role"] == "tool"]
        self.assertTrue(all(t["ok"] for t in tools), [t.get("error") for t in tools])
        cast = view["session"]["cast"]
        self.assertEqual([m["display_name"] for m in cast], ["Nisha", "Sonia Mausi", "Ananya"], "no unnamed character 1 left over")
        self.assertEqual(cast[2]["persona"], "You are Ananya, 19, a college student.", "name without speaker edits that character, not the lead")
        self.assertEqual(cast[0]["persona"], "You are Nisha, 45.")

    async def test_talk_goes_on_until_everyone_has_spoken(self):
        cast = [{"name": "Nisha", "persona": "a"}, {"name": "Sonia Mausi", "persona": "b"}, {"name": "Ananya", "persona": "c"}]
        chat = (await self.http.post("/v1/agent/sessions", json={"cast": cast})).json()["id"]
        spoken = []

        def reply(body):
            who = re.match(r"You are (\w+)", body["messages"][0]["content"]).group(1)
            spoken.append(who)
            say = {"Nisha": "Ananya, tum batao.", "Ananya": "Sonia ka gym body alag hai, beta decide karo.", "Sonia": "Main hi jeetungi."}[who]
            return json.dumps({"say": say, "to": "all", "pause": True})  # every turn tries to hand back to the user

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 5})
        await self.settle(chat)
        self.assertEqual(spoken, ["Nisha", "Ananya", "Sonia"], "Sonia (Sonia Mausi) is named, and a pause counts once all have spoken")

    async def test_join_while_they_talk(self):
        cast = [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}]
        chat = (await self.http.post("/v1/agent/sessions", json={"cast": cast})).json()["id"]

        def reply(body):
            last = body["messages"][-1]["content"]
            if "USER: Main bhi hoon" in last:
                return json.dumps({"lines": [{"speaker": "Riya", "say": "Aao aao!"}], "actions": [], "done": True})
            return json.dumps({"say": "chit chat", "to": "all"})

        self.atlas.reply, self.atlas.delay = reply, 0.3
        await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 10})
        await asyncio.sleep(0.5)
        talking = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()["session"]
        self.assertTrue(talking["talking"] and talking["status"] == "running")
        joined = await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Main bhi hoon"})
        self.assertEqual(joined.status_code, 202, joined.text)
        view = await self.settle(chat)
        notes = [m["content"]["text"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") != "talk"]
        self.assertIn("You joined in.", notes)
        rounds = [m for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "talk"]
        self.assertLess(len(rounds), 10)
        self.assertEqual(view["messages"][-1]["content"]["lines"][0]["say"], "Aao aao!")
        self.assertFalse(view["session"]["talking"])

    async def test_characters_feel_privately_make_things_whisper_and_remember(self):
        cast = [{"name": "Maya", "persona": "stylist"}, {"name": "Riya", "persona": "director"}]
        chat = (await self.http.post("/v1/agent/sessions", json={"cast": cast, "adaptive": True, "whispers": True})).json()
        self.assertTrue(chat["whispers"])
        chat = chat["id"]
        prompts = {}
        turns = iter([
            ("Maya", {"say": "Riya, you always pick for me.", "to": "Riya",
                      "grow": [{"about": "Riya", "note": "Tired of Riya deciding everything."}, {"about": "user", "note": "Trusts the user's eye."}]}),
            ("Riya", {"say": "Fine, let's see you in emerald then.", "to": "Maya", "make": "A photo of Maya in an emerald lehenga at a sangeet"}),
            ("Maya", {"say": "Okay... I actually love it.", "to": "all", "pause": True}),
        ])

        def reply(body):
            system = body["messages"][0]["content"]
            character = re.match(r"You are (\w+), one of the characters", system)
            if character:
                who, turn = next(turns)
                self.assertEqual(character.group(1), who)
                prompts.setdefault(who, []).append(body["messages"])
                return json.dumps(turn)
            if system.startswith("You are Hawk"):
                last = body["messages"][-1]["content"]
                if "STAGE DIRECTION" in last and "TOOL RESULT" not in last:
                    return json.dumps({"say": "", "actions": [{"tool": "generate_image", "by": "Riya", "args": {"prompt": "Maya in emerald"}}]})
                if "whispering privately to Maya" in last:
                    return json.dumps({"lines": [{"speaker": "Maya", "say": "Shh, secret safe."}], "actions": [], "done": True})
                return json.dumps({"lines": [{"speaker": "Riya", "say": "Yeh lo!"}], "actions": [], "done": True})
            return "MEMORY: I remember everything important."

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 3, "makes": 1})
        view = await self.settle(chat)
        members = {m["display_name"]: m for m in view["session"]["cast"]}
        riya_id = members["Riya"]["id"]
        self.assertEqual(members["Maya"]["feelings"], {riya_id: ["Tired of Riya deciding everything."], "user": ["Trusts the user's eye."]})
        self.assertEqual({v["about"] for v in members["Maya"]["feelings_view"]}, {"Riya", "the user"})
        grows = [m["content"] for m in view["messages"] if m["role"] == "note" and m["content"].get("kind") == "grow"]
        self.assertEqual([(g["speaker"], g.get("about")) for g in grows], [("Maya", "Riya"), ("Maya", "the user")])

        riya_prompt = prompts["Riya"][0][0]["content"]
        self.assertNotIn("Tired of Riya", riya_prompt, "Maya's feelings are private to Maya")
        self.assertIn("Tired of Riya", prompts["Maya"][1][0]["content"], "Maya keeps her own feelings")

        kinds = [(m["role"], m["content"].get("kind") or m["content"].get("tool")) for m in view["messages"] if m["role"] in ("note", "tool")]
        self.assertIn(("note", "make"), kinds)
        self.assertIn(("tool", "generate_image"), kinds, "the director made what Riya asked for")
        self.assertIn("[generate_image: images", prompts["Maya"][1][1]["content"], "and Maya saw it before her next line")

        # whispers: only Maya hears it, and her answer is private too
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "@Maya Riya ko mat batana, emerald best hai"})
        view = await self.settle(chat)
        whisper = [m for m in view["messages"] if m["role"] == "user"][-1]["content"]
        self.assertEqual(whisper["private_to"], members["Maya"]["id"])
        self.assertEqual(view["messages"][-1]["content"]["private_to"], members["Maya"]["id"])
        turns = iter([("Riya", {"say": "Maya, final answer?", "to": "Maya"}), ("Maya", {"say": "Trust me on this one.", "to": "Riya", "pause": True})])
        prompts.clear()
        await self.http.post(f"/v1/agent/sessions/{chat}/talk", json={"rounds": 1})
        await self.settle(chat)
        self.assertIn("whispering only to you", prompts["Maya"][0][1]["content"])
        self.assertNotIn("emerald best hai", prompts["Riya"][0][1]["content"], "Riya never hears the whisper")
        self.assertNotIn("secret safe", prompts["Riya"][0][1]["content"])

        # each character's memory is its own, condensed by the cheap model; Compact does all of them
        compacted = (await self.http.post(f"/v1/agent/sessions/{chat}/compact")).json()
        self.assertEqual(sorted(compacted["memories"]), ["Maya", "Riya"])
        members = {m["display_name"]: m for m in compacted["session"]["cast"]}
        self.assertTrue(members["Maya"]["memory"].startswith("MEMORY"))
        memory_calls = [r for r in self.atlas.requests if r["messages"][0]["content"].startswith("You are Riya. Write your own memory")]
        self.assertEqual(memory_calls[-1]["model"], "deepseek-ai/deepseek-v4.1-flash")
        self.assertNotIn("emerald best hai", memory_calls[-1]["messages"][1]["content"], "Riya's memory has no whisper either")

    async def test_image_model_routing(self):
        seedream = (await self.http.post("/v1/images", json={"prompt": "A chai stall at dusk", "model": "seedream", "n": 2, "size": "2048x2048"})).json()
        self.assertEqual((seedream["model"], len(seedream["assets"])), ("bytedance/seedream-v5.0-pro/text-to-image", 2))
        big = self.atlas.image_requests[-2:]
        self.assertEqual([(r["size"], "n" in r) for r in big], [("2048*2048", False)] * 2, "Pro makes one image per request")
        self.assertEqual(seedream["cost_usd"], 0.144, "2048x2048 is the 2K tier")
        cheap = (await self.http.post("/v1/images", json={"prompt": "A chai stall", "engine": "seedream"})).json()
        self.assertEqual((self.atlas.image_requests[-1]["size"], cheap["cost_usd"]), ("1328*1776", 0.036), "default: 1.5K tier, portrait")
        await self.http.post("/v1/images", json={"prompt": "A chai stall", "engine": "seedream", "size": "1024x1024"})
        self.assertEqual(self.atlas.image_requests[-1]["size"], "1536*1536", "same price, the larger 1.5K preset")
        wide = (await self.http.post("/v1/images", json={"prompt": "A chai stall", "engine": "seedream", "size": "1600x896"})).json()
        self.assertEqual(self.atlas.image_requests[-1]["size"], "2048*1152")
        self.assertIn("nearest preset", wide["note"])
        lite = (await self.http.post("/v1/images", json={"prompt": "A chai stall", "engine": "seedream-lite", "size": "1024x1536"})).json()
        self.assertEqual((lite["model"], self.atlas.image_requests[-1]["size"], lite["cost_usd"]), ("bytedance/seedream-v5.0-lite", "1664*2496", 0.032))
        base = seedream["assets"][0]["id"]
        switched = (await self.http.post("/v1/images", json={"prompt": "Same stall in rain", "model": "turbo", "reference_asset_ids": [base]})).json()
        self.assertEqual(switched["model"], "bytedance/seedream-v5.0-pro/edit")
        self.assertIn("can't use reference images", switched["note"])
        lite_edit = (await self.http.post("/v1/images", json={"prompt": "Same stall in rain", "model": "seedream-lite", "reference_asset_ids": [base]})).json()
        self.assertEqual(lite_edit["model"], "bytedance/seedream-v5.0-lite/edit")
        seeded = (await self.http.post("/v1/images", json={"prompt": "Lamp", "n": 2, "seed": 7})).json()
        self.assertEqual(sorted(r["seed"] for r in self.atlas.image_requests[-2:]), [7, 8])
        self.assertEqual((seeded["model"], self.atlas.image_requests[-1]["size"]), ("z-image/turbo", "1024*1536"))
        too_big = await self.http.post("/v1/images", json={"prompt": "Lamp", "size": "4096x4096", "engine": "turbo"})
        self.assertEqual(too_big.status_code, 422)

    async def test_the_engine_order_is_a_setting(self):
        view = (await self.http.get("/v1/images/engines")).json()
        self.assertEqual([r["engine"] for r in view["generate"] if r["enabled"]], ["krea2", "turbo", "seedream"])
        self.assertEqual([r["engine"] for r in view["edit"] if r["enabled"]], ["krea2", "seedream"])
        self.assertEqual(view["busy"]["mode"], "fall_through", "unchanged until the user says otherwise")

        # the agent reads the live order from image_options, so an edited prompt can't leave it stale
        options = (await self.http.get("/v1/images/options")).json()
        self.assertEqual(options["would_use"]["generate"], "krea2")
        self.assertEqual([r["engine"] for r in options["generate_ladder"] if r["enabled"]], ["krea2", "turbo", "seedream"])
        self.assertEqual([r["cost_usd"] for r in options["generate_ladder"] if r["engine"] == "turbo"], [0.01])

        # put Seedream first and the next image goes straight there, with no local attempt to skip past
        await self.http.put("/v1/images/engines", json={"generate": [{"engine": "seedream"}, {"engine": "krea2"}]})
        made = (await self.http.post("/v1/images", json={"prompt": "A chai stall"})).json()
        self.assertEqual(made["engine"], "seedream")
        self.assertNotIn("tried", made, "it was first, so nothing was skipped to reach it")
        self.assertEqual((await self.http.get("/v1/images/options")).json()["would_use"]["generate"], "seedream")

        # every engine is listed with whether it can actually run, so the UI can show them all honestly
        ladder = {r["engine"]: r for r in (await self.http.get("/v1/images/options")).json()["generate_ladder"]}
        self.assertEqual(ladder["klein"]["ready"], False, "its weights are not on this pod")
        self.assertIn("flux-2-klein", ladder["klein"]["why_not"], "say which file is missing")
        self.assertEqual(ladder["seedream"]["ready"], True)
        self.assertEqual(ladder["krea2"]["ready"], False, "Krea 2 is not installed in this fixture")
        self.assertIn("not on this pod", ladder["krea2"]["why_not"])

        # an engine that cannot do the job is named rather than quietly dropped
        bad = await self.http.put("/v1/images/engines", json={"edit": [{"engine": "zimage"}]})
        self.assertEqual(bad.status_code, 422)
        self.assertIn("Z-Image Turbo", bad.json()["error"])
        self.assertEqual([r["engine"] for r in (await self.http.get("/v1/images/engines")).json()["edit"] if r["enabled"]],
                         ["krea2", "seedream"], "a refused save changes nothing")

    async def test_inspect_image_then_upgrade_to_seedream(self):
        agent_module.WAIT_POLL_SECONDS = 0.05
        reviews = []

        def reply(body):
            first = body["messages"][0]["content"]
            if isinstance(first, list):  # the inspect_image vision call
                reviews.append(body)
                ids = [part["text"].split()[-1].rstrip(":") for part in first if part["type"] == "text" and part["text"].startswith("Image asset_id")]
                return json.dumps({"images": [{"asset_id": i, "score": 4, "issues": ["six fingers"], "verdict": "retry"} for i in ids],
                                   "best": ids[0], "advice": "Switch to seedream for cleaner hands."})
            turn = assistant_turns(body)
            if turn == 0:
                return json.dumps({"say": "Drafting with turbo.", "actions": [{"tool": "generate_image", "args": {"prompt": "Portrait of Maya", "n": 2}}]})
            if turn == 1:
                ids = [a["id"] for a in tool_result(body, "generate_image")["assets"]]
                return json.dumps({"say": "", "actions": [{"tool": "inspect_image", "args": {"asset_ids": ids, "brief": "Portrait of Maya, natural hands"}}]})
            if turn == 2:
                advice = tool_result(body, "inspect_image")["advice"]
                model = "seedream" if "seedream" in advice else "turbo"
                return json.dumps({"say": "Upgrading.", "actions": [{"tool": "generate_image", "args": {"prompt": "Portrait of Maya, relaxed hands", "model": model}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat(persona="You are Maya.", model="xai/grok-4.6")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Apni photo banao"})
        view = await self.settle(chat)
        tools = [m["content"] for m in view["messages"] if m["role"] == "tool"]
        self.assertEqual([c["tool"] for c in tools], ["generate_image", "inspect_image", "generate_image"])
        self.assertTrue(all(c["ok"] for c in tools), tools)
        self.assertEqual((tools[0]["result"]["model"], tools[2]["result"]["model"]), ("z-image/turbo", "bytedance/seedream-v5.0-pro/text-to-image"))
        self.assertEqual(tools[1]["result"]["best"], tools[0]["result"]["assets"][0]["id"])
        parts = reviews[0]["messages"][0]["content"]
        self.assertEqual(sum(1 for p in parts if p["type"] == "image_url"), 2)
        self.assertTrue(parts[2]["image_url"]["url"].startswith("data:image/"))
        self.assertEqual(reviews[0]["model"], "xai/grok-4.6", "the chat's model can see images")
        self.assertIn("inspect_image", self.atlas.requests[0]["messages"][0]["content"])

    async def test_inspect_falls_back_and_failed_takes_move_up_engines(self):
        files = self.fake.model_files
        files["diffusion_models"].append("krea2_turbo_fp8_scaled.safetensors")
        files["text_encoders"].append("qwen3vl_4b_fp8_scaled.safetensors")
        files["vae"] = ["qwen_image_vae.safetensors"]
        self.atlas.model_list = MODELS + [DEEPSEEK]
        self.atlas.fail_vision = {"deepseek-ai/deepseek-v4.1-flash"}  # e.g. refuses to review adult images
        reviewers = []

        def reply(body):
            first = body["messages"][0]["content"]
            if isinstance(first, list):
                reviewers.append(body["model"])
                ids = [part["text"].split()[-1].rstrip(":") for part in first if part["type"] == "text" and part["text"].startswith("Image asset_id")]
                return json.dumps({"images": [{"asset_id": f"media-attachment-0-{n}", "score": 4, "issues": ["bad hands"], "verdict": "retry"}
                                              for n, _ in enumerate(ids)],  # Grok sometimes echoes made-up ids
                                   "best": "media-attachment-0-0", "advice": "Retry with a sharper prompt."})
            turn = assistant_turns(body)
            if turn < 6 and turn % 2 == 0:
                return json.dumps({"say": "", "actions": [{"tool": "generate_image", "args": {"prompt": f"Portrait take {turn}", "engine": "auto"}}]})
            if turn < 6:
                ids = [a["id"] for a in tool_result(body, "generate_image")["assets"]]
                return json.dumps({"say": "", "actions": [{"tool": "inspect_image", "args": {"asset_ids": ids, "brief": "Portrait"}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat(model="deepseek-ai/deepseek-v4.1-flash")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Apni photo banao"})
        view = await self.settle(chat)
        tools = [m["content"] for m in view["messages"] if m["role"] == "tool"]
        self.assertTrue(all(c["ok"] for c in tools), tools)
        made = [c["result"]["engine"] for c in tools if c["tool"] == "generate_image"]
        self.assertEqual(made, ["krea2", "z-image", "seedream"], "each failed take moves auto one engine up")
        inspected = [c["result"] for c in tools if c["tool"] == "inspect_image"]
        self.assertEqual(inspected[0]["model"], "xai/grok-4.6", "falls back when the chat's model fails")
        self.assertIn("content policy", inspected[0]["skipped_models"][0])
        self.assertIn("turbo", inspected[0]["next_engine"])
        self.assertIn("seedream", tools[4]["result"]["engine_note"])
        self.assertEqual(reviewers, ["xai/grok-4.6"] * 3)
        self.assertEqual(inspected[0]["best"], tools[0]["result"]["assets"][0]["id"], "made-up ids mapped back by position")

        # a new message starts over at the free local engine
        def once(action):
            """Run action on the first turn, then finish."""
            pending = [action]

            def reply(_body):
                actions = [pending.pop()] if pending else []
                return json.dumps({"say": "ok", "actions": actions, "done": not actions})
            return reply

        self.atlas.reply = once({"tool": "generate_image", "args": {"prompt": "Lamp"}})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Ek lamp bhi"})
        view = await self.settle(chat)
        last = [m["content"] for m in view["messages"] if m["role"] == "tool"][-1]
        self.assertEqual(last["result"]["engine"], "krea2")

        self.atlas.fail_vision.add("xai/grok-4.6")
        self.atlas.fail_vision.add("xai/grok-4.3")
        self.atlas.reply = once({"tool": "inspect_image", "args": {"asset_ids": [last["result"]["assets"][0]["id"]]}})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Check it"})
        view = await self.settle(chat)
        failed = [m["content"] for m in view["messages"] if m["role"] == "tool"][-1]
        self.assertFalse(failed["ok"])
        self.assertIn("seedream", failed["error"])

    async def test_characters_own_what_they_make(self):
        cast = [{"name": "Nisha", "persona": "stylist"}, {"name": "Sonia Mausi", "persona": "aunt"},
                {"name": "Ananya", "persona": "photographer"}]
        chat = await self.new_chat(cast=cast)

        def reply(body):
            if assistant_turns(body):
                return json.dumps({"lines": [{"speaker": "Ananya", "say": "Yeh lo."}], "actions": [], "done": True})
            return json.dumps({"lines": [{"speaker": "Ananya", "say": "Main khichti hoon."}],
                               "actions": [{"tool": "generate_image", "by": "Ananya",
                                            "args": {"prompt": "a marigold garland on a door", "engine": "turbo"}}]})

        self.atlas.reply = reply
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Ek photo chahiye"})
        view = await self.settle(chat)
        tool = [m for m in view["messages"] if m["role"] == "tool"][-1]
        self.assertEqual(tool["content"]["by"]["name"], "Ananya", "the character marked on the call owns it")
        asset_id = tool["content"]["result"]["assets"][0]["id"]
        stored = self.agent.service.store.get_asset(asset_id)
        self.assertEqual(stored["by"]["name"], "Ananya")
        self.assertIn("by ananya", [t.lower() for t in stored["tags"]], "and it is findable in the library")

        inspected = [m for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] != "generate_image"]
        self.assertTrue(all("by" not in m["content"] for m in inspected),
                        "a call that makes nothing belongs to nobody")

        # a picture of several characters: one of the ones in it owns it, never somebody who isn't
        session = self.agent.get_session(chat)
        action = {"tool": "generate_image", "args": {"prompt": "Nisha and Sonia Mausi laughing on a rooftop"}}
        owners = {self.agent._owner_for(session, action)["name"] for _ in range(40)}
        self.assertEqual(owners, {"Nisha", "Sonia Mausi"}, "a group shot is owned by one of the two in it")

        # nobody named anywhere: the character who asked for it while they were talking
        asked = self.agent._owner_for(session, {"tool": "generate_image", "args": {"prompt": "a diya"}}, asked_by="Sonia Mausi")
        self.assertEqual(asked["name"], "Sonia Mausi")

        # a render belongs to a character too, and the job itself carries it
        def render_reply(body):
            made = any("TOOL RESULT render_film" in m["content"] for m in body["messages"] if m["role"] == "user")
            return json.dumps({"lines": [{"speaker": "Nisha", "say": "Bana rahi hoon."}],
                               "actions": [] if made else [{"tool": "render_film", "by": "Nisha",
                                   "args": {"script": "Nisha walks onto the terrace", "settings": {"megapixels": 0.4}}}],
                               "done": made})

        self.atlas.reply = render_reply
        film = await self.new_chat(cast=cast)
        await self.http.post(f"/v1/agent/sessions/{film}/messages", json={"text": "Ek chhota video banao"})
        view = await self.settle(film)
        render = [m for m in view["messages"] if m["role"] == "tool" and m["content"]["tool"] == "render_film"][-1]
        self.assertTrue(render["content"]["ok"], render["content"].get("error"))
        self.assertEqual(render["content"]["by"]["name"], "Nisha")
        job = (await self.http.get(f"/v1/jobs/{render['content']['result']['id']}")).json()
        self.assertEqual(job["by"]["name"], "Nisha", "the job carries its owner, not just the chat message")

    async def test_a_group_reply_without_lines_still_names_its_speakers(self):
        """Kimi K2.5 answered a three-character chat with plain "say", so every bubble showed as the lead."""
        cast = [{"name": "Nisha", "persona": "sharp-tongued stylist"}, {"name": "Sonia Mausi", "persona": "warm aunt"},
                {"name": "Ananya", "persona": "shy photographer"}]
        chat = await self.new_chat(cast=cast)

        self.atlas.reply = lambda body: json.dumps({
            "say": "Nisha: emerald lehenga lo.\nSonia Mausi: beta, pehle khana khao.\nAnanya: ...light achhi hai.",
            "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Kya pehnu?"})
        view = await self.settle(chat)
        reply = [m for m in view["messages"] if m["role"] == "assistant"][-1]["content"]
        self.assertEqual([line["speaker"] for line in reply["lines"]], ["Nisha", "Sonia Mausi", "Ananya"])
        self.assertNotIn("narrator", reply)

        # the group format is restated last, after the editable prompt's own REPLY FORMAT
        system = self.atlas.requests[-1]["messages"][0]["content"]
        self.assertIn("REPLY FORMAT IN THIS GROUP CHAT", system)
        self.assertGreater(system.index("REPLY FORMAT IN THIS GROUP CHAT"), system.index("REPLY FORMAT"))

        # nobody named: kept as narration rather than put in the lead's mouth
        self.atlas.reply = lambda body: json.dumps({"say": "Inspection done, scores below.", "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Aur?"})
        view = await self.settle(chat)
        reply = [m for m in view["messages"] if m["role"] == "assistant"][-1]["content"]
        self.assertNotIn("lines", reply)
        self.assertTrue(reply["narrator"])

    async def test_video_and_image_loras_stay_apart(self):
        """One models/loras folder holds both families: a render must not reach for a Krea 2 LoRA, and the image
        catalogue must not offer MiniMax H3 ones."""
        files = self.fake.model_files
        files["loras"] += ["krea2_mystic_xxx_v3.safetensors", "snofs_krea2.safetensors", "krea2_realism_v2.safetensors",
                           "MysticXXX_MMH3-V4-ref2va.safetensors", "H3_Motion_BoosterV2.safetensors",
                           "Krea2/krea2_identity_edit_v1_2.safetensors"]

        listed = (await self.http.get("/v1/options")).json()["available_loras"]
        self.assertIn("MysticXXX_MMH3-V4-ref2va.safetensors", listed, "video LoRAs are offered to renders")
        self.assertNotIn("krea2_mystic_xxx_v3.safetensors", listed, "image LoRAs are not")
        self.assertNotIn("snofs_krea2.safetensors", listed)
        self.assertNotIn("Krea2/krea2_identity_edit_v1_2.safetensors", listed,
                         "the identity-edit LoRA belongs to image edits, not to renders")

        image_loras = [item["file"] for item in (await self.http.get("/v1/images/options")).json()["local"]["loras"]]
        self.assertIn("krea2_mystic_xxx_v3.safetensors", image_loras)
        self.assertNotIn("MysticXXX_MMH3-V4-ref2va.safetensors", image_loras, "video LoRAs stay out of images")
        self.assertNotIn("H3_Motion_BoosterV2.safetensors", image_loras)

        # "mystic" is ambiguous across the folder, but each family sees only its own file
        render = await self.http.post("/v1/videos", json={"script": "A walk in the rain",
                                                          "settings": {"loras": [{"name": "mystic"}]}})
        self.assertEqual(render.status_code, 202, render.text)
        applied = [lora["file"] for lora in render.json()["loras"]]
        self.assertIn("MysticXXX_MMH3-V4-ref2va.safetensors", applied)

        refused = await self.http.post("/v1/videos", json={"script": "A walk in the rain",
                                                           "settings": {"loras": [{"name": "snofs_krea2.safetensors"}]}})
        self.assertEqual(refused.status_code, 422, refused.text)
        self.assertIn("image LoRA", refused.json()["error"], refused.text)

    async def test_local_krea_images_and_fallbacks(self):
        files = self.fake.model_files
        auto = (await self.http.post("/v1/images", json={"prompt": "A lamp"})).json()
        self.assertEqual(auto["engine"], "z-image")
        self.assertIn("not installed", auto["tried"][0]["skipped"])

        files["diffusion_models"].append("krea2_turbo_fp8_scaled.safetensors")
        files["text_encoders"].append("qwen3vl_4b_fp8_scaled.safetensors")
        files["vae"] = ["qwen_image_vae.safetensors"]
        files["loras"] += ["krea2_realism_v1.safetensors", "krea2_mystic_xxx_v3.safetensors", "snofs_krea2.safetensors",
                           "krea2_darkbrush.safetensors", "styles/krea2_my_custom.safetensors"]
        options = (await self.http.get("/v1/images/options")).json()
        self.assertEqual((options["local"]["installed"], options["local"]["busy"]), (True, False))
        loras = {l["file"]: l for l in options["local"]["loras"]}
        self.assertTrue(loras["krea2_realism_v1.safetensors"]["installed"])
        self.assertFalse(loras["krea2_enhancer.safetensors"]["installed"])
        self.assertEqual(loras["styles/krea2_my_custom.safetensors"]["kind"], "other")

        made = await self.http.post("/v1/images", json={"prompt": "Portrait of Maya", "n": 2, "size": "1024x1536",
                                                        "loras": [{"name": "realism"}, {"name": "darkbrush", "strength": 0.9}]})
        self.assertEqual(made.status_code, 201, made.text)
        made = made.json()
        self.assertEqual((made["engine"], len(made["assets"])), ("krea2", 2))
        self.assertIn("krea2", made["assets"][0]["tags"])
        graph = list(self.fake.prompts.values())[-1]
        nodes = {n["class_type"]: n["inputs"] for n in graph.values() if n["class_type"] != "LoraLoaderModelOnly"}
        chain = [n["inputs"] for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
        self.assertEqual(nodes["UNETLoader"]["unet_name"], "krea2_turbo_fp8_scaled.safetensors")
        self.assertEqual(nodes["CLIPLoader"]["type"], "krea2")
        self.assertEqual(nodes["CLIPLoader"]["clip_name"], "qwen3vl_4b_fp8_scaled.safetensors")
        files["text_encoders"].remove("qwen3vl_4b_fp8_scaled.safetensors")  # swapped for the bf16 encoder
        files["text_encoders"] += ["qwen3vl_8b_nvfp4.safetensors", "qwen3vl_4b_bf16.safetensors"]
        options = (await self.http.get("/v1/images/options")).json()
        self.assertEqual((options["local"]["installed"], options["local"]["files"]["clip"]), (True, "qwen3vl_4b_bf16.safetensors"))
        await self.http.post("/v1/images", json={"prompt": "A lamp", "engine": "local"})
        graph = list(self.fake.prompts.values())[-1]
        self.assertEqual(next(n["inputs"]["clip_name"] for n in graph.values() if n["class_type"] == "CLIPLoader"), "qwen3vl_4b_bf16.safetensors")
        files["text_encoders"].append("qwen3vl_4b_fp8_scaled.safetensors")
        self.assertEqual([(c["lora_name"], c["strength_model"]) for c in chain],
                         [("krea2_realism_v1.safetensors", 0.8), ("krea2_darkbrush.safetensors", 0.9),
                          ("snofs_krea2.safetensors", 0.8), ("krea2_mystic_xxx_v3.safetensors", 0.5)],
                         "the picked LoRAs, then the go-to adult pair that every local generation gets")
        self.assertEqual((nodes["KSampler"]["steps"], nodes["KSampler"]["cfg"], nodes["KSampler"]["scheduler"]), (8, 1.0, "simple"))
        self.assertEqual((nodes["EmptyLatentImage"]["width"], nodes["EmptyLatentImage"]["height"], nodes["EmptyLatentImage"]["batch_size"]), (1024, 1536, 2))
        self.assertTrue(nodes["CLIPTextEncode"]["text"].endswith(", muted minimalist sketch style"), "trigger word added")

        await self.http.post("/v1/images", json={"prompt": "Portrait", "engine": "local", "loras": [{"name": "mystic"}]})
        sampler = next(n["inputs"] for n in list(self.fake.prompts.values())[-1].values() if n["class_type"] == "KSampler")
        self.assertEqual((sampler["steps"], sampler["scheduler"]), (12, "beta"), "the LoRA's recommended sampler settings")
        pair = await self.http.post("/v1/images", json={"prompt": "x", "engine": "local", "loras": [{"name": "mystic"}, {"name": "snofs_krea2"}]})
        self.assertEqual(pair.status_code, 201, "the go-to pair works by default (agent and MCP too)")
        self.assertNotIn("note", pair.json())
        sampler = next(n["inputs"] for n in list(self.fake.prompts.values())[-1].values() if n["class_type"] == "KSampler")
        self.assertEqual((sampler["steps"], sampler["scheduler"]), (12, "beta"), "Mystic's settings apply to the pair")
        strict = await self.http.post("/v1/images", json={"prompt": "x", "loras": [{"name": "mystic"}, {"name": "snofs_krea2"}], "max_adult_loras": 1})
        self.assertEqual(strict.status_code, 422, "a LoRA error is not hidden by falling back to another engine")
        files["loras"].append("krea2_nsfw_master_turbo.safetensors")
        three = await self.http.post("/v1/images", json={"prompt": "x", "engine": "local", "loras": [
            {"name": "mystic", "strength": 0.7}, {"name": "snofs_krea2", "strength": 0.8}, {"name": "nsfw_master", "strength": 0.8}]})
        self.assertEqual(three.status_code, 201, three.text)
        self.assertIn("combined strength of 2.30", three.json()["note"])
        self.assertEqual((await self.http.post("/v1/images", json={"prompt": "x", "max_adult_loras": 4})).status_code, 422)
        spent = len(self.atlas.image_requests)
        refused = await self.http.post("/v1/images", json={"prompt": "a 16 year old girl on a beach"})
        self.assertEqual(refused.status_code, 422, "refused outright, not passed to another engine")
        self.assertIn("under 18", refused.json()["error"])
        # two rungs were still below it; a refusal must not walk down to one of them, least of all a paid one
        self.assertEqual(len(self.atlas.image_requests), spent, "a refused prompt never reaches a paid engine")

        # a LoRA for another local engine must not be reachable from a Krea 2 image, by any name
        self.fake.model_files["loras"] += ["klein_snofs.safetensors", "zit_mystic_xxx.safetensors"]
        for name in ("klein_snofs", "zit_mystic_xxx"):
            wrong = await self.http.post("/v1/images", json={"prompt": "A lamp", "loras": [{"name": name}]})
            self.assertEqual(wrong.status_code, 422, f"{name} is not a Krea 2 LoRA")
        images = (await self.http.get("/v1/images/options")).json()["local"]["loras"]
        self.assertNotIn("klein_snofs.safetensors", [l["file"] for l in images], "another engine's LoRA isn't offered")

        # with its weights present, the local Z-Image builds its own graph rather than Krea 2's
        files = self.fake.model_files
        files["diffusion_models"].append("z_image_turbo_nvfp4.safetensors")
        files["text_encoders"].append("qwen_3_4b_fp4_mixed.safetensors")
        files["vae"].append("z_image_ae.safetensors")
        zimage = await self.http.post("/v1/images", json={"prompt": "A lamp", "engine": "z-image"})
        self.assertEqual(zimage.status_code, 201, zimage.text)
        self.assertEqual(zimage.json()["model"], "zimage/turbo", "never 'z-image/...', which means the Atlas engine")
        graph = list(self.fake.prompts.values())[-1]
        kinds = {node["class_type"] for node in graph.values()}
        self.assertIn("ModelSamplingAuraFlow", kinds, "Z-Image samples through AuraFlow")
        clip = next(n for n in graph.values() if n["class_type"] == "CLIPLoader")
        self.assertEqual(clip["inputs"]["type"], "lumina2", "not a Qwen type, whatever the file name suggests")
        sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
        self.assertEqual(sampler["inputs"]["sampler_name"], "res_multistep")

        for name in ("diffusion_models", "text_encoders", "vae"):
            files[name].pop()
        moved = await self.http.post("/v1/images", json={"prompt": "A lamp", "engine": "z-image"})
        self.assertEqual(moved.status_code, 422, "engine z-image now means the local engine, which isn't wired up")
        self.assertIn("local", moved.json()["error"].lower())

        self.fake.running["render"] = asyncio.get_event_loop().create_future()  # a video render holds ComfyUI
        busy = (await self.http.post("/v1/images", json={"prompt": "A lamp"})).json()
        self.assertEqual(busy["engine"], "z-image")
        self.assertIn("busy", busy["tried"][0]["skipped"])
        self.assertEqual((await self.http.post("/v1/images", json={"prompt": "A lamp", "engine": "local"})).status_code, 422)
        self.fake.running.pop("render")

        # Krea 2 Identity Edit: needs the comfyui-krea2edit nodes and the LoRA; auto falls back to Seedream edit
        character = made["assets"][0]["id"]
        edit = (await self.http.post("/v1/images", json={"prompt": "Same, red dress", "reference_asset_ids": [character]})).json()
        self.assertEqual(edit["model"], "bytedance/seedream-v5.0-pro/edit")
        self.assertIn("krea2_identity_edit", edit["tried"][0]["skipped"])
        self.assertEqual((await self.http.post("/v1/images", json={"prompt": "x", "engine": "local", "reference_asset_ids": [character]})).status_code, 422)
        files["loras"] += ["Krea2/krea2_identity_edit_v1_2_r128.safetensors", "Krea2/krea2_identity_edit_v1_2.safetensors"]
        self.fake.missing_nodes = {"Krea2EditGroundedEncode"}
        options = (await self.http.get("/v1/images/options")).json()["local"]
        self.assertEqual(options["edit"]["missing"], ["custom node Krea2EditGroundedEncode (comfyui-krea2edit)"])
        self.assertNotIn("krea2_identity_edit", json.dumps(options["loras"]), "the edit LoRA isn't offered for text-to-image")
        self.fake.missing_nodes = set()
        options = (await self.http.get("/v1/images/options")).json()["local"]
        self.assertEqual((options["edit"]["installed"], options["edit"]["lora"]), (True, "Krea2/krea2_identity_edit_v1_2.safetensors"))

        edit = await self.http.post("/v1/images", json={"prompt": "Change her outfit to a red raincoat", "reference_asset_ids": [character],
                                                        "loras": [{"name": "realism_v1"}]})
        self.assertEqual(edit.status_code, 201, edit.text)
        edit = edit.json()
        self.assertEqual((edit["engine"], edit["model"]), ("krea2-edit", "krea2/identity-edit"))
        self.assertEqual([l["file"] for l in edit["loras"]],
                         ["Krea2/krea2_identity_edit_v1_2.safetensors", "krea2_realism_v1.safetensors",
                          "snofs_krea2.safetensors", "krea2_mystic_xxx_v3.safetensors"],
                         "editing a picture made here is a fictional character, so it gets the adult pair too")
        graph = list(self.fake.prompts.values())[-1]
        by = {n["class_type"]: n["inputs"] for n in graph.values()}
        chain = [n["inputs"] for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]
        self.assertEqual((chain[0]["lora_name"], chain[0]["strength_model"]), ("Krea2/krea2_identity_edit_v1_2.safetensors", 1.0))
        self.assertEqual(by["LoadImage"]["image"], made["assets"][0]["path"])
        self.assertEqual((by["Krea2EditModelPatch"]["ref_boost"], by["Krea2EditModelPatch"]["fit_mode"]), (4.0, "fit"))
        self.assertEqual(by["Krea2EditModelPatch"]["model"], [f"l{len(chain) - 1}", 0])
        self.assertNotIn("source_latent_b", by["Krea2EditModelPatch"])
        encodes = [n["inputs"] for n in graph.values() if n["class_type"] == "Krea2EditGroundedEncode"]
        self.assertEqual(sorted(e["prompt"] for e in encodes), ["", "Change her outfit to a red raincoat"])
        self.assertEqual((by["KSampler"]["steps"], by["KSampler"]["cfg"], by["KSampler"]["model"]), (10, 1.0, ["patch", 0]))
        self.assertEqual((by["EmptySD3LatentImage"]["width"], by["EmptySD3LatentImage"]["height"]), (992, 992), "about 1 MP")

        person = (await self.http.post("/v1/images", json={"prompt": "A woman", "engine": "local"})).json()["assets"][0]["id"]
        both = (await self.http.post("/v1/images", json={"prompt": "Place this person at the cafe table", "ref_boost": 6,
                                                         "reference_asset_ids": [character, person], "n": 2})).json()
        self.assertEqual(both["engine"], "krea2-edit")
        graphs = list(self.fake.prompts.values())[-2:]
        patch = next(n["inputs"] for n in graphs[-1].values() if n["class_type"] == "Krea2EditModelPatch")
        self.assertEqual((patch["ref_boost"], patch["source_latent_b"]), (6.0, ["enc2", 0]))
        seeds = [next(n["inputs"]["seed"] for n in g.values() if n["class_type"] == "KSampler") for g in graphs]
        self.assertEqual(seeds[1], seeds[0] + 1, "n edits run as separate prompts")
        # Blackwell cuDNN has no plan for the likeness boost's attention mask: retry the edit at 1.0
        def cudnn(prompt):
            patch = next((n["inputs"] for n in prompt.values() if n["class_type"] == "Krea2EditModelPatch"), None)
            return "cuDNN Frontend error: No valid execution plans built." if patch and patch["ref_boost"] != 1.0 else None
        self.fake.fail_image = cudnn
        retried = (await self.http.post("/v1/images", json={"prompt": "Red raincoat", "reference_asset_ids": [character]})).json()
        self.assertEqual(retried["engine"], "krea2-edit")
        self.assertIn("likeness boost (4) fails", retried["note"])
        boosts = [next(n["inputs"]["ref_boost"] for n in p.values() if n["class_type"] == "Krea2EditModelPatch")
                  for p in list(self.fake.prompts.values())[-2:]]
        self.assertEqual(boosts, [4.0, 1.0])
        self.assertFalse((await self.http.get("/v1/images/options")).json()["local"]["edit"]["boost"])
        again = (await self.http.post("/v1/images", json={"prompt": "Blue raincoat", "n": 2, "reference_asset_ids": [character]})).json()
        boosts = [next(n["inputs"]["ref_boost"] for n in p.values() if n["class_type"] == "Krea2EditModelPatch")
                  for p in list(self.fake.prompts.values())[-2:]]
        self.assertEqual((boosts, len(again["assets"])), ([1.0, 1.0], 2), "after a failure, edits go straight to 1.0")
        self.fake.fail_image = lambda prompt: "CUDA out of memory"
        oom = await self.http.post("/v1/images", json={"prompt": "Red raincoat", "engine": "local", "reference_asset_ids": [character]})
        self.assertEqual(oom.status_code, 422)
        self.assertIn("in KSampler (node 7): CUDA out of memory", oom.text)
        self.fake.fail_image = None
        logs = (await self.http.get("/v1/debug/comfy-logs", params={"grep": "cudnn"})).json()
        self.assertEqual(logs["lines"], ["[Hawk H3] masked attention skips cuDNN on this Blackwell GPU"])
        three = (await self.http.post("/v1/images", json={"prompt": "Group shot", "reference_asset_ids": [character, person, character]})).json()
        self.assertEqual(three["model"], "bytedance/seedream-v5.0-pro/edit")

        # uploaded photos may show real people: no sexual edits, no adult LoRAs; generated characters follow the normal rules
        photo = (await self.http.post("/v1/assets", files={"files": ("me.png", tiny_png(color=(10, 20, 30)), "image/png")})).json()["assets"][0]["id"]
        for body in ({"prompt": "Make her naked"}, {"prompt": "Beach photo", "loras": [{"name": "mystic"}]}):
            refused = await self.http.post("/v1/images", json={**body, "reference_asset_ids": [photo]})
            self.assertEqual(refused.status_code, 422, body)
            self.assertIn("Refused", refused.text)
        fine = await self.http.post("/v1/images", json={"prompt": "Change the background to a beach", "reference_asset_ids": [photo]})
        self.assertEqual(fine.json()["engine"], "krea2-edit")
        self.assertEqual([l["file"] for l in fine.json()["loras"]], ["Krea2/krea2_identity_edit_v1_2.safetensors"],
                         "a photo edit never gets the adult pair: it may be a real person")
        derived = fine.json()["assets"][0]["id"]  # made from the photo: still counts as the photo, edit after edit
        again = (await self.http.post("/v1/images", json={"prompt": "Same, sunset", "reference_asset_ids": [derived]})).json()["assets"][0]["id"]
        for ref in (derived, again):
            laundered = await self.http.post("/v1/images", json={"prompt": "Same woman, evening", "reference_asset_ids": [ref],
                                                                 "loras": [{"name": "mystic"}]})
            self.assertEqual(laundered.status_code, 422, "an edit of an upload can't be re-edited with adult LoRAs")
        adult = await self.http.post("/v1/images", json={"prompt": "Same woman, evening", "reference_asset_ids": [character], "loras": [{"name": "mystic"}]})
        self.assertEqual(adult.status_code, 201, "generated (fictional) characters can use adult LoRAs")
        turbo = (await self.http.post("/v1/images", json={"prompt": "x", "engine": "turbo", "reference_asset_ids": [character]})).json()
        self.assertEqual(turbo["model"], "bytedance/seedream-v5.0-pro/edit")

    async def test_generate_and_edit_images(self):
        agent_module.WAIT_POLL_SECONDS = 0.05
        created = (await self.http.post("/v1/images", json={"prompt": "A fit model in her forties, studio portrait", "n": 2, "size": "1536x2048"})).json()
        self.assertEqual(created["model"], "z-image/turbo", "cheap text-to-image by default")
        self.assertEqual(len(created["assets"]), 2)
        first = created["assets"][0]
        self.assertEqual((first["kind"], first["filename"][:4]), ("image", "gen_"))
        self.assertEqual(self.atlas.image_requests[-2:], [  # z-image makes one image per request
            {"model": "z-image/turbo", "prompt": "A fit model in her forties, studio portrait", "size": "1536*2048", "prompt_extend": False, "seed": -1}] * 2)
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

    async def test_prompts_are_editable(self):
        prompts = {p["name"]: p for p in (await self.http.get("/v1/prompts")).json()["prompts"]}
        self.assertTrue(prompts["agent"]["is_default"] and prompts["planner"]["is_default"])
        self.assertIn("{{TOOLS}}", prompts["agent"]["text"])
        self.assertIn('"segments"', prompts["planner"]["text"])
        self.assertIn("under 18", prompts["agent"]["platform_rules"])
        self.assertNotIn("under 18", prompts["agent"]["text"], "platform rules are not part of the editable text")

        custom = ("You are Test Director.\nPERSONA: {{PERSONA}}\nCRITERIA: always use 9:16 and one location.\n{{TOOLS}}\n"
                  'Reply with JSON {"say": "", "actions": [], "done": true}.')
        saved = (await self.http.put("/v1/prompts/agent", json={"text": custom})).json()
        self.assertEqual((saved["is_default"], saved["warnings"]), (False, []))
        self.atlas.reply = lambda body: json.dumps({"say": "ok", "actions": [], "done": True})
        chat = await self.new_chat(persona="Moody noir cinematographer")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "hi"})
        await self.settle(chat)
        system = self.atlas.requests[-1]["messages"][0]["content"]
        self.assertTrue(system.startswith("You are Test Director."))
        self.assertIn("PERSONA: Moody noir cinematographer", system)
        self.assertIn("always use 9:16", system)
        self.assertIn("- render_film:", system)
        self.assertNotIn("HOW YOU WORK", system)
        self.assertTrue(system.rstrip().endswith("including from their photos."))

        without_tools = (await self.http.put("/v1/prompts/agent", json={"text": "Minimal. {{PERSONA}}"})).json()
        self.assertTrue(any("{{TOOLS}}" in w for w in without_tools["warnings"]))
        self.assertEqual(without_tools["history"][0]["text"], custom)
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "again"})
        await self.settle(chat)
        self.assertIn("\n\nTOOLS\n- ", self.atlas.requests[-1]["messages"][0]["content"])

        reset = (await self.http.post("/v1/prompts/agent/reset")).json()
        self.assertTrue(reset["is_default"])
        self.assertEqual(len(reset["history"]), 2)

        planner_text = 'Custom planner guide. Return JSON {"segments": [...]}.'
        self.assertEqual((await self.http.put("/v1/prompts/planner", json={"text": planner_text})).json()["warnings"], [])
        plan = (await self.http.post("/v1/plans", json={"story": "A walk"})).json()
        node = next(n for n in self.fake.prompts[plan["id"]].values() if n["class_type"] == "HawkH3StoryPlanner")
        self.assertTrue(node["inputs"]["system_prompt"].startswith(planner_text))
        self.assertIn("PLATFORM RULES", node["inputs"]["system_prompt"])
        await self.http.post("/v1/prompts/planner/reset")
        plan = (await self.http.post("/v1/plans", json={"story": "A walk"})).json()
        node = next(n for n in self.fake.prompts[plan["id"]].values() if n["class_type"] == "HawkH3StoryPlanner")
        self.assertEqual(node["inputs"]["system_prompt"], "")
        self.assertEqual((await self.http.get("/v1/prompts/nope")).status_code, 404)

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
        self.agent.compact_tokens, self.agent.keep_messages = 1, 2

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
        summaries = [r for r in self.atlas.requests if "response_format" not in r]
        self.assertEqual(summaries[0]["model"], "deepseek-ai/deepseek-v4.1-flash", "summaries use the cheap model")
        last = [r for r in self.atlas.requests if "response_format" in r][-1]
        self.assertTrue(any("SUMMARY OF THE EARLIER CONVERSATION" in m["content"] for m in last["messages"] if m["role"] == "system"))

    async def test_compaction_saves_tokens(self):
        long_prompt = "A very detailed portrait prompt. " * 60

        def reply(body):
            if "response_format" not in body:
                return "SUMMARY: made a portrait of Maya (asset ids in the tool results)."
            turn = assistant_turns(body)
            if turn == 0:
                return json.dumps({"say": "Making it.", "actions": [{"tool": "generate_image", "args": {"prompt": long_prompt}}]})
            return json.dumps({"say": "Done.", "actions": [], "done": True})

        self.atlas.reply = reply
        self.atlas.usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 800}}
        chat = await self.new_chat(model="xai/grok-4.6")
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Photo banao"})
        view = await self.settle(chat)
        asset_id = [m for m in view["messages"] if m["role"] == "tool"][0]["content"]["result"]["assets"][0]["id"]
        usage = [m["content"]["usage"] for m in view["messages"] if m["role"] == "assistant"]
        self.assertEqual((usage[0]["in"], usage[0]["out"], usage[0]["cached"], usage[0]["model"]), (1000, 100, 800, "xai/grok-4.6"))
        self.assertAlmostEqual(usage[0]["cost_usd"], 200 * 2e-6 + 800 * 5e-7 + 100 * 6e-6, places=7, msg="cached input at the cache price")

        # the current turn is sent in full; the tool list shows render_film's schema only after describe_tool
        current = self.atlas.requests[-1]["messages"]
        self.assertIn("sha256", current[-1]["content"])
        system = current[0]["content"]
        self.assertIn('call describe_tool {"name": "render_film"}', system)
        self.assertIn("generate_image", system)

        self.atlas.reply = lambda body: json.dumps({"say": "ok", "actions": [] if assistant_turns(body) > 2 else
                                                    [{"tool": "describe_tool", "args": {"name": "render_film"}}], "done": False})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "Ab video"})
        await self.settle(chat)
        later = self.atlas.requests[-1]["messages"]
        history = "\n".join(m["content"] for m in later[1:])
        self.assertIn(asset_id, history, "ids survive the trim")
        self.assertNotIn("sha256", history, "an earlier turn's raw tool details are trimmed")
        self.assertNotIn(long_prompt, history, "an earlier turn's long arguments are cut")
        self.assertNotIn('call describe_tool {"name": "render_film"}', later[0]["content"], "schema shown once described")

        before = self.agent.store.messages(chat)
        self.atlas.reply = reply
        compacted = await self.http.post(f"/v1/agent/sessions/{chat}/compact")
        self.assertEqual(compacted.status_code, 200, compacted.text)
        body = compacted.json()
        self.assertTrue(body["compacted"])
        self.assertLess(body["after"], body["before"])
        self.assertTrue(body["session"]["summary"].startswith("SUMMARY"))
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()
        self.assertEqual(len(view["messages"]), len(before) + 1, "nothing is deleted; a note is added")
        self.assertEqual(view["messages"][-1]["content"]["kind"], "compact")
        again = (await self.http.post(f"/v1/agent/sessions/{chat}/compact")).json()
        self.assertFalse(again["compacted"])

    async def test_forgetting_a_reply_keeps_it_out_of_the_prompt_until_restored(self):
        self.atlas.reply = lambda body: json.dumps({"say": "Nisha wore the emerald lehenga.", "actions": [], "done": True})
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "What did she wear?"})
        view = await self.settle(chat)
        reply = [m for m in view["messages"] if m["role"] == "assistant"][-1]

        forgotten = await self.http.delete(f"/v1/agent/sessions/{chat}/messages/{reply['id']}")
        self.assertEqual(forgotten.status_code, 200, forgotten.text)
        self.assertEqual(forgotten.json()["forgotten"], [reply["id"]])
        self.assertFalse(forgotten.json()["rebuilt"], "nothing was summarised yet")

        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()
        kept = next(m for m in view["messages"] if m["id"] == reply["id"])
        self.assertTrue(kept["excluded"], "the bubble stays in Studio, marked")

        self.atlas.reply = lambda body: json.dumps({"say": "ok", "actions": [], "done": True})
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "and after that?"})
        await self.settle(chat)
        sent = "\n".join(m["content"] for m in self.atlas.requests[-1]["messages"])
        self.assertNotIn("emerald lehenga", sent, "a forgotten reply is never sent again")
        self.assertIn("What did she wear?", sent, "the rest of the chat is untouched")

        restored = await self.http.post(f"/v1/agent/sessions/{chat}/messages/{reply['id']}/restore")
        self.assertEqual(restored.status_code, 200, restored.text)
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "once more"})
        await self.settle(chat)
        self.assertIn("emerald lehenga", "\n".join(m["content"] for m in self.atlas.requests[-1]["messages"]))

    async def test_forget_last_takes_the_tool_results_with_it_and_purge_removes_the_row(self):
        def reply(body):
            if assistant_turns(body):
                return json.dumps({"say": "Here it is.", "actions": [], "done": True})
            return json.dumps({"say": "Looking.", "actions": [{"tool": "list_references", "args": {}}]})

        self.atlas.reply = reply
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "references dikhao"})
        view = await self.settle(chat)
        with_tool = [m for m in view["messages"] if m["role"] == "assistant"][0]
        tool_row = next(m for m in view["messages"] if m["role"] == "tool")

        result = (await self.http.post(f"/v1/agent/sessions/{chat}/forget_last")).json()
        self.assertEqual(result["forgotten"], [[m for m in view["messages"] if m["role"] == "assistant"][-1]["id"]])

        forgotten = (await self.http.delete(f"/v1/agent/sessions/{chat}/messages/{with_tool['id']}")).json()
        self.assertIn(tool_row["id"], forgotten["forgotten"], "a reply takes its tool results with it")
        hidden = self.agent.store.messages(chat)
        self.assertEqual([m["id"] for m in hidden if m["role"] in ("assistant", "tool")], [],
                         "nothing the model said is left in the prompt")

        purged = await self.http.delete(f"/v1/agent/sessions/{chat}/messages/{with_tool['id']}?mode=purge")
        self.assertEqual(purged.status_code, 200, purged.text)
        view = (await self.http.get(f"/v1/agent/sessions/{chat}")).json()
        self.assertNotIn(with_tool["id"], [m["id"] for m in view["messages"]], "purged rows are gone from Studio too")
        self.assertNotIn(tool_row["id"], [m["id"] for m in view["messages"]])
        self.assertEqual((await self.http.delete(f"/v1/agent/sessions/{chat}/messages/{with_tool['id']}")).status_code, 404)

    async def test_forgetting_a_summarised_reply_rebuilds_the_summary(self):
        self.agent.compact_tokens, self.agent.keep_messages = 1, 2

        def reply(body):
            if "response_format" not in body:
                return "SUMMARY: talked about the emerald lehenga"
            return json.dumps({"say": "Noted.", "actions": [], "done": True})

        self.atlas.reply = reply
        chat = await self.new_chat()
        for text in ("first", "second", "third"):
            await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": text})
            view = await self.settle(chat)
        session = view["session"]
        self.assertTrue(session["summary"], "the chat has been summarised")
        old = next(m for m in view["messages"] if m["id"] <= session["summary_upto"] and m["role"] == "assistant")

        result = (await self.http.delete(f"/v1/agent/sessions/{chat}/messages/{old['id']}")).json()
        self.assertTrue(result["rebuilt"], "the summary covered it, so it is dropped")
        self.assertEqual(result["session"]["summary"], "")
        self.assertEqual(result["session"]["summary_upto"], 0)

    async def test_forgetting_waits_for_the_agent_to_finish(self):
        self.atlas.delay = 0.3
        self.atlas.reply = lambda body: json.dumps({"say": "Working.", "actions": [{"tool": "list_references", "args": {}}]})
        chat = await self.new_chat()
        await self.http.post(f"/v1/agent/sessions/{chat}/messages", json={"text": "go"})
        await asyncio.sleep(0.2)
        busy = await self.http.post(f"/v1/agent/sessions/{chat}/forget_last")
        self.assertEqual(busy.status_code, 409)
        await self.http.post(f"/v1/agent/sessions/{chat}/stop")
        await self.settle(chat)

    def test_databases_from_before_forgetting_still_open(self):
        import sqlite3

        from hawk_api.agent import AgentStore
        path = os.path.join(tempfile.mkdtemp(), "old.sqlite3")
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE agent_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
                   "role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL)")
        db.execute("INSERT INTO agent_messages (session_id, role, content, created_at) VALUES ('s', 'user', '{}', 1)")
        db.commit()
        db.close()
        store = AgentStore(path)  # migrates the column in
        self.assertEqual([m["id"] for m in store.messages("s")], [1])
        store.set_excluded("s", [1], True)
        self.assertEqual(store.messages("s"), [])
        self.assertTrue(store.messages("s", include_hidden=True)[0]["excluded"])


class Pieces(unittest.IsolatedAsyncioTestCase):
    def test_image_lora_catalogue_upgrades(self):
        from hawk_api.local_images import load_catalogue
        path = os.path.join(tempfile.mkdtemp(), "image_loras.json")
        with open(path, "w") as handle:  # a pod's copy of an older catalogue
            json.dump({"loras": [{"file": "krea2_enhancer.safetensors", "kind": "detail", "notes": "old"}]}, handle)
        items = {item.file: item for item in load_catalogue(path)}
        self.assertTrue(items["krea2_enhancer.safetensors"].notes.startswith("AVOID"))
        self.assertTrue(items["snofs_krea2.safetensors"].notes.startswith("GO-TO"))
        self.assertTrue(os.path.exists(path + ".bak"))
        with open(path) as handle:
            data = json.load(handle)
        data["loras"][0]["strength"] = 0.33
        with open(path, "w") as handle:
            json.dump(data, handle)
        self.assertEqual(load_catalogue(path)[0].strength, 0.33, "edits to a current copy are kept")

    async def test_local_generation_attaches_the_adult_pair_by_default(self):
        from hawk_api.local_images import DEFAULT_ADULT_LORAS, LocalImageEngine

        PAIR = DEFAULT_ADULT_LORAS["krea2"]

        class Catalogue(LocalImageEngine):
            def __init__(self, adult_default=True, stored=None):
                self.adult_default = adult_default
                # stored=None means the user has never set defaults, so the shipped pair applies
                self.service = SimpleNamespace(available_models=self._files,
                                               image_engines=SimpleNamespace(defaults=lambda _f: stored))

            async def _files(self, _kind):
                return list(PAIR) + ["krea2_realism_v2.safetensors"]

            async def catalogue(self, family=""):
                from hawk_api.local_images import ImageLora
                return [ImageLora(file=PAIR[0], label="SNOFS", kind="adult", strength=0.8, installed=True),
                        ImageLora(file=PAIR[1], label="Mystic XXX v3", kind="adult", strength=0.5, installed=True),
                        ImageLora(file="krea2_nsfw_v4.safetensors", label="NSFW v4", kind="adult", strength=0.8, installed=True),
                        ImageLora(file="krea2_realism_v2.safetensors", label="Realism v2", kind="realism", strength=0.7, installed=True)]

        engine = Catalogue()
        chosen, _, _ = await engine.resolve_loras(None, adult_default=True)
        self.assertEqual([f for f, _ in chosen], list(PAIR), "nothing asked for: the go-to pair")
        self.assertEqual([s for _, s in chosen], [0.8, 0.5], "at their recommended strengths")

        chosen, _, _ = await engine.resolve_loras([{"name": "krea2_realism_v2.safetensors"}], adult_default=True)
        self.assertEqual([f for f, _ in chosen], ["krea2_realism_v2.safetensors", *PAIR],
                         "a realism pick keeps the pair too")

        chosen, _, _ = await engine.resolve_loras([{"name": "krea2_nsfw_v4.safetensors"}], adult_default=True)
        self.assertEqual([f for f, _ in chosen], ["krea2_nsfw_v4.safetensors"], "an explicit adult pick wins")

        chosen, _, _ = await engine.resolve_loras(None, adult_default=False)
        self.assertEqual(chosen, [], "a photo edit and a disabled default get nothing")

        # a LoRA nobody asked for must not quietly change steps, scheduler or sampler for ordinary images
        _, used, _ = await engine.resolve_loras([{"name": "krea2_realism_v2.safetensors"}], adult_default=True)
        self.assertEqual([i.automatic for i in used], [False, True, True])
        _, used, _ = await engine.resolve_loras([{"name": "snofs_krea2.safetensors"}], adult_default=True)
        self.assertEqual([i.automatic for i in used], [False], "asking for it yourself keeps its sampler hints")

        # Studio's panel replaces the shipped pair for that family, and an empty list switches it off
        picked = Catalogue(stored=[{"name": "krea2_nsfw_v4.safetensors", "strength": 0.6}])
        chosen, _, _ = await picked.resolve_loras(None, adult_default=True)
        self.assertEqual(chosen, [("krea2_nsfw_v4.safetensors", 0.6)], "the stored default wins over the shipped one")
        off = Catalogue(stored=[])
        self.assertEqual(await off.resolve_loras(None, adult_default=True), ([], [], []),
                         "switched off is not the same as never set")

    def test_a_third_character_is_not_talked_over(self):
        """Two characters naming each other every line used to lock the third out of the conversation."""
        from hawk_api.cast_talk import next_speaker
        cast = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        names = ["Nisha", "Sonia Mausi", "Ananya"]
        spoke, index = [], 0
        for _ in range(12):  # Nisha and Ananya keep addressing each other by name
            said = "Ananya, dekh na" if index == 0 else "Nisha, tu bata"
            index = next_speaker(cast, names, index, said, spoke)
            spoke.append(index)
        self.assertIn(1, spoke, "Sonia Mausi gets the floor")
        gaps = [n for n, who in enumerate(spoke) if who == 1]
        self.assertLessEqual(max(b - a for a, b in zip(gaps, gaps[1:])) if len(gaps) > 1 else gaps[0], 4,
                             "and keeps getting it, at least every few turns")

    def test_naming_still_decides_a_normal_turn(self):
        from hawk_api.cast_talk import next_speaker
        cast = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        names = ["Nisha", "Sonia Mausi", "Ananya"]
        self.assertEqual(next_speaker(cast, names, 0, "Ananya, tera kya plan hai?", [1, 2, 0]), 2)

    def test_parse_reply(self):
        self.assertEqual(parse_reply('```json\n{"say": "a", "actions": [{"tool": "x"}]}\n```'),
                         {"say": "a", "actions": [{"tool": "x", "args": {}}], "done": False})
        self.assertEqual(parse_reply('Here: {"say": "b", "actions": [], "done": true} ok')["done"], True)
        self.assertIsNone(parse_reply("no json here"))
        self.assertIsNone(parse_reply('{"actions": [{"name": "x"}]}'))

    async def test_restart_marks_running_chats_idle(self):
        path = os.path.join(tempfile.mkdtemp(), "jobs.sqlite3")
        service = SimpleNamespace(settings=Settings(token=TOKEN, data_dir=os.path.dirname(path)), atlas=None)
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
