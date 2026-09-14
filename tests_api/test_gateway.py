"""End-to-end: real gateway (uvicorn) + real MCP client against a fake ComfyUI.

Needs requirements-api.txt plus aiohttp; skipped otherwise.
    python -m unittest tests_api.test_gateway
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import httpx
    import uvicorn
    from aiohttp import web
    from mcp import Client

    from hawk_api import jobs as jobs_module
    from hawk_api.app import create_app
    from hawk_api.config import Settings
    from hawk_h3.script import parse_script
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"gateway dependencies missing: {exc}")

TOKEN = "test-token-0123456789abcdef"
TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REALISM = "styles/h3-realism-people-t2v-i2v-r2v.safetensors"
PLAN_SCRIPT = json.dumps({
    "style": "Cinematic.",
    "segments": [
        {"title": "One", "duration": 5, "prompt": "<Picture 1> walks in."},
        {"title": "Two", "duration": 5, "prompt": "She sits down."},
    ],
})
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeComfy:
    """Just enough of ComfyUI's HTTP + websocket API, with the real message shapes."""

    def __init__(self):
        self.inputs: dict[str, bytes] = {}
        self.outputs: dict[str, bytes] = {}
        self.history: dict[str, dict] = {}
        self.running: dict[str, asyncio.Task] = {}
        self.cancelled: set[str] = set()
        self.sockets: dict[str, web.WebSocketResponse] = {}
        self.prompts: dict[str, dict] = {}
        self.loras = [TURBO, REALISM]
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.add_routes([
            web.post("/upload/image", self.upload),
            web.post("/prompt", self.prompt),
            web.get("/history/{pid}", self.get_history),
            web.get("/queue", self.queue),
            web.post("/api/jobs/{pid}/cancel", self.cancel),
            web.get("/view", self.view),
            web.get("/models/{folder}", self.models),
            web.get("/object_info/{node}", self.object_info),
            web.get("/ws", self.ws),
        ])

    async def upload(self, request):
        post = await request.post()
        field = post["image"]
        subfolder = post.get("subfolder", "")
        self.inputs[f"{subfolder}/{field.filename}"] = field.file.read()
        return web.json_response({"name": field.filename, "subfolder": subfolder, "type": "input"})

    async def prompt(self, request):
        body = await request.json()
        pid, client, prompt = body["prompt_id"], body["client_id"], body["prompt"]
        errors = {}
        for node_id, node in prompt.items():
            key = {"LoadImage": "image", "LoadAudio": "audio", "LoadVideo": "file"}.get(node["class_type"])
            if key and node["inputs"][key] not in self.inputs:
                errors[node_id] = {"errors": [{"message": "Invalid file", "details": node["inputs"][key]}], "class_type": node["class_type"]}
        if errors:
            return web.json_response({"error": {"message": "Prompt outputs failed validation"}, "node_errors": errors}, status=400)
        self.prompts[pid] = prompt
        self.running[pid] = asyncio.create_task(self.run(pid, client, prompt))
        return web.json_response({"prompt_id": pid, "number": len(self.prompts), "node_errors": {}})

    async def send(self, client, kind, data):
        socket_ = self.sockets.get(client)
        if socket_ is not None and not socket_.closed:
            await socket_.send_json({"type": kind, "data": data})

    async def run(self, pid, client, prompt):
        outputs, messages = {}, []

        async def finish(status):
            self.history[pid] = {"prompt": [0, pid, prompt, {}, []], "outputs": outputs,
                                 "status": {"status_str": status, "completed": status == "success", "messages": messages}}
            self.running.pop(pid, None)

        await self.send(client, "execution_start", {"prompt_id": pid})
        by_class = lambda cls: [(i, n) for i, n in prompt.items() if n["class_type"] == cls]
        script = None
        for node_id, node in by_class("HawkH3StoryPlanner"):
            script = PLAN_SCRIPT
        for node_id, node in by_class("PreviewAny"):
            outputs[node_id] = {"text": [script]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        for node_id, node in by_class("HawkH3LoraStack"):
            lines = [f"{node['inputs'][f'lora_{i}']} @ {node['inputs'][f'strength_{i}']:g}"
                     for i in range(1, 5) if node["inputs"][f"lora_{i}"] != "None"]
            outputs[node_id] = {"text": ["\n".join(lines) or "no LoRAs selected"]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        for node_id, node in by_class("HawkH3Director"):
            text = node["inputs"]["script"] if isinstance(node["inputs"]["script"], str) else script
            run_name = node["inputs"]["run_name"]
            total = len(parse_script(text).segments)
            delay = 0.4 if "SLOW" in text else 0.02
            for number in range(1, total + 1):
                await asyncio.sleep(delay)
                if pid in self.cancelled:
                    data = {"prompt_id": pid, "node_id": node_id, "node_type": "HawkH3Director"}
                    messages.append(["execution_interrupted", data])
                    await self.send(None, "execution_interrupted", data)
                    return await finish("error")
                if "FAILRENDER" in text:
                    data = {"prompt_id": pid, "node_id": node_id, "node_type": "HawkH3Director", "exception_message": "CUDA out of memory"}
                    messages.append(["execution_error", data])
                    await self.send(client, "execution_error", data)
                    return await finish("error")
                self.outputs[f"hawk_h3/{run_name}/segment_{number:03d}.mp4"] = f"segment {number}".encode()
                await self.send(client, "progress", {"value": number, "max": total, "prompt_id": pid, "node": node_id})
            final = f"{run_name}_final.mp4"
            self.outputs[f"hawk_h3/{run_name}/{final}"] = b"FINAL VIDEO BYTES"
            outputs[node_id] = {"images": [{"filename": final, "subfolder": f"hawk_h3/{run_name}", "type": "output"}], "animated": [True]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        await finish("success")
        await self.send(client, "execution_success", {"prompt_id": pid})

    async def get_history(self, request):
        pid = request.match_info["pid"]
        return web.json_response({pid: self.history[pid]} if pid in self.history else {})

    async def queue(self, _request):
        return web.json_response({"queue_running": [[0, pid, {}, {}, []] for pid in self.running], "queue_pending": []})

    async def cancel(self, request):
        pid = request.match_info["pid"]
        self.cancelled.add(pid)
        return web.json_response({"cancelled": pid in self.running})

    async def view(self, request):
        key = f"{request.query.get('subfolder', '')}/{request.query['filename']}"
        if key not in self.outputs:
            return web.Response(status=404)
        return web.Response(body=self.outputs[key], content_type="video/mp4")

    async def models(self, request):
        return web.json_response(self.loras if request.match_info["folder"] == "loras" else [])

    async def object_info(self, request):
        node = request.match_info["node"]
        return web.json_response({node: {"input": {"required": {
            "sampler_name": ["COMBO", {"options": ["euler", "res_multistep"]}],
            "scheduler": [["simple", "beta"], {}],
            "aspect_ratio": [["16:9", "9:16"], {}],
            "continuity": [["off", "tail_22"], {}],
        }}}})

    async def ws(self, request):
        socket_ = web.WebSocketResponse()
        await socket_.prepare(request)
        self.sockets[request.query.get("clientId", "")] = socket_
        async for _ in socket_:
            pass
        return socket_

    def restart(self):
        """ComfyUI restart: running prompts and in-memory history are gone, files stay."""
        for task in self.running.values():
            task.cancel()
        self.running.clear()
        self.history.clear()


class Gateway(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeComfy()
        self.fake_runner = web.AppRunner(self.fake.app)
        await self.fake_runner.setup()
        comfy_port, api_port = free_port(), free_port()
        await web.TCPSite(self.fake_runner, "127.0.0.1", comfy_port).start()

        self.data_dir = tempfile.mkdtemp(prefix="hawk_api_test_")
        self.base = f"http://127.0.0.1:{api_port}"
        settings = Settings(
            token=TOKEN, comfy_url=f"http://127.0.0.1:{comfy_port}", public_base_url=self.base,
            data_dir=self.data_dir, lora_cache_seconds=0, reconcile_seconds=0.3,
        )
        self._lost_after = jobs_module.LOST_AFTER_SECONDS
        jobs_module.LOST_AFTER_SECONDS = 0.5
        self.server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=api_port, log_level="warning"))
        self.server_task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.02)
        self.http = httpx.AsyncClient(base_url=self.base, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10)
        await asyncio.sleep(0.2)  # websocket listener connects

    async def asyncTearDown(self):
        jobs_module.LOST_AFTER_SECONDS = self._lost_after
        await self.http.aclose()
        self.server.should_exit = True
        await self.server_task
        await self.fake_runner.cleanup()

    async def upload_picture(self) -> str:
        response = await self.http.post("/v1/assets", files={"files": ("face.png", PNG, "image/png")})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["assets"][0]["id"]

    async def wait(self, job_id, statuses=("done", "failed", "cancelled"), timeout=10.0, check=None):
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            job = (await self.http.get(f"/v1/jobs/{job_id}")).json()
            if job["status"] in statuses and (check is None or check(job)):
                return job
            if asyncio.get_event_loop().time() > deadline:
                self.fail(f"job {job_id} stuck: {job}")
            await asyncio.sleep(0.05)

    async def test_auth(self):
        async with httpx.AsyncClient(base_url=self.base) as anonymous:
            self.assertEqual((await anonymous.get("/healthz")).json()["loras"], "ok")
            self.assertEqual((await anonymous.get("/openapi.json")).status_code, 200)
            self.assertEqual((await anonymous.get("/v1/options")).status_code, 401)
            self.assertEqual((await anonymous.get("/v1/jobs")).status_code, 401)
            self.assertEqual((await anonymous.get("/v1/jobs", headers={"Authorization": "Bearer wrong"})).status_code, 401)
            self.assertEqual((await anonymous.get(f"/t/{TOKEN}/v1/jobs")).status_code, 200)
            self.assertEqual((await anonymous.get("/t/wrong-token/v1/jobs")).status_code, 401)
            self.assertEqual((await anonymous.post("/mcp", json={})).status_code, 401)

    async def test_plan_then_render_with_loras_and_downloads(self):
        asset = await self.upload_picture()
        references = [{"asset_id": asset, "role": "picture", "label": "her face"}]

        plan = (await self.http.post("/v1/plans", json={"story": "A walk", "references": references, "segment_count": 2})).json()
        self.assertEqual(plan["status"], "planning")
        plan = await self.wait(plan["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertEqual(json.loads(plan["script"])["segments"][0]["title"], "One")

        response = await self.http.post("/v1/videos", json={
            "plan_job_id": plan["id"],
            "settings": {"megapixels": 0.4, "loras": [{"name": "realism", "strength": 0.7}]},
        })
        self.assertEqual(response.status_code, 202, response.text)
        render = response.json()
        self.assertEqual((render["steps"], [l["file"] for l in render["loras"]]), (8, [TURBO, REALISM]))
        self.assertIn("turbo", render["steps_reason"])

        render = await self.wait(render["id"])
        self.assertEqual(render["status"], "done", render)
        self.assertEqual(render["progress"], {"segments_done": 2, "segments_total": 2})
        self.assertEqual(render["loras_applied"], [{"file": TURBO, "strength": 1.0}, {"file": REALISM, "strength": 0.7}])
        self.assertEqual(render["warnings"], [])

        async with httpx.AsyncClient() as browser:  # signed links need no token
            self.assertEqual((await browser.get(render["video_url"])).content, b"FINAL VIDEO BYTES")
            self.assertEqual((await browser.get(render["segment_urls"][1])).content, b"segment 2")
            self.assertEqual((await browser.get(render["video_url"].replace("sig=", "sig=0"))).status_code, 401)

        graph = self.fake.prompts[render["id"]]
        loader = next(n for n in graph.values() if n["class_type"] == "HawkH3ModelLoader")
        self.assertEqual(loader["inputs"]["lora_stack"], "")
        refs = next(n for n in graph.values() if n["class_type"] == "HawkH3References")
        self.assertIn("pictures.picture_0", refs["inputs"])

    async def test_one_call_story(self):
        asset = await self.upload_picture()
        job = (await self.http.post("/v1/videos", json={
            "references": [{"asset_id": asset, "role": "picture"}],
            "story": {"story": "A walk", "segment_count": 2},
        })).json()
        self.assertEqual(job["status"], "planning")
        job = await self.wait(job["id"])
        self.assertEqual((job["status"], job["progress"]["segments_total"]), ("done", 2), job)
        self.assertIn("segments", job["script"])

    async def test_validation_errors_queue_nothing(self):
        asset = await self.upload_picture()
        refs = [{"asset_id": asset, "role": "picture"}]
        bad_tag = await self.http.post("/v1/videos", json={"references": refs, "script": "<Picture 2> smiles"})
        self.assertEqual(bad_tag.status_code, 422)
        self.assertIn("only 1 picture", bad_tag.json()["error"])
        bad_lora = await self.http.post("/v1/videos", json={"references": refs, "script": "A", "settings": {"loras": [{"name": "realsim-people"}]}})
        self.assertEqual(bad_lora.status_code, 422)
        self.assertIn(REALISM, bad_lora.json()["details"]["suggestions"])
        wrong_kind = await self.http.post("/v1/videos", json={"references": [{"asset_id": asset, "role": "audio"}], "script": "A"})
        self.assertEqual(wrong_kind.status_code, 422)
        both = await self.http.post("/v1/videos", json={"script": "A", "plan_job_id": "x"})
        self.assertEqual(both.status_code, 422)
        self.assertEqual(self.fake.prompts, {})

        self.fake.loras = [REALISM]  # turbo LoRA removed from the pod
        refused = await self.http.post("/v1/videos", json={"script": "A"})
        self.assertEqual(refused.status_code, 422)
        self.assertIn("Required default LoRA", refused.json()["error"])
        health = (await self.http.get("/healthz")).json()
        self.assertEqual(health["loras"], "degraded")
        allowed = (await self.http.post("/v1/videos", json={"script": "A", "settings": {"use_default_loras": False}})).json()
        self.assertEqual((allowed["steps"], allowed["loras"]), (30, []))

    async def test_failure_restart_retry_and_cancel(self):
        failed = (await self.http.post("/v1/videos", json={"script": "FAILRENDER"})).json()
        failed = await self.wait(failed["id"])
        self.assertEqual((failed["status"], failed["resumable"]), ("failed", True))
        self.assertIn("CUDA out of memory", failed["error"])

        slow = (await self.http.post("/v1/videos", json={"script": "SLOW one\n---\nSLOW two\n---\nSLOW three"})).json()
        await self.wait(slow["id"], statuses=("rendering",), check=lambda j: j["progress"]["segments_done"] >= 1)
        self.fake.restart()
        lost = await self.wait(slow["id"], timeout=5)
        self.assertEqual((lost["status"], lost["resumable"]), ("failed", True), lost)
        self.assertEqual(len(lost["segment_urls"]), lost["progress"]["segments_done"])

        retried = (await self.http.post(f"/v1/jobs/{slow['id']}/retry")).json()
        self.assertEqual(retried["run_name"], lost["run_name"])
        prompts = [pid for pid, graph in self.fake.prompts.items()
                   if any(n["inputs"].get("run_name") == lost["run_name"] for n in graph.values())]
        self.assertEqual(len(prompts), 2)
        seeds = {n["inputs"]["seed"] for pid in prompts for n in self.fake.prompts[pid].values() if n["class_type"] == "HawkH3Director"}
        self.assertEqual(len(seeds), 1, "a retry must keep the seed so segments resume")

        cancel = (await self.http.post(f"/v1/jobs/{slow['id']}/cancel")).json()
        self.assertEqual(cancel["status"], "cancelled")
        self.assertEqual((await self.http.post(f"/v1/jobs/{slow['id']}/cancel")).status_code, 409)

    async def test_upload_page_and_mcp(self):
        async with Client(f"{self.base}/t/{TOKEN}/mcp") as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            self.assertTrue({"plan_film", "render_film", "get_job", "upload_page_link", "list_options", "retry_job"} <= names)

            async def call(name, args=None):
                result = await client.call_tool(name, args or {})
                self.assertFalse(result.is_error, result)
                if result.structured_content is not None:
                    return result.structured_content
                return json.loads(result.content[0].text)

            link = (await call("upload_page_link"))["upload_url"]
            async with httpx.AsyncClient() as browser:
                self.assertIn("Upload references", (await browser.get(link)).text)
                uploaded = await browser.post(link, files={"files": ("pose.png", PNG, "image/png")})
                self.assertEqual(uploaded.status_code, 201, uploaded.text)
                asset = uploaded.json()["assets"][0]["id"]

            options = await call("list_options")
            self.assertIn(TURBO, options["available_loras"])
            self.assertTrue(options["default_loras"][0]["present"])

            job = await call("render_film", {"references": [{"asset_id": asset, "role": "pose"}], "script": "She ends in <Pose 1>."})
            done = await self.wait(job["id"])
            self.assertEqual(done["status"], "done")
            polled = await call("get_job", {"job_id": job["id"]})
            self.assertTrue(polled["video_url"].startswith(self.base))

            bad = await client.call_tool("render_film", {"script": "<Picture 3>"})
            self.assertTrue(bad.is_error)


if __name__ == "__main__":
    unittest.main()
