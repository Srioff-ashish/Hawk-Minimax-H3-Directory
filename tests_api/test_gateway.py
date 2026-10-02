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
MOTION = "H3_Motion_BoosterV2.safetensors"  # shipped defaults, added to every render
AIO = "HMNSFW_AIO_V25.safetensors"
PLAN_SCRIPT = json.dumps({
    "style": "Cinematic.",
    "segments": [
        {"title": "One", "duration": 5, "prompt": "<Picture 1> walks in."},
        {"title": "Two", "duration": 5, "prompt": "She sits down."},
    ],
})
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
UNET_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
UNET_BF16 = "h3/minimax_h3_ref2va_pruned_bf16.safetensors"
UNET_FL2VA = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
CLIP_BF16 = "qwen3vl_32b_minimax_h3_bf16.safetensors"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeComfy:
    """Just enough of ComfyUI's HTTP + websocket API, with the real message shapes.
    Like ComfyUI, one worker runs prompts one at a time; the rest wait in the queue."""

    def __init__(self):
        self.inputs: dict[str, bytes] = {}
        self.outputs: dict[str, bytes] = {}
        self.history: dict[str, dict] = {}
        self.running: dict[str, asyncio.Task] = {}
        self.pending: list[tuple[int, str, str, dict]] = []
        self.worker: asyncio.Task | None = None
        self.cancelled: set[str] = set()
        self.sockets: dict[str, web.WebSocketResponse] = {}
        self.prompts: dict[str, dict] = {}
        self.stack_cache: set[str] = set()
        self.output_dir: str | None = None  # also write outputs to disk, like real ComfyUI
        self.final_bytes = b"FINAL VIDEO BYTES"
        self.missing_nodes: set[str] = set()  # custom node classes this ComfyUI doesn't have
        self.fail_image = None  # callable(prompt) -> error message for an image graph, or None
        self.log_lines = ["Starting server\n", "[Hawk H3] masked attention skips cuDNN on this Blackwell GPU\n"]
        self.model_files = {
            "loras": [TURBO, REALISM, MOTION, AIO],
            "diffusion_models": [UNET_FL2VA, UNET_INT8, UNET_BF16],
            "text_encoders": [CLIP_INT8, CLIP_BF16, "umt5_xxl.safetensors"],
        }
        #: Planning is an LLM call the gateway makes itself now, so this fake answers /v1/models and
        #: /v1/chat/completions too. One fake server for both keeps the wiring in these tests to one port.
        self.chat_requests: list[dict] = []
        self.llm_models_data = [
            {"id": "xai/grok-4.6", "name": "Grok 4.6", "input_modalities": ["text", "image"],
             "output_modalities": ["text"], "pricing": {"prompt": "0.000002", "completion": "0.000006"}},
            {"id": "xai/grok-4.3", "name": "Grok 4.3", "input_modalities": ["text", "image"],
             "output_modalities": ["text"], "pricing": {"prompt": "0.00000125", "completion": "0.0000025"}},
        ]
        self.chat_reply = PLAN_SCRIPT
        self.chat_status = 200
        self.chat_finish_reason = "stop"  # "length" is a reply the model ran out of room to finish
        self.chat_status_by_model: dict[str, int] = {}  # one model rate-limited while the rest answer
        self.app = web.Application(client_max_size=64 * 1024 * 1024)
        self.app.add_routes([
            web.get("/v1/models", self.llm_models),
            web.post("/v1/chat/completions", self.chat),
            web.post("/upload/image", self.upload),
            web.post("/prompt", self.prompt),
            web.get("/history/{pid}", self.get_history),
            web.get("/queue", self.queue),
            web.post("/api/jobs/{pid}/cancel", self.cancel),
            web.get("/view", self.view),
            web.get("/models/{folder}", self.models),
            web.get("/object_info/{node}", self.object_info),
            web.get("/internal/logs/raw", self.logs),
            web.get("/ws", self.ws),
        ])

    async def llm_models(self, _request):
        return web.json_response({"data": self.llm_models_data})

    async def chat(self, request):
        body = await request.json()
        self.chat_requests.append(body)
        status = self.chat_status_by_model.get(body.get("model"), self.chat_status)
        if status != 200:
            return web.json_response({"error": {"message": "no"}}, status=status)
        return web.json_response({"choices": [{"message": {"content": self.chat_reply},
                                               "finish_reason": self.chat_finish_reason}],
                                  "usage": {"prompt_tokens": 900, "completion_tokens": 120}})

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
        self.pending.append((len(self.prompts), pid, client, prompt))
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self.work())
        return web.json_response({"prompt_id": pid, "number": len(self.prompts), "node_errors": {}})

    async def work(self):
        while self.pending:
            _, pid, client, prompt = self.pending.pop(0)
            self.running[pid] = asyncio.create_task(self.run(pid, client, prompt))
            await self.running[pid]

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
        for node_id, node in by_class("SaveImage"):  # an image graph (local Krea 2)
            await asyncio.sleep(0.05)
            failure = self.fail_image(prompt) if self.fail_image else None
            if failure:
                messages.append(["execution_error", {"prompt_id": pid, "node_id": "7", "node_type": "KSampler", "exception_message": failure}])
                return await finish("error")
            batch = next(n for _, n in by_class("EmptyLatentImage") + by_class("EmptySD3LatentImage"))["inputs"]["batch_size"]
            prefix = node["inputs"]["filename_prefix"]
            folder, stem = prefix.rsplit("/", 1)
            images = []
            for number in range(1, batch + 1):
                name = f"{stem}_{len(self.outputs) + 1:05d}_.png"
                self.save_output(f"{folder}/{name}", PNG + name.encode())  # each output differs, like real images
                images.append({"filename": name, "subfolder": folder, "type": "output"})
            outputs[node_id] = {"images": images}
            return await finish("success")
        script = None
        for node_id, node in by_class("HawkH3StoryPlanner"):
            script = PLAN_SCRIPT
        directors = {node_id for node_id, _ in by_class("HawkH3Director")}
        for node_id, node in by_class("PreviewAny"):
            if node["inputs"]["source"][0] in directors:
                continue  # the Director's prompts output is reported after it runs
            outputs[node_id] = {"text": [script]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        for node_id, node in by_class("HawkH3LoraStack"):
            signature = json.dumps({k: v for k, v in node["inputs"].items() if not isinstance(v, list)}, sort_keys=True)
            if signature in self.stack_cache:  # like ComfyUI: a cached, non-output node reports nothing
                continue
            self.stack_cache.add(signature)
            lines = [f"{node['inputs'][f'lora_{i}']} @ {node['inputs'][f'strength_{i}']:g}"
                     for i in range(1, 5) if node["inputs"][f"lora_{i}"] != "None"]
            outputs[node_id] = {"text": ["\n".join(lines) or "no LoRAs selected"]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        for node_id, node in by_class("HawkH3Director"):
            text = node["inputs"]["script"] if isinstance(node["inputs"]["script"], str) else script
            run_name = node["inputs"]["run_name"]
            total = len(parse_script(text).segments)
            delay = 0.4 if "SLOW" in text else 0.02
            await self.send(client, "hawk_h3.segment", {"prompt_id": pid, "done": 0, "total": total, "title": "", "cached": False})
            for number in range(1, total + 1):
                for step in range(1, 6):  # the sampler's step bar reports under the Director node
                    await asyncio.sleep(delay / 5)
                    await self.send(client, "progress", {"value": step, "max": 5, "prompt_id": pid, "node": node_id})
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
                self.save_output(f"hawk_h3/{run_name}/segment_{number:03d}.mp4", f"segment {number}".encode())
                await self.send(client, "hawk_h3.segment", {"prompt_id": pid, "done": number, "total": total, "title": f"seg {number}", "cached": False})
                await self.send(client, "progress", {"value": number, "max": total, "prompt_id": pid, "node": node_id})
            for preview_id, preview in by_class("PreviewAny"):
                if preview["inputs"]["source"][0] == node_id:
                    outputs[preview_id] = {"text": ["### Segment 1\n" + text.replace("<Pose 1>", "<Picture 2>")]}
                    await self.send(client, "executed", {"node": preview_id, "output": outputs[preview_id], "prompt_id": pid})
            final = f"{run_name}_final.mp4"
            self.save_output(f"hawk_h3/{run_name}/{final}", self.final_bytes)
            outputs[node_id] = {"images": [{"filename": final, "subfolder": f"hawk_h3/{run_name}", "type": "output"}], "animated": [True]}
            await self.send(client, "executed", {"node": node_id, "output": outputs[node_id], "prompt_id": pid})
        await finish("success")
        await self.send(client, "execution_success", {"prompt_id": pid})

    def save_output(self, key: str, data: bytes) -> None:
        self.outputs[key] = data
        if self.output_dir:
            path = os.path.join(self.output_dir, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(data)

    async def get_history(self, request):
        pid = request.match_info["pid"]
        return web.json_response({pid: self.history[pid]} if pid in self.history else {})

    async def queue(self, _request):
        # Reversed on purpose: ComfyUI's pending list is a heap, not in run order.
        return web.json_response({
            "queue_running": [[0, pid, {}, {}, []] for pid in self.running],
            "queue_pending": [[number, pid, {}, {}, []] for number, pid, _, _ in reversed(self.pending)],
        })

    async def cancel(self, request):
        pid = request.match_info["pid"]
        waiting = [item for item in self.pending if item[1] == pid]
        self.pending = [item for item in self.pending if item[1] != pid]
        self.cancelled.add(pid)
        return web.json_response({"cancelled": pid in self.running or bool(waiting)})

    async def view(self, request):
        key = f"{request.query.get('subfolder', '')}/{request.query['filename']}"
        if request.query.get("type") == "input":
            return web.Response(body=self.inputs[key], content_type="application/octet-stream") if key in self.inputs else web.Response(status=404)
        if key not in self.outputs:
            return web.Response(status=404)
        return web.Response(body=self.outputs[key], content_type="video/mp4")

    async def models(self, request):
        return web.json_response(self.model_files.get(request.match_info["folder"], []))

    async def logs(self, _request):
        return web.json_response({"entries": [{"t": "", "m": line} for line in self.log_lines], "size": {}})

    async def object_info(self, request):
        node = request.match_info["node"]
        if node in self.missing_nodes:
            return web.json_response({})
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
        """ComfyUI restart: running and queued prompts and in-memory history are gone, files stay."""
        if self.worker is not None:
            self.worker.cancel()
        for task in self.running.values():
            task.cancel()
        self.running.clear()
        self.pending.clear()
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
            # The planner is a gateway-side call now, so these tests need a chat service; the ComfyUI fake
            # answers for it. Before, the plan came out of the fake's HawkH3StoryPlanner node.
            atlas_url=f"http://127.0.0.1:{comfy_port}/v1", atlas_api_key="test-key",
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
            for studio in ("/studio", f"/t/{TOKEN}/studio"):
                page = await anonymous.get(studio)
                self.assertEqual(page.status_code, 200, studio)
                self.assertIn("Hawk H3 Studio", page.text)
            self.assertEqual((await anonymous.get("/v1/options")).status_code, 401)
            self.assertEqual((await anonymous.get("/v1/jobs")).status_code, 401)
            self.assertEqual((await anonymous.get("/v1/jobs", headers={"Authorization": "Bearer wrong"})).status_code, 401)
            self.assertEqual((await anonymous.get(f"/t/{TOKEN}/v1/jobs")).status_code, 200)
            self.assertEqual((await anonymous.get("/t/wrong-token/v1/jobs")).status_code, 401)
            denied = await anonymous.post("/mcp", json={})
            self.assertEqual(denied.status_code, 401)
            self.assertNotIn("www-authenticate", denied.headers, "would make connector apps ask for OAuth")
            for probe in ("/.well-known/oauth-protected-resource", f"/.well-known/oauth-protected-resource/t/{TOKEN}/mcp",
                          "/.well-known/oauth-authorization-server"):
                self.assertEqual((await anonymous.get(probe)).status_code, 404, probe)

    async def test_the_planner_calls_the_chosen_provider_and_not_comfyui(self):
        # Planning used to happen inside ComfyUI, where the graph hardcoded Atlas's URL and an empty key, so the
        # node fell back to ATLAS_API_KEY in ComfyUI's own environment: choosing a provider in Studio set the
        # planner's model and silently left its provider alone, and its tokens were never counted anywhere.
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertTrue(self.fake.chat_requests, "the gateway should have made the planner call itself")
        self.assertEqual(self.fake.prompts, {}, "a plan should no longer submit any graph to ComfyUI")

        call = self.fake.chat_requests[-1]
        self.assertEqual(call["model"], "xai/grok-4.6",
                         "the planner leads with 4.6; 4.3 writes weaker plans and is only the fallback")
        self.assertIn("BRIEF:", json.dumps(call["messages"][-1]["content"]))
        self.assertNotIn("test-key", json.dumps(call["messages"]), "the key belongs in a header, never in the prompt")

    async def test_a_sexual_brief_reaches_the_planner_with_the_position_guide(self):
        # H3 knows no position names: a plan that says "standing full nelson" and nothing more gets a guess.
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "Standing full nelson, anal"})).json()["id"])
        brief = json.dumps(self.fake.chat_requests[-1]["messages"][-1]["content"])
        self.assertIn("POSITION GUIDE", brief, plan)
        self.assertIn("Standing full nelson", brief)
        self.assertNotIn("Butterfly", brief, "only the position the brief names, not the whole guide")
        await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertNotIn("POSITION GUIDE", json.dumps(self.fake.chat_requests[-1]["messages"][-1]["content"]))

    async def test_a_plan_records_what_its_llm_call_cost(self):
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        usage = plan.get("usage") or {}
        self.assertEqual((usage.get("prompt_tokens"), usage.get("completion_tokens")), (900, 120),
                         "the planner is a paid call and used to be billed invisibly inside ComfyUI")
        self.assertEqual(plan.get("planner_model"), "xai/grok-4.6")

    async def test_the_planner_falls_back_to_a_model_the_provider_lists(self):
        # The provider drops 4.6 -- a real case when switching provider or account.
        self.fake.llm_models_data = [
            {"id": "xai/grok-4.3", "name": "Grok 4.3", "input_modalities": ["text", "image"],
             "output_modalities": ["text"], "pricing": {"prompt": "0.00000125", "completion": "0.0000025"}}]
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertEqual(self.fake.chat_requests[-1]["model"], "xai/grok-4.3",
                         "the chain's second id should run rather than the feature failing")

    async def test_a_plan_that_the_provider_refuses_fails_the_job_rather_than_hanging(self):
        self.fake.chat_status = 503
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertEqual(plan["status"], "failed", plan)
        self.assertTrue(plan["error"], "a failed plan must say why: the caller is holding a job id and polling")
        retried = (await self.http.post(f"/v1/jobs/{plan['id']}/retry")).json()
        self.assertIn(retried["status"], ("planning", "queued"), "a plan retries by running the call again")

    async def test_a_rate_limited_planner_hands_the_plan_to_the_next_model_in_the_chain(self):
        # The chain used to be resolved to one id up front, so a model rate-limited upstream failed the plan
        # with a 429 while the next candidate was free.
        self.fake.chat_status_by_model = {"xai/grok-4.6": 429}
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertEqual(plan.get("planner_model"), "xai/grok-4.3")
        self.assertEqual([c["model"] for c in self.fake.chat_requests][-1], "xai/grok-4.3")

    async def test_a_plan_with_reference_photos_uses_a_model_that_can_read_them(self):
        # The planner sends references as image parts, so with photos attached this call is multimodal. The
        # role carried no image requirement, so the head of the chain won whatever it could do, and every
        # plan with references failed with the provider's own 404: "No endpoints found that support image
        # input". The requirement belongs to the call, not to the role -- hence the second half below.
        self.fake.llm_models_data = [
            {"id": "xai/grok-4.6", "name": "Grok 4.6", "input_modalities": ["text"],
             "output_modalities": ["text"], "pricing": {"prompt": "0.000002", "completion": "0.000006"}},
            {"id": "xai/grok-4.3", "name": "Grok 4.3", "input_modalities": ["text", "image"],
             "output_modalities": ["text"], "pricing": {"prompt": "0.00000125", "completion": "0.0000025"}}]
        asset = await self.upload_picture()

        plan = await self.wait((await self.http.post("/v1/plans", json={
            "story": "A walk", "references": [{"asset_id": asset, "role": "picture", "label": "her face"}]})).json()["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertEqual(self.fake.chat_requests[-1]["model"], "xai/grok-4.3",
                         "a plan carrying photos must go to a model the provider serves image input for")

        plain = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])
        self.assertEqual(plain["status"], "done", plain)
        self.assertEqual(self.fake.chat_requests[-1]["model"], "xai/grok-4.6",
                         "with no photos the cheaper head of the chain should still be used")

    async def test_a_plan_cut_off_mid_sentence_fails_rather_than_arriving_as_done(self):
        # The only check on this path was that the reply was not empty, and a script that stops mid-sentence
        # is not empty: the job said done with no error, Studio drew a half-written plan, and the first thing
        # to actually parse it was the render, minutes later.
        self.fake.chat_finish_reason = "length"
        self.fake.chat_reply = PLAN_SCRIPT[: len(PLAN_SCRIPT) // 2]
        plan = await self.wait((await self.http.post("/v1/plans", json={"story": "A walk"})).json()["id"])

        self.assertEqual(plan["status"], "failed", "a plan that cannot be parsed is not a finished plan")
        self.assertFalse(plan.get("script"), "a half-written script must not be offered as the plan")
        self.assertTrue(plan["error"], "a failed plan must say why: the caller is holding a job id and polling")
        self.assertGreater(len(self.fake.chat_requests), 1,
                           "being cut off is worth asking again for, since nothing else can fix it")

    async def test_plan_then_render_with_loras_and_downloads(self):
        asset = await self.upload_picture()
        references = [{"asset_id": asset, "role": "picture", "label": "her face"}]

        plan = (await self.http.post("/v1/plans", json={"story": "A walk", "references": references, "segment_count": 2})).json()
        self.assertIn(plan["status"], ("queued", "planning"))
        plan = await self.wait(plan["id"])
        self.assertEqual(plan["status"], "done", plan)
        self.assertEqual(json.loads(plan["script"])["segments"][0]["title"], "One")

        response = await self.http.post("/v1/videos", json={
            "plan_job_id": plan["id"],
            "settings": {"megapixels": 0.4, "loras": [{"name": "realism", "strength": 0.7}]},
        })
        self.assertEqual(response.status_code, 202, response.text)
        render = response.json()
        self.assertEqual((render["steps"], [l["file"] for l in render["loras"]]), (8, [TURBO, MOTION, AIO, REALISM]))
        self.assertIn("turbo", render["steps_reason"])

        render = await self.wait(render["id"])
        self.assertEqual(render["status"], "done", render)
        progress = render["progress"]
        self.assertEqual((progress["segments_done"], progress["segments_total"]), (2, 2), progress)
        self.assertEqual(progress["steps_total"], 5, "sampler steps are reported separately from segments")
        self.assertEqual(render["loras_applied"], [{"file": TURBO, "strength": 1.0}, {"file": MOTION, "strength": 1.0},
                                                   {"file": AIO, "strength": 0.8}, {"file": REALISM, "strength": 0.7}])
        self.assertEqual(render["warnings"], [])

        # Same LoRAs again: ComfyUI serves the LoRA Stack from its cache and sends no report.
        again = await self.wait((await self.http.post("/v1/videos", json={
            "script": "Another walk", "settings": {"loras": [{"name": "realism", "strength": 0.7}]},
        })).json()["id"])
        self.assertEqual((again["status"], again["warnings"]), ("done", []), again)
        self.assertEqual(again["loras_applied"], [{"file": TURBO, "strength": 1.0}, {"file": MOTION, "strength": 1.0},
                                                  {"file": AIO, "strength": 0.8}, {"file": REALISM, "strength": 0.7}])

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
        self.assertIn(job["status"], ("queued", "planning", "rendering"))  # the fake plans instantly
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

        self.fake.model_files["loras"] = [REALISM]  # turbo LoRA removed from the pod
        refused = await self.http.post("/v1/videos", json={"script": "A"})
        self.assertEqual(refused.status_code, 422)
        self.assertIn("Required default LoRA", refused.json()["error"])
        health = (await self.http.get("/healthz")).json()
        self.assertEqual(health["loras"], "degraded")
        allowed = (await self.http.post("/v1/videos", json={"script": "A", "settings": {"use_default_loras": False}})).json()
        self.assertEqual((allowed["steps"], allowed["loras"]), (30, []))

    async def test_a_turbo_hybrid_base_renders_at_turbo_steps_without_the_turbo_lora(self):
        # 10Eros-Max beta5: an H3 fl2va/ref2va hybrid with no "ref2va" in its name, and turbo built in.
        eros = "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors"
        self.fake.model_files["diffusion_models"].append(eros)
        options = (await self.http.get("/v1/options")).json()
        self.assertIn(eros, options["diffusion_models"], "a hybrid reads references, so the Director offers it")
        self.assertNotIn(UNET_FL2VA, options["diffusion_models"], "plain fl2va still does not")

        job = (await self.http.post("/v1/videos", json={"script": "A", "settings": {"unet_name": "TURBO-hybrid_beta5"}})).json()
        self.assertEqual(job["unet_name"], eros, job)
        self.assertEqual(job["steps"], 8, job)
        self.assertNotIn(TURBO, [l["file"] for l in job["loras"]], "the turbo LoRA must not be stacked on a turbo base")
        self.assertTrue(any("turbo built in" in w for w in job.get("warnings") or []), job.get("warnings"))
        loader = next(n for n in self.fake.prompts[job["id"]].values() if n["class_type"] == "HawkH3ModelLoader")
        self.assertEqual(loader["inputs"]["unet_name"], eros)
        self.assertEqual((await self.wait(job["id"]))["status"], "done")

    async def test_a_default_encoder_moved_into_a_subfolder_loads_by_its_subfolder_path(self):
        # The configured default is a bare name; sorted into text_encoders/h3/ ComfyUI calls it "h3/...".
        default = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        self.fake.model_files["text_encoders"] = [f"h3/{default}", CLIP_BF16]
        job = (await self.http.post("/v1/videos", json={"script": "A"})).json()
        self.assertEqual(job["clip_name"], f"h3/{default}", job)
        loader = next(n for n in self.fake.prompts[job["id"]].values() if n["class_type"] == "HawkH3ModelLoader")
        self.assertEqual(loader["inputs"]["clip_name"], f"h3/{default}")
        self.assertEqual((await self.wait(job["id"]))["status"], "done")

    async def test_model_choice(self):
        options = (await self.http.get("/v1/options")).json()
        self.assertEqual((options["diffusion_models"], options["text_encoders"]), ([UNET_INT8, UNET_BF16], [CLIP_INT8, CLIP_BF16]))
        self.assertEqual(options["default_unet"], UNET_INT8)

        job = (await self.http.post("/v1/videos", json={"script": "A", "settings": {"unet_name": "bf16", "clip_name": "qwen3vl_32b_minimax_h3_bf16"}})).json()
        self.assertEqual((job["unet_name"], job["clip_name"]), (UNET_BF16, CLIP_BF16), job)
        loader = next(n for n in self.fake.prompts[job["id"]].values() if n["class_type"] == "HawkH3ModelLoader")
        self.assertEqual((loader["inputs"]["unet_name"], loader["inputs"]["clip_name"]), (UNET_BF16, CLIP_BF16))
        self.assertEqual((await self.wait(job["id"]))["status"], "done")

        default = (await self.http.post("/v1/videos", json={"script": "A"})).json()
        self.assertEqual(default["unet_name"], UNET_INT8)
        fl2va = await self.http.post("/v1/videos", json={"script": "A", "settings": {"unet_name": "fl2va_pruned_int8"}})
        self.assertEqual(fl2va.status_code, 422)
        self.assertIn("needs a ref2va model", fl2va.json()["error"])
        encoder = await self.http.post("/v1/videos", json={"script": "A", "settings": {"clip_name": "umt5"}})
        self.assertEqual(encoder.status_code, 422)
        self.assertIn("Text encoder 'umt5'", encoder.json()["error"])

        # The image engines keep their own Qwen3-VL encoders in this folder: Krea 2 loads a 4B and Qwen
        # Image 2.1 an 8B, both narrower than the 32B a render needs (4096 against 5120). Handing one to a
        # render does not fail in the loader, it fails in the first matmul with "mat1 and mat2 shapes
        # cannot be multiplied", which names no file and no reason. They must not be offered or accepted.
        self.fake.model_files["text_encoders"] += ["qwen3vl_8b_bf16.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"]
        listed = (await self.http.get("/v1/options")).json()["text_encoders"]
        self.assertEqual(listed, [CLIP_INT8, CLIP_BF16], "an image engine's encoder is not a video encoder")
        for name in ("qwen3vl_8b_bf16", "qwen3vl_4b_fp8_scaled"):
            wrong = await self.http.post("/v1/videos", json={"script": "A", "settings": {"clip_name": name}})
            self.assertEqual(wrong.status_code, 422, f"{name} should be refused by name, not by matmul")
        # "bf16" is the trap: with the 8B present it is the shorter match, and on a pod without the 32B
        # bf16 encoder it is the only one, so the render would silently load the image engine's file.
        by_precision = await self.http.post("/v1/videos", json={"script": "A", "settings": {"clip_name": "bf16"}})
        self.assertEqual(by_precision.status_code, 202, by_precision.text)
        self.assertEqual(by_precision.json()["clip_name"], CLIP_BF16, "bf16 means the H3 encoder, not the 8B")
        await self.wait(by_precision.json()["id"])
        await self.wait(default["id"])

    async def test_music_bed(self):
        uploaded = await self.http.post("/v1/assets", files={"files": ("beat.mp3", b"ID3" + b"0" * 64, "audio/mpeg")})
        music = uploaded.json()["assets"][0]["id"]
        picture = await self.upload_picture()
        job = (await self.http.post("/v1/videos", json={"script": "A", "settings": {"music_asset_id": music, "music_volume_db": -8}})).json()
        self.assertEqual(job["music_asset_id"], music, job)
        graph = self.fake.prompts[job["id"]]
        director = next(n for n in graph.values() if n["class_type"] == "HawkH3Director")
        loader = graph[director["inputs"]["music"][0]]
        self.assertEqual((loader["class_type"], director["inputs"]["music_volume_db"]), ("LoadAudio", -8))
        self.assertTrue(loader["inputs"]["audio"].endswith("beat.mp3"))
        self.assertEqual((await self.wait(job["id"]))["status"], "done")

        wrong = await self.http.post("/v1/videos", json={"script": "A", "settings": {"music_asset_id": picture}})
        self.assertEqual(wrong.status_code, 422)
        self.assertIn("must be an audio file", wrong.json()["error"])
        missing = await self.http.post("/v1/videos", json={"script": "A", "settings": {"music_asset_id": "nope"}})
        self.assertEqual(missing.status_code, 422)

    async def test_second_job_waits_in_queue(self):
        first = (await self.http.post("/v1/videos", json={"script": "SLOW one\n---\nSLOW two\n---\nSLOW three"})).json()
        await self.wait(first["id"], statuses=("rendering",))
        second = (await self.http.post("/v1/videos", json={"script": "Quick one"})).json()
        third = (await self.http.post("/v1/videos", json={"script": "Quick two"})).json()
        self.assertEqual((second["status"], second["queue_position"]), ("queued", 1), second)
        self.assertEqual((third["status"], third["queue_position"]), ("queued", 2), third)
        self.assertEqual(second["progress"]["segments_done"], 0)

        first = await self.wait(first["id"])
        promoted = await self.wait(third["id"], statuses=("queued", "rendering", "done"), check=lambda j: j["queue_position"] != 2)
        self.assertIn(promoted["queue_position"], (1, None))
        second, third = await self.wait(second["id"]), await self.wait(third["id"])
        self.assertEqual([j["status"] for j in (first, second, third)], ["done", "done", "done"])
        self.assertIsNone(third["queue_position"])

    async def test_failure_restart_retry_and_cancel(self):
        failed = (await self.http.post("/v1/videos", json={"script": "FAILRENDER"})).json()
        failed = await self.wait(failed["id"])
        self.assertEqual((failed["status"], failed["resumable"]), ("failed", True))
        self.assertIn("CUDA out of memory", failed["error"])

        slow = (await self.http.post("/v1/videos", json={"script": "SLOW one\n---\nSLOW two\n---\nSLOW three"})).json()
        mid = await self.wait(slow["id"], statuses=("rendering",), check=lambda j: j["progress"]["segments_done"] >= 1)
        self.assertEqual(mid["progress"]["segments_total"], 3, "step progress must not overwrite the segment count")
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

            guide = await call("pose_guide", {"pose": "standing full nelson"})
            self.assertEqual([p["key"] for p in guide["positions"]], ["full-nelson-standing"])
            self.assertIn("feet off the floor", guide["positions"][0]["guide"])

            link = (await call("upload_page_link"))["upload_url"]
            async with httpx.AsyncClient() as browser:
                self.assertIn("Upload references", (await browser.get(link)).text)
                uploaded = await browser.post(link, files={"files": ("pose.png", PNG, "image/png")})
                self.assertEqual(uploaded.status_code, 201, uploaded.text)
                asset = uploaded.json()["assets"][0]["id"]

            options = await call("list_options")
            self.assertIn(TURBO, options["available_loras"])
            self.assertTrue(options["default_loras"][0]["present"])
            self.assertEqual(list(options)[:3], ["available_loras", "default_loras", "lora_presets"])
            self.assertTrue(all(isinstance(model, str) for model in options["planner_models"]))
            loras = await call("list_loras")
            self.assertEqual(set(loras), {"available_loras", "default_loras", "lora_presets"})
            self.assertIn(REALISM, loras["available_loras"])

            job = await call("render_film", {"references": [{"asset_id": asset, "role": "pose"}], "script": "She ends in <Pose 1>."})
            done = await self.wait(job["id"])
            self.assertEqual(done["status"], "done")
            polled = await call("get_job", {"job_id": job["id"]})
            self.assertIn("<Picture", polled["final_prompts"])
            self.assertNotIn("<Pose", polled["final_prompts"])
            self.assertTrue(polled["video_url"].startswith(self.base))

            bad = await client.call_tool("render_film", {"script": "<Picture 3>"})
            self.assertTrue(bad.is_error)
            message = " ".join(getattr(block, "text", "") for block in bad.content)
            self.assertIn("only 0 picture(s) are connected", message, "the chat model must see why the call failed")

            missing = await client.call_tool("get_job", {"job_id": "no-such-job"})
            self.assertTrue(missing.is_error)
            self.assertIn("No job", " ".join(getattr(block, "text", "") for block in missing.content))


if __name__ == "__main__":
    unittest.main()
