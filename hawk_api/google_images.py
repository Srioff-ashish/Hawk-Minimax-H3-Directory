"""Google's Gemini image models ("Nano Banana") and the SFW gate in front of them.

The Nano Banana engines are for SFW work only. Before a request reaches one, a small Gemini text model reads
the prompt and any reference images and answers sfw or nsfw, following an instruction the user can rewrite in
Studio (Prompts page, "SFW check"). Only an explicit sfw lets the request through. An nsfw verdict, a reply
that is not the expected JSON, a blocked or failed check: every one of them keeps the request away from
Google, and the ladder moves on to the next engine.

Torch-free and Studio-free (httpx only), so jobs.py and the tests can import it without a GPU.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
from dataclasses import dataclass

import httpx

API_URL = "https://generativelanguage.googleapis.com/v1beta"
#: Reads the prompt and the reference images and answers sfw/nsfw. Cheap and able to see images.
GATE_MODEL = "gemini-3.1-flash-lite"

#: The editable part of the gate's instruction: what counts as SFW. Shown and saved on the Prompts page.
DEFAULT_GATE_PROMPT = """You decide whether an image request may be sent to Google's Nano Banana image models, which this server uses for safe-for-work (SFW) images only. Anything that is not SFW is made by a different engine, so a request you mark nsfw is not refused -- it is routed elsewhere. You are deciding where the request goes, not whether it is allowed.

You are given the request type (generate or edit), the prompt, and any reference images. Judge what the finished image would show, taking the prompt and the reference images together.

Mark it NSFW when the finished image would show or emphasise any of these:
- nudity or partial nudity: exposed genitals, exposed buttocks, exposed female nipples or areolae, or clothing so sheer that these show through
- sexual activity, sexual touching, or a pose or framing whose point is sexual
- lingerie, underwear or fetish wear as the subject of the picture
- a sexualised focus on breasts, buttocks or groin
- graphic violence, gore, self-harm, or hateful symbols
- anyone who is or appears to be under 18 in a suggestive context
Mark it NSFW as well when any reference image is itself NSFW by these rules, even if the prompt is innocent.

Mark it SFW when it is ordinary photography or art without any of the above: portraits, fashion (including short dresses, bodycon, crop tops, slits, cut-outs and swimwear worn in an ordinary setting), editorial and magazine shoots, products, food, places, characters, illustrations, posters.

When you are genuinely unsure, choose nsfw: an nsfw verdict only moves the request to another engine, while a wrong sfw verdict sends it to a service that does not accept it."""

#: Appended to the gate instruction by the server and not editable: the verdict has to be machine-readable,
#: and whatever the editable text says, anything that is not exactly this shape counts as nsfw.
GATE_OUTPUT_RULES = """OUTPUT (set by the server; not editable)
Reply with only one JSON object and nothing else:
{"verdict": "sfw" or "nsfw", "reason": "one short sentence"}
Any other reply is treated as nsfw."""

#: What imageConfig.aspectRatio accepts on the Gemini image models.
ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")

#: finishReason / blockReason values that mean Google declined to answer for content reasons.
BLOCKED = frozenset({"SAFETY", "IMAGE_SAFETY", "PROHIBITED_CONTENT", "IMAGE_PROHIBITED_CONTENT", "BLOCKLIST",
                     "SPII", "RECITATION", "IMAGE_RECITATION", "OTHER", "IMAGE_OTHER"})

#: USD per image by engine and output tier, from ai.google.dev/gemini-api/docs/pricing (standard paid tier).
#: The 1K price is the one Studio and the agent quote; larger outputs bill at the higher tiers.
PRICES = {
    "nano-banana": {"1K": 0.067, "2K": 0.101, "4K": 0.151},
    "nano-banana-pro": {"1K": 0.134, "2K": 0.134, "4K": 0.24},
}


class GoogleError(RuntimeError):
    """A Gemini call that produced no image (or no verdict). ``blocked`` when Google declined on content."""

    def __init__(self, message: str, blocked: bool = False):
        super().__init__(message)
        self.blocked = blocked


@dataclass(frozen=True)
class GateVerdict:
    sfw: bool
    reason: str


def gate_instruction(editable: str) -> str:
    """The full system instruction the gate model sees: the user's text, then the fixed output rules."""
    return f"{(editable or DEFAULT_GATE_PROMPT).rstrip()}\n\n{GATE_OUTPUT_RULES}"


def parse_verdict(text: str) -> GateVerdict:
    """Read the gate model's reply. Only an unambiguous {"verdict": "sfw"} passes; everything else is nsfw."""
    raw = (text or "").strip()
    fenced = re.search(r"\{.*\}", raw, re.S)  # tolerate a code fence or a stray word around the object
    try:
        data = json.loads(fenced.group(0) if fenced else raw)
    except (json.JSONDecodeError, AttributeError):
        return GateVerdict(False, "The SFW check did not answer in the expected format, so it counts as nsfw.")
    if not isinstance(data, dict):
        return GateVerdict(False, "The SFW check did not answer in the expected format, so it counts as nsfw.")
    verdict = str(data.get("verdict") or "").strip().lower()
    reason = str(data.get("reason") or "").strip()[:300]
    if verdict == "sfw":
        return GateVerdict(True, reason or "Judged SFW.")
    if verdict == "nsfw":
        return GateVerdict(False, reason or "Judged NSFW.")
    return GateVerdict(False, f"The SFW check answered {verdict!r} rather than sfw or nsfw, so it counts as nsfw.")


def _parse_size(size: str | None) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*[x*×]\s*(\d+)\s*", size or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def image_config(size: str | None) -> tuple[dict, str]:
    """(imageConfig, output tier) for a "WxH" size.

    Gemini takes an aspect ratio and a resolution tier rather than pixels, so the nearest ratio is used, and a
    size over about 1.7 MP asks for 2K (over about 4.5 MP, 4K). The cut sits above every size Studio offers
    (1024x1536 and 896x1600 are 1.4-1.6 MP), because 2K bills half as much again on Nano Banana 2: only a size
    asked for on purpose, like 2048x2048, should cost that. No size, or one that cannot be read, sends nothing
    and leaves Gemini at its own default, a 1K square.
    """
    parsed = _parse_size(size)
    if not parsed or not all(parsed):
        return {}, "1K"
    width, height = parsed
    target = math.log(width / height)
    ratio = min(ASPECT_RATIOS, key=lambda r: abs(math.log(int(r.split(":")[0]) / int(r.split(":")[1])) - target))
    pixels = width * height
    tier = "1K" if pixels <= 1_700_000 else "2K" if pixels <= 4_500_000 else "4K"
    config = {"aspectRatio": ratio}
    if tier != "1K":
        config["imageSize"] = tier
    return config, tier


def price(engine_id: str, tier: str = "1K") -> float:
    table = PRICES.get(engine_id) or {}
    return table.get(tier, table.get("1K", 0.0))


def _inline(data: bytes, mime: str) -> dict:
    return {"inlineData": {"mimeType": mime or "image/png", "data": base64.b64encode(data).decode("ascii")}}


def _blocked_reason(body: dict) -> str:
    """Why Google declined, when it did: a prompt-level block, or a candidate stopped for content."""
    feedback = body.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        return str(feedback["blockReason"])
    for candidate in body.get("candidates") or []:
        reason = str(candidate.get("finishReason") or "")
        if reason in BLOCKED:
            return reason
    return ""


class GoogleImageClient:
    """The Gemini API, as far as image generation, image editing and the SFW gate need it."""

    def __init__(self, api_key: str, url: str = API_URL, timeout: float = 240.0):
        self.api_key = (api_key or "").strip()
        self.url = url.rstrip("/")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _generate_content(self, model: str, payload: dict, max_retries: int = 3) -> dict:
        if not self.configured:
            raise GoogleError("No Google API key: add one under Settings, or as the GOOGLE_API_KEY Colab secret.")
        url = f"{self.url}/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=30.0)) as http:
            for attempt in range(max_retries + 1):
                try:
                    response = await http.post(url, headers=headers, json=payload)
                except httpx.HTTPError as exc:
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise GoogleError(f"Could not reach Google ({type(exc).__name__}).") from None
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if response.status_code >= 400:
                    message = str(((body or {}).get("error") or {}).get("message") or response.text[:300])
                    # 402 is "prepayment credits depleted": say so plainly, it is the usual first failure.
                    raise GoogleError(f"Google answered {response.status_code}: {message.strip()}")
                return body
        raise GoogleError("Google did not answer.")  # pragma: no cover -- the loop always returns or raises

    async def generate(self, model: str, prompt: str, references: list[tuple[bytes, str]] = (),
                       size: str | None = None) -> list[bytes]:
        """One image from a prompt, or an edit when references are given (images first, then the text)."""
        parts = [_inline(data, mime) for data, mime in references] + [{"text": prompt.strip()}]
        config, _ = image_config(size)
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                        **({"imageConfig": config} if config else {})}}
        body = await self._generate_content(model, payload)
        images = []
        for candidate in body.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    images.append(base64.b64decode(inline["data"]))
        if images:
            return images
        blocked = _blocked_reason(body)
        if blocked:
            raise GoogleError(f"Google declined to make this image ({blocked}).", blocked=True)
        said = " ".join(part.get("text", "") for candidate in body.get("candidates") or []
                        for part in (candidate.get("content") or {}).get("parts") or []).strip()
        raise GoogleError("Google returned no image" + (f": {said[:200]}" if said else "."))

    async def classify(self, model: str, instruction: str, prompt: str, action: str,
                       references: list[tuple[bytes, str]] = ()) -> GateVerdict:
        """The SFW gate. Never raises: a check that cannot run is an nsfw verdict, so Google is skipped."""
        parts = [{"text": f"Request type: {action}\nReference images: {len(references)}\nPrompt:\n{prompt.strip()}"}]
        parts += [_inline(data, mime) for data, mime in references]
        payload = {"systemInstruction": {"parts": [{"text": gate_instruction(instruction)}]},
                   "contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        try:
            body = await self._generate_content(model, payload, max_retries=1)
        except GoogleError as exc:
            return GateVerdict(False, f"The SFW check could not run ({exc}), so Google was not used.")
        blocked = _blocked_reason(body)
        if blocked:
            return GateVerdict(False, f"The SFW check's own model declined to read this request ({blocked}).")
        text = "".join(part.get("text", "") for candidate in body.get("candidates") or []
                       for part in (candidate.get("content") or {}).get("parts") or [])
        return parse_verdict(text)
