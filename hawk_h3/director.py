"""HawkH3Director -- a script in, one long MiniMax H3 video with synced audio out.

Per segment this runs exactly what the stock ref2va template runs
(MiniMaxH3ReferenceToVideo -> BasicGuider -> SamplerCustomAdvanced -> both VAE
decodes), and chains the segments into one film:

* **Continuity.** The tail of the previous segment -- frames and their audio -- is
  anchored at frame 0 of the next one with the stock MiniMaxH3AddGuide logic, so
  motion, identity and room tone carry over. The re-rendered head is trimmed and
  its audio crossfaded, so nothing plays twice.
* **One text-encoder pass.** Every segment's conditioning is encoded before any
  sampling, so the 32B text encoder and the DiT swap once rather than per segment.
* **Resume.** Each finished segment is written to disk with a hash of everything
  that produced it. Re-running after a crash or a prompt tweak only renders the
  segments whose inputs -- or whose predecessors -- changed.
* **Bounded memory.** Finished segments live on disk; only the continuity tail and
  the audio stay in RAM, and the final file is assembled by stream copy.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from fractions import Fraction

import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes
import torch
from comfy_api.latest import InputImpl, Types, io, ui
from comfy_extras.nodes_audio import vae_decode_audio
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide, MiniMaxH3ReferenceToVideo

from . import media
from .common import ASPECT_RATIOS, CATEGORY, H3Pipe, H3Refs
from .references import RefBundle
from .script import CONTINUITY_MODES, FPS, MAX_TAIL_FRAMES, Job, build_jobs, job_key, parse_script

logger = logging.getLogger("HawkH3")

#: Bump when a change to this file makes old cached segments invalid.
CACHE_VERSION = 1

EXAMPLE_SCRIPT = """style: Cinematic live-action, soft overcast light, 35mm lens feel, light film grain. Native audio, music N/A. No subtitles.
---
title: Arrival
duration: 8
<Picture 1> defines the woman's face and hair. She steps off a tram onto a rain-wet platform and looks up at the station clock. Slow push-in from a medium-wide shot to a medium shot. Sound: tram brakes hiss, distant announcements, light rain on the canopy.
---
title: The call
duration: 8
She turns and walks toward camera, takes out her phone and answers: "I'm here. Where are you?" The camera tracks backwards at walking pace, holding a medium close-up. Sound: footsteps on wet tiles, her voice clear and close, rain continues.
"""

INTERPOLATION = {"off": None, "48 fps (RIFE)": 48.0, "60 fps (RIFE)": 60.0}
SEED_MODES = ["increment per segment", "same for all"]

_ENCODE_CACHE: OrderedDict = OrderedDict()
_ENCODE_CACHE_MAX = 8


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "hawk_h3"


def _run_directory(run_name: str) -> tuple[str, str]:
    subfolder = os.path.join("hawk_h3", _slug(run_name))
    path = os.path.join(folder_paths.get_output_directory(), subfolder)
    os.makedirs(path, exist_ok=True)
    return path, subfolder


def _segment_paths(run_dir: str, index: int) -> dict[str, str]:
    stem = os.path.join(run_dir, f"segment_{index + 1:03d}")
    return {"video": f"{stem}.mp4", "state": f"{stem}.pt", "meta": f"{stem}.json"}


def _is_cached(paths: dict[str, str], key: str) -> bool:
    if not all(os.path.isfile(path) for path in paths.values()):
        return False
    try:
        with open(paths["meta"], "r", encoding="utf-8") as handle:
            return json.load(handle).get("key") == key
    except (OSError, json.JSONDecodeError):
        return False


def _encode(pipe: dict, bundle: RefBundle, job: Job, width: int, height: int, ref_image_size: str):
    """MiniMaxH3ReferenceToVideo for one job, memoised across runs in this process."""
    cache_key = (
        pipe["signature"],
        id(pipe["clip"]),
        job.prompt,
        width,
        height,
        job.frames,
        ref_image_size,
        tuple(bundle.digest("Picture", n) for n in job.pictures),
        tuple(bundle.digest("Video", n) for n in job.videos),
        tuple(bundle.digest("Audio", n) for n in job.audios),
    )
    if cache_key in _ENCODE_CACHE:
        _ENCODE_CACHE.move_to_end(cache_key)
        return _ENCODE_CACHE[cache_key]

    ref_images = {f"ref_image_{slot}": bundle.pictures[n - 1] for slot, n in enumerate(job.pictures)}
    ref_videos, ref_video_audios = {}, {}
    for slot, n in enumerate(job.videos):
        video = bundle.videos[n - 1]
        ref_videos[f"ref_video_{slot}"] = video["frames"]
        if video["audio"] is not None:
            ref_video_audios[f"ref_video_audio_{slot}"] = video["audio"]
    ref_audios = {f"ref_audio_{slot}": bundle.audios[n - 1] for slot, n in enumerate(job.audios)}

    cond, latent = MiniMaxH3ReferenceToVideo.execute(
        clip=pipe["clip"],
        prompt=job.prompt,
        width=width,
        height=height,
        length=job.frames,
        ref_image_size=ref_image_size,
        vae=pipe["vae"],
        audio_vae=pipe["audio_vae"],
        ref_images=ref_images or None,
        ref_videos=ref_videos or None,
        ref_video_audios=ref_video_audios or None,
        ref_audios=ref_audios or None,
    ).args

    _ENCODE_CACHE[cache_key] = (cond, latent)
    while len(_ENCODE_CACHE) > _ENCODE_CACHE_MAX:
        _ENCODE_CACHE.popitem(last=False)
    return cond, latent


def _add_continuity(cond, latent, tail: dict, job: Job, pipe: dict, carry_audio: bool):
    frames = media.from_uint8(tail["tail_frames"][-job.tail_frames :])
    audio = None
    # A 1-frame guide is 42 ms -- too short to carry meaningful sound.
    if carry_audio and job.tail_frames >= 5:
        sample_rate = int(tail["sample_rate"])
        samples = media.audio_samples(job.tail_frames, sample_rate)
        audio = {"waveform": tail["waveform"][..., -samples:], "sample_rate": sample_rate}
    return MiniMaxH3AddGuide.execute(
        positive=cond,
        latent=latent,
        frame_idx=0,
        vae=pipe["vae"],
        audio_vae=pipe["audio_vae"] if audio is not None else None,
        image=frames,
        audio=audio,
    ).args[0]


def _interpolate(frames: torch.Tensor, target_fps: float) -> torch.Tensor:
    rife = nodes.NODE_CLASS_MAPPINGS["RIFEInterpolation"]
    out = rife().interpolate(frames, float(FPS), float(target_fps), 1.0)[0].cpu()
    # Keep the frame count exactly proportional so audio never drifts across segments.
    return media.fit_frames(out, round(frames.shape[0] * target_fps / FPS))


def _save_segment(path: str, frames: torch.Tensor, waveform: torch.Tensor, sample_rate: int, fps: float) -> None:
    components = Types.VideoComponents(
        images=frames,
        audio={"waveform": waveform, "sample_rate": sample_rate},
        frame_rate=Fraction(fps).limit_denominator(1000),
    )
    video = InputImpl.VideoFromComponents(components)
    try:
        video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=16)
    except TypeError:  # ComfyUI builds whose save_to predates the crf argument
        video.save_to(path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)


class HawkH3Director(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HawkH3Director",
            display_name="Hawk H3 Director",
            category=CATEGORY,
            description=(
                "Renders a MiniMax H3 reference-to-video script -- one segment or many -- into a "
                "single video with synced audio. Segments hand off through a continuity guide, "
                "are saved as they finish, and resume after a crash or an edit."
            ),
            search_aliases=["minimax", "h3", "director", "long video", "ref2va", "hawk"],
            is_output_node=True,
            inputs=[
                H3Pipe.Input("pipe", tooltip="From Hawk H3 Model Loader."),
                io.String.Input(
                    "script",
                    multiline=True,
                    default=EXAMPLE_SCRIPT,
                    tooltip=(
                        "Plain text: segments separated by a line of ---, each with optional "
                        "title/duration/pictures/videos/audios/seed/continuity headers; a "
                        "'style:' block is prepended to every segment. Or JSON from Hawk H3 "
                        "Story Planner. Mention references as <Picture 1>, <Video 1>, <Audio 1>."
                    ),
                ),
                H3Refs.Input("refs", optional=True, tooltip="From Hawk H3 References."),
                io.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, default="16:9"),
                io.Float.Input("megapixels", default=0.98, min=0.1, max=2.2, step=0.01,
                               tooltip="0.98 at 16:9 is H3's native 1344x768. Sizes snap to 32."),
                io.Float.Input("default_seconds", default=10.0, min=0.5, max=15.0, step=0.5,
                               tooltip="Length of segments without a duration. Snaps to H3's 17k+5 frame grid."),
                io.Int.Input("steps", default=8, min=1, max=100,
                             tooltip="8 suits the turbo 4-step LoRA; ~30+ without it."),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="res_multistep"),
                io.Combo.Input("scheduler", options=comfy.samplers.SCHEDULER_NAMES, default="simple",
                               tooltip="beta or normal often beat simple on reference-heavy prompts."),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Keep this fixed to resume or re-render single segments; randomize invalidates every segment.",
                ),
                io.Combo.Input(
                    "continuity",
                    options=CONTINUITY_MODES,
                    default="tail_22",
                    tooltip=(
                        "How segments hand off. tail_N anchors the previous segment's last N frames "
                        "(and audio) at the start of the next -- more frames carry more motion but cost "
                        "more tokens. last_frame carries one still. off makes hard cuts."
                    ),
                ),
                io.Boolean.Input("carry_audio", default=True,
                                 tooltip="Also anchor the previous segment's tail audio, so voices and room tone continue."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="match is faster; max keeps up to a 2048px short edge for stronger identity."),
                io.String.Input("run_name", default="hawk_h3",
                                tooltip="Folder under output/hawk_h3/. Segments and the final video are saved there."),
                io.Boolean.Input("resume", default=True,
                                 tooltip="Reuse saved segments whose inputs have not changed."),
                io.Combo.Input("seed_mode", options=SEED_MODES, default=SEED_MODES[0], optional=True, advanced=True),
                io.Int.Input("audio_crossfade_ms", default=60, min=0, max=1000, optional=True, advanced=True,
                             tooltip="Crossfade across each continuity seam. Hard cuts are never faded."),
                io.Combo.Input("interpolation", options=list(INTERPOLATION), default="off", optional=True, advanced=True,
                               tooltip="RIFE frame interpolation per segment. Needs ComfyUI-VFI."),
                io.Boolean.Input("encode_all_first", default=True, optional=True, advanced=True,
                                 tooltip="Encode every segment's text + references before sampling, so the text encoder and DiT swap once."),
                io.Boolean.Input("output_frames", default=False, optional=True, advanced=True,
                                 tooltip="Return every frame of the full film on `frames`. Off returns the last segment only -- a long film at full size can need tens of GB of RAM."),
            ],
            outputs=[
                io.Video.Output("video", tooltip="The whole film with audio. Connect to Save Video."),
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
                io.String.Output("prompts", tooltip="The exact text each segment was encoded with."),
                io.String.Output("info", tooltip="Resolution, timings, cache hits, warnings, files."),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipe: dict,
        script: str,
        aspect_ratio: str,
        megapixels: float,
        default_seconds: float,
        steps: int,
        sampler_name: str,
        scheduler: str,
        seed: int,
        continuity: str,
        carry_audio: bool,
        ref_image_size: str,
        run_name: str,
        resume: bool,
        refs: RefBundle | None = None,
        seed_mode: str = SEED_MODES[0],
        audio_crossfade_ms: int = 60,
        interpolation: str = "off",
        encode_all_first: bool = True,
        output_frames: bool = False,
    ) -> io.NodeOutput:
        started = time.monotonic()
        bundle = refs if refs is not None else RefBundle()

        # ---- validate everything before touching the GPU ----------------------
        jobs = build_jobs(
            parse_script(script),
            available=bundle.available(),
            video_has_audio=bundle.video_has_audio(),
            default_seconds=default_seconds,
            continuity=continuity,
            base_seed=seed,
            seed_mode="same" if seed_mode == SEED_MODES[1] else "increment",
        )
        width, height = media.resolve_resolution(
            aspect_ratio, megapixels, bundle.pictures[0] if bundle.pictures else None
        )
        out_fps = INTERPOLATION.get(interpolation) or float(FPS)
        if out_fps != FPS and "RIFEInterpolation" not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError(
                "interpolation needs ComfyUI-VFI (github.com/GACLove/ComfyUI-VFI) for RIFE. "
                "Install it, or set interpolation to off."
            )
        warnings = [w for job in jobs for w in job.warnings]
        for warning in warnings:
            logger.warning("HawkH3: %s", warning)

        run_dir, subfolder = _run_directory(run_name)
        context = {
            "version": CACHE_VERSION,
            "pipe": pipe["signature"],
            "size": [width, height],
            "steps": steps,
            "sampler": sampler_name,
            "scheduler": scheduler,
            "ref_image_size": ref_image_size,
            "carry_audio": bool(carry_audio),
            "fps": out_fps,
        }
        keys: list[str] = []
        previous = ""
        for job in jobs:
            refs_digest = {
                "pictures": [bundle.digest("Picture", n) for n in job.pictures],
                "videos": [bundle.digest("Video", n) for n in job.videos],
                "audios": [bundle.digest("Audio", n) for n in job.audios],
            }
            previous = job_key(job, {**context, "refs": refs_digest}, previous)
            keys.append(previous)
        paths = [_segment_paths(run_dir, job.index) for job in jobs]
        cached = [bool(resume) and _is_cached(p, k) for p, k in zip(paths, keys)]
        logger.info(
            "HawkH3: %d segment(s) at %dx%d, %d cached, run folder %s",
            len(jobs), width, height, sum(cached), run_dir,
        )

        model, vae, audio_vae = pipe["model"], pipe["vae"], pipe["audio_vae"]

        # ---- phase 1: conditioning (text encoder + reference VAE encodes) -------
        conditioned: dict[int, tuple] = {}
        if encode_all_first:
            for job, hit in zip(jobs, cached):
                if not hit:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    conditioned[job.index] = _encode(pipe, bundle, job, width, height, ref_image_size)

        # ---- phase 2: sample, decode, save --------------------------------------
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).cpu()
        progress = comfy.utils.ProgressBar(len(jobs))

        accumulated: torch.Tensor | None = None
        sample_rate: int | None = None
        tail: dict | None = None
        last_frames: torch.Tensor | None = None
        report = []

        for job, key, hit, segment_paths in zip(jobs, keys, cached, paths):
            comfy.model_management.throw_exception_if_processing_interrupted()
            segment_started = time.monotonic()

            if hit:
                state = torch.load(segment_paths["state"], map_location="cpu", weights_only=True)
            else:
                cond, latent = conditioned.pop(job.index, None) or _encode(
                    pipe, bundle, job, width, height, ref_image_size
                )
                trim = 0
                if job.tail_frames and tail is not None:
                    cond = _add_continuity(cond, latent, tail, job, pipe, carry_audio)
                    trim = job.tail_frames

                guider = Guider_Basic(model)
                guider.set_conds(cond)
                sampled = SamplerCustomAdvanced.execute(
                    Noise_RandomNoise(job.seed), guider, sampler, sigmas, latent
                ).args[0]
                del cond, latent, guider

                frames = media.fit_frames(nodes.VAEDecode().decode(vae, sampled)[0].cpu(), job.frames)
                audio = vae_decode_audio(audio_vae, sampled)
                del sampled
                segment_rate = int(audio["sample_rate"])
                waveform = media.fit_waveform(
                    audio["waveform"].float().cpu(), media.audio_samples(job.frames, segment_rate)
                )

                state = {
                    "tail_frames": media.to_uint8(frames[-MAX_TAIL_FRAMES:]),
                    "waveform": waveform,
                    "sample_rate": torch.tensor(segment_rate),
                    "trim": torch.tensor(trim),
                }

                body = frames[trim:]
                del frames
                if out_fps != FPS:
                    body = _interpolate(body, out_fps)
                _save_segment(
                    segment_paths["video"],
                    body,
                    waveform[..., media.audio_samples(trim, segment_rate) :],
                    segment_rate,
                    out_fps,
                )
                torch.save(state, segment_paths["state"])
                # Written last: a segment only counts as cached once everything is on disk.
                with open(segment_paths["meta"], "w", encoding="utf-8") as handle:
                    json.dump(
                        {"key": key, "title": job.title, "seconds": job.seconds, "frames": job.frames,
                         "seed": job.seed, "tail_frames": trim, "prompt": job.prompt},
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                last_frames = body
                comfy.model_management.soft_empty_cache()

            segment_rate = int(state["sample_rate"])
            trim = int(state["trim"])
            if sample_rate is None:
                sample_rate = segment_rate
            elif segment_rate != sample_rate:
                raise RuntimeError(
                    f"Segment {job.index + 1} decoded at {segment_rate} Hz but earlier segments at "
                    f"{sample_rate} Hz. Delete {run_dir} or turn resume off after changing the audio VAE."
                )
            accumulated = media.stitch_audio(
                accumulated,
                state["waveform"],
                media.audio_samples(trim, segment_rate),
                int(segment_rate * audio_crossfade_ms / 1000) if trim else 0,
            )
            tail = state

            report.append(
                {
                    "segment": job.index + 1,
                    "title": job.title,
                    "seconds": round(job.seconds, 3),
                    "frames": job.frames,
                    "continuity_frames": trim,
                    "seed": job.seed,
                    "cached": hit,
                    "render_seconds": round(time.monotonic() - segment_started, 1),
                    "file": segment_paths["video"],
                }
            )
            progress.update(1)

        # ---- assemble -----------------------------------------------------------
        final_path = os.path.join(run_dir, f"{_slug(run_name)}_final.mp4")
        complete_audio = {"waveform": accumulated, "sample_rate": sample_rate}
        InputImpl.VideoFromList(
            [InputImpl.VideoFromFile(p["video"]) for p in paths],
            complete_audio=complete_audio,
        ).save_to(final_path, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        video = InputImpl.VideoFromFile(final_path)

        if output_frames:
            frames_out = video.get_components().images
        elif last_frames is not None:
            frames_out = last_frames
        else:
            frames_out = InputImpl.VideoFromFile(paths[-1]["video"]).get_components().images

        prompts = "\n\n".join(
            f"### Segment {job.index + 1}{' - ' + job.title if job.title else ''} "
            f"({job.seconds:.2f}s, {job.frames} frames, seed {job.seed})\n{job.prompt}"
            for job in jobs
        )
        info = json.dumps(
            {
                "resolution": f"{width}x{height}",
                "fps": out_fps,
                "total_seconds": round(accumulated.shape[-1] / sample_rate, 2),
                "segments": report,
                "warnings": warnings,
                "final": final_path,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            indent=2,
            ensure_ascii=False,
        )
        return io.NodeOutput(
            video,
            frames_out,
            complete_audio,
            prompts,
            info,
            ui=ui.PreviewVideo(
                [{"filename": os.path.basename(final_path), "subfolder": subfolder, "type": "output"}]
            ),
        )


__all__ = ["HawkH3Director"]
