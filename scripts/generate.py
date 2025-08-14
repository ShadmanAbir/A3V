#!/usr/bin/env python3
"""
scripts/generate.py

Simple CLI wrapper to:
 - accept prompt + optional input image
 - try to run an AnimateDiff/FramePack-style pipeline if installed
 - otherwise fallback to a frame-by-frame SD img2img generator via diffusers
 - assemble frames into an MP4 using ffmpeg

Example:
python scripts/generate.py --prompt "woman wearing red dress walking on a beach" \
    --input ./examples/garment.png --frames 120 --width 1280 --height 720 --steps 20 --fps 24 --out ./output/myclip.mp4
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:
    torch = None

# ---------- Helpers ----------
def run_cmd(cmd, cwd=None, check=True):
    print("CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return proc.returncode

def assemble_video_ffmpeg(frames_pattern: str, fps: int, output_path: str):
    # Use ffmpeg to assemble frames (frame_0001.png ...)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    run_cmd(cmd)

# ---------- Fallback Img2Img (diffusers) ----------
def fallback_img2img_generate(frames_dir: str,
                              prompt: str,
                              input_image_path: Optional[str],
                              frames: int,
                              width: int,
                              height: int,
                              steps: int,
                              device: str,
                              base_sd_model: str = "runwayml/stable-diffusion-v1-5"):
    """
    Frame-by-frame img2img fallback using diffusers StableDiffusionImg2ImgPipeline.
    It's a naive fallback: each frame uses the previous frame as "input" plus a small noise/seed change
    to create motion. Good for prototyping; replace with AnimateDiff/FramePack for better motion.
    """
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow required. pip install pillow") from e

    try:
        # prefer diffusers import
        from diffusers import StableDiffusionImg2ImgPipeline
        from diffusers import DDIMScheduler
        import numpy as np
    except Exception as e:
        raise RuntimeError("diffusers required for fallback. pip install diffusers[torch]") from e

    print("Loading SD img2img pipeline (fallback). This may take a moment...")
    # Device and dtype
    torch_device = torch.device(device) if torch else "cpu"

    # Load pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_sd_model,
        torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
    )
    pipe = pipe.to(torch_device)
    # reduce memory usage
    pipe.safety_checker = None

    # initial image
    if input_image_path and os.path.exists(input_image_path):
        init_img = Image.open(input_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    else:
        # if no input image, start from random noise via a simple prompt->image pass
        print("No input image provided; creating initial seed image from text prompt.")
        frame0 = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=7.5).images[0]
        frame0 = frame0.resize((width, height), Image.LANCZOS)
        init_img = frame0

    prev_img = init_img
    seed = int(time.time()) % 2**31
    for i in range(frames):
        this_seed = seed + i * 13  # change per-frame
        generator = torch.Generator(device=pipe.device).manual_seed(this_seed)
        # Slightly change prompt to induce motion (naive approach)
        motion_prompt = f"{prompt} --frame {i} --motion subtle"
        out = pipe(
            prompt=motion_prompt,
            init_image=prev_img,
            strength=0.35 if i > 0 else 0.6,  # first frame more faithful
            guidance_scale=7.5,
            num_inference_steps=steps,
            generator=generator,
        )
        image = out.images[0].resize((width, height), Image.LANCZOS)
        fname = os.path.join(frames_dir, f"frame_{i:04d}.png")
        image.save(fname)
        print(f"Saved frame {i+1}/{frames} -> {fname}")
        prev_img = image

    return True

# ---------- Try AnimateDiff / FramePack hooks ----------
def try_animatediff_framegen(frames_dir: str,
                             prompt: str,
                             input_image_path: Optional[str],
                             frames: int,
                             width: int,
                             height: int,
                             steps: int,
                             fps: int,
                             device: str,
                             animatediff_model_dir: Optional[str],
                             framepack_model_dir: Optional[str]) -> bool:
    """
    Uses AnimateDiffPipeline (via diffusers) to generate motion-consistent frames.
    Expects AnimateDiff + MotionAdapter + SD base model installed.
    """
    try:
        import torch
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        from diffusers.utils import export_to_gif  # optional helper
        from PIL import Image
        import numpy as np
        import os
        from typing import Optional
    except ImportError as e:
        print(f"Error importing required packages for AnimateDiff: {e}")
        print("Please install the following packages:")
        print("pip install torch diffusers[torch] pillow numpy")
        return False

    try:
        # Check if CUDA is available and use FP16 if possible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("Loading MotionAdapter from guoyww...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=dtype
        )

        print("Loading base SD model with animate-diff pipeline...")
        pipe = AnimateDiffPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=adapter,
            torch_dtype=dtype
        )

        # Configure scheduler
        scheduler = DDIMScheduler.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            steps_offset=1
        )
        pipe.scheduler = scheduler

        # Move pipeline to the appropriate device
        pipe = pipe.to(device)

        # Enable memory optimizations
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

        # Set random seed (allow user to specify later)
        generator = torch.Generator(device=device).manual_seed(int(time.time()))

        # Handle input image if provided
        init_image = None
        if input_image_path and os.path.exists(input_image_path):
            try:
                init_image = Image.open(input_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
                print(f"Using input image: {input_image_path}")
            except Exception as e:
                print(f"Error loading input image: {e}")
                print("Continuing without input image")

        print("Running AnimateDiff...")
        output = pipe(
            prompt=prompt,
            num_frames=frames,
            guidance_scale=7.5,
            num_inference_steps=steps,
            generator=generator,
            init_image=init_image,
            width=width,
            height=height
        )

        # Check if FramePack is available and should be used
        if framepack_model_dir and os.path.exists(framepack_model_dir):
            try:
                from framepack import FramePackPipeline
                print("FramePack found. Applying temporal consistency optimization...")
                framepack_pipe = FramePackPipeline.from_pretrained(framepack_model_dir)
                framepack_pipe = framepack_pipe.to(device)

                # Apply FramePack to the generated frames
                optimized_frames = []
                for idx, img in enumerate(output.frames[0]):
                    optimized_img = framepack_pipe(img)
                    optimized_frames.append(optimized_img)

                anim_frames = optimized_frames
                print("FramePack optimization applied successfully.")
            except Exception as e:
                print(f"Error applying FramePack: {e}")
                print("Continuing without FramePack optimization")
                anim_frames = output.frames[0]
        else:
            anim_frames = output.frames[0]
            print("FramePack not found or not specified. Continuing without optimization.")

        # Save frames with progress logging
        print(f"Saving {len(anim_frames)} frames to {frames_dir}...")
        for idx, img in enumerate(anim_frames):
            img = img.resize((width, height), Image.LANCZOS)
            frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            img.save(frame_path)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(anim_frames):
                print(f"Saved frame {idx+1}/{len(anim_frames)} -> {frame_path}")

        return True

    except Exception as e:
        print(f"Error during AnimateDiff generation: {e}")
        print("Attempting to clean up temporary files...")
        try:
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
                print(f"Cleaned up temporary directory: {frames_dir}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        return False
