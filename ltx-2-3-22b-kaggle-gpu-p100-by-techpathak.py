# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#fff7ed;border:2px solid #fed7aa;border-radius:16px;padding:32px 36px;margin:8px 0;font-family:'Segoe UI',Arial,sans-serif;">
# 
#   <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
#     <div style="background:#ea580c;color:#fff;font-size:1.1em;font-weight:900;padding:11px 16px;border-radius:10px;letter-spacing:-0.5px;line-height:1;">TP</div>
#     <div>
#       <p style="color:#9a3412;font-size:0.72em;letter-spacing:2px;text-transform:uppercase;margin:0 0 4px;font-weight:700;">TechPathak &nbsp;·&nbsp; Free AI on Kaggle P100</p>
#       <h1 style="color:#1c1917;font-size:1.8em;font-weight:800;margin:0;line-height:1.2;">🎬 LTX-2.3 22B Distilled Video Generator</h1>
#     </div>
#   </div>
# 
#   <p style="color:#44403c;font-size:0.9em;margin:0 0 16px;line-height:1.6;">
#     Run a <strong>22-billion parameter</strong> video generation model for <strong>free</strong> on Kaggle's P100 GPU using the Wan2GP engine with intelligent memory offloading.
#     Supports both <strong>Text-to-Video</strong> and <strong>Image-to-Video</strong> modes.
#   </p>
# 
#   <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:20px;">
#     <div style="background:#fff;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;">
#       <p style="color:#9a3412;font-size:0.7em;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0 0 4px;">Model</p>
#       <p style="color:#1c1917;font-size:0.85em;font-weight:600;margin:0;">LTX-2.3 22B Distilled</p>
#       <p style="color:#78716c;font-size:0.75em;margin:2px 0 0;">quanto int8 quantized</p>
#     </div>
#     <div style="background:#fff;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;">
#       <p style="color:#9a3412;font-size:0.7em;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0 0 4px;">Hardware</p>
#       <p style="color:#1c1917;font-size:0.85em;font-weight:600;margin:0;">Kaggle P100 GPU</p>
#       <p style="color:#78716c;font-size:0.75em;margin:2px 0 0;">16 GB VRAM · 29 GB RAM</p>
#     </div>
#     <div style="background:#fff;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;">
#       <p style="color:#9a3412;font-size:0.7em;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0 0 4px;">Pipeline</p>
#       <p style="color:#1c1917;font-size:0.85em;font-weight:600;margin:0;">Two-Stage Distilled</p>
#       <p style="color:#78716c;font-size:0.75em;margin:2px 0 0;">8 steps → 2× upscale → 3 steps</p>
#     </div>
#     <div style="background:#fff;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;">
#       <p style="color:#9a3412;font-size:0.7em;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:0 0 4px;">Engine</p>
#       <p style="color:#1c1917;font-size:0.85em;font-weight:600;margin:0;">Wan2GP + mmgp</p>
#       <p style="color:#78716c;font-size:0.75em;margin:2px 0 0;">Profile 4 offloading</p>
#     </div>
#   </div>
# 
#   <div style="background:#fff;border:1px solid #fed7aa;border-left:4px solid #ea580c;border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:20px;">
#     <p style="color:#1c1917;font-weight:700;margin:0 0 8px;font-size:0.88em;">⚡ Quick Start</p>
#     <ol style="color:#44403c;margin:0;padding-left:18px;font-size:0.85em;line-height:2;">
#       <li><strong style="color:#1c1917;">Settings → Accelerator → GPU P100 x1</strong></li>
#       <li>Turn on <strong style="color:#1c1917;">Internet</strong> in the Settings sidebar</li>
#       <li>Run all cells <strong style="color:#1c1917;">in order</strong></li>
#       <li>Click the <strong style="color:#1c1917;">public Gradio URL</strong> in the last cell's output</li>
#     </ol>
#   </div>
# 
#   <div style="display:flex;gap:10px;flex-wrap:wrap;">
#     <a href="https://www.youtube.com/@techpathak3617" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:#ea580c;color:#fff;text-decoration:none;padding:10px 22px;border-radius:9px;font-weight:700;font-size:0.85em;box-shadow:0 2px 8px rgba(234,88,12,0.35);">▶ Subscribe on YouTube</a>
#   </div>
# 
# </div>
# 
# ---
# 
# ## Step 1: Environment Setup
# 
# Optimizes memory settings for Kaggle P100 GPU.
# 
# > If no GPU is detected → **Settings → Accelerator → GPU P100 x1**

# %% [code] {"jupyter":{"source_hidden":true}}
# Cell 0: Environment Setup (Kaggle P100)
import os, gc, psutil

print("=== Kaggle P100 Environment Setup ===")
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available")

# Drop filesystem caches
os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
os.system("echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null 2>&1")

gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("✅ Environment optimized!")
print("   Kaggle has 29GB RAM — no swap needed.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#f0f9ff;border:1px solid #bae6fd;border-left:4px solid #0369a1;border-radius:0 10px 10px 0;padding:14px 20px;font-family:'Segoe UI',Arial,sans-serif;margin:4px 0;"><span style="background:#0369a1;color:#fff;font-size:0.68em;font-weight:800;padding:3px 8px;border-radius:5px;letter-spacing:1px;text-transform:uppercase;margin-right:10px;">STEP 02</span><strong style="color:#0369a1;font-size:0.92em;">Clone Wan2GP &amp; Install Dependencies</strong><p style="color:#44403c;margin:6px 0 0;font-size:0.83em;line-height:1.6;">Clones the Wan2GP repository and installs all required packages.</p></div>

# %% [code] {"jupyter":{"source_hidden":true}}
# Cell 1: Clone Wan2GP & install dependencies
import subprocess
try:
    subprocess.run(["nvidia-smi"], check=True)
    print("GPU Active!")
except Exception:
    print("WARNING: No GPU. Go to Settings → Accelerator → GPU P100 x1")

!git clone https://github.com/DeepBeepMeep/Wan2GP.git
!pip install --timeout 120 --retries 5 -q -r Wan2GP/requirements.txt
!pip install --timeout 120 --retries 5 -q mmgp gradio

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #dc2626;border-radius:0 10px 10px 0;padding:14px 20px;font-family:'Segoe UI',Arial,sans-serif;margin:4px 0;"><span style="background:#dc2626;color:#fff;font-size:0.68em;font-weight:800;padding:3px 8px;border-radius:5px;letter-spacing:1px;text-transform:uppercase;margin-right:10px;">STEP 02b</span><strong style="color:#dc2626;font-size:0.92em;">Fix PyTorch for P100 GPU (sm_60 compatibility)</strong><p style="color:#44403c;margin:6px 0 0;font-size:0.83em;line-height:1.6;">Kaggle's default PyTorch 2.5+ dropped support for P100 (sm_60). This installs PyTorch 2.1.2 + CUDA 11.8 — the last version that supports P100. <strong style="color:#1c1917;">Do not skip this cell.</strong></p></div>

# %% [code] {"jupyter":{"source_hidden":true}}
# Cell 1b: Install PyTorch 2.1.2 — required for P100 GPU (sm_60)
# Kaggle default PyTorch 2.5+ only supports sm_70+, breaking P100 entirely
import subprocess, sys

print("Installing PyTorch 2.1.2 with CUDA 11.8 (P100 compatible)...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.1.2",
    "torchvision==0.16.2",
    "torchaudio==2.1.2",
    "--index-url", "https://download.pytorch.org/whl/cu118",
    "--upgrade"
], check=True)

# Verify
import torch
cap = torch.cuda.get_device_capability()
print(f"\nPyTorch version : {torch.__version__}")
print(f"GPU             : {torch.cuda.get_device_name()}")
print(f"Compute cap     : sm_{cap[0]}{cap[1]}")

# Quick tensor test — will crash here (not during generation) if still broken
x = torch.ones(2, 2).cuda()
del x
torch.cuda.empty_cache()
print("✅ P100 GPU tensor test passed — ready to generate!")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #15803d;border-radius:0 10px 10px 0;padding:14px 20px;font-family:'Segoe UI',Arial,sans-serif;margin:4px 0;"><span style="background:#15803d;color:#fff;font-size:0.68em;font-weight:800;padding:3px 8px;border-radius:5px;letter-spacing:1px;text-transform:uppercase;margin-right:10px;">STEP 03</span><strong style="color:#15803d;font-size:0.92em;">Download All Required Models</strong><p style="color:#44403c;margin:6px 0 0;font-size:0.83em;line-height:1.6;">Downloads ~48 GB of model files. Large files go to <code style="background:#dcfce7;padding:1px 5px;border-radius:4px;font-size:0.9em;">/kaggle/tmp</code> and are symlinked back to save disk space.</p></div>

# %% [code] {"jupyter":{"source_hidden":true}}
# Cell 2: Download all required models (Kaggle disk-aware)
import os
from huggingface_hub import hf_hub_download

REPO = "DeepBeepMeep/LTX-2"
MODEL_DIR = "Wan2GP/models"
TMP_DIR = "/kaggle/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# === Large files go to /kaggle/tmp, symlinked back ===
LARGE_FILES = [
    "ltx-2.3-22b-distilled_diffusion_model_quanto_int8.safetensors",  # 19.4 GB
    "ltx-2.3-22b-distilled-lora-384.safetensors",                      # 7.6 GB
    "ltx-2.3-22b_embeddings_connector.safetensors",                     # 4.0 GB
    "ltx-2.3-22b_text_embedding_projection.safetensors",                # 2.3 GB
    "ltx-2.3-22b_vae.safetensors",                                      # 1.5 GB
]

for f in LARGE_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {f}")
        continue
    print(f"Downloading {f} → /kaggle/tmp ...")
    hf_hub_download(repo_id=REPO, filename=f, local_dir=TMP_DIR)
    actual = os.path.join(TMP_DIR, f)
    os.symlink(actual, dest)
    print(f"  ✓ {f} (symlinked)")

# === Small files download normally to /kaggle/working ===
SMALL_FILES = [
    "ltx-2.3-22b_audio_vae.safetensors",
    "ltx-2.3-22b_vocoder.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
]

for f in SMALL_FILES:
    dest = os.path.join(MODEL_DIR, f)
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {f}")
        continue
    print(f"Downloading {f}...")
    hf_hub_download(repo_id=REPO, filename=f, local_dir=MODEL_DIR)
    print(f"  ✓ {f}")

# === Gemma text encoder — large, goes to /kaggle/tmp ===
GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_FILES = [
    "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
    "added_tokens.json",
    "chat_template.json",
    "config_light.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

gemma_dest = os.path.join(MODEL_DIR, GEMMA_FOLDER)
gemma_tmp = os.path.join(TMP_DIR, GEMMA_FOLDER)

if os.path.exists(gemma_dest):
    print(f"  ✓ Already exists: {GEMMA_FOLDER}/")
else:
    os.makedirs(gemma_tmp, exist_ok=True)
    for gf in GEMMA_FILES:
        tmp_file = os.path.join(gemma_tmp, gf)
        if os.path.exists(tmp_file):
            print(f"  ✓ Already exists: gemma/{gf}")
            continue
        print(f"Downloading gemma/{gf} → /kaggle/tmp ...")
        hf_hub_download(
            repo_id=REPO,
            filename=f"{GEMMA_FOLDER}/{gf}",
            local_dir=TMP_DIR,
        )
        print(f"  ✓ gemma/{gf}")
    os.symlink(gemma_tmp, gemma_dest)
    print(f"  ✓ {GEMMA_FOLDER}/ (symlinked)")

# Clean up HF cache
import shutil
for cache_dir in [os.path.join(MODEL_DIR, ".cache"), os.path.join(TMP_DIR, ".cache")]:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

os.system("df -h /kaggle/working /kaggle/tmp")
print("\n✅ All downloads complete!")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #b45309;border-radius:0 10px 10px 0;padding:14px 20px;font-family:'Segoe UI',Arial,sans-serif;margin:4px 0;"><span style="background:#b45309;color:#fff;font-size:0.68em;font-weight:800;padding:3px 8px;border-radius:5px;letter-spacing:1px;text-transform:uppercase;margin-right:10px;">STEP 04</span><strong style="color:#b45309;font-size:0.92em;">Write the Generation Script</strong><p style="color:#44403c;margin:6px 0 0;font-size:0.83em;line-height:1.6;">Creates <code style="background:#fef9c3;padding:1px 5px;border-radius:4px;font-size:0.9em;">run_ltx.py</code> — model loading, mmgp offloading, generation logic, and the Gradio UI.</p></div>

# %% [code] {"jupyter":{"source_hidden":true}}
%%writefile run_ltx.py
import gc
import os
import sys
import json
import random
import tempfile
import glob
import traceback
import numpy as np
import subprocess
import psutil
from PIL import Image

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gradio as gr
from shared.utils.audio_video import save_video

# ==== GPU INFO ====
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

# ==== Force attention backends for P100 (sm_60) ====
# Flash attention is NOT supported on P100 — must disable
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ==== LOAD MODEL VIA WAN2GP ====
print("\nLoading LTX-2.3 22B Distilled (quanto int8)...")
sys.stdout.flush()

from mmgp import offload
from shared.utils import files_locator as fl

fl.set_checkpoints_paths(["models", "ckpts", "."])

from models.ltx2.ltx2_handler import family_handler

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "models/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
quanto_files = [f for f in gemma_files if "quanto" in f]
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}. Check Cell 2.")
print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

transformer_path = os.path.join("models", "ltx-2.3-22b-distilled_diffusion_model_quanto_int8.safetensors")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"Transformer not found at {transformer_path}. Check Cell 2.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    text_encoder_filename=text_encoder_file,
)

# ==== Verify pipeline components ====
print("\n--- Pipeline Components ---")
for name, component in pipe.items():
    if component is not None:
        ctype = type(component).__name__
        if hasattr(component, 'parameters'):
            try:
                p = next(component.parameters())
                print(f"  {name}: {ctype} (dtype={p.dtype})")
            except StopIteration:
                print(f"  {name}: {ctype} (no params)")
        else:
            print(f"  {name}: {ctype}")
    else:
        print(f"  {name}: None")

has_upscaler = pipe.get("spatial_upsampler") is not None
print(f"\n  Spatial Upscaler: {'✅ LOADED' if has_upscaler else '❌ MISSING'}")
print(f"  Note: Distilled LoRA is baked into the quanto int8 checkpoint")
sys.stdout.flush()

# ==== Apply mmgp Profile 4 with upscaler budgets ====
print("\nApplying mmgp Profile 4 with per-model budgets...")
sys.stdout.flush()

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.bfloat16,
    budgets={
        "transformer":  7000,
        "text_encoder": 1500,
        "vae":          2500,
        "spatial_upsampler": 1500,
        "video_encoder": 1500,
        "*":             500,
    },
)
print("✅ mmgp offloading ready!")
sys.stdout.flush()

offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Two-stage distilled pipeline active.")
sys.stdout.flush()

# ==== HELPER FUNCTIONS ====
def get_resolution(base_res_str, aspect_ratio_str):
    base_resolutions = {
        "1080p": 1088,
        "720p": 704,
        "540p": 544,
        "480p": 480,
    }
    ratios = {
        "16:9 Landscape": 16/9,
        "4:3 Standard": 4/3,
        "1:1 Square": 1.0,
        "3:4 Portrait": 3/4,
        "9:16 Portrait": 9/16,
    }
    base = base_resolutions.get(base_res_str, 704)
    ratio = ratios.get(aspect_ratio_str, 16/9)
    if ratio >= 1.0:
        height = base
        width = int(base * ratio)
    else:
        width = base
        height = int(base / ratio)
    width = (width // 32) * 32
    height = (height // 32) * 32
    return width, height

def get_vae_tile_size(height, width):
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    effective_vram = vram_mb / 1.5
    if effective_vram >= 24000:
        vae_config = 1
    elif effective_vram >= 8000:
        vae_config = 2
    else:
        vae_config = 3
    ref_size = max(height, width)
    if ref_size > 480:
        vae_config += 1
    if vae_config <= 1:
        tile_size = 0
    elif vae_config == 2:
        tile_size = 512
    elif vae_config == 3:
        tile_size = 256
    else:
        tile_size = 128
    return tile_size, vae_config

DEVICE = torch.device("cuda")

@torch.inference_mode()
def Video_Generation(prompt, input_image_start, input_image_end, seed, duration_dropdown,
                     resolution_dropdown, aspect_ratio_dropdown, progress=gr.Progress()):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        progress(0, desc="Starting...")

        duration_map = {
            "2 Seconds (49 frames)": 49,
            "3 Seconds (73 frames)": 73,
            "5 Seconds (121 frames)": 121,
            "10 Seconds (241 frames)": 241,
            "15 Seconds (361 frames)": 361,
            "20 Seconds (481 frames)": 481,
        }
        num_frames = duration_map.get(duration_dropdown, 73)
        frame_rate = 24.0

        width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None
        image_end = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")
        if input_image_end is not None:
            image_end = Image.open(input_image_end).convert("RGB")

        free_vram = torch.cuda.mem_get_info()[0] / 1024**3
        ram = psutil.virtual_memory()
        print(f"\n{'='*60}")
        print(f"Generating: {width}x{height}, {num_frames} frames, seed={seed}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"  VRAM free: {free_vram:.2f} GB | RAM free: {ram.available / 1024**3:.1f} GB")
        print(f"  Stage 1 at {width//2}x{height//2} → 2x upscale → Stage 2 at {width}x{height}")
        print(f"{'='*60}")
        sys.stdout.flush()

        vae_tile_size, vae_config = get_vae_tile_size(height, width)
        print(f"  VAE tile size: {vae_tile_size} (auto-calculated, vae_config={vae_config})")
        sys.stdout.flush()

        total_steps = [8]
        current_step = [0]
        current_pass = [1]

        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            stage_name = "Stage 1 (low-res)" if current_pass[0] == 1 else "Stage 2 (full-res refine)"
            free_vram = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | VRAM free: {free_vram:.2f} GB")
            sys.stdout.flush()
            frac = current_step[0] / max(total_steps[0], 1)
            if current_pass[0] == 2:
                frac = 0.7 + 0.3 * frac
            else:
                frac = frac * 0.7
            progress(min(frac, 0.95), desc=f"{stage_name}: {current_step[0]}/{total_steps[0]}")

        gen_kwargs = dict(
            input_prompt=prompt,
            image_start=image_start,
            height=height,
            width=width,
            frame_num=num_frames,
            fps=frame_rate,
            seed=seed,
            callback=cb,
            VAE_tile_size=vae_tile_size,
            enhance_prompt=True,
        )
        if image_end is not None:
            gen_kwargs["image_end"] = image_end

        result = ltx2_model.generate(**gen_kwargs)

        if result is None:
            return None, "Generation failed or was interrupted."

        audio_data = None
        audio_sr = None

        if isinstance(result, dict):
            video_tensor = result.get("x")
            audio_data = result.get("audio")
            audio_sr = result.get("audio_sampling_rate", 24000)
            print(f"  Result dict: x={type(video_tensor)}, audio={type(audio_data)}, sr={audio_sr}")
        elif isinstance(result, tuple):
            video_tensor = result[0]
            if len(result) > 1:
                audio_data = result[1]
            if len(result) > 2:
                audio_sr = result[2]
        else:
            video_tensor = result

        if video_tensor is None or not torch.is_tensor(video_tensor):
            return None, f"❌ No video tensor found. Got: {type(video_tensor)}"

        print(f"  Video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        sys.stdout.flush()

        video_tensor = video_tensor.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        out_path = tempfile.mktemp(suffix=".mp4")
        video_for_save = video_tensor.unsqueeze(0).float()
        video_for_save = video_for_save / 127.5 - 1.0

        save_video(
            tensor=video_for_save,
            save_file=out_path,
            fps=frame_rate,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"  ✅ Video saved: {out_path}")

        if audio_data is not None:
            try:
                import soundfile as sf
                audio_tmp = tempfile.mktemp(suffix=".wav")

                if isinstance(audio_data, np.ndarray):
                    audio_np = audio_data
                    if audio_np.ndim == 2:
                        if audio_np.shape[0] <= 2:
                            audio_np = audio_np.T
                    sr = int(audio_sr) if audio_sr else 24000
                    sf.write(audio_tmp, audio_np, sr)
                    print(f"  Audio: numpy shape={audio_data.shape}, sr={sr}")
                elif torch.is_tensor(audio_data):
                    import torchaudio
                    audio_cpu = audio_data.cpu().float()
                    if audio_cpu.dim() == 1:
                        audio_cpu = audio_cpu.unsqueeze(0)
                    if audio_cpu.dim() == 3:
                        audio_cpu = audio_cpu.squeeze(0)
                    sr = int(audio_sr) if audio_sr else 24000
                    torchaudio.save(audio_tmp, audio_cpu, sr)
                    print(f"  Audio: tensor shape={audio_cpu.shape}, sr={sr}")
                else:
                    raise ValueError(f"Unknown audio type: {type(audio_data)}")

                final_path = out_path.replace(".mp4", "_with_audio.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-i", out_path, "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", final_path
                ], check=True, capture_output=True)

                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    out_path = final_path
                    print(f"  ✅ Audio muxed: {out_path}")
                else:
                    print(f"  ⚠️ Audio mux produced empty file, using video-only")
            except Exception as e:
                print(f"  ⚠️ Audio mux failed: {e}")
                traceback.print_exc()

        del video_tensor, video_for_save
        gc.collect()
        torch.cuda.empty_cache()

        progress(1.0, desc="Done!")
        print(f"  ✅ Final output: {out_path}")
        sys.stdout.flush()
        return out_path, f"✅ Done! Seed: {seed} | {width}x{height} | {num_frames} frames"

    except Exception as e:
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# ==== GRADIO UI — TechPathak Edition ====
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important; }

.gradio-container { max-width: 920px !important; margin: auto !important; }

.tp-wrap {
  background: #fff7ed;
  border: 2px solid #fed7aa;
  border-radius: 16px;
  padding: 28px 32px;
  margin-bottom: 20px;
}
.tp-top { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; flex-wrap: wrap; }
.tp-badge {
  background: #ea580c;
  color: #fff;
  font-size: 1em;
  font-weight: 900;
  padding: 9px 13px;
  border-radius: 9px;
  line-height: 1;
  flex-shrink: 0;
}
.tp-heading { color: #1c1917; font-size: 1.45em; font-weight: 800; margin: 0; }
.tp-sub { color: #78716c; font-size: 0.84em; margin: 0 0 16px; }
.tp-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
.tp-pill {
  background: #fff;
  border: 1px solid #fed7aa;
  border-radius: 20px;
  color: #9a3412;
  font-size: 0.76em;
  font-weight: 600;
  padding: 4px 13px;
}
.tp-btn-yt {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: #ea580c;
  color: #fff !important;
  text-decoration: none !important;
  padding: 9px 20px;
  border-radius: 9px;
  font-weight: 700;
  font-size: 0.82em;
  box-shadow: 0 2px 10px rgba(234,88,12,0.30);
  transition: box-shadow 0.2s, transform 0.15s;
}
.tp-btn-yt:hover { box-shadow: 0 4px 16px rgba(234,88,12,0.50); transform: translateY(-1px); }
.tp-note {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-left: 4px solid #f59e0b;
  border-radius: 0 10px 10px 0;
  padding: 12px 18px;
  margin-bottom: 20px;
  color: #44403c;
  font-size: 0.83em;
  line-height: 1.75;
}
.tp-note strong { color: #1c1917; }
.tp-note .ok  { color: #15803d; font-weight: 600; }
.tp-note .warn { color: #b45309; font-weight: 600; }
button.primary {
  background: #ea580c !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 1em !important;
  border-radius: 10px !important;
  border: none !important;
  box-shadow: 0 3px 14px rgba(234,88,12,0.35) !important;
  transition: box-shadow 0.2s, transform 0.15s !important;
}
button.primary:hover {
  box-shadow: 0 5px 20px rgba(234,88,12,0.55) !important;
  transform: translateY(-1px) !important;
}
.tp-footer {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 12px;
  padding: 20px;
  margin-top: 28px;
  text-align: center;
}
.tp-footer p { margin: 4px 0; color: #78716c; font-size: 0.82em; }
.tp-footer strong { color: #1c1917; }
.tp-footer a { color: #ea580c; text-decoration: none; font-weight: 600; margin: 0 8px; }
.tp-footer a:hover { text-decoration: underline; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX_2_3_22b_kaggle_gpu_p100_TechPathak") as demo:
    gr.HTML('''
    <div class="tp-wrap">
      <div class="tp-top">
        <span class="tp-badge">TP</span>
        <h1 class="tp-heading">&#127916; LTX-2.3 22B Distilled &mdash; Kaggle P100</h1>
      </div>
      <p class="tp-sub">Built by <strong style="color:#ea580c;">TechPathak</strong> &nbsp;&middot;&nbsp; Free AI-Powered Video Generation on Kaggle GPU</p>
      <div class="tp-pills">
        <span class="tp-pill">22B Distilled &middot; quanto int8</span>
        <span class="tp-pill">Kaggle P100 &middot; 16 GB VRAM</span>
        <span class="tp-pill">Wan2GP + mmgp Profile 4</span>
        <span class="tp-pill">Text-to-Video</span>
        <span class="tp-pill">Image-to-Video</span>
      </div>
      <a href="''' + YT + '''" target="_blank" class="tp-btn-yt">&#9654; Subscribe on YouTube</a>
    </div>
    ''')

    gr.HTML('''
    <div class="tp-note">
      <strong>Pipeline:</strong> Stage 1 at half-res (8 steps) &rarr; 2&times; spatial upscale &rarr; Stage 2 refine (3 steps)<br>
      <span class="ok">&#10003; 720p / 3-5 sec</span> &nbsp;&nbsp;
      <span class="ok">&#10003; 540p / 3-10 sec</span> &nbsp;&nbsp;
      <span class="warn">&#9888; 480p</span>: Stage 1 at 240p &mdash; expect artifacts &nbsp;&nbsp;
      <strong>Kaggle advantage:</strong> 29 GB RAM = faster mmgp offloading vs Colab
    </div>
    ''')

    with gr.Column():
        prompt = gr.Textbox(label="🎬 Prompt", lines=3,
                   placeholder="A cinematic shot of a red fox walking through a snowy forest...")

        with gr.Accordion("🖼️ Image to Video (Optional)", open=False):
            with gr.Row():
                input_image_start = gr.Image(type="filepath", label="Start Frame (optional)")
                input_image_end = gr.Image(type="filepath", label="End Frame (optional)")
            gr.Markdown("*Upload start and/or end frames. The model will generate video between them.*")

        with gr.Row():
            seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
            duration_dropdown = gr.Dropdown(
                label="⏱️ Duration",
                choices=[
                   "2 Seconds (49 frames)",
                   "3 Seconds (73 frames)",
                   "5 Seconds (121 frames)",
                   "10 Seconds (241 frames)",
                   "15 Seconds (361 frames)",
                   "20 Seconds (481 frames)",
                ],
                value="3 Seconds (73 frames)",
            )

        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                label="📐 Base Resolution Quality",
                choices=["1080p", "720p", "540p", "480p"],
                value="720p",
            )
            aspect_ratio_dropdown = gr.Dropdown(
                label="📏 Aspect Ratio",
                choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                value="16:9 Landscape",
            )

        gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        video_out = gr.Video(label="🎥 Output")
        status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

        gen_btn.click(
            fn=Video_Generation,
            inputs=[prompt, input_image_start, input_image_end, seed, duration_dropdown,
                    resolution_dropdown, aspect_ratio_dropdown],
            outputs=[video_out, status_out],
        )

    gr.HTML('''
    <div class="tp-footer">
      <p><strong>Built by TechPathak</strong></p>
      <p>Free &amp; Open Source &nbsp;&middot;&nbsp; LTX-2.3 22B Distilled &nbsp;&middot;&nbsp; Kaggle P100 GPU</p>
      <p><a href="''' + YT + '''" target="_blank">&#9654; YouTube</a></p>
    </div>
    ''')

print("\nLaunching Gradio...")
sys.stdout.flush()
demo.queue()
demo.launch(
    share=True,
    inline=False,
    debug=True,
    show_error=True,
    max_threads=1,
    ssr_mode=False,
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#faf5ff;border:1px solid #ddd6fe;border-left:4px solid #7c3aed;border-radius:0 10px 10px 0;padding:14px 20px;font-family:'Segoe UI',Arial,sans-serif;margin:4px 0;"><span style="background:#7c3aed;color:#fff;font-size:0.68em;font-weight:800;padding:3px 8px;border-radius:5px;letter-spacing:1px;text-transform:uppercase;margin-right:10px;">STEP 05</span><strong style="color:#7c3aed;font-size:0.92em;">Launch! 🚀</strong><p style="color:#44403c;margin:6px 0 0;font-size:0.83em;line-height:1.6;">Runs the generation script. Watch for the <strong>public Gradio URL</strong> printed in the output. Model loading takes <strong>5–15 minutes</strong> — the URL appears after setup is complete.</p></div>

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Cell 4: Launch!
!cd /kaggle/working && python -u run_ltx.py 2>&1

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# <div style="background:#fff7ed;border:2px solid #fed7aa;border-radius:14px;padding:28px 32px;text-align:center;font-family:'Segoe UI',Arial,sans-serif;margin-top:8px;"><div style="background:#ea580c;color:#fff;font-size:1em;font-weight:900;padding:9px 14px;border-radius:9px;display:inline-block;margin-bottom:14px;">TP</div><p style="color:#1c1917;font-size:1.05em;font-weight:700;margin:0 0 6px;">Enjoyed this notebook? ⭐ Upvote &amp; Subscribe!</p><p style="color:#78716c;font-size:0.84em;margin:0 0 18px;">More free AI tools on Kaggle every week from <strong style="color:#ea580c;">TechPathak</strong>.</p><a href="https://www.youtube.com/@techpathak3617" target="_blank" style="display:inline-flex;align-items:center;gap:8px;background:#ea580c;color:#fff;text-decoration:none;padding:10px 24px;border-radius:9px;font-weight:700;font-size:0.85em;box-shadow:0 2px 8px rgba(234,88,12,0.35);">▶ Subscribe on YouTube</a></div>