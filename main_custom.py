import os
import sys
import subprocess
import re
import shutil
import ctypes.util
import tempfile
import time
import urllib.request
import venv
import glob
import types
import colorsys

try:
    from importlib import metadata as importlib_metadata
except Exception:
    importlib_metadata = None

try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None

# Evita prompt interattivi (es. pip uninstall "Proceed (Y/n)?") in ambienti non TTY.
os.environ.setdefault("PIP_NO_INPUT", "1")


COMFYUI_MANAGER_REPO_URL = "https://github.com/Comfy-Org/ComfyUI-Manager.git"
COMFYUI_MANAGER_DIRNAME = "comfyui-manager"
COMFYUI_MANAGER_LEGACY_DIRNAME = "ComfyUI-Manager"
LOCAL_VENV_DIRNAME = ".venv"
OPENCV_GUI_PACKAGES = (
    "opencv-python",
    "opencv-contrib-python",
)
OPENCV_HEADLESS_PACKAGE = "opencv-python-headless"
# NOTE: `diffusers` recenti importano `Dinov2WithRegistersConfig` da `transformers`.
# Alcune versioni/build di transformers non lo espongono: o si aggiorna transformers,
# oppure si applica una compat patch runtime (vedi sotto).
TRANSFORMERS_TARGET_VERSION = os.environ.get("COMFYUI_TRANSFORMERS_VERSION", "4.44.0")
ACCELERATE_TARGET_VERSION = os.environ.get("COMFYUI_ACCELERATE_VERSION", "1.6.0")
TRANSFORMERS_TARGET_VERSION = os.environ.get("COMFYUI_TRANSFORMERS_VERSION", "4.54.1")
DIFFUSERS_TARGET_VERSION = os.environ.get("COMFYUI_DIFFUSERS_VERSION", "0.32.1")
HUGGINGFACE_HUB_TARGET_VERSION = os.environ.get("COMFYUI_HUGGINGFACE_HUB_VERSION", "0.34.3")
SAFETENSORS_TARGET_VERSION = os.environ.get("COMFYUI_SAFETENSORS_VERSION", "0.4.5")
PYTORCH_TARGET_VERSION = os.environ.get("COMFYUI_TORCH_VERSION", "2.9.1")
TORCHVISION_TARGET_VERSION = os.environ.get("COMFYUI_TORCHVISION_VERSION", "0.24.1")
TORCHAUDIO_TARGET_VERSION = os.environ.get("COMFYUI_TORCHAUDIO_VERSION", "2.9.1")
PYTORCH_WHEEL_INDEX_URL = os.environ.get("COMFYUI_PYTORCH_INDEX_URL", "").strip()

FLUXTRAINER_FORCE_PACKAGES = [
    f"accelerate=={ACCELERATE_TARGET_VERSION}",
    f"transformers=={TRANSFORMERS_TARGET_VERSION}",
    f"diffusers[torch]=={DIFFUSERS_TARGET_VERSION}",
    f"huggingface-hub=={HUGGINGFACE_HUB_TARGET_VERSION}",
    f"safetensors=={SAFETENSORS_TARGET_VERSION}",
    "sentencepiece>=0.2.0",
]


extra_packages = [
     "requests",
    "PyYAML",  # <-- il pacchetto pip corretto per import yaml
    "tqdm",
    "comfy_aimdo",
          "diffusers>=0.25.0",
    f"transformers=={TRANSFORMERS_TARGET_VERSION}"
        #        "transformers==4.4.1.2"

]


_AUTO_REQUIREMENTS_ALREADY_RAN = False


def _bootstrap_trace(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BOOTSTRAP][TRACE {timestamp}] {message}", flush=True)

SHARED_MODELS_URLS = {
    # =========================
    # CHECKPOINTS
    # =========================
    "checkpoints": [
        {"url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors", "filename": "v1-5-pruned-emaonly-fp16.safetensors"},
        {"url": "https://huggingface.co/webui/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors", "filename": "512-inpainting-ema.safetensors"},
        {"url": "https://huggingface.co/autismanon/modeldump/resolve/main/dreamshaper_8.safetensors", "filename": "dreamshaper_8.safetensors"},

        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "filename": "sd_xl_base_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors", "filename": "sd_xl_refiner_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors", "filename": "sd_xl_turbo_1.0_fp16.safetensors"},
        # URL firmato Civitai: se scade, va aggiornato con un nuovo download URL.
        {"url": "https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/764940/juggernautxlRagnarok.k3mq.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22juggernautXL_ragnarokBy.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20260413/us-east-1/s3/aws4_request&X-Amz-Date=20260413T224358Z&X-Amz-SignedHeaders=host&X-Amz-Signature=735a7f9a95c3a6645f9b5fa5efb4af97916b552ad1ee0308f5029fc040af4976", "filename": "juggernautXL_ragnarokBy.safetensors"},

   ],

    # =========================
    # DIFFUSION MODELS
    # =========================
    "diffusion_models": [
        # FLUX Trainer (set richiesto)
        {"url": "https://huggingface.co/bstungnguyen/Flux/resolve/main/flux1-dev.safetensors", "filename": "flux1-dev.safetensors"},
{"url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors", "filename": "flux1-dev-fp8.safetensors"},





        # >10GB circa (Qwen Image fp8)
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "filename": "qwen_image_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors", "filename": "qwen_image_distill_full_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors", "filename": "qwen_image_edit_fp8_e4m3fn.safetensors"},

        # Wan 2.1
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", "filename": "wan2.1_t2v_1.3B_fp16.safetensors"},

        # >10GB circa (14B fp16)
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "filename": "wan2.1_i2v_480p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", "filename": "wan2.1_i2v_720p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors", "filename": "wan2.1_vace_14B_fp16.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors", "filename": "wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors"},

        # Hunyuan Video (molto pesanti)
        # >10GB circa
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors", "filename": "hunyuan_video_t2v_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_image_to_video_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors"},

        # FLUX full (gated/opzionali, pesanti)
      # >10GB circa (FP8 FLUX)
        {"url": "https://huggingface.co/lllyasviel/flux1_dev/resolve/main/flux1-dev-fp8.safetensors", "filename": "flux1-schnell-fp8.safetensors"},
   
        # >10GB circa
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors", "filename": "flux1-fill-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/flux1-kontext-dev.safetensors", "filename": "flux1-kontext-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/resolve/main/flux1-canny-dev.safetensors", "filename": "flux1-canny-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/resolve/main/flux1-depth-dev.safetensors", "filename": "flux1-depth-dev.safetensors"},
    ],

    # =========================
    # TEXT ENCODERS
    # =========================
    "text_encoders": [
        # FLUX text encoders (richiesti dai workflow FluxTrainer)
        {"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "filename": "clip_l.safetensors"},
        {"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors", "filename": "t5xxl_fp16.safetensors"},
        # Compat workflow salvati su Windows con separatore "\\".
        #{"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors", "filename": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors"},


        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true", "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors?download=true", "filename": "clip_l_hunyuan.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors?download=true", "filename": "llava_llama3_fp8_scaled.safetensors"},
    ],

    # =========================
    # VAE
    # =========================
    "vae": [
        # FLUX VAE (richiesto dai workflow FluxTrainer)
        {"url": "https://huggingface.co/comfyanonymous/flux_vae/resolve/main/flux-vae-bf16.safetensors", "filename": "flux-vae-bf16.safetensors"},

        {"url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "filename": "ae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "filename": "qwen_image_vae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true", "filename": "wan_2.1_vae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors?download=true", "filename": "hunyuan_video_vae_bf16.safetensors"},
         #{"url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors", "filename": "flux2-vae.safetensors"},
    ],




    # =========================
    # CLIP VISION
    # =========================
    "LLM": [
        {"url": "https://huggingface.co/microsoft/Florence-2-large/resolve/main/model.safetensors", "filename": "florence-2-large-model.safetensors"},
        {"url": "https://huggingface.co/microsoft/Florence-2-large/resolve/main/pytorch_model.bin", "filename": "florence-2-large-pytorch_model.bin"},
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true", "filename": "clip_vision_h.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/clip_vision/llava_llama3_vision.safetensors?download=true", "filename": "llava_llama3_vision.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors", "filename": "sigclip_vision_patch14_384.safetensors"},
    ],

    # =========================
    # LORAS
    # =========================
    "loras": [
        #{"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-8steps-V1.0.safetensors"},
        #{"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors"},

        #{"url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors", "filename": "flux1-canny-dev-lora.safetensors"},
        #{"url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/resolve/main/flux1-depth-dev-lora.safetensors", "filename": "flux1-depth-dev-lora.safetensors"},
    ],

    # =========================
    # STYLE MODELS
    # =========================
    "style_models": [
        #{"url": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors", "filename": "flux1-redux-dev.safetensors"},
    ],

    # =========================
    # CONTROLNET
    # =========================
    "controlnet": [
        {"url": "https://huggingface.co/XLabs-AI/flux-controlnet-depth-v3/resolve/main/flux-depth-controlnet-v3.safetensors", "filename": "flux-depth-controlnet-v3.safetensors"},
        {"url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors", "filename": "controlV11pSd15_v10.safetensors"},
        {"url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors", "filename": "control_v11p_sd15_openpose_fp16.safetensors"},
        {"url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors", "filename": "control_v11f1p_sd15_depth_fp16.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/qwen_image_union_diffsynth_lora.safetensors", "filename": "qwen_image_union_diffsynth_lora.safetensors"},
    ],

    # =========================
    # INPAINT
    # =========================
    "inpaint": [
        {"url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth", "filename": "fooocus_inpaint_head.pth"},
        {"url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_lama.safetensors", "filename": "fooocus_lama.safetensors"},
        {"url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch", "filename": "inpaint.fooocus.patch"},
        {"url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch", "filename": "inpaint_v25.fooocus.patch"},
        {"url": "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch", "filename": "inpaint_v26.fooocus.patch"},
    ],

        "clip": [ 
            {"url": "https://huggingface.co/Madespace/clip/resolve/main/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors", "filename": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors"},
       ],
    "embeddings": [],
    "upscale_models": [
        {"url": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth", "filename": "4x-UltraSharp.pth"},
    ],
    "gligen": [],
    "hypernetworks": [],
    "vae_approx": [],

    "unet": [
        # >10GB circa (FLUX full)
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors", "filename": "flux1-schnell.safetensors"},
    ],
}


def _find_fluxtrainer_requirements(custom_nodes_dir):
    explicit_names = (
        "comfyui-fluxtrainer",
        "comfyui-flux-trainer",
        "comfyui_fluxtrainer",
        "comfyui_flux_trainer",
        "fluxtrainer",
        "flux-trainer",
        "flux_trainer",
    )

    if not os.path.isdir(custom_nodes_dir):
        return None

    for entry_name in os.listdir(custom_nodes_dir):
        if entry_name.endswith(".disabled"):
            continue

        full_path = os.path.join(custom_nodes_dir, entry_name)
        if not os.path.isdir(full_path):
            continue

        normalized = entry_name.strip().lower()
        if (
            normalized in explicit_names
            or "fluxtrainer" in normalized
            or "flux-trainer" in normalized
            or "flux_trainer" in normalized
        ):
            req = os.path.join(full_path, "requirements.txt")
            if os.path.isfile(req):
                return req

    return None


def _install_fluxtrainer_runtime_stack(custom_nodes_dir):
    flux_req = _find_fluxtrainer_requirements(custom_nodes_dir)
    if not flux_req:
        print("[BOOTSTRAP] FluxTrainer requirements not found, skip final reconciliation")
        return False

    changed_any = False

    if _requirements_file_needs_install(flux_req):
        print(f"[BOOTSTRAP] Re-installing FluxTrainer requirements as final step: {flux_req}")
        subprocess.check_call(_get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            "--upgrade",
            "--upgrade-strategy", "eager",
            "-r",
            flux_req,
        ))
        changed_any = True
    else:
        print(f"[BOOTSTRAP] FluxTrainer requirements already satisfied, skip: {flux_req}")

    pending_fluxtrainer_packages = _get_pending_requirements(FLUXTRAINER_FORCE_PACKAGES)
    if pending_fluxtrainer_packages:
        print("[BOOTSTRAP] Enforcing FluxTrainer core package versions")
        subprocess.check_call(_get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            "--upgrade",
            "--upgrade-strategy", "eager",
            *pending_fluxtrainer_packages,
        ))
        changed_any = True
    else:
        print("[BOOTSTRAP] FluxTrainer core packages already aligned, skip")

    return changed_any


def _normalize_requirement_entry(requirement):
    if requirement is None:
        return ""

    normalized = str(requirement).strip()
    if not normalized or normalized.startswith("#"):
        return ""

    comment_index = normalized.find(" #")
    if comment_index != -1:
        normalized = normalized[:comment_index].rstrip()

    return normalized


def _is_requirement_satisfied(requirement):
    requirement = _normalize_requirement_entry(requirement)
    if not requirement or Requirement is None or importlib_metadata is None:
        return False

    if requirement.startswith(("-", ".", "/")) or requirement.startswith(("git+", "http://", "https://")):
        return False

    try:
        parsed_requirement = Requirement(requirement)
    except Exception:
        return False

    if parsed_requirement.marker and not parsed_requirement.marker.evaluate():
        return True

    if parsed_requirement.url or parsed_requirement.extras:
        return False

    try:
        installed_version = importlib_metadata.version(parsed_requirement.name)
    except Exception:
        return False

    if parsed_requirement.specifier and installed_version not in parsed_requirement.specifier:
        return False

    return True


def _get_pending_requirements(requirements):
    pending = []
    seen = set()

    for requirement in requirements:
        normalized = _normalize_requirement_entry(requirement)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        if not _is_requirement_satisfied(normalized):
            pending.append(normalized)

    return pending


def _requirements_file_needs_install(requirements_path):
    try:
        with open(requirements_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                normalized = _normalize_requirement_entry(raw_line)
                if not normalized:
                    continue

                if normalized.startswith("-"):
                    return True

                if not _is_requirement_satisfied(normalized):
                    return True
    except Exception as exc:
        print(f"[BOOTSTRAP] Unable to inspect requirements file {requirements_path}, fallback to install: {exc}")
        return True

    return False

def _run_cmd_quiet(command):
    try:
        subprocess.check_call(command)
        return True
    except Exception as exc:
        print(f"[BOOTSTRAP] Command failed: {' '.join(command)} -> {exc}")
        return False


def _get_ollama_base_url():
    base_url = (
        os.environ.get("COMFYUI_OLLAMA_BASE_URL", "").strip()
        or os.environ.get("OLLAMA_HOST", "").strip()
        or "http://127.0.0.1:11434"
    )
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    return base_url.rstrip("/")


def _get_bootstrap_ollama_models():
    raw_models = os.environ.get("COMFYUI_OLLAMA_MODELS", "llama3.2:latest")
    normalized = raw_models.replace("\n", ",").replace(";", ",")
    models = []
    seen = set()
    for item in normalized.split(","):
        model = item.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return models


def _ollama_api_request(path, payload=None, timeout=10):
    import json

    url = f"{_get_ollama_base_url()}{path}"
    data = None
    headers = {}
    method = "GET"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()

    if not content:
        return {}

    return json.loads(content.decode("utf-8"))


def _is_ollama_server_reachable(timeout=3):
    try:
        _ollama_api_request("/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def _is_local_ollama_base_url():
    from urllib.parse import urlparse

    parsed = urlparse(_get_ollama_base_url())
    hostname = (parsed.hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _ensure_ollama_server_running():
    if _is_ollama_server_reachable():
        return True

    if not shutil.which("ollama"):
        return False

    if not _is_local_ollama_base_url():
        print(f"[BOOTSTRAP] Ollama server not reachable at {_get_ollama_base_url()}, skip local auto-start")
        return False

    try:
        print(f"[BOOTSTRAP] Starting Ollama server in background on {_get_ollama_base_url()}")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        print(f"[BOOTSTRAP] Unable to start Ollama server: {exc}")
        return False

    deadline = time.time() + 20
    while time.time() < deadline:
        if _is_ollama_server_reachable(timeout=2):
            print("[BOOTSTRAP] Ollama server is reachable")
            return True
        time.sleep(1)

    print(f"[BOOTSTRAP] Ollama server did not become reachable at {_get_ollama_base_url()}")
    return False


def _get_installed_ollama_models():
    payload = _ollama_api_request("/api/tags", timeout=10)
    models = set()
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or item.get("model") or "").strip()
        if name:
            models.add(name)
    return models


def _ensure_ollama_models_available():
    if os.environ.get("COMFYUI_OLLAMA_AUTO_PULL", "1") != "1":
        return

    models = _get_bootstrap_ollama_models()
    if not models:
        return

    if not _ensure_ollama_server_running():
        print("[BOOTSTRAP] Ollama model pull skipped: server is not reachable")
        return

    try:
        installed_models = _get_installed_ollama_models()
    except Exception as exc:
        print(f"[BOOTSTRAP] Unable to list Ollama models: {exc}")
        return

    for model in models:
        if model in installed_models:
            print(f"[BOOTSTRAP] Ollama model already available: {model}")
            continue

        try:
            print(f"[BOOTSTRAP] Pulling Ollama model: {model}")
            _ollama_api_request(
                "/api/pull",
                payload={"model": model, "stream": False},
                timeout=3600,
            )
            print(f"[BOOTSTRAP] Ollama model ready: {model}")
        except Exception as exc:
            print(f"[BOOTSTRAP] Ollama model pull failed for {model}: {exc}")


def _ensure_ollama_installed():
    if os.environ.get("COMFYUI_AUTO_INSTALL_OLLAMA", "1") != "1":
        return

    if shutil.which("ollama"):
        print("[BOOTSTRAP] Ollama already installed, skip")
        _ensure_ollama_models_available()
        return

    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid) and geteuid() != 0:
        print("[BOOTSTRAP] Ollama auto-install skipped: root privileges are required")
        return

    try:
        print("[BOOTSTRAP] Installing Ollama via official script")
        subprocess.check_call([
            "/bin/sh",
            "-c",
            "curl -fsSL https://ollama.com/install.sh | sh",
        ])
        print("[BOOTSTRAP] Ollama installation completed")
        _ensure_ollama_models_available()
    except Exception as exc:
        print(f"[BOOTSTRAP] Ollama installation failed: {exc}")


def _ensure_bootstrap_python_package(pkg_spec, import_name=None):
    import_name = import_name or pkg_spec.split("[")[0].split("=")[0].replace("-", "_")
    try:
        __import__(import_name)
        return
    except ImportError:
        pass

    print(f"[BOOTSTRAP] Installing Python package: {pkg_spec}")
    subprocess.check_call(_get_bootstrap_install_cmd(
        "--disable-pip-version-check",
        "--upgrade",
        pkg_spec,
    ))


def _run_bootstrap_command(command, cwd=None, env=None):
    printable = " ".join(str(part) for part in command)
    print(f"[BOOTSTRAP] Running command: {printable}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _hf_hub_download_to_local_dir(repo_id, filename, out_dir, hf_token=None):
    _ensure_bootstrap_python_package("huggingface_hub[cli]", "huggingface_hub")
    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    base_kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": "model",
        "token": hf_token,
        "local_dir": out_dir,
    }
    attempts = [
        {**base_kwargs, "local_dir_use_symlinks": False},
        base_kwargs,
    ]

    last_exc = None
    for kwargs in attempts:
        try:
            return hf_hub_download(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception:
            raise

    raise RuntimeError(f"hf_hub_download failed for {repo_id}/{filename}: {last_exc}")


def _snapshot_hf_repo_to_local_dir(repo_id, local_dir, hf_token=None, ignore_patterns=None):
    _ensure_bootstrap_python_package("huggingface_hub[cli]", "huggingface_hub")
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    base_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_dir": local_dir,
        "token": hf_token,
    }
    attempts = [
        {**base_kwargs, "ignore_patterns": ignore_patterns or [], "local_dir_use_symlinks": False},
        {**base_kwargs, "ignore_patterns": ignore_patterns or []},
        base_kwargs,
    ]

    last_exc = None
    for kwargs in attempts:
        try:
            snapshot_download(**kwargs)
            return local_dir
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception:
            raise

    raise RuntimeError(f"snapshot_download failed for {repo_id}: {last_exc}")


def _find_first_existing_path(candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _clone_or_update_llama_cpp_repo(repo_dir):
    if os.path.isdir(repo_dir):
        if os.path.isdir(os.path.join(repo_dir, ".git")):
            _run_bootstrap_command(["git", "pull", "--ff-only"], cwd=repo_dir)
            return
        raise RuntimeError(f"{repo_dir} exists but is not a git repository")

    parent_dir = os.path.dirname(repo_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    _run_bootstrap_command(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", repo_dir])


def _build_llama_cpp_quantize_binary(repo_dir):
    build_dir = os.path.join(repo_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    _run_bootstrap_command(["cmake", "-S", repo_dir, "-B", build_dir])
    _run_bootstrap_command(["cmake", "--build", build_dir, "--config", "Release"])

    is_windows = os.name == "nt"
    binary_name = "llama-quantize.exe" if is_windows else "llama-quantize"
    legacy_name = "quantize.exe" if is_windows else "quantize"
    candidates = [
        os.path.join(build_dir, "bin", binary_name),
        os.path.join(build_dir, "bin", "Release", binary_name),
        os.path.join(repo_dir, binary_name),
        os.path.join(repo_dir, legacy_name),
    ]
    quantize_bin = _find_first_existing_path(candidates)
    if quantize_bin:
        return quantize_bin

    raise FileNotFoundError(
        "Unable to find llama.cpp quantize binary. Checked:\n  - "
        + "\n  - ".join(candidates)
    )


def _convert_llama_hf_to_f16_gguf(llama_cpp_dir, model_dir, out_dir, output_stem=None):
    required_python_packages = (
        ("numpy", "numpy"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "google.protobuf"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("safetensors", "safetensors"),
    )
    for pkg_spec, import_name in required_python_packages:
        _ensure_bootstrap_python_package(pkg_spec, import_name)

    req_file = os.path.join(llama_cpp_dir, "requirements.txt")
    if os.path.isfile(req_file):
        subprocess.check_call(_get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            "--upgrade",
            "-r",
            req_file,
        ))

    converter = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(converter):
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")

    os.makedirs(out_dir, exist_ok=True)
    model_name = output_stem or os.path.basename(os.path.normpath(model_dir))
    out_file = os.path.join(out_dir, f"{model_name}-f16.gguf")

    if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
        return out_file

    _run_bootstrap_command(
        [
            sys.executable,
            converter,
            model_dir,
            "--outfile",
            out_file,
            "--outtype",
            "f16",
        ],
        cwd=llama_cpp_dir,
    )
    return out_file


def _quantize_llama_gguf(quantize_bin, input_gguf, quant_type, out_dir, output_stem=None):
    os.makedirs(out_dir, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(input_gguf))[0]
    output_name = output_stem or input_name.replace("-f16", "")
    out_file = os.path.join(out_dir, f"{output_name}-{quant_type}.gguf")

    if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
        return out_file

    _run_bootstrap_command([quantize_bin, input_gguf, out_file, quant_type])
    return out_file


def _copy_llama_artifact_to_output(src_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dst_path = os.path.join(out_dir, os.path.basename(src_path))

    src_abs = os.path.abspath(src_path)
    dst_abs = os.path.abspath(dst_path)
    if src_abs != dst_abs:
        shutil.copy2(src_path, dst_path)
    return dst_path


def _normalize_llama_bootstrap_mode():
    raw_mode = os.environ.get("COMFYUI_LLAMA_GGUF_MODE", "direct").strip().lower()
    if raw_mode in {"", "0", "false", "off", "none", "disable", "disabled"}:
        return ""
    if raw_mode in {"direct", "convert"}:
        return raw_mode

    logging.warning(f"Unsupported COMFYUI_LLAMA_GGUF_MODE '{raw_mode}', skipping LLaMA bootstrap")
    return ""


def _ensure_llama_gguf_available(model_roots):
    mode = _normalize_llama_bootstrap_mode()
    if not mode:
        _bootstrap_trace("_ensure_llama_gguf_available: skipped because bootstrap is disabled")
        return None

    if not model_roots:
        _bootstrap_trace("_ensure_llama_gguf_available: skipped because model_roots is empty")
        return None

    out_dir = os.environ.get("COMFYUI_LLAMA_GGUF_OUTDIR", "").strip() or os.path.join(model_roots[0], "text_encoders")
    hf_token = (
        os.environ.get("COMFYUI_LLAMA_HF_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
        or None
    )

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to prepare LLaMA GGUF output dir {out_dir}: {exc}")
        _bootstrap_trace(f"_ensure_llama_gguf_available: failed creating out_dir -> {exc}")
        return None

    if mode == "direct":
        repo_id = os.environ.get("COMFYUI_LLAMA_DIRECT_REPO", "bartowski/Llama-3.2-3B-Instruct-GGUF").strip()
        filename = os.environ.get("COMFYUI_LLAMA_DIRECT_FILE", "Llama-3.2-3B-Instruct-Q4_K_M.gguf").strip()
        if not repo_id or not filename:
            logging.warning("COMFYUI_LLAMA_GGUF_MODE=direct requires COMFYUI_LLAMA_DIRECT_REPO and COMFYUI_LLAMA_DIRECT_FILE")
            return None

        final_path = os.path.join(out_dir, filename)
        if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
            _bootstrap_trace(f"_ensure_llama_gguf_available: direct model already present {final_path}")
            return final_path

        try:
            _bootstrap_trace(f"_ensure_llama_gguf_available: direct download {repo_id}/{filename}")
            downloaded = _hf_hub_download_to_local_dir(repo_id, filename, out_dir, hf_token)
            _bootstrap_trace(f"_ensure_llama_gguf_available: direct download completed {downloaded}")
            return downloaded
        except Exception as exc:
            logging.warning(f"Unable to download LLaMA GGUF from {repo_id}/{filename}: {exc}")
            _bootstrap_trace(f"_ensure_llama_gguf_available: direct download failed -> {exc}")
            return None

    repo_id = os.environ.get("COMFYUI_LLAMA_HF_REPO", "meta-llama/Llama-3.2-3B-Instruct").strip()
    quant_type = os.environ.get("COMFYUI_LLAMA_QUANT", "Q4_K_M").strip() or "Q4_K_M"
    if not repo_id:
        logging.warning("COMFYUI_LLAMA_GGUF_MODE=convert requires COMFYUI_LLAMA_HF_REPO")
        return None

    model_name = repo_id.rstrip("/").split("/")[-1].strip() or "llama-model"
    final_path = os.path.join(out_dir, f"{model_name}-{quant_type}.gguf")
    if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
        _bootstrap_trace(f"_ensure_llama_gguf_available: converted model already present {final_path}")
        return final_path

    base_dir = os.path.dirname(os.path.realpath(__file__))
    workdir_root = os.environ.get("COMFYUI_LLAMA_WORKDIR", "").strip() or os.path.join(base_dir, "llm_work", "llama_gguf")
    repo_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.strip("/")) or "llama-model"
    model_workdir = os.path.join(workdir_root, repo_slug)
    snapshot_dir = os.path.join(model_workdir, "hf_model")
    converted_dir = os.path.join(model_workdir, "converted")
    quantized_dir = os.path.join(model_workdir, "quantized")
    llama_cpp_dir = os.path.join(workdir_root, "llama.cpp")

    try:
        _bootstrap_trace(f"_ensure_llama_gguf_available: snapshot repo {repo_id} -> {snapshot_dir}")
        _snapshot_hf_repo_to_local_dir(
            repo_id,
            snapshot_dir,
            hf_token=hf_token,
            ignore_patterns=[
                "*.gguf",
                "*.onnx",
                "*.msgpack",
                "*.h5",
                "*.ot",
                "*.tflite",
                "*.tar.gz",
            ],
        )

        _bootstrap_trace(f"_ensure_llama_gguf_available: prepare llama.cpp in {llama_cpp_dir}")
        _clone_or_update_llama_cpp_repo(llama_cpp_dir)
        quantize_bin = _build_llama_cpp_quantize_binary(llama_cpp_dir)

        _bootstrap_trace(f"_ensure_llama_gguf_available: convert to f16 gguf from {snapshot_dir}")
        f16_gguf = _convert_llama_hf_to_f16_gguf(
            llama_cpp_dir,
            snapshot_dir,
            converted_dir,
            output_stem=model_name,
        )

        _bootstrap_trace(f"_ensure_llama_gguf_available: quantize {f16_gguf} -> {quant_type}")
        quantized_gguf = _quantize_llama_gguf(
            quantize_bin,
            f16_gguf,
            quant_type,
            quantized_dir,
            output_stem=model_name,
        )

        final_copied = _copy_llama_artifact_to_output(quantized_gguf, out_dir)
        _bootstrap_trace(f"_ensure_llama_gguf_available: completed {final_copied}")
        return final_copied
    except Exception as exc:
        logging.warning(f"Unable to build/convert LLaMA GGUF from {repo_id}: {exc}")
        _bootstrap_trace(f"_ensure_llama_gguf_available: convert failed -> {exc}")
        return None


def _get_local_venv_python(base_dir):
    if os.name == "nt":
        return os.path.join(base_dir, LOCAL_VENV_DIRNAME, "Scripts", "python.exe")
    return os.path.join(base_dir, LOCAL_VENV_DIRNAME, "bin", "python")


def _get_bootstrap_pip_cmd(python_executable=None):
    python_executable = python_executable or sys.executable
    python_cmd = [python_executable]

    try:
        subprocess.check_output(
            python_cmd + ["-m", "pip", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return python_cmd + ["-m", "pip"]
    except Exception:
        pass

    try:
        subprocess.check_output(
            python_cmd + ["-m", "uv", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return python_cmd + ["-m", "uv", "pip"]
    except Exception:
        pass

    if python_executable == sys.executable and shutil.which("uv"):
        return ["uv", "pip"]

    raise RuntimeError(f"No supported package installer available for: {python_executable}")


def _python_supports_uv(python_executable):
    try:
        subprocess.check_output(
            [python_executable, "-m", "uv", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _seed_venv_with_get_pip(venv_python):
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="-get-pip.py") as temp_file:
            temp_path = temp_file.name

        print(f"[BOOTSTRAP] Downloading pip bootstrap script: {get_pip_url}")
        urllib.request.urlretrieve(get_pip_url, temp_path)

        print(f"[BOOTSTRAP] Installing pip into local virtualenv via get-pip.py")
        subprocess.check_call([venv_python, temp_path, "pip", "setuptools", "wheel"])
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _ensure_current_python_package_manager():
    try:
        _get_bootstrap_pip_cmd(sys.executable)
        return
    except RuntimeError:
        pass

    host_candidates = list(_iter_host_python_candidates())
    host_python = host_candidates[0] if host_candidates else None

    try:
        _ensure_venv_package_manager(sys.executable, host_python)
        _get_bootstrap_pip_cmd(sys.executable)
        return
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize a package manager for current interpreter {sys.executable}: {exc}"
        ) from exc


def _force_comfyui_cpu_mode(reason):
    if os.environ.get("COMFYUI_CPU_FALLBACK_ACTIVE") == "1":
        return

    os.environ["COMFYUI_CPU_FALLBACK_ACTIVE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""

    if "--cpu" not in sys.argv:
        sys.argv.append("--cpu")

    print(f"[BOOTSTRAP] Forcing ComfyUI CPU mode: {reason}")


def _cuda_failure_requires_cpu_fallback(message):
    if not message:
        return False

    lowered = message.lower()
    indicators = (
        "nvidia driver on your system is too old",
        "cuda initialization",
        "found no nvidia driver",
        "torch not compiled with cuda enabled",
        "cuda driver version is insufficient",
        "torch._c._cuda_init",
    )
    return any(indicator in lowered for indicator in indicators)


def _run_torch_cuda_probe(python_executable=None, timeout=45):
    python_executable = python_executable or sys.executable

    try:
        probe = subprocess.run(
            [
                python_executable,
                "-c",
                (
                    "import torch\n"
                    "if not hasattr(torch, 'cuda'):\n"
                    "    raise SystemExit(0)\n"
                    "try:\n"
                    "    torch.cuda.current_device()\n"
                    "except Exception as exc:\n"
                    "    print(exc)\n"
                    "    raise\n"
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return False, f"CUDA probe skipped for {python_executable}: {exc}"

    output = (probe.stdout or "").strip()
    return probe.returncode == 0, output


def _parse_version_tuple(raw_version):
    if not raw_version:
        return ()

    numbers = re.findall(r"\d+", str(raw_version))
    return tuple(int(part) for part in numbers[:3])


def _probe_nvidia_smi_cuda_version():
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""

    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return match.group(1) if match else ""


def _get_compatible_pytorch_index_url():
    if PYTORCH_WHEEL_INDEX_URL:
        return PYTORCH_WHEEL_INDEX_URL

    cuda_version = _probe_nvidia_smi_cuda_version()
    version_tuple = _parse_version_tuple(cuda_version)
    if not version_tuple:
        return ""

    if version_tuple >= (13, 0):
        return "https://download.pytorch.org/whl/cu130"
    if version_tuple >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    if version_tuple >= (12, 6):
        return "https://download.pytorch.org/whl/cu126"
    if version_tuple >= (12, 4):
        return "https://download.pytorch.org/whl/cu124"
    if version_tuple >= (12, 1):
        return "https://download.pytorch.org/whl/cu121"
    if version_tuple >= (11, 8):
        return "https://download.pytorch.org/whl/cu118"

    return ""


def _inspect_torch_runtime(python_executable=None, timeout=45):
    python_executable = python_executable or sys.executable
    script = (
        "import json\n"
        "data = {}\n"
        "try:\n"
        "    import torch\n"
        "    data['torch_version'] = getattr(torch, '__version__', '')\n"
        "    data['torch_cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', '')\n"
        "    try:\n"
        "        data['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    except Exception as exc:\n"
        "        data['cuda_available'] = False\n"
        "        data['cuda_is_available_error'] = str(exc)\n"
        "    try:\n"
        "        torch.cuda.current_device()\n"
        "        data['cuda_current_device_ok'] = True\n"
        "    except Exception as exc:\n"
        "        data['cuda_current_device_ok'] = False\n"
        "        data['cuda_error'] = str(exc)\n"
        "except Exception as exc:\n"
        "    data['import_error'] = str(exc)\n"
        "print(json.dumps(data))\n"
    )

    try:
        probe = subprocess.run(
            [python_executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {"probe_error": str(exc)}

    output = (probe.stdout or "").strip()
    if probe.returncode != 0:
        return {"probe_error": output or f"torch inspect failed with exit code {probe.returncode}"}

    try:
        import json

        data = json.loads(output) if output else {}
    except Exception:
        data = {"probe_error": output}

    if not isinstance(data, dict):
        data = {"probe_error": output}

    return data


def _ensure_compatible_pytorch_runtime():
    if os.environ.get("COMFYUI_AUTO_INSTALL_PYTORCH_COMPAT", "1") != "1":
        return False

    index_url = _get_compatible_pytorch_index_url()
    if not index_url:
        _bootstrap_trace("pytorch compat: no compatible PyTorch wheel index detected, skip")
        return False

    runtime_info = _inspect_torch_runtime(sys.executable)
    if runtime_info.get("cuda_current_device_ok"):
        _bootstrap_trace(
            "pytorch compat: existing torch runtime already works with CUDA "
            f"({runtime_info.get('torch_version', 'unknown')})"
        )
        return False

    desired_stack = [
        f"torch=={PYTORCH_TARGET_VERSION}",
        f"torchvision=={TORCHVISION_TARGET_VERSION}",
        f"torchaudio=={TORCHAUDIO_TARGET_VERSION}",
    ]

    install_reason = (
        runtime_info.get("cuda_error")
        or runtime_info.get("cuda_is_available_error")
        or runtime_info.get("import_error")
        or runtime_info.get("probe_error")
        or "missing/incompatible torch runtime"
    )
    print(
        "[BOOTSTRAP] Installing a PyTorch runtime compatible with the detected NVIDIA driver: "
        f"{', '.join(desired_stack)} from {index_url}"
    )
    print(f"[BOOTSTRAP] Previous torch runtime status: {install_reason}")

    subprocess.check_call(
        _get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            "--upgrade",
            "--force-reinstall",
            "--index-url",
            index_url,
            *desired_stack,
        )
    )

    runtime_info = _inspect_torch_runtime(sys.executable)
    if runtime_info.get("cuda_current_device_ok"):
        _bootstrap_trace(
            "pytorch compat: installed compatible torch runtime "
            f"({runtime_info.get('torch_version', 'unknown')})"
        )
    else:
        print(
            "[BOOTSTRAP] WARNING: compatible PyTorch install completed but CUDA probe still fails: "
            f"{runtime_info.get('cuda_error') or runtime_info.get('probe_error') or runtime_info}"
        )

    return True


def _maybe_force_cpu_mode_from_torch_probe():
    if os.environ.get("COMFYUI_AUTO_FORCE_CPU_ON_CUDA_FAILURE", "1") != "1":
        return

    if "--cpu" in sys.argv or os.environ.get("COMFYUI_CPU_FALLBACK_ACTIVE") == "1":
        return

    probe_ok, output = _run_torch_cuda_probe(sys.executable)
    if probe_ok:
        return

    if _cuda_failure_requires_cpu_fallback(output):
        _force_comfyui_cpu_mode(output.splitlines()[-1])


def _should_force_headless_opencv():
    return sys.platform.startswith("linux") and ctypes.util.find_library("GL") is None


def _is_headless_opencv_ready():
    if importlib_metadata is None:
        return False

    try:
        importlib_metadata.version(OPENCV_HEADLESS_PACKAGE)
    except Exception:
        return False

    for package_name in OPENCV_GUI_PACKAGES:
        try:
            importlib_metadata.version(package_name)
            return False
        except Exception:
            continue

    ok, info = _check_cv2_import_subprocess()
    if ok:
        print(f"[BOOTSTRAP] OpenCV headless already ready: {info}")
        return True

    return False


def _get_site_packages_dirs():
    paths = []
    try:
        import site

        paths.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            paths.append(user_site)
    except Exception:
        pass

    unique_paths = []
    seen = set()
    for path in paths:
        if not path:
            continue
        normalized = os.path.realpath(path)
        if normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    return unique_paths


def _purge_opencv_site_packages():
    removed_any = False
    patterns = [
        "cv2",
        "cv2.*",
        "opencv_python*",
        "opencv_contrib_python*",
        "opencv_python_headless*",
    ]

    for site_packages_dir in _get_site_packages_dirs():
        for pattern in patterns:
            for candidate in glob.glob(os.path.join(site_packages_dir, pattern)):
                try:
                    if os.path.isdir(candidate):
                        shutil.rmtree(candidate, ignore_errors=True)
                    elif os.path.exists(candidate):
                        os.remove(candidate)
                    removed_any = True
                    print(f"[BOOTSTRAP] Removed stale OpenCV artifact: {candidate}")
                except Exception as exc:
                    print(f"[BOOTSTRAP] Failed removing stale OpenCV artifact {candidate}: {exc}")

    return removed_any


def _can_import_cv2():
    try:
        subprocess.check_output(
            [sys.executable, "-c", "import cv2; print(cv2.__file__)"],
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        return True
    except Exception as exc:
        print(f"[BOOTSTRAP] cv2 import check failed: {exc}")
        return False


def _install_cv2_fallback():
    if "cv2" in sys.modules:
        return False

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Unable to install cv2 fallback module: {exc}") from exc

    if hasattr(Image, "Resampling"):
        resampling = Image.Resampling
    else:
        resampling = Image

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_AREA = 3
    cv2_stub.COLOR_BGR2HSV = 40
    cv2_stub.COLOR_HSV2BGR = 54
    cv2_stub.COLOR_BGRA2RGBA = 5
    cv2_stub.COLOR_BGR2RGB = 4
    cv2_stub.COLOR_RGBA2BGRA = 6
    cv2_stub.COLOR_RGB2BGR = 4

    def _require_uint8_image(image):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _swap_channels(image, order):
        array = _require_uint8_image(image)
        return array[..., order].copy()

    def _bgr_to_hsv(image):
        array = _require_uint8_image(image)
        rgb = array[..., ::-1].astype(np.float32) / 255.0
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        delta = maxc - minc

        hue = np.zeros_like(maxc)
        mask = delta != 0

        red_mask = mask & (maxc == r)
        green_mask = mask & (maxc == g)
        blue_mask = mask & (maxc == b)

        hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6.0
        hue[green_mask] = ((b[green_mask] - r[green_mask]) / delta[green_mask]) + 2.0
        hue[blue_mask] = ((r[blue_mask] - g[blue_mask]) / delta[blue_mask]) + 4.0
        hue = (hue * 30.0) % 180.0

        saturation = np.zeros_like(maxc)
        nonzero = maxc != 0
        saturation[nonzero] = delta[nonzero] / maxc[nonzero]

        hsv = np.empty_like(array)
        hsv[..., 0] = np.round(hue).astype(np.uint8)
        hsv[..., 1] = np.round(saturation * 255.0).astype(np.uint8)
        hsv[..., 2] = np.round(maxc * 255.0).astype(np.uint8)
        return hsv

    def _hsv_to_bgr(image):
        array = _require_uint8_image(image)
        h = array[..., 0].astype(np.float32) / 180.0
        s = array[..., 1].astype(np.float32) / 255.0
        v = array[..., 2].astype(np.float32) / 255.0

        flat_h = h.reshape(-1)
        flat_s = s.reshape(-1)
        flat_v = v.reshape(-1)
        rgb_flat = np.empty((flat_h.size, 3), dtype=np.uint8)

        for index, values in enumerate(zip(flat_h, flat_s, flat_v)):
            red, green, blue = colorsys.hsv_to_rgb(*values)
            rgb_flat[index] = [
                int(round(red * 255.0)),
                int(round(green * 255.0)),
                int(round(blue * 255.0)),
            ]

        rgb = rgb_flat.reshape(array.shape)
        return rgb[..., ::-1].copy()

    def cvtColor(image, code):
        if code == cv2_stub.COLOR_BGR2HSV:
            return _bgr_to_hsv(image)
        if code == cv2_stub.COLOR_HSV2BGR:
            return _hsv_to_bgr(image)
        if code == cv2_stub.COLOR_BGR2RGB or code == cv2_stub.COLOR_RGB2BGR:
            return _swap_channels(image, [2, 1, 0])
        if code == cv2_stub.COLOR_BGRA2RGBA or code == cv2_stub.COLOR_RGBA2BGRA:
            return _swap_channels(image, [2, 1, 0, 3])
        raise NotImplementedError(f"cv2 fallback does not support color code {code}")

    def resize(image, size, interpolation=None):
        del interpolation
        array = _require_uint8_image(image)
        target_size = tuple(int(value) for value in size)
        if array.ndim == 2:
            pil_image = Image.fromarray(array)
            resized = pil_image.resize(target_size, resample=resampling.BOX)
            return np.array(resized, dtype=np.uint8)

        if array.shape[2] == 3:
            pil_image = Image.fromarray(array[..., ::-1], mode="RGB")
            resized = pil_image.resize(target_size, resample=resampling.BOX)
            return np.array(resized, dtype=np.uint8)[..., ::-1].copy()

        if array.shape[2] == 4:
            pil_image = Image.fromarray(array[..., [2, 1, 0, 3]], mode="RGBA")
            resized = pil_image.resize(target_size, resample=resampling.BOX)
            return np.array(resized, dtype=np.uint8)[..., [2, 1, 0, 3]].copy()

        raise NotImplementedError(f"cv2 fallback does not support image shape {array.shape}")

    def _noop(*args, **kwargs):
        del args, kwargs
        return None

    cv2_stub.cvtColor = cvtColor
    cv2_stub.resize = resize
    cv2_stub.imshow = _noop
    cv2_stub.destroyAllWindows = _noop
    cv2_stub.waitKey = lambda *args, **kwargs: -1
    sys.modules["cv2"] = cv2_stub
    print("[BOOTSTRAP] Installed pure-Python cv2 fallback for headless environment")
    return True


def _ensure_headless_opencv():
    if not _should_force_headless_opencv():
        return False

    if _is_headless_opencv_ready():
        return False

    print("[BOOTSTRAP] libGL not found, normalizing OpenCV packages to headless variants")
    pip_cmd = _get_bootstrap_pip_cmd()
    changed_any = False

    if _uninstall_opencv_packages(pip_cmd):
        changed_any = True

    if _purge_opencv_site_packages():
        changed_any = True

    subprocess.check_call(_get_bootstrap_install_cmd(
        "--disable-pip-version-check",
        "--force-reinstall",
        OPENCV_HEADLESS_PACKAGE,
    ))

    if not _can_import_cv2():
        _install_cv2_fallback()

    return changed_any or True

def _apply_llama_cpp_present_penalty_compat():
    """
    Alcuni nodi/custom wrapper passano per errore `present_penalty` a llama_cpp,
    mentre le versioni recenti espongono `presence_penalty` o non supportano
    affatto questo parametro. Rendiamo la chiamata tollerante lato runtime.
    """
    if os.environ.get("COMFYUI_LLAMA_CPP_PRESENT_PENALTY_COMPAT", "1") != "1":
        return

    try:
        from llama_cpp import Llama
    except Exception as exc:
        print(f"[BOOTSTRAP] Skipping llama_cpp present_penalty compat, import failed: {exc}")
        return

    original_create_chat_completion = getattr(Llama, "create_chat_completion", None)
    if not callable(original_create_chat_completion):
        print("[BOOTSTRAP] Skipping llama_cpp present_penalty compat, create_chat_completion not found")
        return

    if getattr(original_create_chat_completion, "_comfyui_present_penalty_patch", False):
        return

    def _wrapped_create_chat_completion(self, *args, **kwargs):
        if "present_penalty" in kwargs:
            present_penalty = kwargs.pop("present_penalty")
            kwargs.setdefault("presence_penalty", present_penalty)

        try:
            return original_create_chat_completion(self, *args, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument 'presence_penalty'" not in message:
                raise

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("presence_penalty", None)
            return original_create_chat_completion(self, *args, **fallback_kwargs)

    _wrapped_create_chat_completion._comfyui_present_penalty_patch = True
    Llama.create_chat_completion = _wrapped_create_chat_completion
    print("[BOOTSTRAP] Applied llama_cpp compat patch: present_penalty -> presence_penalty")


def _check_cv2_import_subprocess():
    try:
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import numpy as np; import cv2; print('ok', np.__version__, getattr(cv2, '__version__', 'unknown'))",
            ],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return True, output.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        raw_output = getattr(exc, "output", b"")
        if isinstance(raw_output, bytes) and raw_output:
            return False, raw_output.decode("utf-8", errors="replace").strip()
        return False, str(exc)


def _check_onnxruntime_import_subprocess():
    try:
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import numpy as np; import onnxruntime as ort; print('ok', np.__version__, ort.__version__)",
            ],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return True, output.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        raw_output = getattr(exc, "output", b"")
        if isinstance(raw_output, bytes) and raw_output:
            return False, raw_output.decode("utf-8", errors="replace").strip()
        return False, str(exc)


def _has_numpy_abi_mismatch(info):
    markers = ("_ARRAY_API not found", "numpy.core.multiarray failed to import")
    return any(marker in info for marker in markers)


def _install_numpy_compat_runtime(log_context):
    numpy_spec = os.environ.get("COMFYUI_NUMPY_COMPAT_SPEC", "numpy<2")
    print(f"[BOOTSTRAP] Detected {log_context} ABI mismatch, installing: {numpy_spec}")
    try:
        subprocess.check_call(
            _get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-cache-dir",
                numpy_spec,
            )
        )
        return True
    except Exception as exc:
        print(f"[BOOTSTRAP] NumPy compat install failed for {log_context}: {exc}")
        return False


def _uninstall_opencv_packages(pip_cmd):
    try:
        subprocess.check_call(
            pip_cmd
            + [
                "uninstall",
                "--no-input",
                "-y",
                "opencv-python",
                "opencv-contrib-python",
                "opencv-python-headless",
            ],
        )
        return True
    except Exception:
        return False


def _ensure_cv2_importable_or_fallback():
    """
    Garantisce che `import cv2` non blocchi l'avvio dei custom nodes.
    Se OpenCV è installato ma rotto (spesso mismatch con NumPy), prova una riparazione
    best-effort; in ultima istanza installa un fallback puro-Python (limitato).
    """
    if os.environ.get("COMFYUI_ENSURE_CV2", "1") != "1":
        return True

    ok, info = _check_cv2_import_subprocess()
    if ok:
        print(f"[BOOTSTRAP] cv2 import OK: {info}")
        return True

    print(f"[BOOTSTRAP] cv2 import FAILED, attempting repair...\n{info}")

    pip_cmd = _get_bootstrap_pip_cmd()
    _uninstall_opencv_packages(pip_cmd)

    # Tentativo 1: reinstalla solo opencv headless (spesso basta).
    try:
        subprocess.check_call(
            _get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-cache-dir",
                OPENCV_HEADLESS_PACKAGE,
            )
        )
    except Exception as exc:
        print(f"[BOOTSTRAP] OpenCV headless reinstall failed: {exc}")

    ok, info = _check_cv2_import_subprocess()
    if ok:
        print(f"[BOOTSTRAP] cv2 import OK after OpenCV reinstall: {info}")
        return True

    # Tentativo 2: se è un errore di ABI NumPy<->OpenCV, forza NumPy 1.x e reinstalla.
    if _has_numpy_abi_mismatch(info) and os.environ.get("COMFYUI_CV2_NUMPY1_FALLBACK", "1") == "1":
        _install_numpy_compat_runtime("NumPy/OpenCV")
        _uninstall_opencv_packages(pip_cmd)
        try:
            subprocess.check_call(
                _get_bootstrap_install_cmd(
                    "--disable-pip-version-check",
                    "--force-reinstall",
                    "--no-cache-dir",
                    OPENCV_HEADLESS_PACKAGE,
                )
            )
        except Exception as exc:
            print(f"[BOOTSTRAP] OpenCV headless reinstall (post-NumPy) failed: {exc}")

        ok, info = _check_cv2_import_subprocess()
        if ok:
            print(f"[BOOTSTRAP] cv2 import OK after NumPy/OpenCV repair: {info}")
            return True

    if os.environ.get("COMFYUI_CV2_FALLBACK", "1") == "1":
        try:
            if _install_cv2_fallback():
                print("[BOOTSTRAP] Using pure-Python cv2 fallback (limited)")
                return True
        except Exception as exc:
            print(f"[BOOTSTRAP] cv2 fallback install failed: {exc}")

    print("[BOOTSTRAP] cv2 import still failing; continuing startup (custom nodes may fail).")
    return False


def _ensure_onnxruntime_importable_or_fallback():
    """
    Garantisce che `import onnxruntime` non fallisca per mismatch ABI con NumPy.
    Questo evita che nodi come comfyui_controlnet_aux/DWPose saltino durante il load.
    """
    if os.environ.get("COMFYUI_ENSURE_ONNXRUNTIME", "1") != "1":
        return True

    ok, info = _check_onnxruntime_import_subprocess()
    if ok:
        print(f"[BOOTSTRAP] onnxruntime import OK: {info}")
        return True

    print(f"[BOOTSTRAP] onnxruntime import FAILED, attempting repair...\n{info}")

    if _has_numpy_abi_mismatch(info) and os.environ.get("COMFYUI_ONNXRUNTIME_NUMPY1_FALLBACK", "1") == "1":
        if _install_numpy_compat_runtime("NumPy/onnxruntime"):
            ok, info = _check_onnxruntime_import_subprocess()
            if ok:
                print(f"[BOOTSTRAP] onnxruntime import OK after NumPy repair: {info}")
                return True

    print("[BOOTSTRAP] onnxruntime import still failing; continuing startup (DWPose/custom nodes may fail).")
    return False


def _iter_host_python_candidates():
    candidates = []

    env_host = os.environ.get("_COMFYUI_BOOTSTRAP_HOST_PYTHON")
    if env_host:
        candidates.append(env_host)

    base_executable = getattr(sys, "_base_executable", None)
    if base_executable:
        candidates.append(base_executable)

    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        if os.name == "nt":
            candidates.append(os.path.join(sys.base_prefix, "python.exe"))
            candidates.append(os.path.join(sys.base_prefix, "Scripts", "python.exe"))
        else:
            candidates.append(os.path.join(sys.base_prefix, "bin", "python3"))
            candidates.append(os.path.join(sys.base_prefix, "bin", "python"))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.realpath(candidate)
        if candidate == os.path.realpath(sys.executable):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            yield candidate


def _get_bootstrap_install_cmd(*install_args, python_executable=None):
    python_executable = python_executable or sys.executable
    python_executable = os.path.realpath(python_executable)

    try:
        effective_args = list(install_args)
        if "--no-input" not in effective_args:
            effective_args.insert(0, "--no-input")
        return _get_bootstrap_pip_cmd(python_executable) + ["install", *effective_args]
    except RuntimeError:
        pass

    fallback_commands = []

    for host_python in _iter_host_python_candidates():
        host_python = os.path.realpath(host_python)
        if host_python == python_executable:
            continue
        if not _python_supports_uv(host_python):
            continue
        fallback_commands.append([
            host_python,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            python_executable,
            "--no-input",
            *install_args,
        ])

    if shutil.which("uv"):
        fallback_commands.append([
            "uv",
            "pip",
            "install",
            "--python",
            python_executable,
            "--no-input",
            *install_args,
        ])

    if fallback_commands:
        return fallback_commands[0]

    raise RuntimeError(f"No supported package installer available for: {python_executable}")


def _ensure_venv_package_manager(venv_python, host_python):
    venv_python = os.path.realpath(venv_python)
    try:
        _get_bootstrap_pip_cmd(venv_python)
        return
    except RuntimeError:
        pass

    install_commands = []
    for candidate in [host_python, *_iter_host_python_candidates()]:
        if not candidate or not os.path.isfile(candidate):
            continue
        candidate = os.path.realpath(candidate)
        if candidate == venv_python:
            continue
        if not _python_supports_uv(candidate):
            continue
        install_commands.append([
            candidate,
            "-m",
            "uv",
            "pip",
            "install",
            "--python",
            venv_python,
            "pip",
            "setuptools",
            "wheel",
            "uv",
        ])

    if shutil.which("uv"):
        install_commands.append([
            "uv",
            "pip",
            "install",
            "--python",
            venv_python,
            "pip",
            "setuptools",
            "wheel",
            "uv",
        ])

    for command in install_commands:
        try:
            print(f"[BOOTSTRAP] Seeding package tools into local virtualenv: {' '.join(command)}")
            subprocess.check_call(command)
            _get_bootstrap_pip_cmd(venv_python)
            return
        except Exception as exc:
            print(f"[BOOTSTRAP] Failed seeding package tools: {' '.join(command)} -> {exc}")

    try:
        _seed_venv_with_get_pip(venv_python)
        _get_bootstrap_pip_cmd(venv_python)
        return
    except Exception as exc:
        print(f"[BOOTSTRAP] Failed seeding package tools with get-pip.py -> {exc}")

    raise RuntimeError(f"Unable to seed pip/uv into local virtualenv: {venv_python}")


def _create_local_venv(venv_dir, venv_python):
    try:
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        return
    except Exception as exc:
        print(f"[BOOTSTRAP] Standard venv creation failed, trying uv fallback: {exc}")

    uv_commands = [
        [sys.executable, "-m", "uv", "venv", venv_dir],
    ]
    if shutil.which("uv"):
        uv_commands.append(["uv", "venv", venv_dir])

    for command in uv_commands:
        try:
            subprocess.check_call(command)
            if os.path.isfile(venv_python):
                return
        except Exception as exc:
            print(f"[BOOTSTRAP] UV venv command failed: {' '.join(command)} -> {exc}")

    raise RuntimeError("Unable to create local virtualenv with stdlib venv or uv")


def _should_reexec_into_local_venv(base_dir):
    if os.environ.get("COMFYUI_AUTO_VENV", "1") != "1":
        return False

    if os.environ.get("_COMFYUI_AUTO_VENV_ACTIVE") == "1":
        return False

    if os.environ.get("VIRTUAL_ENV"):
        return False

    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return False

    current_python = os.path.realpath(sys.executable)
    venv_python = os.path.realpath(_get_local_venv_python(base_dir))
    if current_python == venv_python:
        return False

    return os.path.realpath(sys.prefix).startswith("/usr")


def _delete_local_venv_if_present(base_dir):
    venv_dir = os.path.join(base_dir, LOCAL_VENV_DIRNAME)
    if not os.path.exists(venv_dir):
        return

    real_venv_dir = os.path.realpath(venv_dir)
    if os.path.basename(real_venv_dir) != LOCAL_VENV_DIRNAME:
        raise RuntimeError(f"Refusing to delete unexpected virtualenv path: {real_venv_dir}")

    print(f"[BOOTSTRAP] Removing existing local virtualenv: {real_venv_dir}")
    shutil.rmtree(real_venv_dir)


def _clear_custom_nodes_if_present(base_dir):
    custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
    if not os.path.isdir(custom_nodes_dir):
        return

    real_custom_nodes_dir = os.path.realpath(custom_nodes_dir)
    if os.path.basename(real_custom_nodes_dir) != "custom_nodes":
        raise RuntimeError(f"Refusing to clear unexpected custom_nodes path: {real_custom_nodes_dir}")

    for entry_name in os.listdir(real_custom_nodes_dir):
        entry_path = os.path.join(real_custom_nodes_dir, entry_name)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            print(f"[BOOTSTRAP] Removing custom node directory: {entry_path}")
            shutil.rmtree(entry_path)
        else:
            print(f"[BOOTSTRAP] Removing custom node file: {entry_path}")
            os.unlink(entry_path)


def ensure_local_venv():
    if __name__ != "__main__":
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    if not _should_reexec_into_local_venv(base_dir):
        return

    if os.environ.get("COMFYUI_PREFER_HOST_PYTHON_WITH_WORKING_CUDA", "1") == "1":
        host_cuda_ok, host_cuda_output = _run_torch_cuda_probe(sys.executable)
        if host_cuda_ok:
            print(
                "[BOOTSTRAP] Keeping current Python interpreter because it already has working CUDA support; "
                "skip local virtualenv re-launch."
            )
            return
        if host_cuda_output:
            print(f"[BOOTSTRAP] Host Python CUDA probe before local venv failed: {host_cuda_output}")

    venv_dir = os.path.join(base_dir, LOCAL_VENV_DIRNAME)
    venv_python = _get_local_venv_python(base_dir)
    host_python = os.path.realpath(sys.executable)

    try:
        if not os.path.isfile(venv_python):
            print(f"[BOOTSTRAP] Creating local virtualenv: {venv_dir}")
            _create_local_venv(venv_dir, venv_python)

        if not os.path.isfile(venv_python):
            raise FileNotFoundError(f"Virtualenv interpreter not found: {venv_python}")

        _ensure_venv_package_manager(venv_python, host_python)

        print(f"[BOOTSTRAP] Re-launching with local virtualenv: {venv_python}")
        new_env = os.environ.copy()
        new_env["VIRTUAL_ENV"] = venv_dir
        new_env["_COMFYUI_AUTO_VENV_ACTIVE"] = "1"
        new_env["_COMFYUI_BOOTSTRAP_HOST_PYTHON"] = host_python
        new_env["PATH"] = os.path.dirname(venv_python) + os.pathsep + new_env.get("PATH", "")
        os.execve(venv_python, [venv_python] + sys.argv, new_env)
    except Exception as exc:
        print(f"[BOOTSTRAP] Local virtualenv bootstrap failed, continuing with system Python: {exc}")


def ensure_comfyui_manager_installed():
    if os.environ.get("COMFYUI_MANAGER_AUTO_INSTALL", "1") != "1":
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
    os.makedirs(custom_nodes_dir, exist_ok=True)

    manager_dir = os.path.join(custom_nodes_dir, COMFYUI_MANAGER_DIRNAME)
    manager_disabled_dir = manager_dir + ".disabled"
    legacy_manager_dir = os.path.join(custom_nodes_dir, COMFYUI_MANAGER_LEGACY_DIRNAME)
    legacy_manager_disabled_dir = legacy_manager_dir + ".disabled"

    if os.path.isdir(manager_disabled_dir) or os.path.isdir(legacy_manager_disabled_dir):
        print(f"[BOOTSTRAP] Found disabled ComfyUI Manager, skip install: {manager_disabled_dir}")
        return

    if os.path.isdir(manager_dir):
        print(f"[BOOTSTRAP] ComfyUI Manager already present: {manager_dir}")
        return

    if os.path.isdir(legacy_manager_dir):
        try:
            os.replace(legacy_manager_dir, manager_dir)
            print(f"[BOOTSTRAP] Renamed legacy ComfyUI Manager folder to: {manager_dir}")
            return
        except Exception as exc:
            print(f"[BOOTSTRAP] Failed renaming legacy ComfyUI Manager folder: {exc}")

    if _run_cmd_quiet(["git", "clone", COMFYUI_MANAGER_REPO_URL, manager_dir]):
        print(f"[BOOTSTRAP] ComfyUI Manager installed: {manager_dir}")
        return

    try:
        import io
        import shutil
        import urllib.request
        import zipfile

        zip_url = "https://github.com/ltdrdata/ComfyUI-Manager/archive/refs/heads/main.zip"
        print(f"[BOOTSTRAP] Downloading ComfyUI Manager ZIP fallback: {zip_url}")
        data = urllib.request.urlopen(zip_url, timeout=60).read()

        extract_tmp = os.path.join(custom_nodes_dir, "_comfyui_manager_extract_tmp")
        if os.path.isdir(extract_tmp):
            shutil.rmtree(extract_tmp, ignore_errors=True)
        os.makedirs(extract_tmp, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(extract_tmp)

        extracted_root = None
        for name in os.listdir(extract_tmp):
            candidate = os.path.join(extract_tmp, name)
            if os.path.isdir(candidate):
                extracted_root = candidate
                break

        if not extracted_root:
            raise RuntimeError("Extracted ComfyUI Manager folder not found")

        shutil.move(extracted_root, manager_dir)
        shutil.rmtree(extract_tmp, ignore_errors=True)
        print(f"[BOOTSTRAP] ComfyUI Manager installed from ZIP: {manager_dir}")

    except Exception as exc:
        print(f"[BOOTSTRAP] Failed installing ComfyUI Manager: {exc}")

def auto_install_requirements():
    global _AUTO_REQUIREMENTS_ALREADY_RAN

    if __name__ != "__main__":
        return

    if os.environ.get("COMFYUI_AUTO_INSTALL_REQUIREMENTS", "1") != "1":
        return

    if _AUTO_REQUIREMENTS_ALREADY_RAN:
        _bootstrap_trace("auto_install_requirements: skipped (already ran in this process)")
        return

    _AUTO_REQUIREMENTS_ALREADY_RAN = True

    _bootstrap_trace("auto_install_requirements: start")
    _ensure_current_python_package_manager()
    _bootstrap_trace("auto_install_requirements: python package manager ready")
    installed_any = False
    if _ensure_compatible_pytorch_runtime():
        installed_any = True
        _bootstrap_trace("auto_install_requirements: PyTorch compatibility install completed")
    _ensure_ollama_installed()
    _bootstrap_trace("auto_install_requirements: ollama check completed")

    if _ensure_headless_opencv():
        installed_any = True
    _bootstrap_trace(f"auto_install_requirements: OpenCV normalization completed (installed_any={installed_any})")

    base_dir = os.path.dirname(os.path.realpath(__file__))
    custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
    req_files = []

    main_req = os.path.join(base_dir, "requirements.txt")
    if os.path.isfile(main_req):
        req_files.append(main_req)

    if os.environ.get("COMFYUI_AUTO_INSTALL_CUSTOM_NODE_REQUIREMENTS", "1") == "1":
        if os.path.isdir(custom_nodes_dir):
            for name in os.listdir(custom_nodes_dir):
                req = os.path.join(custom_nodes_dir, name, "requirements.txt")
                if os.path.isfile(req):
                    req_files.append(req)

    pending_extra_packages = _get_pending_requirements(extra_packages)
    for pkg in pending_extra_packages:
        print(f"[BOOTSTRAP] Installing extra package: {pkg}")
        subprocess.check_call(_get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            pkg,
        ))
        installed_any = True
        _bootstrap_trace(f"auto_install_requirements: installed extra package {pkg}")

    if not pending_extra_packages:
        print("[BOOTSTRAP] Extra packages already satisfied, skip")
    _bootstrap_trace(f"auto_install_requirements: extra package phase completed ({len(pending_extra_packages)} pending)")

    seen = set()
    for req in req_files:
        req = os.path.abspath(req)
        if req in seen:
            continue
        seen.add(req)

        if not _requirements_file_needs_install(req):
            print(f"[BOOTSTRAP] Requirements already satisfied, skip: {req}")
            continue

        print(f"[BOOTSTRAP] Installing requirements from: {req}")
        try:
            subprocess.check_call(_get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "-r",
                req,
            ))
            installed_any = True
            _bootstrap_trace(f"auto_install_requirements: requirements install completed for {req}")
        except subprocess.CalledProcessError as exc:
            is_custom_node_req = os.path.realpath(req).startswith(os.path.realpath(custom_nodes_dir) + os.sep)
            strict_custom_req = os.environ.get("COMFYUI_STRICT_CUSTOM_NODE_REQUIREMENTS", "0") == "1"

            if is_custom_node_req and not strict_custom_req:
                print(
                    f"[BOOTSTRAP] WARNING: Failed installing custom node requirements ({req}) -> {exc}. "
                    "Continuing startup. Set COMFYUI_STRICT_CUSTOM_NODE_REQUIREMENTS=1 to fail fast."
                )
                continue
            raise

    # IMPORTANT: riallinea FluxTrainer alla fine, dopo TUTTI gli altri requirements,
    # così eventuali installazioni precedenti non lasciano l'ambiente in stato incoerente.
    if os.environ.get("COMFYUI_FORCE_TRANSFORMERS_FLUXTRAINER_COMPAT", "1") == "1":
        if _install_fluxtrainer_runtime_stack(custom_nodes_dir):
            installed_any = True
        _bootstrap_trace("auto_install_requirements: FluxTrainer final reconciliation completed")

    if _should_force_headless_opencv():
        if _ensure_headless_opencv():
            installed_any = True
        _bootstrap_trace("auto_install_requirements: post-requirements OpenCV normalization completed")

    # Protegge l'avvio da cv2 rotto (mismatch NumPy/OpenCV).
    _bootstrap_trace("auto_install_requirements: checking cv2 importability")
    _ensure_cv2_importable_or_fallback()
    _bootstrap_trace("auto_install_requirements: cv2 importability check completed")
    _bootstrap_trace("auto_install_requirements: checking onnxruntime importability")
    _ensure_onnxruntime_importable_or_fallback()
    _bootstrap_trace("auto_install_requirements: onnxruntime importability check completed")

    if installed_any:
        import importlib, site
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                site.addsitedir(user_site)
        except Exception:
            pass
        importlib.invalidate_caches()

        try:
            import yaml  # noqa
        except ModuleNotFoundError:
            if os.environ.get("_COMFYUI_BOOTSTRAP_REEXEC", "0") != "1":
                os.environ["_COMFYUI_BOOTSTRAP_REEXEC"] = "1"
                _bootstrap_trace("auto_install_requirements: yaml missing after install, re-executing interpreter")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            raise

    _bootstrap_trace(f"auto_install_requirements: completed (installed_any={installed_any})")

def _apply_early_transformers_fluxtrainer_compat():
    """
    Applica la compat transformers il prima possibile, prima degli import ComfyUI.
    Serve a evitare errori di import dei custom nodes FluxTrainer.
    """
    compat_mode = os.environ.get("COMFYUI_EAGER_TRANSFORMERS_COMPAT", "1").strip().lower()
    if compat_mode not in {"0", "1", "auto"}:
        compat_mode = "1"

    should_apply = compat_mode == "1"

    if compat_mode == "auto":
        custom_nodes_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_nodes")
        if os.path.isdir(custom_nodes_dir):
            try:
                for entry_name in os.listdir(custom_nodes_dir):
                    if entry_name.endswith(".disabled"):
                        continue
                    full_path = os.path.join(custom_nodes_dir, entry_name)
                    if not os.path.isdir(full_path):
                        continue
                    normalized = entry_name.strip().lower()
                    if (
                        "fluxtrainer" in normalized
                        or "flux-trainer" in normalized
                        or "flux_trainer" in normalized
                        or "fl-trainer" in normalized
                    ):
                        should_apply = True
                        break
            except Exception as exc:
                print(f"[BOOTSTRAP] Early FluxTrainer detection failed: {exc}")

    if not should_apply:
        return

    try:
        import transformers
    except Exception as exc:
        print(f"[BOOTSTRAP] Skipping early transformers compat, import failed: {exc}")
        return

    if hasattr(transformers, "CLIPFeatureExtractor"):
        _ensure_transformers_encoderdecodercache_compat(transformers, log_prefix="[BOOTSTRAP]")
        return

    clip_image_processor = None
    try:
        if hasattr(transformers, "CLIPImageProcessor"):
            clip_image_processor = transformers.CLIPImageProcessor
    except Exception:
        clip_image_processor = None

    if clip_image_processor is None:
        try:
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor

            clip_image_processor = CLIPImageProcessor
        except Exception as exc:
            print(f"[BOOTSTRAP] Skipping early transformers compat, CLIPImageProcessor unavailable: {exc}")
            return

    try:
        transformers.CLIPFeatureExtractor = clip_image_processor

        # transformers usa un LazyModule: aggiorniamo anche la struttura di import
        # per rendere affidabile `from transformers import CLIPFeatureExtractor`.
        try:
            import_utils = getattr(transformers, "utils", None)
            if import_utils is not None:
                import_utils = getattr(import_utils, "import_utils", None)
            if import_utils is not None and hasattr(import_utils, "_import_structure"):
                clip_exports = import_utils._import_structure.setdefault("models.clip", [])
                if "CLIPFeatureExtractor" not in clip_exports:
                    clip_exports.append("CLIPFeatureExtractor")
            module_all = getattr(transformers, "__all__", None)
            if isinstance(module_all, list) and "CLIPFeatureExtractor" not in module_all:
                module_all.append("CLIPFeatureExtractor")
        except Exception as compat_exc:
            print(f"[BOOTSTRAP] Early compat lazy import patch warning: {compat_exc}")

        print("[BOOTSTRAP] Applied EARLY transformers compat alias: CLIPFeatureExtractor -> CLIPImageProcessor")
    except Exception as exc:
        print(f"[BOOTSTRAP] Failed applying EARLY transformers compat alias: {exc}")

    _ensure_transformers_encoderdecodercache_compat(transformers, log_prefix="[BOOTSTRAP]")


def _apply_llama_cpp_present_penalty_compat():
    """
    Alcuni nodi/custom wrapper passano per errore `present_penalty` a llama_cpp,
    mentre le versioni recenti espongono `presence_penalty` o non supportano
    affatto questo parametro. Rendiamo la chiamata tollerante lato runtime.
    """
    if os.environ.get("COMFYUI_LLAMA_CPP_PRESENT_PENALTY_COMPAT", "1") != "1":
        return

    try:
        from llama_cpp import Llama
    except Exception as exc:
        print(f"[BOOTSTRAP] Skipping llama_cpp present_penalty compat, import failed: {exc}")
        return

    original_create_chat_completion = getattr(Llama, "create_chat_completion", None)
    if not callable(original_create_chat_completion):
        print("[BOOTSTRAP] Skipping llama_cpp present_penalty compat, create_chat_completion not found")
        return

    if getattr(original_create_chat_completion, "_comfyui_present_penalty_patch", False):
        return

    def _wrapped_create_chat_completion(self, *args, **kwargs):
        if "present_penalty" in kwargs:
            present_penalty = kwargs.pop("present_penalty")
            kwargs.setdefault("presence_penalty", present_penalty)

        try:
            return original_create_chat_completion(self, *args, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument 'presence_penalty'" not in message:
                raise

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("presence_penalty", None)
            return original_create_chat_completion(self, *args, **fallback_kwargs)

    _wrapped_create_chat_completion._comfyui_present_penalty_patch = True
    Llama.create_chat_completion = _wrapped_create_chat_completion
    print("[BOOTSTRAP] Applied llama_cpp compat patch: present_penalty -> presence_penalty")


def _ensure_transformers_encoderdecodercache_compat(transformers_module, log_prefix="[WRAPPER]"):
    """
    Alcuni stack peft/diffusers importano EncoderDecoderCache.
    Se manca (transformers vecchio/downgradato), crea un alias di fallback.
    """
    def _module_has_name(mod, name: str) -> bool:
        """Check attribute presence without relying on LazyModule __getattr__ semantics."""
        try:
            d = getattr(mod, "__dict__", None)
            if isinstance(d, dict) and name in d:
                return True
        except Exception:
            pass

        try:
            getattr(mod, name)
            return True
        except AttributeError:
            return False
        except Exception:
            # Se il lazy getattr solleva altro, consideriamo il simbolo come mancante
            # e lasciamo che la compat patch lo inietti esplicitamente.
            return False

    # 1) Crea gli alias sul modulo `transformers` (parte essenziale per far passare
    #    `from transformers import ...`). Evitiamo che un errore nella parte LazyModule
    #    impedisca la creazione degli attributi.
    try:
        if not _module_has_name(transformers_module, "EncoderDecoderCache"):
            try:
                base_cls = getattr(transformers_module, "__dict__", {}).get("DynamicCache")
                if base_cls is None:
                    base_cls = getattr(transformers_module, "DynamicCache")
            except Exception:
                base_cls = object

            class EncoderDecoderCache(base_cls):
                pass

            transformers_module.EncoderDecoderCache = EncoderDecoderCache
            try:
                sys.modules.get("transformers").EncoderDecoderCache = EncoderDecoderCache  # type: ignore[union-attr]
            except Exception:
                pass
            print(f"{log_prefix} Applied transformers compat alias: EncoderDecoderCache")
    except Exception as exc:
        print(f"{log_prefix} Failed applying EncoderDecoderCache compat alias: {exc}")

    try:
        # diffusers nuovi importano questa config; su alcune versioni/build transformers manca.
        if not _module_has_name(transformers_module, "Dinov2WithRegistersConfig"):
            try:
                dinov2_base = getattr(transformers_module, "__dict__", {}).get("Dinov2Config")
                if dinov2_base is None:
                    dinov2_base = getattr(transformers_module, "Dinov2Config")
            except Exception:
                dinov2_base = object

            class Dinov2WithRegistersConfig(dinov2_base):
                model_type = "dinov2_with_registers"

            transformers_module.Dinov2WithRegistersConfig = Dinov2WithRegistersConfig
            try:
                sys.modules.get("transformers").Dinov2WithRegistersConfig = Dinov2WithRegistersConfig  # type: ignore[union-attr]
            except Exception:
                pass
            print(f"{log_prefix} Applied transformers compat alias: Dinov2WithRegistersConfig")

            # Best-effort: inserisci anche nel modulo di configurazione, se presente,
            # così import diretti da quel path funzionano.
            try:
                import importlib

                cfg_mod = importlib.import_module("transformers.models.dinov2.configuration_dinov2")
                if not hasattr(cfg_mod, "Dinov2WithRegistersConfig"):
                    cfg_mod.Dinov2WithRegistersConfig = Dinov2WithRegistersConfig
            except Exception:
                pass
    except Exception as exc:
        print(f"{log_prefix} Failed applying Dinov2WithRegistersConfig compat alias: {exc}")

    # 2) Best-effort: aggiorna la struttura lazy di transformers, così anche gli import
    #    basati su `_import_structure` risultano consistenti.
    try:
        import_utils_pkg = getattr(transformers_module, "utils", None)
        import_utils = getattr(import_utils_pkg, "import_utils", None) if import_utils_pkg is not None else None
        import_structure = getattr(import_utils, "_import_structure", None) if import_utils is not None else None

        def _register_symbol(module_key, symbol_name):
            if isinstance(import_structure, dict):
                exports = import_structure.setdefault(module_key, [])
                if symbol_name not in exports:
                    exports.append(symbol_name)
            module_all = getattr(transformers_module, "__all__", None)
            if isinstance(module_all, list) and symbol_name not in module_all:
                module_all.append(symbol_name)

        if hasattr(transformers_module, "EncoderDecoderCache"):
            _register_symbol("cache_utils", "EncoderDecoderCache")

        if hasattr(transformers_module, "Dinov2WithRegistersConfig"):
            # A seconda della versione, la chiave può essere questa o più generica.
            _register_symbol("models.dinov2", "Dinov2WithRegistersConfig")
            _register_symbol("models.dinov2.configuration_dinov2", "Dinov2WithRegistersConfig")
    except Exception as exc:
        print(f"{log_prefix} Warning: Failed updating transformers lazy import structure: {exc}")


# Install custom nodes PRIMA del bootstrap requirements, così i loro requirements vengono inclusi.
_bootstrap_trace("startup: delete_local_venv_if_present begin")
_delete_local_venv_if_present(os.path.dirname(os.path.realpath(__file__)))
_bootstrap_trace("startup: delete_local_venv_if_present completed")
_bootstrap_trace("startup: clear_custom_nodes_if_present begin")
_clear_custom_nodes_if_present(os.path.dirname(os.path.realpath(__file__)))
_bootstrap_trace("startup: clear_custom_nodes_if_present completed")
_bootstrap_trace("startup: ensure_local_venv begin")
ensure_local_venv()
_bootstrap_trace("startup: ensure_local_venv completed")
_bootstrap_trace("startup: ensure_comfyui_manager_installed begin")
ensure_comfyui_manager_installed()
_bootstrap_trace("startup: ensure_comfyui_manager_installed completed")

# Bootstrap PRIMA degli import ComfyUI
_bootstrap_trace("startup: initial auto_install_requirements begin")
auto_install_requirements()
_bootstrap_trace("startup: initial auto_install_requirements completed")
_bootstrap_trace("startup: cuda probe begin")
_maybe_force_cpu_mode_from_torch_probe()
_bootstrap_trace("startup: cuda probe completed")

# Stabilizza l'allocator CUDA PRIMA di qualunque import Comfy/PyTorch.
# Evita mismatch: runtime cudaMallocAsync vs load-time native.
if "--disable-cuda-malloc" not in sys.argv and os.environ.get("COMFYUI_FORCE_CUDA_MALLOC", "0") != "1":
    sys.argv.append("--disable-cuda-malloc")
_legacy_pytorch_alloc_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", "").strip()
_current_pytorch_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "").strip()
if not _current_pytorch_alloc_conf:
    os.environ["PYTORCH_ALLOC_CONF"] = _legacy_pytorch_alloc_conf or "backend:native"

# Compat FluxTrainer/transformers prima di importare ComfyUI.
_bootstrap_trace("startup: early transformers compat begin")
_apply_early_transformers_fluxtrainer_compat()
_bootstrap_trace("startup: early transformers compat completed")
_bootstrap_trace("startup: llama_cpp compat begin")
_apply_llama_cpp_present_penalty_compat()
_bootstrap_trace("startup: llama_cpp compat completed")

_bootstrap_trace("startup: importing comfy.options")
import comfy.options
comfy.options.enable_args_parsing()
_bootstrap_trace("startup: comfy.options imported and args parsing enabled")

_bootstrap_trace("startup: importing ComfyUI runtime modules")
import os
import importlib.util
import folder_paths
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_api import feature_flags
import urllib.request
import urllib.parse
import urllib.error
from tqdm import tqdm
_bootstrap_trace("startup: ComfyUI runtime modules imported")

if __name__ == "__main__":
    # NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)
_bootstrap_trace("startup: logger configured")


def _infer_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        raise ValueError(f"Impossibile dedurre filename da URL: {url}")
    return filename

def _download_if_missing(url: str, dest_path: str, timeout: int = 120, ignore_http_404: bool = False):
    """
    Scarica il file solo se non esiste già.
    Scrive su .part e poi fa rename atomico.
    Mostra una progress bar con tqdm.
    """
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        logging.info(f"Model already present, skip download: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"

    logging.info(f"Downloading missing model:\n  URL:  {url}\n  DEST: {dest_path}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ComfyUI-ModelBootstrap/1.0"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            # Prova a leggere la dimensione totale (se disponibile)
            total_size = response.headers.get("Content-Length")
            total_size = int(total_size) if total_size is not None else None

            chunk_size = 1024 * 1024  # 1MB
            filename = os.path.basename(dest_path)

            with open(tmp_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=True
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        os.replace(tmp_path, dest_path)
        logging.info(f"Download completed: {dest_path}")

    except urllib.error.HTTPError as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        if ignore_http_404 and e.code == 404:
            logging.info(f"Optional model file not found (404), skip: {url}")
            return

        logging.error(f"Failed downloading model from {url} -> {dest_path}: {e}")
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        logging.error(f"Failed downloading model from {url} -> {dest_path}: {e}")
def _normalize_model_entries(entries):
    """
    Normalizza elementi del tipo:
      - "https://..."
      - {"url": "...", "filename": "..."}
    Restituisce lista di tuple (url, filename)
    """
    normalized = []
    for item in entries or []:
        if isinstance(item, str):
            url = item
            filename = _infer_filename_from_url(url)
            normalized.append((url, filename))
        elif isinstance(item, dict):
            url = item.get("url")
            if not url:
                logging.warning(f"Skipping model entry without 'url': {item}")
                continue
            filename = item.get("filename") or _infer_filename_from_url(url)
            normalized.append((url, filename))
        else:
            logging.warning(f"Unsupported model entry type, skipping: {item}")
    return normalized


def _is_writable_directory(path: str) -> bool:
    """
    Verifica se la directory è realmente scrivibile provando a creare un file temporaneo.
    """
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".comfyui_write_test.tmp")
        with open(test_file, "wb") as f:
            f.write(b"ok")
        os.remove(test_file)
        return True
    except Exception as e:
        logging.warning(f"Directory non scrivibile (skip download): {path} -> {e}")
        return False


def ensure_shared_models_downloaded(shared_root: str):
    """
    Per ogni cartella in SHARED_MODELS_URLS:
      - crea la cartella se non esiste (solo se scrivibile)
      - scarica il modello se manca (solo se scrivibile)
    Se la root è in sola lettura, salta i download senza crashare.
    """
    if not shared_root:
        _bootstrap_trace("ensure_shared_models_downloaded: skipped because shared_root is empty")
        return

    shared_root = os.path.abspath(shared_root)
    _bootstrap_trace(f"ensure_shared_models_downloaded: start for root {shared_root}")

    # Se la root non è scrivibile, salta TUTTI i download (ma ComfyUI potrà comunque leggere i modelli)
    if not _is_writable_directory(shared_root):
        logging.info(f"Shared root in sola lettura, download disabilitato: {shared_root}")
        _bootstrap_trace(f"ensure_shared_models_downloaded: root not writable, skipping {shared_root}")
        return

    for folder_name, entries in SHARED_MODELS_URLS.items():
        target_dir = os.path.join(shared_root, folder_name)
        _bootstrap_trace(f"ensure_shared_models_downloaded: checking folder {folder_name} -> {target_dir}")

        # Prova a creare/validare la cartella; se non scrivibile, skip solo quella cartella
        if not _is_writable_directory(target_dir):
            logging.info(f"Cartella modelli non scrivibile, skip download per '{folder_name}': {target_dir}")
            _bootstrap_trace(f"ensure_shared_models_downloaded: target not writable, skipping folder {folder_name}")
            continue

        for url, filename in _normalize_model_entries(entries):
            dest_path = os.path.join(target_dir, filename)
            _bootstrap_trace(f"ensure_shared_models_downloaded: ensure file {dest_path}")
            _download_if_missing(url, dest_path)
            _bootstrap_trace(f"ensure_shared_models_downloaded: file ready {dest_path}")

    _bootstrap_trace(f"ensure_shared_models_downloaded: completed for root {shared_root}")


def _resolve_model_roots():
    """
    Risolve le root modelli in modo portabile:
    - COMFYUI_MODEL_ROOTS (path separati da os.pathsep) se definita
    - COMFYUI_MODELS_DEFAULT_ROOT forza sempre la root primaria
    - /mnt/default-models viene usata solo se gia' presente
    - altrimenti usa una root locale al progetto per evitare mount non presenti/lenti
    """
    env_primary_root = os.environ.get("COMFYUI_MODELS_DEFAULT_ROOT", "").strip()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    local_primary_root = os.path.join(base_dir, "models-default")
    mnt_primary_root = "/mnt/default-models"

    if env_primary_root:
        primary_root = env_primary_root
    elif os.path.isdir(mnt_primary_root):
        primary_root = mnt_primary_root
    else:
        primary_root = local_primary_root

    secondary_root = os.environ.get("COMFYUI_MODELS_ROOT", "").strip() or os.path.join(base_dir, "models")

    candidates = [primary_root, secondary_root]
    legacy_model_roots = globals().get("MODEL_ROOTS", ("/mnt/default-models", "/vscode/workspace/models"))
    for legacy_root in legacy_model_roots:
        if legacy_root and os.path.isdir(legacy_root):
            candidates.append(legacy_root)

    env_value = os.environ.get("COMFYUI_MODEL_ROOTS", "").strip()
    if env_value:
        candidates.extend([item for item in env_value.split(os.pathsep) if item])

    roots = []
    seen = set()

    def _append_root_candidate(path):
        normalized = os.path.abspath(path)
        if normalized in seen:
            return
        seen.add(normalized)
        roots.append(normalized)

    for candidate in candidates:
        _append_root_candidate(candidate)

        nested_default_models = os.path.join(candidate, "default-models")
        if os.path.isdir(nested_default_models):
            _append_root_candidate(nested_default_models)

    _bootstrap_trace(f"_resolve_model_roots: resolved {roots}")
    return roots


def _ensure_llm_subdirs(model_roots):
    """
    Alcuni nodi (es. Florence2ModelLoader) cercano esplicitamente models/LLM.
    Garantisce che la cartella esista in ogni root registrata.
    """
    for root in model_roots:
        try:
            _bootstrap_trace(f"_ensure_llm_subdirs: ensuring {os.path.join(root, 'LLM')}")
            os.makedirs(os.path.join(root, "LLM"), exist_ok=True)
        except Exception as exc:
            logging.warning(f"Unable to create LLM folder in {root}: {exc}")
            _bootstrap_trace(f"_ensure_llm_subdirs: failed for {root} -> {exc}")


def _sync_llm_primary_to_secondary(model_roots):
    """
    Mantiene download su root primaria (models-default) ma rende disponibili i file
    anche in root secondaria (models) per nodi che usano path hardcoded models/LLM.
    """
    if os.environ.get("COMFYUI_SYNC_LLM_TO_SECONDARY", "0") != "1":
        _bootstrap_trace("_sync_llm_primary_to_secondary: disabled by env")
        return

    if len(model_roots) < 2:
        _bootstrap_trace("_sync_llm_primary_to_secondary: skipped because fewer than 2 model roots")
        return

    primary_llm = os.path.join(model_roots[0], "LLM")
    secondary_llm = os.path.join(model_roots[1], "LLM")

    try:
        _bootstrap_trace(f"_sync_llm_primary_to_secondary: ensuring primary {primary_llm}")
        os.makedirs(primary_llm, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to prepare primary LLM folder {primary_llm}: {exc}")
        _bootstrap_trace(f"_sync_llm_primary_to_secondary: failed preparing primary -> {exc}")
        return

    if os.path.realpath(primary_llm) == os.path.realpath(secondary_llm):
        _bootstrap_trace("_sync_llm_primary_to_secondary: primary and secondary already match")
        return

    if not os.path.exists(secondary_llm):
        try:
            os.symlink(primary_llm, secondary_llm, target_is_directory=True)
            logging.info(f"Linked LLM folder: {secondary_llm} -> {primary_llm}")
            _bootstrap_trace(f"_sync_llm_primary_to_secondary: linked {secondary_llm} -> {primary_llm}")
            return
        except Exception:
            pass

    try:
        _bootstrap_trace(f"_sync_llm_primary_to_secondary: copying contents {primary_llm} -> {secondary_llm}")
        os.makedirs(secondary_llm, exist_ok=True)
        for entry_name in os.listdir(primary_llm):
            src = os.path.join(primary_llm, entry_name)
            dst = os.path.join(secondary_llm, entry_name)
            if os.path.exists(dst):
                continue
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        logging.info(f"Synced LLM files from {primary_llm} to {secondary_llm}")
        _bootstrap_trace(f"_sync_llm_primary_to_secondary: sync completed {primary_llm} -> {secondary_llm}")
    except Exception as exc:
        logging.warning(f"Unable to sync LLM folders {primary_llm} -> {secondary_llm}: {exc}")
        _bootstrap_trace(f"_sync_llm_primary_to_secondary: sync failed -> {exc}")

def _try_link_or_copy_file(src_path: str, dest_path: str) -> bool:
    """
    Prova a rendere disponibile un file grande evitando duplicazioni disco:
    1) hardlink (stesso filesystem)
    2) symlink (fallback)
    3) copia (ultimo fallback)
    """
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    if not os.path.isfile(src_path) or os.path.getsize(src_path) <= 0:
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        if os.path.exists(dest_path) and os.path.getsize(dest_path) <= 0:
            os.remove(dest_path)
    except Exception:
        pass

    try:
        os.link(src_path, dest_path)
        return True
    except Exception:
        pass

    try:
        rel_target = os.path.relpath(src_path, os.path.dirname(dest_path))
        os.symlink(rel_target, dest_path)
        return True
    except Exception:
        pass

    try:
        shutil.copy2(src_path, dest_path)
        return True
    except Exception:
        return False


def _sync_model_alias_directories(model_roots):
    """
    Alcuni selector/nodi continuano a leggere solo le cartelle canoniche ComfyUI
    (es. controlnet) anche se registriamo alias aggiuntivi. Manteniamo quindi i file
    visibili in tutte le root rilevanti e in entrambe le posizioni senza duplicare
    inutilmente il contenuto.
    """
    alias_pairs = [
        ("controlnet", "xlabs/controlnets"),
    ]

    normalized_roots = []
    seen_roots = set()
    for root in model_roots:
        normalized = os.path.abspath(root)
        if normalized in seen_roots:
            continue
        seen_roots.add(normalized)
        normalized_roots.append(normalized)

    for canonical_subdir, alias_subdir in alias_pairs:
        discovered_files = {}

        for root in normalized_roots:
            for subdir in (canonical_subdir, alias_subdir):
                source_dir = os.path.join(root, subdir)
                if not os.path.isdir(source_dir):
                    continue

                try:
                    entry_names = os.listdir(source_dir)
                except Exception:
                    continue

                for entry_name in entry_names:
                    src_path = os.path.join(source_dir, entry_name)
                    if os.path.isdir(src_path):
                        continue
                    discovered_files.setdefault(entry_name, src_path)

        if not discovered_files:
            continue

        for root in normalized_roots:
            target_dirs = [
                os.path.join(root, canonical_subdir),
                os.path.join(root, alias_subdir),
            ]

            for target_dir in target_dirs:
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception as exc:
                    logging.warning(f"Unable to prepare model alias directory {target_dir}: {exc}")
                    continue

                for entry_name, src_path in discovered_files.items():
                    dst_path = os.path.join(target_dir, entry_name)
                    if os.path.abspath(src_path) == os.path.abspath(dst_path):
                        continue

                    if _try_link_or_copy_file(src_path, dst_path):
                        _bootstrap_trace(
                            f"_sync_model_alias_directories: mirrored {src_path} -> {dst_path}"
                        )


def _try_hf_snapshot_download(repo_id: str, local_dir: str, revision: str = "main", ignore_patterns=None) -> bool:
    """
    Scarica l'intero snapshot HF dentro una cartella locale (tutti i file del repo),
    utile per modelli che richiedono tokenizer/processor/config vari.
    """
    if os.environ.get("COMFYUI_FLORENCE2_SNAPSHOT", "1") != "1":
        return False

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        logging.info(f"huggingface_hub non disponibile, salto snapshot_download: {exc}")
        return False

    os.makedirs(local_dir, exist_ok=True)

    # best-effort: alcune versioni possono non supportare alcuni kwargs.
    base_kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": local_dir,
    }
    try_kwargs = []
    try_kwargs.append({**base_kwargs, "ignore_patterns": ignore_patterns or []})
    try_kwargs.append({**base_kwargs})

    last_exc = None
    for kwargs in try_kwargs:
        try:
            snapshot_download(**kwargs)
            logging.info(f"HF snapshot downloaded: {repo_id}@{revision} -> {local_dir}")
            return True
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            break

    logging.warning(f"HF snapshot download failed for {repo_id}@{revision}: {last_exc}")
    return False


def _ensure_florence2_layout(model_roots):
    """
    Florence2ModelLoader in genere cerca una CARTELLA modello dentro LLM,
    non un singolo file .safetensors nella root LLM.
    Costruisce un layout compatibile: LLM/Florence-2-large/...
    """
    repo_id = os.environ.get("COMFYUI_FLORENCE2_REPO", "microsoft/Florence-2-large").strip() or "microsoft/Florence-2-large"
    revision = os.environ.get("COMFYUI_FLORENCE2_REVISION", "main").strip() or "main"
    hf_base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
    required_files = [
        "config.json",
        "configuration_florence2.py",
        "generation_config.json",
        "modeling_florence2.py",
        "preprocessor_config.json",
        "processor_config.json",
        "processing_florence2.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    optional_files = [
        # Alcuni snapshot Florence non espongono questi file; se assenti non e' un errore.
        "special_tokens_map.json",
        "merges.txt",
    ]

    if not model_roots:
        _bootstrap_trace("_ensure_florence2_layout: skipped because model_roots is empty")
        return

    # Scarica SOLO nella root primaria (models-default).
    root = model_roots[0]
    llm_root = os.path.join(root, "LLM")
    model_dir = os.path.join(llm_root, "Florence-2-large")
    _bootstrap_trace(f"_ensure_florence2_layout: start for {model_dir}")

    try:
        os.makedirs(model_dir, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to create Florence-2-large dir in {root}: {exc}")
        _bootstrap_trace(f"_ensure_florence2_layout: failed creating model dir -> {exc}")
        return

    # Compat con i file già presenti nella root LLM (naming legacy del bootstrap),
    # evitando copie inutili (hardlink/symlink se possibile).
    legacy_weights = [
        ("florence-2-large-model.safetensors", "model.safetensors", False),
        ("florence-2-large-pytorch_model.bin", "pytorch_model.bin", True),  # opzionale: non sempre serve
    ]
    for src_name, dst_name, optional in legacy_weights:
        src = os.path.join(llm_root, src_name)
        dst = os.path.join(model_dir, dst_name)
        _bootstrap_trace(f"_ensure_florence2_layout: preparing legacy weight {dst_name}")
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            continue
        if _try_link_or_copy_file(src, dst):
            logging.info(f"Prepared Florence-2-large file: {dst}")
            continue
        if optional:
            continue
        # Fallback: prova a scaricare direttamente il peso se non presente altrove.
        _download_if_missing(f"{hf_base}/{dst_name}", dst, ignore_http_404=True)

    # Snapshot completo del repo HF: scarica tutti i file presenti su
    # https://huggingface.co/<repo_id>/tree/<revision>
    # Evita di scaricare di nuovo i pesi se già preparati sopra.
    ignore_snapshot = []
    if os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        ignore_snapshot.append("model.safetensors")
    if os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")):
        ignore_snapshot.append("pytorch_model.bin")
    _try_hf_snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        revision=revision,
        ignore_patterns=ignore_snapshot,
    )
    _bootstrap_trace(f"_ensure_florence2_layout: snapshot step completed for {model_dir}")

    for filename in required_files:
        _bootstrap_trace(f"_ensure_florence2_layout: ensuring required file {filename}")
        _download_if_missing(f"{hf_base}/{filename}", os.path.join(model_dir, filename))

    for filename in optional_files:
        _bootstrap_trace(f"_ensure_florence2_layout: ensuring optional file {filename}")
        _download_if_missing(
            f"{hf_base}/{filename}",
            os.path.join(model_dir, filename),
            ignore_http_404=True,
        )

    # Compat: alcuni config Florence locali non includono questi campi e alcuni
    # loader vanno in AttributeError su Florence2LanguageConfig.
    # Nota: alcuni checkpoint li mettono in text_config, altri in language_config.
    config_path = os.path.join(model_dir, "config.json")
    try:
        import json

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_data = json.load(config_file)

        changed = False

        compat_keys = (
            "forced_bos_token_id",
            "forced_eos_token_id",
            "suppress_tokens",
            "begin_suppress_tokens",
        )

        for section_name in ("text_config", "language_config"):
            section = config_data.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in compat_keys:
                if key not in section:
                    section[key] = None
                    changed = True

        if changed:
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(config_data, config_file, ensure_ascii=False, indent=2)
                config_file.write("\n")
            logging.info("Patched Florence-2 config compatibility fields in text_config/language_config")
    except Exception as exc:
        logging.warning(f"Unable to patch Florence-2 config compatibility fields: {exc}")

    # Alias lowercase utile per nodi che usano nomi cartella in minuscolo.
    lower_alias = os.path.join(llm_root, "florence-2-large")
    if not os.path.exists(lower_alias):
        try:
            os.symlink(model_dir, lower_alias, target_is_directory=True)
        except Exception:
            try:
                shutil.copytree(model_dir, lower_alias, dirs_exist_ok=True)
            except Exception as exc:
                logging.warning(f"Unable to create lowercase Florence alias {lower_alias}: {exc}")
                _bootstrap_trace(f"_ensure_florence2_layout: lowercase alias failed -> {exc}")

    _bootstrap_trace(f"_ensure_florence2_layout: completed for {model_dir}")


def _ensure_da3_large_layout(model_roots):
    """
    Prepara il layout locale per Depth Anything 3 Large dentro models/LLM.
    Il repo HF espone almeno config.json e model.safetensors, quindi conviene
    mantenere una cartella repo-like invece di un file singolo nella root LLM.
    """
    repo_id = os.environ.get("COMFYUI_DA3_LARGE_REPO", "depth-anything/DA3-LARGE").strip() or "depth-anything/DA3-LARGE"
    revision = os.environ.get("COMFYUI_DA3_LARGE_REVISION", "main").strip() or "main"
    hf_base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
    required_files = [
        "config.json",
        "model.safetensors",
    ]

    if not model_roots:
        _bootstrap_trace("_ensure_da3_large_layout: skipped because model_roots is empty")
        return

    root = model_roots[0]
    llm_root = os.path.join(root, "LLM")
    model_dir = os.path.join(llm_root, "DA3-LARGE")
    _bootstrap_trace(f"_ensure_da3_large_layout: start for {model_dir}")

    try:
        os.makedirs(model_dir, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to create DA3-LARGE dir in {root}: {exc}")
        _bootstrap_trace(f"_ensure_da3_large_layout: failed creating model dir -> {exc}")
        return

    ignore_snapshot = []
    if os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        ignore_snapshot.append("model.safetensors")
    _try_hf_snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        revision=revision,
        ignore_patterns=ignore_snapshot,
    )
    _bootstrap_trace(f"_ensure_da3_large_layout: snapshot step completed for {model_dir}")

    for filename in required_files:
        _bootstrap_trace(f"_ensure_da3_large_layout: ensuring required file {filename}")
        _download_if_missing(f"{hf_base}/{filename}", os.path.join(model_dir, filename))

    lower_alias = os.path.join(llm_root, "da3-large")
    if not os.path.exists(lower_alias):
        try:
            os.symlink(model_dir, lower_alias, target_is_directory=True)
        except Exception:
            try:
                shutil.copytree(model_dir, lower_alias, dirs_exist_ok=True)
            except Exception as exc:
                logging.warning(f"Unable to create lowercase DA3 alias {lower_alias}: {exc}")
                _bootstrap_trace(f"_ensure_da3_large_layout: lowercase alias failed -> {exc}")

    _bootstrap_trace(f"_ensure_da3_large_layout: completed for {model_dir}")


def apply_shared_model_paths():
    """
    Registra più cartelle modelli condivise e scarica automaticamente i modelli mancanti
    dalla cartella principale (prima root) usando SHARED_MODELS_URLS.
    """
    model_roots = _resolve_model_roots()
    _bootstrap_trace(f"apply_shared_model_paths: model_roots={model_roots}")

    if not model_roots:
        _bootstrap_trace("apply_shared_model_paths: skipped because no model roots were resolved")
        return

    # Crea le root (se vuoi che esistano). Se una non esiste, ComfyUI leggerà solo quelle presenti.
    for root in model_roots:
        _bootstrap_trace(f"apply_shared_model_paths: ensuring root {root}")
        os.makedirs(root, exist_ok=True)
        logging.info(f"Using models root: {root}")

    _bootstrap_trace("apply_shared_model_paths: ensure LLM subdirs begin")
    _ensure_llm_subdirs(model_roots)
    _bootstrap_trace("apply_shared_model_paths: ensure LLM subdirs completed")

    # Scarica modelli mancanti SOLO nella prima root (quella principale)
    # così non alteri la seconda cartella
    _bootstrap_trace(f"apply_shared_model_paths: ensure shared models begin on {model_roots[0]}")
    ensure_shared_models_downloaded(model_roots[0])
    _bootstrap_trace("apply_shared_model_paths: ensure shared models completed")
    _bootstrap_trace("apply_shared_model_paths: first LLM sync begin")
    _sync_llm_primary_to_secondary(model_roots)
    _bootstrap_trace("apply_shared_model_paths: first LLM sync completed")
    _bootstrap_trace("apply_shared_model_paths: Florence2 layout begin")
    _ensure_florence2_layout(model_roots)
    _bootstrap_trace("apply_shared_model_paths: Florence2 layout completed")
    _bootstrap_trace("apply_shared_model_paths: DA3 layout begin")
    _ensure_da3_large_layout(model_roots)
    _bootstrap_trace("apply_shared_model_paths: DA3 layout completed")
    _bootstrap_trace("apply_shared_model_paths: second LLM sync begin")
    _sync_llm_primary_to_secondary(model_roots)
    _bootstrap_trace("apply_shared_model_paths: second LLM sync completed")
    _bootstrap_trace("apply_shared_model_paths: model alias sync begin")
    _sync_model_alias_directories(model_roots)
    _bootstrap_trace("apply_shared_model_paths: model alias sync completed")

    iter_model_dir_bindings = globals().get("_iter_model_dir_bindings")
    if callable(iter_model_dir_bindings):
        model_dir_bindings = list(iter_model_dir_bindings())
    else:
        # Fallback sicuro: apply_custom_paths() gira prima dei helper definiti nel wrapper.
        base_model_dirs = {
            "checkpoints": "checkpoints",
            "loras": "loras",
            "vae": "vae",
            "clip": "clip",
            "inpaint": "inpaint",
            "diffusion_models": "diffusion_models",
            "transformer": "diffusion_models",
            "embeddings": "embeddings",
            "controlnet": "controlnet",
            "upscale_models": "upscale_models",
            "clip_vision": "clip_vision",
            "ipadapter": "ipadapter",
            "style_models": "style_models",
            "gligen": "gligen",
            "hypernetworks": "hypernetworks",
            "vae_approx": "vae_approx",
            "unet": "unet",
            "text_encoders": "text_encoders",
            "t5": "text_encoders",
            "clip_l": "text_encoders",
            "LLM": "LLM",
            "llm": "LLM",
        }
        alias_model_dirs = {
            "controlnet": [
                "xlabs/controlnets",
            ],
        }

        model_dir_bindings = []
        seen_bindings = set()
        for model_type, subdir in base_model_dirs.items():
            binding = (model_type, subdir)
            if binding in seen_bindings:
                continue
            seen_bindings.add(binding)
            model_dir_bindings.append(binding)

        for model_type, extra_subdirs in alias_model_dirs.items():
            for subdir in extra_subdirs:
                binding = (model_type, subdir)
                if binding in seen_bindings:
                    continue
                seen_bindings.add(binding)
                model_dir_bindings.append(binding)

    # Aggiunge TUTTE le cartelle per ogni tipo modello
    for root in model_roots:
        for model_type, subdir in model_dir_bindings:
            p = os.path.join(root, subdir)
            if os.path.isdir(p):
                folder_paths.add_model_folder_path(model_type, p)
                logging.info(f"Added model path [{model_type}] -> {p}")
                _bootstrap_trace(f"apply_shared_model_paths: registered {model_type} -> {p}")

    _bootstrap_trace("apply_shared_model_paths: completed")

def apply_custom_paths():
    # extra model paths
    _bootstrap_trace("apply_custom_paths: begin")
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        _bootstrap_trace(f"apply_custom_paths: loading default extra model config {extra_model_paths_config_path}")
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)
        _bootstrap_trace("apply_custom_paths: default extra model config loaded")

    if args.extra_model_paths_config:
        _bootstrap_trace(f"apply_custom_paths: loading CLI extra model configs {args.extra_model_paths_config}")
        for config_path in itertools.chain(*args.extra_model_paths_config):
            _bootstrap_trace(f"apply_custom_paths: loading CLI config {config_path}")
            utils.extra_config.load_extra_path_config(config_path)
            _bootstrap_trace(f"apply_custom_paths: loaded CLI config {config_path}")

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)
        _bootstrap_trace(f"apply_custom_paths: output directory set to {output_dir}")

    # NUOVO: cartella modelli condivisa (+ download automatico se mancano)
    _bootstrap_trace("apply_custom_paths: apply_shared_model_paths begin")
    apply_shared_model_paths()
    _bootstrap_trace("apply_custom_paths: apply_shared_model_paths completed")

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    _bootstrap_trace("apply_custom_paths: registering output subdirectories")
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))
    _bootstrap_trace("apply_custom_paths: output subdirectories registered")

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)
        _bootstrap_trace(f"apply_custom_paths: input directory set to {input_dir}")

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)
        _bootstrap_trace(f"apply_custom_paths: user directory set to {user_dir}")

    _bootstrap_trace("apply_custom_paths: completed")


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        _bootstrap_trace("prestartup: skipped because all custom nodes are disabled")
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            _bootstrap_trace(f"prestartup: executing {script_path}")
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _bootstrap_trace(f"prestartup: completed {script_path}")
            return True
        except Exception as e:
            _bootstrap_trace(f"prestartup: failed {script_path} -> {e}")
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    _bootstrap_trace(f"prestartup: scanning custom node paths {node_paths}")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")


_bootstrap_trace("startup: apply_custom_paths begin")
apply_custom_paths()
_bootstrap_trace("startup: apply_custom_paths completed")
execute_prestartup_script()
_bootstrap_trace("startup: prestartup scripts completed")

# ===== WRAPPER STABILE PER COMFYUI (compatibile con update futuri) =====
# Sostituisce tutto il blocco "# Main code" e il vecchio if __name__ == "__main__"

from pathlib import Path
import runpy
import logging
_current_pytorch_alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF", "").strip()
if "expandable_segments:True" not in _current_pytorch_alloc_conf.split(","):
    os.environ["PYTORCH_ALLOC_CONF"] = ",".join(
        part for part in [_current_pytorch_alloc_conf, "expandable_segments:True"] if part
    )

def _ensure_transformers_clipfeatureextractor_compat():
    """
    Compat per custom nodes legacy (es. comfyui-fluxtrainer) che importano
    CLIPFeatureExtractor, rimosso nelle versioni recenti di transformers.
    """
    try:
        import transformers
    except Exception as e:


        logging.warning(f"[WRAPPER] transformers non importabile per patch compat: {e}")
        return

    if hasattr(transformers, "CLIPFeatureExtractor"):
        _ensure_transformers_encoderdecodercache_compat(transformers)
        return

    clip_image_processor = None
    try:
        if hasattr(transformers, "CLIPImageProcessor"):
            clip_image_processor = transformers.CLIPImageProcessor
    except Exception:
        clip_image_processor = None

    if clip_image_processor is None:
        try:
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor

            clip_image_processor = CLIPImageProcessor
        except Exception as exc:
            logging.warning(f"[WRAPPER] CLIPImageProcessor non disponibile per patch compat: {exc}")
            return

    try:
        transformers.CLIPFeatureExtractor = clip_image_processor

        # transformers usa un LazyModule: aggiorniamo anche la struttura di import
        # per rendere affidabile `from transformers import CLIPFeatureExtractor`.
        try:
            import_utils = getattr(transformers, "utils", None)
            if import_utils is not None:
                import_utils = getattr(import_utils, "import_utils", None)
            if import_utils is not None and hasattr(import_utils, "_import_structure"):
                clip_exports = import_utils._import_structure.setdefault("models.clip", [])
                if "CLIPFeatureExtractor" not in clip_exports:
                    clip_exports.append("CLIPFeatureExtractor")
            module_all = getattr(transformers, "__all__", None)
            if isinstance(module_all, list) and "CLIPFeatureExtractor" not in module_all:
                module_all.append("CLIPFeatureExtractor")
        except Exception as compat_exc:
            logging.warning(f"[WRAPPER] Lazy import compat patch warning: {compat_exc}")

        logging.info("[WRAPPER] Applied transformers compat alias: CLIPFeatureExtractor -> CLIPImageProcessor")
    except Exception as exc:
        logging.warning(f"[WRAPPER] Failed applying transformers compat alias: {exc}")

    _ensure_transformers_encoderdecodercache_compat(transformers)


def _detect_fluxtrainer_custom_node(custom_nodes_dir: str):
    """
    Rileva FluxTrainer in modo tollerante rispetto a varianti naming della cartella.
    Ritorna una tupla (found: bool, reason: str).
    """
    if not os.path.isdir(custom_nodes_dir):
        return False, "custom_nodes directory not found"

    explicit_candidates = {
        "comfyui-fluxtrainer",
        "comfyui-flux-trainer",
        "comfyui_fluxtrainer",
        "comfyui_flux_trainer",
        "comfyui-fluxtrainer-node",
        "comfyui-flux-trainer-node",
        "comfyui-fluxtrainer-main",
        "comfyui-flux-trainer-main",
        "fluxtrainer",
        "flux-trainer",
        "flux_trainer",
    }

    try:
        for entry_name in os.listdir(custom_nodes_dir):
            if entry_name.endswith(".disabled"):
                continue

            full_path = os.path.join(custom_nodes_dir, entry_name)
            if not os.path.isdir(full_path):
                continue

            normalized = entry_name.strip().lower()
            if normalized in explicit_candidates:
                return True, f"matched folder '{entry_name}'"

            if "fluxtrainer" in normalized or "flux-trainer" in normalized or "flux_trainer" in normalized:
                return True, f"matched fuzzy folder '{entry_name}'"
    except Exception as exc:
        return False, f"error while scanning custom_nodes: {exc}"

    return False, "no FluxTrainer-like folder found"

# Mappa cartelle modelli (stessa logica dei tuoi path)
MODEL_DIRS_MAP = {
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "clip": "clip",
    "inpaint": "inpaint",
    "diffusion_models": "diffusion_models",
    "transformer": "diffusion_models",
    "embeddings": "embeddings",
    "controlnet": "controlnet",
    "upscale_models": "upscale_models",
    "clip_vision": "clip_vision",
    "ipadapter": "ipadapter",
    "style_models": "style_models",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "vae_approx": "vae_approx",
    "unet": "unet",
    "text_encoders": "text_encoders",
    "t5": "text_encoders",
    "clip_l": "text_encoders",
    "LLM": "LLM",
    "llm": "LLM",
}

MODEL_DIR_ALIASES_MAP = {
    # XLabs salva i controlnet Flux in una sottocartella dedicata.
    "controlnet": [
        "xlabs/controlnets",
    ],
}


def _iter_model_dir_bindings():
    seen = set()

    for model_type, subdir in MODEL_DIRS_MAP.items():
        binding = (model_type, subdir)
        if binding in seen:
            continue
        seen.add(binding)
        yield binding

    for model_type, extra_subdirs in MODEL_DIR_ALIASES_MAP.items():
        for subdir in extra_subdirs:
            binding = (model_type, subdir)
            if binding in seen:
                continue
            seen.add(binding)
            yield binding


def _build_extra_model_paths_config(model_roots: list[str]):
    data = {}
    for idx, root in enumerate(model_roots, start=1):
        root = os.path.abspath(root)
        entry_name = f"shared_models_{idx}"
        entry = {"base_path": root}
        entry.update(MODEL_DIRS_MAP)
        data[entry_name] = entry

        alias_index = 0
        for model_type, subdir in _iter_model_dir_bindings():
            if MODEL_DIRS_MAP.get(model_type) == subdir:
                continue

            alias_index += 1
            data[f"{entry_name}_alias_{alias_index}"] = {
                "base_path": root,
                model_type: subdir,
            }

    return data


MODEL_FILENAME_ALIASES = {
    # Alcuni workflow esportati referenziano il nome Civitai dello stesso OpenPose SD1.5.
    "controlV11pSd15_v10.safetensors": "control_v11p_sd15_openpose_fp16.safetensors",
    # Compat tra basename del download URL Civitai e filename finale salvato localmente.
    "juggernautxlRagnarok.k3mq.safetensors": "juggernautXL_ragnarokBy.safetensors",
}


def _normalize_known_model_alias(value):
    if not isinstance(value, str):
        return value

    normalized_value = value.replace("\\", "/")

    if os.path.isabs(value):
        dir_name, base_name = os.path.split(value)
        aliased_name = MODEL_FILENAME_ALIASES.get(base_name)
        if aliased_name:
            return os.path.join(dir_name, aliased_name)
        return value

    if "/" in normalized_value:
        dir_name, base_name = normalized_value.rsplit("/", 1)
        aliased_name = MODEL_FILENAME_ALIASES.get(base_name)
        if aliased_name:
            return f"{dir_name}/{aliased_name}"
        return normalized_value

    return MODEL_FILENAME_ALIASES.get(normalized_value, value)


def _normalize_registered_model_path(value):
    value = _normalize_known_model_alias(value)
    if not isinstance(value, str) or not os.path.isabs(value):
        return value

    normalized_value = os.path.abspath(value)
    best_match = None
    best_prefix_len = -1

    for root in _resolve_model_roots():
        for _, subdir in _iter_model_dir_bindings():
            registered_dir = os.path.abspath(os.path.join(root, subdir))
            try:
                if os.path.commonpath([normalized_value, registered_dir]) != registered_dir:
                    continue
            except ValueError:
                continue

            try:
                relative_path = os.path.relpath(normalized_value, registered_dir)
            except ValueError:
                continue

            if relative_path == os.pardir or relative_path.startswith(f"{os.pardir}{os.sep}"):
                continue

            prefix_len = len(registered_dir)
            if prefix_len > best_prefix_len:
                best_match = relative_path.replace("\\", "/")
                best_prefix_len = prefix_len

    return best_match or value


MODEL_ROOTS = [
    # Resta come fallback statico, ma il wrapper usa _resolve_model_roots().
    "/mnt/default-models",
    "/vscode/workspace/models",
]

def _write_auto_extra_model_paths_yaml(config_path: str, model_roots: list[str]):
    """
    Genera un file extra_model_paths YAML che ComfyUI carica nativamente.
    Questo evita di toccare folder_paths.add_model_folder_path nel core runtime.
    """
    try:
        import yaml  # PyYAML (già installato dal tuo bootstrap)
    except Exception as e:
        raise RuntimeError(f"PyYAML non disponibile per generare extra_model_paths: {e}")

    data = _build_extra_model_paths_config(model_roots)

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    logging.info(f"[WRAPPER] Generated extra model paths config: {config_path}")
    return config_path


def _append_extra_model_paths_arg(config_path: str):
    """
    Aggiunge il config auto ai parametri di avvio ComfyUI.
    Non rimuove gli eventuali config già passati dall'utente.
    """
    for index, token in enumerate(sys.argv):
        if token == "--extra-model-paths-config" and index + 1 < len(sys.argv):
            if os.path.abspath(sys.argv[index + 1]) == os.path.abspath(config_path):
                logging.info(f"[WRAPPER] CLI arg already present --extra-model-paths-config {config_path}")
                return

    sys.argv.extend(["--extra-model-paths-config", config_path])
    logging.info(f"[WRAPPER] Added CLI arg --extra-model-paths-config {config_path}")


def _cleanup_broken_manager_json_cache():
    """
    Se un aggiornamento ComfyRegistry viene interrotto, possono restare JSON troncati
    che causano JSONDecodeError in comfyui-manager. Rinomina solo i file corrotti.
    """
    if os.environ.get("COMFYUI_MANAGER_CLEANUP_BROKEN_CACHE", "1") != "1":
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    manager_dir = os.path.join(base_dir, "custom_nodes", COMFYUI_MANAGER_DIRNAME)

    cache_roots = [
        os.path.join(manager_dir, ".cache"),
        os.path.join(manager_dir, "cache"),
        os.path.join(base_dir, ".cache", "comfyui-manager"),
    ]

    renamed = 0
    for cache_root in cache_roots:
        if not os.path.isdir(cache_root):
            continue

        for root, _, files in os.walk(cache_root):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue

                file_path = os.path.join(root, name)
                try:
                    with open(file_path, "r", encoding="utf-8") as json_file:
                        import json

                        json.load(json_file)
                except json.JSONDecodeError as exc:
                    backup_path = f"{file_path}.corrupt"
                    try:
                        if os.path.exists(backup_path):
                            os.remove(backup_path)
                        os.replace(file_path, backup_path)
                        renamed += 1
                        logging.warning(
                            "[WRAPPER] Renamed corrupt manager cache JSON: %s -> %s (%s)",
                            file_path,
                            backup_path,
                            exc,
                        )
                    except Exception as move_exc:
                        logging.warning(
                            "[WRAPPER] Failed handling corrupt manager cache JSON %s: %s",
                            file_path,
                            move_exc,
                        )
                except Exception:
                    # Ignora file non leggibili o lock temporanei.
                    pass

    if renamed:
        logging.info("[WRAPPER] Cleaned %d corrupt ComfyUI-Manager cache JSON file(s)", renamed)


def _get_comfyui_manager_config_path():
    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    if not callable(get_user_directory):
        return None

    try:
        user_dir = os.path.abspath(get_user_directory())
    except Exception as exc:
        logging.warning(f"[WRAPPER] Unable to resolve ComfyUI user directory for manager config: {exc}")
        return None

    if hasattr(folder_paths, "get_system_user_directory"):
        manager_files_path = os.path.join(user_dir, "__manager")
    else:
        manager_files_path = os.path.join(user_dir, "default", "ComfyUI-Manager")

    return os.path.join(manager_files_path, "config.ini")


def _is_comfyregistry_reachable(timeout=5):
    probe_url = (
        os.environ.get("COMFYUI_MANAGER_REGISTRY_PROBE_URL", "").strip()
        or "https://api.comfy.org/nodes?page=1&limit=1"
    )
    request = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "ComfyUI-Manager-NetworkProbe/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 500
    except Exception as exc:
        logging.warning(f"[WRAPPER] ComfyRegistry probe failed ({probe_url}): {exc}")
        return False


def _ensure_comfyui_manager_network_mode():
    """
    Permette di forzare `network_mode` via env e, in auto mode, ripiega su
    `offline` quando ComfyRegistry non e' raggiungibile per evitare startup lenti
    o apparentemente bloccati nel fetch iniziale.
    """
    if os.environ.get("COMFYUI_MANAGER_AUTO_CONFIGURE_NETWORK_MODE", "1") != "1":
        return

    config_path = _get_comfyui_manager_config_path()
    if not config_path:
        _bootstrap_trace("_ensure_comfyui_manager_network_mode: skipped because config path is unavailable")
        return

    import configparser

    config = configparser.ConfigParser(strict=False)
    if os.path.isfile(config_path):
        config.read(config_path)

    if "default" not in config:
        config["default"] = {}

    current_mode = (config["default"].get("network_mode") or "public").strip().lower() or "public"
    requested_mode = os.environ.get("COMFYUI_MANAGER_NETWORK_MODE", "").strip().lower()
    valid_modes = {"public", "private", "offline"}

    if requested_mode and requested_mode not in valid_modes:
        logging.warning(f"[WRAPPER] Ignoring unsupported COMFYUI_MANAGER_NETWORK_MODE='{requested_mode}'")
        requested_mode = ""

    target_mode = current_mode
    change_reason = None

    if requested_mode:
        target_mode = requested_mode
        if target_mode != current_mode:
            change_reason = f"env override ({target_mode})"
    else:
        auto_offline_enabled = os.environ.get("COMFYUI_MANAGER_AUTO_OFFLINE_ON_REGISTRY_FAILURE", "1") == "1"
        try:
            probe_timeout = int(os.environ.get("COMFYUI_MANAGER_REGISTRY_PROBE_TIMEOUT", "5"))
        except ValueError:
            probe_timeout = 5

        if auto_offline_enabled and current_mode == "public" and not _is_comfyregistry_reachable(timeout=probe_timeout):
            target_mode = "offline"
            change_reason = "ComfyRegistry unreachable"

    if target_mode == current_mode and os.path.isfile(config_path):
        _bootstrap_trace(
            f"_ensure_comfyui_manager_network_mode: keeping network_mode={current_mode} ({config_path})"
        )
        return

    config["default"]["network_mode"] = target_mode
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as config_file:
        config.write(config_file)

    logging.info(
        "[WRAPPER] Set ComfyUI-Manager network_mode=%s in %s%s",
        target_mode,
        config_path,
        f" ({change_reason})" if change_reason else "",
    )
    _bootstrap_trace(
        f"_ensure_comfyui_manager_network_mode: wrote network_mode={target_mode} to {config_path}"
    )


def _append_disable_cuda_malloc_arg():
    """
    Evita mismatch allocator (cudaMallocAsync/native) quando torch viene
    importato prima del bootstrap CUDA di ComfyUI.
    """
    if "--disable-cuda-malloc" in sys.argv:
        return

    # Permette override esplicito se serve testare cudaMallocAsync.
    if os.environ.get("COMFYUI_FORCE_CUDA_MALLOC", "0") == "1":
        logging.info("[WRAPPER] COMFYUI_FORCE_CUDA_MALLOC=1, keeping cudaMallocAsync behavior")
        return

    sys.argv.append("--disable-cuda-malloc")
    logging.info("[WRAPPER] Added CLI arg --disable-cuda-malloc to keep allocator stable")


def _normalize_flux_workflow_paths(base_dir: str):
    """
    Su Linux/macOS, alcuni workflow salvati da ambienti Windows possono contenere
    path modello con backslash (es. t5\\google...). Li normalizza a slash.
    """
    if os.name == "nt":
        return

    if os.environ.get("COMFYUI_NORMALIZE_WORKFLOW_PATHS", "1") != "1":
        return

    search_dirs = [
        os.path.join(base_dir, "user"),
        os.path.join(base_dir, "workflows"),
    ]

    replacements = {
        "t5\\\\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
        "t5\\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
    }

    patched_files = 0
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        for root, _, files in os.walk(search_dir):
            for name in files:
                if not name.lower().endswith(".json"):
                    continue

                file_path = os.path.join(root, name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    new_content = content
                    for src, dst in replacements.items():
                        new_content = new_content.replace(src, dst)

                    if new_content == content:
                        continue

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)

                    patched_files += 1
                except Exception as exc:
                    logging.warning(f"[WRAPPER] Unable to normalize workflow file {file_path}: {exc}")

    if patched_files:
        logging.info(f"[WRAPPER] Normalized Flux workflow path separators in {patched_files} JSON file(s)")


FLUX_PROMPT_PATH_REPLACEMENTS = {
    "t5\\\\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
    "t5\\google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors",
}


def _normalize_flux_prompt_value(value):
    if not isinstance(value, str):
        return value

    new_value = value
    for source, target in FLUX_PROMPT_PATH_REPLACEMENTS.items():
        new_value = new_value.replace(source, target)

    new_value = _normalize_registered_model_path(new_value)
    return new_value


def _normalize_prompt_payload_paths(payload):
    if isinstance(payload, dict):
        return {key: _normalize_prompt_payload_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_prompt_payload_paths(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_normalize_prompt_payload_paths(value) for value in payload)
    return _normalize_flux_prompt_value(payload)


_KNOWN_BOOTSTRAP_MODELS_BY_TYPE = None
_KNOWN_BOOTSTRAP_MODEL_SOURCES = None


def _get_bootstrap_model_index():
    global _KNOWN_BOOTSTRAP_MODELS_BY_TYPE, _KNOWN_BOOTSTRAP_MODEL_SOURCES

    if _KNOWN_BOOTSTRAP_MODELS_BY_TYPE is not None and _KNOWN_BOOTSTRAP_MODEL_SOURCES is not None:
        return _KNOWN_BOOTSTRAP_MODELS_BY_TYPE, _KNOWN_BOOTSTRAP_MODEL_SOURCES

    known_by_type = {}
    sources = {}

    for folder_name, entries in SHARED_MODELS_URLS.items():
        aliases = {folder_name}
        for model_type, mapped_folder in MODEL_DIRS_MAP.items():
            if mapped_folder == folder_name:
                aliases.add(model_type)

        for url, filename in _normalize_model_entries(entries):
            for model_type in aliases:
                known_by_type.setdefault(model_type, set()).add(filename)
                sources[(model_type, filename)] = (folder_name, url)

                for alias_name, canonical_name in MODEL_FILENAME_ALIASES.items():
                    if canonical_name != filename:
                        continue
                    known_by_type.setdefault(model_type, set()).add(alias_name)
                    sources[(model_type, alias_name)] = (folder_name, url)

    _KNOWN_BOOTSTRAP_MODELS_BY_TYPE = known_by_type
    _KNOWN_BOOTSTRAP_MODEL_SOURCES = sources
    return _KNOWN_BOOTSTRAP_MODELS_BY_TYPE, _KNOWN_BOOTSTRAP_MODEL_SOURCES


def _get_known_bootstrap_model_filenames(model_type):
    known_by_type, _ = _get_bootstrap_model_index()
    return sorted(known_by_type.get(model_type, set()))


def _invalidate_folder_paths_filename_cache(folder_paths_module, model_type):
    cache_attr_names = (
        "filename_list_cache",
        "cached_filename_list",
        "_filename_list_cache",
    )

    for attr_name in cache_attr_names:
        cache = getattr(folder_paths_module, attr_name, None)
        if isinstance(cache, dict):
            cache.pop(model_type, None)


def _ensure_known_bootstrap_model_available(model_type, filename):
    _, sources = _get_bootstrap_model_index()
    source = sources.get((model_type, filename))
    if not source:
        return None

    folder_name, url = source
    model_roots = _resolve_model_roots()
    if not model_roots:
        return None

    dest_path = os.path.join(model_roots[0], folder_name, filename)
    try:
        _download_if_missing(url, dest_path)
    except Exception as exc:
        logging.warning(
            f"[WRAPPER] Unable to prepare bootstrap model '{filename}' for '{model_type}': {exc}"
        )
        return None

    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path
    return None


def _install_known_model_selector_patch():
    """
    Espone ai selector anche i modelli dichiarati nel bootstrap locale, così
    i prompt non falliscono con "Value not in list" quando il file è previsto
    ma non è ancora stato indicizzato da folder_paths.
    """
    if os.environ.get("COMFYUI_PATCH_KNOWN_MODEL_SELECTORS", "1") != "1":
        logging.info("[WRAPPER] Known model selector patch disabled by env")
        return

    original_get_filename_list = getattr(folder_paths, "get_filename_list", None)
    if callable(original_get_filename_list) and not getattr(original_get_filename_list, "_comfyui_known_model_patch", False):
        def _wrapped_get_filename_list(model_type, *args, **kwargs):
            filenames = original_get_filename_list(model_type, *args, **kwargs) or []
            known = _get_known_bootstrap_model_filenames(model_type)
            if not known:
                return filenames

            merged = []
            seen = set()
            for item in list(filenames) + known:
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
            return merged

        _wrapped_get_filename_list._comfyui_known_model_patch = True
        folder_paths.get_filename_list = _wrapped_get_filename_list
        logging.info("[WRAPPER] Installed known model selector patch for folder_paths.get_filename_list")

    original_get_full_path = getattr(folder_paths, "get_full_path", None)
    if callable(original_get_full_path) and not getattr(original_get_full_path, "_comfyui_known_model_patch", False):
        def _wrapped_get_full_path(model_type, filename, *args, **kwargs):
            filename = _normalize_registered_model_path(filename)
            result = original_get_full_path(model_type, filename, *args, **kwargs)
            if result:
                return result

            if filename not in _get_known_bootstrap_model_filenames(model_type):
                return result

            prepared_path = _ensure_known_bootstrap_model_available(model_type, filename)
            if not prepared_path:
                return result

            _invalidate_folder_paths_filename_cache(folder_paths, model_type)
            retry_result = original_get_full_path(model_type, filename, *args, **kwargs)
            return retry_result or prepared_path

        _wrapped_get_full_path._comfyui_known_model_patch = True
        folder_paths.get_full_path = _wrapped_get_full_path
        logging.info("[WRAPPER] Installed on-demand bootstrap patch for folder_paths.get_full_path")

    original_get_full_path_or_raise = getattr(folder_paths, "get_full_path_or_raise", None)
    if callable(original_get_full_path_or_raise) and not getattr(original_get_full_path_or_raise, "_comfyui_known_model_patch", False):
        def _wrapped_get_full_path_or_raise(model_type, filename, *args, **kwargs):
            filename = _normalize_registered_model_path(filename)
            try:
                return original_get_full_path_or_raise(model_type, filename, *args, **kwargs)
            except Exception:
                prepared_path = _ensure_known_bootstrap_model_available(model_type, filename)
                if not prepared_path:
                    raise

                _invalidate_folder_paths_filename_cache(folder_paths, model_type)
                retry_result = getattr(folder_paths, "get_full_path", None)
                if callable(retry_result):
                    resolved = retry_result(model_type, filename, *args, **kwargs)
                    if resolved:
                        return resolved
                return prepared_path

        _wrapped_get_full_path_or_raise._comfyui_known_model_patch = True
        folder_paths.get_full_path_or_raise = _wrapped_get_full_path_or_raise
        logging.info("[WRAPPER] Installed on-demand bootstrap patch for folder_paths.get_full_path_or_raise")


def _install_prompt_path_normalization_patch():
    """
    Alcuni prompt inviati via API/UI possono contenere path modello con backslash
    (workflow salvati su Windows). Normalizza il payload prima della validazione,
    evitando errori "Value not in list" per selector come FluxTrainModelSelect.
    """
    if os.environ.get("COMFYUI_PATCH_PROMPT_PATH_NORMALIZATION", "1") != "1":
        logging.info("[WRAPPER] Prompt path normalization patch disabled by env")
        return

    try:
        import comfy_execution.validation as validation
    except Exception as exc:
        logging.warning(f"[WRAPPER] Unable to import comfy_execution.validation for prompt patch: {exc}")
        return

    original_validate_prompt = getattr(validation, "validate_prompt", None)
    if not callable(original_validate_prompt):
        logging.warning("[WRAPPER] validate_prompt not found, prompt patch skipped")
        return

    if getattr(original_validate_prompt, "_comfyui_prompt_path_patch", False):
        return

    def _wrapped_validate_prompt(prompt, *args, **kwargs):
        normalized_prompt = _normalize_prompt_payload_paths(prompt)
        return original_validate_prompt(normalized_prompt, *args, **kwargs)

    _wrapped_validate_prompt._comfyui_prompt_path_patch = True
    validation.validate_prompt = _wrapped_validate_prompt
    logging.info("[WRAPPER] Installed prompt path normalization patch")


def _preflight_custom_logic():
    """
    Esegue SOLO la tua logica custom stabile:
    - install requirements
    - env vars
    - crea cartelle model roots
    - download modelli mancanti nella root principale
    - genera extra_model_paths.yaml auto
    """
    # logging base (semplice; il logger vero lo inizializza poi ComfyUI)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _bootstrap_trace("_preflight_custom_logic: begin")

    # 1) bootstrap pip / requirements (la tua logica)
    _bootstrap_trace("_preflight_custom_logic: auto_install_requirements begin")
    auto_install_requirements()
    _bootstrap_trace("_preflight_custom_logic: auto_install_requirements completed")

    # 1b) ripulisce eventuali cache JSON corrotte di ComfyUI-Manager
    _bootstrap_trace("_preflight_custom_logic: cleanup manager cache begin")
    _cleanup_broken_manager_json_cache()
    _bootstrap_trace("_preflight_custom_logic: cleanup manager cache completed")
    _bootstrap_trace("_preflight_custom_logic: manager network mode check begin")
    _ensure_comfyui_manager_network_mode()
    _bootstrap_trace("_preflight_custom_logic: manager network mode check completed")

    # 2) env vars opzionali
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # 2a) Normalizza i workflow salvati con separatori Windows.
    base_dir = os.path.dirname(os.path.realpath(__file__))
    _bootstrap_trace("_preflight_custom_logic: workflow normalization begin")
    _normalize_flux_workflow_paths(base_dir)
    _install_prompt_path_normalization_patch()
    _install_known_model_selector_patch()
    _bootstrap_trace("_preflight_custom_logic: workflow normalization and prompt patch completed")

    # 2b) Compat per custom nodes legacy (es. fluxtrainer).
    # Modalita':
    #   COMFYUI_EAGER_TRANSFORMERS_COMPAT=1    -> forza patch
    #   COMFYUI_EAGER_TRANSFORMERS_COMPAT=0    -> disabilita patch
    #   COMFYUI_EAGER_TRANSFORMERS_COMPAT=auto -> applica patch solo se trova FluxTrainer
    # Default: "1" per evitare ImportError con nodi legacy che importano CLIPFeatureExtractor.
    compat_mode = os.environ.get("COMFYUI_EAGER_TRANSFORMERS_COMPAT", "1").strip().lower()
    if compat_mode not in {"0", "1", "auto"}:
        compat_mode = "1"

    if compat_mode == "1":
        _bootstrap_trace("_preflight_custom_logic: transformers compat forced begin")
        _ensure_transformers_clipfeatureextractor_compat()
        _bootstrap_trace("_preflight_custom_logic: transformers compat forced completed")
    elif compat_mode == "auto":
        custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
        found_fluxtrainer, detect_reason = _detect_fluxtrainer_custom_node(custom_nodes_dir)

        if found_fluxtrainer:
            logging.info(f"[WRAPPER] Enabling transformers compat (auto mode, {detect_reason})")
            _bootstrap_trace(f"_preflight_custom_logic: transformers compat auto begin ({detect_reason})")
            _ensure_transformers_clipfeatureextractor_compat()
            _bootstrap_trace("_preflight_custom_logic: transformers compat auto completed")
        else:
            logging.info(
                "[WRAPPER] Skipping transformers compat patch "
                f"(auto mode, FluxTrainer not detected: {detect_reason})"
            )
            _bootstrap_trace(f"_preflight_custom_logic: transformers compat skipped ({detect_reason})")
    else:
        logging.info("[WRAPPER] Skipping transformers compat patch (COMFYUI_EAGER_TRANSFORMERS_COMPAT=0)")
        _bootstrap_trace("_preflight_custom_logic: transformers compat disabled by env")

    # 3) prepara root modelli
    _bootstrap_trace("_preflight_custom_logic: resolve model roots begin")
    model_roots = _resolve_model_roots()
    _bootstrap_trace(f"_preflight_custom_logic: resolved model roots {model_roots}")
    for root in model_roots:
        try:
            os.makedirs(root, exist_ok=True)
            logging.info(f"[WRAPPER] Using models root: {root}")
        except Exception as e:
            logging.warning(f"[WRAPPER] Cannot create models root {root}: {e}")

    _bootstrap_trace("_preflight_custom_logic: ensure LLM subdirs begin")
    _ensure_llm_subdirs(model_roots)
    _bootstrap_trace("_preflight_custom_logic: ensure LLM subdirs completed")

    # 4) download modelli mancanti SOLO nella root principale (come fai già)
    if model_roots:
        _bootstrap_trace(f"_preflight_custom_logic: shared model bootstrap begin on {model_roots[0]}")
        ensure_shared_models_downloaded(model_roots[0])
        _bootstrap_trace("_preflight_custom_logic: shared model downloads completed")
        _bootstrap_trace("_preflight_custom_logic: llama gguf bootstrap begin")
        _ensure_llama_gguf_available(model_roots)
        _bootstrap_trace("_preflight_custom_logic: llama gguf bootstrap completed")
        _sync_llm_primary_to_secondary(model_roots)
        _bootstrap_trace("_preflight_custom_logic: first LLM sync completed")
        _ensure_florence2_layout(model_roots)
        _bootstrap_trace("_preflight_custom_logic: Florence2 layout completed")
        _ensure_da3_large_layout(model_roots)
        _bootstrap_trace("_preflight_custom_logic: DA3 layout completed")
        _sync_llm_primary_to_secondary(model_roots)
        _bootstrap_trace("_preflight_custom_logic: second LLM sync completed")
        _sync_model_alias_directories(model_roots)
        _bootstrap_trace("_preflight_custom_logic: model alias sync completed")

    # 5) genera config path nativo ComfyUI per le shared folders
    auto_cfg = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "extra_model_paths.auto.yaml"
    )
    _bootstrap_trace(f"_preflight_custom_logic: writing extra model paths config {auto_cfg}")
    _write_auto_extra_model_paths_yaml(auto_cfg, model_roots)
    _bootstrap_trace("_preflight_custom_logic: extra model paths config completed")

    # 6) passa il config auto al main.py ufficiale
    _bootstrap_trace("_preflight_custom_logic: append extra model paths arg begin")
    _append_extra_model_paths_arg(auto_cfg)
    _bootstrap_trace("_preflight_custom_logic: append extra model paths arg completed")

    # 7) stabilizza allocator CUDA nel flusso wrapper.
    _bootstrap_trace("_preflight_custom_logic: disable cuda malloc arg begin")
    _append_disable_cuda_malloc_arg()
    _bootstrap_trace("_preflight_custom_logic: disable cuda malloc arg completed")

    # 8) stampa snapshot pacchetti installati prima del launch del main.
    _bootstrap_trace("_preflight_custom_logic: installed packages snapshot begin")
    _print_installed_packages_snapshot()
    _bootstrap_trace("_preflight_custom_logic: installed packages snapshot completed")


def _print_installed_packages_snapshot():
    """
    Stampa i pacchetti installati nell'interprete corrente (tipicamente .venv)
    prima di avviare il main ufficiale.
    Disattivabile con COMFYUI_PRINT_INSTALLED_PACKAGES=0.
    """
    if os.environ.get("COMFYUI_PRINT_INSTALLED_PACKAGES", "1") != "1":
        return

    print("[BOOTSTRAP] Python executable:", sys.executable)

    try:
        transformers_version = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import transformers; print(transformers.__version__)",
            ],
            stderr=subprocess.STDOUT,
            timeout=30,
            text=True,
        ).strip()
        print(f"[BOOTSTRAP] transformers version: {transformers_version}")
    except Exception as exc:
        print(f"[BOOTSTRAP] Unable to read transformers version: {exc}")

    try:
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            stderr=subprocess.STDOUT,
            timeout=120,
            text=True,
        )
        print("[BOOTSTRAP] Installed packages (pip freeze style):")
        print(packages.rstrip())
    except Exception as exc:
        print(f"[BOOTSTRAP] Unable to list installed packages via pip: {exc}")

        # Fallback: `importlib.metadata` non richiede `pip`.
        try:
            from importlib import metadata as importlib_metadata

            items = []
            for dist in importlib_metadata.distributions():
                name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.name
                version = dist.version or ""
                if name:
                    items.append((name, version))
            items.sort(key=lambda t: (t[0] or "").lower())

            print("[BOOTSTRAP] Installed packages (importlib.metadata fallback):")
            for name, version in items:
                if version:
                    print(f"{name}=={version}")
                else:
                    print(name)
        except Exception as fallback_exc:
            print(f"[BOOTSTRAP] Unable to list installed packages via importlib.metadata: {fallback_exc}")


def _launch_official_comfyui_main():
    """
    Avvia il main.py UFFICIALE di ComfyUI.
    Questo mantiene compatibilità con prompt worker / cache / websocket / Assets.
    """
    comfy_main = Path(__file__).resolve().with_name("main.py")
    if not comfy_main.is_file():
        raise FileNotFoundError(f"main.py ufficiale non trovato: {comfy_main}")

    _bootstrap_trace(f"_launch_official_comfyui_main: runpy begin for {comfy_main}")
    logging.info(f"[WRAPPER] Launching official ComfyUI main: {comfy_main}")
    try:
        runpy.run_path(str(comfy_main), run_name="__main__")
    except RuntimeError as exc:
        message = str(exc)
        if "--cpu" not in sys.argv and _cuda_failure_requires_cpu_fallback(message):
            logging.warning(f"[WRAPPER] CUDA startup failed, retrying in CPU mode: {message}")
            _force_comfyui_cpu_mode(message)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise
    _bootstrap_trace("_launch_official_comfyui_main: runpy returned")


if __name__ == "__main__":
    # Bootstrap PRIMA degli import ComfyUI
    _bootstrap_trace("__main__: entering wrapper main")
    _preflight_custom_logic()
    _bootstrap_trace("__main__: preflight completed")
    _launch_official_comfyui_main()
