import os
import sys
import subprocess


extra_packages = [
     "requests",
    "PyYAML",  # <-- il pacchetto pip corretto per import yaml
    "tqdm",
    "comfy_aimdo",
        "comfyui-frontend-package==1.39.19"
]

SHARED_MODELS_URLS = {
    # =========================
    # CHECKPOINTS
    # =========================
    "checkpoints": [
        {"url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors", "filename": "v1-5-pruned-emaonly-fp16.safetensors"},
        {"url": "https://huggingface.co/webui/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors", "filename": "512-inpainting-ema.safetensors"},

        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "filename": "sd_xl_base_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors", "filename": "sd_xl_refiner_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors", "filename": "sd_xl_turbo_1.0_fp16.safetensors"},

        # >10GB circa (FP8 FLUX)
        # {"url": "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors", "filename": "flux1-dev-fp8.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors", "filename": "flux1-schnell-fp8.safetensors"},
    ],

    # =========================
    # DIFFUSION MODELS
    # =========================
    "diffusion_models": [
        # >10GB circa (Qwen Image fp8)
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "filename": "qwen_image_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors", "filename": "qwen_image_distill_full_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors", "filename": "qwen_image_edit_fp8_e4m3fn.safetensors"},

        # Wan 2.1
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", "filename": "wan2.1_t2v_1.3B_fp16.safetensors"},

        # >10GB circa (14B fp16)
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "filename": "wan2.1_i2v_480p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", "filename": "wan2.1_i2v_720p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors", "filename": "wan2.1_vace_14B_fp16.safetensors"},

        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors", "filename": "wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors"},

        # Hunyuan Video (molto pesanti)
        # >10GB circa
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors", "filename": "hunyuan_video_t2v_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_image_to_video_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors"},

        # FLUX full (gated/opzionali, pesanti)
        # >10GB circa
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors", "filename": "flux1-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors", "filename": "flux1-fill-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/flux1-kontext-dev.safetensors", "filename": "flux1-kontext-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/resolve/main/flux1-canny-dev.safetensors", "filename": "flux1-canny-dev.safetensors"},
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/resolve/main/flux1-depth-dev.safetensors", "filename": "flux1-depth-dev.safetensors"},
    ],

    # =========================
    # TEXT ENCODERS
    # =========================
    "text_encoders": [
        {"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "filename": "clip_l.safetensors"},
        {"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors", "filename": "t5xxl_fp8_e4m3fn_scaled.safetensors"},

        {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors"},

        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true", "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},

        {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors?download=true", "filename": "clip_l_hunyuan.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors?download=true", "filename": "llava_llama3_fp8_scaled.safetensors"},
    ],

    # =========================
    # VAE
    # =========================
    "vae": [
        {"url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "filename": "ae.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "filename": "qwen_image_vae.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true", "filename": "wan_2.1_vae.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors?download=true", "filename": "hunyuan_video_vae_bf16.safetensors"},
         {"url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors", "filename": "flux2-vae.safetensors"},
    ],




    # =========================
    # CLIP VISION
    # =========================
    "clip_vision": [
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true", "filename": "clip_vision_h.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/clip_vision/llava_llama3_vision.safetensors?download=true", "filename": "llava_llama3_vision.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors", "filename": "sigclip_vision_patch14_384.safetensors"},
    ],

    # =========================
    # LORAS
    # =========================
    "loras": [
        {"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-8steps-V1.0.safetensors"},
        {"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors"},

        {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors", "filename": "flux1-canny-dev-lora.safetensors"},
        {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/resolve/main/flux1-depth-dev-lora.safetensors", "filename": "flux1-depth-dev-lora.safetensors"},
    ],

    # =========================
    # STYLE MODELS
    # =========================
    "style_models": [
        {"url": "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors", "filename": "flux1-redux-dev.safetensors"},
    ],

    # =========================
    # CONTROLNET
    # =========================
    "controlnet": [
        {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/qwen_image_union_diffsynth_lora.safetensors", "filename": "qwen_image_union_diffsynth_lora.safetensors"},
    ],

    "clip": [],
    "embeddings": [],
    "upscale_models": [],
    "gligen": [],
    "hypernetworks": [],
    "vae_approx": [],

    "unet": [
        # >10GB circa (FLUX full)
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors", "filename": "flux1-schnell.safetensors"},
    ],
}

def auto_install_requirements():
    if __name__ != "__main__":
        return

    if os.environ.get("COMFYUI_AUTO_INSTALL_REQUIREMENTS", "1") != "1":
        return

    installed_any = False

    for pkg in extra_packages:
        print(f"[BOOTSTRAP] Installing extra package: {pkg}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check",
            pkg
        ])
        installed_any = True

    base_dir = os.path.dirname(os.path.realpath(__file__))
    req_files = []

    main_req = os.path.join(base_dir, "requirements.txt")
    if os.path.isfile(main_req):
        req_files.append(main_req)

    if os.environ.get("COMFYUI_AUTO_INSTALL_CUSTOM_NODE_REQUIREMENTS", "1") == "1":
        custom_nodes_dir = os.path.join(base_dir, "custom_nodes")
        if os.path.isdir(custom_nodes_dir):
            for name in os.listdir(custom_nodes_dir):
                req = os.path.join(custom_nodes_dir, name, "requirements.txt")
                if os.path.isfile(req):
                    req_files.append(req)

    seen = set()
    for req in req_files:
        req = os.path.abspath(req)
        if req in seen:
            continue
        seen.add(req)

        print(f"[BOOTSTRAP] Installing requirements from: {req}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--disable-pip-version-check",
            "-r", req
        ])
        installed_any = True

    # Refresh import system
    if installed_any:
        import importlib, site
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                site.addsitedir(user_site)
        except Exception:
            pass
        importlib.invalidate_caches()

        # Se ancora non vede yaml, riavvia una volta sola
        try:
            import yaml  # noqa
        except ModuleNotFoundError:
            if os.environ.get("_COMFYUI_BOOTSTRAP_REEXEC", "0") != "1":
                os.environ["_COMFYUI_BOOTSTRAP_REEXEC"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            raise
# Bootstrap PRIMA degli import ComfyUI
auto_install_requirements()


import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
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

if __name__ == "__main__":
    # NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)


def _infer_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        raise ValueError(f"Impossibile dedurre filename da URL: {url}")
    return filename

def _download_if_missing(url: str, dest_path: str, timeout: int = 120):
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
        return

    shared_root = os.path.abspath(shared_root)

    # Se la root non è scrivibile, salta TUTTI i download (ma ComfyUI potrà comunque leggere i modelli)
    if not _is_writable_directory(shared_root):
        logging.info(f"Shared root in sola lettura, download disabilitato: {shared_root}")
        return

    for folder_name, entries in SHARED_MODELS_URLS.items():
        target_dir = os.path.join(shared_root, folder_name)

        # Prova a creare/validare la cartella; se non scrivibile, skip solo quella cartella
        if not _is_writable_directory(target_dir):
            logging.info(f"Cartella modelli non scrivibile, skip download per '{folder_name}': {target_dir}")
            continue

        for url, filename in _normalize_model_entries(entries):
            dest_path = os.path.join(target_dir, filename)
            _download_if_missing(url, dest_path)


def apply_shared_model_paths():
    """
    Registra più cartelle modelli condivise e scarica automaticamente i modelli mancanti
    dalla cartella principale (prima root) usando SHARED_MODELS_URLS.
    """
    model_roots = [
        r"/vscode/workspace/models-default",   # cartella shared principale
        r"/vscode/workspace/models",     # seconda cartella modelli (solo lettura/logica di scansione)
    ]

    # Filtra eventuali valori vuoti
    model_roots = [os.path.abspath(p) for p in model_roots if p]

    if not model_roots:
        return

    # Crea le root (se vuoi che esistano). Se una non esiste, ComfyUI leggerà solo quelle presenti.
    for root in model_roots:
        os.makedirs(root, exist_ok=True)
        logging.info(f"Using models root: {root}")

    # Scarica modelli mancanti SOLO nella prima root (quella principale)
    # così non alteri la seconda cartella
    ensure_shared_models_downloaded(model_roots[0])

    model_dirs = {
        "checkpoints": "checkpoints",
        "loras": "loras",
        "vae": "vae",
        "clip": "clip",
        "diffusion_models": "diffusion_models",
        "embeddings": "embeddings",
        "controlnet": "controlnet",
        "upscale_models": "upscale_models",
        "clip_vision": "clip_vision",
        "style_models": "style_models",
        "gligen": "gligen",
        "hypernetworks": "hypernetworks",
        "vae_approx": "vae_approx",
        "unet": "unet",
        "text_encoders": "text_encoders",
    }

    # Aggiunge TUTTE le cartelle per ogni tipo modello
    for root in model_roots:
        for model_type, subdir in model_dirs.items():
            p = os.path.join(root, subdir)
            if os.path.isdir(p):
                folder_paths.add_model_folder_path(model_type, p)
                logging.info(f"Added model path [{model_type}] -> {p}")

def apply_custom_paths():
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # NUOVO: cartella modelli condivisa (+ download automatico se mancano)
    apply_shared_model_paths()

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
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


apply_custom_paths()
execute_prestartup_script()

# ===== WRAPPER STABILE PER COMFYUI (compatibile con update futuri) =====
# Sostituisce tutto il blocco "# Main code" e il vecchio if __name__ == "__main__"

from pathlib import Path
import runpy
import logging

# Mappa cartelle modelli (stessa logica dei tuoi path)
MODEL_DIRS_MAP = {
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "clip": "clip",
    "diffusion_models": "diffusion_models",
    "embeddings": "embeddings",
    "controlnet": "controlnet",
    "upscale_models": "upscale_models",
    "clip_vision": "clip_vision",
    "style_models": "style_models",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "vae_approx": "vae_approx",
    "unet": "unet",
    "text_encoders": "text_encoders",
}

MODEL_ROOTS = [
    "/vscode/workspace/models-default",  # principale (download qui)
    "/vscode/workspace/models",          # secondaria (lettura/scansione)
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

    data = {}
    for idx, root in enumerate(model_roots, start=1):
        root = os.path.abspath(root)
        entry_name = f"shared_models_{idx}"
        entry = {"base_path": root}
        entry.update(MODEL_DIRS_MAP)
        data[entry_name] = entry

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
    sys.argv.extend(["--extra-model-paths-config", config_path])
    logging.info(f"[WRAPPER] Added CLI arg --extra-model-paths-config {config_path}")


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

    # 1) bootstrap pip / requirements (la tua logica)
    auto_install_requirements()

    # 2) env vars opzionali
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # 3) prepara root modelli
    model_roots = [os.path.abspath(p) for p in MODEL_ROOTS if p]
    for root in model_roots:
        try:
            os.makedirs(root, exist_ok=True)
            logging.info(f"[WRAPPER] Using models root: {root}")
        except Exception as e:
            logging.warning(f"[WRAPPER] Cannot create models root {root}: {e}")

    # 4) download modelli mancanti SOLO nella root principale (come fai già)
    if model_roots:
        ensure_shared_models_downloaded(model_roots[0])

    # 5) genera config path nativo ComfyUI per le shared folders
    auto_cfg = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "extra_model_paths.auto.yaml"
    )
    _write_auto_extra_model_paths_yaml(auto_cfg, model_roots)

    # 6) passa il config auto al main.py ufficiale
    _append_extra_model_paths_arg(auto_cfg)


def _launch_official_comfyui_main():
    """
    Avvia il main.py UFFICIALE di ComfyUI.
    Questo mantiene compatibilità con prompt worker / cache / websocket / Assets.
    """
    comfy_main = Path(__file__).resolve().with_name("main.py")
    if not comfy_main.is_file():
        raise FileNotFoundError(f"main.py ufficiale non trovato: {comfy_main}")

    logging.info(f"[WRAPPER] Launching official ComfyUI main: {comfy_main}")
    runpy.run_path(str(comfy_main), run_name="__main__")


if __name__ == "__main__":
    _preflight_custom_logic()
    _launch_official_comfyui_main()