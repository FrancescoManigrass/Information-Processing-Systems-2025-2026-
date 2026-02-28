import os
import sys
import subprocess


extra_packages = [
     "requests",
    "PyYAML",  # <-- il pacchetto pip corretto per import yaml
    "tqdm",
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


# Main code
import asyncio
import shutil
import threading
import gc


if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

if __name__ == "__main__":
    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ','.join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
        os.environ['HIP_VISIBLE_DEVICES'] = str(devices)

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

if 'torch' in sys.modules:
    logging.warning("WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.")

import comfy.utils

import execution
import server
from protocol import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0


def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    import inspect

    kwargs = {"cache_type": cache_type}
    sig = inspect.signature(execution.PromptExecutor)

    # Compat vecchie versioni
    if "cache_size" in sig.parameters:
        kwargs["cache_size"] = args.cache_lru

    # Compat nuove versioni (0.15.x+)
    if "cache_args" in sig.parameters:
        # minimo indispensabile per evitare il crash su self.cache_args["ram"]
        kwargs["cache_args"] = {"ram": 0}

    e = execution.PromptExecutor(server_instance, **kwargs)


    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time

            # Log Time in a more readable way after 10 minutes
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}")
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            # Only send old method if client doesn't support preview metadata
            if not feature_flags.supports_feature(
                server_instance.sockets_metadata,
                server_instance.client_id,
                "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_database():
    try:
        from app.database.db import init_db, dependencies_available
        if dependencies_available():
            init_db()
    except Exception as e:
        logging.error(
            f"Failed to initialize database. Please ensure you have installed the latest requirements. "
            f"If the error persists, please report this as in future the database will be required: {e}"
        )


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
    prompt_server = server.PromptServer(asyncio_loop)

    hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    ))
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    setup_database()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(prompt_server.prompt_queue, prompt_server,)).start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")

    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()