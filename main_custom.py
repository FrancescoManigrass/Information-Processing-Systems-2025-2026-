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
import logging
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

# Disabilita mmap per comfy_aimdo su filesystem remoti/network che non supportano mmap().
# Imposta a "0" per riabilitare se il filesystem locale lo supporta.
os.environ.setdefault("COMFY_AIMDO_DISABLE_MMAP", "1")


def _try_prepare_writable_dir(path: str):
    if not path:
        return None

    try:
        normalized = os.path.abspath(path)
        os.makedirs(normalized, exist_ok=True)
        test_path = os.path.join(normalized, ".comfyui_cache_write_test.tmp")
        with open(test_path, "wb") as handle:
            handle.write(b"ok")
        os.remove(test_path)
        return normalized
    except Exception:
        return None


def _configure_early_disk_caches():
    """
    HuggingFace/Transformers leggono le variabili cache al primo import.
    In CrownLabs la home /vscode può avere una quota piccola: se la cache resta lì,
    FluxTrainer può fallire anche scrivendo solo i metadata dei tokenizer.
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    env_cache_root = os.environ.get("COMFYUI_CACHE_ROOT", "").strip()
    env_model_root = os.environ.get("COMFYUI_MODELS_DEFAULT_ROOT", "").strip()

    candidates = [
        env_cache_root,
        os.path.join(env_model_root, ".cache") if env_model_root else "",
        "/mnt/default-models/.cache",
        "/mnt/shared/default-models/.cache",
        "/vscode/workspace/.cache",
        os.path.join(base_dir, ".cache"),
    ]

    cache_root = None
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        cache_root = _try_prepare_writable_dir(normalized)
        if cache_root:
            break

    if not cache_root:
        return

    hf_home = os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    hf_hub_cache = os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_hub_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(cache_root, "pip"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)

    for env_name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        try:
            os.makedirs(os.environ[env_name], exist_ok=True)
        except Exception:
            pass


_configure_early_disk_caches()


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

HUGGINGFACE_RUNTIME_FORCE_PACKAGES = [
    f"transformers=={TRANSFORMERS_TARGET_VERSION}",
    f"huggingface-hub=={HUGGINGFACE_HUB_TARGET_VERSION}",
]

FLUXTRAINER_FORCE_PACKAGES = [
    f"accelerate=={ACCELERATE_TARGET_VERSION}",
    *HUGGINGFACE_RUNTIME_FORCE_PACKAGES,
    f"diffusers[torch]=={DIFFUSERS_TARGET_VERSION}",
    f"safetensors=={SAFETENSORS_TARGET_VERSION}",
    "came-pytorch==0.1.3",
    "sentencepiece>=0.2.0",
    "protobuf>=3.20.0",
]


extra_packages = [
    "requests",
    "PyYAML",
    "tqdm",
    "comfy_aimdo",
    "comfy-env",        # richiesto da custom nodes V3 (es. ComfyUI-DepthAnythingV3)
    "comfy-3d-viewers",
    #"comfy-dynamic-widgets",
    "came-pytorch",
    "numba",
    "diffusers>=0.25.0",
    f"transformers=={TRANSFORMERS_TARGET_VERSION}",
    f"huggingface-hub=={HUGGINGFACE_HUB_TARGET_VERSION}",
    f"accelerate>={ACCELERATE_TARGET_VERSION}",
    "scikit-image",     # richiesto da ComfyUI_Swwan (layerstyle_utils)
    "imagesize",        # richiesto da comfyui-fluxtrainer (train_util)
    "protobuf>=3.20.0", # richiesto da transformers T5TokenizerFast (tokenizer .model files)
    "voluptuous",
    "matplotlib"
]


_AUTO_REQUIREMENTS_ALREADY_RAN = False


def _bootstrap_trace(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BOOTSTRAP][TRACE {timestamp}] {message}", flush=True)


# ── Bootstrap probe cache ──────────────────────────────────────────────────
# Evita subprocess ripetuti (torch probe, cv2 probe) ad ogni avvio.
# TTL configurabile via COMFYUI_BOOTSTRAP_CACHE_TTL (default 4h).
_BOOTSTRAP_CACHE_TTL = int(os.environ.get("COMFYUI_BOOTSTRAP_CACHE_TTL", "14400"))

def _bootstrap_cache_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), ".bootstrap_cache.json")

def _load_bootstrap_cache():
    try:
        import json as _j
        p = _bootstrap_cache_path()
        if not os.path.isfile(p):
            return {}
        with open(p, "r", encoding="utf-8") as _f:
            d = _j.load(_f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def _save_bootstrap_cache(updates: dict):
    try:
        import json as _j
        p = _bootstrap_cache_path()
        d = _load_bootstrap_cache()
        d.update(updates)
        with open(p, "w", encoding="utf-8") as _f:
            _j.dump(d, _f, separators=(",", ":"))
    except Exception:
        pass

def _bootstrap_probe_cached(key: str) -> bool:
    d = _load_bootstrap_cache()
    return bool(d.get(key)) and (time.time() - d.get(key + "_ts", 0) < _BOOTSTRAP_CACHE_TTL)

def _bootstrap_probe_set(key: str):
    _save_bootstrap_cache({key: True, key + "_ts": time.time()})

def _bootstrap_probe_invalidate():
    """Da chiamare dopo qualsiasi install per forzare ri-verifica al prossimo avvio."""
    _save_bootstrap_cache({
        "cv2_ok": False, "cv2_ok_ts": 0,
        f"torch_cuda_ok_{PYTORCH_TARGET_VERSION}": False,
        f"cuda_probe_ok_{PYTORCH_TARGET_VERSION}": False,
    })


def _patch_xflux_vram_management():
    """
    x-flux-comfyui/nodes.py chiama `inmodel.diffusion_model.to(device)` che sposta
    l'intero modello su GPU ignorando il VRAM manager di ComfyUI, causando OOM.
    Sostituisce quella riga con comfy.model_management.load_models_gpu() che e'
    VRAM-aware e gestisce l'offload automatico.
    Patch idempotente: usa regex per trovare la riga indipendentemente dall'indentazione.
    """
    if os.environ.get("COMFYUI_PATCH_XFLUX_VRAM", "1") != "1":
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    nodes_path = os.path.join(base_dir, "custom_nodes", "x-flux-comfyui", "nodes.py")
    if not os.path.isfile(nodes_path):
        return

    try:
        with open(nodes_path, "r", encoding="utf-8") as _f:
            content = _f.read()
    except Exception as _exc:
        print(f"[BOOTSTRAP] x-flux VRAM patch: could not read {nodes_path}: {_exc}", flush=True)
        return

    # Se il vecchio patch (load_models_gpu) è presente, ripristina dal backup e riapplica.
    if "_xflux_mm.load_models_gpu([inmodel])" in content:
        backup = nodes_path + ".pre_vram_patch"
        if os.path.isfile(backup):
            with open(backup, "r", encoding="utf-8") as _f:
                content = _f.read()
            print("[BOOTSTRAP] x-flux VRAM patch: old patch detected, restoring from backup to re-patch", flush=True)
        else:
            print("[BOOTSTRAP] x-flux VRAM patch: old patch detected but no backup found, skip", flush=True)
            return

    # Controllo idempotenza: se il nuovo patch è già presente, skip.
    if "_xflux_mm.unload_all_models()" in content:
        return

    # Cerca la riga con regex tollerante all'indentazione (spazi o tab).
    import re as _re
    pattern = _re.compile(
        r'^([ \t]*)inmodel\.diffusion_model\.to\(device\)[ \t]*$',
        _re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        print(
            f"[BOOTSTRAP] x-flux VRAM patch: target line not found in {nodes_path} "
            "(different version or already patched differently), skip",
            flush=True,
        )
        return

    indent = match.group(1)
    replacement = (
        f"{indent}try:\n"
        f"{indent}    import comfy.model_management as _xflux_mm\n"
        f"{indent}    _xflux_mm.unload_all_models()\n"
        f"{indent}    _xflux_mm.soft_empty_cache()\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
        f"{indent}inmodel.diffusion_model.to(device)"
    )
    patched = pattern.sub(replacement, content, count=1)

    try:
        backup = nodes_path + ".pre_vram_patch"
        if not os.path.exists(backup):
            with open(backup, "w", encoding="utf-8") as _f:
                _f.write(content)
        with open(nodes_path, "w", encoding="utf-8") as _f:
            _f.write(patched)
        print(f"[BOOTSTRAP] Patched x-flux-comfyui VRAM management: {nodes_path}", flush=True)
    except Exception as _exc:
        print(f"[BOOTSTRAP] Warning: x-flux VRAM patch write failed: {_exc}", flush=True)


def _ensure_comfy_env_stub():
    """
    Assicura compatibilita' con custom nodes V3 che importano `comfy_env`.
    Se il pacchetto PyPI `comfy-env` e' installato, rimuove il vecchio stub
    generato dal bootstrap cosi' Python puo' usare il pacchetto reale.
    Altrimenti crea/aggiorna uno stub minimo compatibile.
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    comfy_env_path = os.path.join(base_dir, "comfy_env.py")
    marker = "comfy_env bootstrap compatibility stub"

    def _loaded_local_comfy_env():
        loaded = sys.modules.get("comfy_env")
        if loaded is None:
            return None
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file and os.path.abspath(loaded_file) == os.path.abspath(comfy_env_path):
            return loaded
        return None

    def _find_external_comfy_env_spec():
        try:
            import importlib.machinery as _machinery

            excluded = {os.path.abspath(base_dir), os.path.abspath(os.getcwd())}
            for entry in sys.path:
                search_entry = entry or os.getcwd()
                try:
                    abs_entry = os.path.abspath(search_entry)
                except Exception:
                    continue
                if abs_entry in excluded:
                    continue

                spec = _machinery.PathFinder.find_spec("comfy_env", [search_entry])
                if spec is None or not getattr(spec, "origin", None):
                    continue
                if os.path.abspath(spec.origin) != os.path.abspath(comfy_env_path):
                    return spec
        except Exception:
            pass

        return None

    def _fallback_setup_env(*_args, **_kwargs):
        return None

    def _fallback_copy_files(src_dir, dst_dir, pattern="**/*"):
        from pathlib import Path as _Path
        import shutil as _shutil

        src_path = _Path(src_dir)
        dst_path = _Path(dst_dir)
        copied = []
        if not src_path.exists():
            return copied

        for src in src_path.glob(pattern):
            if not src.is_file():
                continue
            rel_path = src.relative_to(src_path)
            dst = dst_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if (
                not dst.exists()
                or src.stat().st_mtime > dst.stat().st_mtime
                or src.stat().st_size != dst.stat().st_size
            ):
                _shutil.copy2(src, dst)
            copied.append(dst)

        return copied

    def _fallback_install(*_args, **_kwargs):
        return None

    def _fallback_wrap_isolated_nodes(node_class_mappings, *_args, **_kwargs):
        return node_class_mappings

    def _fallback_register_nodes(module_or_globals=None, globals_dict=None):
        import sys as _sys

        target = {}
        if globals_dict is not None and isinstance(globals_dict, dict):
            target = globals_dict
        elif isinstance(module_or_globals, dict):
            target = module_or_globals
        elif isinstance(module_or_globals, str):
            mod = _sys.modules.get(module_or_globals)
            if mod is not None:
                target = vars(mod)
        return target.get("NODE_CLASS_MAPPINGS", {}), target.get("NODE_DISPLAY_NAME_MAPPINGS", {})

    def _patch_loaded_stub_module():
        loaded = _loaded_local_comfy_env()
        if loaded is None:
            return
        fallbacks = {
            "setup_env": _fallback_setup_env,
            "copy_files": _fallback_copy_files,
            "install": _fallback_install,
            "wrap_isolated_nodes": _fallback_wrap_isolated_nodes,
            "register_nodes": _fallback_register_nodes,
        }
        for name, func in fallbacks.items():
            if not hasattr(loaded, name):
                setattr(loaded, name, func)

    external_spec = _find_external_comfy_env_spec()
    if external_spec is not None:
        if os.path.isfile(comfy_env_path):
            try:
                with open(comfy_env_path, "r", encoding="utf-8") as _f:
                    existing_content = _f.read()
                if marker in existing_content or "generato automaticamente dal bootstrap" in existing_content:
                    backup_path = f"{comfy_env_path}.bootstrap_stub"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.replace(comfy_env_path, backup_path)
                    print(f"[BOOTSTRAP] Disabled local comfy_env stub: {backup_path}", flush=True)
            except Exception as _exc:
                print(f"[BOOTSTRAP] Warning: could not disable local comfy_env stub: {_exc}", flush=True)

        if _loaded_local_comfy_env() is not None:
            sys.modules.pop("comfy_env", None)
        try:
            import importlib as _importlib
            _importlib.invalidate_caches()
        except Exception:
            pass
        return

    stub_content = '''\
"""comfy_env bootstrap compatibility stub."""


def setup_env(*_args, **_kwargs):
    return None


def copy_files(src_dir, dst_dir, pattern="**/*"):
    from pathlib import Path as _Path
    import shutil as _shutil

    src_path = _Path(src_dir)
    dst_path = _Path(dst_dir)
    copied = []
    if not src_path.exists():
        return copied

    for src in src_path.glob(pattern):
        if not src.is_file():
            continue
        rel_path = src.relative_to(src_path)
        dst = dst_path / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (
            not dst.exists()
            or src.stat().st_mtime > dst.stat().st_mtime
            or src.stat().st_size != dst.stat().st_size
        ):
            _shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def install(*_args, **_kwargs):
    return None


def wrap_isolated_nodes(node_class_mappings, *_args, **_kwargs):
    return node_class_mappings


def register_nodes(module_or_globals=None, globals_dict=None):
    import sys as _sys

    target = {}
    if globals_dict is not None and isinstance(globals_dict, dict):
        target = globals_dict
    elif isinstance(module_or_globals, dict):
        target = module_or_globals
    elif isinstance(module_or_globals, str):
        mod = _sys.modules.get(module_or_globals)
        if mod is not None:
            target = vars(mod)
    return target.get("NODE_CLASS_MAPPINGS", {}), target.get("NODE_DISPLAY_NAME_MAPPINGS", {})
'''
    try:
        if os.path.isfile(comfy_env_path):
            with open(comfy_env_path, "r", encoding="utf-8") as _f:
                existing_content = _f.read()
            required = ("def setup_env", "def copy_files", "def register_nodes")
            if all(token in existing_content for token in required):
                _patch_loaded_stub_module()
                return
            if marker not in existing_content and "generato automaticamente dal bootstrap" not in existing_content:
                print(f"[BOOTSTRAP] Warning: existing comfy_env.py is not a bootstrap stub: {comfy_env_path}", flush=True)
                _patch_loaded_stub_module()
                return

        with open(comfy_env_path, "w", encoding="utf-8") as _f:
            _f.write(stub_content)
        _patch_loaded_stub_module()
        print(f"[BOOTSTRAP] Updated comfy_env compatibility stub: {comfy_env_path}", flush=True)
    except Exception as _exc:
        print(f"[BOOTSTRAP] Warning: could not create/update comfy_env stub: {_exc}", flush=True)

SHARED_MODELS_URLS = {
    # =========================
    # CHECKPOINTS
    # =========================
    "checkpoints": [
       # {"url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors", "filename": "v1-5-pruned-emaonly-fp16.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly.safetensors", "filename": "v1-5-pruned-emaonly.safetensors"},
        {"url": "https://huggingface.co/webui/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors", "filename": "512-inpainting-ema.safetensors"},

        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors", "filename": "sd_xl_base_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors", "filename": "sd_xl_refiner_1.0.safetensors"},
        {"url": "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors", "filename": "sd_xl_turbo_1.0_fp16.safetensors"},
        {"url": "https://huggingface.co/voxiliummusic/cyberrealistic_V4.0/resolve/main/cyberrealistic_v40.safetensors", "filename": "cyberrealistic_v40.safetensors"},
        {"url": "https://huggingface.co/Hishambarakat/checkpoint/resolve/fa9be0812fb75f2646096c2833d1236c80751d34/flux/flux1-dev-fp8.safetensors", "filename": "flux1-dev-fp8.safetensors"},

   ],

    # =========================
    # DIFFUSION MODELS
    # =========================
    "diffusion_models": [
        # FLUX Trainer (set richiesto)
        {"url": "https://huggingface.co/bstungnguyen/Flux/resolve/main/flux1-dev.safetensors", "filename": "flux1-dev.safetensors"},
        {"url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors", "filename": "flux1-dev-fp8.safetensors"},
        {"url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors", "filename": "flux1-schnell.safetensors"},





        # >10GB circa (Qwen Image fp8)
        {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", "filename": "qwen_image_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors", "filename": "qwen_image_distill_full_fp8_e4m3fn.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors", "filename": "qwen_image_edit_fp8_e4m3fn.safetensors"},

        # Wan 2.1
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors", "filename": "wan2.1_t2v_1.3B_fp16.safetensors"},

        # >10GB circa (14B fp16)
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors", "filename": "wan2.1_i2v_480p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors", "filename": "wan2.1_i2v_720p_14B_fp16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors", "filename": "wan2.1_vace_14B_fp16.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors", "filename": "wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors"},

        # Wan 2.2 Image-to-Video (14B fp8_scaled, ~10GB ciascuno)
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "filename": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},

        # Hunyuan Video (molto pesanti)
        # >10GB circa
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors", "filename": "hunyuan_video_t2v_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_image_to_video_720p_bf16.safetensors"},
        # {"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors", "filename": "hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors"},

        # FLUX full (gated/opzionali, pesanti)
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

        {"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},

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
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "filename": "wan_2.1_vae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "filename": "qwen_image_vae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors?download=true", "filename": "hunyuan_video_vae_bf16.safetensors"},
         #{"url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors", "filename": "flux2-vae.safetensors"},
    ],




    # =========================
    # CLIP VISION
    # =========================
    "LLM": [
        # Florence2 auto-download disabilitato: lasciare i file LLM gestiti manualmente.
        # {"url": "https://huggingface.co/microsoft/Florence-2-large/resolve/main/model.safetensors", "filename": "florence-2-large-model.safetensors"},
        # {"url": "https://huggingface.co/microsoft/Florence-2-large/resolve/main/pytorch_model.bin", "filename": "florence-2-large-pytorch_model.bin"},
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true", "filename": "clip_vision_h.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/clip_vision/llava_llama3_vision.safetensors?download=true", "filename": "llava_llama3_vision.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors", "filename": "sigclip_vision_patch14_384.safetensors"},
    ],

    # =========================
    # LORAS
    # =========================
    "loras": [
        *(
            [{"url": os.environ["COMFYUI_YOUR_SD15_LORA_URL"], "filename": "your_sd15_lora.safetensors"}]
            if os.environ.get("COMFYUI_YOUR_SD15_LORA_URL")
            else []
        ),
        #{"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-8steps-V1.0.safetensors"},
        #{"url": "https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors", "filename": "Qwen-Image-Lightning-4steps-V1.0.safetensors"},

        #{"url": "https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors", "filename": "flux1-canny-dev-lora.safetensors"},
        #{"url": "https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/resolve/main/flux1-depth-dev-lora.safetensors", "filename": "flux1-depth-dev-lora.safetensors"},
        # Wan 2.2 Image-to-Video 4-step LoRA (Lightx2v)
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "filename": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"},
        {"url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "filename": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"},
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
        {"url": "https://huggingface.co/XLabs-AI/flux-controlnet-canny-v3/resolve/main/flux-canny-controlnet-v3.safetensors", "filename": "flux-canny-controlnet-v3.safetensors"},
        {"url": "https://huggingface.co/XLabs-AI/flux-controlnet-hed-v3/resolve/main/flux-hed-controlnet-v3.safetensors", "filename": "flux-hed-controlnet-v3.safetensors"},
        {"url": "https://huggingface.co/XLabs-AI/flux-controlnet-seg/resolve/main/flux-seg-controlnet.safetensors", "filename": "flux-seg-controlnet.safetensors"},
        {"url": "https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/99c135d84d5aa22fc202ebfb0fba83091b08c224/flux.1-dev-controlnet-lineart-14000.safetensors", "filename": "flux-lineart-controlnet.safetensors"},
        {"url": "https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/v11/control_v11f1p_sd15_depth.safetensors", "filename": "control_v11f1p_sd15_depth.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/qwen_image_union_diffsynth_lora.safetensors", "filename": "qwen_image_union_diffsynth_lora.safetensors"},
    ],

        "clip": [ 
            {"url": "https://huggingface.co/Madespace/clip/resolve/main/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors", "filename": "t5/google_t5-v1_1-xxl_encoderonly-fp8_e4m3fn.safetensors"},
       ],
    "embeddings": [],
    "upscale_models": [],
    "gligen": [],
    "hypernetworks": [],
    "vae_approx": [],

    "unet": [
        # >10GB circa (FLUX full)
        # {"url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors", "filename": "flux1-schnell.safetensors"},
    ],

    # =========================
    # DIFFUSERS REPOS
    # =========================
    "diffusers": [
        {"url": "https://huggingface.co/manycore-research/FLUX.1-Layout-ControlNet/resolve/main/config.json", "filename": "manycore-research/FLUX.1-Layout-ControlNet/config.json"},
        {"url": "https://huggingface.co/manycore-research/FLUX.1-Layout-ControlNet/resolve/main/diffusion_pytorch_model.safetensors", "filename": "manycore-research/FLUX.1-Layout-ControlNet/diffusion_pytorch_model.safetensors"},
    ],

    # =========================
    # GGUF MODELS (llama.cpp quantized)
    # =========================
    "gguf": [
        # Gemma auto-download disabilitato: lasciare i GGUF gestiti manualmente.
        # Gemma 4 31B Q5_K_L (~24GB) — bartowski quant
        # {"url": "https://huggingface.co/bartowski/google_gemma-4-31B-it-GGUF/resolve/main/google_gemma-4-31B-it-Q5_K_L.gguf",
        #  "filename": "google_gemma-4-31B-it-Q5_K_L.gguf"},
    ],
}


MODEL_FILENAME_ALIASES = {
    # Hugging Face stores this as flux-dev-controlnet-union-pro.safetensors,
    # while some workflows use the shorter legacy/widget name.
    "flux-Union-controlnet.safetensors": (
        "flux-dev-controlnet-union-pro.safetensors",
        "flux-union-controlnet.safetensors",
    ),
}


def _equivalent_model_filenames(filename: str):
    names = [filename]
    for canonical, aliases in MODEL_FILENAME_ALIASES.items():
        group = [canonical, *aliases]
        if filename in group:
            names = group
            break

    result = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


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


def _get_huggingface_runtime_import_error():
    # NOTE: 'tqdm_class' fu rimosso da hf_hub_download in huggingface_hub >= 0.22.
    # Non si controlla più la sua presenza: la compat patch (sotto) gestisce runtime
    # i chiamanti che lo passano ancora. Il probe verifica solo che gli import funzionino.
    script = (
        "import transformers\n"
        "import huggingface_hub\n"
        "from huggingface_hub import hf_hub_download\n"
        "print(getattr(transformers, '__version__', 'unknown'), "
        "getattr(huggingface_hub, '__version__', 'unknown'))\n"
    )

    try:
        probe = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
        )
    except Exception as exc:
        return str(exc)

    if probe.returncode == 0:
        return ""

    return (probe.stdout or "").strip() or f"import probe failed with exit code {probe.returncode}"


def _enforce_huggingface_runtime_stack():
    pending_packages = _get_pending_requirements(HUGGINGFACE_RUNTIME_FORCE_PACKAGES)
    force_reinstall = False
    if not pending_packages:
        import_error = _get_huggingface_runtime_import_error()
        if not import_error:
            print("[BOOTSTRAP] Hugging Face runtime packages already aligned, skip")
            return False

        print(f"[BOOTSTRAP] Hugging Face runtime import check failed, reinstalling pinned stack: {import_error}")
        pending_packages = list(HUGGINGFACE_RUNTIME_FORCE_PACKAGES)
        force_reinstall = True

    print(
        "[BOOTSTRAP] Enforcing Hugging Face runtime package versions: "
        + ", ".join(HUGGINGFACE_RUNTIME_FORCE_PACKAGES)
    )
    install_args = [
        "--disable-pip-version-check",
        "--upgrade",
    ]
    if force_reinstall:
        install_args.append("--force-reinstall")

    subprocess.check_call(_get_bootstrap_install_cmd(
        *install_args,
        *pending_packages,
    ))
    return True


def _patch_huggingface_hub_download_tqdm_class_compat():
    """
    Custom nodes (es. ComfyUI-DepthAnythingV3) possono chiamare hf_hub_download
    passando tqdm_class=..., parametro rimosso in huggingface_hub >= 0.22.
    La patch:
      1. Installa un hook builtins.__import__ che intercetta qualsiasi
         `from huggingface_hub import hf_hub_download` e sostituisce il
         riferimento nel modulo sorgente con il wrapper prima che il caller
         lo legga. Questo è immuno ai reset del LazyModule e alle reimport.
      2. Sostituisce hf_hub_download nei moduli huggingface_hub noti usando
         __dict__ direttamente (bypassa __setattr__ del LazyModule).
      3. Fa il sweep di sys.modules per aggiornare riferimenti già bindati.
    """
    import builtins
    import importlib
    import inspect

    # ── Costruisci il wrapper ────────────────────────────────────────────────
    def _make_tqdm_class_wrapper(original):
        if getattr(original, "_comfyui_tqdm_class_compat", False):
            return original

        def _wrapped_hf_hub_download(*args, __orig=original, **kwargs):
            try:
                supports_tqdm_class = "tqdm_class" in inspect.signature(__orig).parameters
            except Exception:
                supports_tqdm_class = False
            if not supports_tqdm_class:
                kwargs.pop("tqdm_class", None)
            return __orig(*args, **kwargs)

        _wrapped_hf_hub_download.__name__ = getattr(original, "__name__", "hf_hub_download")
        _wrapped_hf_hub_download.__doc__ = getattr(original, "__doc__", None)
        _wrapped_hf_hub_download._comfyui_tqdm_class_compat = True
        return _wrapped_hf_hub_download

    # ── Recupera/crea il wrapper canonico ───────────────────────────────────
    try:
        hf_module = importlib.import_module("huggingface_hub")
    except Exception:
        return False

    canonical = getattr(hf_module, "hf_hub_download", None)
    if not callable(canonical):
        return False

    if getattr(canonical, "_comfyui_tqdm_class_compat", False):
        wrapper = canonical
    else:
        wrapper = _make_tqdm_class_wrapper(canonical)

    # ── 1) builtins.__import__ interceptor ──────────────────────────────────
    # Intercetta qualsiasi `from huggingface_hub import hf_hub_download`
    # anche se avviene DOPO questa patch (lazy custom node imports).
    # L'intercettore patcha il __dict__ del modulo sorgente prima che il
    # caller Python legga il simbolo, quindi il caller riceve sempre il wrapper.
    _orig_import = builtins.__import__
    if not getattr(_orig_import, "_comfyui_hfhub_compat", False):

        def _intercepting_import(name, glob=None, loc=None, fromlist=(), level=0):
            result = _orig_import(name, glob, loc, fromlist, level)
            if not fromlist or "hf_hub_download" not in fromlist:
                return result
            # IMPORTANT: `IMPORT_FROM` always reads from `result` (the object
            # returned by __import__), NOT from a navigated submodule.
            # We must patch result.__dict__ directly so the subsequent
            # `getattr(result, 'hf_hub_download')` in the bytecode gets our wrapper.
            try:
                fn = result.__dict__.get("hf_hub_download")
                if fn is None:
                    fn = getattr(result, "hf_hub_download", None)
                if callable(fn) and not getattr(fn, "_comfyui_tqdm_class_compat", False):
                    w = _make_tqdm_class_wrapper(fn)
                    try:
                        result.__dict__["hf_hub_download"] = w
                    except Exception:
                        pass
            except Exception:
                pass
            return result

        _intercepting_import._comfyui_hfhub_compat = True
        builtins.__import__ = _intercepting_import
        logging.info("[BOOTSTRAP] Installed hf_hub_download import interceptor (builtins.__import__)")

    # ── 2) Patcha moduli huggingface_hub noti via __dict__ ──────────────────
    # Usa __dict__ direttamente per bypassare eventuale __setattr__ del LazyModule.
    already_patched_in_known = False
    for module_name in (
        "huggingface_hub",
        "huggingface_hub.file_download",
        "huggingface_hub._snapshot_download",
    ):
        try:
            mod = sys.modules.get(module_name) or importlib.import_module(module_name)
            fn = mod.__dict__.get("hf_hub_download") or getattr(mod, "hf_hub_download", None)
            if callable(fn) and not getattr(fn, "_comfyui_tqdm_class_compat", False):
                w = _make_tqdm_class_wrapper(fn)
                try:
                    mod.__dict__["hf_hub_download"] = w
                except Exception:
                    try:
                        setattr(mod, "hf_hub_download", w)
                    except Exception:
                        pass
                already_patched_in_known = True
        except Exception:
            pass

    if already_patched_in_known:
        logging.info("[BOOTSTRAP] Applied hf_hub_download tqdm_class patch to huggingface_hub modules")

    # ── 3) Sweep sys.modules ─────────────────────────────────────────────────
    patched_sysmods = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        fn = mod.__dict__.get("hf_hub_download") if hasattr(mod, "__dict__") else None
        if fn is None:
            fn = getattr(mod, "hf_hub_download", None)
        if not callable(fn):
            continue
        if getattr(fn, "_comfyui_tqdm_class_compat", False):
            continue
        w = _make_tqdm_class_wrapper(fn)
        try:
            mod.__dict__["hf_hub_download"] = w
            patched_sysmods += 1
        except Exception:
            try:
                setattr(mod, "hf_hub_download", w)
                patched_sysmods += 1
            except Exception:
                pass

    if patched_sysmods:
        logging.info(
            f"[BOOTSTRAP] Swept sys.modules: updated hf_hub_download in {patched_sysmods} additional module(s)"
        )
    return True


def _oneformer_cuda_runtime_error_requires_cpu(message):
    if not message:
        return False

    lowered = str(message).lower()
    if "cuda" not in lowered:
        return False

    indicators = (
        "invalid argument",
        "illegal memory access",
        "device-side assert",
        "driver error",
        "cuda error",
        "cublas",
        "cusparse",
    )
    return any(indicator in lowered for indicator in indicators)


def _get_controlnet_aux_oneformer_device():
    device = os.environ.get("COMFYUI_CONTROLNET_AUX_ONEFORMER_DEVICE", "cpu").strip()
    if not device:
        return "cpu"

    normalized = device.lower()
    if normalized in {"auto", "comfy", "default"}:
        try:
            import comfy.model_management as model_management

            return model_management.get_torch_device()
        except Exception:
            return "cpu"

    return device


def _patch_controlnet_aux_oneformer_module(module):
    """
    OneFormer in comfyui_controlnet_aux can hit CUDA driver/runtime edge cases on
    some lab GPUs even when the main ComfyUI CUDA probe passes. Keep only these
    semantic preprocessors on CPU by default; the rest of ComfyUI stays on GPU.
    """
    if module is None or getattr(module, "_comfyui_oneformer_cpu_compat", False):
        return False

    common_annotator_call = getattr(module, "common_annotator_call", None)
    if not callable(common_annotator_call):
        return False

    class_specs = {
        "OneFormer_COCO_SemSegPreprocessor": "150_16_swin_l_oneformer_coco_100ep.pth",
        "OneFormer_ADE20K_SemSegPreprocessor": "250_16_swin_l_oneformer_ade20k_160k.pth",
    }

    patched_any = False
    for class_name, filename in class_specs.items():
        node_class = getattr(module, class_name, None)
        if node_class is None:
            continue

        original = getattr(node_class, "semantic_segmentate", None)
        if not callable(original) or getattr(original, "_comfyui_oneformer_cpu_compat", False):
            continue

        def _wrapped_semantic_segmentate(self, image, resolution=512, __filename=filename):
            from custom_controlnet_aux.oneformer import OneformerSegmentor

            device = _get_controlnet_aux_oneformer_device()
            model = OneformerSegmentor.from_pretrained(filename=__filename)
            try:
                model = model.to(device)
                out = common_annotator_call(model, image, resolution=resolution)
            except RuntimeError as exc:
                if str(device).lower() != "cpu" and _oneformer_cuda_runtime_error_requires_cpu(str(exc)):
                    logging.warning(
                        "[WRAPPER] OneFormer CUDA execution failed on %s, retrying on CPU: %s",
                        device,
                        exc,
                    )
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    model = model.to("cpu")
                    out = common_annotator_call(model, image, resolution=resolution)
                else:
                    raise
            finally:
                try:
                    del model
                except Exception:
                    pass
            return (out,)

        _wrapped_semantic_segmentate._comfyui_oneformer_cpu_compat = True
        node_class.semantic_segmentate = _wrapped_semantic_segmentate
        patched_any = True

    if patched_any:
        module._comfyui_oneformer_cpu_compat = True
        logging.info(
            "[WRAPPER] Patched comfyui_controlnet_aux OneFormer preprocessors "
            "to use %s by default",
            os.environ.get("COMFYUI_CONTROLNET_AUX_ONEFORMER_DEVICE", "cpu"),
        )
    return patched_any


def _patch_loaded_controlnet_aux_oneformer_modules():
    patched_any = False
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith("node_wrappers.oneformer"):
            continue
        patched_any = _patch_controlnet_aux_oneformer_module(module) or patched_any
    return patched_any


def _install_controlnet_aux_oneformer_cpu_compat():
    if os.environ.get("COMFYUI_PATCH_CONTROLNET_AUX_ONEFORMER_CPU", "1") != "1":
        return False

    try:
        import importlib
    except Exception:
        return False

    _patch_loaded_controlnet_aux_oneformer_modules()

    original_import_module = importlib.import_module
    if getattr(original_import_module, "_comfyui_oneformer_cpu_import_hook", False):
        return True

    def _wrapped_import_module(name, package=None):
        module = original_import_module(name, package)
        try:
            module_name = getattr(module, "__name__", "")
            if module_name.endswith("node_wrappers.oneformer"):
                _patch_controlnet_aux_oneformer_module(module)
        except Exception as exc:
            logging.debug("[WRAPPER] OneFormer CPU compat patch skipped: %s", exc)
        return module

    _wrapped_import_module._comfyui_oneformer_cpu_import_hook = True
    _wrapped_import_module._comfyui_original_import_module = original_import_module
    importlib.import_module = _wrapped_import_module
    return True


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


def _get_requirement_package_name(requirement):
    requirement = _normalize_requirement_entry(requirement)
    if not requirement or Requirement is None:
        return ""

    if requirement.startswith(("-", ".", "/")) or requirement.startswith(("git+", "http://", "https://")):
        return ""

    try:
        return Requirement(requirement).name.lower()
    except Exception:
        return ""


def _should_skip_bootstrap_managed_requirement(requirement):
    if os.environ.get("COMFYUI_SKIP_BOOTSTRAP_MANAGED_REQUIREMENTS", "1") != "1":
        return False
    if os.environ.get("COMFYUI_ENFORCE_PYTORCH_VERSION", "1") != "1":
        return False

    return _get_requirement_package_name(requirement) in PYTORCH_REQUIREMENT_PACKAGE_NAMES


def _write_filtered_requirements_file(requirements_path):
    skipped = []
    kept_lines = []

    with open(requirements_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            if _should_skip_bootstrap_managed_requirement(raw_line):
                skipped.append(_normalize_requirement_entry(raw_line))
                continue
            kept_lines.append(raw_line)

    if not skipped:
        return requirements_path, None, skipped

    temp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        prefix="comfyui-filtered-requirements-",
        suffix=".txt",
    )
    try:
        temp_file.writelines(kept_lines)
        return temp_file.name, temp_file.name, skipped
    finally:
        temp_file.close()


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

                if _should_skip_bootstrap_managed_requirement(normalized):
                    continue

                if normalized.startswith("-"):
                    if normalized.startswith(("-r ", "--requirement ")):
                        return True
                    continue

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

    _torch_cache_key = f"torch_cuda_ok_{PYTORCH_TARGET_VERSION}"
    if _bootstrap_probe_cached(_torch_cache_key):
        _bootstrap_trace(f"pytorch compat: cache hit for torch {PYTORCH_TARGET_VERSION}, skip subprocess probe")
        return False

    runtime_info = _inspect_torch_runtime(sys.executable)
    if runtime_info.get("cuda_current_device_ok"):
        _bootstrap_probe_set(_torch_cache_key)
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

    _cuda_cache_key = f"cuda_probe_ok_{PYTORCH_TARGET_VERSION}"
    if _bootstrap_probe_cached(_cuda_cache_key):
        _bootstrap_trace("cuda probe: cache hit, skip subprocess probe")
        return

    probe_ok, output = _run_torch_cuda_probe(sys.executable)
    if probe_ok:
        _bootstrap_probe_set(_cuda_cache_key)
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

    if _bootstrap_probe_cached("cv2_ok"):
        return True

    ok, info = _check_cv2_import_subprocess()
    if ok:
        print(f"[BOOTSTRAP] cv2 import OK: {info}")
        _bootstrap_probe_set("cv2_ok")
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
    numpy_abi_markers = ("_ARRAY_API not found", "numpy.core.multiarray failed to import")
    if any(marker in info for marker in numpy_abi_markers) and os.environ.get("COMFYUI_CV2_NUMPY1_FALLBACK", "1") == "1":
        numpy_spec = os.environ.get("COMFYUI_NUMPY_COMPAT_SPEC", "numpy<2")
        print(f"[BOOTSTRAP] Detected NumPy/OpenCV ABI mismatch, installing: {numpy_spec}")
        try:
            subprocess.check_call(
                _get_bootstrap_install_cmd(
                    "--disable-pip-version-check",
                    "--force-reinstall",
                    "--no-cache-dir",
                    numpy_spec,
                )
            )
        except Exception as exc:
            print(f"[BOOTSTRAP] NumPy compat install failed: {exc}")

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
        pip_cmd = _get_bootstrap_pip_cmd(python_executable)
        uv_target = _get_uv_user_site_target(pip_cmd, python_executable)
        if uv_target:
            return pip_cmd + ["install", "--target", uv_target, *effective_args]
        # Fallback --user: quando il sistema non e' scrivibile e non siamo in un venv.
        # Copre sia pip puro che uv quando _get_uv_user_site_target non ha applicato --target.
        if (
            "--user" not in effective_args
            and not os.environ.get("VIRTUAL_ENV")
            and getattr(sys, "base_prefix", sys.prefix) == sys.prefix
            and not (hasattr(os, "geteuid") and os.geteuid() == 0)
            and python_executable == os.path.realpath(sys.executable)
            and not _system_site_packages_writable()
        ):
            return pip_cmd + ["install", "--user", *effective_args]
        return pip_cmd + ["install", *effective_args]
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


def _is_uv_pip_cmd(command):
    command = [os.path.basename(str(part)) for part in command]
    if len(command) >= 2 and command[-2:] == ["uv", "pip"]:
        return True
    if len(command) >= 4 and command[-4:] == ["python", "-m", "uv", "pip"]:
        return True
    return len(command) >= 3 and command[-3:] == ["-m", "uv", "pip"]


def _system_site_packages_writable():
    """
    Prova a creare un file temporaneo nelle cartelle site-packages di sistema.
    Ritorna True se almeno una e' scrivibile dall'utente corrente.
    """
    try:
        import site as _site
        candidates = list(_site.getsitepackages()) if hasattr(_site, "getsitepackages") else []
    except Exception:
        return True  # Fallback conservativo: assume scrivibile

    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        test_path = os.path.join(path, ".comfyui_perm_test.tmp")
        try:
            with open(test_path, "wb") as _f:
                _f.write(b"x")
            os.remove(test_path)
            return True
        except Exception:
            continue
    return False


def _get_uv_user_site_target(pip_cmd, python_executable):
    """
    uv non fa fallback automatico allo user-site quando il sistema non e' scrivibile.
    In quel caso installiamo direttamente nello user-site importato da Python
    (es. /vscode/.local/lib/python3.10/site-packages).
    Usa un check reale di scrivibilita' invece del fragile prefisso /usr.
    """
    if os.environ.get("COMFYUI_UV_TARGET_USER_SITE", "1") != "1":
        return None
    if not _is_uv_pip_cmd(pip_cmd):
        return None
    if os.environ.get("VIRTUAL_ENV") or getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return None
    if os.path.realpath(python_executable) != os.path.realpath(sys.executable):
        return None
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return None
    # Verifica effettiva: se il sistema e' scrivibile non serve redirectare.
    if _system_site_packages_writable():
        return None

    try:
        import site

        user_site = site.getusersitepackages()
        if not user_site:
            return None
        os.makedirs(user_site, exist_ok=True)
        return user_site
    except Exception as exc:
        print(f"[BOOTSTRAP] Warning: unable to resolve Python user site for uv install: {exc}")
        return None


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

        install_req = req
        temp_req = None
        skipped_bootstrap_managed = []
        try:
            install_req, temp_req, skipped_bootstrap_managed = _write_filtered_requirements_file(req)
        except Exception as exc:
            print(f"[BOOTSTRAP] WARNING: Unable to filter bootstrap-managed requirements from {req}: {exc}")

        if skipped_bootstrap_managed:
            print(
                "[BOOTSTRAP] Skipping bootstrap-managed requirement(s) from "
                f"{req}: {', '.join(skipped_bootstrap_managed)}"
            )

        print(f"[BOOTSTRAP] Installing requirements from: {req}")
        try:
            subprocess.check_call(_get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "--timeout", "600",   # 10 min per chunk (pacchetti CUDA > 200MB)
                "--retries", "5",
                "-r",
                install_req,
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
        finally:
            if temp_req:
                try:
                    os.remove(temp_req)
                except Exception:
                    pass

    # IMPORTANT: riallinea FluxTrainer alla fine, dopo TUTTI gli altri requirements,
    # così eventuali installazioni precedenti non lasciano l'ambiente in stato incoerente.
    if os.environ.get("COMFYUI_FORCE_TRANSFORMERS_FLUXTRAINER_COMPAT", "1") == "1":
        if _install_fluxtrainer_runtime_stack(custom_nodes_dir):
            installed_any = True
        _bootstrap_trace("auto_install_requirements: FluxTrainer final reconciliation completed")

    if os.environ.get("COMFYUI_ENFORCE_HUGGINGFACE_RUNTIME", "1") == "1":
        if _enforce_huggingface_runtime_stack():
            installed_any = True
        _bootstrap_trace("auto_install_requirements: Hugging Face runtime enforcement completed")

    _patch_huggingface_hub_download_tqdm_class_compat()
    _bootstrap_trace("auto_install_requirements: Hugging Face runtime compat patch completed")

    if _should_force_headless_opencv():
        if _ensure_headless_opencv():
            installed_any = True
        _bootstrap_trace("auto_install_requirements: post-requirements OpenCV normalization completed")

    # Pin safetensors a una versione stabile dopo tutti i requirements dei custom nodes.
    # x-flux-comfyui e altri possono installare pre-release (es. 0.8.0rc0) che causano crash.
    safetensors_pin = f"safetensors=={SAFETENSORS_TARGET_VERSION}"
    pending_safetensors = _get_pending_requirements([safetensors_pin])
    if pending_safetensors:
        print(f"[BOOTSTRAP] Pinning safetensors to stable version: {safetensors_pin}")
        try:
            subprocess.check_call(_get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "--force-reinstall",
                safetensors_pin,
            ))
            installed_any = True
        except Exception as exc:
            print(f"[BOOTSTRAP] Warning: failed pinning safetensors: {exc}")
    _bootstrap_trace("auto_install_requirements: safetensors stable pin completed")

    # Protegge l'avvio da cv2 rotto (mismatch NumPy/OpenCV).
    # Riporta PyTorch ai pin del requirements dopo i custom nodes, evitando
    # rimbalzi tra stack CUDA diversi a ogni avvio.
    if os.environ.get("COMFYUI_ENFORCE_PYTORCH_VERSION", "1") == "1":
        _pytorch_index_url = _get_compatible_pytorch_index_url()
        if _pytorch_index_url:
            _pytorch_stack = [
                f"torch=={PYTORCH_TARGET_VERSION}",
                f"torchvision=={TORCHVISION_TARGET_VERSION}",
                f"torchaudio=={TORCHAUDIO_TARGET_VERSION}",
            ]
            _pending_pytorch = _get_pending_requirements(_pytorch_stack)
            if _pending_pytorch:
                print(
                    f"[BOOTSTRAP] Enforcing GPU-compatible PyTorch after requirements: "
                    f"{', '.join(_pytorch_stack)} from {_pytorch_index_url}"
                )
                try:
                    subprocess.check_call(_get_bootstrap_install_cmd(
                        "--disable-pip-version-check",
                        "--force-reinstall",
                        "--index-url", _pytorch_index_url,
                        *_pytorch_stack,
                    ))
                    installed_any = True
                except Exception as _exc:
                    print(f"[BOOTSTRAP] Warning: PyTorch enforcement failed: {_exc}")
            else:
                print(f"[BOOTSTRAP] GPU-compatible PyTorch already at target version, skip")
        _bootstrap_trace("auto_install_requirements: pytorch version enforcement completed")

    _bootstrap_trace("auto_install_requirements: checking cv2 importability")
    _ensure_cv2_importable_or_fallback()
    _bootstrap_trace("auto_install_requirements: cv2 importability check completed")

    if installed_any:
        _bootstrap_probe_invalidate()
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


class _TeeBinaryBuffer:
    """Buffer binario che duplica le scritture sia sull'originale che sul file di log."""
    def __init__(self, original_buffer, log_file):
        self._original = original_buffer
        self._log_file = log_file

    def write(self, data):
        try:
            self._original.write(data)
        except Exception:
            pass
        try:
            text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data
            self._log_file.write(text)
            self._log_file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


class _Tee:
    """Duplica stdout/stderr sia sul terminale che su file di log."""
    def __init__(self, original_stream, log_file):
        self._original = original_stream
        self._log_file = log_file

    def write(self, data):
        try:
            self._original.write(data)
            self._original.flush()
        except Exception:
            pass
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._log_file.flush()
        except Exception:
            pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    @property
    def buffer(self):
        orig_buffer = getattr(self._original, "buffer", None)
        if orig_buffer is not None:
            return _TeeBinaryBuffer(orig_buffer, self._log_file)
        return self._original

    def __getattr__(self, name):
        return getattr(self._original, name)


def _setup_crash_logging():
    import faulthandler

    base_dir = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(base_dir, "comfyui_crash.log")

    try:
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        header = (
            f"\n{'='*60}\n"
            f"[CRASH LOG] Session: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[CRASH LOG] Python: {sys.version}\n"
            f"[CRASH LOG] Args: {' '.join(sys.argv)}\n"
            f"{'='*60}\n"
        )
        log_file.write(header)
        log_file.flush()

        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)

        faulthandler.enable(file=log_file, all_threads=True)

    except Exception as exc:
        print(f"[BOOTSTRAP] Warning: crash logging setup failed: {exc}", flush=True)
        return

    print(f"[CRASH LOG] Logging to: {log_path}", flush=True)

    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            meminfo = f.read()
        mem_total = re.search(r"MemTotal:\s+(\d+)", meminfo)
        mem_avail = re.search(r"MemAvailable:\s+(\d+)", meminfo)
        if mem_total and mem_avail:
            total_gb = int(mem_total.group(1)) / 1024 / 1024
            avail_gb = int(mem_avail.group(1)) / 1024 / 1024
            print(f"[CRASH LOG] RAM: {avail_gb:.1f}GB available / {total_gb:.1f}GB total", flush=True)
    except Exception:
        pass

    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        for line in gpu_info.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                print(f"[CRASH LOG] GPU: {parts[0]} | VRAM {parts[2]}/{parts[1]} MB free | {parts[3]}°C", flush=True)
    except Exception:
        pass

    # Mostra i processi che usano la GPU — utile per identificare processi estranei che consumano VRAM.
    try:
        pmon = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "m"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        for pline in pmon.strip().splitlines():
            if pline.startswith("#") or not pline.strip():
                continue
            fields = pline.split()
            if len(fields) >= 4:
                pid, gpu_idx, mem_mb = fields[1], fields[0], fields[3]
                if mem_mb not in ("-", "0") and pid != "-":
                    try:
                        cmdline_path = f"/proc/{pid}/cmdline"
                        if os.path.isfile(cmdline_path):
                            with open(cmdline_path, "rb") as _pf:
                                cmd = _pf.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()[:120]
                        else:
                            cmd = "(unavailable)"
                    except Exception:
                        cmd = "(unavailable)"
                    print(f"[CRASH LOG] GPU process: PID={pid} GPU={gpu_idx} MEM={mem_mb}MiB CMD={cmd}", flush=True)
    except Exception:
        pass


# Install custom nodes PRIMA del bootstrap requirements, così i loro requirements vengono inclusi.
if __name__ == "__main__":
    _setup_crash_logging()

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
_bootstrap_trace("startup: Hugging Face runtime compat patch begin")
_patch_huggingface_hub_download_tqdm_class_compat()
_bootstrap_trace("startup: Hugging Face runtime compat patch completed")
_bootstrap_trace("startup: ControlNet Aux OneFormer CPU compat patch begin")
_install_controlnet_aux_oneformer_cpu_compat()
_bootstrap_trace("startup: ControlNet Aux OneFormer CPU compat patch completed")
_bootstrap_trace("startup: cuda probe begin")
_maybe_force_cpu_mode_from_torch_probe()
_bootstrap_trace("startup: cuda probe completed")

# Stabilizza l'allocator CUDA PRIMA di qualunque import Comfy/PyTorch.
# Evita mismatch: runtime cudaMallocAsync vs load-time native.
if "--disable-cuda-malloc" not in sys.argv and os.environ.get("COMFYUI_FORCE_CUDA_MALLOC", "0") != "1":
    sys.argv.append("--disable-cuda-malloc")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")


# Compat FluxTrainer/transformers prima di importare ComfyUI.
_bootstrap_trace("startup: early transformers compat begin")
_apply_early_transformers_fluxtrainer_compat()
_bootstrap_trace("startup: early transformers compat completed")

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
_ensure_comfy_env_stub()
try:
    import comfy_env  # noqa: F401 — side-effect import, non presente in tutte le versioni
except ImportError:
    pass
_bootstrap_trace("startup: ComfyUI runtime modules imported")

if __name__ == "__main__":
    # NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)
_bootstrap_trace("startup: logger configured")

# Patch JSON encoder per evitare che oggetti non serializzabili (es. eccezioni Python)
# nel publish_loop di server.py causino TypeError → asyncio.gather crash → CUDA abort.
import json as _json_module

class _PermissiveJSONEncoder(_json_module.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)

_json_module._default_encoder = _PermissiveJSONEncoder(
    skipkeys=False, ensure_ascii=True, check_circular=True,
    allow_nan=True, sort_keys=False, indent=None, separators=None,
    default=None,
)


def _infer_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed.path)
    if not filename:
        raise ValueError(f"Impossibile dedurre filename da URL: {url}")
    return filename

def _is_safetensors_header_valid(path: str) -> bool:
    """
    Valida che il file safetensors non sia corrotto o incompleto.
    Legge solo i primi 8 byte (header length) e verifica che il file
    contenga almeno header_length + 8 byte totali.
    """
    try:
        import json
        import struct
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False
            header_size = struct.unpack("<Q", raw)[0]
            if header_size < 2 or header_size > 100 * 1024 * 1024:
                return False
            header = f.read(header_size)
            if len(header) < header_size:
                return False
            header_data = json.loads(header.decode("utf-8"))
            if not isinstance(header_data, dict):
                return False
            f.seek(0, 2)
            return f.tell() >= 8 + header_size
    except Exception:
        return False


def _is_existing_model_file_valid(path: str) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            return False
    except Exception:
        return False

    if path.lower().endswith(".safetensors"):
        return _is_safetensors_header_valid(path)

    return True


def _is_shared_model_placeholder(filename: str) -> bool:
    normalized = (filename or "").strip().lower()
    if not normalized:
        return True
    if normalized in {".gitkeep", ".keep"}:
        return True
    if normalized.startswith("put_") and normalized.endswith("_here"):
        return True
    return False


def _download_if_missing(url: str, dest_path: str, timeout: int = 120, ignore_http_404: bool = False):
    """
    Scarica il file solo se non esiste già o se è corrotto.
    Scrive su .part e poi fa rename atomico.
    Mostra una progress bar con tqdm.
    """
    if _is_existing_model_file_valid(dest_path):
        logging.info(f"Model already present, skip download: {dest_path}")
        return

    if os.path.isfile(dest_path):
        logging.warning(f"Model file corrotto o incompleto, riscarico: {dest_path}")
        try:
            os.remove(dest_path)
        except Exception:
            pass

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
        if dest_path.lower().endswith(".safetensors") and not _is_safetensors_header_valid(dest_path):
            logging.error(f"Downloaded safetensors is invalid, removing: {dest_path}")
            try:
                os.remove(dest_path)
            except Exception:
                pass
            return
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
    test_file = os.path.join(path, ".comfyui_write_test.tmp")
    try:
        os.makedirs(path, exist_ok=True)
        with open(test_file, "wb") as f:
            f.write(b"ok")
        os.remove(test_file)
        return True
    except Exception as e:
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
        except Exception:
            pass
        logging.warning(f"Directory non scrivibile (skip download): {path} -> {e}")
        return False


def _preferred_shared_model_roots(model_roots=None):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    candidates = [
        os.environ.get("COMFYUI_SHARED_MODELS_ROOT", "").strip(),
        os.path.join(base_dir, "shared", "default-models"),
        "/vscode/workspace/shared/default-models",
        "/mnt/shared/default-models",
    ]

    for root in model_roots or []:
        normalized = os.path.abspath(root)
        if normalized.endswith(os.path.join("shared", "default-models")):
            candidates.append(root)

    roots = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append(normalized)
    return roots


def _local_models_root():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(
        os.environ.get("COMFYUI_MODELS_ROOT", "").strip()
        or os.path.join(base_dir, "models")
    )


def _path_is_in_shared_model_root(path: str, model_roots=None) -> bool:
    try:
        normalized_path = os.path.abspath(path)
    except Exception:
        return False

    for root in _preferred_shared_model_roots(model_roots):
        try:
            normalized_root = os.path.abspath(root)
            if normalized_path == normalized_root or normalized_path.startswith(normalized_root + os.sep):
                return True
        except Exception:
            continue
    return False


def _select_writable_model_root(model_roots):
    """
    Usa prima una root condivisa scrivibile. In CrownLabs /mnt/default-models
    puo' esistere ma avere quota esaurita; i download devono stare in shared.
    """
    for root in _preferred_shared_model_roots(model_roots):
        if _is_writable_directory(root):
            return os.path.abspath(root)
        _bootstrap_trace(f"_select_writable_model_root: shared root not writable {root}")

    if os.environ.get("COMFYUI_ALLOW_NON_SHARED_MODEL_DOWNLOAD", "0") != "1":
        _bootstrap_trace("_select_writable_model_root: no writable shared root found; downloads disabled")
        return None

    for root in model_roots or []:
        if _is_writable_directory(root):
            return os.path.abspath(root)
        _bootstrap_trace(f"_select_writable_model_root: not writable {root}")
    return None


def _find_existing_model_file(model_roots, folder_name: str, filename: str):
    union_names = set(_equivalent_model_filenames("flux-Union-controlnet.safetensors"))
    for root in model_roots or []:
        for candidate_filename in _equivalent_model_filenames(filename):
            candidate = os.path.join(root, folder_name, candidate_filename)
            if _is_existing_model_file_valid(candidate):
                if candidate_filename in union_names and not _path_is_in_shared_model_root(candidate, model_roots):
                    continue
                return candidate
            if os.path.isfile(candidate):
                _bootstrap_trace(f"_find_existing_model_file: ignoring invalid file {candidate}")
    return None


def ensure_shared_models_downloaded(shared_root: str, available_roots=None):
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

    available_shared_roots = _preferred_shared_model_roots(available_roots)
    if shared_root not in available_shared_roots:
        available_shared_roots.insert(0, shared_root)

    for folder_name, entries in SHARED_MODELS_URLS.items():
        target_dir = os.path.join(shared_root, folder_name)
        _bootstrap_trace(f"ensure_shared_models_downloaded: checking folder {folder_name} -> {target_dir}")

        # Prova a creare/validare la cartella; se non scrivibile, skip solo quella cartella
        if not _is_writable_directory(target_dir):
            logging.info(f"Cartella modelli non scrivibile, skip download per '{folder_name}': {target_dir}")
            _bootstrap_trace(f"ensure_shared_models_downloaded: target not writable, skipping folder {folder_name}")
            continue

        for url, filename in _normalize_model_entries(entries):
            existing_path = _find_existing_model_file(available_shared_roots, folder_name, filename)
            if existing_path:
                _bootstrap_trace(f"ensure_shared_models_downloaded: available elsewhere {filename} -> {existing_path}")
                continue

            dest_path = os.path.join(target_dir, filename)
            _bootstrap_trace(f"ensure_shared_models_downloaded: ensure file {dest_path}")
            _download_if_missing(url, dest_path)
            _bootstrap_trace(f"ensure_shared_models_downloaded: file ready {dest_path}")

    _bootstrap_trace(f"ensure_shared_models_downloaded: completed for root {shared_root}")


def _find_shared_model_entry(folder_name: str, filename: str):
    target_names = set(_equivalent_model_filenames(filename))
    for url, entry_filename in _normalize_model_entries(SHARED_MODELS_URLS.get(folder_name, [])):
        if entry_filename in target_names or filename in _equivalent_model_filenames(entry_filename):
            return url, entry_filename
    return None


def _canonical_shared_controlnet_path(model_roots, filename: str):
    if filename not in set(_equivalent_model_filenames("flux-Union-controlnet.safetensors")):
        return None

    download_root = _select_writable_model_root(model_roots)
    if not download_root:
        return None

    canonical_filename = "flux-Union-controlnet.safetensors"
    canonical_path = os.path.join(download_root, "controlnet", canonical_filename)
    if not _is_existing_model_file_valid(canonical_path):
        model_entry = _find_shared_model_entry("controlnet", canonical_filename)
        if not model_entry:
            return None
        url, _ = model_entry
        _bootstrap_trace(f"_canonical_shared_controlnet_path: downloading {canonical_filename} to {canonical_path}")
        _download_if_missing(url, canonical_path)

    if not _is_existing_model_file_valid(canonical_path):
        return None

    for alias_filename in _equivalent_model_filenames(canonical_filename):
        alias_path = os.path.join(download_root, "controlnet", alias_filename)
        if os.path.abspath(alias_path) == os.path.abspath(canonical_path):
            continue
        _try_symlink_file(canonical_path, alias_path, replace_existing=True)

    return canonical_path


def _resolve_model_roots():
    """
    Risolve le root modelli in modo portabile:
    - COMFYUI_MODEL_ROOTS (path separati da os.pathsep) se definita
    - COMFYUI_MODELS_DEFAULT_ROOT forza sempre la root primaria
    - /mnt/default-models o /mnt/shared/default-models vengono usate solo se gia' presenti
    - altrimenti usa una root locale al progetto per evitare mount non presenti/lenti
    """
    env_primary_root = os.environ.get("COMFYUI_MODELS_DEFAULT_ROOT", "").strip()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    local_shared_root = os.path.join(base_dir, "shared", "default-models")
    local_primary_root = local_shared_root
    vscode_shared_root = "/vscode/workspace/shared/default-models"
    mnt_default_root = "/mnt/default-models"
    mnt_shared_root = "/mnt/shared/default-models"

    if env_primary_root:
        primary_root = env_primary_root
    elif os.path.isdir(mnt_default_root):
        primary_root = mnt_default_root
    elif os.path.isdir(mnt_shared_root):
        primary_root = mnt_shared_root
    elif os.path.isdir(vscode_shared_root):
        primary_root = vscode_shared_root
    else:
        primary_root = local_primary_root

    secondary_root = _local_models_root()

    candidates = [primary_root, secondary_root]
    for shared_root in (local_shared_root, vscode_shared_root, mnt_default_root, mnt_shared_root):
        if os.path.abspath(shared_root) == os.path.abspath(vscode_shared_root) and not os.path.isdir(vscode_shared_root):
            continue
        if os.path.isdir(shared_root) or shared_root == local_shared_root:
            if os.path.abspath(shared_root) == os.path.abspath(primary_root):
                continue
            candidates.append(shared_root)

    env_value = os.environ.get("COMFYUI_MODEL_ROOTS", "").strip()
    if env_value:
        candidates.extend([item for item in env_value.split(os.pathsep) if item])

    roots = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append(normalized)

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
    Mantiene download su root primaria (shared) ma rende disponibili i file
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
    if _is_existing_model_file_valid(dest_path):
        return True

    if not _is_existing_model_file_valid(src_path):
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Rimuove path esistente non valido: file vuoto, symlink rotto, o symlink che punta
    # a una sorgente diversa (il file sorgente potrebbe essere cambiato).
    try:
        is_broken_symlink = os.path.islink(dest_path) and not os.path.exists(dest_path)
        is_invalid_file = os.path.isfile(dest_path) and not _is_existing_model_file_valid(dest_path)
        if is_broken_symlink or is_invalid_file:
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


def _try_symlink_file(src_path: str, dest_path: str, replace_existing: bool = False) -> bool:
    if not _is_existing_model_file_valid(src_path):
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        if os.path.islink(dest_path):
            if os.path.realpath(dest_path) == os.path.realpath(src_path):
                return _is_existing_model_file_valid(dest_path)
            os.remove(dest_path)
        elif os.path.exists(dest_path):
            if replace_existing:
                os.remove(dest_path)
            else:
                return (
                    os.path.realpath(dest_path) == os.path.realpath(src_path)
                    and _is_existing_model_file_valid(dest_path)
                )
    except Exception:
        return False

    try:
        rel_target = os.path.relpath(src_path, os.path.dirname(dest_path))
        os.symlink(rel_target, dest_path)
        return True
    except Exception:
        return False


def _ensure_real_directory(path: str) -> bool:
    try:
        if os.path.islink(path):
            os.remove(path)
        elif os.path.exists(path) and not os.path.isdir(path):
            os.remove(path)
        os.makedirs(path, exist_ok=True)
        return os.path.isdir(path) and not os.path.islink(path)
    except Exception as exc:
        _bootstrap_trace(f"_ensure_real_directory: failed for {path}: {exc}")
        return False


def _remove_invalid_model_file(path: str):
    try:
        if os.path.islink(path):
            if not os.path.exists(path) or not _is_existing_model_file_valid(path):
                os.remove(path)
                _bootstrap_trace(f"_remove_invalid_model_file: removed invalid symlink {path}")
            return

        if os.path.isfile(path) and not _is_existing_model_file_valid(path):
            os.remove(path)
            _bootstrap_trace(f"_remove_invalid_model_file: removed invalid file {path}")
    except Exception as exc:
        _bootstrap_trace(f"_remove_invalid_model_file: failed for {path}: {exc}")


def _ensure_direct_xlabs_controlnet_links(model_roots, filenames):
    local_target_dir = os.path.join(_local_models_root(), "xlabs", "controlnets")
    if not _ensure_real_directory(local_target_dir):
        return

    source_roots = []
    source_roots.extend(_preferred_shared_model_roots(model_roots))
    source_roots.extend(model_roots or [])

    seen_roots = set()
    for root in source_roots:
        if not root:
            continue
        root = os.path.abspath(root)
        if root in seen_roots or root == os.path.abspath(_local_models_root()):
            continue
        seen_roots.add(root)

        source_dir = os.path.join(root, "xlabs", "controlnets")
        if not os.path.isdir(source_dir):
            continue

        for filename in filenames:
            dest_path = os.path.join(local_target_dir, filename)
            for source_filename in _equivalent_model_filenames(filename):
                source_path = os.path.join(source_dir, source_filename)
                if not _is_existing_model_file_valid(source_path):
                    if os.path.isfile(source_path):
                        _bootstrap_trace(f"_ensure_direct_xlabs_controlnet_links: ignoring invalid source {source_path}")
                    continue

                if _try_symlink_file(source_path, dest_path, replace_existing=True):
                    _bootstrap_trace(f"_ensure_direct_xlabs_controlnet_links: linked {dest_path} -> {source_path}")
                break


def _ensure_xlabs_controlnet_layout(model_roots):
    """
    x-flux-comfyui non usa sempre folder_paths: alcuni loader cercano
    direttamente models/xlabs/controlnets/<file>. Espone li' i ControlNet XLabs.
    """
    if os.environ.get("COMFYUI_XLABS_CONTROLNET_COMPAT", "1") != "1":
        _bootstrap_trace("_ensure_xlabs_controlnet_layout: disabled by env")
        return

    if not model_roots:
        _bootstrap_trace("_ensure_xlabs_controlnet_layout: skipped because model_roots is empty")
        return

    filenames = [
        "flux-depth-controlnet-v3.safetensors",
        "flux-canny-controlnet-v3.safetensors",
        "flux-hed-controlnet-v3.safetensors",
        "flux-lineart-controlnet.safetensors",
        "flux-seg-controlnet.safetensors",
        "flux-Union-controlnet.safetensors",
        "flux-dev-controlnet-union-pro.safetensors",
        "flux-union-controlnet.safetensors",
        "manycore-FLUX.1-Layout-ControlNet.safetensors",
    ]
    local_models_root = _local_models_root()
    source_roots = []
    source_roots.extend(_preferred_shared_model_roots(model_roots))
    source_roots.append(local_models_root)
    source_roots.extend(model_roots)

    candidate_source_dirs = []
    seen_source_dirs = set()
    for root in source_roots:
        if not root:
            continue
        for source_dir in (
            os.path.join(root, "controlnet"),
            os.path.join(root, "controlnets"),
            os.path.join(root, "controlnet", "FLUX.1-Layout-ControlNet"),
            os.path.join(root, "controlnet", "manycore-research", "FLUX.1-Layout-ControlNet"),
            os.path.join(root, "xlabs", "flux"),
            os.path.join(root, "xlabs", "controlnets"),
            os.path.join(root, "xlabs", "controlnets", "FLUX.1-Layout-ControlNet"),
        ):
            normalized = os.path.abspath(source_dir)
            if normalized in seen_source_dirs:
                continue
            seen_source_dirs.add(normalized)
            candidate_source_dirs.append(normalized)

    download_root = _select_writable_model_root(model_roots)
    target_dir = os.path.join(local_models_root, "xlabs", "controlnets")
    if not _ensure_real_directory(target_dir):
        return

    # x-flux cerca direttamente models/xlabs/controlnets: deve essere una directory
    # locale reale, non un symlink di directory verso una root con file vecchi.
    for filename in filenames:
        _remove_invalid_model_file(os.path.join(target_dir, filename))

    _ensure_direct_xlabs_controlnet_links(model_roots, filenames)
    for root in [local_models_root]:
        target_dir = os.path.join(root, "xlabs", "controlnets")
        if not _ensure_real_directory(target_dir):
            continue
        for filename in filenames:
            source_path = None
            equivalent_filenames = _equivalent_model_filenames(filename)
            union_names = set(_equivalent_model_filenames("flux-Union-controlnet.safetensors"))
            if filename in union_names:
                source_path = _canonical_shared_controlnet_path(model_roots, filename)

            for source_dir in candidate_source_dirs:
                if source_path is not None:
                    break
                for source_filename in equivalent_filenames:
                    candidate = os.path.join(source_dir, source_filename)
                    if _is_existing_model_file_valid(candidate):
                        if filename in union_names and not _path_is_in_shared_model_root(candidate, model_roots):
                            continue
                        source_path = candidate
                        break
                    if os.path.isfile(candidate):
                        _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: ignoring invalid source {candidate}")
                if source_path is not None:
                    break

            if source_path is None:
                for root in source_roots:
                    if source_path is not None:
                        break
                    if not root:
                        continue
                    for rel_dir in ("controlnet", "controlnets", "xlabs"):
                        recursive_dir = os.path.join(root, rel_dir)
                        if not os.path.isdir(recursive_dir):
                            continue
                        for dirpath, dirnames, walk_filenames in os.walk(recursive_dir):
                            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                            for source_filename in equivalent_filenames:
                                if source_filename not in walk_filenames:
                                    continue
                                candidate = os.path.join(dirpath, source_filename)
                                if _is_existing_model_file_valid(candidate):
                                    source_path = candidate
                                    _bootstrap_trace(
                                        f"_ensure_xlabs_controlnet_layout: found recursive source {candidate}"
                                    )
                                    break
                                _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: ignoring invalid source {candidate}")
                            if source_path is not None:
                                break

            if source_path is None:
                model_entry = _find_shared_model_entry("controlnet", filename)
                if download_root and model_entry:
                    url, source_filename = model_entry
                    download_source = os.path.join(download_root, "controlnet", source_filename)
                    _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: downloading missing {filename} to {download_source}")
                    _download_if_missing(url, download_source)
                    if _is_existing_model_file_valid(download_source):
                        for alias_filename in equivalent_filenames:
                            alias_path = os.path.join(download_root, "controlnet", alias_filename)
                            if os.path.abspath(alias_path) != os.path.abspath(download_source):
                                _try_symlink_file(download_source, alias_path, replace_existing=True)
                        source_path = download_source
            if source_path is None and filename == "manycore-FLUX.1-Layout-ControlNet.safetensors":
                model_entry = _find_shared_model_entry(
                    "diffusers",
                    "manycore-research/FLUX.1-Layout-ControlNet/diffusion_pytorch_model.safetensors",
                )
                if download_root and model_entry:
                    url, source_filename = model_entry
                    download_source = os.path.join(download_root, "diffusers", source_filename)
                    _bootstrap_trace(
                        f"_ensure_xlabs_controlnet_layout: downloading missing {filename} to {download_source}"
                    )
                    _download_if_missing(url, download_source)
                    if _is_existing_model_file_valid(download_source):
                        source_path = download_source

            if source_path is None:
                _remove_invalid_model_file(os.path.join(target_dir, filename))
                _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: source not found for {filename}")
                continue

            dest_path = os.path.join(target_dir, filename)
            if os.path.realpath(source_path) == os.path.realpath(dest_path):
                continue

            if _try_symlink_file(source_path, dest_path, replace_existing=True):
                logging.info(f"Prepared XLabs ControlNet path: {dest_path} -> {source_path}")
                _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: prepared {dest_path} -> {source_path}")
            else:
                logging.warning(f"Unable to prepare XLabs ControlNet path {dest_path} from {source_path}")
                _bootstrap_trace(f"_ensure_xlabs_controlnet_layout: failed {dest_path} from {source_path}")


def _try_hf_snapshot_download(repo_id: str, local_dir: str, revision: str = "main", ignore_patterns=None) -> bool:
    """
    Scarica l'intero snapshot HF dentro una cartella locale (tutti i file del repo),
    utile per modelli che richiedono tokenizer/processor/config vari.
    """
    if os.environ.get("COMFYUI_FLORENCE2_SNAPSHOT", "1") != "1":
        return False

    try:
        _patch_huggingface_hub_download_tqdm_class_compat()
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
    Auto-download disabilitato di default: abilita solo con
    COMFYUI_ENABLE_FLORENCE2_AUTO_DOWNLOAD=1.
    """
    if os.environ.get("COMFYUI_ENABLE_FLORENCE2_AUTO_DOWNLOAD", "0") != "1":
        _bootstrap_trace("_ensure_florence2_layout: skipped (auto-download disabled)")
        return

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

    # Scarica SOLO nella root primaria (shared).
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
        if _is_existing_model_file_valid(dst):
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


def _ensure_llama3_layout(model_roots):
    """
    Rende disponibile Llama-3-8B-Instruct per i custom nodes.
    Crea symlink sia in LLM/<name> che in diffusers/<name> (richiesto da ComfyUI_Llama3_8B
    che hardcoda il path di caricamento come <comfyui_root>/models/diffusers/<name>).
    Se non trovato e COMFYUI_LLAMA3_AUTO_DOWNLOAD=1 con HF_TOKEN, scarica da HuggingFace.
    """
    if os.environ.get("COMFYUI_LLAMA3_LAYOUT", "1") != "1":
        _bootstrap_trace("_ensure_llama3_layout: disabled by env")
        return

    if not model_roots:
        _bootstrap_trace("_ensure_llama3_layout: skipped, no model roots")
        return

    repo_id = os.environ.get("COMFYUI_LLAMA3_REPO", "meta-llama/Meta-Llama-3-8B-Instruct").strip()
    canonical_name = repo_id.split("/")[-1]
    alias_names = [canonical_name, "Llama-3-8B-Instruct", "Meta-Llama-3-8B-Instruct"]

    source_dir = None
    for root in model_roots:
        for subdir in ("diffusers", "LLM", "llm", ""):
            for name in alias_names:
                candidate = os.path.join(root, subdir, name) if subdir else os.path.join(root, name)
                if not os.path.isdir(candidate):
                    continue
                try:
                    files = os.listdir(candidate)
                    if any(f.endswith((".safetensors", ".bin")) or f == "config.json" for f in files):
                        source_dir = os.path.abspath(candidate)
                        _bootstrap_trace(f"_ensure_llama3_layout: found at {source_dir}")
                        break
                except Exception:
                    pass
            if source_dir:
                break
        if source_dir:
            break

    if source_dir is None:
        hf_token = (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
            or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        )
        if hf_token and os.environ.get("COMFYUI_LLAMA3_AUTO_DOWNLOAD", "0") == "1":
            target_dir = os.path.join(model_roots[0], "LLM", canonical_name)
            os.makedirs(target_dir, exist_ok=True)
            _bootstrap_trace(f"_ensure_llama3_layout: downloading {repo_id} to {target_dir}")
            try:
                _patch_huggingface_hub_download_tqdm_class_compat()
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=target_dir, token=hf_token)
                logging.info(f"[BOOTSTRAP] Downloaded Llama-3: {repo_id} -> {target_dir}")
                source_dir = target_dir
            except Exception as exc:
                logging.warning(f"[BOOTSTRAP] Llama-3 download failed: {exc}")
        else:
            _bootstrap_trace(
                "_ensure_llama3_layout: model not found locally "
                "(set COMFYUI_LLAMA3_AUTO_DOWNLOAD=1 + HF_TOKEN to auto-download)"
            )
            return

    if source_dir is None:
        return

    for root in model_roots:
        # Symlink in LLM/ (per nodi che cercano in LLM/)
        llm_root = os.path.join(root, "LLM")
        try:
            os.makedirs(llm_root, exist_ok=True)
        except Exception:
            pass
        else:
            for link_name in alias_names:
                dest = os.path.join(llm_root, link_name)
                if os.path.realpath(source_dir) == os.path.realpath(os.path.abspath(dest)):
                    break
                if os.path.exists(dest) or os.path.islink(dest):
                    break
                try:
                    rel_target = os.path.relpath(source_dir, llm_root)
                    os.symlink(rel_target, dest, target_is_directory=True)
                    logging.info(f"[BOOTSTRAP] Linked Llama-3 (LLM): {dest} -> {source_dir}")
                    _bootstrap_trace(f"_ensure_llama3_layout: linked LLM/{link_name}")
                    break
                except Exception as exc:
                    _bootstrap_trace(f"_ensure_llama3_layout: symlink LLM/{link_name} failed: {exc}")

        # Symlink in diffusers/ (richiesto da ComfyUI_Llama3_8B che hardcoda
        # il path come <comfyui_root>/models/diffusers/<name>).
        diffusers_root = os.path.join(root, "diffusers")
        try:
            os.makedirs(diffusers_root, exist_ok=True)
        except Exception:
            continue

        for link_name in alias_names:
            dest = os.path.join(diffusers_root, link_name)
            if os.path.realpath(source_dir) == os.path.realpath(os.path.abspath(dest)):
                break
            if os.path.exists(dest) or os.path.islink(dest):
                break
            try:
                rel_target = os.path.relpath(source_dir, diffusers_root)
                os.symlink(rel_target, dest, target_is_directory=True)
                logging.info(f"[BOOTSTRAP] Linked Llama-3 (diffusers): {dest} -> {source_dir}")
                _bootstrap_trace(f"_ensure_llama3_layout: linked diffusers/{link_name}")
                break
            except Exception as exc:
                _bootstrap_trace(f"_ensure_llama3_layout: symlink diffusers/{link_name} failed: {exc}")

    _bootstrap_trace("_ensure_llama3_layout: completed")


def _ensure_llama32_1b_layout(model_roots):
    """
    Rende disponibile Llama-3.2-1B-Instruct (~2GB) nella cartella LLM.
    Modello gated Meta: richiede HF_TOKEN e accesso approvato su HuggingFace.
    Download automatico abilitato con COMFYUI_LLAMA32_1B_AUTO_DOWNLOAD=1 + HF_TOKEN.
    """
    if os.environ.get("COMFYUI_LLAMA32_1B_LAYOUT", "1") != "1":
        _bootstrap_trace("_ensure_llama32_1b_layout: disabled by env")
        return

    if not model_roots:
        _bootstrap_trace("_ensure_llama32_1b_layout: skipped, no model roots")
        return

    repo_id = os.environ.get("COMFYUI_LLAMA32_1B_REPO", "meta-llama/Llama-3.2-1B-Instruct").strip()
    canonical_name = repo_id.split("/")[-1]
    alias_names = [canonical_name, "Llama-3.2-1B-Instruct", "llama-3.2-1b-instruct"]

    source_dir = None
    for root in model_roots:
        for subdir in ("diffusers", "LLM", "llm", ""):
            for name in alias_names:
                candidate = os.path.join(root, subdir, name) if subdir else os.path.join(root, name)
                if not os.path.isdir(candidate):
                    continue
                try:
                    files = os.listdir(candidate)
                    if any(f.endswith((".safetensors", ".bin")) or f == "config.json" for f in files):
                        source_dir = os.path.abspath(candidate)
                        _bootstrap_trace(f"_ensure_llama32_1b_layout: found at {source_dir}")
                        break
                except Exception:
                    pass
            if source_dir:
                break
        if source_dir:
            break

    if source_dir is None:
        hf_token = (
            os.environ.get("HF_TOKEN", "").strip()
            or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
            or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
        )
        if hf_token and os.environ.get("COMFYUI_LLAMA32_1B_AUTO_DOWNLOAD", "0") == "1":
            target_dir = os.path.join(model_roots[0], "LLM", canonical_name)
            os.makedirs(target_dir, exist_ok=True)
            _bootstrap_trace(f"_ensure_llama32_1b_layout: downloading {repo_id} to {target_dir}")
            try:
                _patch_huggingface_hub_download_tqdm_class_compat()
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=target_dir, token=hf_token)
                logging.info(f"[BOOTSTRAP] Downloaded Llama-3.2-1B: {repo_id} -> {target_dir}")
                source_dir = target_dir
            except Exception as exc:
                logging.warning(f"[BOOTSTRAP] Llama-3.2-1B download failed: {exc}")
        else:
            _bootstrap_trace(
                "_ensure_llama32_1b_layout: model not found locally "
                "(set COMFYUI_LLAMA32_1B_AUTO_DOWNLOAD=1 + HF_TOKEN to auto-download)"
            )
            return

    if source_dir is None:
        return

    for root in model_roots:
        llm_root = os.path.join(root, "LLM")
        try:
            os.makedirs(llm_root, exist_ok=True)
        except Exception:
            continue

        for link_name in alias_names:
            dest = os.path.join(llm_root, link_name)
            if os.path.realpath(source_dir) == os.path.realpath(os.path.abspath(dest)):
                break
            if os.path.exists(dest) or os.path.islink(dest):
                break
            try:
                rel_target = os.path.relpath(source_dir, llm_root)
                os.symlink(rel_target, dest, target_is_directory=True)
                logging.info(f"[BOOTSTRAP] Linked Llama-3.2-1B: {dest} -> {source_dir}")
                _bootstrap_trace(f"_ensure_llama32_1b_layout: linked {dest}")
                break
            except Exception as exc:
                _bootstrap_trace(f"_ensure_llama32_1b_layout: symlink {dest} failed: {exc}")

    _bootstrap_trace("_ensure_llama32_1b_layout: completed")


def _ensure_local_model_symlinks(shared_root: str, local_models_root: str):
    """
    Per ogni file in shared_root crea un symlink corrispondente in local_models_root.
    I download avvengono solo su shared; models/ contiene solo symlink.
    """
    if not shared_root or not local_models_root:
        return
    if os.path.abspath(shared_root) == os.path.abspath(local_models_root):
        return
    if not os.path.isdir(shared_root):
        return

    _bootstrap_trace(f"_ensure_local_model_symlinks: {shared_root} -> {local_models_root}")

    for dirpath, dirnames, filenames in os.walk(shared_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, shared_root)
        dst_dir = os.path.join(local_models_root, rel_dir) if rel_dir != "." else local_models_root

        try:
            os.makedirs(dst_dir, exist_ok=True)
        except Exception:
            continue

        for filename in filenames:
            if filename.startswith(".") or _is_shared_model_placeholder(filename):
                continue
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_dir, filename)

            if not os.path.isfile(src_file):
                continue

            if os.path.islink(dst_file):
                if os.path.realpath(dst_file) == os.path.realpath(src_file):
                    if _is_existing_model_file_valid(dst_file):
                        continue
                    try:
                        os.remove(dst_file)
                    except Exception:
                        continue
                else:
                    try:
                        os.remove(dst_file)
                    except Exception:
                        continue
            elif os.path.isfile(dst_file):
                if _is_existing_model_file_valid(dst_file):
                    continue
                try:
                    os.remove(dst_file)
                except Exception:
                    continue

            if not _is_existing_model_file_valid(src_file):
                _remove_invalid_model_file(dst_file)
                _bootstrap_trace(f"_ensure_local_model_symlinks: skip invalid source {src_file}")
                continue

            if _try_symlink_file(src_file, dst_file):
                _bootstrap_trace(f"_ensure_local_model_symlinks: linked {dst_file} -> {src_file}")
            else:
                _bootstrap_trace(f"_ensure_local_model_symlinks: unable to link {dst_file} -> {src_file}")

    _bootstrap_trace(f"_ensure_local_model_symlinks: completed")


def _ensure_shared_models_visible_in_local_models(model_roots, download_root=None):
    local_models_root = _local_models_root()
    try:
        os.makedirs(local_models_root, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to prepare local models symlink root {local_models_root}: {exc}")
        _bootstrap_trace(f"_ensure_shared_models_visible_in_local_models: cannot prepare {local_models_root}: {exc}")
        return

    candidates = []
    if download_root:
        candidates.append(download_root)
    candidates.extend(_preferred_shared_model_roots(model_roots))

    seen = set()
    for shared_root in candidates:
        if not shared_root:
            continue
        shared_root = os.path.abspath(shared_root)
        if shared_root in seen or shared_root == os.path.abspath(local_models_root):
            continue
        seen.add(shared_root)
        if not os.path.isdir(shared_root):
            continue
        _ensure_local_model_symlinks(shared_root, local_models_root)


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

    download_root = _select_writable_model_root(model_roots)
    if download_root:
        _bootstrap_trace(f"apply_shared_model_paths: ensure shared models begin on {download_root}")
        ensure_shared_models_downloaded(download_root, model_roots)
    else:
        _bootstrap_trace("apply_shared_model_paths: skipped downloads because shared/default-models is not writable")
    _bootstrap_trace("apply_shared_model_paths: ensure shared models completed")
    _bootstrap_trace("apply_shared_model_paths: first LLM sync begin")
    _sync_llm_primary_to_secondary(model_roots)
    _bootstrap_trace("apply_shared_model_paths: first LLM sync completed")
    # _bootstrap_trace("apply_shared_model_paths: Florence2 layout begin")
    # _ensure_florence2_layout(model_roots)
    # _bootstrap_trace("apply_shared_model_paths: Florence2 layout completed")
    _bootstrap_trace("apply_shared_model_paths: second LLM sync begin")
    _sync_llm_primary_to_secondary(model_roots)
    _bootstrap_trace("apply_shared_model_paths: second LLM sync completed")
    _ensure_xlabs_controlnet_layout(model_roots)
    _bootstrap_trace("apply_shared_model_paths: XLabs ControlNet layout completed")
    _ensure_llama3_layout(model_roots)
    _bootstrap_trace("apply_shared_model_paths: Llama3 layout completed")
    _ensure_llama32_1b_layout(model_roots)
    _bootstrap_trace("apply_shared_model_paths: Llama3.2-1B layout completed")
    _ensure_shared_models_visible_in_local_models(model_roots, download_root)
    _bootstrap_trace("apply_shared_model_paths: local model symlinks completed")

    model_dirs = {
        "checkpoints": "checkpoints",
        "loras": "loras",
        "vae": "vae",
        "clip": "clip",
        "diffusion_models": "diffusion_models",
        "transformer": "diffusion_models",
        "embeddings": "embeddings",
        "controlnet": "controlnet",
        "control_net": "controlnet",
        "controlnets": "controlnet",
        "xlabs_controlnet": "controlnet",
        "xlabs_controlnets": "controlnet",
        "upscale_models": "upscale_models",
        "clip_vision": "clip_vision",
        "style_models": "style_models",
        "gligen": "gligen",
        "hypernetworks": "hypernetworks",
        "vae_approx": "vae_approx",
        "unet": "unet",
        "text_encoders": "text_encoders",
        "t5": "text_encoders",
        "clip_l": "text_encoders",
        # Compat Florence/LLM: alcuni nodi cercano "LLM", altri "llm".
        "LLM": "LLM",
        "llm": "LLM",
        # HuggingFace-style models (es. Llama-3 in diffusers/).
        "diffusers": "diffusers",
        # GGUF quantized models (llama.cpp / ComfyUI-GGUF nodes).
        "gguf": "gguf",
        # IP-Adapter models.
        "ipadapter": "ipadapter",
        "ip_adapter": "ipadapter",
        # Inpainting models.
        "inpaint": "inpaint",
    }

    # Aggiunge TUTTE le cartelle per ogni tipo modello
    for root in model_roots:
        for model_type, subdir in model_dirs.items():
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


def _patch_comfy_utils_mmap_fallback():
    """
    comfy_aimdo.model_mmap.ModelMMAP usa mmap() che fallisce su filesystem
    di rete/condivisi (NFS, virtio-fs, FUSE, ecc.) perché non supportano mmap().
    Questo patch intercetta il RuntimeError da ModelMMAP in comfy.utils.load_safetensors
    e fa fallback al caricamento standard safetensors senza mmap.
    """
    if os.environ.get("COMFY_AIMDO_DISABLE_MMAP", "1") != "1":
        return

    try:
        import comfy.utils as _cu
    except ImportError:
        return

    _orig = getattr(_cu, "load_safetensors", None)
    if not callable(_orig) or getattr(_orig, "_mmap_fallback_patched", False):
        return

    def _patched_load_safetensors(ckpt):
        try:
            return _orig(ckpt)
        except RuntimeError as _exc:
            msg = str(_exc)
            if "ModelMMAP" not in msg and "mmap" not in msg.lower():
                raise
            # mmap non supportato dal filesystem: fallback a caricamento standard
            try:
                from safetensors import safe_open as _safe_open
                sd = {}
                metadata = {}
                with _safe_open(ckpt, framework="pt", device="cpu") as _f:
                    metadata = _f.metadata() or {}
                    for k in _f.keys():
                        sd[k] = _f.get_tensor(k)
                logging.info(f"[BOOTSTRAP] mmap fallback: loaded {ckpt} via safe_open (no mmap)")
                return sd, metadata
            except Exception as _fe:
                _fe_msg = str(_fe)
                # "header too small" / "header invalid" → file corrotto o download incompleto
                _is_corrupt = any(x in _fe_msg.lower() for x in (
                    "header too small", "header invalid", "header",
                    "not a safetensors file", "unexpected end",
                ))
                if _is_corrupt:
                    import os as _os
                    _size = _os.path.getsize(ckpt) if _os.path.isfile(ckpt) else -1
                    raise RuntimeError(
                        f"File corrotto o download incompleto ({_size} bytes): {ckpt}\n"
                        f"Elimina il file e riavvia ComfyUI per riscaricarlo."
                    ) from _fe
                raise RuntimeError(
                    f"mmap fallback failed for {ckpt}: {_fe}"
                ) from _exc

    _patched_load_safetensors._mmap_fallback_patched = True
    _cu.load_safetensors = _patched_load_safetensors
    print("[BOOTSTRAP] Installed comfy.utils.load_safetensors mmap fallback patch", flush=True)


_ensure_comfy_env_stub()
_patch_xflux_vram_management()
_patch_comfy_utils_mmap_fallback()
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
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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
    "diffusion_models": "diffusion_models",
    "transformer": "diffusion_models",
    "embeddings": "embeddings",
    "controlnet": "controlnet",
    "control_net": "controlnet",
    "controlnets": "controlnet",
    "xlabs_controlnet": "controlnet",
    "xlabs_controlnets": "controlnet",
    "upscale_models": "upscale_models",
    "clip_vision": "clip_vision",
    "style_models": "style_models",
    "gligen": "gligen",
    "hypernetworks": "hypernetworks",
    "vae_approx": "vae_approx",
    "unet": "unet",
    "text_encoders": "text_encoders",
    "t5": "text_encoders",
    "clip_l": "text_encoders",
    # Compat Florence/LLM: espone la stessa cartella con entrambe le chiavi.
    "LLM": "LLM",
    "llm": "LLM",
    # GGUF quantized models (llama.cpp / ComfyUI-GGUF nodes).
    "gguf": "gguf",
    # IP-Adapter models.
    "ipadapter": "ipadapter",
    "ip_adapter": "ipadapter",
    # Inpainting models.
    "inpaint": "inpaint",
}

MODEL_ROOTS = [
    # Resta come fallback statico, ma il wrapper usa _resolve_model_roots().
    "/mnt/shared/default-models",
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
    o cache custom-node-list senza schema valido che causano errori in comfyui-manager.
    Rinomina solo i file cache corrotti/invalidi, cosi' Manager li puo' rigenerare.
    """
    if os.environ.get("COMFYUI_MANAGER_CLEANUP_BROKEN_CACHE", "1") != "1":
        return

    import json

    base_dir = os.path.dirname(os.path.realpath(__file__))
    manager_dir = os.path.join(base_dir, "custom_nodes", COMFYUI_MANAGER_DIRNAME)
    user_dir = os.path.join(base_dir, "user")

    for index, arg in enumerate(sys.argv):
        if arg == "--user-directory" and index + 1 < len(sys.argv):
            user_dir = os.path.abspath(sys.argv[index + 1])
            break
        if arg.startswith("--user-directory="):
            user_dir = os.path.abspath(arg.split("=", 1)[1])
            break

    cache_roots = []
    for cache_root in [
        os.path.join(manager_dir, ".cache"),
        os.path.join(manager_dir, "cache"),
        os.path.join(base_dir, ".cache", "comfyui-manager"),
        os.path.join(user_dir, "__manager", "cache"),
        os.path.join(user_dir, "default", "ComfyUI-Manager", "cache"),
    ]:
        cache_root = os.path.abspath(cache_root)
        if cache_root not in cache_roots:
            cache_roots.append(cache_root)

    def _rename_bad_cache(file_path, reason):
        backup_path = f"{file_path}.corrupt"
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.replace(file_path, backup_path)
            logging.warning(
                "[WRAPPER] Renamed invalid manager cache JSON: %s -> %s (%s)",
                file_path,
                backup_path,
                reason,
            )
            return True
        except Exception as move_exc:
            logging.warning(
                "[WRAPPER] Failed handling invalid manager cache JSON %s: %s",
                file_path,
                move_exc,
            )
            return False

    def _custom_node_list_schema_error(name, json_obj):
        lower_name = name.lower()
        if lower_name != "custom-node-list.json" and not lower_name.endswith("_custom-node-list.json"):
            return None

        if not isinstance(json_obj, dict):
            return "custom-node-list root is not an object"
        if not isinstance(json_obj.get("custom_nodes"), list):
            return "custom-node-list is missing custom_nodes list"
        return None

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
                        json_obj = json.load(json_file)
                    schema_error = _custom_node_list_schema_error(name, json_obj)
                    if schema_error and _rename_bad_cache(file_path, schema_error):
                        renamed += 1
                except json.JSONDecodeError as exc:
                    if _rename_bad_cache(file_path, exc):
                        renamed += 1
                except Exception:
                    # Ignora file non leggibili o lock temporanei.
                    pass

    if renamed:
        logging.info("[WRAPPER] Cleaned %d corrupt ComfyUI-Manager cache JSON file(s)", renamed)


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
                for exposed_filename in _equivalent_model_filenames(filename):
                    known_by_type.setdefault(model_type, set()).add(exposed_filename)
                    sources[(model_type, exposed_filename)] = (folder_name, url, filename)

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

    if len(source) == 2:
        folder_name, url = source
        source_filename = filename
    else:
        folder_name, url, source_filename = source
    model_roots = _resolve_model_roots()
    if not model_roots:
        return None

    canonical_controlnet_path = _canonical_shared_controlnet_path(model_roots, filename)
    if canonical_controlnet_path:
        _ensure_xlabs_controlnet_layout(model_roots)
        return canonical_controlnet_path

    download_root = _select_writable_model_root(model_roots)
    if not download_root:
        return None
    source_path = os.path.join(download_root, folder_name, source_filename)
    dest_path = os.path.join(download_root, folder_name, filename)
    try:
        _download_if_missing(url, source_path)
        if os.path.abspath(source_path) != os.path.abspath(dest_path):
            _try_symlink_file(source_path, dest_path, replace_existing=True)
        if folder_name == "controlnet":
            _ensure_xlabs_controlnet_layout(model_roots)
    except Exception as exc:
        logging.warning(
            f"[WRAPPER] Unable to prepare bootstrap model '{filename}' for '{model_type}': {exc}"
        )
        return None

    if _is_existing_model_file_valid(dest_path):
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


def _get_known_flux_controlnet_filenames():
    names = set()
    incompatible_xflux_names = {
        "manycore-flux.1-layout-controlnet.safetensors",
        "diffusion_pytorch_model.safetensors",
    }
    for model_type in (
        "controlnet",
        "control_net",
        "controlnets",
        "xlabs_controlnet",
        "xlabs_controlnets",
    ):
        names.update(_get_known_bootstrap_model_filenames(model_type))

    expanded = set()
    for name in names:
        for equivalent in _equivalent_model_filenames(name):
            normalized = equivalent.lower()
            if normalized in incompatible_xflux_names:
                continue
            if "manycore" in normalized or "layout-controlnet" in normalized:
                continue
            if "flux" in normalized:
                expanded.add(equivalent)

    return sorted(expanded)


def _merge_selector_options(input_spec, extra_options):
    if not isinstance(input_spec, (tuple, list)) or not input_spec:
        return input_spec

    current_options = input_spec[0]
    if not isinstance(current_options, (tuple, list)):
        return input_spec

    merged = []
    seen = set()
    for option in list(current_options) + list(extra_options):
        if not isinstance(option, str) or option in seen:
            continue
        seen.add(option)
        merged.append(option)

    if isinstance(input_spec, tuple):
        return (merged, *input_spec[1:])
    return [merged, *input_spec[1:]]


def _extend_flux_controlnet_input_types(input_types):
    if not isinstance(input_types, dict):
        return input_types

    extra_options = _get_known_flux_controlnet_filenames()
    if not extra_options:
        return input_types

    patched = dict(input_types)
    for section_name in ("required", "optional"):
        section = patched.get(section_name)
        if not isinstance(section, dict):
            continue

        section = dict(section)
        for input_name, input_spec in list(section.items()):
            normalized_name = str(input_name).lower()
            if normalized_name not in {"controlnet_path", "controlnet_name", "controlnet"}:
                continue
            section[input_name] = _merge_selector_options(input_spec, extra_options)

        patched[section_name] = section

    return patched


def _patch_flux_controlnet_loader_selectors():
    if os.environ.get("COMFYUI_PATCH_XFLUX_CONTROLNET_SELECTOR", "1") != "1":
        return

    try:
        import nodes as _comfy_nodes
    except Exception as exc:
        logging.debug(f"[WRAPPER] LoadFluxControlNet selector patch waiting for nodes module: {exc}")
        return

    mappings = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        return

    patched_count = 0
    for node_name, node_class in list(mappings.items()):
        if node_name != "LoadFluxControlNet":
            continue
        if getattr(node_class, "_comfyui_flux_controlnet_selector_patch", False):
            continue

        original_input_types = getattr(node_class, "INPUT_TYPES", None)
        if not callable(original_input_types):
            continue

        def _wrapped_input_types(cls, __orig=original_input_types):
            try:
                input_types = __orig()
            except TypeError:
                input_types = __orig(cls)
            return _extend_flux_controlnet_input_types(input_types)

        node_class.INPUT_TYPES = classmethod(_wrapped_input_types)
        node_class._comfyui_flux_controlnet_selector_patch = True
        patched_count += 1

    if patched_count:
        logging.info("[WRAPPER] Patched LoadFluxControlNet selectors with bootstrap Flux ControlNet filenames")


def _iter_prompt_nodes(payload):
    if isinstance(payload, dict):
        if "class_type" in payload and "inputs" in payload:
            yield payload
            return

        for value in payload.values():
            yield from _iter_prompt_nodes(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_prompt_nodes(value)


def _ensure_prompt_flux_controlnets_available(prompt):
    if os.environ.get("COMFYUI_PREPARE_PROMPT_FLUX_CONTROLNETS", "1") != "1":
        return

    known = set(_get_known_flux_controlnet_filenames())
    if not known:
        return

    for node in _iter_prompt_nodes(prompt):
        if node.get("class_type") != "LoadFluxControlNet":
            continue

        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        for input_name in ("controlnet_path", "controlnet_name", "controlnet"):
            value = inputs.get(input_name)
            if not isinstance(value, str) or value not in known:
                continue
            _ensure_known_bootstrap_model_available("controlnet", value)


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
        _patch_flux_controlnet_loader_selectors()
        normalized_prompt = _normalize_prompt_payload_paths(prompt)
        _ensure_prompt_flux_controlnets_available(normalized_prompt)
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
    _bootstrap_trace("_preflight_custom_logic: Hugging Face runtime compat patch begin")
    _patch_huggingface_hub_download_tqdm_class_compat()
    _bootstrap_trace("_preflight_custom_logic: Hugging Face runtime compat patch completed")
    _bootstrap_trace("_preflight_custom_logic: ControlNet Aux OneFormer CPU compat patch begin")
    _install_controlnet_aux_oneformer_cpu_compat()
    _bootstrap_trace("_preflight_custom_logic: ControlNet Aux OneFormer CPU compat patch completed")

    # 1b) ripulisce eventuali cache JSON corrotte di ComfyUI-Manager
    _bootstrap_trace("_preflight_custom_logic: cleanup manager cache begin")
    _cleanup_broken_manager_json_cache()
    _bootstrap_trace("_preflight_custom_logic: cleanup manager cache completed")

    # 2) env vars opzionali
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # Punta HF_HOME alla root modelli condivisa (persistente tra i restart).
    # Così i modelli scaricati da nodi come DepthAnythingV3 vengono cachati
    # in /mnt/shared/default-models/.huggingface e non riscaricati a ogni avvio.
    # Override con HF_HOME nell'environment per disabilitare.
    if not os.environ.get("HF_HOME"):
        _hf_cache_roots = _resolve_model_roots()
        if _hf_cache_roots:
            _hf_home = os.path.join(_hf_cache_roots[0], ".huggingface")
            try:
                os.makedirs(_hf_home, exist_ok=True)
                os.environ["HF_HOME"] = _hf_home
                logging.info(f"[WRAPPER] Set HF_HOME to persistent cache: {_hf_home}")
            except Exception as _hf_exc:
                logging.warning(f"[WRAPPER] Could not set HF_HOME to {_hf_home}: {_hf_exc}")

    # 2a) Normalizza i workflow salvati con separatori Windows.
    base_dir = os.path.dirname(os.path.realpath(__file__))
    _ensure_comfy_env_stub()
    _patch_xflux_vram_management()
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

    # 4) download modelli mancanti in shared/default-models quando possibile.
    if model_roots:
        download_root = _select_writable_model_root(model_roots)
        if download_root:
            _bootstrap_trace(f"_preflight_custom_logic: shared model bootstrap begin on {download_root}")
            ensure_shared_models_downloaded(download_root, model_roots)
        else:
            _bootstrap_trace("_preflight_custom_logic: skipped downloads because shared/default-models is not writable")
        _bootstrap_trace("_preflight_custom_logic: shared model downloads completed")
        _sync_llm_primary_to_secondary(model_roots)
        _bootstrap_trace("_preflight_custom_logic: first LLM sync completed")
        # _ensure_florence2_layout(model_roots)
        # _bootstrap_trace("_preflight_custom_logic: Florence2 layout completed")
        _sync_llm_primary_to_secondary(model_roots)
        _bootstrap_trace("_preflight_custom_logic: second LLM sync completed")
        _ensure_xlabs_controlnet_layout(model_roots)
        _bootstrap_trace("_preflight_custom_logic: XLabs ControlNet layout completed")
        _ensure_llama3_layout(model_roots)
        _bootstrap_trace("_preflight_custom_logic: Llama3 layout completed")
        _ensure_llama32_1b_layout(model_roots)
        _bootstrap_trace("_preflight_custom_logic: Llama3.2-1B layout completed")
        _ensure_shared_models_visible_in_local_models(model_roots, download_root)
        _bootstrap_trace("_preflight_custom_logic: local model symlinks completed")

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
    if os.environ.get("COMFYUI_PRINT_INSTALLED_PACKAGES", "0") != "1":
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
