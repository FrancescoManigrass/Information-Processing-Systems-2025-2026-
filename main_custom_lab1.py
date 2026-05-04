import os
import sys
import subprocess
import shutil
import ctypes.util
import tempfile
import urllib.request
import venv
import glob
import types
import colorsys


COMFYUI_MANAGER_REPO_URL = "https://github.com/Comfy-Org/ComfyUI-Manager.git"
COMFYUI_MANAGER_DIRNAME = "comfyui-manager"
COMFYUI_MANAGER_LEGACY_DIRNAME = "ComfyUI-Manager"
LOCAL_VENV_DIRNAME = ".venv"
OPENCV_GUI_PACKAGES = (
    "opencv-python",
    "opencv-contrib-python",
)
OPENCV_HEADLESS_PACKAGE = "opencv-python-headless"


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
        #{"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "filename": "clip_l.safetensors"},
        #{"url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors", "filename": "t5xxl_fp8_e4m3fn_scaled.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true", "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"},

        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors?download=true", "filename": "clip_l_hunyuan.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors?download=true", "filename": "llava_llama3_fp8_scaled.safetensors"},
    ],

    # =========================
    # VAE
    # =========================
    "vae": [
        #{"url": "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "filename": "ae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors", "filename": "qwen_image_vae.safetensors"},
        #{"url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true", "filename": "wan_2.1_vae.safetensors"},
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
        #{"url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/qwen_image_union_diffsynth_lora.safetensors", "filename": "qwen_image_union_diffsynth_lora.safetensors"},
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


def _run_cmd_quiet(command):
    try:
        subprocess.check_call(command)
        return True
    except Exception as exc:
        print(f"[BOOTSTRAP] Command failed: {' '.join(command)} -> {exc}")
        return False


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


def _should_force_headless_opencv():
    return sys.platform.startswith("linux") and ctypes.util.find_library("GL") is None


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

    print("[BOOTSTRAP] libGL not found, normalizing OpenCV packages to headless variants")
    pip_cmd = _get_bootstrap_pip_cmd()
    changed_any = False

    try:
        subprocess.check_call(pip_cmd + [
            "uninstall",
            "-y",
            *OPENCV_GUI_PACKAGES,
            OPENCV_HEADLESS_PACKAGE,
        ])
        changed_any = True
    except Exception:
        pass

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
        return _get_bootstrap_pip_cmd(python_executable) + ["install", *install_args]
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
            *install_args,
        ])

    if shutil.which("uv"):
        fallback_commands.append([
            "uv",
            "pip",
            "install",
            "--python",
            python_executable,
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


def ensure_local_venv():
    if __name__ != "__main__":
        return

    base_dir = os.path.dirname(os.path.realpath(__file__))
    if not _should_reexec_into_local_venv(base_dir):
        return

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
    if __name__ != "__main__":
        return

    if os.environ.get("COMFYUI_AUTO_INSTALL_REQUIREMENTS", "1") != "1":
        return

    _ensure_current_python_package_manager()

    installed_any = False

    if _ensure_headless_opencv():
        installed_any = True



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

    
    
    for pkg in extra_packages:
        print(f"[BOOTSTRAP] Installing extra package: {pkg}")
        subprocess.check_call(_get_bootstrap_install_cmd(
            "--disable-pip-version-check",
            pkg,
        ))
        installed_any = True
    seen = set()
    for req in req_files:
        req = os.path.abspath(req)
        if req in seen:
            continue
        seen.add(req)

        print(f"[BOOTSTRAP] Installing requirements from: {req}")
        try:
            subprocess.check_call(_get_bootstrap_install_cmd(
                "--disable-pip-version-check",
                "-r",
                req,
            ))
            installed_any = True
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

    if _should_force_headless_opencv():
        if _ensure_headless_opencv():
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


# Install custom nodes PRIMA del bootstrap requirements, così i loro requirements vengono inclusi.
ensure_local_venv()
ensure_comfyui_manager_installed()

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


def _resolve_model_roots():
    """
    Risolve le root modelli in modo portabile:
    - COMFYUI_MODEL_ROOTS (path separati da os.pathsep) se definita
    - fallback locale relativo al file main_custom.py
    """
    base_dir = os.path.dirname(os.path.realpath(__file__))
    primary_root = os.environ.get("COMFYUI_MODELS_DEFAULT_ROOT", "").strip() or os.path.join(base_dir, "models-default")
    secondary_root = os.environ.get("COMFYUI_MODELS_ROOT", "").strip() or os.path.join(base_dir, "models")

    candidates = [primary_root, secondary_root]

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

    return roots


def _ensure_llm_subdirs(model_roots):
    """
    Alcuni nodi (es. Florence2ModelLoader) cercano esplicitamente models/LLM.
    Garantisce che la cartella esista in ogni root registrata.
    """
    for root in model_roots:
        try:
            os.makedirs(os.path.join(root, "LLM"), exist_ok=True)
        except Exception as exc:
            logging.warning(f"Unable to create LLM folder in {root}: {exc}")


def _sync_llm_primary_to_secondary(model_roots):
    """
    Mantiene download su root primaria (models-default) ma rende disponibili i file
    anche in root secondaria (models) per nodi che usano path hardcoded models/LLM.
    """
    if os.environ.get("COMFYUI_SYNC_LLM_TO_SECONDARY", "0") != "1":
        return

    if len(model_roots) < 2:
        return

    primary_llm = os.path.join(model_roots[0], "LLM")
    secondary_llm = os.path.join(model_roots[1], "LLM")

    try:
        os.makedirs(primary_llm, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to prepare primary LLM folder {primary_llm}: {exc}")
        return

    if os.path.realpath(primary_llm) == os.path.realpath(secondary_llm):
        return

    if not os.path.exists(secondary_llm):
        try:
            os.symlink(primary_llm, secondary_llm, target_is_directory=True)
            logging.info(f"Linked LLM folder: {secondary_llm} -> {primary_llm}")
            return
        except Exception:
            pass

    try:
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
    except Exception as exc:
        logging.warning(f"Unable to sync LLM folders {primary_llm} -> {secondary_llm}: {exc}")


def _ensure_florence2_layout(model_roots):
    """
    Florence2ModelLoader in genere cerca una CARTELLA modello dentro LLM,
    non un singolo file .safetensors nella root LLM.
    Costruisce un layout compatibile: LLM/Florence-2-large/...
    """
    if os.environ.get("COMFYUI_ENABLE_FLORENCE2_AUTO_DOWNLOAD", "0") != "1":
        _bootstrap_trace("_ensure_florence2_layout: skipped (auto-download disabled)")
        return

    hf_base = "https://huggingface.co/microsoft/Florence-2-large/resolve/main"
    required_files = [
        "config.json",
        "generation_config.json",
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
        return

    # Scarica SOLO nella root primaria (models-default).
    root = model_roots[0]
    llm_root = os.path.join(root, "LLM")
    model_dir = os.path.join(llm_root, "Florence-2-large")

    try:
        os.makedirs(model_dir, exist_ok=True)
    except Exception as exc:
        logging.warning(f"Unable to create Florence-2-large dir in {root}: {exc}")
        return

    # Compat con i file già presenti nella root LLM (naming legacy del bootstrap).
    legacy_pairs = [
        ("florence-2-large-model.safetensors", "model.safetensors"),
        ("florence-2-large-pytorch_model.bin", "pytorch_model.bin"),
    ]
    for src_name, dst_name in legacy_pairs:
        src = os.path.join(llm_root, src_name)
        dst = os.path.join(model_dir, dst_name)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
                logging.info(f"Prepared Florence-2-large file: {dst}")
            except Exception as exc:
                logging.warning(f"Unable to copy Florence file {src} -> {dst}: {exc}")

    for filename in required_files:
        _download_if_missing(f"{hf_base}/{filename}", os.path.join(model_dir, filename))

    for filename in optional_files:
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


def apply_shared_model_paths():
    """
    Registra più cartelle modelli condivise e scarica automaticamente i modelli mancanti
    dalla cartella principale (prima root) usando SHARED_MODELS_URLS.
    """
    model_roots = _resolve_model_roots()

    if not model_roots:
        return

    # Crea le root (se vuoi che esistano). Se una non esiste, ComfyUI leggerà solo quelle presenti.
    for root in model_roots:
        os.makedirs(root, exist_ok=True)
        logging.info(f"Using models root: {root}")

    _ensure_llm_subdirs(model_roots)

    # Scarica modelli mancanti SOLO nella prima root (quella principale)
    # così non alteri la seconda cartella
    ensure_shared_models_downloaded(model_roots[0])
    _sync_llm_primary_to_secondary(model_roots)
    _bootstrap_trace("apply_shared_model_paths: Florence2 layout skipped (auto-download disabled)")
    # _ensure_florence2_layout(model_roots)
    _sync_llm_primary_to_secondary(model_roots)

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
        # Compat Florence/LLM: alcuni nodi cercano "LLM", altri "llm".
        "LLM": "LLM",
        "llm": "LLM",
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
        return

    if hasattr(transformers, "CLIPImageProcessor"):
        transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor
        logging.info("[WRAPPER] Applied transformers compat alias: CLIPFeatureExtractor -> CLIPImageProcessor")

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
    # Compat Florence/LLM: espone la stessa cartella con entrambe le chiavi.
    "LLM": "LLM",
    "llm": "LLM",
}

MODEL_ROOTS = [
    # Resta come fallback statico, ma il wrapper usa _resolve_model_roots().
    "/vscode/workspace/models-default",
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

    # 1b) ripulisce eventuali cache JSON corrotte di ComfyUI-Manager
    _cleanup_broken_manager_json_cache()

    # 2) env vars opzionali
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # 2b) evita side-effect: importare transformers qui puo' importare torch
    # troppo presto e causare mismatch dell'allocator CUDA.
    if os.environ.get("COMFYUI_EAGER_TRANSFORMERS_COMPAT", "0") == "1":
        _ensure_transformers_clipfeatureextractor_compat()
    else:
        logging.info("[WRAPPER] Skipping eager transformers compat patch (set COMFYUI_EAGER_TRANSFORMERS_COMPAT=1 to enable)")

    # 3) prepara root modelli
    model_roots = _resolve_model_roots()
    for root in model_roots:
        try:
            os.makedirs(root, exist_ok=True)
            logging.info(f"[WRAPPER] Using models root: {root}")
        except Exception as e:
            logging.warning(f"[WRAPPER] Cannot create models root {root}: {e}")

    _ensure_llm_subdirs(model_roots)

    # 4) download modelli mancanti SOLO nella root principale (come fai già)
    if model_roots:
        ensure_shared_models_downloaded(model_roots[0])
        _sync_llm_primary_to_secondary(model_roots)
        _bootstrap_trace("_preflight_custom_logic: Florence2 layout skipped (auto-download disabled)")
        # _ensure_florence2_layout(model_roots)
        _sync_llm_primary_to_secondary(model_roots)

    # 5) genera config path nativo ComfyUI per le shared folders
    auto_cfg = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "extra_model_paths.auto.yaml"
    )
    _write_auto_extra_model_paths_yaml(auto_cfg, model_roots)

    # 6) passa il config auto al main.py ufficiale
    _append_extra_model_paths_arg(auto_cfg)

    # 7) stabilizza allocator CUDA nel flusso wrapper.
    _append_disable_cuda_malloc_arg()


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
    # Bootstrap PRIMA degli import ComfyUI
    _preflight_custom_logic()
    _launch_official_comfyui_main()
